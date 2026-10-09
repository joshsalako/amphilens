"""Durable orchestration for consented cloud GPU training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..core import CheckpointManifest, ProjectStore, ValidationError, atomic_write_json, read_json
from ..dataset import DatasetSnapshot, prepare_yolo_training_dataset, split_snapshot_images
from .estimate import CostEstimate, estimate_training_cost
from .models import TERMINAL_CLOUD_JOB_STATES, CloudConsent, CloudJobRecord
from .pack import pack_training_payload
from .transport import CloudTransport


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_digest() -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _effective_fingerprint(configuration: dict[str, Any]) -> str:
    provided = configuration.get("fingerprint")
    if isinstance(provided, str) and provided:
        return provided
    return _canonical_hash(
        {key: value for key, value in configuration.items() if key != "fingerprint"}
    )


class CloudTrainingService:
    """Prepare, submit, poll, verify, and register one durable training job."""

    def __init__(
        self,
        store: ProjectStore,
        transport: CloudTransport,
        *,
        now: Callable[[], datetime] | None = None,
    ):
        self.store = store
        self.transport = transport
        self._now = now or (lambda: datetime.now(timezone.utc))

    def _run_path(self, run_id: str) -> Path:
        if not run_id.startswith("cloud-") or "/" in run_id or ".." in run_id:
            raise ValidationError(f"Invalid cloud training run id: {run_id}")
        return self.store.root / "runs" / run_id

    def _record_path(self, run_id: str) -> Path:
        return self._run_path(run_id) / "cloud-job.json"

    def _save(self, record: CloudJobRecord) -> CloudJobRecord:
        record.updated_at = self._now().astimezone(timezone.utc).replace(microsecond=0).isoformat()
        atomic_write_json(self._record_path(record.run_id), record.to_dict())
        return record

    def get_job(self, run_id: str) -> CloudJobRecord:
        path = self._record_path(run_id)
        if not path.is_file():
            raise ValidationError(f"Cloud training run not found: {run_id}")
        return CloudJobRecord.from_dict(read_json(path))

    def list_jobs(self) -> list[CloudJobRecord]:
        records = []
        for path in (self.store.root / "runs").glob("*/cloud-job.json"):
            records.append(CloudJobRecord.from_dict(read_json(path)))
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def active_jobs(self) -> list[CloudJobRecord]:
        return [
            record for record in self.list_jobs() if record.state not in TERMINAL_CLOUD_JOB_STATES
        ]

    def _safe_error(self, error: Exception) -> str:
        message = str(error)
        redactor = getattr(self.transport, "redact", None)
        return str(redactor(message)) if callable(redactor) else message

    def estimate(
        self,
        snapshot_path: str | Path,
        *,
        gpu: str = "L4",
        epochs: int | None = None,
        max_cost_usd: float = 5.0,
        image_size: int = 640,
        preprocessing: dict[str, Any] | None = None,
        validation_snapshot_path: str | Path | None = None,
        test_snapshot_path: str | Path | None = None,
        data_mode: str = "separate-snapshots",
        seed: int = 42,
        allow_image_level_fallback: bool = False,
    ) -> CostEstimate:
        snapshot = DatasetSnapshot.load(snapshot_path)
        project = self.store.load_manifest()
        if not snapshot.manifest.images:
            raise ValidationError("Cloud training requires a non-empty labelled dataset snapshot")
        if preprocessing is None:
            selected_preprocessing = project.project_config.preprocessing.to_dict()
            selected_preprocessing["short_side_dimension"] = image_size
        else:
            selected_preprocessing = preprocessing
        training_canvas_size = snapshot.training_canvas_size(selected_preprocessing)
        train_count = len(snapshot.manifest.images)
        byte_count = sum(
            (snapshot.root / image.relative_path).stat().st_size
            for image in snapshot.manifest.images
        )
        if data_mode == "auto-split":
            split = split_snapshot_images(
                snapshot,
                seed=seed,
                allow_image_level_fallback=allow_image_level_fallback,
            )
            train_count = len(split["train"])
            byte_count = sum(
                (snapshot.root / image.relative_path).stat().st_size
                for role in ("train", "validation", "test")
                for image in split[role]
            )
        elif data_mode == "training-monitor":
            # The prepared cloud bundle contains separate train and validation copies.
            byte_count *= 2
        elif data_mode == "separate-snapshots":
            for selected_path in (validation_snapshot_path, test_snapshot_path):
                if not selected_path:
                    continue
                selected = DatasetSnapshot.load(selected_path)
                if selected.manifest.classes != snapshot.manifest.classes:
                    raise ValidationError("Dataset classes do not match the training snapshot")
                byte_count += sum(
                    (selected.root / image.relative_path).stat().st_size
                    for image in selected.manifest.images
                )
        else:
            raise ValidationError(f"Unsupported training data mode: {data_mode}")
        if validation_snapshot_path and data_mode == "separate-snapshots":
            validation = DatasetSnapshot.load(validation_snapshot_path)
            training_canvas_size = max(
                training_canvas_size,
                validation.training_canvas_size(selected_preprocessing),
            )
        if test_snapshot_path and data_mode == "separate-snapshots":
            test = DatasetSnapshot.load(test_snapshot_path)
            training_canvas_size = max(
                training_canvas_size,
                test.training_canvas_size(selected_preprocessing),
            )
        return estimate_training_cost(
            image_count=train_count,
            dataset_bytes=byte_count,
            epochs=project.project_config.epochs if epochs is None else epochs,
            gpu=gpu,
            max_cost_usd=max_cost_usd,
            image_size=training_canvas_size,
        )

    def submit(
        self,
        *,
        snapshot_path: str | Path,
        effective_configuration: dict[str, Any],
        training_config: dict[str, Any],
        consent: CloudConsent | None,
        gpu: str = "L4",
        base_checkpoint: str | Path | None = None,
        base_manifest: CheckpointManifest | None = None,
        validation_snapshot_path: str | Path | None = None,
        test_snapshot_path: str | Path | None = None,
        data_mode: str = "separate-snapshots",
        allow_image_level_fallback: bool = False,
    ) -> CloudJobRecord:
        if consent is None or not consent.acknowledged or not consent.uploads_dataset:
            raise ValidationError("Explicit cloud training and dataset upload consent is required")
        snapshot = DatasetSnapshot.load(snapshot_path)
        project = self.store.load_manifest()
        if not snapshot.manifest.images:
            raise ValidationError("Cloud training requires a non-empty labelled dataset snapshot")
        if list(snapshot.manifest.classes) != list(project.classes):
            raise ValidationError("Dataset classes do not match the project class order")
        effective = dict(effective_configuration)
        effective.setdefault("device", "cuda")
        if effective["device"] != "cuda":
            raise ValidationError("Cloud GPU training requires the effective device to be cuda")
        fingerprint = _effective_fingerprint(effective)
        normalized_training = dict(training_config)
        epochs = int(normalized_training.get("epochs", effective.get("epochs", 100)))
        image_size = int(normalized_training.get("image_size", effective.get("image_size", 640)))
        normalized_training.setdefault("epochs", epochs)
        normalized_training.setdefault("image_size", image_size)
        normalized_training.setdefault("batch_size", int(effective.get("batch_size", 16)))
        normalized_training.setdefault("patience", int(effective.get("patience", 25)))
        normalized_training.setdefault("seed", int(effective.get("seed", 42)))
        normalized_training.setdefault("val", True)
        if epochs <= 0 or image_size <= 0 or int(normalized_training["batch_size"]) <= 0:
            raise ValidationError(
                "Cloud training epochs, image_size, and batch_size must be positive"
            )
        if int(normalized_training["patience"]) < 0:
            raise ValidationError("Cloud training patience cannot be negative")
        if normalized_training.get("device", "cuda") != "cuda":
            raise ValidationError("Cloud GPU training requires the training device to be cuda")
        normalized_training["device"] = "cuda"
        if normalized_training["val"] is not True:
            raise ValidationError("Cloud training requires a validation dataset for early stopping")
        normalized_training.setdefault(
            "evaluation",
            "test set"
            if test_snapshot_path or data_mode == "auto-split"
            else "not evaluated",
        )
        normalized_training["data_mode"] = data_mode
        normalized_training["allow_image_level_split"] = bool(allow_image_level_fallback)

        checkpoint_path = Path(base_checkpoint).expanduser().resolve() if base_checkpoint else None
        if checkpoint_path is None and base_manifest is not None:
            checkpoint_path = Path(base_manifest.checkpoint_path).expanduser().resolve()
        if checkpoint_path is not None and not checkpoint_path.is_file():
            raise ValidationError(f"Base checkpoint does not exist: {checkpoint_path}")
        base_hash = _file_hash(checkpoint_path) if checkpoint_path else ""
        if base_manifest is not None and base_hash != base_manifest.sha256:
            raise ValidationError("Base checkpoint hash does not match its manifest")

        estimate = self.estimate(
            snapshot.root,
            gpu=gpu,
            epochs=epochs * 2,
            max_cost_usd=consent.max_cost_usd,
            image_size=image_size,
            preprocessing=effective.get("preprocessing"),
            validation_snapshot_path=validation_snapshot_path,
            test_snapshot_path=test_snapshot_path,
            data_mode=data_mode,
            seed=int(normalized_training.get("seed", 42)),
            allow_image_level_fallback=allow_image_level_fallback,
        )
        if abs(consent.estimated_usd - estimate.high_usd) > 0.02:
            raise ValidationError("Cloud consent estimate is stale; review the current estimate")

        code_digest = _code_digest()
        identity = {
            "project": project.name,
            "snapshot_id": snapshot.manifest.snapshot_id,
            "validation_snapshot_id": (
                DatasetSnapshot.load(validation_snapshot_path).manifest.snapshot_id
                if validation_snapshot_path else None
            ),
            "test_snapshot_id": (
                DatasetSnapshot.load(test_snapshot_path).manifest.snapshot_id
                if test_snapshot_path else None
            ),
            "image_digests": sorted(
                image.sha256
                for role_snapshot in [
                    snapshot,
                    *(
                        [DatasetSnapshot.load(validation_snapshot_path)]
                        if validation_snapshot_path else []
                    ),
                    *([DatasetSnapshot.load(test_snapshot_path)] if test_snapshot_path else []),
                ]
                for image in role_snapshot.manifest.images
            ),
            "effective_fingerprint": fingerprint,
            "training_config": normalized_training,
            "gpu": gpu,
            "max_cost_usd": consent.max_cost_usd,
            "timeout_seconds": estimate.time_limit_seconds,
            "estimate": estimate.to_dict(),
            "base_checkpoint_sha256": base_hash,
            "code_digest": code_digest,
            "image_pins": self._image_pins(),
            "operation": "train",
        }
        job_key = _canonical_hash(identity)
        existing = next((item for item in self.list_jobs() if item.job_key == job_key), None)
        if existing is not None:
            if (
                existing.snapshot_id != snapshot.manifest.snapshot_id
                or existing.effective_fingerprint != fingerprint
                or existing.training_config != normalized_training
            ):
                raise ValidationError(
                    "Existing cloud job key does not match the requested snapshot/config"
                )
            return existing

        run_id = f"cloud-{job_key[:16]}"
        run_dir = self._run_path(run_id)
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            existing_path = run_dir / "cloud-job.json"
            if not existing_path.is_file():
                raise ValidationError(
                    "An identical cloud request is being prepared; refresh its job status shortly"
                )
            existing = CloudJobRecord.from_dict(read_json(existing_path))
            if existing.job_key != job_key:
                raise ValidationError("Cloud job id collision; no new job was submitted")
            return existing
        created = self._now().astimezone(timezone.utc).replace(microsecond=0)
        record = CloudJobRecord(
            run_id=run_id,
            job_key=job_key,
            state="created",
            phase="created",
            created_at=created.isoformat(),
            updated_at=created.isoformat(),
            project_name=project.name,
            snapshot_id=snapshot.manifest.snapshot_id,
            effective_fingerprint=fingerprint,
            effective_configuration=effective,
            training_config=normalized_training,
            consent=consent,
            estimate=estimate.to_dict(),
            gpu=gpu,
            timeout_seconds=estimate.time_limit_seconds,
            base_checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            base_checkpoint_sha256=base_hash,
            code_digest=code_digest,
            expected_echo={
                "job_key": job_key,
                "payload_sha256": "",
                "effective_fingerprint": fingerprint,
                "base_checkpoint_sha256": base_hash,
                "classes": list(effective.get("classes", project.classes)),
            },
        )
        self._save(record)
        try:
            record.state, record.phase = "prepared", "materializing-dataset"
            self._save(record)
            attempt = run_dir / "upload" / "attempt-1"
            prepared = attempt / "prepared-dataset"
            prepare_yolo_training_dataset(
                prepared,
                training_snapshot=snapshot,
                validation_snapshot=(
                    DatasetSnapshot.load(validation_snapshot_path)
                    if validation_snapshot_path else None
                ),
                test_snapshot=(
                    DatasetSnapshot.load(test_snapshot_path) if test_snapshot_path else None
                ),
                data_mode=data_mode,
                seed=int(normalized_training.get("seed", 42)),
                allow_image_level_fallback=allow_image_level_fallback,
                preprocessing=effective.get("preprocessing"),
            )
            mount = f"/mnt/amphilens/jobs/{job_key}/dataset"
            payload_path = run_dir / "upload" / "payload.zip"
            payload = pack_training_payload(
                prepared,
                payload_path,
                container_mount=mount,
                base_checkpoint=checkpoint_path,
                base_manifest=base_manifest.to_dict() if base_manifest else None,
            )
            record.payload_hash = payload.sha256
            record.payload_size_bytes = payload.size_bytes
            record.expected_echo["payload_sha256"] = payload.sha256
            record.state, record.phase = "uploading", "uploading-payload"
            self._save(record)
            remote_path = f"jobs/{job_key}/payload.zip"
            self.transport.upload(payload.path, remote_path, payload.sha256)
            record.state, record.phase = "uploaded", "payload-uploaded"
            self._save(record)
            record.deadline_at = (
                (self._now() + timedelta(seconds=record.timeout_seconds))
                .astimezone(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            )
            remote_payload = {
                "run_id": run_id,
                "job_key": job_key,
                "payload_sha256": payload.sha256,
                "payload_remote_path": remote_path,
                "effective_configuration": effective,
                "effective_fingerprint": fingerprint,
                "training_config": normalized_training,
                "base_checkpoint_sha256": base_hash,
                "code_digest": code_digest,
                "code_version": "working-tree",
                "gpu": gpu,
                "timeout_seconds": record.timeout_seconds,
                "deadline_at": record.deadline_at,
                "dataset_mount": mount,
                "volume_name": self._volume_name(),
                "remote_prefix": f"jobs/{job_key}",
                "echo": dict(record.expected_echo),
            }
            record.state, record.phase = "submitted", "remote-submitted"
            record.call_id = self.transport.submit(remote_payload)
            record.dashboard_url = self.transport.dashboard_url(record.call_id)
            return self._save(record)
        except Exception as exc:
            record.state, record.phase = "failed", "submission-failed"
            record.error = self._safe_error(exc)
            self._save(record)
            raise

    def refresh(self, run_id: str) -> CloudJobRecord:
        record = self.get_job(run_id)
        if record.state in TERMINAL_CLOUD_JOB_STATES or not record.call_id:
            return record
        if record.deadline_at and self._now().astimezone(timezone.utc) >= datetime.fromisoformat(
            record.deadline_at
        ):
            try:
                self.transport.cancel(record.call_id)
            except Exception as exc:
                record.phase = "deadline-cancel-pending"
                record.error = f"Deadline reached; cancellation failed: {self._safe_error(exc)}"
                return self._save(record)
            else:
                record.error = "The budget-derived server-side time limit was reached"
            record.state, record.phase = "timed_out", "server-deadline-reached"
            return self._save(record)
        try:
            result = self.transport.poll(record.call_id, remote_prefix=f"jobs/{record.job_key}")
        except Exception as exc:
            record.phase = "status-poll-failed"
            record.error = self._safe_error(exc)
            return self._save(record)
        if result is None:
            record.error = ""
            record.state, record.phase = "running", "remote-running"
            return self._save(record)
        record.error = ""
        record.remote_state = str(result.get("remote_state", result.get("state", "")))
        record.progress = result.get("progress")
        details = result.get("progress_details")
        if not isinstance(details, dict):
            details = {
                key: result[key]
                for key in (
                    "updated_at",
                    "phase",
                    "message",
                    "epoch",
                    "epochs",
                    "metrics",
                    "progress",
                    "device",
                    "gpu",
                    "error",
                    "environment",
                )
                if key in result
            }
        progress_keys = {
            "updated_at",
            "phase",
            "message",
            "error",
            "progress",
            "phase_progress",
            "completed",
            "failed",
            "total",
            "remaining",
            "eta_seconds",
            "epoch",
            "epochs",
            "metrics",
            "device",
            "gpu",
            "batch",
        }
        record.progress_details = {
            key: value for key, value in details.items() if key in progress_keys
        }
        record.log_tail = str(details.get("log_tail", result.get("log_tail", "")))[-8000:]
        if details.get("error"):
            record.error = str(details["error"])[-2000:]
        event = {
            key: value
            for key, value in record.progress_details.items()
            if key
            in {
                "phase",
                "message",
                "error",
                "epoch",
                "epochs",
                "metrics",
                "device",
                "gpu",
                "progress",
                "completed",
                "failed",
                "total",
                "remaining",
                "updated_at",
            }
        }
        if event:
            event_key = (
                event.get("phase"),
                event.get("epoch"),
                event.get("completed"),
                event.get("failed"),
                event.get("message"),
            )
            prior = record.progress_events[-1] if record.progress_events else {}
            prior_key = (
                prior.get("phase"),
                prior.get("epoch"),
                prior.get("completed"),
                prior.get("failed"),
                prior.get("message"),
            )
            if event_key != prior_key:
                record.progress_events.append(event)
                del record.progress_events[:-100]
        record.environment = {
            str(key): str(value) for key, value in dict(result.get("environment", {})).items()
        }
        if result.get("state") in {"finished", "success", "completed"}:
            record.state, record.phase = "finished", "remote-finished"
            record.remote_result = result
        elif result.get("state") in {"canceled", "cancelled"}:
            record.state, record.phase = "canceled", "remote-canceled"
        elif result.get("state") in {"timed_out", "timeout"}:
            record.state, record.phase = "timed_out", "remote-timeout"
        elif result.get("state") in {"failed", "error"}:
            record.state, record.phase = "failed", "remote-failed"
            record.error = str(result.get("error", "Remote training failed"))
        else:
            record.state, record.phase = "running", "remote-running"
        return self._save(record)

    def collect(self, run_id: str) -> CloudJobRecord:
        record = self.get_job(run_id)
        retryable_result = (
            record.state == "incomplete"
            and record.remote_result.get("state") in {"finished", "success", "completed"}
            and isinstance(record.remote_result.get("artifacts"), dict)
        )
        if record.state != "finished" and not retryable_result:
            raise ValidationError("Cloud job must be finished before collecting its results")
        try:
            result = record.remote_result
            self._verify_echo(record, result.get("echo"))
            artifacts = result.get("artifacts")
            required = {"best.pt", "last.pt", "metrics.json", "checkpoint.json"}
            if not isinstance(artifacts, dict) or not required.issubset(artifacts):
                missing = sorted(required - set(artifacts or {}))
                raise ValidationError(f"Remote training is missing required artifacts: {missing}")

            results_dir = self._run_path(run_id) / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            downloaded: dict[str, Path] = {}
            for name, metadata in artifacts.items():
                if Path(name).name != name or name in {".", ".."}:
                    raise ValidationError(f"Remote artifact name is unsafe: {name}")
                remote_ref = metadata.get("remote_ref")
                expected_hash = metadata.get("sha256")
                if not isinstance(remote_ref, str) or not isinstance(expected_hash, str):
                    raise ValidationError(f"Remote artifact metadata is invalid: {name}")
                destination = results_dir / name
                size = int(metadata.get("size_bytes", -1))
                if (
                    destination.is_file()
                    and destination.stat().st_size == size
                    and _file_hash(destination) == expected_hash
                ):
                    downloaded[name] = destination
                    continue
                temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
                try:
                    self.transport.download(remote_ref, temporary)
                    if not temporary.is_file():
                        raise ValidationError(f"Remote artifact was not downloaded: {name}")
                    digest = _file_hash(temporary)
                    if digest != expected_hash:
                        raise ValidationError(f"Remote artifact hash mismatch: {name}")
                    if temporary.stat().st_size != size:
                        raise ValidationError(f"Remote artifact size mismatch: {name}")
                    os.replace(temporary, destination)
                    downloaded[name] = destination
                finally:
                    temporary.unlink(missing_ok=True)

            remote_manifest = CheckpointManifest.from_dict(read_json(downloaded["checkpoint.json"]))
            best_path = downloaded["best.pt"]
            if remote_manifest.sha256 != _file_hash(best_path):
                raise ValidationError("Remote checkpoint manifest hash does not match best.pt")
            effective = record.effective_configuration
            expected_preprocessing = dict(effective.get("preprocessing", {}))
            remote_manifest.validate_compatibility(
                architecture=str(effective["architecture"]),
                classes=list(effective["classes"]),
                preprocessing=expected_preprocessing,
            )
            if remote_manifest.model_id != effective["model_id"]:
                raise ValidationError("Remote checkpoint model id does not match the submitted job")
            cloud_provenance = {
                "provider": "modal",
                "job_key": record.job_key,
                "payload_sha256": record.payload_hash,
                "effective_fingerprint": record.effective_fingerprint,
                "base_checkpoint_sha256": record.base_checkpoint_sha256,
                "code_digest": record.code_digest,
                "code_version": record.code_version,
                "gpu": record.gpu,
                "timeout_seconds": record.timeout_seconds,
                "environment": record.environment,
                "evaluation": str(record.training_config.get("evaluation", "not evaluated")),
                "checkpoint_selection": "best-validation",
            }
            training_config = dict(record.training_config)
            training_config["effective_configuration"] = record.effective_configuration
            training_config["cloud"] = cloud_provenance
            record.training_config = training_config
            local_manifest = CheckpointManifest.create(
                best_path,
                model_id=remote_manifest.model_id,
                architecture=remote_manifest.architecture,
                classes=remote_manifest.classes,
                preprocessing=remote_manifest.preprocessing,
                parent_checkpoint=record.base_checkpoint_path,
                training_config=training_config,
            )
            local_manifest.validate_compatibility(
                architecture=str(effective["architecture"]),
                classes=list(effective["classes"]),
                preprocessing=expected_preprocessing,
            )
            atomic_write_json(results_dir / "checkpoint.json", local_manifest.to_dict())
            checkpoint_dir = self.store.root / "checkpoints" / run_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_path, checkpoint_dir / "best.pt")
            atomic_write_json(checkpoint_dir / "checkpoint.json", local_manifest.to_dict())
            self._register_results(run_id, results_dir, checkpoint_dir)
            record.state, record.phase = "verified", "artifacts-verified-and-registered"
            record.error = ""
            record.checkpoint_manifest = local_manifest
            self._cleanup(record)
            return self._save(record)
        except Exception as exc:
            record.state, record.phase = "incomplete", "artifact-verification-failed"
            record.error = self._safe_error(exc)
            self._save(record)
            raise

    def cancel(self, run_id: str) -> CloudJobRecord:
        record = self.get_job(run_id)
        if record.state in TERMINAL_CLOUD_JOB_STATES:
            return record
        if record.call_id:
            self.transport.cancel(record.call_id)
        record.state, record.phase = "canceled", "canceled-by-user"
        return self._save(record)

    def retry_cleanup(self, run_id: str) -> CloudJobRecord:
        record = self.get_job(run_id)
        if record.state not in TERMINAL_CLOUD_JOB_STATES:
            raise ValidationError("Uploaded files can only be removed after the container stops")
        self._cleanup(record)
        return self._save(record)

    @staticmethod
    def _verify_echo(record: CloudJobRecord, echo: Any) -> None:
        if not isinstance(echo, dict):
            raise ValidationError("Remote result is missing its job identity echo")
        expected = dict(record.expected_echo)
        if echo != expected:
            raise ValidationError("Remote result identity does not match this cloud training job")

    def _register_results(self, run_id: str, results_dir: Path, checkpoint_dir: Path) -> None:
        paths = {
            results_dir / "best.pt": "cloud-checkpoint-best",
            results_dir / "last.pt": "cloud-checkpoint-last",
            results_dir / "metrics.json": "cloud-training-metrics",
            results_dir / "checkpoint.json": "cloud-checkpoint-manifest",
            checkpoint_dir / "best.pt": "cloud-checkpoint-best-registered",
            checkpoint_dir / "checkpoint.json": "cloud-checkpoint-manifest-registered",
        }
        current = {item.relative_path: item for item in self.store.load_artifact_index()}
        from ..core import ArtifactRecord

        for path, artifact_type in paths.items():
            if not path.is_file() or not path.resolve().is_relative_to(self.store.root):
                raise ValidationError(f"Cloud artifact is missing or outside the project: {path}")
            relative = path.relative_to(self.store.root).as_posix()
            current[relative] = ArtifactRecord(
                relative_path=relative,
                sha256=_file_hash(path),
                artifact_type=artifact_type,
                producer_run=run_id,
            )
        atomic_write_json(
            self.store.root / "artifacts" / "index.json",
            {
                "schema_version": 1,
                "artifacts": [
                    item.to_dict()
                    for item in sorted(current.values(), key=lambda value: value.relative_path)
                ],
            },
        )

    def _cleanup(self, record: CloudJobRecord) -> None:
        try:
            report = self.transport.cleanup(record.job_key)
            record.cleanup_succeeded = bool(report.get("success"))
            record.cleanup_error = (
                "" if record.cleanup_succeeded else str(report.get("error", "Cleanup failed"))
            )
        except Exception as exc:
            record.cleanup_succeeded = False
            record.cleanup_error = str(exc)

    @staticmethod
    def _image_pins() -> dict[str, str]:
        from .constants import IMAGE_PINS

        return dict(IMAGE_PINS)

    @staticmethod
    def _volume_name() -> str:
        from .constants import VOLUME_NAME

        return VOLUME_NAME
