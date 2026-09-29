"""Remote training worker logic, kept separate from the Modal SDK entry point."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import platform
import shutil
import signal
import tempfile
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from ..core import CheckpointManifest, ValidationError, atomic_write_json, read_json
from ..models import ModelCatalog, load_detector, load_preset_detector
from ..preprocessing import PreprocessingConfig
from ..training import TrainingConfig, train_and_register
from .constants import VOLUME_MOUNT, VOLUME_NAME


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_payload(
    payload: bytes | str | Path,
    destination: str | Path,
    *,
    expected_sha256: str,
) -> Path:
    """Verify and safely extract the deterministic training archive into a job directory."""
    archive_source = payload
    if isinstance(payload, (str, Path)):
        archive_source = Path(payload)
        if _sha256_file(archive_source) != expected_sha256:
            raise ValidationError("Cloud payload SHA-256 does not match the submitted identity")
    else:
        if _sha256_bytes(payload) != expected_sha256:
            raise ValidationError("Cloud payload SHA-256 does not match the submitted identity")
        archive_source = io.BytesIO(payload)
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_source) as archive:
        seen: set[str] = set()
        resolved_targets: set[Path] = set()
        members = []
        for item in archive.infolist():
            name = item.filename
            path = PurePosixPath(name)
            normalized_name = "/".join(path.parts)
            if (
                not name
                or "\\" in name
                or path.is_absolute()
                or ".." in path.parts
                or "." in path.parts
                or not path.parts
                or path.parts[0] not in {"dataset", "model"}
                or name in seen
                or item.is_dir()
                or normalized_name != name
            ):
                raise ValidationError("Cloud payload contains an unsafe or duplicate archive path")
            seen.add(name)
            mode = (item.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValidationError("Cloud payload may not contain symbolic links")
            if path.parts[0] == "model" and name not in {
                "model/base.pt",
                "model/checkpoint.json",
            }:
                raise ValidationError("Cloud payload contains an unexpected model file")
            target = (root / Path(*path.parts)).resolve()
            if not target.is_relative_to(root) or target in resolved_targets:
                raise ValidationError("Cloud payload contains a path outside its job directory")
            resolved_targets.add(target)
            members.append(item)
        if "dataset/dataset.yaml" not in seen:
            raise ValidationError("Cloud payload is missing dataset/dataset.yaml")
        archive.extractall(root, members=members)
    return root


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _code_digest() -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_progress(job_root: Path, volume_commit: Callable[[], Any], **values) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        **values,
    }
    atomic_write_json(job_root / "progress.json", payload)
    volume_commit()


class _TailBuffer:
    """Keep only the most recent log text while streaming training output."""

    encoding = "utf-8"

    def __init__(self, max_chars: int):
        self.max_chars = max(1, int(max_chars))
        self._value = ""

    def write(self, value: str) -> int:
        rendered = str(value)
        self._value = (self._value + rendered)[-self.max_chars :]
        return len(rendered)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def getvalue(self) -> str:
        return self._value


def _environment() -> dict[str, str]:
    names = ["torch", "torchvision", "ultralytics", "Pillow", "PyYAML", "modal"]
    versions = {name.lower().replace("-", "_"): "unknown" for name in names}
    for name in names:
        try:
            versions[name.lower().replace("-", "_")] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    versions["python"] = platform.python_version()
    try:
        import torch

        if torch.cuda.is_available():
            versions["gpu"] = torch.cuda.get_device_name(0)
            versions["cuda"] = str(torch.version.cuda or "unknown")
    except Exception:
        pass
    return versions


def _remote_result(payload: dict[str, Any], state: str, **fields) -> dict[str, Any]:
    return {
        "state": state,
        "remote_state": state,
        "echo": dict(payload.get("echo", {})),
        **fields,
    }


def run_remote_training(
    payload: dict[str, Any],
    *,
    volume_root: str | Path = VOLUME_MOUNT,
    volume_commit: Callable[[], Any] = lambda: None,
) -> dict[str, Any]:
    """Run one verified cloud training job and return volume-backed artifact references."""
    job_key = str(payload.get("job_key", ""))
    if not job_key or any(part in job_key for part in ("/", "\\", "..")):
        return _remote_result(payload, "failed", error="Cloud job identity is invalid")
    expected_echo = payload.get("echo")
    if not isinstance(expected_echo, dict) or expected_echo.get("job_key") != job_key:
        return _remote_result(payload, "failed", error="Cloud job identity echo is invalid")
    if payload.get("volume_name") != VOLUME_NAME:
        return _remote_result(payload, "failed", error="Cloud volume identity does not match")
    prefix = f"jobs/{job_key}"
    if payload.get("remote_prefix") != prefix:
        return _remote_result(payload, "failed", error="Cloud result path identity does not match")
    expected_mount = f"{VOLUME_MOUNT}/{prefix}/dataset"
    if payload.get("dataset_mount") != expected_mount:
        return _remote_result(payload, "failed", error="Cloud dataset path identity does not match")
    if _code_digest() != payload.get("code_digest"):
        return _remote_result(
            payload, "failed", error="Remote AmphiLens source digest does not match"
        )

    try:
        deadline = datetime.fromisoformat(str(payload["deadline_at"])).astimezone(timezone.utc)
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    except (KeyError, ValueError, TypeError):
        return _remote_result(payload, "failed", error="Cloud job deadline is invalid")
    if remaining <= 0:
        return _remote_result(
            payload, "timed_out", error="Cloud job deadline elapsed before startup"
        )

    root = Path(volume_root).expanduser().resolve() / prefix
    payload_path = (root / "payload.zip").resolve()
    if not payload_path.is_relative_to(Path(volume_root).expanduser().resolve()):
        return _remote_result(payload, "failed", error="Cloud payload path is unsafe")
    try:
        if payload.get("payload_remote_path") != f"{prefix}/payload.zip":
            raise ValidationError("Cloud payload path identity does not match")
        safe_extract_payload(
            payload_path,
            root,
            expected_sha256=str(payload.get("payload_sha256", "")),
        )
        echo = payload["echo"]
        if echo.get("payload_sha256") != payload.get("payload_sha256"):
            raise ValidationError("Cloud payload SHA-256 does not match its identity echo")
        effective = payload["effective_configuration"]
        if not isinstance(effective, dict):
            raise ValidationError("Effective training configuration is invalid")
        if effective.get("fingerprint") != payload.get("effective_fingerprint"):
            raise ValidationError("Effective configuration fingerprint does not match")
        if echo.get("effective_fingerprint") != payload.get("effective_fingerprint"):
            raise ValidationError("Effective configuration identity does not match")
        classes = list(effective.get("classes", []))
        if not classes or classes != list(echo.get("classes", [])):
            raise ValidationError("Cloud training class order does not match")
        dataset_yaml = root / "dataset" / "dataset.yaml"
        try:
            import yaml

            dataset_config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValidationError("Remote training dataset configuration is invalid") from exc
        if (
            not isinstance(dataset_config, dict)
            or str(dataset_config.get("path")) != expected_mount
        ):
            raise ValidationError("Remote training dataset path does not match the mounted volume")
        remote_names = dataset_config.get("names", [])
        if isinstance(remote_names, dict):
            remote_classes = [
                remote_names.get(str(index), remote_names.get(index))
                for index in range(len(classes))
            ]
        elif isinstance(remote_names, list):
            remote_classes = remote_names
        else:
            remote_classes = []
        if remote_classes != classes:
            raise ValidationError("Remote training dataset classes do not match the submitted job")

        base_path = root / "model" / "base.pt"
        base_manifest_path = root / "model" / "checkpoint.json"
        base_digest = str(payload.get("base_checkpoint_sha256", ""))
        if base_digest:
            if not base_path.is_file() or _sha256_file(base_path) != base_digest:
                raise ValidationError("Base checkpoint hash does not match the submitted job")
            if echo.get("base_checkpoint_sha256") != base_digest:
                raise ValidationError("Base checkpoint identity does not match")
        elif base_path.exists() or base_manifest_path.exists():
            raise ValidationError("Unexpected base checkpoint in cloud payload")

        base_manifest = None
        if base_manifest_path.is_file():
            base_manifest = CheckpointManifest.from_dict(read_json(base_manifest_path))
            if base_manifest.sha256 != base_digest:
                raise ValidationError("Base checkpoint manifest hash does not match")

        if base_path.is_file():
            detector = load_detector(
                base_path,
                architecture=str(effective["architecture"]),
                classes=classes,
                model_id=str(effective["model_id"]),
                checkpoint_manifest=base_manifest,
                preprocessing=dict(effective["preprocessing"]),
            )
        else:
            detector = load_preset_detector(
                ModelCatalog().get(str(effective["model_preset"])),
                classes=classes,
            )

        submitted_config = dict(payload.get("training_config", {}))
        training = TrainingConfig(
            epochs=int(submitted_config.get("epochs", effective["epochs"])),
            image_size=int(submitted_config.get("image_size", effective["image_size"])),
            batch_size=int(submitted_config.get("batch_size", effective["batch_size"])),
            patience=int(submitted_config.get("patience", effective["patience"])),
            seed=int(submitted_config.get("seed", effective["seed"])),
            device="cuda",
            run_name=str(payload.get("run_id", "cloud-train")),
            preprocessing=PreprocessingConfig.from_any(effective["preprocessing"]),
            freeze_strategy=str(effective.get("freeze_strategy", "none")),
            evaluation="not evaluated",
            metadata={"effective_configuration": effective},
        )
        if (
            submitted_config.get("evaluation", "not evaluated") != "not evaluated"
            or submitted_config.get("val", False) is not False
        ):
            raise ValidationError("Cloud training does not evaluate or create validation splits")

        environment = _environment()
        output_root = Path(tempfile.mkdtemp(prefix="amphilens-cloud-training-"))
        result_root = root / "results"
        result_root.mkdir(parents=True, exist_ok=True)
        _write_progress(
            root,
            volume_commit,
            state="running",
            phase="training",
            progress=0.0,
            environment=environment,
        )
        cloud_provenance = {
            "provider": "modal",
            "job_key": job_key,
            "payload_sha256": payload["payload_sha256"],
            "effective_fingerprint": payload["effective_fingerprint"],
            "base_checkpoint_sha256": base_digest,
            "code_digest": payload["code_digest"],
            "code_version": str(payload.get("code_version", "working-tree")),
            "gpu": str(payload["gpu"]),
            "timeout_seconds": int(payload["timeout_seconds"]),
            "environment": environment,
            "evaluation": "not evaluated",
            "checkpoint_selection": "last-no-validation",
        }

        def report_progress(values: dict[str, Any]) -> None:
            _write_progress(
                root,
                volume_commit,
                state="running",
                phase="training",
                progress=float(values.get("progress", 0.0)),
                epoch=values.get("epoch"),
                epochs=values.get("epochs"),
                metrics={
                    key: value
                    for key, value in values.items()
                    if key not in {"epoch", "epochs", "progress"}
                },
                environment=environment,
            )

        timer_seconds = max(1, min(int(remaining), int(payload.get("timeout_seconds", remaining))))
        old_handler = signal.getsignal(signal.SIGALRM)

        def on_deadline(signum, frame):
            del signum, frame
            raise TimeoutError("Budget-derived server-side deadline reached")

        logs = _TailBuffer(8000)
        try:
            signal.signal(signal.SIGALRM, on_deadline)
            signal.setitimer(signal.ITIMER_REAL, timer_seconds)
            with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
                train_and_register(
                    detector,
                    dataset_yaml=dataset_yaml,
                    output_dir=output_root,
                    config=training,
                    preprocessing=effective["preprocessing"],
                    resume_from=base_manifest,
                    progress_callback=report_progress,
                    extra_training_config={"cloud": cloud_provenance},
                )
        except TimeoutError as exc:
            _write_progress(
                root,
                volume_commit,
                state="timed_out",
                phase="deadline-reached",
                progress=None,
                error=str(exc),
            )
            return _remote_result(
                payload,
                "timed_out",
                error=str(exc),
                log_tail=logs.getvalue()[-8000:],
                environment=environment,
            )
        except Exception as exc:
            safe_error = (
                str(exc)
                .replace(str(root), "[remote-job]")
                .replace(str(output_root), "[training-output]")
            )
            _write_progress(
                root,
                volume_commit,
                state="failed",
                phase="training-failed",
                progress=None,
                error=safe_error[-2000:],
            )
            return _remote_result(
                payload,
                "failed",
                error=safe_error[-2000:],
                log_tail=logs.getvalue()[-8000:],
                environment=environment,
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

        if datetime.now(timezone.utc) >= deadline:
            return _remote_result(payload, "timed_out", error="Cloud job deadline was reached")
        for name in ("best.pt", "last.pt", "metrics.json", "checkpoint.json"):
            source = output_root / name
            if not source.is_file():
                raise ValidationError(f"Remote training did not produce required artifact: {name}")
            shutil.copy2(source, result_root / name)
        remote_manifest = CheckpointManifest.from_dict(read_json(result_root / "checkpoint.json"))
        remote_manifest.checkpoint_path = f"{Path(expected_mount).parent}/results/best.pt"
        atomic_write_json(result_root / "checkpoint.json", remote_manifest.to_dict())
        volume_commit()

        artifacts = {}
        for name in ("best.pt", "last.pt", "metrics.json", "checkpoint.json"):
            artifact = result_root / name
            artifacts[name] = {
                "remote_ref": f"modal-volume://{VOLUME_NAME}/{prefix}/results/{name}",
                "sha256": _sha256_file(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        _write_progress(
            root,
            volume_commit,
            state="finished",
            phase="artifacts-ready",
            progress=1.0,
            environment=environment,
        )
        return _remote_result(
            payload,
            "finished",
            remote_state="success",
            progress=1.0,
            environment=environment,
            log_tail=logs.getvalue()[-8000:],
            artifacts=artifacts,
        )
    except Exception as exc:
        safe_error = str(exc).replace(str(root), "[remote-job]")
        try:
            _write_progress(
                root,
                volume_commit,
                state="failed",
                phase="validation-failed",
                progress=None,
                error=safe_error[-2000:],
            )
        except Exception:
            pass
        return _remote_result(payload, "failed", error=safe_error[-2000:])
    finally:
        if "output_root" in locals():
            shutil.rmtree(output_root, ignore_errors=True)
