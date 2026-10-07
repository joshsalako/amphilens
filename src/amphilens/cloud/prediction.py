"""Bounded, resumable Modal prediction batches and their local detector adapter."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Any

from ..core import DetectionRecord, InferenceConfig, ValidationError, atomic_write_json
from ..runs import InferenceInterruption
from .constants import PREDICTION_MOUNT
from .estimate import (
    CPU_CORES,
    GPU_RATES_PER_SECOND,
    MEMORY_GIB,
    MODAL_CPU_RATE_PER_CORE_SECOND,
    MODAL_MEMORY_RATE_PER_GIB_SECOND,
)

_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")


class PredictionBudgetReached(InferenceInterruption):
    """The best-effort prediction spending limit has been reached."""


class PredictionCancelled(InferenceInterruption):
    """A user requested that the current Modal batch stop."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified_hosted_checkpoint(
    model,
    cache_root: str | Path,
    *,
    hf_hub_download=None,
    cache_commit=lambda: None,
) -> Path:
    """Fetch a public checkpoint at its pinned revision and cache only verified bytes."""
    if hf_hub_download is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Modal prediction image is missing huggingface_hub") from exc
    root = Path(cache_root).expanduser().resolve()
    verified = root / "verified" / f"{model.sha256}.pt"
    verified.parent.mkdir(parents=True, exist_ok=True)
    if verified.is_file():
        if _sha256_file(verified) == model.sha256:
            return verified
        verified.unlink()
    revision = model.to_summary()["revision"]
    downloaded = Path(
        hf_hub_download(
            repo_id=model.repo_id,
            filename=model.artifact,
            repo_type="model",
            revision=revision,
            token=False,
            cache_dir=str(root / "huggingface"),
        )
    )
    if not downloaded.is_file() or _sha256_file(downloaded) != model.sha256:
        raise RuntimeError(
            f"The public Hugging Face checkpoint failed SHA-256 verification: {model.model_id}"
        )
    shutil.copy2(downloaded, verified)
    cache_commit()
    return verified


def _safe_error(error: Exception, root: Path) -> str:
    return f"{type(error).__name__}: {str(error).replace(str(root), '[remote-batch]')[:1500]}"


def run_remote_prediction_batch(
    payload: dict[str, Any],
    *,
    volume_root: str | Path = PREDICTION_MOUNT,
    model_cache_root: str | Path,
    detector,
    volume_reload=lambda: None,
    volume_commit=lambda: None,
) -> dict[str, Any]:
    """Verify and predict one committed image batch, then remove its image data."""
    root = Path(volume_root).resolve()
    job_key = str(payload.get("job_key", ""))
    batch_id = str(payload.get("batch_id", ""))
    if not _ID_PATTERN.fullmatch(job_key) or not _ID_PATTERN.fullmatch(batch_id):
        return {"state": "failed", "error": "Prediction batch identity is invalid"}
    batch_root = (root / "prediction-jobs" / job_key / batch_id).resolve()
    if not batch_root.is_relative_to(root):
        return {"state": "failed", "error": "Prediction batch path is unsafe"}
    started = time.monotonic()
    try:
        volume_reload()
        images = payload.get("images")
        if not isinstance(images, list) or not images or len(images) > 32:
            raise ValidationError("Prediction batch must contain between 1 and 32 images")
        manifest_path = batch_root / "manifest.json"
        if not manifest_path.is_file():
            raise ValidationError("Committed prediction batch manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("images") != images:
            raise ValidationError("Committed prediction manifest does not match the request")
        config_data = payload.get("config")
        if not isinstance(config_data, dict):
            raise ValidationError("Prediction configuration is invalid")
        paths: list[Path] = []
        id_by_path: dict[str, str] = {}
        seen_ids: set[str] = set()
        image_directory = (batch_root / "images").resolve()
        for item in images:
            if not isinstance(item, dict):
                raise ValidationError("Prediction image manifest is invalid")
            image_id = str(item.get("image_id", ""))
            remote_path = str(item.get("remote_path", ""))
            path_parts = PurePosixPath(remote_path)
            if (
                not _ID_PATTERN.fullmatch(image_id)
                or image_id in seen_ids
                or path_parts.is_absolute()
                or ".." in path_parts.parts
                or path_parts.parent.as_posix() != f"prediction-jobs/{job_key}/{batch_id}/images"
                or path_parts.stem != image_id
            ):
                raise ValidationError("Prediction image identity or path is invalid")
            image_path = (root / Path(*path_parts.parts)).resolve()
            if not image_path.is_relative_to(image_directory) or not image_path.is_file():
                raise ValidationError("A transferred prediction image is missing or unsafe")
            sentinel_path = Path(f"{image_path}.sha256")
            sentinel = (
                sentinel_path.read_text(encoding="ascii").strip() if sentinel_path.is_file() else ""
            )
            if not sentinel_path.is_file() or sentinel != str(item.get("sha256", "")):
                raise ValidationError("Transferred image SHA-256 manifest is missing or mismatched")
            if _sha256_file(image_path) != str(item.get("sha256", "")):
                raise ValidationError("Transferred image SHA-256 verification failed")
            seen_ids.add(image_id)
            paths.append(image_path)
            id_by_path[str(image_path)] = image_id

        config = InferenceConfig(
            model_id=str(config_data["model_id"]),
            image_size=int(config_data["image_size"]),
            confidence=float(config_data["confidence"]),
            preprocessing=config_data.get("preprocessing", {}),
            batch_size=len(images),
            device="cuda",
            run_id=str(config_data["run_id"]),
            metadata=dict(config_data.get("metadata", {})),
        )

        records = []
        for record in detector.predict(paths, config):
            image_id = id_by_path.get(str(Path(record.image_path).resolve()))
            if image_id is None:
                raise ValidationError("Model returned a result for an unknown prediction image")
            row = record.to_dict()
            row.pop("image_path", None)
            row.pop("image_id", None)
            records.append({"local_image_id": image_id, **row})
        return {
            "state": "finished",
            "records": records,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001 - return only sanitized, bounded error details
        return {
            "state": "failed",
            "error": _safe_error(exc, root),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        try:
            if batch_root.exists():
                shutil.rmtree(batch_root)
                volume_commit()
        except OSError:
            # The local transport makes a second cleanup attempt after every invocation.
            pass


class ModalPredictionDetector:
    """Detector adapter that transfers only its current bounded batch to Modal."""

    def __init__(
        self,
        transport,
        *,
        job_key: str,
        gpu: str,
        model_spec: dict[str, Any],
        timeout_seconds: int,
        max_cost_usd: float,
        output_dir: str | Path,
        progress_callback=None,
        image_count: int = 0,
        cancellation_requested=None,
    ):
        self.transport = transport
        self.job_key = job_key
        self.gpu = gpu
        self.model_spec = dict(model_spec)
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.max_cost_usd = float(max_cost_usd)
        self.output_dir = Path(output_dir).resolve()
        self.progress_callback = progress_callback
        self.image_count = image_count
        self.cancellation_requested = cancellation_requested or (lambda: False)
        self.model_id = str(model_spec["model_id"])
        self._batch_number = 0
        self._uploaded_images = 0
        self.device_name = "Modal GPU"
        self._cost_path = self.output_dir / "cloud-cost.json"
        self.estimated_cost_usd = 0.0
        self.elapsed_seconds = 0.0
        self.completed_image_count = 0
        if self._cost_path.is_file():
            try:
                self.estimated_cost_usd = float(
                    json.loads(self._cost_path.read_text(encoding="utf-8")).get(
                        "estimated_cost_usd", 0.0
                    )
                )
                self.elapsed_seconds = float(
                    json.loads(self._cost_path.read_text(encoding="utf-8")).get(
                        "elapsed_seconds", 0.0
                    )
                )
                self.completed_image_count = int(
                    json.loads(self._cost_path.read_text(encoding="utf-8")).get(
                        "completed_images", 0
                    )
                )
                self._uploaded_images = self.completed_image_count
            except (OSError, ValueError, TypeError):
                self.estimated_cost_usd = 0.0

    def _rate_per_second(self) -> float:
        return (
            GPU_RATES_PER_SECOND[self.gpu]
            + MODAL_CPU_RATE_PER_CORE_SECOND * CPU_CORES
            + MODAL_MEMORY_RATE_PER_GIB_SECOND * MEMORY_GIB
        )

    def _record_cost(
        self,
        *,
        elapsed_seconds: float,
        setup_seconds: float = 0.0,
        completed_images: int = 0,
    ) -> None:
        recorded_seconds = max(0.0, elapsed_seconds + setup_seconds)
        self.elapsed_seconds += recorded_seconds
        self.completed_image_count += completed_images
        self.estimated_cost_usd += recorded_seconds * self._rate_per_second()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self._cost_path,
            {
                "provider": "modal",
                "gpu": self.gpu,
                "model_id": self.model_id,
                "elapsed_seconds": round(self.elapsed_seconds, 3),
                "completed_images": self.completed_image_count,
                "estimated_cost_usd": round(self.estimated_cost_usd, 6),
                "max_cost_usd": self.max_cost_usd,
                "disclaimer": (
                    "Runtime-based estimate; startup, platform billing, and timeouts may vary."
                ),
            },
        )

    def predict(self, image_paths, config: InferenceConfig):
        paths = [Path(path).resolve() for path in image_paths]
        if self.cancellation_requested():
            raise PredictionCancelled(
                "Prediction canceled. Completed images and results are saved locally."
            )
        remaining = self.max_cost_usd - self.estimated_cost_usd
        if remaining <= 0:
            raise PredictionBudgetReached(
                f"Estimated Modal cost reached your ${self.max_cost_usd:.2f} spending limit. "
                "Completed images are saved locally; increase the limit to resume."
            )
        timeout = min(self.timeout_seconds, math.floor(remaining / self._rate_per_second()))
        if timeout < 1:
            raise PredictionBudgetReached(
                "Less than one estimated Modal GPU-second remains under the "
                f"${self.max_cost_usd:.2f} "
                "limit. Completed images are saved locally; increase the limit to resume."
            )
        self._batch_number += 1
        batch_id = f"batch-{self._batch_number:08d}"
        local_paths: dict[str, Path] = {}
        try:
            uploaded = self.transport.upload_prediction_batch(self.job_key, batch_id, paths)
            for item, path in zip(uploaded, paths, strict=True):
                local_paths[str(item["image_id"])] = path
            payload = {
                "job_key": self.job_key,
                "batch_id": batch_id,
                "gpu": self.gpu,
                "timeout_seconds": timeout,
                "model_spec": self.model_spec,
                "images": uploaded,
                "config": config.to_dict(),
                "first_batch": self._uploaded_images == 0,
            }
            if self.progress_callback:
                self.progress_callback(
                    {
                        "message": (
                            f"Uploading batch {self._batch_number}; "
                            f"{self._uploaded_images} of {self.image_count} images completed."
                        ),
                        "completed": self._uploaded_images,
                        "total": self.image_count,
                    }
                )
            result = self.transport.predict_batch(
                payload, cancellation_requested=self.cancellation_requested
            )
        finally:
            self.transport.cleanup_prediction_batch(self.job_key, batch_id)
        if not isinstance(result, dict) or result.get("state") != "finished":
            elapsed = float(result.get("elapsed_seconds", 0.0)) if isinstance(result, dict) else 0.0
            setup = float(result.get("setup_seconds", 0.0)) if isinstance(result, dict) else 0.0
            self._record_cost(elapsed_seconds=elapsed, setup_seconds=setup)
            if isinstance(result, dict):
                message = result.get("error", "Modal prediction returned an invalid response")
            else:
                message = "Modal prediction returned an invalid response"
            raise RuntimeError(str(message))
        self._record_cost(
            elapsed_seconds=float(result.get("elapsed_seconds", 0.0)),
            setup_seconds=float(result.get("setup_seconds", 0.0)),
            completed_images=len(paths),
        )
        if result.get("gpu_name"):
            self.device_name = f"Modal {self.gpu} · {result['gpu_name']}"
        records = []
        for row in result.get("records", []):
            if not isinstance(row, dict) or row.get("local_image_id") not in local_paths:
                raise RuntimeError("Modal prediction returned an unknown local image ID")
            path = local_paths[str(row["local_image_id"])]
            records.append(
                DetectionRecord(
                    image_path=str(path),
                    image_id=path.name,
                    class_id=int(row["class_id"]),
                    class_name=str(row["class_name"]),
                    confidence=(
                        float(row["confidence"]) if row.get("confidence") is not None else None
                    ),
                    bbox_xyxy=[float(value) for value in row["bbox_xyxy"]],
                    image_width=int(row["image_width"]),
                    image_height=int(row["image_height"]),
                    model_id=config.model_id,
                    run_id=config.run_id,
                    cycle=row.get("cycle"),
                    preprocessing=row.get("preprocessing"),
                )
            )
        self._uploaded_images += len(paths)
        if self.progress_callback:
            self.progress_callback(
                {
                    "message": f"Processed {self._uploaded_images} of {self.image_count} images; "
                    f"estimated Modal spend ${self.estimated_cost_usd:.2f} of "
                    f"${self.max_cost_usd:.2f}.",
                    "completed": self._uploaded_images,
                    "total": self.image_count,
                    "estimated_cost_usd": round(self.estimated_cost_usd, 4),
                }
            )
        return records
