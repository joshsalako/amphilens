"""Resumable inference artifacts and run summaries."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from .core import (
    DetectionRecord,
    InferenceConfig,
    RunManifest,
    SourceCollisionError,
    atomic_write_json,
    read_json,
    utc_now,
)
from .inference import write_predictions_csv


@dataclass(slots=True)
class RunSummary:
    run_id: str
    image_count: int
    completed_images: int
    failed_images: list[str]
    detection_count: int
    predictions_csv: Path
    summary_json: Path
    run_manifest: Path
    interruption_reason: str | None = None


class InferenceInterruption(RuntimeError):
    """Stop before scheduling more images while preserving completed progress."""


def _load_jsonl(path: Path) -> list[DetectionRecord]:
    if not path.is_file():
        return []
    return [
        DetectionRecord.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line
    ]


def _is_out_of_memory(error: Exception) -> bool:
    name = type(error).__name__.lower()
    message = str(error).lower()
    return "outofmemory" in name or "out of memory" in message or "memory allocation" in message


def _predict_resiliently(detector, paths: list[Path], config: InferenceConfig):
    """Yield successful sub-batches and per-image failures after shrinking failed batches."""
    try:
        batch_config = replace(config, batch_size=len(paths))
        return [(paths, list(detector.predict(paths, batch_config)), None)]
    except Exception as exc:  # noqa: BLE001 - isolated below and persisted per image
        if isinstance(exc, InferenceInterruption):
            raise
        if len(paths) == 1:
            return [(paths, [], exc)]
        if _is_out_of_memory(exc):
            midpoint = max(1, len(paths) // 2)
            return _predict_resiliently(detector, paths[:midpoint], config) + _predict_resiliently(
                detector, paths[midpoint:], config
            )
        return [
            result for path in paths for result in _predict_resiliently(detector, [path], config)
        ]


def _report_inference_progress(
    callback: Callable[[dict], None] | None,
    *,
    phase: str,
    message: str,
    total: int,
    completed: int,
    failed: int,
    **details,
) -> None:
    if callback is None:
        return
    remaining = max(0, total - completed - failed)
    callback(
        {
            "phase": phase,
            "message": message,
            "completed": completed,
            "failed": failed,
            "total": total,
            "remaining": remaining,
            "progress": (completed + failed) / total if total else 1.0,
            **details,
        }
    )


def run_resumable_inference(
    detector,
    image_paths: Iterable[str | Path],
    config: InferenceConfig,
    artifact_dir: str | Path,
    *,
    progress_callback: Callable[[dict], None] | None = None,
) -> RunSummary:
    paths = sorted(Path(path).expanduser().resolve() for path in image_paths)
    artifact = Path(artifact_dir).expanduser().resolve()
    for path in paths:
        if artifact == path or (path.is_dir() and artifact.is_relative_to(path)):
            raise SourceCollisionError(f"Inference artifact directory overlaps source: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    if not config.run_id:
        config.run_id = f"run-{uuid.uuid4().hex[:12]}"
    run_manifest_path = artifact / "run.json"
    if run_manifest_path.is_file():
        run_manifest = RunManifest.from_dict(read_json(run_manifest_path))
        if (
            run_manifest.run_id != config.run_id
            or run_manifest.config.get("model_id") != config.model_id
        ):
            raise ValueError("Existing run metadata belongs to a different run or model")
    else:
        run_manifest = RunManifest.create("inference", config.to_dict())
        run_manifest.run_id = config.run_id
        atomic_write_json(run_manifest_path, run_manifest.to_dict())
    progress_path = artifact / "progress.json"
    records_path = artifact / "predictions.jsonl"
    progress = {
        "run_id": config.run_id,
        "model_id": config.model_id,
        "completed_images": [],
        "failed_images": {},
    }
    if progress_path.is_file():
        progress = read_json(progress_path)
        if progress.get("run_id") != config.run_id or progress.get("model_id") != config.model_id:
            raise ValueError("Existing inference artifacts belong to a different run or model")
    completed = set(progress.get("completed_images", []))
    failures = dict(progress.get("failed_images", {}))
    records = _load_jsonl(records_path)
    current_paths = {str(path) for path in paths}
    pending = [path for path in paths if str(path) not in completed]
    attempted_this_run: set[str] = set()
    batch_size = min(config.batch_size, 32)
    if config.device == "cpu":
        batch_size = 1
    effective_batch_size = int(progress.get("effective_batch_size", 0))

    _report_inference_progress(
        progress_callback,
        phase="prediction",
        message="Preparing image prediction",
        total=len(paths),
        completed=sum(path in completed for path in current_paths),
        failed=0,
        device=getattr(detector, "device_name", config.device),
        effective_batch_size=effective_batch_size or batch_size,
    )

    interruption_reason = None
    for offset in range(0, len(pending), batch_size):
        requested_paths = pending[offset : offset + batch_size]
        try:
            results = _predict_resiliently(detector, requested_paths, config)
        except InferenceInterruption as exc:
            interruption_reason = str(exc)
            break
        for successful_paths, new_records, error in results:
            if error is None:
                with records_path.open("a", encoding="utf-8") as handle:
                    for record in new_records:
                        handle.write(json.dumps(record.to_dict()) + "\n")
                records.extend(new_records)
                completed.update(str(path) for path in successful_paths)
                for path in successful_paths:
                    failures.pop(str(path), None)
                attempted_this_run.update(str(path) for path in successful_paths)
                effective_batch_size = max(effective_batch_size, len(successful_paths))
            else:
                failures[str(successful_paths[0])] = f"{type(error).__name__}: {error}"
                attempted_this_run.add(str(successful_paths[0]))
            atomic_write_json(
                progress_path,
                {
                    "run_id": config.run_id,
                    "model_id": config.model_id,
                    "completed_images": sorted(completed),
                    "failed_images": failures,
                    "requested_batch_size": config.batch_size,
                    "effective_batch_size": effective_batch_size,
                },
            )
            attempted_failures = sum(
                1 for image in attempted_this_run if image in failures and image not in completed
            )
            completed_count = sum(str(path) in completed for path in paths)
            _report_inference_progress(
                progress_callback,
                phase="prediction",
                message=f"Predicted {completed_count} of {len(paths)} images",
                total=len(paths),
                completed=completed_count,
                failed=attempted_failures,
                device=getattr(detector, "device_name", config.device),
                effective_batch_size=effective_batch_size or batch_size,
            )

    _report_inference_progress(
        progress_callback,
        phase="saving_results",
        message="Saving prediction results",
        total=len(paths),
        completed=sum(str(path) in completed for path in paths),
        failed=sum(
            1 for image in attempted_this_run if image in failures and image not in completed
        ),
        device=getattr(detector, "device_name", config.device),
        effective_batch_size=effective_batch_size or batch_size,
    )
    run_name = config.metadata.get("run_name") if isinstance(config.metadata, dict) else None
    predictions_csv = write_predictions_csv(
        records, artifact / "predictions.csv", run_name=run_name
    )
    summary = {
        "run_id": config.run_id,
        "model_id": config.model_id,
        "image_count": len(paths),
        "completed_images": len(completed),
        "failed_images": failures,
        "detection_count": len(records),
        "requested_batch_size": config.batch_size,
        "effective_batch_size": effective_batch_size,
        "status": "interrupted"
        if interruption_reason
        else ("completed_with_failures" if failures else "completed"),
        "interruption_reason": interruption_reason,
        "predictions_csv": str(predictions_csv),
    }
    summary_path = artifact / "summary.json"
    run_manifest.status = summary["status"]
    run_manifest.finished_at = utc_now()
    atomic_write_json(run_manifest_path, run_manifest.to_dict())
    atomic_write_json(summary_path, summary)
    return RunSummary(
        run_id=config.run_id,
        image_count=len(paths),
        completed_images=len(completed),
        failed_images=sorted(failures),
        detection_count=len(records),
        predictions_csv=predictions_csv,
        summary_json=summary_path,
        run_manifest=run_manifest_path,
        interruption_reason=interruption_reason,
    )
