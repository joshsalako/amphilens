"""Resumable inference artifacts and run summaries."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .core import DetectionRecord, InferenceConfig, SourceCollisionError
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


def _atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> list[DetectionRecord]:
    if not path.is_file():
        return []
    return [
        DetectionRecord.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line
    ]


def run_resumable_inference(
    detector,
    image_paths: Iterable[str | Path],
    config: InferenceConfig,
    artifact_dir: str | Path,
) -> RunSummary:
    paths = sorted(Path(path).expanduser().resolve() for path in image_paths)
    artifact = Path(artifact_dir).expanduser().resolve()
    for path in paths:
        if artifact == path or (path.is_dir() and artifact.is_relative_to(path)):
            raise SourceCollisionError(f"Inference artifact directory overlaps source: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    progress_path = artifact / "progress.json"
    records_path = artifact / "predictions.jsonl"
    progress = {
        "run_id": config.run_id,
        "model_id": config.model_id,
        "completed_images": [],
        "failed_images": {},
    }
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("run_id") != config.run_id or progress.get("model_id") != config.model_id:
            raise ValueError("Existing inference artifacts belong to a different run or model")
    completed = set(progress.get("completed_images", []))
    failures = dict(progress.get("failed_images", {}))
    records = _load_jsonl(records_path)
    recorded_paths = {record.image_path for record in records}

    for path in paths:
        image_path = str(path)
        if image_path in completed:
            continue
        try:
            new_records = list(detector.predict([path], config))
            with records_path.open("a", encoding="utf-8") as handle:
                for record in new_records:
                    handle.write(json.dumps(record.to_dict()) + "\n")
            records.extend(new_records)
            recorded_paths.add(image_path)
            completed.add(image_path)
            failures.pop(image_path, None)
        except Exception as exc:  # noqa: BLE001 - persisted as a user-visible image failure
            failures[image_path] = f"{type(exc).__name__}: {exc}"
        _atomic_json(
            progress_path,
            {
                "run_id": config.run_id,
                "model_id": config.model_id,
                "completed_images": sorted(completed),
                "failed_images": failures,
            },
        )

    predictions_csv = write_predictions_csv(records, artifact / "predictions.csv")
    summary = {
        "run_id": config.run_id,
        "model_id": config.model_id,
        "image_count": len(paths),
        "completed_images": len(completed),
        "failed_images": failures,
        "detection_count": len(records),
        "predictions_csv": str(predictions_csv),
    }
    summary_path = artifact / "summary.json"
    _atomic_json(summary_path, summary)
    return RunSummary(
        run_id=config.run_id,
        image_count=len(paths),
        completed_images=len(completed),
        failed_images=sorted(failures),
        detection_count=len(records),
        predictions_csv=predictions_csv,
        summary_json=summary_path,
    )
