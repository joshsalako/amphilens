"""Discover saved prediction results and restore their existing job actions."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .core import ProjectStore, RunManifest, atomic_write_json
from .webapp_jobs import DownloadArtifact, JobOutput

_COMPLETE = {"completed", "completed_with_failures", "completed_with_skips"}


def _index_dir(store: ProjectStore) -> Path:
    return store.root / ".amphilens" / "prediction-runs"


def _manifest(folder: Path, *, completed: bool = True) -> RunManifest:
    try:
        manifest = RunManifest.from_dict(json.loads((folder / "run.json").read_text()))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError("Prediction run metadata is missing or invalid") from exc
    if completed and (manifest.kind != "inference" or manifest.status not in _COMPLETE):
        raise ValueError("Choose a completed prediction run folder")
    return manifest


def remember_prediction(store: ProjectStore, folder: Path) -> None:
    folder = folder.expanduser().resolve()
    manifest = _manifest(folder)
    if folder.parent == (store.root / "runs").resolve():
        return
    key = hashlib.sha256(str(folder).encode()).hexdigest()
    metadata = manifest.config.get("metadata") or {}
    stored_path = (
        str(folder.relative_to(store.root)) if folder.is_relative_to(store.root) else str(folder)
    )
    atomic_write_json(
        _index_dir(store) / f"{key}.json",
        {
            "path": stored_path,
            "name": metadata.get("run_name") or folder.name,
            "finished_at": manifest.finished_at,
        },
    )


def saved_predictions(store: ProjectStore) -> list[dict]:
    folders = {path.resolve(): {} for path in (store.root / "runs").glob("*") if path.is_dir()}
    for entry in _index_dir(store).glob("*.json"):
        try:
            data = json.loads(entry.read_text())
            path = Path(data["path"]).expanduser()
            folders[(path if path.is_absolute() else store.root / path).resolve()] = data
        except (OSError, ValueError, TypeError, KeyError):
            continue
    results = []
    for folder, saved in folders.items():
        try:
            manifest = _manifest(folder, completed=False)
        except ValueError:
            if saved:
                results.append(
                    {
                        **saved,
                        "path": str(folder),
                        "error": "Run folder or metadata is missing or invalid.",
                    }
                )
            elif (folder / "summary.json").exists() or (folder / "predictions.csv").exists():
                results.append(
                    {
                        "path": str(folder),
                        "name": folder.name,
                        "error": "Run metadata is missing or invalid.",
                    }
                )
            continue
        if manifest.kind != "inference" or manifest.status not in _COMPLETE:
            continue
        metadata = manifest.config.get("metadata") or {}
        try:
            summary = json.loads((folder / "summary.json").read_text())
            if summary.get("run_id") != manifest.run_id:
                raise ValueError("Run summary does not match metadata")
        except (OSError, ValueError, TypeError, AttributeError):
            summary = {}
        results.append(
            {
                "path": str(folder),
                "name": metadata.get("run_name") or folder.name,
                "finished_at": manifest.finished_at or manifest.started_at,
                "image_count": summary.get("image_count"),
                "detection_count": summary.get("detection_count"),
                "failed_count": len(summary.get("failed_images") or {}),
                "skipped_count": len(summary.get("skipped_images") or {}),
                "error": "Run summary or predictions CSV is missing or invalid."
                if not summary or not (folder / "predictions.csv").is_file()
                else "",
            }
        )
    return sorted(results, key=lambda item: item.get("finished_at") or "", reverse=True)


def restore_prediction(store: ProjectStore, folder: Path) -> JobOutput:
    folder = folder.expanduser().resolve()
    manifest = _manifest(folder)
    try:
        summary = json.loads((folder / "summary.json").read_text())
        if summary.get("run_id") != manifest.run_id:
            raise ValueError("Run summary does not match metadata")
        csv_path = folder / "predictions.csv"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "image_path" not in reader.fieldnames:
                raise ValueError("Prediction CSV is invalid")
            timestamp_done = {"date", "time"}.issubset(reader.fieldnames)
            unresolved = (
                sum(1 for row in reader if not row.get("date") or not row.get("time"))
                if timestamp_done
                else 0
            )
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Run summary or predictions CSV is missing or invalid") from exc
    metadata = manifest.config.get("metadata") or {}
    image_root = metadata.get("image_root") or ""
    name = metadata.get("run_name") or folder.name
    downloads = [DownloadArtifact(csv_path, "Predictions CSV", "predictions.csv", "text/csv")]
    for filename, label, download, media in (
        ("report.md", "Summary report", "report.md", "text/markdown"),
        ("summary.json", "Report data", "report.json", "application/json"),
    ):
        downloads.append(DownloadArtifact(folder / "report" / filename, label, download, media))
    failed = len(summary.get("failed_images") or {})
    skipped = len(summary.get("skipped_images") or {})
    message = (
        f"Processed {summary.get('completed_images', 0)} of {summary.get('image_count', 0)} "
        f"images; found {summary.get('detection_count', 0)} detections. "
        f"{failed} failed; {skipped} skipped."
    )
    return JobOutput(
        result={
            "message": message,
            "paths": {"csv": str(csv_path), "run_summary": str(folder / "summary.json")},
            "prediction": {
                "run_id": manifest.run_id,
                "run_name": name,
                "csv": str(csv_path),
                "project_root": str(store.root),
                "image_root": image_root,
                "image_root_available": bool(image_root and Path(image_root).is_dir()),
                "detection_count": summary.get("detection_count", 0),
                "execution": metadata.get("execution", "local"),
                "gpu": metadata.get("gpu"),
                "compute_device": manifest.config.get("device", "cpu"),
                "timestamps_extracted": timestamp_done,
                "timestamp_unresolved_rows": unresolved,
                "missing_reports": not all(
                    (folder / "report" / name).is_file() for name in ("report.md", "summary.json")
                ),
            },
        },
        downloads=downloads,
    )
