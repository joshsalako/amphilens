"""Persistent active-learning queue and calibration artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .active_learning import HybridPPALConfig, PPALCalibration, SelectedImage


def write_selection_artifacts(
    selected: Iterable[SelectedImage],
    calibration: PPALCalibration,
    config: HybridPPALConfig,
    output_dir: str | Path,
) -> dict[str, Path]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = list(selected)
    queue_csv = destination / "selection_queue.csv"
    with queue_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_path", "class_name", "score", "curation_reason"]
        )
        writer.writeheader()
        for item in rows:
            writer.writerow(asdict(item))
    calibration_json = destination / "calibration.json"
    calibration_json.write_text(json.dumps(asdict(calibration), indent=2) + "\n")
    selection_json = destination / "selection.json"
    selection_json.write_text(
        json.dumps(
            {
                "strategy": "hybrid_ppal",
                "config": asdict(config),
                "calibration": asdict(calibration),
                "selected_count": len(rows),
                "selected": [asdict(item) for item in rows],
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "queue_csv": queue_csv,
        "calibration_json": calibration_json,
        "selection_json": selection_json,
    }

