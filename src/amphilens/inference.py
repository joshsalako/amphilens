"""Model-independent inference and result export services."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from .core import DetectionRecord, InferenceConfig


class Detector(Protocol):
    model_id: str

    def predict(
        self, image_paths: Iterable[Path], config: InferenceConfig
    ) -> Iterable[DetectionRecord]: ...


def run_inference(
    detector: Detector, image_paths: Iterable[str | Path], config: InferenceConfig
) -> Iterable[DetectionRecord]:
    paths = sorted(Path(path).expanduser().resolve() for path in image_paths)
    yield from detector.predict(paths, config)


CSV_FIELDS = [
    "image_path",
    "image_id",
    "class_id",
    "class_name",
    "confidence",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "bbox_xmin_norm",
    "bbox_ymin_norm",
    "bbox_xmax_norm",
    "bbox_ymax_norm",
    "image_width",
    "image_height",
    "model_id",
    "run_id",
    "cycle",
    "preprocessing",
]


def write_predictions_csv(records: Iterable[DetectionRecord], output: str | Path) -> Path:
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    return destination


def read_predictions_csv(path: str | Path) -> list[DetectionRecord]:
    source = Path(path).expanduser().resolve()
    records = []
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                DetectionRecord(
                    image_path=row["image_path"],
                    image_id=row["image_id"],
                    class_id=int(row["class_id"]),
                    class_name=row["class_name"],
                    confidence=float(row["confidence"]) if row.get("confidence") else None,
                    bbox_xyxy=[
                        float(row["bbox_xmin"]),
                        float(row["bbox_ymin"]),
                        float(row["bbox_xmax"]),
                        float(row["bbox_ymax"]),
                    ],
                    image_width=int(row["image_width"]),
                    image_height=int(row["image_height"]),
                    model_id=row.get("model_id", "csv-import"),
                    run_id=row.get("run_id", "csv-import"),
                    cycle=int(row["cycle"]) if row.get("cycle") else None,
                    preprocessing=row.get("preprocessing") or None,
                )
            )
    return records


def write_overlays(records: Iterable[DetectionRecord], output_dir: str | Path) -> list[Path]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Overlay generation requires the 'inference' extra") from exc

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[DetectionRecord]] = {}
    for record in records:
        grouped.setdefault(record.image_path, []).append(record)
    outputs: list[Path] = []
    for image_path, image_records in sorted(grouped.items()):
        source = Path(image_path)
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        for record in image_records:
            box = tuple(record.bbox_xyxy)
            label = (
                record.class_name
                if record.confidence is None
                else f"{record.class_name} {record.confidence:.3f}"
            )
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], max(0, box[1] - 14)), label, fill="red")
        output = destination / source.name
        image.save(output)
        outputs.append(output)
    return outputs
