"""Model-independent inference and result export services."""

from __future__ import annotations

import csv
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
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
    "run_name",
    "model_id",
    "run_id",
    "cycle",
    "preprocessing",
]


def write_predictions_csv(
    records: Iterable[DetectionRecord], output: str | Path, *, run_name: str | None = None
) -> Path:
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = record.to_row()
            row["run_name"] = run_name or ""
            writer.writerow(row)
    return destination


@dataclass(frozen=True, slots=True)
class ImageCollectionSummary:
    destination: Path
    copied_count: int
    already_present_count: int
    missing_count: int


def collect_prediction_images(
    predictions_csv: str | Path,
    image_root: str | Path,
    destination: str | Path,
) -> ImageCollectionSummary:
    """Copy unique source images referenced by prediction rows into a managed folder."""
    csv_path = Path(predictions_csv).expanduser().resolve()
    source_root = Path(image_root).expanduser().resolve()
    output_root = Path(destination).expanduser().absolute()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Predictions CSV was not found: {csv_path}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Image folder was not found: {source_root}")
    if (
        output_root == source_root
        or output_root.is_relative_to(source_root)
        or source_root.is_relative_to(output_root)
    ):
        raise ValueError("Image collection folder must not overlap the source image folder")
    if output_root != output_root.resolve():
        raise ValueError("Image collection folder must not contain symlinks")

    unique_sources: dict[Path, Path] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError("Predictions CSV must contain an image_path column")
        for row in reader:
            value = row.get("image_path")
            if not value:
                raise ValueError("Predictions CSV contains an empty image_path")
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = source_root / candidate
            source = candidate.resolve()
            try:
                relative_path = source.relative_to(source_root)
            except ValueError as exc:
                raise ValueError("Prediction image is outside the selected image folder") from exc
            unique_sources[source] = relative_path

    output_root.mkdir(parents=True, exist_ok=True)
    resolved_output_root = output_root.resolve()
    copied_count = 0
    already_present_count = 0
    missing_count = 0
    for source, relative_path in unique_sources.items():
        target_parent = output_root
        for part in relative_path.parts[:-1]:
            target_parent = target_parent / part
            if target_parent.is_symlink():
                raise ValueError("Image collection destination contains a symlink")
            target_parent.mkdir(exist_ok=True)
            if not target_parent.resolve().is_relative_to(resolved_output_root):
                raise ValueError("Image collection destination is outside its collection folder")
        target = target_parent / relative_path.name
        if target.is_symlink():
            raise ValueError("Image collection destination contains a symlink")
        if target.exists():
            if not target.is_file():
                raise ValueError("Existing image collection target is not a regular file")
            already_present_count += 1
            continue
        if not source.is_file():
            missing_count += 1
            continue
        shutil.copy2(source, target)
        copied_count += 1

    return ImageCollectionSummary(
        destination=output_root,
        copied_count=copied_count,
        already_present_count=already_present_count,
        missing_count=missing_count,
    )


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
