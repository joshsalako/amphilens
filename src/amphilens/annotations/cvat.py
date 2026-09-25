"""Portable COCO/manifest exchange for CVAT."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

from ..core import DetectionRecord, SourceCollisionError, ValidationError


def _safe_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("_")
    return stem or "image"


def _mapped_name(source: Path, used: set[str]) -> str:
    candidate = _safe_filename(source.name)
    if candidate not in used:
        return candidate
    digest = hashlib.sha1(str(source).encode()).hexdigest()[:10]
    return f"{Path(candidate).stem}_{digest}{Path(candidate).suffix}"


def export_cvat(
    records: Iterable[DetectionRecord], output_dir: str | Path, *, classes: list[str]
) -> Path:
    rows = list(records)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    images_dir = destination / "images"
    images_dir.mkdir(exist_ok=True)
    source_paths = {Path(record.image_path).expanduser().resolve() for record in rows}
    if any(path == destination or destination.is_relative_to(path) for path in source_paths):
        raise SourceCollisionError("CVAT export directory overlaps a source image")

    image_ids: dict[str, int] = {}
    image_entries: list[dict] = []
    mapping: list[dict] = []
    used_names: set[str] = set()
    for record in rows:
        source = Path(record.image_path).expanduser().resolve()
        key = str(source)
        if key in image_ids:
            continue
        if not source.is_file():
            raise ValidationError(f"Source image does not exist: {source}")
        image_id = len(image_ids) + 1
        image_ids[key] = image_id
        filename = _mapped_name(source, used_names)
        used_names.add(filename)
        shutil.copy2(source, images_dir / filename)
        image_entries.append(
            {
                "id": image_id,
                "file_name": filename,
                "width": record.image_width,
                "height": record.image_height,
            }
        )
        mapping.append(
            {
                "image_id": image_id,
                "file_name": filename,
                "source_path": str(source),
                "width": record.image_width,
                "height": record.image_height,
            }
        )

    category_ids = {name: index + 1 for index, name in enumerate(classes)}
    annotations = []
    for index, record in enumerate(rows, start=1):
        x1, y1, x2, y2 = record.bbox_xyxy
        annotations.append(
            {
                "id": index,
                "image_id": image_ids[str(Path(record.image_path).expanduser().resolve())],
                "category_id": category_ids[record.class_name],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            }
        )
    coco = {
        "info": {"description": "AmphiLens CVAT task"},
        "images": image_entries,
        "annotations": annotations,
        "categories": [{"id": value, "name": name} for name, value in category_ids.items()],
    }
    (destination / "annotations.json").write_text(json.dumps(coco, indent=2) + "\n")
    (destination / "manifest.json").write_text(
        json.dumps({"format": "amphilens-cvat-coco-v1", "classes": classes, "images": mapping}, indent=2)
        + "\n"
    )
    (destination / "classes.txt").write_text("\n".join(classes) + "\n")
    return destination


def import_cvat(task_dir: str | Path) -> list[DetectionRecord]:
    directory = Path(task_dir).expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    coco = json.loads((directory / "annotations.json").read_text())
    image_map = {item["image_id"]: item for item in manifest["images"]}
    category_map = {item["id"]: item["name"] for item in coco["categories"]}
    records: list[DetectionRecord] = []
    for annotation in coco["annotations"]:
        image = image_map[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        records.append(
            DetectionRecord(
                image_path=image["source_path"],
                image_id=Path(image["source_path"]).name,
                class_id=annotation["category_id"] - 1,
                class_name=category_map[annotation["category_id"]],
                confidence=None,
                bbox_xyxy=[float(x), float(y), float(x + width), float(y + height)],
                image_width=int(image["width"]),
                image_height=int(image["height"]),
                model_id="human-cvat",
                run_id="annotation-import",
            )
        )
    return records


def export_yolo(
    records: Iterable[DetectionRecord], output_dir: str | Path, *, classes: list[str]
) -> Path:
    destination = export_cvat(records, output_dir, classes=classes)
    labels_dir = destination / "labels"
    labels_dir.mkdir(exist_ok=True)
    manifest = json.loads((destination / "manifest.json").read_text())
    image_map = {item["source_path"]: item for item in manifest["images"]}
    grouped: dict[str, list[DetectionRecord]] = {}
    for record in records:
        grouped.setdefault(str(Path(record.image_path).expanduser().resolve()), []).append(record)
    class_ids = {name: index for index, name in enumerate(classes)}
    for source_path, source_records in grouped.items():
        image = image_map[source_path]
        lines = []
        for record in source_records:
            x1, y1, x2, y2 = record.bbox_xyxy
            width, height = record.image_width, record.image_height
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            lines.append(
                f"{class_ids[record.class_name]} {center_x:.6g} {center_y:.6g} "
                f"{box_width:.6g} {box_height:.6g}"
            )
        (labels_dir / f"{Path(image['file_name']).stem}.txt").write_text("\n".join(lines) + "\n")
    (destination / "dataset.yaml").write_text(
        f"path: {destination}\ntrain: images\nval: images\nnames:\n"
        + "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
        + "\n"
    )
    return destination


def import_yolo(task_dir: str | Path) -> list[DetectionRecord]:
    directory = Path(task_dir).expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    classes = (directory / "classes.txt").read_text().splitlines()
    records: list[DetectionRecord] = []
    for image in manifest["images"]:
        label_path = directory / "labels" / f"{Path(image['file_name']).stem}.txt"
        if not label_path.is_file():
            continue
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                raise ValidationError(f"Invalid YOLO annotation line: {line}")
            class_id, center_x, center_y, box_width, box_height = parts
            class_index = int(class_id)
            width, height = int(image["width"]), int(image["height"])
            center_x, center_y = float(center_x) * width, float(center_y) * height
            box_width, box_height = float(box_width) * width, float(box_height) * height
            records.append(
                DetectionRecord(
                    image_path=image["source_path"],
                    image_id=Path(image["source_path"]).name,
                    class_id=class_index,
                    class_name=classes[class_index],
                    confidence=None,
                    bbox_xyxy=[
                        round(center_x - box_width / 2, 4),
                        round(center_y - box_height / 2, 4),
                        round(center_x + box_width / 2, 4),
                        round(center_y + box_height / 2, 4),
                    ],
                    image_width=width,
                    image_height=height,
                    model_id="human-yolo",
                    run_id="annotation-import",
                )
            )
    return records
