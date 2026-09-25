"""Portable COCO/manifest exchange for CVAT."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Iterable
from pathlib import Path

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


def _validate_classes(classes: Iterable[str]) -> list[str]:
    names = [str(value).strip() for value in classes]
    if not names or any(not name for name in names):
        raise ValidationError("CVAT classes must contain at least one non-empty name")
    if len(names) != len(set(names)):
        raise ValidationError("CVAT classes must be unique")
    return names


def _read_json(path: Path, label: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"CVAT {label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"CVAT {label} is not valid JSON: {path}") from exc


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValidationError(
            "CVAT image validation requires Pillow; install 'amphilens[inference]'"
        ) from exc
    try:
        with Image.open(path) as image:
            return image.size
    except Exception as exc:  # noqa: BLE001 - normalize decoder-specific errors
        raise ValidationError(f"CVAT image cannot be read: {path}") from exc


def _task_filename(value: object) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValidationError(f"Invalid CVAT image filename: {value!r}")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"CVAT {label} must be a positive integer")
    return value


def _validate_cvat_payload(
    directory: Path, manifest: dict, coco: dict
) -> tuple[dict[int, dict], dict[int, str]]:
    classes = _validate_classes(manifest.get("classes", []))
    image_map: dict[int, dict] = {}
    for image in manifest.get("images", []):
        image_id = _positive_int(image.get("image_id"), "manifest image id")
        if image_id in image_map:
            raise ValidationError(f"CVAT manifest contains duplicate image id: {image_id}")
        filename = _task_filename(image.get("file_name"))
        width = _positive_int(image.get("width"), f"width for image {image_id}")
        height = _positive_int(image.get("height"), f"height for image {image_id}")
        source_path = image.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            raise ValidationError(f"CVAT source mapping is missing for image {image_id}")
        packaged = directory / "images" / filename
        if not packaged.is_file():
            raise ValidationError(f"CVAT image is missing: {packaged}")
        if _image_size(packaged) != (width, height):
            raise ValidationError(f"CVAT image dimensions do not match manifest: {filename}")
        image_map[image_id] = image

    coco_images: dict[int, dict] = {}
    for image in coco.get("images", []):
        image_id = _positive_int(image.get("id"), "COCO image id")
        if image_id in coco_images:
            raise ValidationError(f"CVAT COCO contains duplicate image id: {image_id}")
        if image_id not in image_map:
            raise ValidationError(f"CVAT COCO references unknown image id: {image_id}")
        if _task_filename(image.get("file_name")) != image_map[image_id]["file_name"]:
            raise ValidationError(f"CVAT image filename mappings do not match: {image_id}")
        if (
            image.get("width") != image_map[image_id]["width"]
            or image.get("height") != image_map[image_id]["height"]
        ):
            raise ValidationError(f"CVAT image dimensions do not match manifest: {image_id}")
        coco_images[image_id] = image
    if set(coco_images) != set(image_map):
        raise ValidationError("CVAT manifest and COCO image sets do not match")

    category_map: dict[int, str] = {}
    for category in coco.get("categories", []):
        category_id = _positive_int(category.get("id"), "category id")
        name = category.get("name")
        if category_id in category_map:
            raise ValidationError(f"CVAT contains duplicate category id: {category_id}")
        if not isinstance(name, str) or name not in classes:
            raise ValidationError(f"Unknown CVAT class: {name!r}")
        if name in category_map.values():
            raise ValidationError(f"CVAT contains duplicate class name: {name}")
        category_map[category_id] = name
    expected_categories = {index + 1: name for index, name in enumerate(classes)}
    if category_map != expected_categories:
        raise ValidationError("CVAT categories do not match the manifest classes")

    annotation_ids: set[int] = set()
    for annotation in coco.get("annotations", []):
        annotation_id = _positive_int(annotation.get("id"), "annotation id")
        if annotation_id in annotation_ids:
            raise ValidationError(f"CVAT contains duplicate annotation id: {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = annotation.get("image_id")
        if image_id not in image_map:
            raise ValidationError(f"CVAT annotation references unknown image id: {image_id}")
        category_id = annotation.get("category_id")
        if category_id not in category_map:
            raise ValidationError(f"CVAT annotation references unknown category id: {category_id}")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValidationError(f"CVAT annotation {annotation_id} has an invalid bbox")
        x, y, width, height = bbox
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in bbox):
            raise ValidationError(f"CVAT annotation {annotation_id} has an invalid bbox")
        image = image_map[image_id]
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValidationError(f"CVAT annotation {annotation_id} has an invalid bbox")
        if x + width > image["width"] or y + height > image["height"]:
            raise ValidationError(f"CVAT annotation {annotation_id} is outside image bounds")
    return image_map, category_map


def export_cvat(
    records: Iterable[DetectionRecord], output_dir: str | Path, *, classes: list[str]
) -> Path:
    classes = _validate_classes(classes)
    rows = sorted(
        list(records),
        key=lambda record: (
            str(Path(record.image_path).expanduser().resolve()),
            record.class_id,
            tuple(record.bbox_xyxy),
        ),
    )
    destination = Path(output_dir).expanduser().resolve()
    source_paths = {Path(record.image_path).expanduser().resolve() for record in rows}
    if any(destination == path or destination.is_relative_to(path) for path in source_paths):
        raise SourceCollisionError("CVAT export directory overlaps a source image")
    for record in rows:
        if record.class_name not in classes:
            raise ValidationError(f"Record class {record.class_name!r} is not present in classes")
    destination.mkdir(parents=True, exist_ok=True)
    images_dir = destination / "images"
    images_dir.mkdir(exist_ok=True)

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
        json.dumps(
            {"format": "amphilens-cvat-coco-v1", "classes": classes, "images": mapping}, indent=2
        )
        + "\n"
    )
    (destination / "classes.txt").write_text("\n".join(classes) + "\n")
    return destination


def import_cvat(task_dir: str | Path) -> list[DetectionRecord]:
    directory = Path(task_dir).expanduser().resolve()
    manifest = _read_json(directory / "manifest.json", "manifest")
    coco = _read_json(directory / "annotations.json", "annotations")
    image_map, category_map = _validate_cvat_payload(directory, manifest, coco)
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
    rows = list(records)
    destination = export_cvat(rows, output_dir, classes=classes)
    labels_dir = destination / "labels"
    labels_dir.mkdir(exist_ok=True)
    manifest = _read_json(destination / "manifest.json", "manifest")
    image_map = {item["source_path"]: item for item in manifest["images"]}
    grouped: dict[str, list[DetectionRecord]] = {}
    for record in rows:
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
            normalized = [float(center_x), float(center_y), float(box_width), float(box_height)]
            if any(value < 0 or value > 1 for value in normalized):
                raise ValidationError("YOLO normalized coordinates must be between 0 and 1")
            if (
                float(center_x) - float(box_width) / 2 < 0
                or float(center_y) - float(box_height) / 2 < 0
                or float(center_x) + float(box_width) / 2 > 1
                or float(center_y) + float(box_height) / 2 > 1
            ):
                raise ValidationError("YOLO normalized bounding box must remain within the image")
            if class_index < 0 or class_index >= len(classes):
                raise ValidationError(f"Unknown YOLO class id: {class_index}")
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
