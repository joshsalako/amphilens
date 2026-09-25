"""Immutable, format-neutral annotated dataset snapshots."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import ValidationError, atomic_write_json, read_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classes(values: Iterable[str]) -> list[str]:
    result = [str(value).strip() for value in values]
    if not result or any(not value for value in result) or len(result) != len(set(result)):
        raise ValidationError("Dataset classes must be non-empty and unique")
    return result


@dataclass(frozen=True, slots=True)
class DatasetAnnotation:
    class_id: int
    class_name: str
    bbox_xyxy: list[float]

    def validate(self, classes: list[str], image_width: int, image_height: int) -> None:
        if self.class_id < 0 or self.class_id >= len(classes):
            raise ValidationError(f"Dataset annotation class id is out of range: {self.class_id}")
        if self.class_name != classes[self.class_id]:
            raise ValidationError("Dataset annotation class id and name do not match")
        if len(self.bbox_xyxy) != 4 or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in self.bbox_xyxy
        ):
            raise ValidationError("Dataset annotation bbox must contain four finite numbers")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValidationError("Dataset annotation bbox must be a positive box")
        if x2 > image_width or y2 > image_height:
            raise ValidationError("Dataset annotation bbox is outside image bounds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox_xyxy": list(self.bbox_xyxy),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetAnnotation:
        return cls(
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
            bbox_xyxy=[float(value) for value in data["bbox_xyxy"]],
        )


@dataclass(frozen=True, slots=True)
class DatasetImage:
    image_id: str
    relative_path: str
    source_path: str
    width: int
    height: int
    sha256: str
    reviewed: bool
    annotations: list[DatasetAnnotation] = field(default_factory=list)

    def validate(self, classes: list[str]) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValidationError("Dataset image paths must be relative and safe")
        if self.width <= 0 or self.height <= 0:
            raise ValidationError("Dataset image dimensions must be positive")
        if len(self.sha256) != 64:
            raise ValidationError("Dataset image sha256 must be a 64-character digest")
        for annotation in self.annotations:
            annotation.validate(classes, self.width, self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "relative_path": self.relative_path,
            "source_path": self.source_path,
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
            "reviewed": self.reviewed,
            "annotations": [annotation.to_dict() for annotation in self.annotations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetImage:
        return cls(
            image_id=str(data["image_id"]),
            relative_path=str(data["relative_path"]),
            source_path=str(data["source_path"]),
            width=int(data["width"]),
            height=int(data["height"]),
            sha256=str(data["sha256"]),
            reviewed=bool(data["reviewed"]),
            annotations=[DatasetAnnotation.from_dict(item) for item in data.get("annotations", [])],
        )


@dataclass(slots=True)
class DatasetManifest:
    snapshot_id: str
    source_format: str
    classes: list[str]
    images: list[DatasetImage]
    source_archive: str
    source_archive_sha256: str
    created_at: str = field(default_factory=_now)
    parent_snapshot: str | None = None
    schema_version: int = 1

    def validate(self) -> None:
        if not self.snapshot_id.strip() or self.schema_version != 1:
            raise ValidationError("Unsupported or missing dataset snapshot metadata")
        classes = _classes(self.classes)
        image_ids = [image.image_id for image in self.images]
        paths = [image.relative_path for image in self.images]
        hashes = [image.sha256 for image in self.images]
        if len(image_ids) != len(set(image_ids)):
            raise ValidationError("Dataset snapshot contains duplicate image ids")
        if len(paths) != len(set(paths)):
            raise ValidationError("Dataset snapshot contains duplicate image paths")
        if len(hashes) != len(set(hashes)):
            raise ValidationError("Dataset snapshot contains duplicate image content")
        for image in self.images:
            image.validate(classes)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "source_format": self.source_format,
            "classes": list(self.classes),
            "images": [image.to_dict() for image in self.images],
            "source_archive": self.source_archive,
            "source_archive_sha256": self.source_archive_sha256,
            "created_at": self.created_at,
            "parent_snapshot": self.parent_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        manifest = cls(
            snapshot_id=str(data["snapshot_id"]),
            source_format=str(data["source_format"]),
            classes=list(data["classes"]),
            images=[DatasetImage.from_dict(item) for item in data.get("images", [])],
            source_archive=str(data.get("source_archive", "")),
            source_archive_sha256=str(data.get("source_archive_sha256", "")),
            created_at=str(data.get("created_at", _now())),
            parent_snapshot=data.get("parent_snapshot"),
            schema_version=int(data.get("schema_version", 0)),
        )
        manifest.validate()
        return manifest


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    root: Path
    manifest: DatasetManifest

    @classmethod
    def load(cls, root: str | Path) -> DatasetSnapshot:
        path = Path(root).expanduser().resolve()
        manifest = DatasetManifest.from_dict(read_json(path / "manifest.json"))
        return cls(path, manifest)

    def to_yolo_dataset(
        self,
        destination: str | Path,
        *,
        preprocessing=None,
    ) -> Path:
        """Materialize a training directory without modifying the snapshot."""
        from .preprocessing import PreprocessingConfig, PreprocessingService

        target = Path(destination).expanduser().resolve()
        if target.exists():
            raise ValidationError(f"Prepared dataset destination already exists: {target}")
        image_dir = target / "images"
        label_dir = target / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir()
        service = PreprocessingService(PreprocessingConfig.from_any(preprocessing))
        for item in self.manifest.images:
            source = self.root / item.relative_path
            transformed = service.transform(source)
            filename = Path(item.relative_path).name
            transformed.image.save(image_dir / filename)
            width, height = transformed.processed_size
            lines = []
            for annotation in item.annotations:
                x1, y1, x2, y2 = transformed.map_box_to_processed(annotation.bbox_xyxy)
                box_width = x2 - x1
                box_height = y2 - y1
                center_x = x1 + box_width / 2
                center_y = y1 + box_height / 2
                lines.append(
                    f"{annotation.class_id} {center_x / width:.6f} {center_y / height:.6f} "
                    f"{box_width / width:.6f} {box_height / height:.6f}"
                )
            (label_dir / f"{Path(filename).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
        yaml_path = target / "dataset.yaml"
        yaml_path.write_text(
            json.dumps(
                {
                    "path": str(target),
                    "train": "images",
                    "labels": "labels",
                    "names": {index: name for index, name in enumerate(self.manifest.classes)},
                    "evaluation": "not evaluated",
                    "preprocessing": service.config.to_dict(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return yaml_path


class DatasetImporter:
    """Import CVAT XML, COCO, or YOLO ZIP archives into immutable snapshots."""

    def import_archive(
        self,
        archive_path: str | Path,
        destination: str | Path,
        *,
        classes: Iterable[str],
        class_mapping: dict[str, str] | None = None,
    ) -> DatasetSnapshot:
        archive = Path(archive_path).expanduser().resolve()
        if not archive.is_file() or archive.suffix.lower() != ".zip":
            raise ValidationError("Dataset import requires a ZIP archive")
        target = Path(destination).expanduser().resolve()
        if target.exists():
            raise ValidationError(f"Dataset snapshot destination already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target_id = f"snapshot-{_sha256(archive)[:12]}"
        target = target.parent / target_id
        if target.exists():
            raise ValidationError(f"Dataset snapshot already exists: {target}")
        expected_classes = _classes(classes)
        with tempfile.TemporaryDirectory(prefix="amphilens-import-") as temporary:
            extracted = Path(temporary)
            self._extract(archive, extracted)
            images = self._find_images(extracted)
            if not images:
                raise ValidationError("Dataset archive must include image files")
            source_format, parsed = self._parse(extracted, images, expected_classes, class_mapping)
            manifest_images = self._copy_images(parsed, target)
        manifest = DatasetManifest(
            snapshot_id=target_id,
            source_format=source_format,
            classes=expected_classes,
            images=manifest_images,
            source_archive=archive.name,
            source_archive_sha256=_sha256(archive),
        )
        atomic_write_json(target / "manifest.json", manifest.to_dict())
        return DatasetSnapshot(target, manifest)

    @staticmethod
    def _extract(archive: Path, destination: Path) -> None:
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            if len(names) != len(set(names)):
                raise ValidationError("Dataset archive contains duplicate filenames")
            for name in names:
                path = Path(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValidationError("Dataset archive contains an unsafe filename")
            handle.extractall(destination)

    @staticmethod
    def _find_images(root: Path) -> list[Path]:
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _parse(
        self,
        root: Path,
        images: list[Path],
        expected_classes: list[str],
        class_mapping: dict[str, str] | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        xml_files = sorted(root.rglob("*.xml"))
        if xml_files:
            return "cvat-xml", self._parse_cvat(
                xml_files[0], root, images, expected_classes, class_mapping
            )
        json_files = sorted(root.rglob("*.json"))
        for path in json_files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if {"images", "annotations", "categories"}.issubset(data):
                return "coco", self._parse_coco(data, root, images, expected_classes, class_mapping)
        return "yolo", self._parse_yolo(root, images, expected_classes, class_mapping)

    def _class_map(
        self,
        source_classes: list[str],
        expected_classes: list[str],
        class_mapping: dict[str, str] | None,
    ) -> dict[str, str]:
        mapping = dict(class_mapping or {})
        for source in source_classes:
            destination = mapping.get(source, source)
            if destination not in expected_classes:
                raise ValidationError(
                    f"Dataset classes require an explicit class mapping: {source!r}"
                )
            mapping[source] = destination
        return mapping

    @staticmethod
    def _image_lookup(root: Path, images: list[Path]) -> dict[str, Path]:
        lookup: dict[str, Path] = {}
        for image in images:
            relative = image.relative_to(root).as_posix()
            lookup[relative] = image
            lookup.setdefault(image.name, image)
        return lookup

    def _parse_cvat(self, xml_path, root, images, expected_classes, class_mapping):
        tree = ET.parse(xml_path)
        labels = [item.text.strip() for item in tree.findall(".//label/name") if item.text]
        mapping = self._class_map(
            _classes(labels) if labels else expected_classes,
            expected_classes,
            class_mapping,
        )
        lookup = self._image_lookup(root, images)
        parsed: list[dict[str, Any]] = []
        used_hashes: set[str] = set()
        for image_node in tree.findall(".//image"):
            source_name = image_node.attrib.get("name", "")
            image = self._resolve_image(source_name, lookup)
            width = self._positive_int(image_node.attrib.get("width"), "image width")
            height = self._positive_int(image_node.attrib.get("height"), "image height")
            boxes = [
                {
                    "label": box.attrib.get("label", ""),
                    "xtl": box.attrib.get("xtl"),
                    "ytl": box.attrib.get("ytl"),
                    "xbr": box.attrib.get("xbr"),
                    "ybr": box.attrib.get("ybr"),
                }
                for box in image_node.findall("box")
            ]
            parsed.append(
                self._parsed_image(
                    image,
                    root,
                    width,
                    height,
                    boxes,
                    mapping,
                    expected_classes,
                    used_hashes,
                )
            )
        if not parsed:
            raise ValidationError("CVAT XML contains no image annotations")
        return parsed

    def _parse_coco(self, data, root, images, expected_classes, class_mapping):
        lookup = self._image_lookup(root, images)
        categories = {int(item["id"]): str(item["name"]) for item in data["categories"]}
        mapping = self._class_map(list(categories.values()), expected_classes, class_mapping)
        annotations: dict[int, list[dict[str, Any]]] = {}
        for item in data["annotations"]:
            annotations.setdefault(int(item["image_id"]), []).append(item)
        parsed = []
        used_hashes: set[str] = set()
        for item in data["images"]:
            image = self._resolve_image(str(item["file_name"]), lookup)
            width = self._positive_int(item.get("width"), "image width")
            height = self._positive_int(item.get("height"), "image height")
            boxes = []
            for annotation in annotations.get(int(item["id"]), []):
                category = categories.get(int(annotation["category_id"]))
                if category is None:
                    raise ValidationError("COCO annotation references an unknown category")
                x, y, box_width, box_height = self._bbox(annotation.get("bbox"))
                boxes.append(
                    {
                        "label": category,
                        "xtl": x,
                        "ytl": y,
                        "xbr": x + box_width,
                        "ybr": y + box_height,
                    }
                )
            parsed.append(
                self._parsed_image(
                    image, root, width, height, boxes, mapping, expected_classes, used_hashes
                )
            )
        return parsed

    def _parse_yolo(self, root, images, expected_classes, class_mapping):
        names = self._yolo_classes(root)
        mapping = self._class_map(names, expected_classes, class_mapping)
        label_files = {
            path.stem: path for path in root.rglob("*.txt") if path.name != "classes.txt"
        }
        parsed = []
        used_hashes: set[str] = set()
        for image in images:
            width, height = self._actual_size(image)
            boxes = []
            label = label_files.get(image.stem)
            if label:
                for line_number, line in enumerate(
                    label.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValidationError(
                            f"YOLO label has five fields at line {line_number}: {label}"
                        )
                    class_index = int(parts[0])
                    if class_index < 0 or class_index >= len(names):
                        raise ValidationError(f"YOLO class id is out of range: {class_index}")
                    center_x, center_y, box_width, box_height = (
                        float(value) for value in parts[1:]
                    )
                    if any(
                        value < 0 or value > 1
                        for value in (center_x, center_y, box_width, box_height)
                    ):
                        raise ValidationError("YOLO coordinates must be between 0 and 1")
                    x1 = (center_x - box_width / 2) * width
                    y1 = (center_y - box_height / 2) * height
                    x2 = (center_x + box_width / 2) * width
                    y2 = (center_y + box_height / 2) * height
                    boxes.append(
                        {
                            "label": names[class_index],
                            "xtl": x1,
                            "ytl": y1,
                            "xbr": x2,
                            "ybr": y2,
                        }
                    )
            parsed.append(
                self._parsed_image(
                    image, root, width, height, boxes, mapping, expected_classes, used_hashes
                )
            )
        return parsed

    @staticmethod
    def _yolo_classes(root: Path) -> list[str]:
        classes_file = next(iter(sorted(root.rglob("classes.txt"))), None)
        if classes_file:
            return _classes(classes_file.read_text(encoding="utf-8").splitlines())
        yaml_files = list(root.rglob("dataset.yaml")) + list(root.rglob("data.yaml"))
        yaml_file = next(iter(sorted(yaml_files)), None)
        if yaml_file:
            try:
                import yaml

                data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
                names = data.get("names")
                if isinstance(names, dict):
                    return _classes(
                        names[index] for index in sorted(names, key=lambda value: int(value))
                    )
                if isinstance(names, list):
                    return _classes(names)
            except ImportError:
                text = yaml_file.read_text(encoding="utf-8")
                match = re.search(r"names:\s*\[(.*?)\]", text, flags=re.DOTALL)
                if match:
                    return _classes(ast.literal_eval("[" + match.group(1) + "]"))
        raise ValidationError("YOLO archive must include classes.txt or dataset.yaml")

    def _parsed_image(
        self, image, root, width, height, boxes, mapping, expected_classes, used_hashes
    ):
        actual_width, actual_height = self._actual_size(image)
        if (width, height) != (actual_width, actual_height):
            raise ValidationError(f"Dataset image dimensions do not match: {image.name}")
        digest = _sha256(image)
        if digest in used_hashes:
            raise ValidationError("Dataset contains duplicate image content")
        used_hashes.add(digest)
        annotations = []
        for box in boxes:
            label = str(box["label"])
            target = mapping.get(label, label)
            if target not in expected_classes:
                raise ValidationError(f"Dataset annotation class is not configured: {target}")
            x1, y1, x2, y2 = (float(box[key]) for key in ("xtl", "ytl", "xbr", "ybr"))
            annotation = DatasetAnnotation(
                expected_classes.index(target),
                target,
                [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)],
            )
            annotation.validate(expected_classes, width, height)
            annotations.append(annotation)
        return {
            "source": image,
            "source_name": image.relative_to(root).as_posix(),
            "width": width,
            "height": height,
            "sha256": digest,
            "annotations": annotations,
        }

    @staticmethod
    def _resolve_image(name: str, lookup: dict[str, Path]) -> Path:
        clean = name.lstrip("/").replace("\\", "/")
        if clean in lookup:
            return lookup[clean]
        basename = Path(clean).name
        if basename in lookup:
            return lookup[basename]
        raise ValidationError(f"Dataset annotation references missing image: {name}")

    @staticmethod
    def _positive_int(value: object, label: str) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Dataset {label} must be a positive integer") from exc
        if number <= 0:
            raise ValidationError(f"Dataset {label} must be a positive integer")
        return number

    @staticmethod
    def _bbox(value: object) -> tuple[float, float, float, float]:
        if not isinstance(value, list) or len(value) != 4:
            raise ValidationError("COCO annotation bbox must contain four values")
        try:
            values = tuple(float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValidationError("COCO annotation bbox is invalid") from exc
        if any(not math.isfinite(item) for item in values) or values[2] <= 0 or values[3] <= 0:
            raise ValidationError("COCO annotation bbox is invalid")
        return values

    @staticmethod
    def _actual_size(path: Path) -> tuple[int, int]:
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                return image.size
        except Exception as exc:  # noqa: BLE001 - normalize decoder errors
            raise ValidationError(f"Dataset image cannot be read: {path.name}") from exc

    @staticmethod
    def _copy_images(parsed: list[dict[str, Any]], target: Path) -> list[DatasetImage]:
        image_dir = target / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        result: list[DatasetImage] = []
        for index, item in enumerate(
            sorted(parsed, key=lambda value: value["source_name"]), start=1
        ):
            source = item["source"]
            name = Path(item["source_name"]).name
            if name in used_names:
                name = f"{Path(name).stem}_{item['sha256'][:10]}{Path(name).suffix}"
            used_names.add(name)
            destination = image_dir / name
            shutil.copy2(source, destination)
            result.append(
                DatasetImage(
                    image_id=f"image-{index:06d}",
                    relative_path=destination.relative_to(target).as_posix(),
                    source_path=item["source_name"],
                    width=item["width"],
                    height=item["height"],
                    sha256=item["sha256"],
                    reviewed=True,
                    annotations=item["annotations"],
                )
            )
        return result


class DatasetMerger:
    """Create a new snapshot by unioning immutable parent and incoming snapshots."""

    def merge(
        self,
        parent: DatasetSnapshot,
        incoming: DatasetSnapshot,
        destination: str | Path,
    ) -> DatasetSnapshot:
        target = Path(destination).expanduser().resolve()
        if target.exists():
            raise ValidationError(f"Dataset snapshot destination already exists: {target}")
        if parent.manifest.classes != incoming.manifest.classes:
            raise ValidationError("Dataset snapshots have incompatible classes")
        target.mkdir(parents=True, exist_ok=False)
        image_dir = target / "images"
        image_dir.mkdir()
        by_hash: dict[str, tuple[DatasetSnapshot, DatasetImage]] = {}
        for snapshot in (parent, incoming):
            for image in snapshot.manifest.images:
                by_hash[image.sha256] = (snapshot, image)
        images: list[DatasetImage] = []
        used_names: set[str] = set()
        for index, (snapshot, image) in enumerate(
            sorted(by_hash.values(), key=lambda item: item[1].source_path), start=1
        ):
            name = Path(image.relative_path).name
            if name in used_names:
                name = f"{Path(name).stem}_{image.sha256[:10]}{Path(name).suffix}"
            used_names.add(name)
            relative = f"images/{name}"
            shutil.copy2(snapshot.root / image.relative_path, target / relative)
            images.append(
                DatasetImage(
                    image_id=f"image-{index:06d}",
                    relative_path=relative,
                    source_path=image.source_path,
                    width=image.width,
                    height=image.height,
                    sha256=image.sha256,
                    reviewed=image.reviewed,
                    annotations=list(image.annotations),
                )
            )
        source_hash = hashlib.sha256(
            (parent.manifest.snapshot_id + incoming.manifest.snapshot_id).encode("utf-8")
        ).hexdigest()
        manifest = DatasetManifest(
            snapshot_id=f"merged-{source_hash[:12]}",
            source_format="merged",
            classes=list(parent.manifest.classes),
            images=images,
            source_archive=f"{parent.manifest.snapshot_id}+{incoming.manifest.snapshot_id}",
            source_archive_sha256=source_hash,
            parent_snapshot=parent.manifest.snapshot_id,
        )
        atomic_write_json(target / "manifest.json", manifest.to_dict())
        return DatasetSnapshot(target, manifest)
