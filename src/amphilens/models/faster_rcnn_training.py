"""Optional Faster R-CNN training over a portable YOLO directory layout."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import ValidationError, atomic_write_json
from .backends import OptionalDependencyError, _select_torch_device

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_CLASSIFIER_HEAD_PREFIXES = (
    "roi_heads.box_predictor.cls_score.",
    "roi_heads.box_predictor.bbox_pred.",
)


def adapt_faster_rcnn_state_dict(
    source_state: dict[str, Any],
    target_state: dict[str, Any],
    *,
    source_classes: list[str] | tuple[str, ...] | None,
    target_classes: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Keep compatible Faster R-CNN weights and leave an incompatible output head fresh."""
    reset_head = source_classes is not None and list(source_classes) != list(target_classes)
    compatible: dict[str, Any] = {}
    for key, value in source_state.items():
        if key not in target_state:
            continue
        is_head = key.startswith(_CLASSIFIER_HEAD_PREFIXES)
        if is_head and reset_head:
            continue
        source_shape = getattr(value, "shape", None)
        target_shape = getattr(target_state[key], "shape", None)
        if source_shape != target_shape:
            if is_head:
                continue
            raise ValueError(f"Faster R-CNN checkpoint shape differs for {key}")
        compatible[key] = value
    return compatible


def _classes_from_names(value: Any) -> list[str]:
    if isinstance(value, dict):
        try:
            items = sorted(value.items(), key=lambda item: int(item[0]))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Dataset class names must use integer ids") from exc
        names = [str(name).strip() for _, name in items]
    elif isinstance(value, list):
        names = [str(name).strip() for name in value]
    else:
        raise ValidationError("Dataset YAML must define class names as a list or mapping")
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValidationError("Dataset class names must be non-empty and unique")
    return names


def parse_yolo_label_lines(
    lines: list[str], image_width: int, image_height: int, *, class_count: int
) -> list[tuple[list[float], int]]:
    """Convert YOLO normalized labels into pixel boxes and Faster R-CNN labels."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    parsed: list[tuple[list[float], int]] = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO annotation line: {line}")
        try:
            class_id = int(parts[0])
            center_x, center_y, box_width, box_height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO annotation line: {line}") from exc
        values = (center_x, center_y, box_width, box_height)
        if class_id < 0 or class_id >= class_count:
            raise ValueError(f"Unknown YOLO class id: {class_id}")
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
            raise ValueError("YOLO normalized coordinates must be between 0 and 1")
        x1 = center_x - box_width / 2
        y1 = center_y - box_height / 2
        x2 = center_x + box_width / 2
        y2 = center_y + box_height / 2
        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1 or x2 <= x1 or y2 <= y1:
            raise ValueError("YOLO normalized bounding box must remain within the image")
        parsed.append(
            (
                [
                    x1 * image_width,
                    y1 * image_height,
                    x2 * image_width,
                    y2 * image_height,
                ],
                class_id + 1,
            )
        )
    return parsed


@dataclass(frozen=True, slots=True)
class FasterRCNNDatasetSpec:
    """Resolved image/label paths and class order for a training cycle."""

    dataset_yaml: Path
    root: Path
    image_dir: Path
    label_dir: Path
    classes: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls,
        dataset_yaml: str | Path,
        mapping: dict[str, Any],
        *,
        expected_classes: list[str] | None = None,
    ) -> FasterRCNNDatasetSpec:
        yaml_path = Path(dataset_yaml).expanduser().resolve()
        root_value = mapping.get("path", yaml_path.parent)
        root = Path(root_value).expanduser()
        if not root.is_absolute():
            root = yaml_path.parent / root
        root = root.resolve()
        train_value = mapping.get("train", "images")
        if not isinstance(train_value, str):
            raise ValidationError(
                "Faster R-CNN training currently requires one train image directory"
            )
        image_dir = (root / train_value).resolve()
        label_value = mapping.get("labels", "labels")
        if not isinstance(label_value, str):
            raise ValidationError("Dataset labels must be one directory")
        label_dir = (root / label_value).resolve()
        classes = _classes_from_names(mapping.get("names"))
        if expected_classes is not None and classes != list(expected_classes):
            raise ValidationError("Dataset classes do not match the detector classes")
        if not image_dir.is_dir():
            raise ValidationError(f"Dataset image directory does not exist: {image_dir}")
        if not label_dir.is_dir():
            raise ValidationError(f"Dataset label directory does not exist: {label_dir}")
        return cls(yaml_path, root, image_dir, label_dir, tuple(classes))

    @classmethod
    def from_yaml(
        cls,
        dataset_yaml: str | Path,
        *,
        expected_classes: list[str] | None = None,
    ) -> FasterRCNNDatasetSpec:
        yaml_path = Path(dataset_yaml).expanduser().resolve()
        if not yaml_path.is_file():
            raise ValidationError(f"Dataset YAML does not exist: {yaml_path}")
        try:
            import yaml
        except ImportError as exc:
            raise OptionalDependencyError(
                "Faster R-CNN training requires the 'training' extra"
            ) from exc
        try:
            mapping = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValidationError(f"Dataset YAML is invalid: {yaml_path}") from exc
        if not isinstance(mapping, dict):
            raise ValidationError("Dataset YAML must contain a mapping")
        return cls.from_mapping(yaml_path, mapping, expected_classes=expected_classes)

    def image_paths(self) -> list[Path]:
        return sorted(
            path
            for path in self.image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )


class FasterRCNNDataset:
    """Torchvision dataset that consumes :class:`FasterRCNNDatasetSpec`."""

    def __init__(self, spec: FasterRCNNDatasetSpec):
        try:
            import numpy as np
            import torch
            from PIL import Image
        except ImportError as exc:
            raise OptionalDependencyError(
                "Faster R-CNN training requires the 'training' extra"
            ) from exc
        self._np = np
        self._torch = torch
        self._image = Image
        self.spec = spec
        self.paths = spec.image_paths()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            with self._image.open(path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
            array = self._np.array(rgb, copy=True)
        except OSError as exc:
            raise ValidationError(f"Dataset image cannot be read: {path}") from exc
        tensor = self._torch.from_numpy(array).permute(2, 0, 1).float().div(255)
        label_path = self.spec.label_dir / f"{path.stem}.txt"
        lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.is_file() else []
        try:
            parsed = parse_yolo_label_lines(
                lines, width, height, class_count=len(self.spec.classes)
            )
        except ValueError as exc:
            raise ValidationError(f"Invalid labels for dataset image {path}: {exc}") from exc
        boxes = self._torch.tensor([box for box, _ in parsed], dtype=self._torch.float32)
        if not parsed:
            boxes = self._torch.zeros((0, 4), dtype=self._torch.float32)
        labels = self._torch.tensor([label for _, label in parsed], dtype=self._torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": self._torch.tensor([index]),
            "area": area,
            "iscrowd": self._torch.zeros((len(parsed),), dtype=self._torch.int64),
        }
        return tensor, target


def detection_collate(batch):
    """Collate variable-size images and detection targets for torchvision."""
    return tuple(zip(*batch))


def _save_torch_checkpoint(torch, path: Path, value: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(value, temporary)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _selected_checkpoint_name(config: dict[str, Any]) -> str:
    return "last.pt" if isinstance(config.get("cloud"), dict) else "best.pt"


class FasterRCNNTrainer:
    """Train a torchvision Faster R-CNN model from a CVAT/YOLO bundle."""

    def __init__(self, classes: list[str]):
        if not classes or len(classes) != len(set(classes)):
            raise ValidationError("Faster R-CNN classes must be non-empty and unique")
        self.classes = list(classes)

    def train(
        self,
        dataset_yaml: str | Path,
        output_dir: str | Path,
        config: dict[str, Any],
        initial_checkpoint: str | Path | None = None,
        resume_from: str | Path | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        try:
            import torch
            from torch.utils.data import DataLoader
            from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        except ImportError as exc:
            raise OptionalDependencyError(
                "Faster R-CNN training requires the 'training' extra"
            ) from exc
        spec = FasterRCNNDatasetSpec.from_yaml(dataset_yaml, expected_classes=self.classes)
        dataset = FasterRCNNDataset(spec)
        if not len(dataset):
            raise ValidationError("Faster R-CNN training dataset contains no images")
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        device = _select_torch_device(torch, str(config.get("device", "auto")))
        torch.manual_seed(int(config.get("seed", 42)))
        weights = None
        if config.get("use_official_weights"):
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.classes) + 1)
        model.to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(config.get("learning_rate", 0.005)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 0.0005)),
        )
        start_epoch = 0
        if initial_checkpoint and not resume_from:
            initial_state = torch.load(
                Path(initial_checkpoint), map_location="cpu", weights_only=False
            )
            checkpoint_metadata = initial_state if isinstance(initial_state, dict) else {}
            source_classes = checkpoint_metadata.get("classes")
            effective = (
                config.get("metadata", {}).get("effective_configuration", {})
                if isinstance(config.get("metadata"), dict)
                else {}
            )
            hosted_model = effective.get("hosted_model", {})
            if source_classes is None and isinstance(hosted_model, dict):
                source_classes = hosted_model.get("source_class_order")
            source_state = checkpoint_metadata.get("model_state_dict", checkpoint_metadata)
            compatible = adapt_faster_rcnn_state_dict(
                source_state,
                model.state_dict(),
                source_classes=source_classes,
                target_classes=self.classes,
            )
            result = model.load_state_dict(compatible, strict=False)
            unexpected_missing = [
                key for key in result.missing_keys if not key.startswith(_CLASSIFIER_HEAD_PREFIXES)
            ]
            if result.unexpected_keys or unexpected_missing:
                incompatible = result.unexpected_keys + unexpected_missing
                raise ValueError(
                    "Faster R-CNN checkpoint is incompatible outside its output head: "
                    + ", ".join(incompatible[:5])
                )
        if resume_from:
            state = torch.load(Path(resume_from), map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
                if "optimizer_state_dict" in state:
                    optimizer.load_state_dict(state["optimizer_state_dict"])
                start_epoch = int(state.get("epoch", -1)) + 1
            else:
                model.load_state_dict(state)
        loader = DataLoader(
            dataset,
            batch_size=int(config.get("batch_size", 2)),
            shuffle=True,
            num_workers=int(config.get("num_workers", 0)),
            collate_fn=detection_collate,
        )
        epochs = int(config.get("epochs", 1))
        if epochs <= 0:
            raise ValidationError("epochs must be positive")
        best_loss = float("inf")
        metrics: list[dict[str, float | int]] = []
        for epoch in range(start_epoch, start_epoch + epochs):
            model.train()
            total_loss = 0.0
            for images, targets in loader:
                image_batch = [image.to(device) for image in images]
                target_batch = [
                    {key: value.to(device) for key, value in target.items()} for target in targets
                ]
                losses = model(image_batch, target_batch)
                loss = sum(loss_value for loss_value in losses.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
            average_loss = total_loss / len(loader)
            metrics.append({"epoch": epoch, "train_loss": average_loss})
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "classes": self.classes,
                "architecture": "faster_rcnn",
            }
            _save_torch_checkpoint(torch, output / "last.pt", state)
            if average_loss <= best_loss:
                best_loss = average_loss
                _save_torch_checkpoint(torch, output / "best.pt", state)
            if progress_callback is not None:
                completed = epoch - start_epoch + 1
                progress_callback(
                    {
                        "epoch": completed,
                        "epochs": epochs,
                        "progress": min(1.0, completed / epochs),
                        "train_loss": average_loss,
                    }
                )
        atomic_write_json(
            output / "metrics.json",
            {"evaluation": "not evaluated", "epochs": metrics, "classes": self.classes},
        )
        return output / _selected_checkpoint_name(config)
