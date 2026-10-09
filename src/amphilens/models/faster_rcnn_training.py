"""Optional Faster R-CNN training over a portable YOLO directory layout."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..core import ValidationError, atomic_write_json
from .backends import (
    OptionalDependencyError,
    _configure_faster_rcnn_transform,
    _select_torch_device,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_CLASSIFIER_HEAD_PREFIXES = (
    "roi_heads.box_predictor.cls_score.",
    "roi_heads.box_predictor.bbox_pred.",
)
_FASTER_RCNN_AUGMENTATION = {
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.5,
    "color_jitter_probability": 0.8,
    "brightness": 0.4,
    "contrast": 0.4,
    "saturation": 0.4,
    "affine_probability": 0.5,
    "affine_scale": [0.5, 1.5],
    "affine_translation_fraction": 0.1,
    "affine_rotation_degrees": 10.0,
    "minimum_box_visibility": 0.2,
    "fill": 114,
}


def _build_faster_rcnn_augmentation():
    from torchvision.transforms import v2

    color_jitter = v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0)
    affine = v2.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.5, 1.5),
        fill=(114, 114, 114),
    )
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([color_jitter], p=0.8),
            v2.RandomApply([affine], p=0.5),
        ]
    )


def _apply_faster_rcnn_augmentation(image, target, *, transform=None):
    """Apply WTL-style joint image and box augmentation to a training sample."""
    import torch
    from torchvision import tv_tensors

    height, width = image.shape[-2:]
    selected_transform = transform or _build_faster_rcnn_augmentation()
    boxes = tv_tensors.BoundingBoxes(target["boxes"], format="XYXY", canvas_size=(height, width))
    image, transformed = selected_transform(
        tv_tensors.Image(image), {"boxes": boxes, "labels": target["labels"]}
    )
    raw_boxes = transformed["boxes"].as_subclass(torch.Tensor).to(dtype=torch.float32)
    if raw_boxes.numel():
        clipped = raw_boxes.clone()
        clipped[:, 0::2].clamp_(0, width)
        clipped[:, 1::2].clamp_(0, height)
        raw_area = (raw_boxes[:, 2] - raw_boxes[:, 0]).clamp(min=0) * (
            raw_boxes[:, 3] - raw_boxes[:, 1]
        ).clamp(min=0)
        visible_area = (clipped[:, 2] - clipped[:, 0]).clamp(min=0) * (
            clipped[:, 3] - clipped[:, 1]
        ).clamp(min=0)
        keep = (
            (clipped[:, 2] > clipped[:, 0])
            & (clipped[:, 3] > clipped[:, 1])
            & (visible_area >= raw_area * _FASTER_RCNN_AUGMENTATION["minimum_box_visibility"])
        )
        clipped = clipped[keep]
        labels = transformed["labels"][keep]
    else:
        clipped = raw_boxes.reshape((0, 4))
        labels = transformed["labels"]

    output_target = dict(target)
    output_target["boxes"] = clipped
    output_target["labels"] = labels
    output_target["area"] = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    output_target["iscrowd"] = (
        target.get("iscrowd", torch.zeros(len(target["labels"]), dtype=torch.int64))[keep]
        if raw_boxes.numel()
        else target.get("iscrowd", torch.zeros(0, dtype=torch.int64))
    )
    return image.as_subclass(torch.Tensor), output_target


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
    short_side_dimension: int = 640
    training_image_size: int | None = None
    validation_image_dir: Path | None = None
    validation_label_dir: Path | None = None
    test_image_dir: Path | None = None
    test_label_dir: Path | None = None

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
        label_root = (root / label_value).resolve()
        label_dir = label_root / "train" if (label_root / "train").is_dir() else label_root
        validation_value = mapping.get("val")
        test_value = mapping.get("test")
        if not isinstance(validation_value, str):
            raise ValidationError("Faster R-CNN training requires a validation directory")
        validation_image_dir = (root / validation_value).resolve()
        validation_label_dir = (
            label_root / "validation" if (label_root / "validation").is_dir() else label_root
        )
        test_image_dir = (root / test_value).resolve() if isinstance(test_value, str) else None
        test_label_dir = (
            label_root / "test"
            if test_image_dir is not None and (label_root / "test").is_dir()
            else label_root if test_image_dir is not None else None
        )
        classes = _classes_from_names(mapping.get("names"))
        short_side_dimension = mapping.get("short_side_dimension", 640)
        training_image_size = mapping.get("training_image_size")
        for name, value in (
            ("short_side_dimension", short_side_dimension),
            ("training_image_size", training_image_size),
        ):
            if value is None and name == "training_image_size":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"Dataset {name} must be a positive integer")
        if expected_classes is not None and classes != list(expected_classes):
            raise ValidationError("Dataset classes do not match the detector classes")
        if not image_dir.is_dir():
            raise ValidationError(f"Dataset image directory does not exist: {image_dir}")
        if not label_dir.is_dir():
            raise ValidationError(f"Dataset label directory does not exist: {label_dir}")
        if not validation_image_dir.is_dir() or not validation_label_dir.is_dir():
            raise ValidationError("Faster R-CNN validation images and labels are required")
        if test_image_dir is not None and not test_image_dir.is_dir():
            raise ValidationError(
                f"Faster R-CNN test image directory does not exist: {test_image_dir}"
            )
        return cls(
            yaml_path,
            root,
            image_dir,
            label_dir,
            tuple(classes),
            short_side_dimension,
            training_image_size,
            validation_image_dir,
            validation_label_dir,
            test_image_dir,
            test_label_dir,
        )

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

    def __init__(self, spec: FasterRCNNDatasetSpec, *, augment: bool = False, transform=None):
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
        self.augment = augment
        self.transform = transform or (_build_faster_rcnn_augmentation() if augment else None)

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
        tensor = self._torch.from_numpy(array).permute(2, 0, 1)
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
        if self.augment:
            tensor, target = _apply_faster_rcnn_augmentation(
                tensor, target, transform=self.transform
            )
        tensor = tensor.float().div(255)
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
        config["training_image_size"] = spec.training_image_size or spec.short_side_dimension
        augment = bool(config.get("augment", True))
        dataset = FasterRCNNDataset(spec, augment=augment)
        validation_spec = replace(
            spec,
            image_dir=spec.validation_image_dir or spec.image_dir,
            label_dir=spec.validation_label_dir or spec.label_dir,
        )
        validation_dataset = FasterRCNNDataset(validation_spec, augment=False)
        test_dataset = None
        if spec.test_image_dir is not None and spec.test_label_dir is not None:
            test_dataset = FasterRCNNDataset(
                replace(spec, image_dir=spec.test_image_dir, label_dir=spec.test_label_dir),
                augment=False,
            )
        config["augmentation"] = {
            "backend": "torchvision-v2",
            "enabled": augment,
            "parameters": dict(_FASTER_RCNN_AUGMENTATION) if augment else {},
        }
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
        _configure_faster_rcnn_transform(
            model,
            spec.training_image_size or spec.short_side_dimension,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.classes) + 1)
        model.to(device)
        optimizer = None
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
            else:
                model.load_state_dict(state)
        batch_size = int(config.get("batch_size", 16))
        if batch_size <= 0:
            raise ValidationError("batch_size must be positive")
        loader_options = {
            "batch_size": batch_size,
            "num_workers": int(config.get("num_workers", 0)),
            "collate_fn": detection_collate,
        }
        train_loader = DataLoader(dataset, shuffle=True, **loader_options)
        validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
        test_loader = (
            DataLoader(test_dataset, shuffle=False, **loader_options) if test_dataset else None
        )
        epochs_per_phase = int(config.get("epochs", 100))
        if epochs_per_phase <= 0:
            raise ValidationError("epochs must be positive")
        patience = int(config.get("patience", 25))
        phase_specs = (
            [("backbone-frozen", True, float(config.get("phase1_learning_rate", 0.0001))),
             ("full-fine-tuning", False, float(config.get("phase2_learning_rate", 0.00005)))]
            if config.get("freeze_strategy", "paper-phased") == "paper-phased"
            else [("training", False, float(config.get("learning_rate", 0.0001)))]
        )
        phase_history: list[dict[str, Any]] = []

        def run_loss(loader, *, training: bool) -> float:
            model.train()
            if not training:
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
            total = 0.0
            batches = 0
            context = torch.enable_grad() if training else torch.no_grad()
            with context:
                for images, targets in loader:
                    image_batch = [image.to(device) for image in images]
                    target_batch = [
                        {key: value.to(device) for key, value in target.items()}
                        for target in targets
                    ]
                    losses = model(image_batch, target_batch)
                    loss = sum(loss_value for loss_value in losses.values())
                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    total += float(loss.detach().cpu())
                    batches += 1
            return total / max(1, batches)

        for phase_index, (phase_name, freeze_backbone, learning_rate) in enumerate(phase_specs, 1):
            if phase_index > 1:
                previous_best = output / f"phase-{phase_index - 1}" / "best.pt"
                phase_state = torch.load(previous_best, map_location="cpu", weights_only=False)
                model.load_state_dict(phase_state["model_state_dict"])
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            if freeze_backbone:
                for parameter in model.backbone.parameters():
                    parameter.requires_grad_(False)
            optimizer = torch.optim.SGD(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=learning_rate,
                momentum=float(config.get("momentum", 0.9)),
                weight_decay=float(config.get("weight_decay", 0.0005)),
            )
            phase_dir = output / f"phase-{phase_index}"
            phase_dir.mkdir(parents=True, exist_ok=True)
            best_validation_loss = float("inf")
            epochs_without_improvement = 0
            epochs_run = 0
            epoch_metrics: list[dict[str, float | int]] = []
            for epoch in range(1, epochs_per_phase + 1):
                train_loss = run_loss(train_loader, training=True)
                validation_loss = run_loss(validation_loader, training=False)
                if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
                    raise ValidationError(
                        f"{phase_name} produced a non-finite training or validation loss"
                    )
                epoch_metrics.append(
                    {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
                )
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "classes": self.classes,
                    "architecture": "faster_rcnn",
                    "phase": phase_name,
                    "validation_loss": validation_loss,
                }
                _save_torch_checkpoint(torch, phase_dir / "last.pt", state)
                _save_torch_checkpoint(torch, output / "last.pt", state)
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    epochs_without_improvement = 0
                    _save_torch_checkpoint(torch, phase_dir / "best.pt", state)
                    _save_torch_checkpoint(torch, output / "best.pt", state)
                else:
                    epochs_without_improvement += 1
                epochs_run = epoch
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": phase_name,
                            "training_phase": phase_name,
                            "phase_index": phase_index,
                            "phase_count": len(phase_specs),
                            "message": (
                                f"{phase_name.replace('-', ' ').title()} · epoch "
                                f"{epoch} of {epochs_per_phase}"
                            ),
                            "epoch": epoch,
                            "epochs": epochs_per_phase,
                            "phase_progress": epoch / epochs_per_phase,
                            "progress": ((phase_index - 1) + epoch / epochs_per_phase)
                            / len(phase_specs),
                            "metrics": {
                                "train_loss": train_loss,
                                "validation_loss": validation_loss,
                            },
                            "device": str(device),
                        }
                    )
                if patience > 0 and epochs_without_improvement >= patience:
                    break
            phase_history.append(
                {
                    "phase": phase_name,
                    "backbone_frozen": freeze_backbone,
                    "learning_rate": learning_rate,
                    "max_epochs": epochs_per_phase,
                    "epochs_run": epochs_run,
                    "patience": patience,
                    "best_validation_loss": best_validation_loss,
                    "epochs": epoch_metrics,
                }
            )
        test_loss = None
        if test_loader is not None:
            best_state = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(best_state["model_state_dict"])
            test_loss = run_loss(test_loader, training=False)
        config["training_phases"] = [
            {key: value for key, value in phase.items() if key != "epochs"}
            for phase in phase_history
        ]
        config["evaluation"] = "test set" if test_loss is not None else "not evaluated"
        metrics_payload = {
            "evaluation": config["evaluation"],
            "validation_strategy": "validation loss with early stopping",
            "training_phases": phase_history,
            "test_loss": test_loss,
            "classes": self.classes,
            "augmentation": config["augmentation"],
        }
        atomic_write_json(
            output / "metrics.json",
            metrics_payload,
        )
        return output / "best.pt"
