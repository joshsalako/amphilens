"""Lazy-loading detector adapters for supported model families."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from ..core import (
    CheckpointManifest,
    DetectionRecord,
    InferenceConfig,
    UnsupportedCheckpointError,
    ValidationError,
)
from ..preprocessing import PreprocessingService
from .catalog import ModelPreset


class OptionalDependencyError(RuntimeError):
    """Raised when a selected backend's optional ML dependency is absent."""


def _select_torch_device(torch, requested: str):
    available = bool(torch.cuda.is_available())
    if requested == "auto":
        return torch.device("cuda" if available else "cpu")
    if requested == "cuda" or requested.startswith("cuda:"):
        if not available:
            raise OptionalDependencyError("CUDA was requested but no CUDA device is available")
        return torch.device(requested)
    if requested == "cpu":
        return torch.device("cpu")
    raise ValidationError(f"Unsupported device: {requested}")


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("Image inference requires the 'inference' extra") from exc
    with Image.open(path) as image:
        return image.width, image.height


class UltralyticsDetector:
    """YOLO or RT-DETR adapter using Ultralytics' lazy runtime."""

    def __init__(
        self,
        checkpoint: str | Path,
        architecture: str,
        classes: list[str],
        model_id: str | None = None,
    ):
        if architecture not in {"yolo", "rtdetr"}:
            raise ValidationError("UltralyticsDetector architecture must be 'yolo' or 'rtdetr'")
        checkpoint_path = Path(checkpoint).expanduser()
        self.checkpoint = (
            checkpoint_path.resolve()
            if checkpoint_path.is_absolute() or checkpoint_path.is_file()
            else checkpoint_path
        )
        self.architecture = architecture
        self.classes = list(classes)
        self.model_id = model_id or self.checkpoint.stem
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import RTDETR, YOLO
        except ImportError as exc:
            raise OptionalDependencyError(
                "YOLO/RT-DETR inference requires the 'inference' extra"
            ) from exc
        self._model = (RTDETR if self.architecture == "rtdetr" else YOLO)(str(self.checkpoint))
        return self._model

    def predict(self, image_paths: Iterable[Path], config: InferenceConfig):
        model = self._load()
        for path in image_paths:
            width, height = _image_size(path)
            transformed = PreprocessingService(config.preprocessing_config).transform(path)
            results = model.predict(
                source=transformed.image,
                imgsz=config.image_size,
                conf=config.confidence,
                device=None if config.device == "auto" else config.device,
                verbose=False,
            )
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", getattr(model, "names", {}))
            for box, confidence, class_id in zip(
                boxes.xyxy.cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.cls.cpu().tolist(),
            ):
                class_index = int(class_id)
                class_name = (
                    names[class_index]
                    if isinstance(names, (list, tuple))
                    else names.get(
                        class_index,
                        self.classes[class_index]
                        if class_index < len(self.classes)
                        else f"class_{class_index}",
                    )
                )
                yield DetectionRecord(
                    image_path=str(path),
                    image_id=path.name,
                    class_id=class_index,
                    class_name=str(class_name),
                    confidence=float(confidence),
                    bbox_xyxy=transformed.map_box_to_original(box),
                    image_width=width,
                    image_height=height,
                    model_id=self.model_id,
                    run_id=config.run_id,
                    preprocessing=config.preprocessing_fingerprint,
                )

    def train(
        self,
        dataset_yaml: str | Path,
        output_dir: str | Path,
        config: dict,
        resume_from: CheckpointManifest | None = None,
    ) -> Path:
        model = self._load()
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        if resume_from:
            resume_from.validate_compatibility(
                architecture=self.architecture,
                classes=self.classes,
                preprocessing=config.get("preprocessing", {}),
            )
        results = model.train(
            data=str(dataset_yaml),
            project=str(output),
            name=str(config.get("run_name", "train")),
            epochs=int(config.get("epochs", 100)),
            imgsz=int(config.get("image_size", 640)),
            batch=int(config.get("batch_size", 16)),
            device=config.get("device", "auto"),
            patience=int(config.get("patience", 25)),
            seed=int(config.get("seed", 42)),
            exist_ok=True,
        )
        save_dir = Path(getattr(results, "save_dir", output / str(config.get("run_name", "train"))))
        checkpoint = save_dir / "weights" / "best.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"Training completed without a best checkpoint at {checkpoint}")
        return checkpoint


class FasterRCNNDetector:
    """Torchvision Faster R-CNN adapter with no import-time torch dependency."""

    architecture = "faster_rcnn"

    def __init__(self, checkpoint: str | Path, classes: list[str], model_id: str | None = None):
        checkpoint_path = Path(checkpoint).expanduser()
        self.checkpoint = (
            checkpoint_path.resolve()
            if checkpoint_path.is_absolute() or checkpoint_path.is_file()
            else None
        )
        self.checkpoint_reference = str(checkpoint)
        self.classes = list(classes)
        self.model_id = model_id or self.checkpoint.stem
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        except ImportError as exc:
            raise OptionalDependencyError(
                "Faster R-CNN inference requires the 'inference' extra"
            ) from exc
        if self.checkpoint is None and self.checkpoint_reference.startswith("torchvision://"):
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

            model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        else:
            model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.classes) + 1)
        if self.checkpoint is not None:
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
        model.eval()
        self._model = model
        return model

    def predict(self, image_paths: Iterable[Path], config: InferenceConfig):
        try:
            import numpy as np
            import torch
            from PIL import Image
        except ImportError as exc:
            raise OptionalDependencyError(
                "Faster R-CNN inference requires the 'inference' extra"
            ) from exc
        model = self._load()
        device = _select_torch_device(torch, config.device)
        model.to(device)
        for path in image_paths:
            with Image.open(path) as image:
                width, height = image.size
            transformed = PreprocessingService(config.preprocessing_config).transform(path)
            rgb = np.asarray(transformed.image)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)
            with torch.no_grad():
                output = model([tensor])[0]
            for box, confidence, label in zip(output["boxes"], output["scores"], output["labels"]):
                score = float(confidence.cpu())
                if score < config.confidence:
                    continue
                class_id = int(label.cpu()) - 1
                yield DetectionRecord(
                    image_path=str(path),
                    image_id=path.name,
                    class_id=class_id,
                    class_name=(
                        self.classes[class_id]
                        if 0 <= class_id < len(self.classes)
                        else f"class_{class_id}"
                    ),
                    confidence=score,
                    bbox_xyxy=transformed.map_box_to_original(box.cpu().tolist()),
                    image_width=width,
                    image_height=height,
                    model_id=self.model_id,
                    run_id=config.run_id,
                    preprocessing=config.preprocessing_fingerprint,
                )

    def train(self, dataset_yaml, output_dir, config, resume_from=None) -> Path:
        from .faster_rcnn_training import FasterRCNNTrainer

        if self.checkpoint is not None and not self.checkpoint.is_file():
            raise UnsupportedCheckpointError(f"Checkpoint is missing: {self.checkpoint}")
        if resume_from is not None:
            resume_from.validate_compatibility(
                architecture=self.architecture,
                classes=self.classes,
                preprocessing=config.get("preprocessing", {}),
            )
            resume_checkpoint = resume_from.checkpoint_path
        else:
            resume_checkpoint = None
        return FasterRCNNTrainer(self.classes).train(
            dataset_yaml,
            output_dir,
            config,
            initial_checkpoint=self.checkpoint,
            resume_from=resume_checkpoint,
        )


def load_detector(
    checkpoint: str | Path,
    *,
    architecture: str,
    classes: list[str],
    model_id: str | None = None,
    checkpoint_manifest: CheckpointManifest | None = None,
    preprocessing: dict | None = None,
):
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise UnsupportedCheckpointError(f"Checkpoint is missing: {checkpoint_path}")
    if not classes or len(classes) != len(set(classes)):
        raise ValidationError("Detector classes must be non-empty and unique")
    if checkpoint_manifest is not None:
        checkpoint_manifest.validate_compatibility(
            architecture=architecture,
            classes=classes,
            preprocessing=preprocessing or {},
        )
        if model_id is not None and checkpoint_manifest.model_id != model_id:
            raise UnsupportedCheckpointError(
                "Checkpoint model id does not match the requested model"
            )
        current_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if current_hash != checkpoint_manifest.sha256:
            raise UnsupportedCheckpointError("Checkpoint hash does not match its manifest")
    if architecture in {"yolo", "rtdetr"}:
        return UltralyticsDetector(checkpoint_path, architecture, classes, model_id)
    if architecture == "faster_rcnn":
        return FasterRCNNDetector(checkpoint_path, classes, model_id)
    raise ValidationError(f"Unsupported detector architecture: {architecture}")


def load_preset_detector(
    preset: ModelPreset, *, classes: list[str], checkpoint: str | Path | None = None
):
    """Create a detector from a catalog preset without downloading weights at import time."""
    selected = checkpoint or preset.checkpoint_reference
    if checkpoint is not None:
        return load_detector(
            checkpoint,
            architecture=preset.architecture,
            classes=classes,
            model_id=preset.model_id,
        )
    if preset.architecture in {"yolo", "rtdetr"}:
        return UltralyticsDetector(selected, preset.architecture, classes, model_id=preset.model_id)
    if preset.architecture == "faster_rcnn":
        return FasterRCNNDetector(selected, classes, model_id=preset.model_id)
    raise ValidationError(f"Unsupported detector architecture: {preset.architecture}")
