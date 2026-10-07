"""Lazy-loading detector adapters for supported model families."""

from __future__ import annotations

import hashlib
import math
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


def _training_only_trainer(base_trainer):
    """Keep Ultralytics' mandatory final-epoch hooks from evaluating cloud runs."""

    class TrainingOnlyTrainer(base_trainer):
        def validate(self):
            self.metrics = {}
            self.fitness = 0.0
            if self.best_fitness is None or self.best_fitness < self.fitness:
                self.best_fitness = self.fitness
            return self.metrics, self.fitness

        def final_eval(self):
            if not self.last.exists() and not self.best.exists():
                return None
            from ultralytics.utils.torch_utils import strip_optimizer

            last_checkpoint = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best,
                    updates={"train_results": last_checkpoint.get("train_results")},
                )

    TrainingOnlyTrainer.__name__ = "AmphiLensTrainingOnlyTrainer"
    return TrainingOnlyTrainer


def _trainer_metrics(trainer) -> dict[str, float]:
    """Extract finite scalar metrics from an Ultralytics epoch callback."""
    metrics: dict[str, float] = {}
    source_metrics = getattr(trainer, "metrics", {})
    if isinstance(source_metrics, dict):
        for key, value in source_metrics.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                metrics[str(key)] = numeric

    losses = getattr(trainer, "tloss", None)
    if losses is not None:
        if hasattr(losses, "detach"):
            losses = losses.detach().cpu()
        if hasattr(losses, "tolist"):
            losses = losses.tolist()
        if not isinstance(losses, (list, tuple)):
            losses = [losses]
        names = list(getattr(trainer, "loss_names", []))
        for index, value in enumerate(losses):
            if index >= len(names):
                break
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                metrics[f"train/{names[index]}"] = numeric
    return metrics


def _select_torch_device(torch, requested: str):
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    if requested == "auto":
        return torch.device("cuda" if cuda_available else "mps" if mps_available else "cpu")
    if requested == "cuda" or requested.startswith("cuda:"):
        if not cuda_available:
            raise OptionalDependencyError("CUDA was requested but no CUDA device is available")
        return torch.device(requested)
    if requested == "mps":
        if not mps_available:
            raise OptionalDependencyError("Apple MPS was requested but no MPS device is available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValidationError(f"Unsupported device: {requested}")


def resolve_device_name(requested: str) -> str:
    """Resolve a requested local device while keeping torch optional at import time."""
    try:
        import torch
    except ImportError as exc:
        if requested == "auto":
            return "cpu"
        raise OptionalDependencyError(
            f"{requested.upper()} was requested but PyTorch is not installed"
        ) from exc
    return str(_select_torch_device(torch, requested))


def describe_device_name(requested: str) -> str:
    """Return a concise device label suitable for saved run and UI summaries."""
    if requested.startswith("cuda"):
        try:
            import torch

            return f"CUDA · {torch.cuda.get_device_name(0)}"
        except Exception:
            return "CUDA GPU"
    if requested == "mps":
        return "Apple MPS GPU"
    return "CPU"


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("Image inference requires the 'inference' extra") from exc
    with Image.open(path) as image:
        return image.width, image.height


def reset_ultralytics_classification_head(
    detector_model,
    *,
    architecture: str,
    source_classes: list[str] | tuple[str, ...] | None,
    target_classes: list[str] | tuple[str, ...],
) -> bool:
    """Reinitialize class outputs when checkpoint labels or class order changes."""
    if source_classes is None or list(source_classes) == list(target_classes):
        return False
    if architecture not in {"yolo", "rtdetr"}:
        raise ValidationError(f"Unsupported Ultralytics head architecture: {architecture}")
    try:
        import torch
    except ImportError as exc:
        raise OptionalDependencyError("Fine-tuning requires the 'training' extra") from exc

    core = getattr(detector_model, "model", None)
    modules = getattr(core, "model", None)
    if modules is None or len(modules) == 0:
        raise RuntimeError("Cannot find the Ultralytics detector output head")
    head = modules[-1]
    reset_count = 0
    if architecture == "yolo":
        for name in ("cv3", "one2one_cv3"):
            branches = getattr(head, name, None)
            if branches is None:
                continue
            for branch in branches:
                convolutions = [
                    module for module in branch.modules() if isinstance(module, torch.nn.Conv2d)
                ]
                if convolutions:
                    convolutions[-1].reset_parameters()
                    reset_count += 1
    else:
        for name in ("enc_score_head", "dec_score_head", "denoising_class_embed"):
            classifier = getattr(head, name, None)
            if classifier is None:
                continue
            for module in classifier.modules():
                if isinstance(
                    module,
                    (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding),
                ):
                    module.reset_parameters()
                    reset_count += 1
    if not reset_count:
        raise RuntimeError(
            f"Cannot identify the {architecture} class output head for label adaptation"
        )
    return True


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
            checkpoint_path.absolute()
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
        paths = list(image_paths)
        batch_size = max(1, config.batch_size)
        for offset in range(0, len(paths), batch_size):
            batch_paths = paths[offset : offset + batch_size]
            dimensions = [_image_size(path) for path in batch_paths]
            transformed_images = [
                PreprocessingService(config.preprocessing_config).transform(path)
                for path in batch_paths
            ]
            results = model.predict(
                source=[item.image for item in transformed_images],
                imgsz=config.image_size,
                conf=config.confidence,
                device=None if config.device == "auto" else config.device,
                batch=len(batch_paths),
                verbose=False,
            )
            if len(results) != len(batch_paths):
                raise RuntimeError("Model returned a different number of results than input images")
            for path, (width, height), transformed, result in zip(
                batch_paths, dimensions, transformed_images, results
            ):
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
        progress_callback=None,
    ) -> Path:
        model = self._load()
        metadata = config.get("metadata", {})
        effective_configuration = (
            metadata.get("effective_configuration", {}) if isinstance(metadata, dict) else {}
        )
        hosted_model = effective_configuration.get("hosted_model", {})
        source_classes = (
            hosted_model.get("source_class_order") if isinstance(hosted_model, dict) else None
        )
        if source_classes is None:
            names = getattr(model, "names", None)
            if isinstance(names, dict):
                source_classes = [str(names[key]) for key in sorted(names)]
            elif isinstance(names, (list, tuple)):
                source_classes = [str(name) for name in names]
        reset_ultralytics_classification_head(
            model,
            architecture=self.architecture,
            source_classes=source_classes,
            target_classes=self.classes,
        )
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        if resume_from:
            resume_from.validate_compatibility(
                architecture=self.architecture,
                classes=self.classes,
                preprocessing=config.get("preprocessing", {}),
            )
        cloud_provenance = config.get("cloud")
        is_cloud_training = isinstance(cloud_provenance, dict)
        train_config = {
            "data": str(dataset_yaml),
            "project": str(output),
            "name": str(config.get("run_name", "train")),
            "epochs": int(config.get("epochs", 100)),
            "imgsz": int(config.get("image_size", 640)),
            "batch": int(config.get("batch_size", 16)),
            "device": config.get("device", "auto"),
            "val": bool(config.get("val", not is_cloud_training)),
            "patience": int(config.get("patience", 25)),
            "seed": int(config.get("seed", 42)),
            "exist_ok": True,
        }
        if is_cloud_training:
            train_config["trainer"] = _training_only_trainer(model._smart_load("trainer"))
        if resume_from is not None:
            train_config["resume"] = str(resume_from.checkpoint_path)
        if progress_callback is not None:

            def report_epoch(trainer):
                total = max(1, int(getattr(trainer, "epochs", train_config["epochs"])))
                epoch = max(0, int(getattr(trainer, "epoch", -1)) + 1)
                progress_callback(
                    {
                        "phase": "training",
                        "message": f"Epoch {min(epoch, total)} of {total}",
                        "epoch": min(epoch, total),
                        "epochs": total,
                        "progress": min(1.0, max(0.0, epoch / total)),
                        "metrics": _trainer_metrics(trainer),
                        "device": str(getattr(trainer, "device", "")),
                    }
                )

            model.add_callback("on_fit_epoch_end", report_epoch)
        results = model.train(
            **train_config,
        )
        save_dir = Path(getattr(results, "save_dir", output / str(config.get("run_name", "train"))))
        checkpoint = save_dir / "weights" / "best.pt"
        if is_cloud_training:
            checkpoint = save_dir / "weights" / "last.pt"
            cloud_provenance["checkpoint_selection"] = "last-no-validation"
        if not checkpoint.is_file():
            raise RuntimeError(
                f"Training completed without the selected checkpoint at {checkpoint}"
            )
        return checkpoint


class FasterRCNNDetector:
    """Torchvision Faster R-CNN adapter with no import-time torch dependency."""

    architecture = "faster_rcnn"

    def __init__(self, checkpoint: str | Path, classes: list[str], model_id: str | None = None):
        checkpoint_path = Path(checkpoint).expanduser()
        self.checkpoint = (
            checkpoint_path.absolute()
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
        paths = list(image_paths)
        batch_size = 1 if str(device) == "cpu" else max(1, config.batch_size)
        for offset in range(0, len(paths), batch_size):
            batch_paths = paths[offset : offset + batch_size]
            dimensions = []
            transformed_images = []
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    dimensions.append(image.size)
                transformed = PreprocessingService(config.preprocessing_config).transform(path)
                transformed_images.append(transformed)
                rgb = np.array(transformed.image, copy=True)
                tensors.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device))
            with torch.no_grad():
                outputs = model(tensors)
            if len(outputs) != len(batch_paths):
                raise RuntimeError("Model returned a different number of results than input images")
            for path, (width, height), transformed, output in zip(
                batch_paths, dimensions, transformed_images, outputs
            ):
                for box, confidence, label in zip(
                    output["boxes"], output["scores"], output["labels"]
                ):
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

    def train(
        self, dataset_yaml, output_dir, config, resume_from=None, progress_callback=None
    ) -> Path:
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
        if self.checkpoint is None and self.checkpoint_reference.startswith("torchvision://"):
            config = dict(config)
            config["use_official_weights"] = True
        return FasterRCNNTrainer(self.classes).train(
            dataset_yaml,
            output_dir,
            config,
            initial_checkpoint=self.checkpoint,
            resume_from=resume_checkpoint,
            progress_callback=progress_callback,
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
    checkpoint_path = Path(checkpoint).expanduser().absolute()
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
