"""Resolve reproducible project and checkpoint runtime configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import (
    CheckpointManifest,
    ProjectManifest,
    UnsupportedCheckpointError,
    ValidationError,
    read_json,
)
from .models import ModelCatalog
from .models.hosted_models import HostedModel
from .preprocessing import PreprocessingConfig


@dataclass(frozen=True, slots=True)
class EffectiveRunConfiguration:
    """The complete configuration used by one training or inference run."""

    model_preset: str
    model_id: str
    architecture: str
    classes: tuple[str, ...]
    preprocessing: PreprocessingConfig
    image_size: int
    confidence: float
    device: str
    epochs: int
    batch_size: int
    patience: int
    seed: int
    freeze_strategy: str
    source: str
    warnings: tuple[str, ...] = ()
    hosted_model: dict[str, Any] | None = None
    base_classes: tuple[str, ...] | None = None

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(include_fingerprint=False), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_preset": self.model_preset,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "classes": list(self.classes),
            "preprocessing": self.preprocessing.to_dict(),
            "image_size": self.image_size,
            "confidence": self.confidence,
            "device": self.device,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "seed": self.seed,
            "freeze_strategy": self.freeze_strategy,
            "source": self.source,
            "warnings": list(self.warnings),
            "hosted_model": self.hosted_model,
            "base_classes": list(self.base_classes) if self.base_classes is not None else None,
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload


def _checkpoint_input_size(manifest: CheckpointManifest) -> int | None:
    value = manifest.training_config.get("image_size")
    if value is None:
        return None
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Checkpoint image_size must be a positive integer") from exc
    if size <= 0:
        raise ValidationError("Checkpoint image_size must be a positive integer")
    return size


def _metadata_preprocessing(value: Any = None) -> PreprocessingConfig:
    """Normalize model metadata, using the web workflow's explicit fallback."""
    fallback = PreprocessingConfig(clahe_enabled=True).to_dict(include_fingerprint=False)
    if isinstance(value, PreprocessingConfig):
        return value
    if isinstance(value, dict):
        if set(value) == {"name"}:
            return PreprocessingConfig.from_any(value)
        fallback.update(value)
    elif value is not None:
        return PreprocessingConfig.from_any(value)
    return PreprocessingConfig.from_any(fallback)


def _preprocessing_override(base: PreprocessingConfig, value: Any) -> PreprocessingConfig:
    """Apply a partial run override while preserving unexposed model settings."""
    if isinstance(value, dict):
        if set(value) == {"name"}:
            return PreprocessingConfig.from_any(value)
        merged = base.to_dict(include_fingerprint=False)
        merged.update(value)
        return PreprocessingConfig.from_any(merged)
    return PreprocessingConfig.from_any(value)


def resolve_effective_configuration(
    project: ProjectManifest,
    *,
    checkpoint_manifest: CheckpointManifest | None = None,
    checkpoint_path: str | Path | None = None,
    hosted_model: HostedModel | None = None,
    use_source_classes: bool = False,
    target_classes: tuple[str, ...] | list[str] | None = None,
    use_model_metadata_defaults: bool = False,
    allow_checkpoint_preprocessing_override: bool = False,
    overrides: dict[str, Any] | None = None,
) -> EffectiveRunConfiguration:
    """Resolve project defaults, checkpoint authority, and explicit run overrides."""
    project_config = project.project_config
    values = dict(overrides or {})
    warnings: list[str] = []
    hosted_metadata = hosted_model.to_summary() if hosted_model else None
    base_classes: tuple[str, ...] | None = None

    def selected_target_classes(default: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        values = default if target_classes is None else target_classes
        selected = tuple(str(value).strip() for value in values)
        if not selected or any(not value for value in selected):
            raise ValidationError("At least one non-empty target class is required")
        if len(selected) != len(set(selected)):
            raise ValidationError("Target classes must be unique")
        return selected

    if checkpoint_manifest is not None and hosted_model is not None:
        raise UnsupportedCheckpointError("Choose either a project checkpoint or an AmphiLens model")
    if checkpoint_manifest is not None:
        checkpoint_manifest.validate()
        model_id = checkpoint_manifest.model_id
        architecture = checkpoint_manifest.architecture
        model_preset = model_id
        base_classes = tuple(checkpoint_manifest.classes)
        classes = selected_target_classes(tuple(checkpoint_manifest.classes))
        preprocessing = _metadata_preprocessing(checkpoint_manifest.preprocessing)
        checkpoint_size = _checkpoint_input_size(checkpoint_manifest)
        image_size = checkpoint_size or (
            640 if use_model_metadata_defaults else project_config.image_size
        )
        source = "checkpoint"

        requested_model = values.pop("model_preset", None)
        if requested_model is not None and requested_model != model_id:
            raise UnsupportedCheckpointError(
                "Checkpoint model cannot be overridden by a different model preset"
            )
        requested_model_id = values.pop("model_id", None)
        if requested_model_id is not None and requested_model_id != model_id:
            raise UnsupportedCheckpointError("Checkpoint model ID cannot be overridden")
        requested_architecture = values.pop("architecture", None)
        if requested_architecture is not None and requested_architecture != architecture:
            raise UnsupportedCheckpointError("Checkpoint architecture cannot be overridden")
        requested_preprocessing = values.pop("preprocessing", None)
        if requested_preprocessing is not None:
            requested_preprocessing = _preprocessing_override(
                preprocessing, requested_preprocessing
            )
            if (
                not allow_checkpoint_preprocessing_override
                and requested_preprocessing.to_dict() != preprocessing.to_dict()
            ):
                raise UnsupportedCheckpointError(
                    "Checkpoint preprocessing cannot be overridden for this run"
                )
            preprocessing = requested_preprocessing
        requested_size = values.pop("image_size", None)
        if requested_size is not None and checkpoint_size is not None:
            if int(requested_size) != checkpoint_size:
                raise UnsupportedCheckpointError(
                    "Checkpoint image_size cannot be overridden for this run"
                )
    elif hosted_model is not None:
        model_id = hosted_model.model_id
        model_preset = hosted_model.model_id
        architecture = hosted_model.architecture
        base_classes = tuple(hosted_model.source_classes)
        default_classes = hosted_model.source_classes if use_source_classes else project.classes
        classes = selected_target_classes(default_classes)
        preprocessing = hosted_model.preprocessing
        image_size = hosted_model.inference_image_size
        source = "huggingface"
        requested_model = values.pop("model_preset", None)
        if requested_model is not None and requested_model != hosted_model.model_id:
            raise UnsupportedCheckpointError(
                "AmphiLens model architecture cannot be overridden for this run"
            )
        requested_architecture = values.pop("architecture", None)
        if requested_architecture is not None and requested_architecture != architecture:
            raise UnsupportedCheckpointError(
                "AmphiLens model architecture cannot be overridden for this run"
            )
        requested_size = values.pop("image_size", None)
        if requested_size is not None and int(requested_size) != image_size:
            raise UnsupportedCheckpointError(
                "AmphiLens model input size cannot be overridden for this run"
            )
        requested_preprocessing = values.pop("preprocessing", None)
        if requested_preprocessing is not None:
            preprocessing = _preprocessing_override(preprocessing, requested_preprocessing)
    else:
        selected_model = use_model_metadata_defaults and "model_preset" in values
        model_preset = str(values.pop("model_preset", project_config.model_preset))
        preset = ModelCatalog().get(model_preset)
        model_id = str(values.pop("model_id", preset.model_id))
        architecture = str(values.pop("architecture", preset.architecture))
        classes = selected_target_classes(tuple(project.classes))
        if architecture not in {"yolo", "rtdetr", "faster_rcnn"}:
            raise ValidationError(f"Unsupported detector architecture: {architecture}")
        default_preprocessing = (
            _metadata_preprocessing(preset.metadata.get("preprocessing"))
            if selected_model
            else project_config.preprocessing
        )
        requested_preprocessing = values.pop("preprocessing", None)
        preprocessing = (
            _preprocessing_override(default_preprocessing, requested_preprocessing)
            if requested_preprocessing is not None
            else default_preprocessing
        )
        default_image_size = (
            preset.inference_image_size if selected_model else project_config.image_size
        )
        image_size = int(values.pop("image_size", default_image_size))
        source = "project"
        if checkpoint_path is not None:
            warnings.append("Checkpoint metadata unavailable; project configuration was used.")

    confidence = float(values.pop("confidence", project_config.confidence_threshold))
    if not 0 <= confidence <= 1:
        raise ValidationError("confidence must be between 0 and 1")
    device = str(values.pop("device", project_config.device))
    epochs = int(values.pop("epochs", project_config.epochs))
    batch_size = int(values.pop("batch_size", project_config.batch_size))
    patience = int(values.pop("patience", project_config.patience))
    seed = int(values.pop("seed", project_config.random_seed))
    freeze_strategy = str(values.pop("freeze_strategy", project_config.freeze_strategy))
    if values:
        raise ValidationError(f"Unsupported runtime configuration override: {next(iter(values))}")
    if image_size <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValidationError("image_size, epochs, and batch_size must be positive")
    if patience < 0:
        raise ValidationError("patience cannot be negative")

    return EffectiveRunConfiguration(
        model_preset=model_preset,
        model_id=model_id,
        architecture=architecture,
        classes=tuple(classes),
        preprocessing=preprocessing,
        image_size=image_size,
        confidence=confidence,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seed=seed,
        freeze_strategy=freeze_strategy,
        source=source,
        warnings=tuple(warnings),
        hosted_model=hosted_metadata,
        base_classes=base_classes,
    )


def discover_checkpoint_manifest(checkpoint: str | Path) -> CheckpointManifest | None:
    """Load the conventional sibling checkpoint manifest when it exists."""
    path = Path(checkpoint).expanduser().resolve()
    candidate = path / "checkpoint.json" if path.is_dir() else path
    if candidate.name != "checkpoint.json":
        candidate = path.parent / "checkpoint.json"
    if not candidate.is_file():
        return None
    try:
        return CheckpointManifest.from_dict(read_json(candidate))
    except Exception as exc:  # noqa: BLE001 - normalize malformed manifests
        raise ValidationError(f"Cannot read checkpoint manifest: {candidate}") from exc
