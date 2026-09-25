"""Explicit model presets and checkpoint provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..core import ValidationError


@dataclass(frozen=True, slots=True)
class ModelSource:
    kind: str
    identifier: str
    is_paper_specific: bool = False
    license: str = "unknown"

    def __post_init__(self) -> None:
        if self.kind not in {"official-general-purpose", "paper-specific", "local"}:
            raise ValidationError(f"Unsupported model source kind: {self.kind}")
        if not self.identifier.strip():
            raise ValidationError("Model source identifier cannot be empty")
        if self.kind == "paper-specific" and not self.is_paper_specific:
            raise ValidationError("Paper-specific model sources must be marked as such")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelPreset:
    model_id: str
    architecture: str
    size: str
    checkpoint_reference: str
    checkpoint_source: str = "official-general-purpose"
    license: str = "unknown"
    supports_inference: bool = True
    supports_training: bool = True
    supported_preprocessing: tuple[str, ...] = ("rgb", "grayscale", "clahe")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.architecture not in {"yolo", "rtdetr", "faster_rcnn"}:
            raise ValidationError(f"Unsupported model architecture: {self.architecture}")
        if self.checkpoint_source not in {
            "official-general-purpose",
            "paper-specific",
            "local",
        }:
            raise ValidationError(f"Unsupported checkpoint source: {self.checkpoint_source}")
        if (
            not self.model_id.strip()
            or not self.size.strip()
            or not self.checkpoint_reference.strip()
        ):
            raise ValidationError("Model preset id, size, and checkpoint reference are required")
        if not self.supports_inference and not self.supports_training:
            raise ValidationError("Model preset must support inference or training")

    @property
    def source(self) -> ModelSource:
        return ModelSource(
            kind=self.checkpoint_source,
            identifier=self.checkpoint_reference,
            is_paper_specific=self.checkpoint_source == "paper-specific",
            license=self.license,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["supported_preprocessing"] = list(self.supported_preprocessing)
        return data


class ModelCatalog:
    """The deliberately small v1 catalog; unsupported sizes stay out of the UI."""

    def __init__(self, presets: list[ModelPreset] | None = None):
        values = presets or [
            ModelPreset(
                model_id="yolo26-l",
                architecture="yolo",
                size="large",
                checkpoint_reference="yolo26l.pt",
                license="Ultralytics license; verify for deployment",
                metadata={"family": "YOLO26", "paper_equivalent": False},
            ),
            ModelPreset(
                model_id="rtdetr-l",
                architecture="rtdetr",
                size="large",
                checkpoint_reference="rtdetr-l.pt",
                license="Ultralytics license; verify for deployment",
                metadata={"family": "RT-DETR", "paper_equivalent": False},
            ),
            ModelPreset(
                model_id="faster-rcnn-resnet50",
                architecture="faster_rcnn",
                size="resnet50-fpn-v2",
                checkpoint_reference="torchvision://fasterrcnn_resnet50_fpn_v2",
                license="BSD-3-Clause",
                metadata={"family": "Torchvision Faster R-CNN", "paper_equivalent": False},
            ),
        ]
        self._presets = {preset.model_id: preset for preset in values}
        if len(self._presets) != len(values):
            raise ValidationError("Model preset ids must be unique")

    @property
    def default(self) -> ModelPreset:
        return self.get("yolo26-l")

    def get(self, model_id: str) -> ModelPreset:
        try:
            return self._presets[model_id]
        except KeyError as exc:
            raise ValidationError(f"Unknown model preset: {model_id}") from exc

    def list(self) -> list[ModelPreset]:
        return [self._presets[key] for key in sorted(self._presets)]
