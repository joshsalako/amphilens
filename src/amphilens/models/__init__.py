"""Detector backends and model metadata."""

from .backends import (
    FasterRCNNDetector,
    UltralyticsDetector,
    load_detector,
    load_preset_detector,
)
from .catalog import ModelCatalog, ModelPreset, ModelSource
from .faster_rcnn_training import FasterRCNNDatasetSpec, FasterRCNNTrainer

__all__ = [
    "FasterRCNNDatasetSpec",
    "FasterRCNNDetector",
    "FasterRCNNTrainer",
    "UltralyticsDetector",
    "load_detector",
    "load_preset_detector",
    "ModelCatalog",
    "ModelPreset",
    "ModelSource",
]
