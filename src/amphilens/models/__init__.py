"""Detector backends and model metadata."""

from .backends import (
    FasterRCNNDetector,
    UltralyticsDetector,
    load_detector,
)
from .faster_rcnn_training import FasterRCNNDatasetSpec, FasterRCNNTrainer

__all__ = [
    "FasterRCNNDatasetSpec",
    "FasterRCNNDetector",
    "FasterRCNNTrainer",
    "UltralyticsDetector",
    "load_detector",
]
