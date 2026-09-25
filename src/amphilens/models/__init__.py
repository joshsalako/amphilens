"""Detector backends and model metadata."""

from .backends import (
    FasterRCNNDetector,
    UltralyticsDetector,
    load_detector,
)

__all__ = ["FasterRCNNDetector", "UltralyticsDetector", "load_detector"]
