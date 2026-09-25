"""AmphiLens: reproducible wildlife camera-trap detection."""

from .core import (
    CheckpointManifest,
    DetectionRecord,
    InferenceConfig,
    ModelManifest,
    ProjectManifest,
    ProjectStore,
    RunManifest,
)

__all__ = [
    "CheckpointManifest",
    "DetectionRecord",
    "InferenceConfig",
    "ModelManifest",
    "ProjectManifest",
    "ProjectStore",
    "RunManifest",
]

__version__ = "0.1.0"

