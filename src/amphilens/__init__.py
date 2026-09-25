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
from .runs import RunSummary, run_resumable_inference

__all__ = [
    "CheckpointManifest",
    "DetectionRecord",
    "InferenceConfig",
    "ModelManifest",
    "ProjectManifest",
    "ProjectStore",
    "RunManifest",
    "RunSummary",
    "run_resumable_inference",
]

__version__ = "0.1.0"
