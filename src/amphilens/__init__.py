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
from .execution import JobHandle, JobSpec, JobStatus, LocalExecutionBackend
from .curation import write_selection_artifacts

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
    "JobHandle",
    "JobSpec",
    "JobStatus",
    "LocalExecutionBackend",
    "write_selection_artifacts",
]

__version__ = "0.1.0"
