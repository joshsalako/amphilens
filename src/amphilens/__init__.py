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
from .curation import write_selection_artifacts
from .execution import JobHandle, JobSpec, JobStatus, LocalExecutionBackend
from .inference import read_predictions_csv
from .reporting import write_report
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
    "read_predictions_csv",
    "JobHandle",
    "JobSpec",
    "JobStatus",
    "LocalExecutionBackend",
    "write_selection_artifacts",
    "write_report",
]

__version__ = "0.1.0"
