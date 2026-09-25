"""AmphiLens: reproducible wildlife camera-trap detection."""

from .core import (
    ArtifactRecord,
    CheckpointManifest,
    DetectionRecord,
    InferenceConfig,
    ModelManifest,
    ProjectConfig,
    ProjectManifest,
    ProjectStore,
    RunManifest,
    atomic_write_json,
)
from .curation import write_selection_artifacts
from .execution import JobHandle, JobSpec, JobStatus, LocalExecutionBackend
from .inference import read_predictions_csv
from .preprocessing import PreprocessedImage, PreprocessingConfig, PreprocessingService
from .reporting import write_report
from .runs import RunSummary, run_resumable_inference

__all__ = [
    "CheckpointManifest",
    "ArtifactRecord",
    "DetectionRecord",
    "InferenceConfig",
    "ProjectConfig",
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
    "atomic_write_json",
    "PreprocessedImage",
    "PreprocessingConfig",
    "PreprocessingService",
]

__version__ = "0.1.0"
