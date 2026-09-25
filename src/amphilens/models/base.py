"""Protocols shared by detector implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

from ..core import CheckpointManifest, DetectionRecord, InferenceConfig


class DetectorBackend(Protocol):
    model_id: str
    architecture: str

    def predict(
        self, image_paths: Iterable[Path], config: InferenceConfig
    ) -> Iterable[DetectionRecord]: ...

    def train(
        self,
        dataset_yaml: str | Path,
        output_dir: str | Path,
        config: dict,
        resume_from: CheckpointManifest | None = None,
    ) -> Path: ...
