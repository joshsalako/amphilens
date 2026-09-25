"""Model and checkpoint registry with compatibility and hash checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .core import (
    CheckpointManifest,
    ModelManifest,
    UnsupportedCheckpointError,
    ValidationError,
    atomic_write_json,
    read_json,
)


class ModelRegistry:
    """Filesystem model registry; weights remain at their recorded paths."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, model: ModelManifest, checkpoint: CheckpointManifest) -> None:
        model.validate()
        checkpoint.validate_compatibility(
            architecture=model.architecture,
            classes=model.classes,
            preprocessing=model.preprocessing,
        )
        if checkpoint.model_id != model.model_id:
            raise ValidationError("Model and checkpoint ids must match")
        directory = self.root / model.model_id
        atomic_write_json(directory / "model.json", model.to_dict())
        atomic_write_json(directory / "checkpoint.json", checkpoint.to_dict())

    def list_models(self) -> list[str]:
        return sorted(
            directory.name
            for directory in self.root.iterdir()
            if directory.is_dir() and (directory / "model.json").is_file()
        )

    def get(self, model_id: str) -> tuple[ModelManifest, CheckpointManifest]:
        directory = self.root / model_id
        try:
            model = ModelManifest.from_dict(read_json(directory / "model.json"))
            checkpoint = CheckpointManifest.from_dict(read_json(directory / "checkpoint.json"))
        except FileNotFoundError as exc:
            raise ValidationError(f"Registered model not found: {model_id}") from exc
        return model, checkpoint

    def resolve(
        self,
        model_id: str,
        *,
        architecture: str,
        classes: list[str],
        preprocessing: dict,
    ) -> CheckpointManifest:
        model, checkpoint = self.get(model_id)
        if model.architecture != architecture or model.classes != classes:
            raise UnsupportedCheckpointError(
                "Registered model metadata does not match the requested run"
            )
        checkpoint.validate_compatibility(
            architecture=architecture, classes=classes, preprocessing=preprocessing
        )
        checkpoint_path = Path(checkpoint.checkpoint_path)
        if not checkpoint_path.is_file():
            raise UnsupportedCheckpointError(f"Registered checkpoint is missing: {checkpoint_path}")
        current_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if current_hash != checkpoint.sha256:
            raise UnsupportedCheckpointError("Registered checkpoint hash does not match the file")
        return checkpoint
