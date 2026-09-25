"""Checkpoint-aware training orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .core import CheckpointManifest, ValidationError


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 100
    image_size: int = 640
    batch_size: int = 16
    patience: int = 25
    seed: int = 42
    device: str = "auto"
    run_name: str = "train"
    preprocessing: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.image_size <= 0 or self.batch_size <= 0:
            raise ValidationError("epochs, image_size, and batch_size must be positive")
        if self.patience < 0:
            raise ValidationError("patience cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingResult:
    checkpoint: Path
    manifest: CheckpointManifest


def train_and_register(
    detector,
    *,
    dataset_yaml: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
    preprocessing: dict[str, Any],
    resume_from: CheckpointManifest | None = None,
) -> TrainingResult:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    training_config = config.to_dict()
    training_config["preprocessing"] = preprocessing
    checkpoint = (
        Path(
            detector.train(
                dataset_yaml,
                output,
                training_config,
                resume_from=resume_from,
            )
        )
        .expanduser()
        .resolve()
    )
    manifest = CheckpointManifest.create(
        checkpoint,
        model_id=detector.model_id,
        architecture=detector.architecture,
        classes=detector.classes,
        preprocessing=preprocessing,
        parent_checkpoint=resume_from.checkpoint_path if resume_from else None,
        training_config=training_config,
    )
    (output / "checkpoint.json").write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    return TrainingResult(checkpoint=checkpoint, manifest=manifest)


def load_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "checkpoint.json"
    return CheckpointManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
