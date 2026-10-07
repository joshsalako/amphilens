"""Checkpoint-aware training orchestration."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .core import CheckpointManifest, ValidationError, atomic_write_json, read_json
from .dataset import DatasetSnapshot
from .preprocessing import PreprocessingConfig


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 100
    image_size: int = 640
    batch_size: int = 16
    patience: int = 25
    seed: int = 42
    device: str = "auto"
    run_name: str = "train"
    preprocessing: PreprocessingConfig | dict[str, Any] | str = field(
        default_factory=PreprocessingConfig
    )
    freeze_strategy: str = "none"
    evaluation: str = "not evaluated"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.image_size <= 0 or self.batch_size <= 0:
            raise ValidationError("epochs, image_size, and batch_size must be positive")
        if self.patience < 0:
            raise ValidationError("patience cannot be negative")
        self.preprocessing = PreprocessingConfig.from_any(self.preprocessing)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["preprocessing"] = PreprocessingConfig.from_any(self.preprocessing).to_dict()
        return data


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
    preprocessing: PreprocessingConfig | dict[str, Any] | str,
    resume_from: CheckpointManifest | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    extra_training_config: dict[str, Any] | None = None,
) -> TrainingResult:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    training_config = config.to_dict()
    preprocessing_data = (
        preprocessing.to_dict()
        if isinstance(preprocessing, PreprocessingConfig)
        else dict(preprocessing)
        if isinstance(preprocessing, dict)
        else PreprocessingConfig.from_any(preprocessing).to_dict()
    )
    training_config["preprocessing"] = preprocessing_data
    for key, value in (extra_training_config or {}).items():
        if key == "cloud" and not isinstance(value, dict):
            raise ValidationError("Cloud training provenance must be a mapping")
        training_config[key] = value
    def report(values: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        progress_callback({"phase": "training", **values})

    train_options = {"resume_from": resume_from}
    if progress_callback is not None:
        progress_callback(
            {"phase": "training", "message": "Starting model training", "progress": 0.0}
        )
        train_options["progress_callback"] = report
    try:
        checkpoint = (
            Path(detector.train(dataset_yaml, output, training_config, **train_options))
            .expanduser()
            .resolve()
        )
    except Exception as exc:
        if progress_callback is not None:
            progress_callback(
                {"phase": "failed", "message": "Model training failed", "error": str(exc)}
            )
        raise
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "finalizing",
                "message": "Saving the trained model and metrics",
                "progress": 1.0,
            }
        )
    return finalize_training_outputs(
        detector,
        checkpoint=checkpoint,
        output_dir=output,
        config=config,
        preprocessing=preprocessing_data,
        resume_from=resume_from,
        training_config=training_config,
    )


def finalize_training_outputs(
    detector,
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
    preprocessing: dict[str, Any],
    resume_from: CheckpointManifest | None = None,
    training_config: dict[str, Any] | None = None,
) -> TrainingResult:
    """Materialize the canonical local artifacts and provenance for one training run."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValidationError(f"Training checkpoint does not exist: {checkpoint}")
    final_training_config = dict(training_config or config.to_dict())
    final_training_config["preprocessing"] = dict(preprocessing)
    best_checkpoint = output / "best.pt"
    if checkpoint != best_checkpoint:
        shutil.copy2(checkpoint, best_checkpoint)
    last_checkpoint = output / "last.pt"
    source_last = checkpoint.parent / "last.pt"
    if not last_checkpoint.is_file():
        shutil.copy2(source_last if source_last.is_file() else best_checkpoint, last_checkpoint)
    metrics_path = output / "metrics.json"
    if not metrics_path.is_file():
        atomic_write_json(
            metrics_path,
            {
                "evaluation": config.evaluation,
                "status": "not provided by detector adapter",
                "checkpoint": str(best_checkpoint),
            },
        )
    manifest = CheckpointManifest.create(
        best_checkpoint,
        model_id=detector.model_id,
        architecture=detector.architecture,
        classes=detector.classes,
        preprocessing=preprocessing,
        parent_checkpoint=resume_from.checkpoint_path if resume_from else None,
        training_config=final_training_config,
    )
    atomic_write_json(output / "checkpoint.json", manifest.to_dict())
    return TrainingResult(checkpoint=best_checkpoint, manifest=manifest)


def train_snapshot_and_register(
    detector,
    *,
    snapshot: DatasetSnapshot,
    output_dir: str | Path,
    config: TrainingConfig,
    preprocessing: PreprocessingConfig | dict[str, Any] | str | None = None,
    resume_from: CheckpointManifest | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    extra_training_config: dict[str, Any] | None = None,
) -> TrainingResult:
    """Prepare an immutable snapshot and train through the existing adapter contract."""
    output = Path(output_dir).expanduser().resolve()
    selected = PreprocessingConfig.from_any(preprocessing or config.preprocessing)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "dataset_preparation",
                "message": f"Preparing {len(snapshot.manifest.images)} labeled images",
                "completed": 0,
                "total": len(snapshot.manifest.images),
                "progress": 0.0,
            }
        )
    try:
        dataset_yaml = snapshot.to_yolo_dataset(
            output / "prepared-dataset",
            preprocessing=selected,
        )
    except Exception as exc:
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "failed",
                    "message": "Preparing the training dataset failed",
                    "error": str(exc),
                }
            )
        raise
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "dataset_preparation",
                "message": "Labeled images are ready for training",
                "completed": len(snapshot.manifest.images),
                "total": len(snapshot.manifest.images),
                "progress": 1.0,
            }
        )
    return train_and_register(
        detector,
        dataset_yaml=dataset_yaml,
        output_dir=output,
        config=config,
        preprocessing=selected.to_dict(),
        resume_from=resume_from,
        progress_callback=progress_callback,
        extra_training_config=extra_training_config,
    )


def load_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "checkpoint.json"
    return CheckpointManifest.from_dict(read_json(manifest_path))
