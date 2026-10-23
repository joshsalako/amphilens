"""Project summaries and shared workflow resolution helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .cancellation import check_cancellation
from .core import ProjectStore
from .locations import open_project
from .models import ModelCatalog
from .webapp_contracts import (
    CloudEstimateRequest,
    CloudTrainingRequest,
    PredictionsRequest,
    TrainingRequest,
)


def _model_summaries() -> list[dict[str, Any]]:
    return [
        {
            "model_id": item.model_id,
            "name": item.display_name,
            "architecture": item.architecture,
            "inference_image_size": item.inference_image_size,
            "preprocessing": item.metadata.get("preprocessing"),
        }
        for item in ModelCatalog().list()
    ]


def _hosted_model_summaries() -> list[dict[str, Any]]:
    from .models.hosted_models import list_hosted_models

    return [model.to_summary() for model in list_hosted_models()]


def select_training_source(
    training_source: str | None,
    checkpoint: str | None,
    hosted_model_id: str | None,
) -> str:
    """Resolve the training initializer while preserving old checkpoint requests."""
    selected = training_source or ("project-checkpoint" if checkpoint else "general-pretrained")
    if selected not in {"general-pretrained", "amphilens-pretrained", "project-checkpoint"}:
        raise ValueError(
            "Choose General pretrained weights, an AmphiLens model, or a project checkpoint"
        )
    if selected == "general-pretrained":
        if checkpoint or hosted_model_id:
            raise ValueError(
                "General pretrained weights cannot be combined with another checkpoint"
            )
    elif selected == "amphilens-pretrained":
        if checkpoint:
            raise ValueError("Choose either an AmphiLens model or a project checkpoint")
        if not hosted_model_id:
            raise ValueError("Choose an AmphiLens pretrained model")
    else:
        if not checkpoint:
            raise ValueError("Choose an existing project checkpoint")
        if hosted_model_id:
            raise ValueError("Choose either a project checkpoint or an AmphiLens model")
    return selected


def _cloud_estimate_image_size(request: CloudEstimateRequest) -> int:
    """Resolve the model's recorded input size for estimates and remote training."""
    source = request.training_source or (
        "amphilens-pretrained" if request.hosted_model_id else "general-pretrained"
    )
    if source not in {"general-pretrained", "amphilens-pretrained", "project-checkpoint"}:
        raise ValueError("Choose a supported training source before requesting an estimate")
    if source == "amphilens-pretrained":
        if not request.hosted_model_id:
            raise ValueError("Choose an AmphiLens pretrained model before requesting an estimate")
        from .models.hosted_models import get_hosted_model

        return get_hosted_model(request.hosted_model_id).inference_image_size
    if source == "project-checkpoint":
        if not request.checkpoint:
            raise ValueError("Choose a project checkpoint before requesting an estimate")
        from .configuration import _checkpoint_input_size, discover_checkpoint_manifest

        manifest = discover_checkpoint_manifest(request.checkpoint)
        if manifest is not None:
            return _checkpoint_input_size(manifest) or request.image_size
        return request.image_size
    if request.hosted_model_id:
        raise ValueError("Choose either General pretrained weights or an AmphiLens model")
    if request.model_preset:
        return ModelCatalog().get(request.model_preset).inference_image_size
    return request.image_size


def _project_summary(store: ProjectStore) -> dict[str, Any]:
    from .core import iter_images
    from .dataset import DatasetSnapshot
    from .training import load_checkpoint_manifest

    manifest = store.load_manifest()
    models = _model_summaries()
    datasets = []
    for path in sorted(
        candidate
        for candidate in (store.root / "datasets").glob("*")
        if candidate.is_dir() and (candidate / "manifest.json").is_file()
    ):
        snapshot = DatasetSnapshot.load(path)
        datasets.append(
            {
                "path": str(path),
                "name": store.dataset_display_name(path),
                "snapshot_id": snapshot.manifest.snapshot_id,
                "image_count": len(snapshot.manifest.images),
                "source_type": snapshot.manifest.source_provenance.get("source_type"),
                "task_id": snapshot.manifest.source_provenance.get("task_id"),
                "task_name": snapshot.manifest.source_provenance.get("task_name"),
            }
        )
    checkpoints = []
    checkpoint_root = store.root / "checkpoints"
    if checkpoint_root.is_dir():
        for path in sorted(checkpoint_root.rglob("checkpoint.json")):
            checkpoint = load_checkpoint_manifest(path)
            checkpoints.append(
                {
                    "path": str(checkpoint.checkpoint_path),
                    "name": path.parent.name,
                    "model_id": checkpoint.model_id,
                    "architecture": checkpoint.architecture,
                    "classes": list(checkpoint.classes),
                    "preprocessing": checkpoint.preprocessing,
                    "image_size": checkpoint.training_config.get("image_size"),
                }
            )
    image_count = sum(1 for _ in iter_images(manifest.image_roots))
    run_count = sum(1 for _ in (store.root / "runs").glob("*/run.json"))
    return {
        "path": str(store.root),
        "name": manifest.name,
        "classes": list(manifest.classes),
        "model_preset": manifest.project_config.model_preset,
        "preprocessing": manifest.project_config.preprocessing.to_dict(),
        "image_roots": list(manifest.image_roots),
        "counts": {"images": image_count, "datasets": len(datasets), "runs": run_count},
        "datasets": datasets,
        "checkpoints": checkpoints,
        "models": models,
        "hosted_models": _hosted_model_summaries(),
    }


def _read_queue_paths(queue_or_folder: str) -> list[Path]:
    source = Path(queue_or_folder).expanduser().resolve()
    if source.is_dir():
        from .core import iter_images

        return iter_images([source])
    if source.is_file() and source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "image_path" not in reader.fieldnames:
                raise ValueError("Selection CSV must contain an image_path column")
            return [Path(row["image_path"]).expanduser().resolve() for row in reader]
    raise ValueError("Choose an image folder or selection_queue.csv")


def _active_store(active_path: Path | None) -> ProjectStore:
    if active_path is None:
        raise ValueError("Create or open a project before using this workflow")
    return open_project(active_path)


def _project_snapshot(store: ProjectStore, snapshot_path: str | Path):
    from .dataset import DatasetSnapshot

    snapshot = DatasetSnapshot.load(snapshot_path)
    dataset_root = (store.root / "datasets").resolve()
    if not snapshot.root.is_relative_to(dataset_root):
        raise ValueError("Choose a labelled dataset snapshot saved in the active project")
    if list(snapshot.manifest.classes) != list(store.load_manifest().classes):
        raise ValueError("Dataset snapshot classes do not match the active project class order")
    if not snapshot.manifest.images:
        raise ValueError("The selected labelled dataset snapshot contains no images")
    return snapshot


def _training_snapshots(store: ProjectStore, request):
    """Resolve role snapshots and reject mode-incompatible selectors before model setup."""
    from .dataset import split_snapshot_images

    training = _project_snapshot(store, request.snapshot_path)
    validation = None
    test = None
    if request.data_mode == "separate-snapshots":
        if request.validation_snapshot_path:
            validation = _project_snapshot(store, request.validation_snapshot_path)
        if request.test_snapshot_path:
            test = _project_snapshot(store, request.test_snapshot_path)
        if validation and validation.manifest.snapshot_id == training.manifest.snapshot_id:
            raise ValueError("Choose a separate validation snapshot or leave validation empty")
        if test and test.manifest.snapshot_id == training.manifest.snapshot_id:
            raise ValueError("Choose a separate test snapshot or leave test empty")
        if validation and test and validation.manifest.snapshot_id == test.manifest.snapshot_id:
            raise ValueError("Choose different validation and test snapshots")
        role_images = {
            "train": {item.sha256 for item in training.manifest.images},
            "validation": (
                {item.sha256 for item in validation.manifest.images}
                if validation
                else {item.sha256 for item in training.manifest.images}
            ),
        }
        if test:
            role_images["test"] = {item.sha256 for item in test.manifest.images}
        for index, role in enumerate(role_images):
            for other in list(role_images)[index + 1 :]:
                if (role, other) == ("train", "validation") and validation is None:
                    continue
                if role_images[role] & role_images[other]:
                    raise ValueError(f"The {role} and {other} snapshots contain duplicate images")
    elif request.validation_snapshot_path or request.test_snapshot_path:
        raise ValueError(
            "Validation and test snapshot selectors are only used in separate-snapshot mode"
        )
    elif request.data_mode == "auto-split":
        project_config = store.load_manifest().project_config
        split_snapshot_images(
            training,
            seed=project_config.random_seed,
            allow_image_level_fallback=request.allow_image_level_split,
        )
    return training, validation, test


def _resolve_prediction_image_root(project, request: PredictionsRequest) -> Path:
    image_root = request.image_root
    if image_root is None or not image_root.strip():
        if not project.image_roots:
            raise ValueError("Choose an image folder in Find Wildlife before running detection.")
        image_root = project.image_roots[0]
    return Path(image_root).expanduser().resolve()


def _modal_prediction_timing_samples(
    store: ProjectStore, model_id: str, gpu: str, provider: str = "modal"
) -> list[float]:
    """Return per-image runtimes for the same model, GPU, and cloud provider."""
    timing_samples = []
    for path in (store.root / "runs").glob("*/cloud-cost.json"):
        try:
            sample = json.loads(path.read_text(encoding="utf-8"))
            if (
                sample.get("provider", "modal") == provider
                and sample.get("gpu") == gpu
                and sample.get("model_id") == model_id
                and int(sample.get("completed_images", 0)) > 0
                and float(sample.get("elapsed_seconds", 0.0)) > 0
            ):
                timing_samples.append(
                    float(sample["elapsed_seconds"]) / int(sample["completed_images"])
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return timing_samples


def _training_checkpoint(
    request: TrainingRequest | CloudTrainingRequest,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    download_hosted: bool = True,
):
    from .configuration import discover_checkpoint_manifest
    from .models.hosted_models import download_hosted_checkpoint, get_hosted_model

    checkpoint = request.checkpoint.strip() if request.checkpoint else ""
    source = select_training_source(
        request.training_source,
        checkpoint or None,
        request.hosted_model_id,
    )
    if source == "amphilens-pretrained":
        hosted_model = get_hosted_model(request.hosted_model_id or "")
        if not download_hosted:
            return source, hosted_model, "", None
        checkpoint_path = download_hosted_checkpoint(
            hosted_model.model_id,
            progress_callback=progress_callback,
        )
        return source, hosted_model, str(checkpoint_path), None
    if source == "project-checkpoint":
        return source, None, checkpoint, discover_checkpoint_manifest(checkpoint)
    return source, None, "", None


def _phase_progress(
    callback: Callable[[dict[str, Any]], None] | None, phase: str
) -> Callable[[dict[str, Any]], None] | None:
    if callback is None:
        return None

    def report(values: dict[str, Any]) -> None:
        check_cancellation(callback)
        event = dict(values)
        if phase == "model_download" and "progress" in event:
            event["phase_progress"] = event.pop("progress")
        callback({"phase": phase, **event})
        check_cancellation(callback)

    report.cancel_requested = getattr(callback, "cancel_requested", None)
    return report
