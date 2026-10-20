"""Dataset, training, active-learning, CVAT, and cloud training workflows."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .cancellation import WorkflowCancelled, cancellation_requested, check_cancellation
from .core import ProjectStore
from .models import ModelCatalog
from .webapp_cloud import _cloud_job_dict
from .webapp_contracts import (
    ActiveLearningRequest,
    CloudTrainingRequest,
    CvatCycleRequest,
    CvatProjectImportRequest,
    DatasetImportRequest,
    TrainingRequest,
)
from .webapp_jobs import DownloadArtifact, JobOutput
from .webapp_project import _phase_progress, _read_queue_paths, _training_checkpoint


def run_dataset_import_job(
    store: ProjectStore,
    request: DatasetImportRequest,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    snapshot = store.import_dataset(
        request.archive,
        class_mapping=request.class_mapping,
        display_name=request.display_name,
        progress_callback=progress_callback,
    )
    return {
        "message": (
            f"Imported {len(snapshot.manifest.images)} images as "
            f"{store.dataset_display_name(snapshot.root)}."
        ),
        "paths": {"snapshot": str(snapshot.root)},
        "snapshot": snapshot.manifest.to_dict(),
    }


def run_cvat_import_job(
    store: ProjectStore,
    request: CvatProjectImportRequest,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from .annotations.initial import CVATProjectImportService
    from .annotations.managed import CVATSdkTransport

    transport = CVATSdkTransport(server_url=request.server_url)
    snapshots = CVATProjectImportService(store, transport).import_project_tasks(
        request.project_id,
        class_mapping=request.class_mapping,
        display_name=request.display_name,
        progress_callback=progress_callback,
    )
    snapshot_summaries = [
        {
            "path": str(snapshot.root),
            "snapshot_id": snapshot.manifest.snapshot_id,
            "name": store.dataset_display_name(snapshot.root),
            "task_id": snapshot.manifest.source_provenance.get("task_id"),
            "task_name": snapshot.manifest.source_provenance.get("task_name"),
            "image_count": len(snapshot.manifest.images),
        }
        for snapshot in snapshots
    ]
    details = ", ".join(
        f"{item['name']} ({item['image_count']} images)" for item in snapshot_summaries
    )
    total_images = sum(item["image_count"] for item in snapshot_summaries)
    return {
        "message": (
            f"Imported {len(snapshots)} CVAT task lists ({total_images} images): {details}."
        ),
        "snapshots": snapshot_summaries,
    }


def _run_training_job(
    store: ProjectStore,
    request: TrainingRequest,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    training_snapshots,
) -> JobOutput:
    from .configuration import resolve_effective_configuration
    from .models import load_detector, load_preset_detector
    from .training import TrainingConfig, train_snapshot_and_register

    check_cancellation(progress_callback, "Training canceled before setup started.")
    project = store.load_manifest()
    snapshot, validation_snapshot, test_snapshot = training_snapshots(store, request)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "model_setup",
                "message": "Preparing the selected training model",
                "device": request.device or project.project_config.device,
            }
        )
    source, hosted_model, checkpoint, checkpoint_manifest = _training_checkpoint(
        request,
        progress_callback=_phase_progress(progress_callback, "model_download"),
    )
    check_cancellation(progress_callback, "Training canceled after model preparation.")
    overrides: dict[str, Any] = {}
    if request.epochs is not None:
        overrides["epochs"] = request.epochs
    if request.batch_size is not None:
        overrides["batch_size"] = request.batch_size
    if request.device is not None:
        overrides["device"] = request.device
    if request.patience is not None:
        overrides["patience"] = request.patience
    if request.model_preset is not None:
        overrides["model_preset"] = request.model_preset
    if request.preprocessing is not None:
        overrides["preprocessing"] = request.preprocessing
    overrides["freeze_strategy"] = "paper-phased"
    effective = resolve_effective_configuration(
        project,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_path=checkpoint or None,
        hosted_model=hosted_model,
        target_classes=tuple(project.classes),
        use_model_metadata_defaults=True,
        allow_checkpoint_preprocessing_override=True,
        overrides=overrides,
    )
    detector = (
        load_detector(
            checkpoint,
            architecture=effective.architecture,
            classes=list(effective.classes),
            model_id=effective.model_id,
            checkpoint_manifest=checkpoint_manifest,
            preprocessing=effective.preprocessing.to_dict(),
            allow_preprocessing_override=True,
            allow_class_override=checkpoint_manifest is not None,
        )
        if checkpoint
        else load_preset_detector(
            ModelCatalog().get(effective.model_preset), classes=list(effective.classes)
        )
    )
    check_cancellation(progress_callback, "Training canceled before dataset preparation.")
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "dataset_preparation",
                "message": "Model ready; preparing the labeled dataset",
                "device": effective.device,
            }
        )
    default_output = (
        store.root / "checkpoints" / f"cycle-{len(list((store.root / 'checkpoints').glob('*')))}"
    )
    output_dir = Path(request.output_dir or default_output)
    result = train_snapshot_and_register(
        detector,
        snapshot=snapshot,
        output_dir=output_dir,
        config=TrainingConfig(
            epochs=effective.epochs,
            batch_size=effective.batch_size,
            image_size=effective.image_size,
            patience=effective.patience,
            seed=effective.seed,
            device=effective.device,
            freeze_strategy="paper-phased",
            preprocessing=effective.preprocessing,
            evaluation=(
                "test set"
                if test_snapshot or request.data_mode == "auto-split"
                else "not evaluated"
            ),
            metadata={
                "effective_configuration": effective.to_dict(),
                "allow_checkpoint_preprocessing_override": checkpoint_manifest is not None,
                "allow_checkpoint_class_override": checkpoint_manifest is not None,
            },
        ),
        preprocessing=effective.preprocessing,
        validation_snapshot=validation_snapshot,
        test_snapshot=test_snapshot,
        data_mode=request.data_mode,
        allow_image_level_fallback=request.allow_image_level_split,
        resume_from=checkpoint_manifest,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "completed",
                "message": "Training and checkpoint registration completed",
                "progress": 1.0,
                "epoch": effective.epochs,
                "epochs": effective.epochs,
                "device": effective.device,
            }
        )
    return JobOutput(
        result={
            "message": f"Training finished from {source}: {result.checkpoint}",
            "paths": {
                "checkpoint": str(result.checkpoint),
                "manifest": str(output_dir / "checkpoint.json"),
            },
            "evaluation": result.manifest.training_config.get("evaluation", "not evaluated"),
        },
        downloads=[
            DownloadArtifact(result.checkpoint, "Model checkpoint", "best.pt"),
            DownloadArtifact(
                output_dir / "checkpoint.json",
                "Checkpoint manifest",
                "checkpoint.json",
                "application/json",
            ),
        ],
    )


def run_active_learning_job(
    store: ProjectStore,
    request: ActiveLearningRequest,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> JobOutput:
    from .active_learning import HybridPPALConfig, HybridPPALStrategy, PPALCalibration
    from .curation import write_selection_artifacts
    from .inference import read_predictions_csv

    check_cancellation(progress_callback, "Active learning selection canceled before setup.")
    if progress_callback is not None:
        progress_callback(
            {"phase": "selection", "message": "Loading calibration and prediction evidence"}
        )
    calibration_path = Path(request.calibration).expanduser().resolve()
    features_path = Path(request.features).expanduser().resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration evidence was not found: {calibration_path}")
    if not features_path.is_file():
        raise FileNotFoundError(f"Feature evidence was not found: {features_path}")
    predictions_path = Path(request.predictions).expanduser().resolve()
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Prediction CSV was not found: {predictions_path}")
    calibration = PPALCalibration(**json.loads(calibration_path.read_text(encoding="utf-8")))
    config = HybridPPALConfig(
        budget=request.budget,
        seed=request.seed,
        pool_multiplier=request.pool_multiplier,
    )
    check_cancellation(progress_callback, "Active learning selection canceled before ranking.")
    if progress_callback is not None:
        progress_callback({"phase": "selection", "message": "Ranking images for annotation"})
    selected = HybridPPALStrategy(config).select(
        read_predictions_csv(predictions_path),
        calibration,
        features=json.loads(features_path.read_text(encoding="utf-8")),
        cancellation_requested=getattr(progress_callback, "cancel_requested", None),
    )
    check_cancellation(progress_callback, "Active learning selection canceled before saving.")
    if progress_callback is not None:
        progress_callback({"phase": "saving_results", "message": "Saving the annotation queue"})
    artifacts = write_selection_artifacts(selected, calibration, config, request.output)
    return JobOutput(
        result={
            "message": f"Selected {len(selected)} images for annotation.",
            "paths": {key: str(path) for key, path in artifacts.items()},
        },
        downloads=[
            DownloadArtifact(
                artifacts["queue_csv"], "Selection queue", "selection_queue.csv", "text/csv"
            ),
            DownloadArtifact(
                artifacts["calibration_json"], "Calibration", "calibration.json", "application/json"
            ),
            DownloadArtifact(
                artifacts["selection_json"],
                "Selection details",
                "selection.json",
                "application/json",
            ),
        ],
    )


def run_cvat_cycle_job(store: ProjectStore, request: CvatCycleRequest) -> dict[str, Any]:
    from .annotations.managed import CVATSdkTransport, ManagedCVATCycleService, selection_hash

    transport = CVATSdkTransport(server_url=request.server_url)
    service = ManagedCVATCycleService(store, transport, server_url=transport.server_url)
    if request.action == "start":
        if not request.queue:
            raise ValueError("Select a queue CSV or image folder before sending it to CVAT")
        paths = _read_queue_paths(request.queue)
        if not paths:
            raise ValueError("The selection queue contains no images")
        manifest = service.start(
            cycle=request.cycle,
            image_paths=paths,
            selection_hash=selection_hash(paths),
        )
        return {"message": "CVAT task is ready for annotation.", "manifest": manifest.to_dict()}
    if request.action == "refresh":
        manifest = service.refresh(request.cycle)
        return {"message": f"CVAT cycle status: {manifest.state}.", "manifest": manifest.to_dict()}
    snapshot = service.continue_cycle(request.cycle)
    return {
        "message": "CVAT annotations were imported into a new immutable dataset snapshot.",
        "snapshot": str(snapshot.root),
        "manifest": snapshot.manifest.to_dict(),
    }


def _run_cloud_training_job(
    store: ProjectStore,
    request: CloudTrainingRequest,
    credentials_store,
    provider_settings_store=None,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cloud_service,
    training_snapshots,
) -> dict[str, Any]:
    from .cloud.models import CloudConsent
    from .configuration import resolve_effective_configuration

    service, credentials = cloud_service(
        store, credentials_store, provider_settings_store, request.provider
    )
    if request.provider == "modal" and credentials is None:
        raise ValueError(
            "Modal credentials are not configured. Save credentials or set both "
            "Modal environment variables."
        )
    check_cancellation(progress_callback, "Cloud training canceled before setup started.")
    snapshot, validation_snapshot, test_snapshot = training_snapshots(store, request)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "model_setup",
                "message": f"Preparing the model and dataset for {request.provider}",
            }
        )
    source, hosted_model, checkpoint, checkpoint_manifest = _training_checkpoint(
        request,
        progress_callback=_phase_progress(progress_callback, "model_download"),
        download_hosted=False,
    )
    project = store.load_manifest()
    overrides: dict[str, Any] = {"device": "cuda"}
    if request.epochs is not None:
        overrides["epochs"] = request.epochs
    if request.batch_size is not None:
        overrides["batch_size"] = request.batch_size
    if request.patience is not None:
        overrides["patience"] = request.patience
    if request.model_preset is not None:
        overrides["model_preset"] = request.model_preset
    if request.preprocessing is not None:
        overrides["preprocessing"] = request.preprocessing
    overrides["freeze_strategy"] = "paper-phased"
    effective = resolve_effective_configuration(
        project,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_path=checkpoint or None,
        hosted_model=hosted_model,
        target_classes=tuple(project.classes),
        use_model_metadata_defaults=True,
        allow_checkpoint_preprocessing_override=True,
        overrides=overrides,
    )
    image_size = effective.image_size
    training_config = {
        "epochs": effective.epochs,
        "image_size": image_size,
        "batch_size": effective.batch_size,
        "patience": effective.patience,
        "seed": effective.seed,
        "device": "cuda",
        "evaluation": (
            "test set" if test_snapshot or request.data_mode == "auto-split" else "not evaluated"
        ),
        "data_mode": request.data_mode,
        "validation_strategy": (
            "training-set-monitor"
            if request.data_mode == "training-monitor" or validation_snapshot is None
            else "separate-snapshot"
        ),
        "allow_checkpoint_preprocessing_override": checkpoint_manifest is not None,
        "allow_checkpoint_class_override": checkpoint_manifest is not None,
        "snapshot_ids": {
            "train": snapshot.manifest.snapshot_id,
            **(
                {"validation": validation_snapshot.manifest.snapshot_id}
                if validation_snapshot
                else {}
            ),
            **({"test": test_snapshot.manifest.snapshot_id} if test_snapshot else {}),
        },
        "freeze_strategy": "paper-phased",
    }
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "uploading_dataset",
                "message": f"Uploading the approved training snapshot to {request.provider}",
                "gpu": request.gpu,
            }
        )
    check_cancellation(progress_callback, "Cloud training canceled before provider submission.")
    job = service.submit(
        snapshot_path=request.snapshot_path,
        validation_snapshot_path=(str(validation_snapshot.root) if validation_snapshot else None),
        test_snapshot_path=str(test_snapshot.root) if test_snapshot else None,
        data_mode=request.data_mode,
        allow_image_level_fallback=request.allow_image_level_split,
        effective_configuration=effective.to_dict(),
        training_config=training_config,
        consent=CloudConsent(
            acknowledged=request.acknowledged,
            uploads_dataset=request.uploads_dataset,
            estimated_usd=request.estimated_usd,
            max_cost_usd=request.max_cost_usd,
        ),
        gpu=request.gpu,
        base_checkpoint=checkpoint or None,
        base_manifest=checkpoint_manifest,
    )
    if cancellation_requested(progress_callback):
        service.cancel(job.run_id)
        raise WorkflowCancelled("Cloud training canceled and provider cancellation requested.")
    if progress_callback is not None:
        try:
            progress_callback(
                {
                    "phase": "submitted",
                    "message": (
                        f"{request.provider} accepted the training job; "
                        "waiting for its first update"
                    ),
                    "gpu": request.gpu,
                    "progress": 0.0,
                }
            )
        except WorkflowCancelled:
            service.cancel(job.run_id)
            raise
    if cancellation_requested(progress_callback):
        service.cancel(job.run_id)
        raise WorkflowCancelled("Cloud training canceled and provider cancellation requested.")
    return {"message": f"Cloud training submitted: {job.run_id}.", "job": _cloud_job_dict(job)}
