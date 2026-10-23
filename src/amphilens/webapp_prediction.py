"""Prediction preflight and execution workflows."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .core import ProjectStore
from .models import ModelCatalog
from .webapp_contracts import (
    PredictionsRequest,
    _ensure_prediction_run_name_is_unique,
    _prediction_run_name,
)
from .webapp_jobs import DownloadArtifact, JobOutput, redact_sensitive_text
from .webapp_project import (
    _modal_prediction_timing_samples,
    _phase_progress,
    _resolve_prediction_image_root,
)


def prediction_preflight(
    store: ProjectStore,
    request: PredictionsRequest,
    *,
    credentials_store=None,
    provider_settings_store=None,
    check_connection: bool = False,
) -> dict[str, Any]:
    """Count local inputs and report available history before any cloud upload."""
    if request.execution == "local":
        raise ValueError("Cloud prediction preflight requires a cloud provider")
    from .cloud.estimate import (
        CPU_CORES,
        GPU_RATES_PER_SECOND,
        MEMORY_GIB,
        MODAL_CPU_RATE_PER_CORE_SECOND,
        MODAL_MEMORY_RATE_PER_GIB_SECOND,
    )
    from .configuration import discover_checkpoint_manifest
    from .core import iter_images

    project = store.load_manifest()
    _ensure_prediction_run_name_is_unique(store, request.run_name)
    image_root_path = _resolve_prediction_image_root(project, request)
    if not image_root_path.is_dir():
        raise ValueError(f"Image folder was not found: {image_root_path}")
    paths = iter_images([image_root_path])
    if not paths:
        raise ValueError(f"No supported images were found in: {image_root_path}")
    total_bytes = 0
    for path in paths:
        try:
            total_bytes += path.stat().st_size
        except OSError as exc:
            raise ValueError(
                "An image changed or became unavailable while preparing consent"
            ) from exc

    if request.hosted_model_id:
        model_id = request.hosted_model_id
    elif request.checkpoint:
        checkpoint = Path(request.checkpoint).expanduser().resolve()
        checkpoint_manifest = discover_checkpoint_manifest(checkpoint)
        model_id = (
            checkpoint_manifest.model_id
            if checkpoint_manifest
            else request.model_preset or project.project_config.model_preset
        )
    else:
        model_id = request.model_preset or project.project_config.model_preset

    provider = request.execution
    if provider == "modal":
        rate_per_second = (
            GPU_RATES_PER_SECOND[request.gpu]
            + MODAL_CPU_RATE_PER_CORE_SECOND * CPU_CORES
            + MODAL_MEMORY_RATE_PER_GIB_SECOND * MEMORY_GIB
        )
        rate_per_hour = rate_per_second * 3600
        price_source = "Modal GPU price table"
        region = ""
    else:
        from .cloud.provider_settings import AzureMLSettings, VertexAISettings
        from .cloud.providers import create_cloud_transport

        if provider_settings_store is None:
            raise ValueError("Cloud provider settings are unavailable")
        values = provider_settings_store.load().get(provider, {})
        settings_type = {"azure_ml": AzureMLSettings, "vertex_ai": VertexAISettings}[provider]
        transport = create_cloud_transport(provider, settings_type.from_mapping(values))
        provider_rate = transport.price_rate(request.gpu)
        rate_per_hour = provider_rate.usd_per_hour
        rate_per_second = rate_per_hour / 3600
        price_source = provider_rate.price_source
        region = provider_rate.region
        if check_connection:
            transport.probe()
    timing_samples = _modal_prediction_timing_samples(store, model_id, request.gpu, provider)
    estimated_cost = None
    if timing_samples:
        estimated_cost = round(
            sum(timing_samples) / len(timing_samples) * len(paths) * rate_per_second, 4
        )

    if check_connection:
        if provider != "modal":
            # The managed provider probe above validates identity, workspace/project, and storage.
            pass
        elif credentials_store is None:
            raise ValueError("Modal credentials are not configured")
        else:
            credentials = credentials_store.resolve()
            if credentials is None:
                raise ValueError("Modal credentials are not configured; save them in System health")
            from .cloud.modal_transport import ModalTransport

            ModalTransport(credentials).probe()
    return {
        "image_count": len(paths),
        "total_bytes": total_bytes,
        "gpu": request.gpu,
        "model_id": model_id,
        "timing_history_count": len(timing_samples),
        "estimated_cost_usd": estimated_cost,
        "rate_usd_per_hour": round(rate_per_hour, 6),
        "price_source": price_source,
        "region": region,
        "max_cost_usd": request.max_cost_usd,
        "estimate_note": (
            f"Estimate based on prior {provider} timing for this model and GPU; "
            "actual billing can vary."
            if estimated_cost is not None
            else f"No timing history is available for {provider} on this model and GPU."
        ),
    }


def run_prediction_job(
    store: ProjectStore,
    request: PredictionsRequest,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cloud_transport=None,
) -> JobOutput:
    from .configuration import discover_checkpoint_manifest, resolve_effective_configuration
    from .core import InferenceConfig, iter_images
    from .models import load_detector, load_preset_detector
    from .models.backends import describe_device_name, resolve_device_name
    from .models.hosted_models import (
        HostedClassMappedDetector,
        download_hosted_checkpoint,
        download_hosted_ncnn_artifact,
        get_hosted_model,
        resolve_class_mapping,
    )
    from .reporting import write_report
    from .runs import run_resumable_inference

    run_name = _prediction_run_name(request.run_name)
    project = store.load_manifest()
    _ensure_prediction_run_name_is_unique(store, run_name)
    image_root_path = _resolve_prediction_image_root(project, request)
    if not image_root_path.is_dir():
        raise ValueError(f"Image folder was not found: {image_root_path}")
    image_paths = iter_images([image_root_path])
    if not image_paths:
        raise ValueError(f"No supported images were found in: {image_root_path}")
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "model_setup",
                "message": f"Preparing model for {len(image_paths)} images",
                "completed": 0,
                "failed": 0,
                "total": len(image_paths),
                "remaining": len(image_paths),
            }
        )
    if request.execution != "local":
        if not request.acknowledged or not request.uploads_dataset:
            raise ValueError("Explicit image upload and cloud spending-limit consent is required")
        if cloud_transport is None:
            raise ValueError("Cloud provider is not configured for prediction")
        total_bytes = sum(path.stat().st_size for path in image_paths)
        if (
            request.expected_image_count != len(image_paths)
            or request.expected_total_bytes != total_bytes
        ):
            raise ValueError(
                "The selected images changed after cloud prediction consent; review the total again"
            )
    checkpoint = request.checkpoint.strip() if request.checkpoint else ""
    if request.hosted_model_id and (checkpoint or request.model_preset):
        raise ValueError("Choose either an AmphiLens pretrained model or another model")
    if request.class_mapping is not None and not request.hosted_model_id:
        raise ValueError("Class mappings are only used with an AmphiLens pretrained model")
    hosted_model = get_hosted_model(request.hosted_model_id) if request.hosted_model_id else None
    checkpoint_manifest = discover_checkpoint_manifest(checkpoint) if checkpoint else None
    if request.runtime == "ncnn":
        import importlib.util

        from .doctor import is_arm_linux

        if request.execution != "local" or request.device != "cpu":
            raise ValueError("NCNN is available for local CPU prediction only")
        if not is_arm_linux():
            raise ValueError("NCNN prediction is supported on ARM Linux devices only")
        if hosted_model is None:
            raise ValueError("NCNN prediction requires an AmphiLens hosted model")
        if not hosted_model.ncnn_available:
            raise ValueError(hosted_model.ncnn_unavailable_reason)
        if importlib.util.find_spec("ncnn") is None:
            raise ValueError(
                "NCNN is not installed. Install the `edge-ncnn` profile on ARM Linux first."
            )
    overrides: dict[str, Any] = {}
    if request.confidence is not None:
        overrides["confidence"] = request.confidence
    if request.execution != "local":
        actual_device = "cuda"
        requested_batch_size = request.batch_size or 8
    elif request.runtime == "ncnn":
        actual_device = "cpu"
        requested_batch_size = 1
    else:
        actual_device = resolve_device_name(request.device or project.project_config.device)
        requested_batch_size = request.batch_size or (
            8 if actual_device.startswith("cuda") or actual_device == "mps" else 1
        )
    effective_batch_size = 1 if actual_device == "cpu" else requested_batch_size
    overrides["device"] = actual_device
    overrides["batch_size"] = effective_batch_size
    if request.model_preset is not None:
        overrides["model_preset"] = request.model_preset
    if request.preprocessing is not None:
        overrides["preprocessing"] = request.preprocessing
    class_mapping = None
    if hosted_model is not None:
        class_mapping = resolve_class_mapping(
            hosted_model.source_classes,
            project.classes,
            request.class_mapping,
        )
        effective = resolve_effective_configuration(
            project,
            hosted_model=hosted_model,
            use_source_classes=True,
            use_model_metadata_defaults=True,
            overrides=overrides,
        )
        if request.execution == "local":
            if request.runtime == "ncnn":
                from .models.ncnn_backend import load_ncnn_detector

                artifact = download_hosted_ncnn_artifact(
                    hosted_model.model_id,
                    progress_callback=_phase_progress(progress_callback, "model_download"),
                )
                detector = load_ncnn_detector(
                    artifact,
                    architecture=effective.architecture,
                    classes=list(effective.classes),
                    model_id=effective.model_id,
                )
            else:
                checkpoint = str(
                    download_hosted_checkpoint(
                        hosted_model.model_id,
                        progress_callback=_phase_progress(progress_callback, "model_download"),
                    )
                )
                detector = load_detector(
                    checkpoint,
                    architecture=effective.architecture,
                    classes=list(effective.classes),
                    model_id=effective.model_id,
                    preprocessing=effective.preprocessing.to_dict(),
                )
            detector = HostedClassMappedDetector(detector, class_mapping, project.classes)
    else:
        effective = resolve_effective_configuration(
            project,
            checkpoint_manifest=checkpoint_manifest,
            checkpoint_path=checkpoint or None,
            use_model_metadata_defaults=True,
            allow_checkpoint_preprocessing_override=True,
            overrides=overrides,
        )
        if request.execution == "local":
            if not checkpoint and progress_callback is not None:
                progress_callback(
                    {
                        "phase": "model_setup",
                        "message": "Loading pretrained weights; first use may download them",
                        "progress": None,
                    }
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
                )
                if checkpoint
                else load_preset_detector(
                    ModelCatalog().get(effective.model_preset), classes=list(effective.classes)
                )
            )
    run_metadata = {
        "run_name": run_name,
        "image_root": str(image_root_path),
        "effective_configuration": effective.to_dict(),
        "class_mapping": class_mapping,
        "execution": request.execution,
        "runtime": request.runtime,
        "runtime_artifact": (
            {
                "format": "ncnn",
                "revision": artifact.revision,
                "adapter": artifact.manifest.adapter,
                "precision": artifact.manifest.precision,
                "files": [item.to_dict() for item in artifact.manifest.files],
            }
            if request.runtime == "ncnn"
            else (
                {
                    "format": "pytorch",
                    "revision": hosted_model.to_summary()["revision"],
                    "filename": hosted_model.artifact,
                    "sha256": hosted_model.sha256,
                }
                if hosted_model is not None
                else {
                    "format": "pytorch",
                    "checkpoint": checkpoint or effective.model_preset,
                }
            )
        ),
        "gpu": request.gpu if request.execution != "local" else None,
        "requested_batch_size": request.batch_size or "auto",
    }
    import hashlib
    import json

    run_signature = hashlib.sha256(
        json.dumps(run_metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    run_id = f"predict-{effective.model_id}-{run_signature}"
    output_dir = request.output_dir or str(store.root / "runs" / run_name)
    output_path = Path(output_dir).expanduser().resolve()
    provider_rate = None
    rate_per_second = None
    price_source = ""
    region = ""
    if request.execution != "local":
        from .cloud.estimate import MAX_FUNCTION_TIMEOUT_SECONDS
        from .cloud.prediction import CloudPredictionDetector, ModalPredictionDetector

        cancel_check = getattr(progress_callback, "cancel_requested", None)
        if callable(cancel_check) and cancel_check():
            from .runs import InferenceInterruption

            raise InferenceInterruption("Prediction canceled before starting cloud work.")

        if request.execution == "modal":
            from .cloud.estimate import (
                CPU_CORES,
                GPU_RATES_PER_SECOND,
                MEMORY_GIB,
                MODAL_CPU_RATE_PER_CORE_SECOND,
                MODAL_MEMORY_RATE_PER_GIB_SECOND,
            )

            rate_per_second = (
                GPU_RATES_PER_SECOND[request.gpu]
                + MODAL_CPU_RATE_PER_CORE_SECOND * CPU_CORES
                + MODAL_MEMORY_RATE_PER_GIB_SECOND * MEMORY_GIB
            )
            price_source = "Modal GPU price table"
        else:
            provider_rate = cloud_transport.price_rate(request.gpu)
            rate_per_second = provider_rate.usd_per_hour / 3600
            price_source = provider_rate.price_source
            region = provider_rate.region
        if request.expected_rate_usd_per_hour is not None and abs(
            rate_per_second * 3600 - request.expected_rate_usd_per_hour
        ) > max(0.01, request.expected_rate_usd_per_hour * 0.01):
            raise ValueError("Cloud GPU pricing changed after review; request a fresh preflight")

        prediction_job_key = f"pred-{uuid.uuid4().hex}"
        if hosted_model is not None:
            model_spec = {
                "source": "hosted",
                "model_id": effective.model_id,
                "hosted_model_id": hosted_model.model_id,
                "classes": list(project.classes),
                "class_mapping": class_mapping,
                "preprocessing": effective.preprocessing.to_dict(),
            }
        elif checkpoint:
            import hashlib

            checkpoint_path = Path(checkpoint).expanduser().resolve()
            digest = hashlib.sha256()
            with checkpoint_path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            checkpoint_sha256 = digest.hexdigest()
            checkpoint_remote_path = cloud_transport.upload_model_checkpoint(
                checkpoint_path, checkpoint_sha256, job_key=prediction_job_key
            )
            model_spec = {
                "source": "checkpoint",
                "model_id": effective.model_id,
                "architecture": effective.architecture,
                "classes": list(effective.classes),
                "preprocessing": effective.preprocessing.to_dict(),
                "checkpoint_remote_path": checkpoint_remote_path,
                "checkpoint_sha256": checkpoint_sha256,
            }
        else:
            model_spec = {
                "source": "preset",
                "model_id": effective.model_preset,
                "classes": list(effective.classes),
                "preprocessing": effective.preprocessing.to_dict(),
            }
        timing_samples = _modal_prediction_timing_samples(
            store, str(model_spec["model_id"]), request.gpu, request.execution
        )
        detector_type = (
            ModalPredictionDetector if request.execution == "modal" else CloudPredictionDetector
        )
        detector = detector_type(
            cloud_transport,
            job_key=prediction_job_key,
            gpu=request.gpu,
            model_spec=model_spec,
            timeout_seconds=MAX_FUNCTION_TIMEOUT_SECONDS,
            max_cost_usd=request.max_cost_usd,
            output_dir=output_dir,
            seconds_per_image=(
                sum(timing_samples) / len(timing_samples) if timing_samples else None
            ),
            progress_callback=progress_callback,
            image_count=len(image_paths),
            cancellation_requested=cancel_check if callable(cancel_check) else None,
            provider=request.execution,
            rate_per_second=rate_per_second,
            region=region,
            price_source=price_source,
        )
    run_directory_reserved = False
    if request.output_dir is None:
        try:
            output_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ValueError(
                f"Prediction run name '{run_name}' has already been used in this project. "
                "Choose a unique name."
            ) from exc
        run_directory_reserved = True
    try:
        summary = run_resumable_inference(
            detector,
            image_paths,
            InferenceConfig(
                model_id=effective.model_id,
                image_size=effective.image_size,
                confidence=effective.confidence,
                device=effective.device,
                batch_size=effective.batch_size,
                preprocessing=effective.preprocessing,
                run_id=run_id,
                metadata=run_metadata,
            ),
            output_dir,
            progress_callback=progress_callback,
        )
    finally:
        cleanup_progress = getattr(detector, "cleanup_progress", None)
        if callable(cleanup_progress):
            try:
                cleanup_progress()
            except Exception as exc:
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "cleanup",
                            "message": "Could not remove temporary Modal progress data",
                            "error": redact_sensitive_text(str(exc))[:240],
                        }
                    )
        if run_directory_reserved and not (output_path / "run.json").exists():
            try:
                output_path.rmdir()
            except OSError:
                pass
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "saving_results",
                "message": "Writing reports and updating the project",
                "completed": summary.completed_images,
                "failed": len(summary.failed_images),
                "skipped": len(summary.skipped_images),
                "total": summary.image_count,
                "remaining": max(
                    0,
                    summary.image_count
                    - summary.completed_images
                    - len(summary.failed_images)
                    - len(summary.skipped_images),
                ),
                "progress": 1.0,
                "device": getattr(detector, "device_name", effective.device),
            }
        )
    report = write_report(summary.predictions_csv, Path(output_dir) / "report")
    markdown = Path(report["markdown"])
    json_report = Path(report["summary_json"])
    skipped_count = len(summary.skipped_images)
    skipped_note = ""
    if skipped_count:
        examples = ", ".join(Path(path).name for path in summary.skipped_images[:3])
        extra = f" and {skipped_count - 3} more" if skipped_count > 3 else ""
        skipped_note = (
            f" Skipped {skipped_count} unreadable image(s): {examples}{extra}. "
            "Full paths and decoder errors are recorded in the run summary."
        )
    if summary.interruption_reason:
        status = "canceled" if "cancel" in summary.interruption_reason.lower() else "paused"
        message = (
            f"Prediction {status} after {summary.completed_images} of "
            f"{summary.image_count} images. {summary.interruption_reason}{skipped_note}"
        )
    elif request.execution != "local":
        provider_label = getattr(
            detector,
            "provider_label",
            {"modal": "Modal", "azure_ml": "Azure ML", "vertex_ai": "Vertex AI"}.get(
                request.execution, request.execution
            ),
        )
        message = (
            f"Processed {summary.completed_images} images on {detector.device_name}; "
            f"found {summary.detection_count} detections. Estimated {provider_label} spend: "
            f"${detector.estimated_cost_usd:.2f} of ${request.max_cost_usd:.2f}."
            f"{skipped_note}"
        )
    else:
        message = (
            f"Processed {summary.completed_images} images on "
            f"{describe_device_name(effective.device)} with batch size "
            f"{effective.batch_size}; found {summary.detection_count} detections."
            f"{skipped_note}"
        )
    return JobOutput(
        result={
            "message": message,
            "completed_images": summary.completed_images,
            "image_count": summary.image_count,
            "skipped_image_count": skipped_count,
            "cancelled": bool(
                summary.interruption_reason and "cancel" in summary.interruption_reason.lower()
            ),
            "effective_batch_size": json.loads(summary.summary_json.read_text()).get(
                "effective_batch_size", 0
            ),
            "estimated_cloud_cost_usd": (
                round(detector.estimated_cost_usd, 4) if request.execution != "local" else None
            ),
            "device": (
                detector.device_name
                if request.execution != "local"
                else describe_device_name(effective.device)
            ),
            "paths": {
                "csv": str(summary.predictions_csv),
                "run_summary": str(summary.summary_json),
                "report_markdown": str(markdown),
                "report_json": str(json_report),
            },
            "prediction": {
                "run_id": summary.run_id,
                "csv": str(summary.predictions_csv),
                "image_root": str(image_root_path),
                "project_root": str(store.root),
                "detection_count": summary.detection_count,
                "skipped_image_count": skipped_count,
                "run_name": run_name,
                "execution": request.execution,
                "gpu": request.gpu if request.execution != "local" else None,
                "compute_device": (
                    "cuda"
                    if str(effective.device).startswith("cuda")
                    else str(effective.device)
                    if effective.device in {"cpu", "mps"}
                    else "cpu"
                ),
            },
        },
        downloads=[
            DownloadArtifact(
                summary.predictions_csv, "Predictions CSV", "predictions.csv", "text/csv"
            ),
            DownloadArtifact(markdown, "Summary report", "report.md", "text/markdown"),
            DownloadArtifact(json_report, "Report data", "report.json", "application/json"),
        ],
    )
