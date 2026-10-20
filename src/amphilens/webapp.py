"""Loopback web API for the guided AmphiLens browser application."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.staticfiles import StaticFiles

from .core import AmphiLensError, ProjectConfig, ProjectManifest, ProjectStore
from .cvat_config import configured_cvat_server_url
from .doctor import run_doctor
from .locations import (
    ProjectLocationError,
    default_projects_root,
    find_source_checkout,
    open_project,
    validate_new_project_path,
)
from .models import ModelCatalog
from .models.backends import OptionalDependencyError
from .preprocessing import PreprocessingConfig, PreprocessingDependencyError
from .state import UserStateError, UserStateStore
from .webapp_cloud import _cloud_job_dict, _cloud_service, _create_cloud_action_impl
from .webapp_contracts import (
    ActiveLearningRequest,
    CloudCredentialsRequest,
    CloudEstimateRequest,
    CloudJobActionRequest,
    CloudTrainingRequest,
    CvatCycleRequest,
    CvatProjectDetailsRequest,
    CvatProjectImportRequest,
    CvatProjectListRequest,
    DatasetDeleteRequest,
    DatasetImportRequest,
    DatasetRenameRequest,
    PredictionsRequest,
    ProjectCreateRequest,
    ProjectOpenRequest,
    TimestampOCRRequest,
    TrainingRequest,
    _boolean,  # noqa: F401
    _class_names,
    _cloud_gpus,  # noqa: F401
    _ensure_prediction_run_name_is_unique,
    _integer,  # noqa: F401
    _mapping,  # noqa: F401
    _number,  # noqa: F401
    _optional_string,  # noqa: F401
    _prediction_run_name,  # noqa: F401
    _prediction_run_name_key,  # noqa: F401
    _preprocessing_override,  # noqa: F401
    _read_body,
    _request,
    _string,
)  # noqa: F401 - retain the established helper exports from this facade.
from .webapp_jobs import (
    _CREDENTIAL_PATTERN,  # noqa: F401
    _URL_CREDENTIAL_PATTERN,  # noqa: F401
    DownloadArtifact,
    JobManager,
    JobOutput,
    LocalPathPickerError,
    _Job,  # noqa: F401
    _native_path_picker,
    redact_sensitive_text,
    safe_error,
)
from .webapp_prediction import prediction_preflight, run_prediction_job
from .webapp_project import (
    _active_store,
    _cloud_estimate_image_size,
    _hosted_model_summaries,
    _modal_prediction_timing_samples,  # noqa: F401
    _model_summaries,
    _phase_progress,  # noqa: F401
    _project_snapshot,  # noqa: F401
    _project_summary,
    _read_queue_paths,  # noqa: F401
    _resolve_prediction_image_root,  # noqa: F401
    _training_checkpoint,  # noqa: F401
    _training_snapshots,
    select_training_source,
)  # noqa: F401 - callers and tests use these helpers through amphilens.webapp.
from .webapp_training import (
    _run_cloud_training_job,
    _run_training_job,
    run_active_learning_job,
    run_cvat_cycle_job,
    run_cvat_import_job,
    run_dataset_import_job,
)

_LOG = logging.getLogger(__name__)
_GENERIC_FAILURE = (
    "AmphiLens could not complete this request. Check your inputs and try again. "
    "If it continues, check the local terminal output."
)

__all__ = [
    "ActiveLearningRequest",
    "AmphiLensError",
    "CloudCredentialsRequest",
    "CloudEstimateRequest",
    "CloudJobActionRequest",
    "CloudTrainingRequest",
    "CvatCycleRequest",
    "CvatProjectDetailsRequest",
    "CvatProjectImportRequest",
    "CvatProjectListRequest",
    "DatasetDeleteRequest",
    "DatasetImportRequest",
    "DatasetRenameRequest",
    "DownloadArtifact",
    "HTTPException",
    "JobManager",
    "JobOutput",
    "LocalPathPickerError",
    "ModelCatalog",
    "OptionalDependencyError",
    "PredictionsRequest",
    "PreprocessingConfig",
    "PreprocessingDependencyError",
    "ProjectConfig",
    "ProjectCreateRequest",
    "ProjectLocationError",
    "ProjectManifest",
    "ProjectOpenRequest",
    "ProjectStore",
    "Request",
    "Response",
    "Starlette",
    "StaticFiles",
    "TimestampOCRRequest",
    "TrainingRequest",
    "UserStateError",
    "UserStateStore",
    "app",
    "asyncio",
    "configured_cvat_server_url",
    "create_app",
    "default_projects_root",
    "find_source_checkout",
    "open_project",
    "prediction_preflight",
    "run_active_learning_job",
    "run_cloud_training_job",
    "run_cvat_cycle_job",
    "run_cvat_import_job",
    "run_dataset_import_job",
    "run_doctor",
    "run_prediction_job",
    "run_training_job",
    "redact_sensitive_text",
    "safe_error",
    "select_training_source",
    "shutil",
    "subprocess",
    "sys",
    "validate_new_project_path",
]


def run_training_job(
    store: ProjectStore,
    request: TrainingRequest,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> JobOutput:
    return _run_training_job(
        store,
        request,
        progress_callback=progress_callback,
        training_snapshots=_training_snapshots,
    )


def run_cloud_training_job(
    store: ProjectStore,
    request: CloudTrainingRequest,
    credentials_store,
    provider_settings_store=None,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    return _run_cloud_training_job(
        store,
        request,
        credentials_store,
        provider_settings_store,
        progress_callback=progress_callback,
        cloud_service=_cloud_service,
        training_snapshots=_training_snapshots,
    )


def _create_cloud_action(
    store: ProjectStore, credentials_store, run_id: str, action: str, provider_settings_store=None
):
    return _create_cloud_action_impl(
        store,
        credentials_store,
        run_id,
        action,
        provider_settings_store,
        cloud_service=_cloud_service,
    )


def _json(value: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(value, status_code=status_code)


def _failure(error: BaseException, status_code: int | None = None) -> JSONResponse:
    if isinstance(error, HTTPException):
        status_code = error.status_code
        detail = error.detail if isinstance(error.detail, str) else _GENERIC_FAILURE
        if status_code >= 500:
            _LOG.error(
                "Request failed (HTTP %s): %s",
                status_code,
                redact_sensitive_text(detail)[:1000],
            )
            detail = _GENERIC_FAILURE
        else:
            detail = redact_sensitive_text(detail)[:240] or _GENERIC_FAILURE
    elif isinstance(
        error,
        (
            AmphiLensError,
            ValueError,
            FileNotFoundError,
            PermissionError,
            OptionalDependencyError,
            PreprocessingDependencyError,
        ),
    ):
        status_code = status_code or 400
        detail = safe_error(error)
    else:
        status_code = status_code or 500
        _LOG.error(
            "Request failed (%s): %s",
            type(error).__name__,
            redact_sensitive_text(str(error))[:1000],
        )
        detail = _GENERIC_FAILURE
    return _json({"detail": detail}, status_code)


def create_app(
    *,
    user_state_store: UserStateStore | None = None,
    cloud_credentials_store=None,
    cloud_provider_settings_store=None,
    jobs: JobManager | None = None,
    source_checkout: str | Path | None = None,
) -> Starlette:
    """Create a local app instance; the CLI launcher binds it to loopback only."""

    @asynccontextmanager
    async def lifespan(application):
        yield
        application.state.jobs.shutdown()

    async def handle_unexpected_exception(request: Request, error: Exception) -> Response:
        del request
        return _failure(error)

    app = Starlette(
        debug=False,
        lifespan=lifespan,
        exception_handlers={
            Exception: handle_unexpected_exception,
            HTTPException: handle_unexpected_exception,
        },
    )
    app.state.jobs = jobs or JobManager()
    app.state.user_state_store = user_state_store or UserStateStore()
    app.state.cloud_credentials_store = cloud_credentials_store
    if app.state.cloud_credentials_store is None:
        from .cloud.credentials import CloudCredentialsStore

        app.state.cloud_credentials_store = CloudCredentialsStore()
    app.state.cloud_provider_settings_store = cloud_provider_settings_store
    if app.state.cloud_provider_settings_store is None:
        from .cloud.provider_settings import CloudProviderSettingsStore

        app.state.cloud_provider_settings_store = CloudProviderSettingsStore()
    app.state.source_checkout = (
        Path(source_checkout).expanduser().resolve()
        if source_checkout is not None
        else find_source_checkout(Path(__file__))
    )
    app.state.active_project_path = None
    app.state.project_restored = False
    app.state.restore_warning = None
    app.state.project_lock = threading.RLock()

    def restore_project() -> None:
        with app.state.project_lock:
            if app.state.project_restored:
                return
            app.state.project_restored = True
            try:
                remembered = app.state.user_state_store.last_active_project()
            except UserStateError as exc:
                app.state.user_state_store.clear()
                app.state.restore_warning = safe_error(exc)
                return
            if remembered is None:
                return
            try:
                store = open_project(remembered)
            except ProjectLocationError as exc:
                app.state.user_state_store.clear()
                app.state.restore_warning = (
                    "The remembered project could not be opened and was forgotten: "
                    + safe_error(exc)
                )
                return
            app.state.active_project_path = store.root

    def active_store() -> ProjectStore:
        restore_project()
        with app.state.project_lock:
            return _active_store(app.state.active_project_path)

    def submit(
        operation: Callable[..., JobOutput | dict[str, Any]], *, with_progress: bool = False
    ) -> dict[str, str]:
        try:
            job_id = app.state.jobs.submit(operation, with_progress=with_progress)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=safe_error(exc)) from exc
        return {"job_id": job_id, "state": "queued"}

    async def path_picker(request: Request) -> Response:
        if request.client is None:
            return _json(
                {"detail": "File and folder selection is available only from this computer."},
                403,
            )
        try:
            client_ip = ipaddress.ip_address(request.client.host)
        except ValueError:
            client_ip = None
        if client_ip is None or not client_ip.is_loopback:
            return _json(
                {"detail": "File and folder selection is available only from this computer."},
                403,
            )
        try:
            body = await _read_body(request)
            kind = _string(body, "kind")
            if kind not in {"directory", "file", "save-file"}:
                raise ValueError("Choose whether to browse for a folder or a file")
            selected_path = await asyncio.to_thread(_native_path_picker, kind)
            return _json({"path": selected_path, "cancelled": selected_path is None})
        except Exception as exc:
            return _failure(exc)

    async def bootstrap(request: Request) -> Response:
        del request
        restore_project()
        try:
            with app.state.project_lock:
                project = (
                    _project_summary(open_project(app.state.active_project_path))
                    if app.state.active_project_path is not None
                    else None
                )
            report = run_doctor(".", check_cloud=False).to_dict()
            from .cloud.estimate import GPU_RATES_PER_SECOND

            payload = {
                "project": project,
                "doctor": report,
                "default_project_root": str(default_projects_root()),
                "models": _model_summaries(),
                "hosted_models": _hosted_model_summaries(),
                "default_model_id": ModelCatalog().default.model_id,
                "cvat_server_url": configured_cvat_server_url(),
                "modal_gpus": [
                    {"name": name, "usd_per_hour": round(rate * 3600, 3)}
                    for name, rate in GPU_RATES_PER_SECOND.items()
                ],
                "cloud_provider_settings": app.state.cloud_provider_settings_store.load(),
            }
            if app.state.restore_warning:
                payload["restore_warning"] = app.state.restore_warning
            return _json(payload)
        except Exception as exc:
            return _failure(exc)

    async def project_open(request: Request) -> Response:
        try:
            body = _request(ProjectOpenRequest, await _read_body(request))
            store = open_project(body.path)
            app.state.user_state_store.remember_project(store.root)
            with app.state.project_lock:
                app.state.active_project_path = store.root
                app.state.project_restored = True
                app.state.restore_warning = None
            return _json({"project": _project_summary(store)})
        except Exception as exc:
            return _failure(exc)

    async def project_create(request: Request) -> Response:
        try:
            body = _request(ProjectCreateRequest, await _read_body(request))
            classes = _class_names(body.classes)
            catalog = ModelCatalog()
            catalog.get(body.model_preset)
            preprocessing = PreprocessingConfig(
                short_side_dimension=body.image_dimension,
                grayscale_enabled=body.grayscale,
                clahe_enabled=body.clahe,
            )
            config = ProjectConfig(
                classes=classes,
                model_preset=body.model_preset,
                preprocessing=preprocessing,
            )
            manifest = ProjectManifest.create(
                body.name,
                [body.image_root] if body.image_root else [],
                classes,
                project_config=config,
            )
            destination = validate_new_project_path(
                body.path, source_checkout=app.state.source_checkout
            )
            store = ProjectStore(destination)
            store.create(manifest)
            app.state.user_state_store.remember_project(store.root)
            with app.state.project_lock:
                app.state.active_project_path = store.root
                app.state.project_restored = True
                app.state.restore_warning = None
            return _json({"project": _project_summary(store)})
        except Exception as exc:
            return _failure(exc)

    async def project_close(request: Request) -> Response:
        del request
        try:
            app.state.user_state_store.clear()
            with app.state.project_lock:
                app.state.active_project_path = None
                app.state.project_restored = True
                app.state.restore_warning = None
            return _json({"project": None})
        except Exception as exc:
            return _failure(exc)

    async def predictions(request: Request) -> Response:
        try:
            body = _request(PredictionsRequest, await _read_body(request))
            store = active_store()
            _ensure_prediction_run_name_is_unique(store, body.run_name)
            if body.execution != "local":
                if not body.acknowledged or not body.uploads_dataset:
                    raise ValueError(
                        "Explicit consent is required. Review and approve the cloud image "
                        "transfer and spending limit first"
                    )
                if (
                    body.expected_image_count is None
                    or body.expected_total_bytes is None
                    or body.expected_rate_usd_per_hour is None
                ):
                    raise ValueError(
                        "Review the image count, transfer size, and GPU price "
                        "before cloud prediction"
                    )
                preview = prediction_preflight(
                    store,
                    body,
                    credentials_store=app.state.cloud_credentials_store,
                    provider_settings_store=app.state.cloud_provider_settings_store,
                    check_connection=True,
                )
                if (
                    preview["image_count"] != body.expected_image_count
                    or preview["total_bytes"] != body.expected_total_bytes
                    or abs(preview["rate_usd_per_hour"] - body.expected_rate_usd_per_hour)
                    > max(0.01, body.expected_rate_usd_per_hour * 0.01)
                ):
                    raise ValueError(
                        "The image selection or GPU price changed; review the updated preflight"
                    )
                service, credentials = _cloud_service(
                    store,
                    app.state.cloud_credentials_store,
                    app.state.cloud_provider_settings_store,
                    provider=body.execution,
                )
                if body.execution == "modal" and credentials is None:
                    raise ValueError(
                        "Modal credentials are not configured; save them in System health"
                    )
                transport = service.transport
                return _json(
                    submit(
                        lambda report: run_prediction_job(
                            store,
                            body,
                            progress_callback=report,
                            cloud_transport=transport,
                        ),
                        with_progress=True,
                    )
                )
            return _json(
                submit(
                    lambda report: run_prediction_job(
                        store,
                        body,
                        progress_callback=report,
                    ),
                    with_progress=True,
                )
            )
        except Exception as exc:
            return _failure(exc)

    async def predictions_preflight(request: Request) -> Response:
        try:
            body = _request(PredictionsRequest, await _read_body(request))
            store = active_store()
            result = prediction_preflight(
                store,
                body,
                credentials_store=app.state.cloud_credentials_store,
                provider_settings_store=app.state.cloud_provider_settings_store,
                check_connection=True,
            )
            return _json(result)
        except Exception as exc:
            return _failure(exc)

    async def dataset_import(request: Request) -> Response:
        try:
            body = _request(DatasetImportRequest, await _read_body(request))
            store = active_store()
            return _json(
                submit(
                    lambda progress: run_dataset_import_job(store, body, progress),
                    with_progress=True,
                )
            )
        except Exception as exc:
            return _failure(exc)

    async def dataset_import_cvat(request: Request) -> Response:
        try:
            body = _request(CvatProjectImportRequest, await _read_body(request))
            store = active_store()
            return _json(
                submit(
                    lambda progress: run_cvat_import_job(store, body, progress),
                    with_progress=True,
                )
            )
        except Exception as exc:
            return _failure(exc)

    async def dataset_rename(request: Request) -> Response:
        try:
            body = _request(DatasetRenameRequest, await _read_body(request))
            display_name = active_store().set_dataset_display_name(
                body.snapshot_path, body.display_name
            )
            return _json({"display_name": display_name})
        except Exception as exc:
            return _failure(exc)

    async def dataset_delete(request: Request) -> Response:
        try:
            body = _request(DatasetDeleteRequest, await _read_body(request))
            result = active_store().delete_dataset_snapshot(body.snapshot_path)
            return _json(result)
        except Exception as exc:
            return _failure(exc)

    async def cvat_projects(request: Request) -> Response:
        try:
            body = _request(CvatProjectListRequest, await _read_body(request))
            store = active_store()
            from .annotations.initial import CVATProjectImportService
            from .annotations.managed import CVATSdkTransport

            transport = CVATSdkTransport(server_url=body.server_url)
            projects = CVATProjectImportService(store, transport).list_projects()
            return _json({"projects": [item.to_dict() for item in projects]})
        except Exception as exc:
            return _failure(exc)

    async def cvat_project(request: Request) -> Response:
        try:
            body = _request(CvatProjectDetailsRequest, await _read_body(request))
            from .annotations.managed import CVATSdkTransport

            transport = CVATSdkTransport(server_url=body.server_url)
            project = transport.get_project(body.project_id)
            return _json({"project": project.to_dict()})
        except Exception as exc:
            return _failure(exc)

    async def training(request: Request) -> Response:
        try:
            body = _request(TrainingRequest, await _read_body(request))
            store = active_store()
            return _json(
                submit(
                    lambda report: run_training_job(
                        store,
                        body,
                        progress_callback=report,
                    ),
                    with_progress=True,
                )
            )
        except Exception as exc:
            return _failure(exc)

    async def active_learning(request: Request) -> Response:
        try:
            body = _request(ActiveLearningRequest, await _read_body(request))
            store = active_store()
            return _json(
                submit(
                    lambda report: run_active_learning_job(
                        store,
                        body,
                        progress_callback=report,
                    ),
                    with_progress=True,
                )
            )
        except Exception as exc:
            return _failure(exc)

    async def cvat_cycle(request: Request) -> Response:
        try:
            body = _request(CvatCycleRequest, await _read_body(request))
            store = active_store()
            return _json(submit(lambda: run_cvat_cycle_job(store, body)))
        except Exception as exc:
            return _failure(exc)

    async def job_status(request: Request) -> Response:
        job = app.state.jobs.snapshot(request.path_params["job_id"])
        if job is None:
            return _json({"detail": "Workflow job was not found"}, 404)
        return _json(job)

    async def job_cancel(request: Request) -> Response:
        job_id = request.path_params["job_id"]
        if not app.state.jobs.cancel(job_id):
            current = app.state.jobs.snapshot(job_id)
            if current is None:
                return _json({"detail": "Workflow job was not found"}, 404)
            return _json({"detail": "Workflow job is already finished"}, 409)
        return _json({"job_id": job_id, "cancel_requested": True})

    async def job_collect_images(request: Request) -> Response:
        try:
            job = app.state.jobs.snapshot(request.path_params["job_id"])
            if job is None:
                return _json({"detail": "Workflow job was not found"}, 404)
            if job["state"] != "completed":
                return _json(
                    {"detail": "Images can only be collected from a completed prediction"}, 409
                )
            result = job.get("result") or {}
            prediction = result.get("prediction")
            if not isinstance(prediction, dict):
                raise ValueError("This job does not contain prediction results")
            try:
                detection_count = int(prediction.get("detection_count", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError("Prediction detection count is invalid") from exc
            if detection_count <= 0:
                raise ValueError("There are no detected images to collect")

            from .core import ProjectStore, _validate_run_id
            from .inference import collect_prediction_images

            project_root = prediction.get("project_root")
            run_id = prediction.get("run_id")
            run_name = prediction.get("run_name")
            if not isinstance(project_root, str) or not isinstance(run_id, str) or not run_id:
                raise ValueError("Prediction project or run information is missing")
            if not isinstance(run_name, str) or not run_name.strip():
                raise ValueError("Prediction run name is missing")
            _validate_run_id(run_id)
            run_folder_name = _prediction_run_name(run_name)
            store = ProjectStore(project_root)
            store.load_manifest()
            destination = store.root / "runs" / run_folder_name / "images"
            if not destination.is_relative_to(store.root):
                raise ValueError("Prediction run folder is outside the project")
            summary = collect_prediction_images(
                prediction.get("csv", ""),
                prediction.get("image_root", ""),
                destination,
            )
            return _json(
                {
                    "folder": str(summary.destination),
                    "copied_count": summary.copied_count,
                    "already_present_count": summary.already_present_count,
                    "missing_count": summary.missing_count,
                }
            )
        except Exception as exc:
            return _failure(exc)

    async def job_ocr_timestamps(request: Request) -> Response:
        try:
            body = _request(TimestampOCRRequest, await _read_body(request))
            job = app.state.jobs.snapshot(request.path_params["job_id"])
            if job is None:
                return _json({"detail": "Workflow job was not found"}, 404)
            if job["state"] != "completed":
                return _json(
                    {"detail": "Timestamps can only be extracted from a completed prediction"},
                    409,
                )
            result = job.get("result") or {}
            prediction = result.get("prediction")
            if not isinstance(prediction, dict):
                raise ValueError("This job does not contain prediction results")
            try:
                detection_count = int(prediction.get("detection_count", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError("Prediction detection count is invalid") from exc
            if detection_count <= 0:
                raise ValueError("There are no detections to process")
            execution = prediction.get("execution", "local")
            if execution not in {"local", "modal", "azure_ml", "vertex_ai"}:
                raise ValueError("Prediction cloud provider information is invalid")
            if execution != "local" and not body.acknowledged:
                raise ValueError(
                    "Confirm cloud timestamp OCR before sending timestamp crops to the provider"
                )

            from .core import RunManifest, _validate_run_id, read_json

            project_root = prediction.get("project_root")
            run_id = prediction.get("run_id")
            csv_value = prediction.get("csv")
            image_root_value = prediction.get("image_root")
            if not all(isinstance(value, str) and value for value in (project_root, run_id)):
                raise ValueError("Prediction project or run information is missing")
            if not isinstance(csv_value, str) or not csv_value:
                raise ValueError("Prediction CSV path is missing")
            if not isinstance(image_root_value, str) or not image_root_value:
                raise ValueError("Prediction image folder is missing")

            _validate_run_id(run_id)
            store = ProjectStore(project_root)
            store.load_manifest()
            csv_path = Path(csv_value).expanduser().resolve()
            image_root = Path(image_root_value).expanduser().resolve()
            if csv_path.name != "predictions.csv" or not csv_path.is_file():
                raise ValueError("Prediction CSV is missing or invalid")
            if not image_root.is_dir():
                raise ValueError("Prediction image folder was not found")
            run_manifest_path = csv_path.parent / "run.json"
            if not run_manifest_path.is_file():
                raise ValueError("Prediction run metadata was not found")
            run_manifest = RunManifest.from_dict(read_json(run_manifest_path))
            if (
                run_manifest.kind != "inference"
                or run_manifest.run_id != run_id
                or run_manifest.status
                not in {"completed", "completed_with_failures", "completed_with_skips"}
            ):
                raise ValueError("Prediction run is not complete or does not match this job")

            provider_name = {
                "modal": "Modal",
                "azure_ml": "Azure ML",
                "vertex_ai": "Vertex AI",
            }.get(execution, "")
            gpu = prediction.get("gpu")
            if execution != "local" and (not isinstance(gpu, str) or not gpu.strip()):
                raise ValueError("Prediction GPU information is missing")

            def run_ocr(progress_callback):
                if execution == "local":
                    from .timestamp_ocr import process_prediction_csv

                    raw_device = str(prediction.get("compute_device", "cpu"))
                    device = (
                        "cuda"
                        if raw_device.startswith("cuda")
                        else raw_device
                        if raw_device in {"cpu", "mps"}
                        else "cpu"
                    )
                    summary = process_prediction_csv(
                        csv_path,
                        image_root,
                        device=device,
                        progress_callback=progress_callback,
                    )
                else:
                    service, credentials = _cloud_service(
                        store,
                        app.state.cloud_credentials_store,
                        app.state.cloud_provider_settings_store,
                        execution,
                    )
                    if execution == "modal" and credentials is None:
                        raise ValueError(
                            "Modal credentials are not configured. Save credentials or set both "
                            "Modal environment variables."
                        )
                    from .cloud.timestamp_ocr import run_cloud_timestamp_ocr

                    summary = run_cloud_timestamp_ocr(
                        csv_path,
                        image_root,
                        transport=service.transport,
                        gpu=gpu,
                        progress_callback=progress_callback,
                        cancellation_requested=getattr(progress_callback, "cancel_requested", None),
                    )
                return JobOutput(
                    result={
                        "message": (
                            f"Read timestamps for {summary.recognized_count} of "
                            f"{summary.image_count} unique detected images; "
                            f"{summary.unparsed_count} were unresolved."
                            + (
                                f" Only local bottom crops were sent to {provider_name} "
                                f"({gpu}); additional provider charges may apply."
                                if execution != "local"
                                else ""
                            )
                        ),
                        "prediction_csv": str(summary.csv_path),
                        "image_count": summary.image_count,
                        "recognized_count": summary.recognized_count,
                        "unparsed_count": summary.unparsed_count,
                    },
                    downloads=[
                        DownloadArtifact(
                            summary.csv_path,
                            "Predictions CSV with timestamps",
                            "predictions.csv",
                            "text/csv",
                        )
                    ],
                )

            job_id = app.state.jobs.submit(run_ocr, with_progress=True)
            return _json({"job_id": job_id, "message": "Timestamp OCR started."})
        except Exception as exc:
            return _failure(exc)

    async def job_download(request: Request) -> Response:
        artifact = app.state.jobs.download(
            request.path_params["job_id"], request.path_params["download_id"]
        )
        if artifact is None or not artifact.path.is_file():
            return _json({"detail": "Download is not available for this job"}, 404)
        return FileResponse(
            artifact.path,
            media_type=artifact.media_type,
            filename=artifact.filename,
        )

    async def environment_check_cloud(request: Request) -> Response:
        del request
        try:
            return _json({"doctor": run_doctor(".", check_cloud=True).to_dict()})
        except Exception as exc:
            return _failure(exc)

    async def cloud_credentials(request: Request) -> Response:
        try:
            body = _request(CloudCredentialsRequest, await _read_body(request))
            credentials_store = app.state.cloud_credentials_store
            credentials_store.save(body.token_id, body.token_secret)
            credentials = credentials_store.resolve()
            description = credentials.describe() if credentials else {}
            return _json(
                {
                    "credentials": {
                        "configured": credentials is not None,
                        "token_id": description.get("token_id", ""),
                        "source": description.get("source", ""),
                    }
                }
            )
        except Exception as exc:
            return _failure(exc)

    async def cloud_provider_settings(request: Request) -> Response:
        settings_store = app.state.cloud_provider_settings_store
        try:
            if request.method == "GET":
                return _json({"providers": settings_store.load()})
            payload = await _read_body(request)
            provider = _string(payload, "provider")
            settings = payload.get("settings")
            if not isinstance(settings, dict):
                raise ValueError("settings must be an object")
            from .cloud.provider_settings import AzureMLSettings, VertexAISettings

            settings_type = {"azure_ml": AzureMLSettings, "vertex_ai": VertexAISettings}.get(
                provider
            )
            if settings_type is None:
                raise ValueError("provider must be azure_ml or vertex_ai")
            normalized = settings_type.from_mapping(settings).to_mapping()
            settings_store.update(provider, normalized)
            return _json({"provider": provider, "settings": normalized})
        except Exception as exc:
            return _failure(exc)

    async def cloud_provider_check(request: Request) -> Response:
        try:
            payload = await _read_body(request)
            provider = _string(payload, "provider")
            service, _ = _cloud_service(
                active_store(),
                app.state.cloud_credentials_store,
                app.state.cloud_provider_settings_store,
                provider=provider,
            )
            transport = service.transport
            status = transport.probe()
            return _json({"connection": status})
        except Exception as exc:
            return _failure(exc)

    async def cloud_estimate(request: Request) -> Response:
        try:
            body = _request(CloudEstimateRequest, await _read_body(request))
            store = active_store()
            snapshot, validation_snapshot, test_snapshot = _training_snapshots(store, body)
            service, _ = _cloud_service(
                store,
                app.state.cloud_credentials_store,
                app.state.cloud_provider_settings_store,
                provider=body.provider,
            )
            project_config = store.load_manifest().project_config
            estimate = service.estimate(
                snapshot.root,
                gpu=body.gpu,
                epochs=(body.epochs or project_config.epochs) * 2,
                max_cost_usd=body.max_cost_usd,
                image_size=_cloud_estimate_image_size(body),
                preprocessing=body.preprocessing,
                validation_snapshot_path=(
                    validation_snapshot.root if validation_snapshot else None
                ),
                test_snapshot_path=test_snapshot.root if test_snapshot else None,
                data_mode=body.data_mode,
                seed=project_config.random_seed,
                allow_image_level_fallback=body.allow_image_level_split,
            )
            return _json({"estimate": estimate.to_dict()})
        except Exception as exc:
            return _failure(exc)

    async def cloud_training(request: Request) -> Response:
        try:
            body = _request(CloudTrainingRequest, await _read_body(request))
            if not body.acknowledged or not body.uploads_dataset:
                raise ValueError(
                    "Confirm both cloud training and dataset upload consent before submitting."
                )
            store = active_store()
            return _json(
                submit(
                    lambda report: run_cloud_training_job(
                        store,
                        body,
                        app.state.cloud_credentials_store,
                        app.state.cloud_provider_settings_store,
                        progress_callback=report,
                    ),
                    with_progress=True,
                )
            )
        except Exception as exc:
            return _failure(exc)

    async def cloud_jobs(request: Request) -> Response:
        del request
        try:
            store = active_store()
            service, _ = _cloud_service(
                store,
                app.state.cloud_credentials_store,
                app.state.cloud_provider_settings_store,
            )
            return _json({"jobs": [_cloud_job_dict(job) for job in service.list_jobs()]})
        except Exception as exc:
            return _failure(exc)

    async def cloud_job_action(request: Request) -> Response:
        try:
            run_id = request.path_params["run_id"]
            action = request.path_params["action"]
            if action not in {"refresh", "cancel", "cleanup"}:
                return _json({"detail": "Unsupported cloud job action"}, 404)
            store = active_store()
            payload = _create_cloud_action(
                store,
                app.state.cloud_credentials_store,
                run_id,
                action,
                app.state.cloud_provider_settings_store,
            )
            response: dict[str, Any] = {"job": payload}
            return _json(response)
        except Exception as exc:
            return _failure(exc)

    app.add_route("/api/path-picker", path_picker, methods=["POST"])
    app.add_route("/api/bootstrap", bootstrap, methods=["GET"])
    app.add_route("/api/projects/open", project_open, methods=["POST"])
    app.add_route("/api/projects/create", project_create, methods=["POST"])
    app.add_route("/api/projects/close", project_close, methods=["POST"])
    app.add_route("/api/predictions", predictions, methods=["POST"])
    app.add_route("/api/predictions/preflight", predictions_preflight, methods=["POST"])
    app.add_route("/api/datasets/import", dataset_import, methods=["POST"])
    app.add_route("/api/datasets/import-cvat", dataset_import_cvat, methods=["POST"])
    app.add_route("/api/datasets/rename", dataset_rename, methods=["POST"])
    app.add_route("/api/datasets/delete", dataset_delete, methods=["POST"])
    app.add_route("/api/cvat/projects", cvat_projects, methods=["POST"])
    app.add_route("/api/cvat/project", cvat_project, methods=["POST"])
    app.add_route("/api/training", training, methods=["POST"])
    app.add_route("/api/active-learning/select", active_learning, methods=["POST"])
    app.add_route("/api/cvat/cycle", cvat_cycle, methods=["POST"])
    app.add_route("/api/jobs/{job_id:str}", job_status, methods=["GET"])
    app.add_route("/api/jobs/{job_id:str}/cancel", job_cancel, methods=["POST"])
    app.add_route("/api/jobs/{job_id:str}/collect-images", job_collect_images, methods=["POST"])
    app.add_route("/api/jobs/{job_id:str}/ocr-timestamps", job_ocr_timestamps, methods=["POST"])
    app.add_route(
        "/api/jobs/{job_id:str}/downloads/{download_id:str}", job_download, methods=["GET"]
    )
    app.add_route("/api/environment/check-cloud", environment_check_cloud, methods=["POST"])
    app.add_route("/api/cloud/credentials", cloud_credentials, methods=["POST"])
    app.add_route(
        "/api/cloud/providers/settings",
        cloud_provider_settings,
        methods=["GET", "POST"],
    )
    app.add_route("/api/cloud/providers/check", cloud_provider_check, methods=["POST"])
    app.add_route("/api/cloud/estimate", cloud_estimate, methods=["POST"])
    app.add_route("/api/cloud/training", cloud_training, methods=["POST"])
    app.add_route("/api/cloud/jobs", cloud_jobs, methods=["GET"])
    app.add_route("/api/cloud/jobs/{run_id:str}/{action:str}", cloud_job_action, methods=["POST"])
    web_root = Path(__file__).with_name("web")
    app.mount("/static", StaticFiles(directory=web_root), name="assets")
    app.mount("/", StaticFiles(directory=web_root, html=True), name="static")
    return app


app = create_app()
