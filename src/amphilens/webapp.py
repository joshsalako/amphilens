"""Loopback web API for the guided AmphiLens browser application."""

from __future__ import annotations

import asyncio
import csv
import ipaddress
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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

_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(authorization\s*:\s*bearer\s+|[a-z0-9_.-]*(?:token|secret|password|api[_-]?key)"
    r"[a-z0-9_.-]*\s*[:=]\s*)([^\s,;]+)"
)
_URL_CREDENTIAL_PATTERN = re.compile(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@")
_LOG = logging.getLogger(__name__)
_GENERIC_FAILURE = (
    "AmphiLens could not complete this request. Check your inputs and try again. "
    "If it continues, check the local terminal output."
)


class LocalPathPickerError(AmphiLensError):
    """Raised when the computer cannot open its native file chooser."""


def _native_path_picker(kind: str) -> str | None:
    """Open a native local dialog without exposing arbitrary command execution."""
    if kind not in {"directory", "file", "save-file"}:
        raise ValueError("Choose whether to browse for a folder or a file")

    if sys.platform == "darwin":
        scripts = {
            "directory": (
                'POSIX path of (choose folder with prompt "Choose a folder for AmphiLens")'
            ),
            "file": ('POSIX path of (choose file with prompt "Choose a file for AmphiLens")'),
            "save-file": (
                'POSIX path of (choose file name with prompt "Save a file from AmphiLens" '
                'default name "selection_queue.csv")'
            ),
        }
        command = ["osascript", "-e", scripts[kind]]
    elif sys.platform == "win32":
        powershell = (
            shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh.exe")
        )
        if not powershell:
            raise LocalPathPickerError(
                "The Windows file picker is unavailable. Paste the path into the field instead."
            )
        dialogs = {
            "directory": """$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Choose a folder for AmphiLens'
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
  [Console]::Out.Write($dialog.SelectedPath)
}""",
            "file": """$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Choose a file for AmphiLens'
$dialog.Filter = 'All files (*.*)|*.*'
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
  [Console]::Out.Write($dialog.FileName)
}""",
            "save-file": """$dialog = New-Object System.Windows.Forms.SaveFileDialog
$dialog.Title = 'Save a file from AmphiLens'
$dialog.FileName = 'selection_queue.csv'
$dialog.Filter = 'CSV files (*.csv)|*.csv|All files (*.*)|*.*'
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
  [Console]::Out.Write($dialog.FileName)
}""",
        }
        command = [
            powershell,
            "-NoProfile",
            "-STA",
            "-Command",
            "Add-Type -AssemblyName System.Windows.Forms; " + dialogs[kind],
        ]
    else:
        zenity = shutil.which("zenity")
        kdialog = shutil.which("kdialog")
        if zenity:
            command = [zenity, "--file-selection", "--title=AmphiLens"]
            if kind == "directory":
                command.append("--directory")
            elif kind == "save-file":
                command.extend(["--save", "--confirm-overwrite", "--filename=selection_queue.csv"])
        elif kdialog:
            dialog_option = {
                "directory": "--getexistingdirectory",
                "file": "--getopenfilename",
                "save-file": "--getsavefilename",
            }[kind]
            command = [kdialog, dialog_option, ".", "--title", "AmphiLens"]
            if kind == "save-file":
                command[2] = "selection_queue.csv"
        else:
            raise LocalPathPickerError(
                "No native file picker is available here. Paste a path, or install "
                "zenity or kdialog."
            )

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LocalPathPickerError(
            "The file picker could not open. Paste the path into the field instead."
        ) from exc

    if result.returncode != 0:
        if "-128" in result.stderr or "User canceled" in result.stderr:
            return None
        raise LocalPathPickerError(
            "The file picker could not open. Paste the path into the field instead."
        )
    selected = result.stdout.strip()
    return os.path.normpath(selected) if selected else None


def safe_error(error: BaseException) -> str:
    """Map failures to concise user guidance; unexpected exception text stays server-side."""
    try:
        raw_message = str(error).strip().splitlines()[0]
    except Exception:
        raw_message = ""
    message = redact_sensitive_text(raw_message)[:240]
    lowered = message.lower()

    if isinstance(error, FileNotFoundError):
        if "calibration" in lowered:
            return (
                "The selected calibration JSON was not found. Choose an existing calibration file."
            )
        if "feature" in lowered:
            return "The selected feature JSON was not found. Choose an existing feature file."
        if "prediction" in lowered:
            return (
                "The selected predictions CSV was not found. Choose an existing predictions file."
            )
        if "checkpoint" in lowered:
            return (
                "The selected model checkpoint was not found. Choose an existing checkpoint file."
            )
        if "archive" in lowered or "dataset" in lowered:
            return "The selected archive was not found. Choose an existing archive file."
        return "The selected file or folder was not found. Check the path and try again."
    if isinstance(error, PermissionError):
        return "AmphiLens cannot access that location. Check its permissions and try again."
    if isinstance(error, ProjectLocationError):
        if "inside the AmphiLens source checkout" in message:
            return "Choose a project folder that is not inside the AmphiLens source checkout."
        if "not empty" in lowered:
            return "Choose an empty folder for the new project."
        if "not a folder" in lowered:
            return "The selected project location is not a folder. Choose a folder and try again."
        return (
            "That folder could not be used as an AmphiLens project. Check the folder and try again."
        )
    if isinstance(
        error,
        (AmphiLensError, ValueError, OptionalDependencyError, PreprocessingDependencyError),
    ):
        return message or _GENERIC_FAILURE
    if isinstance(error, RuntimeError) and message.startswith(
        (
            "Too many workflows are already queued.",
            "The local workflow service is shutting down.",
            "The workflow history is full.",
        )
    ):
        return message
    return _GENERIC_FAILURE


def redact_sensitive_text(message: str) -> str:
    """Redact known local secrets and credential-shaped text from logs or errors."""
    for key in ("CVAT_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        value = os.environ.get(key)
        if value:
            message = message.replace(value, "[redacted]")
    message = _URL_CREDENTIAL_PATTERN.sub(r"\1[redacted]@", message)
    return _CREDENTIAL_PATTERN.sub(r"\1[redacted]", message)


@dataclass(frozen=True, slots=True)
class DownloadArtifact:
    path: Path
    label: str
    filename: str
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        if Path(self.filename).name != self.filename or self.filename in {"", ".", ".."}:
            raise ValueError("Download filename must be a single safe path component")


@dataclass(slots=True)
class JobOutput:
    result: dict[str, Any]
    downloads: list[DownloadArtifact] = field(default_factory=list)


@dataclass(slots=True)
class _Job:
    job_id: str
    state: str
    created_at: float
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: dict[str, Any] | None = None
    progress_events: list[dict[str, Any]] = field(default_factory=list)
    downloads: dict[str, DownloadArtifact] = field(default_factory=dict)
    future: Future | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobManager:
    """Run a bounded number of local workflows and retain a bounded status history."""

    MAX_PROGRESS_EVENTS = 100
    MAX_PROGRESS_LOG_CHARS = 8000

    def __init__(self, *, max_workers: int = 2, max_pending: int = 6, max_history: int = 100):
        if max_workers <= 0 or max_pending < max_workers or max_history <= 0:
            raise ValueError("Job limits must be positive and pending capacity must cover workers")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="amphilens")
        self._capacity = threading.BoundedSemaphore(max_pending)
        self._max_history = max_history
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()
        self._closed = False

    def submit(
        self,
        operation: Callable[..., JobOutput | dict[str, Any]],
        *,
        with_progress: bool = False,
    ) -> str:
        if not self._capacity.acquire(blocking=False):
            raise RuntimeError(
                "Too many workflows are already queued. Try again when one finishes."
            )
        with self._lock:
            if self._closed:
                self._capacity.release()
                raise RuntimeError("The local workflow service is shutting down.")
            self._prune()
            if len(self._jobs) >= self._max_history:
                self._capacity.release()
                raise RuntimeError("The workflow history is full. Restart AmphiLens to clear it.")
            job_id = uuid.uuid4().hex
            job = _Job(job_id, "queued", time.time())
            self._jobs[job_id] = job
            job.future = self._executor.submit(self._run, job_id, operation, with_progress)
        return job_id

    def _run(
        self,
        job_id: str,
        operation: Callable[..., JobOutput | dict[str, Any]],
        with_progress: bool,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.state = "running"
        try:
            if job.cancel_event.is_set():
                with self._lock:
                    job.state = "canceled"
                    job.error = "Canceled before the workflow started."
                return

            def report_progress(value: dict[str, Any]) -> None:
                self.update_progress(job_id, value)

            report_progress.cancel_requested = job.cancel_event.is_set

            output = operation(report_progress) if with_progress else operation()
            if isinstance(output, JobOutput):
                result = dict(output.result)
                download_specs = list(output.downloads)
            else:
                result = dict(output)
                download_specs = []
            download_map: dict[str, DownloadArtifact] = {}
            links = []
            for index, artifact in enumerate(download_specs, start=1):
                if not artifact.path.is_file():
                    continue
                download_id = f"{index}-{artifact.filename}"
                download_map[download_id] = artifact
                links.append(
                    {
                        "label": artifact.label,
                        "url": f"/api/jobs/{job_id}/downloads/{download_id}",
                        "filename": artifact.filename,
                    }
                )
            result["downloads"] = links
            with self._lock:
                job = self._jobs[job_id]
                job.result = result
                job.downloads = download_map
                job.state = "canceled" if result.get("cancelled") else "completed"
        except Exception as exc:  # keep workflow failures inside the job status contract
            _LOG.error(
                "Local job failed (%s): %s",
                type(exc).__name__,
                redact_sensitive_text(str(exc))[:1000],
            )
            with self._lock:
                job = self._jobs[job_id]
                job.error = safe_error(exc)
                job.state = "canceled" if job.cancel_event.is_set() else "failed"
        finally:
            self._capacity.release()

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            payload: dict[str, Any] = {
                "job_id": job.job_id,
                "state": job.state,
                "cancel_requested": job.cancel_event.is_set(),
            }
            if job.result is not None:
                payload["result"] = dict(job.result)
            if job.error is not None:
                payload["error"] = job.error
            if job.progress is not None:
                payload["progress"] = dict(job.progress)
            if job.progress_events:
                payload["progress_events"] = [dict(event) for event in job.progress_events]
            return payload

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in {"completed", "failed", "canceled"}:
                return False
            job.cancel_event.set()
            return True

    def update_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                safe_progress = dict(progress)
                for key in ("message", "phase", "error"):
                    value = safe_progress.get(key)
                    if isinstance(value, str):
                        safe_progress[key] = redact_sensitive_text(value)[:1000]
                log_tail = safe_progress.get("log_tail")
                if isinstance(log_tail, str):
                    safe_progress["log_tail"] = redact_sensitive_text(log_tail)[
                        -self.MAX_PROGRESS_LOG_CHARS :
                    ]
                for key in ("completed", "failed", "total", "remaining", "epoch", "epochs"):
                    value = safe_progress.get(key)
                    if value is not None:
                        try:
                            safe_progress[key] = max(0, int(value))
                        except (TypeError, ValueError):
                            safe_progress.pop(key, None)
                for key in ("progress", "phase_progress"):
                    progress_value = safe_progress.get(key)
                    if isinstance(progress_value, (int, float)) and math.isfinite(progress_value):
                        safe_progress[key] = max(0.0, min(1.0, float(progress_value)))
                eta_seconds = safe_progress.get("eta_seconds")
                if eta_seconds is not None:
                    try:
                        safe_progress["eta_seconds"] = max(0, min(31_536_000, int(eta_seconds)))
                    except (TypeError, ValueError):
                        safe_progress.pop("eta_seconds", None)
                metrics = safe_progress.get("metrics")
                if isinstance(metrics, dict):
                    safe_progress["metrics"] = {
                        str(key)[:80]: (
                            redact_sensitive_text(value)[:240] if isinstance(value, str) else value
                        )
                        for key, value in list(metrics.items())[:24]
                        if isinstance(value, (str, int, bool))
                        or (isinstance(value, float) and math.isfinite(value))
                    }
                job.progress = safe_progress
                event = {key: value for key, value in safe_progress.items() if key != "log_tail"}
                event["updated_at"] = time.time()
                job.progress_events.append(event)
                del job.progress_events[: -self.MAX_PROGRESS_EVENTS]

    def register_result(self, output: JobOutput) -> dict[str, Any]:
        """Expose a completed synchronous action using the same safe download contract."""
        job_id = uuid.uuid4().hex
        download_map: dict[str, DownloadArtifact] = {}
        links = []
        for index, artifact in enumerate(output.downloads, start=1):
            if not artifact.path.is_file():
                continue
            download_id = f"{index}-{artifact.filename}"
            download_map[download_id] = artifact
            links.append(
                {
                    "label": artifact.label,
                    "url": f"/api/jobs/{job_id}/downloads/{download_id}",
                    "filename": artifact.filename,
                }
            )
        result = dict(output.result)
        result["downloads"] = links
        with self._lock:
            self._prune()
            if len(self._jobs) >= self._max_history:
                raise RuntimeError("The workflow history is full. Restart AmphiLens to clear it.")
            self._jobs[job_id] = _Job(
                job_id,
                "completed",
                time.time(),
                result=result,
                downloads=download_map,
            )
        return self.snapshot(job_id) or {"job_id": job_id, "state": "completed"}

    def download(self, job_id: str, download_id: str) -> DownloadArtifact | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.downloads.get(download_id) if job else None

    def _prune(self) -> None:
        terminal = sorted(
            (
                job
                for job in self._jobs.values()
                if job.state in {"completed", "failed", "canceled"}
            ),
            key=lambda job: job.created_at,
        )
        while len(self._jobs) >= self._max_history and terminal:
            self._jobs.pop(terminal.pop(0).job_id, None)

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)


@dataclass(slots=True)
class ProjectOpenRequest:
    path: str


@dataclass(slots=True)
class ProjectCreateRequest:
    name: str
    path: str
    image_root: str
    classes: list[str]
    model_preset: str = "yolo26-l"
    image_dimension: int = 640
    grayscale: bool = True
    clahe: bool = False


@dataclass(slots=True)
class PredictionsRequest:
    image_root: str | None = None
    checkpoint: str | None = None
    output_dir: str | None = None
    confidence: float | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] | None = None
    execution: Literal["local", "modal"] = "local"
    batch_size: int | None = None
    gpu: str = "L4"
    max_cost_usd: float = 5.0
    acknowledged: bool = False
    uploads_dataset: bool = False
    expected_image_count: int | None = None
    expected_total_bytes: int | None = None
    model_preset: str | None = None
    hosted_model_id: str | None = None
    class_mapping: dict[str, str | None] | None = None
    run_name: str = ""


@dataclass(slots=True)
class DatasetImportRequest:
    archive: str
    class_mapping: dict[str, str] | None = None
    display_name: str | None = None


@dataclass(slots=True)
class CvatProjectListRequest:
    server_url: str | None = None


@dataclass(slots=True)
class CvatProjectImportRequest:
    project_id: str
    class_mapping: dict[str, str] | None = None
    server_url: str | None = None
    display_name: str | None = None


@dataclass(slots=True)
class DatasetRenameRequest:
    snapshot_path: str
    display_name: str


@dataclass(slots=True)
class TrainingRequest:
    snapshot_path: str
    checkpoint: str | None = None
    output_dir: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] | None = None
    training_source: str | None = None
    hosted_model_id: str | None = None
    data_mode: Literal[
        "auto-split", "training-monitor", "separate-snapshots"
    ] = "separate-snapshots"
    validation_snapshot_path: str | None = None
    test_snapshot_path: str | None = None
    patience: int | None = None
    allow_image_level_split: bool = False


@dataclass(slots=True)
class ActiveLearningRequest:
    predictions: str
    calibration: str
    features: str
    output: str
    budget: int = 100
    seed: int = 42
    pool_multiplier: int = 200


@dataclass(slots=True)
class CvatCycleRequest:
    action: Literal["start", "refresh", "continue"]
    cycle: int
    queue: str | None = None
    server_url: str | None = None


@dataclass(slots=True)
class CloudCredentialsRequest:
    token_id: str
    token_secret: str


@dataclass(slots=True)
class CloudEstimateRequest:
    snapshot_path: str
    gpu: str = "L4"
    epochs: int | None = None
    max_cost_usd: float = 5.0
    image_size: int = 640
    training_source: str | None = None
    hosted_model_id: str | None = None
    data_mode: Literal[
        "auto-split", "training-monitor", "separate-snapshots"
    ] = "separate-snapshots"
    validation_snapshot_path: str | None = None
    test_snapshot_path: str | None = None
    patience: int | None = None
    allow_image_level_split: bool = False


@dataclass(slots=True)
class CloudTrainingRequest(CloudEstimateRequest):
    checkpoint: str | None = None
    batch_size: int | None = None
    device: Literal["auto", "cpu", "cuda"] | None = None
    estimated_usd: float = 0.0
    acknowledged: bool = False
    uploads_dataset: bool = False


@dataclass(slots=True)
class CloudJobActionRequest:
    action: Literal["refresh", "cancel", "collect", "cleanup"]


def _class_names(values: list[str]) -> list[str]:
    classes = [str(value).strip() for value in values]
    if not classes or any(not value for value in classes):
        raise ValueError("At least one non-empty class name is required")
    if len(classes) != len(set(classes)):
        raise ValueError("Class names must be unique")
    return classes


async def _read_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise ValueError("Request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return payload


def _string(data: dict[str, Any], key: str, *, required: bool = True, default: str = "") -> str:
    value = data.get(key, default)
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    if required and not value.strip():
        raise ValueError(f"{key} is required")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    return _string(data, key)


def _prediction_run_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("run_name must be text")
    name = value.strip()
    if not name:
        raise ValueError("run_name is required")
    if len(name) > 120:
        raise ValueError("run_name must be 120 characters or fewer")
    return name


def _integer(
    data: dict[str, Any], key: str, default: int | None = None, *, minimum: int | None = None
) -> int | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


def _number(
    data: dict[str, Any], key: str, default: float | None = None, *, minimum: float | None = None
) -> float | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    selected = float(value)
    if minimum is not None and selected < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return selected


def _boolean(data: dict[str, Any], key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def _mapping(data: dict[str, Any], key: str) -> dict[str, str | None] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict) or any(
        not isinstance(source, str) or (target is not None and not isinstance(target, str))
        for source, target in value.items()
    ):
        raise ValueError(f"{key} must map source labels to project labels or null for Ignore")
    return value


def _request(cls, payload: dict[str, Any]):
    """Parse the small local API contract without adding a schema dependency."""
    if cls is ProjectOpenRequest:
        return cls(path=_string(payload, "path"))
    if cls is ProjectCreateRequest:
        classes = payload.get("classes")
        if not isinstance(classes, list) or any(not isinstance(value, str) for value in classes):
            raise ValueError("classes must be a list of names")
        dimension_key = (
            "short_side_dimension" if "short_side_dimension" in payload else "max_dimension"
        )
        request = cls(
            name=_string(payload, "name"),
            path=_string(payload, "path"),
            image_root=_string(payload, "image_root"),
            classes=classes,
            model_preset=_string(payload, "model_preset", required=False, default="yolo26-l"),
            image_dimension=_integer(payload, dimension_key, 640, minimum=32),
            grayscale=_boolean(payload, "grayscale", True),
            clahe=_boolean(payload, "clahe", False),
        )
        return request
    if cls is PredictionsRequest:
        run_name = _prediction_run_name(_string(payload, "run_name"))
        confidence = _number(payload, "confidence")
        if confidence is not None and confidence > 1:
            raise ValueError("confidence must be between 0 and 1")
        device = _optional_string(payload, "device")
        if device is not None and device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        execution = _string(payload, "execution", required=False, default="local")
        if execution not in {"local", "modal"}:
            raise ValueError("execution must be local or modal")
        raw_batch_size = payload.get("batch_size", "auto")
        batch_size = None
        if raw_batch_size != "auto":
            batch_size = _integer(payload, "batch_size", minimum=1)
            if batch_size is not None and batch_size > 32:
                raise ValueError("batch_size must be between 1 and 32")
        gpu = _string(payload, "gpu", required=False, default="L4")
        from .cloud.estimate import GPU_RATES_PER_SECOND

        if gpu not in GPU_RATES_PER_SECOND:
            raise ValueError(f"Unsupported Modal GPU: {gpu}")
        max_cost_usd = _number(payload, "max_cost_usd", 5.0, minimum=0.01)
        if execution == "modal" and "max_cost_usd" not in payload:
            raise ValueError("Set a spending limit before choosing Modal prediction")
        if max_cost_usd is None or not math.isfinite(max_cost_usd):
            raise ValueError("max_cost_usd must be a positive finite number")
        return cls(
            run_name=run_name,
            image_root=_optional_string(payload, "image_root"),
            checkpoint=_optional_string(payload, "checkpoint"),
            output_dir=_optional_string(payload, "output_dir"),
            confidence=confidence,
            device=device,
            execution=execution,
            batch_size=batch_size,
            gpu=gpu,
            max_cost_usd=max_cost_usd,
            acknowledged=_boolean(payload, "acknowledged", False),
            uploads_dataset=_boolean(payload, "uploads_dataset", False),
            expected_image_count=_integer(payload, "expected_image_count", minimum=1),
            expected_total_bytes=_integer(payload, "expected_total_bytes", minimum=0),
            model_preset=_optional_string(payload, "model_preset"),
            hosted_model_id=_optional_string(payload, "hosted_model_id"),
            class_mapping=_mapping(payload, "class_mapping"),
        )
    if cls is DatasetImportRequest:
        return cls(
            archive=_string(payload, "archive"),
            class_mapping=_mapping(payload, "class_mapping"),
            display_name=_optional_string(payload, "display_name"),
        )
    if cls is CvatProjectListRequest:
        return cls(server_url=_optional_string(payload, "server_url"))
    if cls is CvatProjectImportRequest:
        return cls(
            project_id=_string(payload, "project_id"),
            class_mapping=_mapping(payload, "class_mapping"),
            server_url=_optional_string(payload, "server_url"),
            display_name=_optional_string(payload, "display_name"),
        )
    if cls is DatasetRenameRequest:
        return cls(
            snapshot_path=_string(payload, "snapshot_path"),
            display_name=_string(payload, "display_name"),
        )
    if cls is TrainingRequest:
        epochs = _integer(payload, "epochs", minimum=1)
        batch_size = _integer(payload, "batch_size", minimum=1)
        patience = _integer(payload, "patience", 25, minimum=0)
        device = _optional_string(payload, "device")
        if device is not None and device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        data_mode = _string(
            payload, "data_mode", required=False, default="separate-snapshots"
        )
        if data_mode not in {"auto-split", "training-monitor", "separate-snapshots"}:
            raise ValueError(
                "data_mode must be auto-split, training-monitor, or separate-snapshots"
            )
        return cls(
            snapshot_path=_string(payload, "snapshot_path"),
            checkpoint=_optional_string(payload, "checkpoint"),
            output_dir=_optional_string(payload, "output_dir"),
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            training_source=_optional_string(payload, "training_source"),
            hosted_model_id=_optional_string(payload, "hosted_model_id"),
            data_mode=data_mode,
            validation_snapshot_path=_optional_string(payload, "validation_snapshot_path"),
            test_snapshot_path=_optional_string(payload, "test_snapshot_path"),
            patience=patience,
            allow_image_level_split=_boolean(payload, "allow_image_level_split", False),
        )
    if cls is ActiveLearningRequest:
        return cls(
            predictions=_string(payload, "predictions"),
            calibration=_string(payload, "calibration"),
            features=_string(payload, "features"),
            output=_string(payload, "output"),
            budget=_integer(payload, "budget", 100, minimum=1),
            seed=_integer(payload, "seed", 42, minimum=0),
            pool_multiplier=_integer(payload, "pool_multiplier", 200, minimum=1),
        )
    if cls is CvatCycleRequest:
        action = _string(payload, "action")
        if action not in {"start", "refresh", "continue"}:
            raise ValueError("action must be start, refresh, or continue")
        cycle = _integer(payload, "cycle", minimum=0)
        if cycle is None:
            raise ValueError("cycle is required")
        return cls(
            action=action,
            cycle=cycle,
            queue=_optional_string(payload, "queue"),
            server_url=_optional_string(payload, "server_url"),
        )
    if cls is CloudCredentialsRequest:
        return cls(
            token_id=_string(payload, "token_id"), token_secret=_string(payload, "token_secret")
        )
    if cls is CloudEstimateRequest:
        data_mode = _string(
            payload, "data_mode", required=False, default="separate-snapshots"
        )
        if data_mode not in {"auto-split", "training-monitor", "separate-snapshots"}:
            raise ValueError(
                "data_mode must be auto-split, training-monitor, or separate-snapshots"
            )
        return cls(
            snapshot_path=_string(payload, "snapshot_path"),
            gpu=_string(payload, "gpu", required=False, default="L4"),
            epochs=_integer(payload, "epochs", minimum=1),
            max_cost_usd=_number(payload, "max_cost_usd", 5.0, minimum=0.001),
            image_size=_integer(payload, "image_size", 640, minimum=1),
            training_source=_optional_string(payload, "training_source"),
            hosted_model_id=_optional_string(payload, "hosted_model_id"),
            data_mode=data_mode,
            validation_snapshot_path=_optional_string(payload, "validation_snapshot_path"),
            test_snapshot_path=_optional_string(payload, "test_snapshot_path"),
            patience=_integer(payload, "patience", 25, minimum=0),
            allow_image_level_split=_boolean(payload, "allow_image_level_split", False),
        )
    if cls is CloudTrainingRequest:
        base = _request(CloudEstimateRequest, payload)
        batch_size = _integer(payload, "batch_size", minimum=1)
        device = _optional_string(payload, "device")
        if device is not None and device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        estimated_usd = _number(payload, "estimated_usd", minimum=0)
        if estimated_usd is None:
            raise ValueError("estimated_usd is required")
        return cls(
            snapshot_path=base.snapshot_path,
            gpu=base.gpu,
            epochs=base.epochs,
            max_cost_usd=base.max_cost_usd,
            image_size=base.image_size,
            training_source=base.training_source,
            hosted_model_id=base.hosted_model_id,
            data_mode=base.data_mode,
            validation_snapshot_path=base.validation_snapshot_path,
            test_snapshot_path=base.test_snapshot_path,
            patience=base.patience,
            allow_image_level_split=base.allow_image_level_split,
            checkpoint=_optional_string(payload, "checkpoint"),
            batch_size=batch_size,
            device=device,
            estimated_usd=estimated_usd,
            acknowledged=_boolean(payload, "acknowledged", False),
            uploads_dataset=_boolean(payload, "uploads_dataset", False),
        )
    raise TypeError(f"Unsupported request model: {cls.__name__}")


def _model_summaries() -> list[dict[str, str]]:
    return [
        {
            "model_id": item.model_id,
            "name": str(item.metadata.get("family", item.model_id)),
            "architecture": item.architecture,
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
    """Use the requested project short-side dimension for every model source."""
    source = request.training_source or (
        "amphilens-pretrained" if request.hosted_model_id else "general-pretrained"
    )
    if source not in {"general-pretrained", "amphilens-pretrained", "project-checkpoint"}:
        raise ValueError("Choose a supported training source before requesting an estimate")
    if source == "amphilens-pretrained":
        if not request.hosted_model_id:
            raise ValueError("Choose an AmphiLens pretrained model before requesting an estimate")
        from .models.hosted_models import get_hosted_model

        get_hosted_model(request.hosted_model_id)
        return request.image_size
    if request.hosted_model_id:
        raise ValueError("Choose either General pretrained weights or an AmphiLens model")
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


def prediction_preflight(
    store: ProjectStore,
    request: PredictionsRequest,
    *,
    credentials_store=None,
    check_connection: bool = False,
) -> dict[str, Any]:
    """Count local inputs and report available history before any cloud upload."""
    if request.execution != "modal":
        raise ValueError("Cloud prediction preflight is only available for Modal execution")
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
    image_root = request.image_root or project.image_roots[0]
    image_root_path = Path(image_root).expanduser().resolve()
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

    timing_samples = _modal_prediction_timing_samples(store, model_id, request.gpu)
    estimated_cost = None
    if timing_samples:
        rate = (
            GPU_RATES_PER_SECOND[request.gpu]
            + MODAL_CPU_RATE_PER_CORE_SECOND * CPU_CORES
            + MODAL_MEMORY_RATE_PER_GIB_SECOND * MEMORY_GIB
        )
        estimated_cost = round(sum(timing_samples) / len(timing_samples) * len(paths) * rate, 4)

    if check_connection:
        if credentials_store is None:
            raise ValueError("Modal credentials are not configured")
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
        "max_cost_usd": request.max_cost_usd,
        "estimate_note": (
            "Estimate based on prior Modal timing for this model and GPU; actual billing can vary."
            if estimated_cost is not None
            else "No timing history is available for this model and GPU."
        ),
    }


def _modal_prediction_timing_samples(store: ProjectStore, model_id: str, gpu: str) -> list[float]:
    """Return per-image Modal runtimes for the same model and GPU only."""
    timing_samples = []
    for path in (store.root / "runs").glob("*/cloud-cost.json"):
        try:
            sample = json.loads(path.read_text(encoding="utf-8"))
            if (
                sample.get("provider") == "modal"
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
        get_hosted_model,
        resolve_class_mapping,
    )
    from .reporting import write_report
    from .runs import run_resumable_inference

    run_name = _prediction_run_name(request.run_name)
    project = store.load_manifest()
    image_root = request.image_root or project.image_roots[0]
    image_root_path = Path(image_root).expanduser().resolve()
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
    if request.execution == "modal":
        if not request.acknowledged or not request.uploads_dataset:
            raise ValueError("Explicit image upload and Modal spending-limit consent is required")
        if cloud_transport is None:
            raise ValueError("Modal credentials are not configured for cloud prediction")
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
    overrides: dict[str, Any] = {}
    if request.confidence is not None:
        overrides["confidence"] = request.confidence
    if request.execution == "modal":
        actual_device = "cuda"
        requested_batch_size = request.batch_size or 8
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
            overrides=overrides,
        )
        if request.execution == "local":
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
        "gpu": request.gpu if request.execution == "modal" else None,
        "requested_batch_size": request.batch_size or "auto",
    }
    import hashlib
    import json

    run_signature = hashlib.sha256(
        json.dumps(run_metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    run_id = f"predict-{effective.model_id}-{run_signature}"
    output_dir = request.output_dir or str(store.root / "runs" / run_id)
    if request.execution == "modal":
        from .cloud.estimate import MAX_FUNCTION_TIMEOUT_SECONDS
        from .cloud.prediction import ModalPredictionDetector

        cancel_check = getattr(progress_callback, "cancel_requested", None)
        if callable(cancel_check) and cancel_check():
            from .runs import InferenceInterruption

            raise InferenceInterruption("Prediction canceled before starting cloud work.")

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
                checkpoint_path, checkpoint_sha256
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
            store, str(model_spec["model_id"]), request.gpu
        )
        detector = ModalPredictionDetector(
            cloud_transport,
            job_key=f"pred-{uuid.uuid4().hex}",
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
        )
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
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "saving_results",
                "message": "Writing reports and updating the project",
                "completed": summary.completed_images,
                "failed": len(summary.failed_images),
                "total": summary.image_count,
                "remaining": max(
                    0, summary.image_count - summary.completed_images - len(summary.failed_images)
                ),
                "progress": 1.0,
                "device": getattr(detector, "device_name", effective.device),
            }
        )
    report = write_report(summary.predictions_csv, Path(output_dir) / "report")
    markdown = Path(report["markdown"])
    json_report = Path(report["summary_json"])
    if summary.interruption_reason:
        message = (
            f"Prediction paused after {summary.completed_images} of {summary.image_count} images. "
            f"{summary.interruption_reason}"
        )
    elif request.execution == "modal":
        message = (
            f"Processed {summary.completed_images} images on {detector.device_name}; "
            f"found {summary.detection_count} detections. Estimated Modal spend: "
            f"${detector.estimated_cost_usd:.2f} of ${request.max_cost_usd:.2f}."
        )
    else:
        message = (
            f"Processed {summary.completed_images} images on "
            f"{describe_device_name(effective.device)} with batch size "
            f"{effective.batch_size}; found {summary.detection_count} detections."
        )
    return JobOutput(
        result={
            "message": message,
            "completed_images": summary.completed_images,
            "image_count": summary.image_count,
            "cancelled": bool(
                summary.interruption_reason and "cancel" in summary.interruption_reason.lower()
            ),
            "effective_batch_size": json.loads(summary.summary_json.read_text()).get(
                "effective_batch_size", 0
            ),
            "estimated_modal_cost_usd": (
                round(detector.estimated_cost_usd, 4) if request.execution == "modal" else None
            ),
            "device": (
                detector.device_name
                if request.execution == "modal"
                else describe_device_name(effective.device)
            ),
            "paths": {
                "csv": str(summary.predictions_csv),
                "report_markdown": str(markdown),
                "report_json": str(json_report),
            },
            "prediction": {
                "run_id": summary.run_id,
                "csv": str(summary.predictions_csv),
                "image_root": str(image_root_path),
                "project_root": str(store.root),
                "detection_count": summary.detection_count,
                "run_name": run_name,
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


def run_dataset_import_job(store: ProjectStore, request: DatasetImportRequest) -> dict[str, Any]:
    snapshot = store.import_dataset(
        request.archive,
        class_mapping=request.class_mapping,
        display_name=request.display_name,
    )
    return {
        "message": (
            f"Imported {len(snapshot.manifest.images)} images as "
            f"{store.dataset_display_name(snapshot.root)}."
        ),
        "paths": {"snapshot": str(snapshot.root)},
        "snapshot": snapshot.manifest.to_dict(),
    }


def run_cvat_import_job(store: ProjectStore, request: CvatProjectImportRequest) -> dict[str, Any]:
    from .annotations.initial import CVATProjectImportService
    from .annotations.managed import CVATSdkTransport

    transport = CVATSdkTransport(server_url=request.server_url)
    snapshots = CVATProjectImportService(store, transport).import_project_tasks(
        request.project_id,
        class_mapping=request.class_mapping,
        display_name=request.display_name,
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
        event = dict(values)
        if phase == "model_download" and "progress" in event:
            event["phase_progress"] = event.pop("progress")
        callback({"phase": phase, **event})

    return report


def run_training_job(
    store: ProjectStore,
    request: TrainingRequest,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> JobOutput:
    from .configuration import resolve_effective_configuration
    from .models import load_detector, load_preset_detector
    from .training import TrainingConfig, train_snapshot_and_register

    project = store.load_manifest()
    snapshot, validation_snapshot, test_snapshot = _training_snapshots(store, request)
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
    overrides: dict[str, Any] = {}
    if request.epochs is not None:
        overrides["epochs"] = request.epochs
    if request.batch_size is not None:
        overrides["batch_size"] = request.batch_size
    if request.device is not None:
        overrides["device"] = request.device
    if request.patience is not None:
        overrides["patience"] = request.patience
    overrides["freeze_strategy"] = "paper-phased"
    effective = resolve_effective_configuration(
        project,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_path=checkpoint or None,
        hosted_model=hosted_model,
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
        )
        if checkpoint
        else load_preset_detector(
            ModelCatalog().get(effective.model_preset), classes=list(effective.classes)
        )
    )
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
            metadata={"effective_configuration": effective.to_dict()},
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


def run_active_learning_job(store: ProjectStore, request: ActiveLearningRequest) -> JobOutput:
    from .active_learning import HybridPPALConfig, HybridPPALStrategy, PPALCalibration
    from .curation import write_selection_artifacts
    from .inference import read_predictions_csv

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
    selected = HybridPPALStrategy(config).select(
        read_predictions_csv(predictions_path),
        calibration,
        features=json.loads(features_path.read_text(encoding="utf-8")),
    )
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


def _cloud_service(store: ProjectStore, credentials_store):
    from .cloud.modal_transport import ModalTransport
    from .cloud.service import CloudTrainingService

    credentials = credentials_store.resolve()
    return CloudTrainingService(store, ModalTransport(credentials)), credentials


def run_cloud_training_job(
    store: ProjectStore,
    request: CloudTrainingRequest,
    credentials_store,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from .cloud.models import CloudConsent
    from .configuration import resolve_effective_configuration

    service, credentials = _cloud_service(store, credentials_store)
    if credentials is None:
        raise ValueError(
            "Modal credentials are not configured. Save credentials or set both "
            "Modal environment variables."
        )
    snapshot, validation_snapshot, test_snapshot = _training_snapshots(store, request)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "model_setup",
                "message": "Preparing the model and dataset for Modal",
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
    overrides["freeze_strategy"] = "paper-phased"
    effective = resolve_effective_configuration(
        project,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_path=checkpoint or None,
        hosted_model=hosted_model,
        overrides=overrides,
    )
    image_size = request.image_size
    if hosted_model is not None:
        if request.image_size != effective.image_size:
            raise ValueError(
                f"The selected AmphiLens model uses {effective.image_size}px input. "
                "Request a fresh cloud estimate for that model before submitting."
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
            "test set"
            if test_snapshot or request.data_mode == "auto-split"
            else "not evaluated"
        ),
        "data_mode": request.data_mode,
        "validation_strategy": (
            "training-set-monitor"
            if request.data_mode == "training-monitor" or validation_snapshot is None
            else "separate-snapshot"
        ),
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
                "message": "Uploading the approved training snapshot to Modal",
                "gpu": request.gpu,
            }
        )
    job = service.submit(
        snapshot_path=request.snapshot_path,
        validation_snapshot_path=(
            str(validation_snapshot.root) if validation_snapshot else None
        ),
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
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "submitted",
                "message": "Modal accepted the training job; waiting for its first update",
                "gpu": request.gpu,
                "progress": 0.0,
            }
        )
    return {"message": f"Cloud training submitted: {job.run_id}.", "job": _cloud_job_dict(job)}


def _cloud_job_dict(job) -> dict[str, Any]:
    payload = job.to_dict()
    # Job logs can contain echoed provider output; redact values before the browser sees them.
    if payload.get("error"):
        payload["error"] = safe_error(RuntimeError(payload["error"])).split(": ", 1)[-1]
    if payload.get("log_tail"):
        payload["log_tail"] = redact_sensitive_text(str(payload["log_tail"]))[-8000:]
    for key in ("progress_details",):
        details = payload.get(key)
        if isinstance(details, dict):
            payload[key] = dict(details)
            for text_key in ("phase", "message", "error"):
                if isinstance(payload[key].get(text_key), str):
                    payload[key][text_key] = redact_sensitive_text(payload[key][text_key])[:1000]
            metrics = payload[key].get("metrics")
            if isinstance(metrics, dict):
                payload[key]["metrics"] = {
                    str(metric_key)[:80]: (
                        redact_sensitive_text(metric_value)[:240]
                        if isinstance(metric_value, str)
                        else metric_value
                    )
                    for metric_key, metric_value in list(metrics.items())[:24]
                    if isinstance(metric_value, (str, int, bool))
                    or (isinstance(metric_value, float) and math.isfinite(metric_value))
                }
    events = payload.get("progress_events")
    if isinstance(events, list):
        safe_events = []
        for event in events[-100:]:
            if not isinstance(event, dict):
                continue
            safe_event = dict(event)
            for text_key in ("phase", "message", "error"):
                if isinstance(safe_event.get(text_key), str):
                    safe_event[text_key] = redact_sensitive_text(safe_event[text_key])[:1000]
            metrics = safe_event.get("metrics")
            if isinstance(metrics, dict):
                safe_event["metrics"] = {
                    str(metric_key)[:80]: (
                        redact_sensitive_text(metric_value)[:240]
                        if isinstance(metric_value, str)
                        else metric_value
                    )
                    for metric_key, metric_value in list(metrics.items())[:24]
                    if isinstance(metric_value, (str, int, bool))
                    or (isinstance(metric_value, float) and math.isfinite(metric_value))
                }
            safe_events.append(safe_event)
        payload["progress_events"] = safe_events
    return payload


def _create_cloud_action(store: ProjectStore, credentials_store, run_id: str, action: str):
    from .core import ValidationError

    service, credentials = _cloud_service(store, credentials_store)
    if credentials is None:
        raise ValueError(
            "Modal credentials are not configured. Save credentials or set both "
            "Modal environment variables."
        )
    if action == "refresh":
        record = service.refresh(run_id)
    elif action == "cancel":
        record = service.cancel(run_id)
    elif action == "collect":
        record = service.collect(run_id)
    elif action == "cleanup":
        record = service.retry_cleanup(run_id)
    else:
        raise ValidationError(f"Unsupported cloud job action: {action}")
    return _cloud_job_dict(record)


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
                [body.image_root],
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
            # Snapshot availability is deliberately not checked: prediction-only works alone.
            if body.execution == "modal":
                if not body.acknowledged or not body.uploads_dataset:
                    raise ValueError(
                        "Explicit consent is required. Review and approve the Modal image "
                        "transfer and spending limit first"
                    )
                if body.expected_image_count is None or body.expected_total_bytes is None:
                    raise ValueError(
                        "Review the image count and transfer size before cloud prediction"
                    )
                preview = prediction_preflight(store, body)
                if (
                    preview["image_count"] != body.expected_image_count
                    or preview["total_bytes"] != body.expected_total_bytes
                ):
                    raise ValueError(
                        "The selected images changed; review the updated transfer total"
                    )
                credentials = app.state.cloud_credentials_store.resolve()
                if credentials is None:
                    raise ValueError(
                        "Modal credentials are not configured; save them in System health"
                    )
                from .cloud.modal_transport import ModalTransport

                transport = ModalTransport(credentials)
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
                check_connection=True,
            )
            return _json(result)
        except Exception as exc:
            return _failure(exc)

    async def dataset_import(request: Request) -> Response:
        try:
            body = _request(DatasetImportRequest, await _read_body(request))
            store = active_store()
            return _json(submit(lambda: run_dataset_import_job(store, body)))
        except Exception as exc:
            return _failure(exc)

    async def dataset_import_cvat(request: Request) -> Response:
        try:
            body = _request(CvatProjectImportRequest, await _read_body(request))
            store = active_store()
            return _json(submit(lambda: run_cvat_import_job(store, body)))
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
            return _json(submit(lambda: run_active_learning_job(store, body)))
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
            if not isinstance(project_root, str) or not isinstance(run_id, str) or not run_id:
                raise ValueError("Prediction project or run information is missing")
            _validate_run_id(run_id)
            store = ProjectStore(project_root)
            store.load_manifest()
            destination = store.root / "runs" / run_id / "images"
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
                or run_manifest.status not in {"completed", "completed_with_failures"}
            ):
                raise ValueError("Prediction run is not complete or does not match this job")

            def run_ocr(progress_callback):
                from .timestamp_ocr import process_prediction_csv

                summary = process_prediction_csv(
                    csv_path,
                    image_root,
                    progress_callback=progress_callback,
                )
                return JobOutput(
                    result={
                        "message": (
                            f"Read timestamps for {summary.recognized_count} of "
                            f"{summary.image_count} unique detected images; "
                            f"{summary.unparsed_count} were unresolved."
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

    async def cloud_estimate(request: Request) -> Response:
        try:
            body = _request(CloudEstimateRequest, await _read_body(request))
            store = active_store()
            snapshot, validation_snapshot, test_snapshot = _training_snapshots(store, body)
            service, _ = _cloud_service(store, app.state.cloud_credentials_store)
            project_config = store.load_manifest().project_config
            estimate = service.estimate(
                snapshot.root,
                gpu=body.gpu,
                epochs=(body.epochs or project_config.epochs) * 2,
                max_cost_usd=body.max_cost_usd,
                image_size=_cloud_estimate_image_size(body),
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
            service, _ = _cloud_service(store, app.state.cloud_credentials_store)
            return _json({"jobs": [_cloud_job_dict(job) for job in service.list_jobs()]})
        except Exception as exc:
            return _failure(exc)

    async def cloud_job_action(request: Request) -> Response:
        try:
            run_id = request.path_params["run_id"]
            action = request.path_params["action"]
            if action not in {"refresh", "cancel", "collect", "cleanup"}:
                return _json({"detail": "Unsupported cloud job action"}, 404)
            store = active_store()
            payload = _create_cloud_action(store, app.state.cloud_credentials_store, run_id, action)
            response: dict[str, Any] = {"job": payload}
            if action == "collect":
                checkpoint_dir = store.root / "checkpoints" / run_id
                checkpoint = checkpoint_dir / "best.pt"
                manifest = checkpoint_dir / "checkpoint.json"
                local_job = app.state.jobs.register_result(
                    JobOutput(
                        result={
                            "message": f"Verified checkpoint registered at {checkpoint}.",
                            "paths": {"checkpoint": str(checkpoint), "manifest": str(manifest)},
                        },
                        downloads=[
                            DownloadArtifact(checkpoint, "Verified model checkpoint", "best.pt"),
                            DownloadArtifact(
                                manifest,
                                "Checkpoint manifest",
                                "checkpoint.json",
                                "application/json",
                            ),
                        ],
                    )
                )
                response["result"] = local_job.get("result")
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
    app.add_route("/api/cvat/projects", cvat_projects, methods=["POST"])
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
    app.add_route("/api/cloud/estimate", cloud_estimate, methods=["POST"])
    app.add_route("/api/cloud/training", cloud_training, methods=["POST"])
    app.add_route("/api/cloud/jobs", cloud_jobs, methods=["GET"])
    app.add_route("/api/cloud/jobs/{run_id:str}/{action:str}", cloud_job_action, methods=["POST"])
    web_root = Path(__file__).with_name("web")
    app.mount("/static", StaticFiles(directory=web_root), name="assets")
    app.mount("/", StaticFiles(directory=web_root, html=True), name="static")
    return app


app = create_app()
