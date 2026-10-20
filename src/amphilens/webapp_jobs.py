"""Local API errors, native path selection, and bounded background jobs."""

from __future__ import annotations

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cancellation import WorkflowCancelled
from .core import AmphiLensError
from .locations import ProjectLocationError
from .models.backends import OptionalDependencyError
from .preprocessing import PreprocessingDependencyError

_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(authorization\s*:\s*bearer\s+|[a-z0-9_.-]*(?:token|secret|password|api[_-]?key)"
    r"[a-z0-9_.-]*\s*[:=]\s*)([^\s,;]+)"
)
_URL_CREDENTIAL_PATTERN = re.compile(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@")
_LOG = logging.getLogger("amphilens.webapp")
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
        # The local server is often started from Terminal. Activate Finder first
        # so the native chooser is not left behind the terminal window.
        command = [
            "osascript",
            "-e",
            'do shell script "/usr/bin/open -a Finder >/dev/null 2>&1 || true"',
            "-e",
            scripts[kind],
        ]
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
    message = re.sub(r"(?i)(https?://[^\s?#]+)\?[^\s\"'<>)]*", r"\1?[redacted]", message)
    message = re.sub(
        r"(?i)((?:token|secret|password|api[_-]?key|access[_-]?token|client[_-]?secret|sig|signature)\b\s*[:=]\s*)[^\s,;&]+",
        r"\1[redacted]",
        message,
    )
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
        except WorkflowCancelled as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.result = {"message": str(exc), "cancelled": True}
                job.state = "canceled"
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
