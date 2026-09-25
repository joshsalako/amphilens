"""Provider-neutral job contracts and the local reference backend."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .core import ValidationError


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class JobSpec:
    operation: str
    project_dir: Path
    config: dict[str, Any]
    model_ref: str | None = None
    input_refs: list[str] = field(default_factory=list)
    code_version: str = "working-tree"
    job_id: str = field(default_factory=lambda: f"job-{uuid.uuid4().hex[:12]}")
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.operation.strip():
            raise ValidationError("Job operation cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["project_dir"] = str(self.project_dir.expanduser().resolve())
        return data


@dataclass(slots=True)
class JobHandle:
    job_id: str
    backend: str
    state: str


@dataclass(slots=True)
class JobStatus:
    job_id: str
    state: str
    updated_at: str
    message: str = ""
    artifacts: list[str] = field(default_factory=list)


class ExecutionBackend(Protocol):
    def submit(self, job: JobSpec) -> JobHandle: ...

    def status(self, job_id: str) -> JobStatus: ...


class LocalExecutionBackend:
    """Persist jobs for local workers and future remote-compatible runners."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def submit(self, job: JobSpec) -> JobHandle:
        directory = self.root / job.job_id
        if directory.exists():
            raise ValidationError(f"Job already exists: {job.job_id}")
        directory.mkdir(parents=True)
        (directory / "job.json").write_text(json.dumps(job.to_dict(), indent=2) + "\n")
        status = JobStatus(job.job_id, "queued", _now())
        self._write_status(status)
        return JobHandle(job.job_id, "local", "queued")

    def status(self, job_id: str) -> JobStatus:
        path = self.root / job_id / "status.json"
        if not path.is_file():
            raise ValidationError(f"Job not found: {job_id}")
        return JobStatus(**json.loads(path.read_text()))

    def update(
        self, job_id: str, state: str, message: str = "", artifacts: list[str] | None = None
    ) -> JobStatus:
        status = JobStatus(job_id, state, _now(), message, artifacts or [])
        self._write_status(status)
        return status

    def _write_status(self, status: JobStatus) -> None:
        path = self.root / status.job_id / "status.json"
        path.write_text(json.dumps(asdict(status), indent=2) + "\n")
