"""Provider-neutral cloud transport boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .models import TERMINAL_CLOUD_JOB_STATES, CloudJobRecord


class CloudTransport(Protocol):
    """Operations required by cloud training; provider SDK types stay behind adapters."""

    def upload(self, source: Path, remote_path: str, sha256: str) -> bool: ...

    def submit(self, payload: dict[str, Any]) -> str: ...

    def poll(self, call_id: str, *, remote_prefix: str) -> dict[str, Any] | None: ...

    def cancel(self, call_id: str) -> None: ...

    def download(self, remote_ref: str, destination: Path) -> None: ...

    def cleanup(self, job_key: str) -> dict[str, Any]: ...

    def dashboard_url(self, call_id: str) -> str: ...


def active_jobs(records: list[CloudJobRecord]) -> list[CloudJobRecord]:
    return [record for record in records if record.state not in TERMINAL_CLOUD_JOB_STATES]
