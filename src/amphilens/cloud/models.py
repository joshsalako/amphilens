"""Serializable cloud-training records with no provider SDK objects."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any

from ..core import CheckpointManifest, ValidationError


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class CloudConsent:
    acknowledged: bool
    uploads_dataset: bool
    estimated_usd: float
    max_cost_usd: float
    accepted_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.acknowledged or not self.uploads_dataset:
            raise ValidationError("Cloud training consent and dataset-upload consent are required")
        if (
            not math.isfinite(self.estimated_usd)
            or not math.isfinite(self.max_cost_usd)
            or self.estimated_usd < 0
            or self.max_cost_usd <= 0
        ):
            raise ValidationError("Cloud cost estimate and ceiling must be valid positive amounts")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CloudConsent:
        return cls(**data)


CLOUD_JOB_STATES = {
    "created",
    "prepared",
    "uploading",
    "uploaded",
    "submitted",
    "running",
    "finished",
    "verified",
    "incomplete",
    "failed",
    "canceled",
    "timed_out",
}
TERMINAL_CLOUD_JOB_STATES = {
    "verified",
    "incomplete",
    "failed",
    "canceled",
    "timed_out",
}


@dataclass(slots=True)
class CloudJobRecord:
    run_id: str
    job_key: str
    state: str
    created_at: str
    updated_at: str
    project_name: str
    snapshot_id: str
    effective_fingerprint: str
    effective_configuration: dict[str, Any]
    training_config: dict[str, Any]
    consent: CloudConsent
    estimate: dict[str, Any]
    gpu: str
    timeout_seconds: int
    deadline_at: str | None = None
    payload_hash: str = ""
    payload_size_bytes: int = 0
    call_id: str = ""
    dashboard_url: str = ""
    remote_state: str = ""
    phase: str = "created"
    progress: float | None = None
    progress_details: dict[str, Any] = field(default_factory=dict)
    progress_events: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    log_tail: str = ""
    environment: dict[str, str] = field(default_factory=dict)
    remote_result: dict[str, Any] = field(default_factory=dict)
    expected_echo: dict[str, Any] = field(default_factory=dict)
    base_checkpoint_path: str | None = None
    base_checkpoint_sha256: str = ""
    code_digest: str = ""
    code_version: str = "working-tree"
    cleanup_succeeded: bool | None = None
    cleanup_error: str = ""
    checkpoint_manifest: CheckpointManifest | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.state not in CLOUD_JOB_STATES:
            raise ValidationError(f"Unknown cloud job state: {self.state}")
        if not self.run_id or not self.job_key:
            raise ValidationError("Cloud job run_id and job_key are required")
        if self.timeout_seconds <= 0:
            raise ValidationError("Cloud job timeout_seconds must be positive")
        if self.progress is not None and not 0 <= self.progress <= 1:
            raise ValidationError("Cloud job progress must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["checkpoint_manifest"] = (
            self.checkpoint_manifest.to_dict() if self.checkpoint_manifest else None
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CloudJobRecord:
        accepted = {item.name for item in fields(cls)}
        values = {key: value for key, value in data.items() if key in accepted}
        consent_data = values.get("consent")
        if isinstance(consent_data, dict):
            values["consent"] = CloudConsent.from_dict(consent_data)
        manifest_data = values.get("checkpoint_manifest")
        if isinstance(manifest_data, dict):
            values["checkpoint_manifest"] = CheckpointManifest.from_dict(manifest_data)
        return cls(**values)
