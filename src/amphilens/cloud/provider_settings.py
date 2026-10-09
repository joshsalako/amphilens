"""Non-secret Azure and Google Cloud settings stored with user-level app state."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..state import default_user_state_path

PROVIDER_SETTINGS_SCHEMA_VERSION = 1
_PROVIDER_FIELDS = {
    "azure_ml": {
        "subscription_id",
        "resource_group",
        "workspace_name",
        "location",
        "managed_identity_client_id",
    },
    "vertex_ai": {"project_id", "location", "staging_bucket", "service_account"},
}


@dataclass(frozen=True, slots=True)
class AzureMLSettings:
    """Non-secret coordinates for an existing Azure ML workspace."""

    subscription_id: str
    resource_group: str
    workspace_name: str
    location: str
    managed_identity_client_id: str

    @classmethod
    def from_mapping(cls, values: dict[str, str]) -> AzureMLSettings:
        invalid = set(values) - _PROVIDER_FIELDS["azure_ml"]
        if invalid:
            raise ValueError(f"Unsupported Azure ML settings: {', '.join(sorted(invalid))}")
        required = tuple(_PROVIDER_FIELDS["azure_ml"])
        normalized = {key: str(values.get(key, "")).strip() for key in required}
        missing = [key for key, value in normalized.items() if not value]
        if missing:
            raise ValueError(f"Azure ML settings require: {', '.join(sorted(missing))}")
        return cls(**normalized)

    def to_mapping(self) -> dict[str, str]:
        return {
            "subscription_id": self.subscription_id,
            "resource_group": self.resource_group,
            "workspace_name": self.workspace_name,
            "location": self.location,
            "managed_identity_client_id": self.managed_identity_client_id,
        }


@dataclass(frozen=True, slots=True)
class VertexAISettings:
    """Non-secret coordinates and job identity for an existing Vertex project."""

    project_id: str
    location: str
    staging_bucket: str
    service_account: str

    @classmethod
    def from_mapping(cls, values: dict[str, str]) -> VertexAISettings:
        invalid = set(values) - _PROVIDER_FIELDS["vertex_ai"]
        if invalid:
            raise ValueError(f"Unsupported Vertex AI settings: {', '.join(sorted(invalid))}")
        required = tuple(_PROVIDER_FIELDS["vertex_ai"])
        normalized = {key: str(values.get(key, "")).strip() for key in required}
        bucket = normalized["staging_bucket"]
        if bucket.startswith("gs://"):
            bucket = bucket[5:]
        normalized["staging_bucket"] = bucket.strip("/")
        missing = [key for key, value in normalized.items() if not value]
        if missing:
            raise ValueError(f"Vertex AI settings require: {', '.join(sorted(missing))}")
        if "/" in normalized["staging_bucket"]:
            raise ValueError("Vertex AI staging_bucket must be a bucket name, not an object path")
        return cls(**normalized)

    def to_mapping(self) -> dict[str, str]:
        return {
            "project_id": self.project_id,
            "location": self.location,
            "staging_bucket": self.staging_bucket,
            "service_account": self.service_account,
        }


def default_provider_settings_path() -> Path:
    """Return a per-user settings path outside any AmphiLens project."""
    return default_user_state_path().with_name("cloud-providers.json")


class CloudProviderSettingsStore:
    """Read and write non-secret configuration for cloud job providers."""

    def __init__(self, path: str | Path | None = None):
        self.path = (
            Path(path).expanduser().resolve()
            if path is not None
            else default_provider_settings_path()
        )

    @staticmethod
    def _validate(settings: Any) -> dict[str, dict[str, str]]:
        if not isinstance(settings, dict):
            raise ValueError("Cloud provider settings must be an object")
        normalized: dict[str, dict[str, str]] = {}
        for provider, values in settings.items():
            if provider not in _PROVIDER_FIELDS:
                raise ValueError(f"Unsupported cloud provider: {provider}")
            if not isinstance(values, dict):
                raise ValueError(f"Settings for {provider} must be an object")
            invalid = set(values) - _PROVIDER_FIELDS[provider]
            if invalid:
                raise ValueError(
                    f"Unsupported or secret cloud setting field for {provider}: "
                    f"{', '.join(sorted(invalid))}"
                )
            normalized[provider] = {}
            for key, value in values.items():
                if not isinstance(value, str):
                    raise ValueError(f"Cloud setting {provider}.{key} must be text")
                normalized[provider][key] = value.strip()
        return normalized

    def load(self) -> dict[str, dict[str, str]]:
        if not self.path.is_file():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read cloud provider settings: {self.path}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != (
            PROVIDER_SETTINGS_SCHEMA_VERSION
        ):
            raise ValueError("Cloud provider settings have an unsupported schema")
        settings = value.get("providers")
        return self._validate(settings)

    def save(self, settings: dict[str, dict[str, str]]) -> None:
        normalized = self._validate(settings)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".cloud-providers-", dir=self.path.parent)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": PROVIDER_SETTINGS_SCHEMA_VERSION,
                        "providers": normalized,
                    },
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def update(self, provider: str, settings: dict[str, str]) -> None:
        if provider not in _PROVIDER_FIELDS:
            raise ValueError(f"Unsupported cloud provider: {provider}")
        providers = self.load()
        providers[provider] = settings
        self.save(providers)
