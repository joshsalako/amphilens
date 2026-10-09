from __future__ import annotations

import json

import pytest

import amphilens.cloud as cloud
import amphilens.cloud.provider_settings as provider_settings
from amphilens.cloud.models import CloudJobRecord


def test_cloud_provider_settings_round_trip_outside_the_project(tmp_path):
    settings_type = getattr(cloud, "CloudProviderSettingsStore", None)
    assert settings_type is not None, "provider settings store is not implemented"

    path = tmp_path / "cloud-providers.json"
    store = settings_type(path)
    settings = {
        "azure_ml": {
            "subscription_id": "subscription-123",
            "resource_group": "wildlife-rg",
            "workspace_name": "amphilens-workspace",
            "location": "eastus",
            "managed_identity_client_id": "identity-123",
        },
        "vertex_ai": {
            "project_id": "wildlife-project",
            "location": "us-central1",
            "staging_bucket": "amphilens-staging",
            "service_account": "amphilens-job@example.iam.gserviceaccount.com",
        },
    }

    store.save(settings)

    assert store.load() == settings
    assert path.stat().st_mode & 0o777 == 0o600
    saved_text = path.read_text(encoding="utf-8")
    assert json.loads(saved_text)["providers"] == settings


def test_cloud_provider_settings_reject_secret_values(tmp_path):
    settings_type = getattr(cloud, "CloudProviderSettingsStore", None)
    assert settings_type is not None, "provider settings store is not implemented"

    store = settings_type(tmp_path / "cloud-providers.json")

    with pytest.raises(ValueError, match="secret|credential|token"):
        store.save({"azure_ml": {"client_secret": "must-not-be-stored"}})

    assert not store.path.exists()


def test_cloud_job_record_without_provider_remains_a_modal_job():
    saved = {
        "run_id": "cloud-legacy",
        "job_key": "legacy",
        "state": "submitted",
        "created_at": "2026-10-09T00:00:00+00:00",
        "updated_at": "2026-10-09T00:00:00+00:00",
        "project_name": "field-study",
        "snapshot_id": "snapshot-001",
        "effective_fingerprint": "fingerprint",
        "effective_configuration": {},
        "training_config": {},
        "consent": {
            "acknowledged": True,
            "uploads_dataset": True,
            "estimated_usd": 0.1,
            "max_cost_usd": 1.0,
        },
        "estimate": {},
        "gpu": "T4",
        "timeout_seconds": 60,
    }

    legacy = CloudJobRecord.from_dict(saved)
    saved["provider"] = "azure_ml"
    azure = CloudJobRecord.from_dict(saved)

    assert legacy.provider == "modal"
    assert azure.to_dict()["provider"] == "azure_ml"


def test_provider_settings_require_existing_workspace_and_job_identity():
    azure_type = getattr(provider_settings, "AzureMLSettings", None)
    vertex_type = getattr(provider_settings, "VertexAISettings", None)
    assert azure_type is not None, "Azure ML settings are not implemented"
    assert vertex_type is not None, "Vertex AI settings are not implemented"

    with pytest.raises(ValueError, match="subscription_id|workspace_name"):
        azure_type.from_mapping({"resource_group": "wildlife-rg"})
    with pytest.raises(ValueError, match="project_id|staging_bucket"):
        vertex_type.from_mapping({"location": "us-central1"})

    azure = azure_type.from_mapping(
        {
            "subscription_id": "subscription-123",
            "resource_group": "wildlife-rg",
            "workspace_name": "amphilens-workspace",
            "location": "eastus",
            "managed_identity_client_id": "identity-123",
        }
    )
    vertex = vertex_type.from_mapping(
        {
            "project_id": "wildlife-project",
            "location": "us-central1",
            "staging_bucket": "gs://amphilens-staging/",
            "service_account": "amphilens-job@example.iam.gserviceaccount.com",
        }
    )
    assert azure.location == "eastus"
    assert vertex.staging_bucket == "amphilens-staging"
