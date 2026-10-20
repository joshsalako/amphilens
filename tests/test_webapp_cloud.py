from __future__ import annotations

import pytest

pytest.importorskip("starlette")

import amphilens.webapp as webapp
from amphilens.core import ProjectManifest, ProjectStore


def test_managed_provider_settings_are_local_nonsecret_and_allowlisted(client, tmp_path):
    settings = {
        "subscription_id": "subscription",
        "resource_group": "wildlife-rg",
        "workspace_name": "amphilens",
        "location": "eastus",
        "managed_identity_client_id": "identity-client-id",
    }

    response = client.post(
        "/api/cloud/providers/settings",
        json={"provider": "azure_ml", "settings": settings},
    )

    assert response.status_code == 200
    assert response.json()["settings"] == settings
    saved = tmp_path / "cloud-providers.json"
    assert "subscription" in saved.read_text(encoding="utf-8")
    assert "access_token" not in saved.read_text(encoding="utf-8")
    assert saved.stat().st_mode & 0o777 == 0o600

    rejected = client.post(
        "/api/cloud/providers/settings",
        json={"provider": "azure_ml", "settings": {**settings, "access_token": "secret"}},
    )
    assert rejected.status_code == 400
    assert "access_token" not in saved.read_text(encoding="utf-8")


def test_managed_provider_check_reports_connection_from_adapter(client, monkeypatch, tmp_path):
    class FakeTransport:
        def probe(self):
            return {"provider": "vertex_ai", "connectivity": "connected", "region": "us-central1"}

    monkeypatch.setattr(
        webapp,
        "_cloud_service",
        lambda *_args, **_kwargs: (type("Service", (), {"transport": FakeTransport()})(), None),
    )
    configured = client.post(
        "/api/cloud/providers/settings",
        json={
            "provider": "vertex_ai",
            "settings": {
                "project_id": "project",
                "location": "us-central1",
                "staging_bucket": "staging",
                "service_account": "runner@example.com",
            },
        },
    )
    assert configured.status_code == 200
    image_root = tmp_path / "images"
    image_root.mkdir()
    project = ProjectStore(tmp_path / "project")
    project.create(ProjectManifest.create("field-study", [image_root], ["frog"]))
    opened = client.post("/api/projects/open", json={"path": str(project.root)})
    assert opened.status_code == 200

    response = client.post("/api/cloud/providers/check", json={"provider": "vertex_ai"})

    assert response.status_code == 200, response.text
    assert response.json()["connection"]["region"] == "us-central1"


def test_prediction_preflight_reports_bytes_and_only_uses_matching_timing_history(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "one.jpg").write_bytes(b"12345")
    (image_root / "two.jpg").write_bytes(b"1234567")
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("sample", [image_root], ["toad"]))
    request = webapp.PredictionsRequest(
        image_root=str(image_root),
        execution="modal",
        gpu="L4",
        max_cost_usd=2.0,
        model_preset="yolo26-l",
        run_name="Cloud cost estimate",
    )

    preview = webapp.prediction_preflight(store, request)

    assert preview["image_count"] == 2
    assert preview["total_bytes"] == 12
    assert preview["gpu"] == "L4"
    assert preview["estimated_cost_usd"] is None
    assert "No timing history" in preview["estimate_note"]

    history = store.root / "runs" / "predict-prior" / "cloud-cost.json"
    history.parent.mkdir(parents=True)
    history.write_text(
        '{"provider":"modal","gpu":"L4","model_id":"yolo26-l",'
        '"elapsed_seconds":8,"completed_images":4}',
        encoding="utf-8",
    )
    preview_with_history = webapp.prediction_preflight(store, request)
    assert preview_with_history["timing_history_count"] == 1
    assert preview_with_history["estimated_cost_usd"] > 0


def test_modal_prediction_api_requires_preflight_consent_before_job_submission(client, tmp_path):
    image_root = tmp_path / "consent-images"
    image_root.mkdir()
    project_path = tmp_path / "consent-project"
    created = client.post(
        "/api/projects/create",
        json={
            "name": "Consent",
            "path": str(project_path),
            "image_root": str(image_root),
            "classes": ["toad"],
        },
    )
    assert created.status_code == 200

    response = client.post(
        "/api/predictions",
        json={"run_name": "Consent test", "execution": "modal", "max_cost_usd": 1.0},
    )

    assert response.status_code == 400
    assert "consent" in response.json()["detail"].lower()


def test_cloud_estimate_uses_recorded_model_input_size_for_hosted_model():
    estimate = webapp._request(
        webapp.CloudEstimateRequest,
        {
            "snapshot_path": "/project/dataset",
            "image_size": 640,
            "training_source": "amphilens-pretrained",
            "hosted_model_id": "amphilens-yolo26-m",
        },
    )
    assert estimate.hosted_model_id == "amphilens-yolo26-m"
    assert webapp._cloud_estimate_image_size(estimate) == 1152

    estimate.hosted_model_id = "amphilens-rtdetr-l"
    assert webapp._cloud_estimate_image_size(estimate) == 640

    estimate.hosted_model_id = "missing-model"
    with pytest.raises(ValueError, match="Unknown AmphiLens pretrained model"):
        webapp._cloud_estimate_image_size(estimate)

    cloud_training = webapp._request(
        webapp.CloudTrainingRequest,
        {
            "snapshot_path": "/project/dataset",
            "training_source": "amphilens-pretrained",
            "hosted_model_id": "amphilens-yolo26-m",
            "image_size": 1152,
            "preprocessing": {
                "short_side_dimension": 384,
                "grayscale_enabled": False,
                "clahe_enabled": False,
            },
            "batch_size": 2,
            "estimated_usd": 1.0,
        },
    )
    assert cloud_training.training_source == "amphilens-pretrained"
    assert cloud_training.hosted_model_id == "amphilens-yolo26-m"
    assert cloud_training.image_size == 1152
    assert cloud_training.preprocessing["short_side_dimension"] == 384


def test_managed_prediction_blocks_unknown_price_before_upload(tmp_path):
    from PIL import Image

    from amphilens.webapp import PredictionsRequest, run_prediction_job

    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "sample.jpg"
    Image.new("RGB", (32, 24), color="gray").save(image_path)
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("field-study", [image_root], ["frog"]))

    class FakeTransport:
        uploads = []

        def price_rate(self, _gpu):
            raise ValueError("current Azure GPU rate is unavailable; submission is blocked")

        def upload_prediction_batch(self, *args):
            self.uploads.append(args)

    transport = FakeTransport()
    request = PredictionsRequest(
        run_name="Managed prediction",
        image_root=str(image_root),
        output_dir=str(tmp_path / "output"),
        execution="azure_ml",
        gpu="T4",
        max_cost_usd=1.0,
        acknowledged=True,
        uploads_dataset=True,
        expected_image_count=1,
        expected_total_bytes=image_path.stat().st_size,
    )

    with pytest.raises(ValueError, match="submission is blocked"):
        run_prediction_job(store, request, cloud_transport=transport)

    assert transport.uploads == []
