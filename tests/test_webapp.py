from __future__ import annotations

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient

from amphilens.core import ProjectStore
from amphilens.webapp import JobManager, create_app


def test_cvat_project_endpoint_returns_fresh_selected_project_details(client, monkeypatch):
    from amphilens.annotations import managed
    from amphilens.annotations.managed import CVATProjectSummary, CVATTaskSummary

    received = {}

    class FreshTransport:
        def __init__(self, server_url=None):
            received["server_url"] = server_url

        def get_project(self, project_id):
            received["project_id"] = project_id
            return CVATProjectSummary(
                project_id="17",
                name="WLT-UP",
                labels=["Small_Mammal", "Western_Leopard_Toad"],
                tasks=[CVATTaskSummary("31", "train", 12, "completed")],
            )

    monkeypatch.setattr(managed, "CVATSdkTransport", FreshTransport)
    response = client.post(
        "/api/cvat/project",
        json={"project_id": "17", "server_url": "http://localhost:8080"},
    )

    assert response.status_code == 200
    assert received == {"server_url": "http://localhost:8080", "project_id": "17"}
    assert response.json()["project"] == {
        "project_id": "17",
        "name": "WLT-UP",
        "labels": ["Small_Mammal", "Western_Leopard_Toad"],
        "task_count": 1,
        "tasks": [{"task_id": "31", "name": "train", "size": 12, "status": "completed"}],
    }


def test_bootstrap_includes_cvat_url_from_dotenv_but_never_token(client, tmp_path, monkeypatch):
    monkeypatch.delenv("CVAT_URL", raising=False)
    monkeypatch.delenv("CVAT_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "CVAT_URL=https://cvat.example.org\nCVAT_TOKEN=fixture-token-must-not-leak\n",
        encoding="utf-8",
    )

    response = client.get("/api/bootstrap")

    assert response.status_code == 200
    assert response.json()["cvat_server_url"] == "https://cvat.example.org"
    assert "fixture-token-must-not-leak" not in response.text


def test_bootstrap_and_project_create_open_close_persist(client, tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    project_path = tmp_path / "projects" / "toads"

    bootstrap = client.get("/api/bootstrap").json()
    assert bootstrap["project"] is None
    assert bootstrap["doctor"] == {"ok": True}
    assert bootstrap["default_project_root"].endswith("AmphiLens/projects")
    assert {model["model_id"] for model in bootstrap["models"]} == {
        "yolo26-n",
        "yolo26-s",
        "yolo26-m",
        "yolo26-l",
        "rtdetr-l",
        "rtdetr-x",
        "faster-rcnn-resnet50",
    }
    assert bootstrap["default_model_id"] == "yolo26-l"
    assert {item["model_id"] for item in bootstrap["hosted_models"]} == {
        "amphilens-yolo26-m",
        "amphilens-rtdetr-l",
        "amphilens-faster-rcnn-resnet50",
        "amphilens-yolo26-m-domain",
        "amphilens-rtdetr-l-domain",
        "amphilens-faster-rcnn-resnet50-domain",
    }
    assert all(len(item["revision"]) == 40 for item in bootstrap["hosted_models"])

    created = client.post(
        "/api/projects/create",
        json={
            "name": "Toads",
            "path": str(project_path),
            "image_root": str(image_root),
            "classes": ["toad", "frog"],
            "model_preset": "yolo26-l",
            "max_dimension": 640,
            "grayscale": True,
            "clahe": False,
        },
    )
    assert created.status_code == 200, created.text
    payload = created.json()["project"]
    assert payload["path"] == str(project_path.resolve())
    assert payload["classes"] == ["toad", "frog"]
    assert payload["model_preset"] == "yolo26-l"
    assert payload["counts"] == {"images": 0, "datasets": 0, "runs": 0}

    client.post("/api/projects/close")
    opened = client.post("/api/projects/open", json={"path": str(project_path)})
    assert opened.status_code == 200, opened.text
    assert opened.json()["project"]["name"] == "Toads"

    reopened_app = create_app(
        user_state_store=client.app.state.user_state_store,
        jobs=JobManager(max_workers=1, max_pending=2),
        source_checkout=tmp_path / "checkout",
    )
    reopened = TestClient(reopened_app).get("/api/bootstrap").json()
    assert reopened["project"]["path"] == str(project_path.resolve())
    reopened_app.state.jobs.shutdown()

    assert client.post("/api/projects/close").json() == {"project": None}
    assert client.get("/api/bootstrap").json()["project"] is None


def test_project_create_does_not_require_an_image_folder(client, tmp_path):
    project_path = tmp_path / "projects" / "field-notes"

    response = client.post(
        "/api/projects/create",
        json={
            "name": "Field notes",
            "path": str(project_path),
            "classes": ["toad"],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["project"]["image_roots"] == []
    assert ProjectStore(project_path).load_manifest().image_roots == []


def test_prediction_requires_a_folder_when_project_has_no_saved_folder(client, tmp_path):
    from amphilens.webapp import PredictionsRequest, run_prediction_job

    project_path = tmp_path / "projects" / "field-notes"
    created = client.post(
        "/api/projects/create",
        json={
            "name": "Field notes",
            "path": str(project_path),
            "classes": ["toad"],
        },
    )
    assert created.status_code == 200, created.text

    with pytest.raises(ValueError, match="Choose an image folder"):
        run_prediction_job(
            ProjectStore(project_path),
            PredictionsRequest(run_name="Camera survey"),
        )


def test_project_create_rejects_source_checkout_path(client, tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    project_path = checkout / "unsafe-project"
    image_root = tmp_path / "images"
    image_root.mkdir()
    client.app.state.source_checkout = checkout.resolve()

    response = client.post(
        "/api/projects/create",
        json={
            "name": "Unsafe",
            "path": str(project_path),
            "image_root": str(image_root),
            "classes": ["toad"],
        },
    )
    assert response.status_code == 400
    assert "inside the AmphiLens source checkout" in response.json()["detail"]
    assert not project_path.exists()
