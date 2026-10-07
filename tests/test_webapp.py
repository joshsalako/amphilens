from __future__ import annotations

import time
from subprocess import CompletedProcess

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient

import amphilens.webapp as webapp
from amphilens.core import ProjectManifest, ProjectStore
from amphilens.runs import InferenceInterruption
from amphilens.state import UserStateStore
from amphilens.webapp import DownloadArtifact, JobManager, JobOutput, create_app


@pytest.fixture
def jobs():
    manager = JobManager(max_workers=1, max_pending=3)
    try:
        yield manager
    finally:
        manager.shutdown()


@pytest.fixture
def client(tmp_path, jobs, monkeypatch):
    monkeypatch.setattr(
        "amphilens.webapp.run_doctor",
        lambda *args, **kwargs: type("Report", (), {"to_dict": lambda self: {"ok": True}})(),
    )
    app = create_app(
        user_state_store=UserStateStore(tmp_path / "state.json"),
        jobs=jobs,
        source_checkout=tmp_path / "checkout",
    )
    return TestClient(app, client=("127.0.0.1", 50000))


def _wait_for_job(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["state"] in {"completed", "failed"}:
            return job
        time.sleep(0.01)
    pytest.fail("job did not finish before test deadline")


def test_static_frontend_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "AmphiLens" in page.text
    assert 'data-path-picker="directory" data-path-target="open-path"' in page.text
    assert 'data-path-picker="directory" data-path-target="project-path-input"' in page.text
    assert 'src="/static/prediction_state.js"' in page.text
    assert 'src="/static/activity_view.js"' in page.text
    assert client.get("/static/prediction_state.js").status_code == 200
    assert client.get("/static/activity_view.js").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert 'api("/api/path-picker"' in client.get("/static/app.js").text
    script = client.get("/static/app.js").text
    assert 'name="training_source"' in script
    assert 'name="hosted_model_id"' in script
    assert "__ignore__" in script
    assert "downloaded directly by Modal" in script
    assert "private Hugging Face repository" not in script
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/favicon.svg").status_code == 200


def test_local_path_picker_returns_the_selected_path(client, monkeypatch, tmp_path):
    selected = tmp_path / "images"
    selected.mkdir()
    monkeypatch.setattr(webapp, "_native_path_picker", lambda kind: str(selected), raising=False)

    response = client.post("/api/path-picker", json={"kind": "directory"})

    assert response.status_code == 200
    assert response.json() == {"path": str(selected), "cancelled": False}


def test_local_path_picker_rejects_remote_clients(client):
    remote_client = TestClient(client.app, client=("192.0.2.10", 50000))

    response = remote_client.post("/api/path-picker", json={"kind": "directory"})

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "File and folder selection is available only from this computer."
    )


def test_local_path_picker_treats_cancel_as_a_normal_result(client, monkeypatch):
    monkeypatch.setattr(webapp, "_native_path_picker", lambda kind: None, raising=False)

    response = client.post("/api/path-picker", json={"kind": "file"})

    assert response.status_code == 200
    assert response.json() == {"path": None, "cancelled": True}


def test_job_manager_exposes_download_progress_to_status_clients(jobs):
    import threading

    ready = threading.Event()
    finish = threading.Event()

    def operation(report_progress):
        report_progress(
            {"message": "Downloading private model", "completed": 1, "total": 4, "progress": 0.25}
        )
        ready.set()
        finish.wait(timeout=2)
        return {"message": "Done"}

    job_id = jobs.submit(operation, with_progress=True)
    assert ready.wait(timeout=1)
    snapshot = jobs.snapshot(job_id)
    assert snapshot["progress"] == {
        "message": "Downloading private model",
        "completed": 1,
        "total": 4,
        "progress": 0.25,
    }
    finish.set()


def test_job_manager_exposes_cancellation_to_an_active_workflow(jobs):
    import threading

    started = threading.Event()

    def operation(report_progress):
        started.set()
        while not report_progress.cancel_requested():
            time.sleep(0.005)
        raise InferenceInterruption("Prediction canceled")

    job_id = jobs.submit(operation, with_progress=True)
    assert started.wait(timeout=1)
    assert jobs.cancel(job_id)
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        snapshot = jobs.snapshot(job_id)
        if snapshot["state"] == "canceled":
            break
        time.sleep(0.01)
    assert snapshot["state"] == "canceled"


def test_api_parses_hosted_inference_and_training_source_fields():
    prediction = webapp._request(
        webapp.PredictionsRequest,
        {
            "hosted_model_id": "amphilens-yolo26-m",
            "class_mapping": {"Other_Amphibian": None},
        },
    )
    training = webapp._request(
        webapp.TrainingRequest,
        {
            "snapshot_path": "/project/dataset",
            "training_source": "amphilens-pretrained",
            "hosted_model_id": "amphilens-yolo26-m",
        },
    )

    assert prediction.hosted_model_id == "amphilens-yolo26-m"
    assert prediction.class_mapping == {"Other_Amphibian": None}
    assert training.training_source == "amphilens-pretrained"
    assert training.hosted_model_id == "amphilens-yolo26-m"
    assert webapp.select_training_source(None, None, None) == "general-pretrained"


def test_prediction_request_parses_local_gpu_batching_and_modal_execution():
    local = webapp._request(
        webapp.PredictionsRequest,
        {"device": "mps", "batch_size": "auto"},
    )
    cloud = webapp._request(
        webapp.PredictionsRequest,
        {
            "execution": "modal",
            "gpu": "L4",
            "batch_size": 6,
            "max_cost_usd": 3.5,
            "acknowledged": True,
            "uploads_dataset": True,
        },
    )

    assert local.device == "mps"
    assert local.batch_size is None
    assert cloud.execution == "modal"
    assert cloud.gpu == "L4"
    assert cloud.batch_size == 6
    assert cloud.max_cost_usd == 3.5
    assert cloud.acknowledged is True
    assert cloud.uploads_dataset is True


def test_prediction_request_rejects_unsupported_modal_gpu_and_batch_size():
    with pytest.raises(ValueError, match="Unsupported Modal GPU"):
        webapp._request(webapp.PredictionsRequest, {"execution": "modal", "gpu": "invented"})
    with pytest.raises(ValueError, match="between 1 and 32"):
        webapp._request(webapp.PredictionsRequest, {"batch_size": 33})
    with pytest.raises(ValueError, match="spending limit"):
        webapp._request(webapp.PredictionsRequest, {"execution": "modal"})


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
        json={"execution": "modal", "max_cost_usd": 1.0},
    )

    assert response.status_code == 400
    assert "consent" in response.json()["detail"].lower()


def test_cloud_estimate_uses_catalog_input_size_for_hosted_model():
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
            "batch_size": 2,
            "estimated_usd": 1.0,
        },
    )
    assert cloud_training.training_source == "amphilens-pretrained"
    assert cloud_training.hosted_model_id == "amphilens-yolo26-m"
    assert cloud_training.image_size == 1152


def test_macos_picker_uses_the_native_folder_dialog(monkeypatch):
    monkeypatch.setattr(webapp.sys, "platform", "darwin")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, stdout="/Users/example/images/\n", stderr="")

    monkeypatch.setattr(webapp.subprocess, "run", run)

    selected = webapp._native_path_picker("directory")

    assert selected == "/Users/example/images"
    assert calls[0][0][0] == "osascript"
    assert "choose folder" in calls[0][0][-1]


def test_macos_picker_treats_cancel_as_no_selection(monkeypatch):
    monkeypatch.setattr(webapp.sys, "platform", "darwin")

    def run(command, **kwargs):
        return CompletedProcess(command, 1, stdout="", stderr="User canceled. (-128)")

    monkeypatch.setattr(webapp.subprocess, "run", run)

    assert webapp._native_path_picker("file") is None


def test_windows_picker_uses_a_sta_windows_forms_dialog(monkeypatch):
    monkeypatch.setattr(webapp.sys, "platform", "win32")
    monkeypatch.setattr(webapp.shutil, "which", lambda name: "powershell.exe")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, stdout="C:\\Users\\example\\images\r\n", stderr="")

    monkeypatch.setattr(webapp.subprocess, "run", run)

    selected = webapp._native_path_picker("directory")

    assert selected == "C:\\Users\\example\\images"
    command, kwargs = calls[0]
    assert command[:3] == ["powershell.exe", "-NoProfile", "-STA"]
    assert "System.Windows.Forms.FolderBrowserDialog" in command[-1]
    assert kwargs["timeout"] == 120


@pytest.mark.parametrize(
    ("picker", "expected_args"),
    [
        ("zenity", ["--file-selection", "--title=AmphiLens", "--save", "--confirm-overwrite"]),
        ("kdialog", ["--getsavefilename", "selection_queue.csv"]),
    ],
)
def test_linux_picker_uses_an_available_native_dialog(monkeypatch, picker, expected_args):
    monkeypatch.setattr(webapp.sys, "platform", "linux")
    monkeypatch.setattr(
        webapp.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name == picker else None,
    )
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, stdout="/tmp/selection_queue.csv\n", stderr="")

    monkeypatch.setattr(webapp.subprocess, "run", run)

    selected = webapp._native_path_picker("save-file")

    assert selected == "/tmp/selection_queue.csv"
    command, _ = calls[0]
    assert command[0] == f"/usr/bin/{picker}"
    assert all(argument in command for argument in expected_args)


def test_linux_picker_explains_when_no_native_dialog_is_installed(monkeypatch):
    monkeypatch.setattr(webapp.sys, "platform", "linux")
    monkeypatch.setattr(webapp.shutil, "which", lambda name: None)

    with pytest.raises(webapp.LocalPathPickerError, match="install zenity or kdialog"):
        webapp._native_path_picker("directory")


def test_bootstrap_and_project_create_open_close_persist(client, tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    project_path = tmp_path / "projects" / "toads"

    bootstrap = client.get("/api/bootstrap").json()
    assert bootstrap["project"] is None
    assert bootstrap["doctor"] == {"ok": True}
    assert bootstrap["default_project_root"].endswith("AmphiLens/projects")
    assert {model["model_id"] for model in bootstrap["models"]} == {
        "yolo26-l",
        "rtdetr-l",
        "faster-rcnn-resnet50",
    }
    assert bootstrap["default_model_id"] == "yolo26-l"
    assert {item["model_id"] for item in bootstrap["hosted_models"]} == {
        "amphilens-yolo26-m",
        "amphilens-rtdetr-l",
        "amphilens-faster-rcnn-resnet50",
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


def test_prediction_job_result_and_downloads_are_allowlisted(client, monkeypatch, tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    created = client.post(
        "/api/projects/create",
        json={
            "name": "Toads",
            "path": str(tmp_path / "project"),
            "image_root": str(image_root),
            "classes": ["toad"],
        },
    )
    assert created.status_code == 200, created.text

    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text("image_path,class_name,confidence\n", encoding="utf-8")
    monkeypatch.setattr(
        "amphilens.webapp.run_prediction_job",
        lambda *args, **kwargs: JobOutput(
            result={"message": "Detection complete", "paths": {"csv": str(csv_path)}},
            downloads=[
                DownloadArtifact(csv_path, "Predictions CSV", "predictions.csv", "text/csv")
            ],
        ),
    )
    started = client.post("/api/predictions", json={"confidence": 0.4, "device": "cpu"})
    assert started.status_code == 200, started.text
    job = _wait_for_job(client, started.json()["job_id"])
    assert job["state"] == "completed"
    assert job["result"]["message"] == "Detection complete"
    download = job["result"]["downloads"][0]
    assert download["label"] == "Predictions CSV"
    response = client.get(download["url"])
    assert response.status_code == 200
    assert response.text.startswith("image_path")
    assert client.get(f"/api/jobs/{job['job_id']}/downloads/secret.csv").status_code == 404


def test_hosted_prediction_maps_labels_and_records_revision(client, tmp_path, monkeypatch):
    import json

    from PIL import Image

    from amphilens.core import DetectionRecord
    from amphilens.webapp import PredictionsRequest, run_prediction_job

    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "sample.jpg"
    Image.new("RGB", (100, 80), color="gray").save(image_path)
    project_path = tmp_path / "project"
    created = client.post(
        "/api/projects/create",
        json={
            "name": "Toads",
            "path": str(project_path),
            "image_root": str(image_root),
            "classes": ["toad", "mammal"],
        },
    )
    assert created.status_code == 200, created.text
    store = webapp.open_project(project_path)
    checkpoint = tmp_path / "hosted.pt"
    checkpoint.write_bytes(b"fixture")
    observed = {}

    class FakeDetector:
        model_id = "amphilens-yolo26-m"

        def predict(self, paths, config):
            observed["image_size"] = config.image_size
            path = next(iter(paths))
            yield DetectionRecord(
                image_path=str(path),
                image_id=path.name,
                class_id=2,
                class_name="Western_Leopard_Toad",
                confidence=0.91,
                bbox_xyxy=[10, 15, 30, 35],
                image_width=100,
                image_height=80,
                model_id=self.model_id,
                run_id=config.run_id,
                preprocessing=config.preprocessing_fingerprint,
            )

    import amphilens.models as models
    import amphilens.models.hosted_models as hosted

    monkeypatch.setattr(hosted, "download_hosted_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(models, "load_detector", lambda *args, **kwargs: FakeDetector())
    output = run_prediction_job(
        store,
        PredictionsRequest(
            image_root=str(image_root),
            output_dir=str(tmp_path / "output"),
            hosted_model_id="amphilens-yolo26-m",
            class_mapping={
                "Other_Amphibian": None,
                "Small_Mammal": "mammal",
                "Western_Leopard_Toad": "toad",
            },
        ),
    )

    run_manifest = json.loads((tmp_path / "output" / "run.json").read_text())
    predictions = (tmp_path / "output" / "predictions.csv").read_text()
    assert observed["image_size"] == 1152
    assert ",toad," in predictions
    assert run_manifest["config"]["metadata"]["class_mapping"]["Other_Amphibian"] is None
    assert run_manifest["config"]["metadata"]["effective_configuration"]["hosted_model"]["revision"]
    assert output.result["message"].startswith("Processed 1 images")


def test_hosted_training_uses_selected_architecture_checkpoint_and_target_classes(
    client, tmp_path, monkeypatch
):
    from pathlib import Path
    from types import SimpleNamespace

    from amphilens.webapp import TrainingRequest, run_training_job

    image_root = tmp_path / "images"
    image_root.mkdir()
    project_path = tmp_path / "project"
    created = client.post(
        "/api/projects/create",
        json={
            "name": "Toads",
            "path": str(project_path),
            "image_root": str(image_root),
            "classes": ["toad", "mammal"],
        },
    )
    assert created.status_code == 200, created.text
    store = webapp.open_project(project_path)
    hosted_checkpoint = tmp_path / "hosted.pt"
    hosted_checkpoint.write_bytes(b"fixture")
    trained_checkpoint = tmp_path / "best.pt"
    trained_checkpoint.write_bytes(b"trained")
    observed = {}

    import amphilens.models as models
    import amphilens.models.hosted_models as hosted
    import amphilens.training as training

    monkeypatch.setattr(
        hosted, "download_hosted_checkpoint", lambda *args, **kwargs: hosted_checkpoint
    )
    monkeypatch.setattr(webapp, "_project_snapshot", lambda *_args, **_kwargs: object())

    def load_detector(checkpoint, **kwargs):
        observed["checkpoint"] = checkpoint
        observed.update(kwargs)
        return object()

    def train_snapshot_and_register(detector, *, config, output_dir, **kwargs):
        observed["effective"] = config.metadata["effective_configuration"]
        manifest_path = Path(output_dir) / "checkpoint.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(checkpoint=trained_checkpoint)

    monkeypatch.setattr(models, "load_detector", load_detector)
    monkeypatch.setattr(
        models,
        "load_preset_detector",
        lambda *_args, **_kwargs: pytest.fail("hosted source must load the hosted checkpoint"),
    )
    monkeypatch.setattr(training, "train_snapshot_and_register", train_snapshot_and_register)

    output = run_training_job(
        store,
        TrainingRequest(
            snapshot_path=str(tmp_path / "snapshot"),
            training_source="amphilens-pretrained",
            hosted_model_id="amphilens-yolo26-m",
            epochs=1,
        ),
    )

    assert observed["checkpoint"] == str(hosted_checkpoint)
    assert observed["architecture"] == "yolo"
    assert observed["classes"] == ["toad", "mammal"]
    assert observed["model_id"] == "amphilens-yolo26-m"
    assert observed["effective"]["image_size"] == 1152
    assert observed["effective"]["hosted_model"]["revision"]
    assert output.result["message"].startswith("Training finished from amphilens-pretrained:")


def test_job_errors_are_actionable_and_redact_credentials(client, jobs):
    job_id = jobs.submit(
        lambda: (_ for _ in ()).throw(RuntimeError("token=very-secret\nTraceback details"))
    )
    job = _wait_for_job(client, job_id)
    assert job["state"] == "failed"
    assert "very-secret" not in job["error"]
    assert "Traceback" not in job["error"]
    assert job["error"] == (
        "AmphiLens could not complete this request. Check your inputs and try again. "
        "If it continues, check the local terminal output."
    )


def test_unexpected_api_errors_are_caught_and_sanitized(client, monkeypatch):
    def fail_doctor(*args, **kwargs):
        raise RuntimeError("secret=private-token /Users/person/internal-config\nTraceback omitted")

    monkeypatch.setattr(webapp, "run_doctor", fail_doctor)

    response = client.get("/api/bootstrap")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "AmphiLens could not complete this request. Check your inputs and try again. "
        "If it continues, check the local terminal output."
    )
    assert "private-token" not in response.text
    assert "/Users/person" not in response.text
    assert "Traceback" not in response.text


def test_active_learning_missing_evidence_reports_clear_failure(client, tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    client.post(
        "/api/projects/create",
        json={
            "name": "Toads",
            "path": str(tmp_path / "project"),
            "image_root": str(image_root),
            "classes": ["toad"],
        },
    )
    started = client.post(
        "/api/active-learning/select",
        json={
            "predictions": str(tmp_path / "missing.csv"),
            "calibration": str(tmp_path / "missing-calibration.json"),
            "features": str(tmp_path / "missing-features.json"),
            "output": str(tmp_path / "queue"),
        },
    )
    assert started.status_code == 200
    job = _wait_for_job(client, started.json()["job_id"])
    assert job["state"] == "failed"
    assert "calibration" in job["error"].lower()
    assert str(tmp_path) not in job["error"]
    assert "traceback" not in job["error"].lower()
