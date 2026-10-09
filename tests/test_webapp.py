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
    assert '<span class="nav-label">Choose images to review</span>' in page.text
    assert client.get("/static/prediction_state.js").status_code == 200
    assert client.get("/static/activity_view.js").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert 'api("/api/path-picker"' in client.get("/static/app.js").text
    script = client.get("/static/app.js").text
    assert 'name="run_name" type="text" maxlength="120"' in script
    assert "run_name: values.run_name" in script
    assert "Prediction run:" in script
    assert "Collect detected images" in script
    assert "/collect-images" in script
    assert 'detectionCount <= 0 ? "disabled" : ""' in script
    assert "There are no detected images to collect." in script
    assert 'name="training_source"' in script
    assert 'name="hosted_model_id"' in script
    assert 'app.cvatServerUrl = data.cvat_server_url || app.cvatServerUrl || "";' in script
    assert 'value="${escapeHtml(app.cvatServerUrl)}"' in script
    assert "__ignore__" in script
    assert 'app.doctor?.mps_available ? `<option value="mps">Apple MPS GPU</option>` : ""' in script
    assert "model card revision recorded by AmphiLens" not in script
    assert "CVAT credentials are not stored in your project." not in script
    assert "Credentials must be configured locally before connecting." not in script
    assert "Credentials are sent only to the local AmphiLens service" not in script
    assert "private Hugging Face repository" not in script
    stylesheet = client.get("/static/app.css")
    assert stylesheet.status_code == 200
    assert (
        ".nav-label { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; }"
        in stylesheet.text
    )
    assert client.get("/static/favicon.svg").status_code == 200


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
            "run_name": "Hosted inference",
            "hosted_model_id": "amphilens-yolo26-m",
            "class_mapping": {"Other_Amphibian": None},
        },
    )
    training = webapp._request(
        webapp.TrainingRequest,
        {
            "snapshot_path": "/project/dataset",
            "device": "mps",
            "training_source": "amphilens-pretrained",
            "hosted_model_id": "amphilens-yolo26-m",
        },
    )

    assert prediction.hosted_model_id == "amphilens-yolo26-m"
    assert prediction.class_mapping == {"Other_Amphibian": None}
    assert training.training_source == "amphilens-pretrained"
    assert training.hosted_model_id == "amphilens-yolo26-m"
    assert training.device == "mps"
    assert webapp.select_training_source(None, None, None) == "general-pretrained"


def test_prediction_request_parses_local_gpu_batching_and_modal_execution():
    local = webapp._request(
        webapp.PredictionsRequest,
        {"run_name": "Local inference", "device": "mps", "batch_size": "auto"},
    )
    cloud = webapp._request(
        webapp.PredictionsRequest,
        {
            "run_name": "Cloud inference",
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


def test_prediction_request_requires_a_trimmed_run_name_of_at_most_120_characters():
    with pytest.raises(ValueError, match="run_name is required"):
        webapp._request(webapp.PredictionsRequest, {})
    with pytest.raises(ValueError, match="run_name is required"):
        webapp._request(webapp.PredictionsRequest, {"run_name": "   "})

    request = webapp._request(webapp.PredictionsRequest, {"run_name": "  Spring survey  "})
    assert request.run_name == "Spring survey"

    with pytest.raises(ValueError, match="120 characters"):
        webapp._request(webapp.PredictionsRequest, {"run_name": "x" * 121})


def test_prediction_job_can_collect_unique_images_into_its_project_run_folder(
    client, jobs, tmp_path
):
    import csv

    image_root = tmp_path / "images"
    image = image_root / "camera" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"source image")
    project_root = tmp_path / "project"
    created = client.post(
        "/api/projects/create",
        json={
            "name": "Survey",
            "path": str(project_root),
            "image_root": str(image_root),
            "classes": ["toad"],
        },
    )
    assert created.status_code == 200, created.text

    predictions_csv = tmp_path / "custom-results" / "predictions.csv"
    predictions_csv.parent.mkdir()
    with predictions_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path"])
        writer.writeheader()
        writer.writerows([{"image_path": str(image)}, {"image_path": str(image)}])

    # Collection uses server-owned job data, not paths from the browser.
    job_id = jobs.submit(
        lambda: JobOutput(
            result={
                "prediction": {
                    "run_id": "predict-yolo26-l-test123",
                    "csv": str(predictions_csv),
                    "image_root": str(image_root),
                    "project_root": str(project_root),
                    "detection_count": 2,
                }
            }
        )
    )
    assert _wait_for_job(client, job_id)["state"] == "completed"

    response = client.post(f"/api/jobs/{job_id}/collect-images")
    assert response.status_code == 200, response.text
    result = response.json()
    destination = project_root / "runs" / "predict-yolo26-l-test123" / "images"
    assert result["folder"] == str(destination)
    assert result["copied_count"] == 1
    assert result["already_present_count"] == 0
    assert result["missing_count"] == 0
    assert (destination / "camera" / "frame.jpg").read_bytes() == b"source image"
    assert image.read_bytes() == b"source image"

    repeated = client.post(f"/api/jobs/{job_id}/collect-images")
    assert repeated.status_code == 200, repeated.text
    assert repeated.json()["copied_count"] == 0
    assert repeated.json()["already_present_count"] == 1

    empty_job_id = jobs.submit(
        lambda: JobOutput(
            result={
                "prediction": {
                    "run_id": "predict-yolo26-l-empty123",
                    "csv": str(predictions_csv),
                    "image_root": str(image_root),
                    "project_root": str(project_root),
                    "detection_count": 0,
                }
            }
        )
    )
    assert _wait_for_job(client, empty_job_id)["state"] == "completed"
    no_detections = client.post(f"/api/jobs/{empty_job_id}/collect-images")
    assert no_detections.status_code == 400
    assert "no detected images" in no_detections.json()["detail"].lower()


def test_prediction_request_rejects_unsupported_modal_gpu_and_batch_size():
    with pytest.raises(ValueError, match="Unsupported Modal GPU"):
        webapp._request(
            webapp.PredictionsRequest,
            {"run_name": "Test", "execution": "modal", "gpu": "invented"},
        )
    with pytest.raises(ValueError, match="between 1 and 32"):
        webapp._request(webapp.PredictionsRequest, {"run_name": "Test", "batch_size": 33})
    with pytest.raises(ValueError, match="spending limit"):
        webapp._request(webapp.PredictionsRequest, {"run_name": "Test", "execution": "modal"})


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
        json={"run_name": "Consent test", "execution": "modal", "max_cost_usd": 1.0},
    )

    assert response.status_code == 400
    assert "consent" in response.json()["detail"].lower()


def test_cloud_estimate_uses_project_input_size_for_hosted_model():
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
    assert webapp._cloud_estimate_image_size(estimate) == 640

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
    started = client.post(
        "/api/predictions",
        json={"run_name": "Test prediction", "confidence": 0.4, "device": "cpu"},
    )
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
            run_name="Hosted model traceability",
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
    other_image_root = tmp_path / "other-images"
    other_image_root.mkdir()
    Image.new("RGB", (100, 80), color="gray").save(other_image_root / "sample.jpg")
    other_output = run_prediction_job(
        store,
        PredictionsRequest(
            run_name="Hosted model traceability",
            image_root=str(other_image_root),
            output_dir=str(tmp_path / "other-output"),
            hosted_model_id="amphilens-yolo26-m",
            class_mapping={
                "Other_Amphibian": None,
                "Small_Mammal": "mammal",
                "Western_Leopard_Toad": "toad",
            },
        ),
    )
    assert output.result["prediction"]["run_id"] != other_output.result["prediction"]["run_id"]

    run_manifest = json.loads((tmp_path / "output" / "run.json").read_text())
    predictions_path = tmp_path / "output" / "predictions.csv"
    predictions = predictions_path.read_text()
    assert observed["image_size"] == 640
    assert ",toad," in predictions
    assert run_manifest["config"]["metadata"]["run_name"] == "Hosted model traceability"
    assert run_manifest["config"]["metadata"]["class_mapping"]["Other_Amphibian"] is None
    assert run_manifest["config"]["metadata"]["effective_configuration"]["hosted_model"]["revision"]
    import csv

    with predictions_path.open(newline="", encoding="utf-8") as handle:
        prediction_row = next(csv.DictReader(handle))
    assert prediction_row["run_name"] == "Hosted model traceability"
    assert prediction_row["model_id"] == "amphilens-yolo26-m"
    assert output.result["prediction"]["run_id"] == prediction_row["run_id"]
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
    monkeypatch.setattr(
        webapp,
        "_training_snapshots",
        lambda *_args, **_kwargs: (object(), None, None),
    )

    def load_detector(checkpoint, **kwargs):
        observed["checkpoint"] = checkpoint
        observed.update(kwargs)
        return object()

    def train_snapshot_and_register(detector, *, config, output_dir, **kwargs):
        observed["effective"] = config.metadata["effective_configuration"]
        manifest_path = Path(output_dir) / "checkpoint.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            checkpoint=trained_checkpoint,
            manifest=SimpleNamespace(training_config={"evaluation": "not evaluated"}),
        )

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
    assert observed["effective"]["image_size"] == 640
    assert observed["effective"]["hosted_model"]["revision"]
    assert output.result["message"].startswith("Training finished from amphilens-pretrained:")


def test_cloud_hosted_training_defers_checkpoint_download_to_remote_worker(monkeypatch):
    import amphilens.models.hosted_models as hosted
    from amphilens.webapp import CloudTrainingRequest

    monkeypatch.setattr(
        hosted,
        "download_hosted_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("cloud training must download weights remotely"),
    )

    source, model, checkpoint, manifest = webapp._training_checkpoint(
        CloudTrainingRequest(
            snapshot_path="unused",
            training_source="amphilens-pretrained",
            hosted_model_id="amphilens-yolo26-m",
        ),
        download_hosted=False,
    )

    assert source == "amphilens-pretrained"
    assert model.model_id == "amphilens-yolo26-m"
    assert checkpoint == ""
    assert manifest is None


def test_cloud_training_submits_hosted_model_without_local_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from amphilens.models.hosted_models import get_hosted_model
    from amphilens.webapp import CloudTrainingRequest, run_cloud_training_job

    hosted = get_hosted_model("amphilens-yolo26-m")
    project_path = tmp_path / "project"
    image_root = tmp_path / "images"
    image_root.mkdir()
    store = ProjectStore(project_path)
    store.create(ProjectManifest.create("study", [image_root], list(hosted.source_classes)))
    snapshots = tuple(
        SimpleNamespace(
            root=tmp_path / snapshot_id,
            manifest=SimpleNamespace(snapshot_id=snapshot_id, classes=list(hosted.source_classes))
        )
        for snapshot_id in ("train-id", "validation-id", "test-id")
    )
    observed = {}

    class FakeService:
        def submit(self, **kwargs):
            observed.update(kwargs)
            return SimpleNamespace(run_id="cloud-test", to_dict=lambda: {"run_id": "cloud-test"})

    monkeypatch.setattr(webapp, "_cloud_service", lambda *_args: (FakeService(), object()))
    monkeypatch.setattr(webapp, "_training_snapshots", lambda *_args: snapshots)

    import amphilens.models.hosted_models as hosted_module

    monkeypatch.setattr(
        hosted_module,
        "download_hosted_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("Modal must fetch public weights remotely"),
    )

    response = run_cloud_training_job(
        store,
        CloudTrainingRequest(
            snapshot_path=str(tmp_path / "train"),
            gpu="T4",
            epochs=1,
            image_size=640,
            training_source="amphilens-pretrained",
            hosted_model_id=hosted.model_id,
            validation_snapshot_path=str(tmp_path / "validation"),
            test_snapshot_path=str(tmp_path / "test"),
            patience=25,
            batch_size=1,
            estimated_usd=0.04,
            max_cost_usd=2.0 / 3.0,
            acknowledged=True,
            uploads_dataset=True,
        ),
        object(),
    )

    assert response["job"]["run_id"] == "cloud-test"
    assert observed["base_checkpoint"] is None
    assert observed["base_manifest"] is None
    assert observed["effective_configuration"]["hosted_model"]["revision"]


def test_hosted_modal_prediction_sends_project_preprocessing(tmp_path, monkeypatch):
    from PIL import Image

    from amphilens.core import ProjectConfig, ProjectManifest, ProjectStore
    from amphilens.models.hosted_models import get_hosted_model
    from amphilens.preprocessing import PreprocessingConfig
    from amphilens.webapp import PredictionsRequest, run_prediction_job

    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "sample.jpg"
    Image.new("RGB", (80, 40), color="gray").save(image_path)
    hosted = get_hosted_model("amphilens-yolo26-m")
    project = ProjectStore(tmp_path / "project")
    project.create(
        ProjectManifest.create(
            "study",
            [image_root],
            list(hosted.source_classes),
            project_config=ProjectConfig(
                classes=list(hosted.source_classes),
                preprocessing=PreprocessingConfig(short_side_dimension=512, clahe_enabled=True),
            ),
        )
    )
    observed = {}

    class FakeModalDetector:
        device_name = "Modal GPU"
        estimated_cost_usd = 0.0

        def __init__(self, _transport, *, model_spec, **_kwargs):
            observed["model_spec"] = model_spec

        def predict(self, _paths, _config):
            return iter(())

        def cleanup_progress(self):
            pass

    import amphilens.cloud.prediction as cloud_prediction

    monkeypatch.setattr(cloud_prediction, "ModalPredictionDetector", FakeModalDetector)
    run_prediction_job(
        project,
        PredictionsRequest(
            run_name="Modal image prediction",
            image_root=str(image_root),
            output_dir=str(tmp_path / "output"),
            execution="modal",
            hosted_model_id=hosted.model_id,
            max_cost_usd=1.0,
            acknowledged=True,
            uploads_dataset=True,
            expected_image_count=1,
            expected_total_bytes=image_path.stat().st_size,
        ),
        cloud_transport=object(),
    )

    assert observed["model_spec"]["preprocessing"]["short_side_dimension"] == 512
    assert observed["model_spec"]["preprocessing"]["clahe_enabled"] is True


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
