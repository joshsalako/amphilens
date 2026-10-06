from __future__ import annotations

import time
from subprocess import CompletedProcess

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient

import amphilens.webapp as webapp
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
    assert client.get("/static/app.js").status_code == 200
    assert 'api("/api/path-picker"' in client.get("/static/app.js").text
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
