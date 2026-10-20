from __future__ import annotations

import time

import pytest

pytest.importorskip("starlette")

from webapp_test_support import _wait_for_job

import amphilens.webapp as webapp
from amphilens.core import ProjectManifest, ProjectStore
from amphilens.runs import InferenceInterruption
from amphilens.webapp import DownloadArtifact, JobOutput


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


def test_dataset_import_routes_run_progress_aware_jobs(client, tmp_path, monkeypatch):
    image_root = tmp_path / "images"
    image_root.mkdir()
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("study", [image_root], ["toad"]))
    client.app.state.active_project_path = store.root
    client.app.state.project_restored = True

    def reportable_job(_store, _request, progress_callback):
        progress_callback(
            {
                "phase": "validating-images",
                "message": "Validated 1 of 1 images",
                "completed": 1,
                "total": 1,
                "progress": 1.0,
                "phase_progress": 1.0,
            }
        )
        return {"message": "Import complete"}

    monkeypatch.setattr(webapp, "run_dataset_import_job", reportable_job)
    monkeypatch.setattr(webapp, "run_cvat_import_job", reportable_job)

    for endpoint, payload in (
        ("/api/datasets/import", {"archive": str(tmp_path / "dataset.zip")}),
        ("/api/datasets/import-cvat", {"project_id": "7"}),
    ):
        response = client.post(endpoint, json=payload)
        assert response.status_code == 200
        job = _wait_for_job(client, response.json()["job_id"])
        assert job["state"] == "completed"
        assert job["progress"]["message"] == "Validated 1 of 1 images"
        assert job["progress"]["phase_progress"] == 1.0


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
