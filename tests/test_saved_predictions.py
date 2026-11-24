from __future__ import annotations

import csv
import shutil
import threading
from types import SimpleNamespace

from amphilens.core import ProjectManifest, ProjectStore, RunManifest, atomic_write_json


def _run(folder, images, name, *, timestamps=False, detections=1):
    folder.mkdir(parents=True)
    manifest = RunManifest.create(
        "inference",
        {
            "model_id": "model",
            "device": "cpu",
            "metadata": {"run_name": name, "image_root": str(images), "execution": "local"},
        },
    )
    manifest.status = "completed_with_skips"
    atomic_write_json(folder / "run.json", manifest.to_dict())
    atomic_write_json(
        folder / "summary.json",
        {
            "run_id": manifest.run_id,
            "image_count": 2,
            "completed_images": 1,
            "detection_count": detections,
            "failed_images": {},
            "skipped_images": {"b": "bad"},
        },
    )
    with (folder / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_path"] + (["date", "time"] if timestamps else [])
        )
        writer.writeheader()
        if detections:
            writer.writerow({"image_path": str(images / "a.jpg")})


def test_saved_runs_restore_actions_across_app_restart(client, tmp_path):
    from amphilens.webapp_saved_predictions import saved_predictions

    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"image")
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("Project", [], ["toad"]))
    first = store.root / "runs" / "first"
    custom = tmp_path / "custom"
    _run(first, images, "first", timestamps=True)
    _run(custom, images, "custom", detections=0)
    client.app.state.active_project_path = store.root
    client.app.state.project_restored = True

    first_job = client.post("/api/prediction-runs/open", json={"path": str(first)}).json()["job"]
    prediction = first_job["result"]["prediction"]
    assert prediction["timestamps_extracted"] and prediction["timestamp_unresolved_rows"] == 1
    assert client.get(first_job["result"]["downloads"][0]["url"]).status_code == 200
    response = client.post(f"/api/jobs/{first_job['job_id']}/collect-images", json={})
    assert response.json()["copied_count"] == 1
    assert (store.root / "runs" / "first" / "images" / "a.jpg").is_file()

    response = client.post("/api/prediction-runs/open", json={"path": str(custom), "add": True})
    assert response.json()["job"]["result"]["prediction"]["detection_count"] == 0
    assert {item["name"] for item in saved_predictions(ProjectStore(store.root))} == {
        "first",
        "custom",
    }
    client.app.state.jobs = type(client.app.state.jobs)()
    reopened = client.post("/api/prediction-runs/open", json={"path": str(custom)})
    assert reopened.status_code == 200
    client.app.state.jobs.shutdown()

    (first / "predictions.csv").unlink()
    assert (
        "missing or invalid"
        in client.post("/api/prediction-runs/open", json={"path": str(first)}).json()["detail"]
    )
    images.rename(tmp_path / "moved")
    from amphilens.webapp_saved_predictions import restore_prediction

    assert not restore_prediction(store, custom).result["prediction"]["image_root_available"]


def test_duplicate_names_cannot_mix_collected_images(client, tmp_path):
    import pytest

    from amphilens.webapp_contracts import _ensure_prediction_run_name_is_unique
    from amphilens.webapp_saved_predictions import remember_prediction

    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"image")
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("Project", [], ["toad"]))
    first, second = tmp_path / "one", tmp_path / "two"
    _run(first, images, "same")
    _run(second, images, "same")
    remember_prediction(store, first)
    with pytest.raises(ValueError, match="already been used"):
        _ensure_prediction_run_name_is_unique(store, "same")
    remember_prediction(store, second)
    client.app.state.active_project_path = store.root
    client.app.state.project_restored = True
    job = client.post("/api/prediction-runs/open", json={"path": str(first)}).json()["job"]
    response = client.post(f"/api/jobs/{job['job_id']}/collect-images", json={})
    assert response.status_code == 400
    assert "Multiple prediction runs" in response.json()["detail"]
    assert not (store.root / "runs" / "same" / "images").exists()


def test_indexed_run_inside_project_follows_project_relocation(tmp_path):
    from amphilens.webapp_saved_predictions import remember_prediction, saved_predictions

    images = tmp_path / "images"
    images.mkdir()
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("Project", [], ["toad"]))
    run = store.root / "custom" / "run"
    _run(run, images, "saved")
    remember_prediction(store, run)
    relocated = tmp_path / "relocated"
    shutil.copytree(store.root, relocated)
    assert saved_predictions(ProjectStore(relocated))[0]["path"] == str(
        relocated / "custom" / "run"
    )


def test_reopened_run_cannot_start_concurrent_timestamp_extraction(client, tmp_path, monkeypatch):
    import amphilens.timestamp_ocr as ocr

    images = tmp_path / "images"
    images.mkdir()
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("Project", [], ["toad"]))
    run = store.root / "runs" / "first"
    _run(run, images, "first")
    client.app.state.active_project_path = store.root
    client.app.state.project_restored = True
    first = client.post("/api/prediction-runs/open", json={"path": str(run)}).json()["job"]
    second = client.post("/api/prediction-runs/open", json={"path": str(run)}).json()["job"]
    started, release = threading.Event(), threading.Event()

    def extract(csv_path, image_root, **kwargs):
        started.set()
        release.wait(5)
        return SimpleNamespace(
            csv_path=csv_path, recognized_count=0, image_count=1, unparsed_count=1
        )

    monkeypatch.setattr(ocr, "process_prediction_csv", extract)
    try:
        response = client.post(f"/api/jobs/{first['job_id']}/ocr-timestamps", json={})
        assert response.status_code == 200
        assert started.wait(2)
        response = client.post(f"/api/jobs/{second['job_id']}/ocr-timestamps", json={})
        assert response.status_code == 400
        assert "already running" in response.json()["detail"]
    finally:
        release.set()


def test_missing_default_run_metadata_is_visible(tmp_path):
    from amphilens.webapp_saved_predictions import saved_predictions

    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("Project", [], ["toad"]))
    folder = store.root / "runs" / "damaged"
    folder.mkdir()
    (folder / "predictions.csv").write_text("image_path\n")
    run = saved_predictions(store)[0]
    assert run["name"] == "damaged"
    assert "metadata is missing" in run["error"]


def test_interrupted_run_is_not_listed_as_damaged(tmp_path):
    from amphilens.webapp_saved_predictions import saved_predictions

    images = tmp_path / "images"
    images.mkdir()
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("Project", [], ["toad"]))
    folder = store.root / "runs" / "interrupted"
    _run(folder, images, "interrupted")
    import json

    manifest = json.loads((folder / "run.json").read_text())
    manifest["status"] = "interrupted"
    atomic_write_json(folder / "run.json", manifest)
    assert saved_predictions(store) == []


def test_same_prediction_name_cannot_run_concurrently(client, tmp_path, monkeypatch):
    import amphilens.webapp as webapp

    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("Project", [], ["toad"]))
    client.app.state.active_project_path = store.root
    client.app.state.project_restored = True
    started, release = threading.Event(), threading.Event()

    def run(*args, **kwargs):
        started.set()
        release.wait(5)
        return {}

    monkeypatch.setattr(webapp, "run_prediction_job", run)
    try:
        response = client.post(
            "/api/predictions", json={"run_name": "same", "output_dir": "/tmp/a"}
        )
        assert response.status_code == 200
        assert started.wait(2)
        response = client.post(
            "/api/predictions", json={"run_name": "same", "output_dir": "/tmp/b"}
        )
        assert response.status_code == 400
        assert "already running" in response.json()["detail"]
    finally:
        release.set()
