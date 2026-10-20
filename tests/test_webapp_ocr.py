from __future__ import annotations

import csv
import sys
import threading

import pytest

pytest.importorskip("starlette")

from webapp_test_support import _wait_for_job

import amphilens.webapp as webapp
from amphilens.core import ProjectManifest, ProjectStore, RunManifest, atomic_write_json
from amphilens.webapp import JobOutput


def test_ocr_job_updates_prediction_csv_and_returns_fresh_download(
    client, jobs, tmp_path, monkeypatch
):
    import cv2
    import numpy as np

    import amphilens.timestamp_ocr as timestamp_ocr

    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((200, 300, 3), dtype=np.uint8))
    project_root = tmp_path / "project"
    project = ProjectStore(project_root)
    project.create(ProjectManifest.create("Survey", [image_root], ["toad"]))
    run_id = "run-ocr-test123"
    output_dir = project_root / "runs" / run_id
    output_dir.mkdir(parents=True)
    run = RunManifest.create("inference", {"model_id": "test"})
    run.run_id = run_id
    run.status = "completed"
    atomic_write_json(output_dir / "run.json", run.to_dict())
    predictions_csv = output_dir / "predictions.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "class_name"])
        writer.writeheader()
        writer.writerows(
            [
                {"image_path": str(image_path), "class_name": "toad"},
                {"image_path": str(image_path), "class_name": "frog"},
            ]
        )

    rapidocr_factory = timestamp_ocr.create_rapidocr_engine
    monkeypatch.setattr(
        timestamp_ocr,
        "create_rapidocr_engine",
        lambda *, device="cpu": lambda crop: "09/10/2026 18:42:07",
    )
    prediction_job = jobs.submit(
        lambda: JobOutput(
            result={
                "prediction": {
                    "run_id": run_id,
                    "csv": str(predictions_csv),
                    "image_root": str(image_root),
                    "project_root": str(project_root),
                    "detection_count": 2,
                }
            }
        )
    )
    assert _wait_for_job(client, prediction_job)["state"] == "completed"

    started = client.post(
        f"/api/jobs/{prediction_job}/ocr-timestamps",
        json={"csv": str(tmp_path / "untrusted.csv"), "image_root": str(tmp_path)},
    )
    assert started.status_code == 200, started.text
    ocr_job_id = started.json()["job_id"]
    ocr_job = _wait_for_job(client, ocr_job_id)
    assert ocr_job["state"] == "completed"
    assert ocr_job["progress"]["completed"] == 1
    assert ocr_job["progress"]["recognized"] == 1
    assert ocr_job["progress"]["unparsed"] == 0
    assert ocr_job["result"]["recognized_count"] == 1
    assert ocr_job["result"]["unparsed_count"] == 0
    assert len(ocr_job["result"]["downloads"]) == 1
    download = client.get(ocr_job["result"]["downloads"][0]["url"])
    assert download.status_code == 200
    assert b"date,time" in download.content
    with predictions_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["date"], row["time"]) for row in rows] == [
        ("2026-10-09", "18:42:07"),
        ("2026-10-09", "18:42:07"),
    ]

    updated_csv = predictions_csv.read_bytes()
    monkeypatch.setattr(timestamp_ocr, "create_rapidocr_engine", rapidocr_factory)
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    missing_extra = client.post(f"/api/jobs/{prediction_job}/ocr-timestamps", json={})
    assert missing_extra.status_code == 200
    dependency_job = _wait_for_job(client, missing_extra.json()["job_id"])
    assert dependency_job["state"] == "failed"
    assert "uv sync --extra web --extra inference --extra ocr" in dependency_job["error"]
    assert predictions_csv.read_bytes() == updated_csv


def test_cloud_ocr_requires_confirmation_and_sends_only_local_crops(
    client, jobs, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import cv2
    import numpy as np

    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((200, 300, 3), dtype=np.uint8))
    original_image = image_path.read_bytes()
    project_root = tmp_path / "project"
    project = ProjectStore(project_root)
    project.create(ProjectManifest.create("Survey", [image_root], ["toad"]))
    run_id = "run-cloud-ocr123"
    output_dir = project_root / "runs" / run_id
    output_dir.mkdir(parents=True)
    run = RunManifest.create("inference", {"model_id": "test"})
    run.run_id = run_id
    run.status = "completed"
    atomic_write_json(output_dir / "run.json", run.to_dict())
    predictions_csv = output_dir / "predictions.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "class_name"])
        writer.writeheader()
        writer.writerow({"image_path": str(image_path), "class_name": "toad"})

    class FakeTransport:
        submitted = None
        cleaned = []

        def run_timestamp_ocr(
            self,
            *,
            job_key,
            gpu,
            crops,
            progress_callback=None,
            cancellation_requested=None,
            job_id_callback=None,
        ):
            del progress_callback, cancellation_requested, job_id_callback
            self.submitted = (job_key, gpu, dict(crops))
            crop = cv2.imread(str(next(iter(crops.values()))))
            assert crop.shape[:2] == (56, 300)
            return {
                "state": "finished",
                "records": [
                    {
                        "image_id": image_id,
                        "date": "2026-10-09",
                        "time": "18:42:07",
                    }
                    for image_id in crops
                ],
            }

        def cleanup_timestamp_ocr(self, job_key):
            self.cleaned.append(job_key)

    transport = FakeTransport()
    monkeypatch.setattr(
        webapp,
        "_cloud_service",
        lambda *_args, **_kwargs: (SimpleNamespace(transport=transport), None),
    )
    prediction_job = jobs.submit(
        lambda: JobOutput(
            result={
                "prediction": {
                    "run_id": run_id,
                    "csv": str(predictions_csv),
                    "image_root": str(image_root),
                    "project_root": str(project_root),
                    "detection_count": 1,
                    "execution": "azure_ml",
                    "gpu": "T4",
                }
            }
        )
    )
    assert _wait_for_job(client, prediction_job)["state"] == "completed"

    unconfirmed = client.post(
        f"/api/jobs/{prediction_job}/ocr-timestamps", json={"acknowledged": False}
    )
    assert unconfirmed.status_code == 400
    assert "confirm cloud timestamp ocr" in unconfirmed.json()["detail"].lower()
    assert transport.submitted is None

    started = client.post(f"/api/jobs/{prediction_job}/ocr-timestamps", json={"acknowledged": True})
    assert started.status_code == 200, started.text
    ocr_job = _wait_for_job(client, started.json()["job_id"])
    assert ocr_job["state"] == "completed"
    assert transport.submitted[1] == "T4"
    assert len(transport.submitted[2]) == 1
    assert transport.cleaned == [transport.submitted[0]]
    assert "Only local bottom crops were sent to Azure ML (T4)" in ocr_job["result"]["message"]
    assert image_path.read_bytes() == original_image


def test_ocr_action_requires_completed_prediction_with_detections(client, jobs):
    gate = threading.Event()
    running_job = jobs.submit(lambda: gate.wait(1))
    try:
        response = client.post(f"/api/jobs/{running_job}/ocr-timestamps", json={})
        assert response.status_code == 409
        assert "completed prediction" in response.json()["detail"].lower()
    finally:
        gate.set()

    non_prediction = jobs.submit(lambda: {"message": "not a prediction"})
    assert _wait_for_job(client, non_prediction)["state"] == "completed"
    response = client.post(f"/api/jobs/{non_prediction}/ocr-timestamps", json={})
    assert response.status_code == 400
    assert "prediction results" in response.json()["detail"].lower()

    empty = jobs.submit(
        lambda: {
            "prediction": {
                "run_id": "run-no-detections",
                "csv": "unused.csv",
                "image_root": "unused-images",
                "project_root": "unused-project",
                "detection_count": 0,
            }
        }
    )
    assert _wait_for_job(client, empty)["state"] == "completed"
    response = client.post(f"/api/jobs/{empty}/ocr-timestamps", json={})
    assert response.status_code == 400
    assert "no detections" in response.json()["detail"].lower()
