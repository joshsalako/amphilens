from __future__ import annotations

import pytest

pytest.importorskip("starlette")

from webapp_test_support import _wait_for_job

import amphilens.webapp as webapp
from amphilens.webapp import JobOutput


def test_api_parses_hosted_inference_and_training_source_fields():
    prediction = webapp._request(
        webapp.PredictionsRequest,
        {
            "run_name": "Hosted inference",
            "hosted_model_id": "amphilens-yolo26-m",
            "class_mapping": {"Other_Amphibian": None},
            "preprocessing": {
                "short_side_dimension": 384,
                "grayscale_enabled": False,
                "clahe_enabled": True,
            },
        },
    )
    training = webapp._request(
        webapp.TrainingRequest,
        {
            "snapshot_path": "/project/dataset",
            "device": "mps",
            "training_source": "amphilens-pretrained",
            "hosted_model_id": "amphilens-yolo26-m",
            "model_preset": "rtdetr-l",
            "preprocessing": {
                "short_side_dimension": 512,
                "grayscale_enabled": True,
                "clahe_enabled": False,
            },
        },
    )

    assert prediction.hosted_model_id == "amphilens-yolo26-m"
    assert prediction.class_mapping == {"Other_Amphibian": None}
    assert prediction.preprocessing["short_side_dimension"] == 384
    assert training.training_source == "amphilens-pretrained"
    assert training.hosted_model_id == "amphilens-yolo26-m"
    assert training.model_preset == "rtdetr-l"
    assert training.preprocessing["short_side_dimension"] == 512
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
                    "run_name": "Field Camera Survey",
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
            preprocessing={
                "short_side_dimension": 384,
                "grayscale_enabled": False,
                "clahe_enabled": False,
            },
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
    assert observed["image_size"] == 1152
    assert ",toad," in predictions
    assert run_manifest["config"]["metadata"]["run_name"] == "Hosted model traceability"
    assert run_manifest["config"]["metadata"]["class_mapping"]["Other_Amphibian"] is None
    assert run_manifest["config"]["metadata"]["effective_configuration"]["hosted_model"]["revision"]
    effective_preprocessing = run_manifest["config"]["metadata"]["effective_configuration"][
        "preprocessing"
    ]
    assert effective_preprocessing["short_side_dimension"] == 384
    assert effective_preprocessing["grayscale_enabled"] is False
    assert effective_preprocessing["clahe_enabled"] is False
    import csv

    with predictions_path.open(newline="", encoding="utf-8") as handle:
        prediction_row = next(csv.DictReader(handle))
    assert prediction_row["run_name"] == "Hosted model traceability"
    assert prediction_row["model_id"] == "amphilens-yolo26-m"
    assert output.result["prediction"]["run_id"] == prediction_row["run_id"]
    assert output.result["message"].startswith("Processed 1 images")


def test_hosted_modal_prediction_sends_run_preprocessing(tmp_path, monkeypatch):
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
            preprocessing={
                "short_side_dimension": 384,
                "grayscale_enabled": False,
                "clahe_enabled": False,
            },
            max_cost_usd=1.0,
            acknowledged=True,
            uploads_dataset=True,
            expected_image_count=1,
            expected_total_bytes=image_path.stat().st_size,
        ),
        cloud_transport=object(),
    )

    assert observed["model_spec"]["preprocessing"]["short_side_dimension"] == 384
    assert observed["model_spec"]["preprocessing"]["grayscale_enabled"] is False
    assert observed["model_spec"]["preprocessing"]["clahe_enabled"] is False
