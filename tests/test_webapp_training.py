from __future__ import annotations

import pytest

pytest.importorskip("starlette")

import amphilens.webapp as webapp
from amphilens.core import ProjectManifest, ProjectStore


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
            preprocessing={
                "short_side_dimension": 384,
                "grayscale_enabled": False,
                "clahe_enabled": False,
            },
            epochs=1,
        ),
    )

    assert observed["checkpoint"] == str(hosted_checkpoint)
    assert observed["architecture"] == "yolo"
    assert observed["classes"] == ["toad", "mammal"]
    assert observed["model_id"] == "amphilens-yolo26-m"
    assert observed["effective"]["classes"] == ["toad", "mammal"]
    assert observed["effective"]["hosted_model"]["source_class_order"] == list(
        hosted.get_hosted_model("amphilens-yolo26-m").source_classes
    )
    assert observed["effective"]["image_size"] == 1152
    assert observed["effective"]["preprocessing"]["short_side_dimension"] == 384
    assert observed["effective"]["preprocessing"]["grayscale_enabled"] is False
    assert observed["effective"]["preprocessing"]["clahe_enabled"] is False
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
            manifest=SimpleNamespace(snapshot_id=snapshot_id, classes=list(hosted.source_classes)),
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
            preprocessing={
                "short_side_dimension": 384,
                "grayscale_enabled": False,
                "clahe_enabled": False,
            },
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
    assert observed["effective_configuration"]["image_size"] == 1152
    assert observed["effective_configuration"]["preprocessing"]["short_side_dimension"] == 384
    assert observed["effective_configuration"]["preprocessing"]["grayscale_enabled"] is False
    assert observed["training_config"]["image_size"] == 1152
