import json
from dataclasses import asdict
from pathlib import Path

import pytest

import amphilens.core as core
from amphilens.core import (
    ArtifactRecord,
    CheckpointManifest,
    DetectionRecord,
    InferenceConfig,
    ProjectManifest,
    ProjectStore,
    RunManifest,
    SourceCollisionError,
    UnsupportedCheckpointError,
    ValidationError,
)
from amphilens.preprocessing import PreprocessingConfig


def test_project_store_creates_portable_manifest_and_run(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "trap_001.jpg").write_bytes(b"not-an-image")

    manifest = ProjectManifest.create(
        name="tunnel-study",
        image_roots=[image_dir],
        classes=["toad", "other_amphibian"],
    )
    store = ProjectStore(tmp_path / "project")
    store.create(manifest)
    run = store.start_run(InferenceConfig(model_id="demo", image_size=640, confidence=0.25))

    assert store.load_manifest().name == "tunnel-study"
    assert run.run_id.startswith("run-")
    assert (tmp_path / "project" / "manifest.json").exists()
    assert (tmp_path / "project" / "runs" / run.run_id / "run.json").exists()


def test_inference_config_preserves_effective_configuration_metadata():
    config = InferenceConfig(
        model_id="demo",
        metadata={"effective_configuration": {"fingerprint": "abc123"}},
    )

    assert config.to_dict()["metadata"]["effective_configuration"]["fingerprint"] == "abc123"


def test_project_store_rejects_source_output_collision(tmp_path: Path):
    source = tmp_path / "images"
    source.mkdir()
    manifest = ProjectManifest.create("collision", [source], ["toad"])
    store = ProjectStore(tmp_path / "project")
    with pytest.raises(SourceCollisionError):
        store.create(manifest, output_dir=source)


def test_detection_record_serializes_pixel_and_normalized_boxes():
    record = DetectionRecord(
        image_path="/data/camera/image.jpg",
        image_id="image.jpg",
        class_id=0,
        class_name="toad",
        confidence=0.91,
        bbox_xyxy=[10, 20, 110, 220],
        image_width=200,
        image_height=400,
        model_id="demo-yolo",
        run_id="run-1",
    )

    assert record.bbox_xyxyn == [0.05, 0.05, 0.55, 0.55]
    row = record.to_row()
    assert row["bbox_xmin"] == 10.0
    assert row["bbox_ymax_norm"] == 0.55
    assert row["class_name"] == "toad"


def test_checkpoint_compatibility_requires_matching_classes_and_architecture(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = CheckpointManifest.create(
        checkpoint,
        model_id="wlt-yolo",
        architecture="yolo",
        classes=["toad"],
        preprocessing={"name": "clahe"},
    )
    manifest.validate_compatibility(
        architecture="yolo", classes=["toad"], preprocessing={"name": "clahe"}
    )

    with pytest.raises(UnsupportedCheckpointError):
        manifest.validate_compatibility(
            architecture="rtdetr", classes=["toad"], preprocessing={"name": "clahe"}
        )


def test_checkpoint_compatibility_accepts_legacy_preprocessing_labels(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = CheckpointManifest.create(
        checkpoint,
        model_id="legacy",
        architecture="yolo",
        classes=["toad"],
        preprocessing={"name": "none"},
    )

    manifest.validate_compatibility(
        architecture="yolo",
        classes=["toad"],
        preprocessing=PreprocessingConfig().to_dict(),
    )


def test_unversioned_manifests_are_migrated_and_future_versions_fail(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    project = ProjectManifest.create("legacy", [image_dir], ["toad"])
    legacy_project = project.to_dict()
    legacy_project.pop("schema_version")
    assert ProjectManifest.from_dict(legacy_project).schema_version == 1

    run = RunManifest.create("inference", {"model_id": "fixture"})
    legacy_run = asdict(run)
    legacy_run.pop("schema_version")
    assert RunManifest.from_dict(legacy_run).schema_version == 1

    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint = CheckpointManifest.create(
        checkpoint_path,
        model_id="fixture",
        architecture="yolo",
        classes=["toad"],
        preprocessing={},
    )
    legacy_checkpoint = asdict(checkpoint)
    legacy_checkpoint.pop("schema_version")
    assert CheckpointManifest.from_dict(legacy_checkpoint).schema_version == 1

    future = project.to_dict()
    future["schema_version"] = 99
    with pytest.raises(ValidationError, match="schema version"):
        ProjectManifest.from_dict(future)


def test_atomic_json_write_keeps_previous_document_on_replace_failure(tmp_path: Path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"state": "old"}) + "\n", encoding="utf-8")

    def fail_replace(*args, **kwargs):
        raise OSError("simulated interruption")

    monkeypatch.setattr(core.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        core.atomic_write_json(path, {"state": "new"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"state": "old"}
    assert list(tmp_path.glob(".state.json.*")) == []


def test_artifact_index_records_relative_path_hash_and_producer(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("artifacts", [image_dir], ["toad"]))
    artifact = store.root / "artifacts" / "predictions.csv"
    artifact.write_text("image_id\ntrap-1\n", encoding="utf-8")

    record = store.register_artifact(
        artifact, artifact_type="predictions", cycle=2, producer_run="run-123"
    )
    assert record == ArtifactRecord(
        relative_path="artifacts/predictions.csv",
        sha256=record.sha256,
        artifact_type="predictions",
        cycle=2,
        producer_run="run-123",
        created_at=record.created_at,
    )
    saved = store.load_artifact_index()
    assert [item.relative_path for item in saved] == ["artifacts/predictions.csv"]
    assert saved[0].sha256 == record.sha256


def test_project_store_rejects_unsafe_run_ids_and_missing_source_roots(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("runs", [image_dir], ["toad"]))
    with pytest.raises(ValidationError, match="run_id"):
        store.start_run(InferenceConfig(model_id="fixture", run_id="../escape"))

    missing = ProjectManifest(
        name="missing",
        classes=["toad"],
        image_roots=[str(tmp_path / "does-not-exist")],
    )
    with pytest.raises(ValidationError, match="does not exist"):
        missing.validate()
