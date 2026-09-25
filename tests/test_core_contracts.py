from pathlib import Path

import pytest

from amphilens.core import (
    CheckpointManifest,
    DetectionRecord,
    InferenceConfig,
    ProjectManifest,
    ProjectStore,
    SourceCollisionError,
    UnsupportedCheckpointError,
)


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
