import json
from pathlib import Path

from PIL import Image

from amphilens.core import CheckpointManifest
from amphilens.dataset import DatasetAnnotation, DatasetImage, DatasetManifest, DatasetSnapshot
from amphilens.preprocessing import PreprocessingConfig
from amphilens.training import TrainingConfig, train_and_register, train_snapshot_and_register


class FixtureTrainer:
    model_id = "fixture-yolo"
    architecture = "yolo"
    classes = ["toad"]

    def train(self, dataset_yaml, output_dir, config, resume_from=None, progress_callback=None):
        self.dataset_yaml = Path(dataset_yaml)
        self.resume_from = resume_from
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "best.pt"
        checkpoint.write_bytes(b"fixture-checkpoint")
        if progress_callback is not None:
            progress_callback({"epoch": 1, "epochs": int(config.get("epochs", 1)), "progress": 1.0})
        return checkpoint


def test_training_registers_reusable_checkpoint_manifest(tmp_path: Path):
    result = train_and_register(
        FixtureTrainer(),
        dataset_yaml=tmp_path / "dataset.yaml",
        output_dir=tmp_path / "cycle-0",
        config=TrainingConfig(epochs=2, image_size=640, seed=17),
        preprocessing={"name": "none"},
    )

    assert result.checkpoint.is_file()
    assert (tmp_path / "cycle-0" / "best.pt").is_file()
    assert (tmp_path / "cycle-0" / "last.pt").is_file()
    assert (tmp_path / "cycle-0" / "metrics.json").is_file()
    assert result.manifest.architecture == "yolo"
    saved = json.loads((tmp_path / "cycle-0" / "checkpoint.json").read_text())
    assert saved["training_config"]["seed"] == 17
    CheckpointManifest(**saved).validate_compatibility(
        architecture="yolo", classes=["toad"], preprocessing={"name": "none"}
    )


def test_training_config_preserves_effective_configuration_metadata():
    config = TrainingConfig(metadata={"effective_configuration": {"fingerprint": "abc123"}})

    assert config.to_dict()["metadata"]["effective_configuration"]["fingerprint"] == "abc123"


def test_shared_training_finalization_persists_cloud_provenance_and_reports_progress(
    tmp_path: Path,
):
    progress = []
    result = train_and_register(
        FixtureTrainer(),
        dataset_yaml=tmp_path / "dataset.yaml",
        output_dir=tmp_path / "cloud-run",
        config=TrainingConfig(epochs=2, image_size=640, seed=17),
        preprocessing={"name": "none"},
        progress_callback=progress.append,
        extra_training_config={
            "cloud": {
                "provider": "modal",
                "job_key": "job-123",
                "gpu": "L4",
                "effective_fingerprint": "config-456",
            }
        },
    )

    saved = json.loads((tmp_path / "cloud-run" / "checkpoint.json").read_text())
    assert saved["training_config"]["cloud"]["job_key"] == "job-123"
    assert result.manifest.training_config["cloud"]["effective_fingerprint"] == "config-456"
    assert progress[0]["phase"] == "training"
    assert progress[1]["epoch"] == 1
    assert progress[1]["epochs"] == 2
    assert progress[1]["progress"] == 1.0
    assert progress[-1]["phase"] == "finalizing"


def test_training_from_snapshot_prepares_labels_with_the_same_preprocessing(tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (80, 40), color=(100, 20, 20)).save(source)
    snapshot_root = tmp_path / "snapshot"
    (snapshot_root / "images").mkdir(parents=True)
    snapshot_image = snapshot_root / "images" / "source.png"
    source.replace(snapshot_image)
    annotation = DatasetAnnotation(0, "toad", [20, 10, 60, 30])
    image = DatasetImage(
        image_id="image-1",
        relative_path="images/source.png",
        source_path="source.png",
        width=80,
        height=40,
        sha256="a" * 64,
        reviewed=True,
        annotations=[annotation],
    )
    manifest = DatasetManifest(
        snapshot_id="snapshot-1",
        source_format="fixture",
        classes=["toad"],
        images=[image],
        source_archive="fixture.zip",
        source_archive_sha256="b" * 64,
    )
    snapshot = DatasetSnapshot(snapshot_root, manifest)
    trainer = FixtureTrainer()

    result = train_snapshot_and_register(
        trainer,
        snapshot=snapshot,
        output_dir=tmp_path / "cycle-0",
        config=TrainingConfig(epochs=1, image_size=640, batch_size=1),
        preprocessing=PreprocessingConfig(short_side_dimension=20),
    )

    assert result.checkpoint.is_file()
    assert trainer.dataset_yaml.is_file()
    assert trainer.dataset_yaml.parent.name == "prepared-dataset"
    saved = json.loads((tmp_path / "cycle-0" / "checkpoint.json").read_text())
    assert saved["training_config"]["preprocessing"]["short_side_dimension"] == 20


def test_training_passes_parent_checkpoint_for_resume_lineage(tmp_path: Path):
    trainer = FixtureTrainer()
    preprocessing = PreprocessingConfig(max_dimension=320)
    first = train_and_register(
        trainer,
        dataset_yaml=tmp_path / "dataset.yaml",
        output_dir=tmp_path / "cycle-0",
        config=TrainingConfig(epochs=1, image_size=320, preprocessing=preprocessing),
        preprocessing=preprocessing,
    )

    resumed = train_and_register(
        trainer,
        dataset_yaml=tmp_path / "dataset.yaml",
        output_dir=tmp_path / "cycle-1",
        config=TrainingConfig(epochs=1, image_size=320, preprocessing=preprocessing),
        preprocessing=preprocessing,
        resume_from=first.manifest,
    )

    assert trainer.resume_from == first.manifest
    assert resumed.manifest.parent_checkpoint == str(first.checkpoint)
