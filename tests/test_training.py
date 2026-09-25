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

    def train(self, dataset_yaml, output_dir, config, resume_from=None):
        self.dataset_yaml = Path(dataset_yaml)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "best.pt"
        checkpoint.write_bytes(b"fixture-checkpoint")
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
    assert result.manifest.architecture == "yolo"
    saved = json.loads((tmp_path / "cycle-0" / "checkpoint.json").read_text())
    assert saved["training_config"]["seed"] == 17
    CheckpointManifest(**saved).validate_compatibility(
        architecture="yolo", classes=["toad"], preprocessing={"name": "none"}
    )


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
        preprocessing=PreprocessingConfig(max_dimension=40),
    )

    assert result.checkpoint.is_file()
    assert trainer.dataset_yaml.is_file()
    assert trainer.dataset_yaml.parent.name == "prepared-dataset"
    saved = json.loads((tmp_path / "cycle-0" / "checkpoint.json").read_text())
    assert saved["training_config"]["preprocessing"]["max_dimension"] == 40
