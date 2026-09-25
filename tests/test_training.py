import json
from pathlib import Path

from amphilens.core import CheckpointManifest
from amphilens.training import TrainingConfig, train_and_register


class FixtureTrainer:
    model_id = "fixture-yolo"
    architecture = "yolo"
    classes = ["toad"]

    def train(self, dataset_yaml, output_dir, config, resume_from=None):
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

