from pathlib import Path

from amphilens.core import ProjectConfig, ProjectManifest, ProjectStore
from amphilens.preprocessing import PreprocessingConfig


def test_project_config_round_trips_in_project_manifest(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    config = ProjectConfig(
        classes=["toad", "frog"],
        model_preset="yolo26-l",
        checkpoint_source="official-general-purpose",
        preprocessing=PreprocessingConfig(clahe_enabled=True),
        epochs=4,
        batch_size=2,
        freeze_strategy="paper-phased",
    )
    manifest = ProjectManifest.create(
        "configured",
        [image_dir],
        config.classes,
        project_config=config,
    )
    store = ProjectStore(tmp_path / "project")
    store.create(manifest)

    restored = store.load_manifest().project_config
    assert restored.model_preset == "yolo26-l"
    assert restored.preprocessing.grayscale_enabled is True
    assert restored.preprocessing.clahe_enabled is True
    assert restored.freeze_strategy == "paper-phased"


def test_project_config_defaults_match_paper_aligned_preprocessing():
    config = ProjectConfig(classes=["toad"])

    assert config.preprocessing.grayscale_enabled is True
    assert config.preprocessing.max_dimension == 640
    assert config.preprocessing.clahe_enabled is False
    assert config.evaluation == "not evaluated"

