from pathlib import Path

import pytest

from amphilens.configuration import (
    discover_checkpoint_manifest,
    resolve_effective_configuration,
)
from amphilens.core import (
    CheckpointManifest,
    ProjectConfig,
    ProjectManifest,
    UnsupportedCheckpointError,
)
from amphilens.preprocessing import PreprocessingConfig


def _project(tmp_path: Path, config: ProjectConfig | None = None) -> ProjectManifest:
    images = tmp_path / "images"
    images.mkdir()
    return ProjectManifest.create(
        "study",
        [images],
        ["toad"],
        project_config=config or ProjectConfig(classes=["toad"]),
    )


def _checkpoint(tmp_path: Path, *, preprocessing: dict, image_size: int = 512):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    return CheckpointManifest.create(
        checkpoint,
        model_id="rtdetr-l",
        architecture="rtdetr",
        classes=["toad"],
        preprocessing=preprocessing,
        training_config={"image_size": image_size, "epochs": 3},
    )


def test_effective_configuration_uses_project_defaults_and_overrides(tmp_path: Path):
    project = _project(
        tmp_path,
        ProjectConfig(
            classes=["toad"],
            model_preset="rtdetr-l",
            preprocessing=PreprocessingConfig(clahe_enabled=True, max_dimension=512),
            image_size=512,
            confidence_threshold=0.4,
            device="cpu",
            epochs=7,
            batch_size=4,
        ),
    )

    effective = resolve_effective_configuration(
        project,
        overrides={"confidence": 0.7, "device": "auto"},
    )

    assert effective.model_id == "rtdetr-l"
    assert effective.architecture == "rtdetr"
    assert effective.preprocessing.clahe_enabled is True
    assert effective.preprocessing.max_dimension == 512
    assert effective.image_size == 512
    assert effective.confidence == 0.7
    assert effective.device == "auto"
    assert effective.epochs == 7
    assert effective.batch_size == 4
    assert effective.warnings == ()


def test_checkpoint_configuration_is_authoritative(tmp_path: Path):
    project = _project(
        tmp_path,
        ProjectConfig(
            classes=["toad"],
            model_preset="yolo26-l",
            preprocessing=PreprocessingConfig(clahe_enabled=False, max_dimension=640),
            image_size=640,
        ),
    )
    manifest = _checkpoint(
        tmp_path,
        preprocessing=PreprocessingConfig(clahe_enabled=True, max_dimension=320).to_dict(),
        image_size=320,
    )

    effective = resolve_effective_configuration(
        project,
        checkpoint_manifest=manifest,
        overrides={"confidence": 0.6, "device": "cpu"},
    )

    assert effective.model_id == "rtdetr-l"
    assert effective.architecture == "rtdetr"
    assert effective.preprocessing.clahe_enabled is True
    assert effective.preprocessing.max_dimension == 320
    assert effective.image_size == 320
    assert effective.confidence == 0.6
    assert effective.device == "cpu"
    assert effective.source == "checkpoint"


def test_checkpoint_rejects_incompatible_model_or_preprocessing_override(tmp_path: Path):
    project = _project(tmp_path)
    manifest = _checkpoint(
        tmp_path,
        preprocessing=PreprocessingConfig(clahe_enabled=True).to_dict(),
    )

    with pytest.raises(UnsupportedCheckpointError, match="preprocessing"):
        resolve_effective_configuration(
            project,
            checkpoint_manifest=manifest,
            overrides={"preprocessing": PreprocessingConfig(clahe_enabled=False)},
        )

    with pytest.raises(UnsupportedCheckpointError, match="model"):
        resolve_effective_configuration(
            project,
            checkpoint_manifest=manifest,
            overrides={"model_preset": "yolo26-l"},
        )


def test_checkpoint_accepts_explicit_overrides_that_match_its_metadata(tmp_path: Path):
    project = _project(tmp_path)
    preprocessing = PreprocessingConfig(clahe_enabled=True).to_dict()
    manifest = _checkpoint(tmp_path, preprocessing=preprocessing)

    effective = resolve_effective_configuration(
        project,
        checkpoint_manifest=manifest,
        overrides={
            "model_preset": "rtdetr-l",
            "preprocessing": preprocessing,
            "image_size": 512,
        },
    )

    assert effective.model_id == "rtdetr-l"
    assert effective.image_size == 512


def test_checkpoint_class_mismatch_fails_before_a_run_can_start(tmp_path: Path):
    project = _project(tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    manifest = CheckpointManifest.create(
        checkpoint,
        model_id="rtdetr-l",
        architecture="rtdetr",
        classes=["frog"],
        preprocessing=PreprocessingConfig().to_dict(),
    )

    with pytest.raises(UnsupportedCheckpointError, match="classes"):
        resolve_effective_configuration(project, checkpoint_manifest=manifest)


def test_discover_checkpoint_manifest_next_to_checkpoint(tmp_path: Path):
    manifest = _checkpoint(tmp_path, preprocessing=PreprocessingConfig().to_dict())
    manifest_path = tmp_path / "checkpoint.json"
    manifest_path.write_text(__import__("json").dumps(manifest.to_dict()))

    discovered = discover_checkpoint_manifest(manifest.checkpoint_path)

    assert discovered is not None
    assert discovered.sha256 == manifest.sha256


def test_unmanifested_checkpoint_is_explicitly_less_reproducible(tmp_path: Path):
    project = _project(tmp_path)
    checkpoint = tmp_path / "external.pt"
    checkpoint.write_bytes(b"external")

    effective = resolve_effective_configuration(project, checkpoint_path=checkpoint)

    assert effective.source == "project"
    assert any("checkpoint metadata" in warning.lower() for warning in effective.warnings)
