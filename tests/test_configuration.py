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


def test_effective_configuration_accepts_explicit_cli_model_identity(tmp_path: Path):
    project = _project(tmp_path)

    effective = resolve_effective_configuration(
        project,
        overrides={"model_id": "fixture", "architecture": "yolo"},
    )

    assert effective.model_id == "fixture"
    assert effective.architecture == "yolo"


def test_selected_general_model_uses_metadata_or_requested_fallback_not_project_defaults(
    tmp_path: Path,
):
    project = _project(
        tmp_path,
        ProjectConfig(
            classes=["toad"],
            model_preset="rtdetr-l",
            preprocessing=PreprocessingConfig(
                short_side_dimension=384,
                grayscale_enabled=False,
                clahe_enabled=False,
            ),
            image_size=320,
        ),
    )

    effective = resolve_effective_configuration(
        project,
        use_model_metadata_defaults=True,
        overrides={"model_preset": "yolo26-l"},
    )

    assert effective.model_preset == "yolo26-l"
    assert effective.image_size == 640
    assert effective.preprocessing.short_side_dimension == 640
    assert effective.preprocessing.grayscale_enabled is True
    assert effective.preprocessing.clahe_enabled is True

    legacy_cli = resolve_effective_configuration(
        project,
        overrides={"model_preset": "yolo26-l"},
    )
    assert legacy_cli.image_size == 320
    assert legacy_cli.preprocessing.short_side_dimension == 384
    assert legacy_cli.preprocessing.grayscale_enabled is False
    assert legacy_cli.preprocessing.clahe_enabled is False


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


def test_checkpoint_rejects_model_override_and_scopes_preprocessing_override(tmp_path: Path):
    project = _project(tmp_path)
    manifest = _checkpoint(
        tmp_path,
        preprocessing=PreprocessingConfig(clahe_enabled=True).to_dict(),
    )

    requested_preprocessing = PreprocessingConfig(clahe_enabled=False)
    with pytest.raises(UnsupportedCheckpointError, match="preprocessing"):
        resolve_effective_configuration(
            project,
            checkpoint_manifest=manifest,
            overrides={"preprocessing": requested_preprocessing},
        )

    effective = resolve_effective_configuration(
        project,
        checkpoint_manifest=manifest,
        allow_checkpoint_preprocessing_override=True,
        overrides={"preprocessing": requested_preprocessing},
    )
    assert effective.preprocessing.clahe_enabled is False
    assert effective.image_size == 512

    with pytest.raises(UnsupportedCheckpointError, match="model"):
        resolve_effective_configuration(
            project,
            checkpoint_manifest=manifest,
            overrides={"model_preset": "yolo26-l"},
        )


def test_checkpoint_without_preprocessing_metadata_uses_web_fallback(tmp_path: Path):
    project = _project(tmp_path, config=ProjectConfig(classes=["toad"], image_size=1024))
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    manifest = CheckpointManifest.create(
        checkpoint,
        model_id="rtdetr-l",
        architecture="rtdetr",
        classes=["toad"],
        preprocessing={},
        training_config={},
    )

    effective = resolve_effective_configuration(
        project,
        checkpoint_manifest=manifest,
        use_model_metadata_defaults=True,
    )

    assert effective.image_size == 640
    assert effective.preprocessing.short_side_dimension == 640
    assert effective.preprocessing.grayscale_enabled is True
    assert effective.preprocessing.clahe_enabled is True

    legacy_cli = resolve_effective_configuration(project, checkpoint_manifest=manifest)
    assert legacy_cli.image_size == 1024


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


def test_checkpoint_training_uses_project_classes_and_prediction_uses_checkpoint_classes(
    tmp_path: Path,
):
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

    prediction = resolve_effective_configuration(project, checkpoint_manifest=manifest)
    training = resolve_effective_configuration(
        project,
        checkpoint_manifest=manifest,
        target_classes=tuple(project.classes),
    )

    assert prediction.classes == ("frog",)
    assert training.classes == tuple(project.classes)
    assert training.base_classes == ("frog",)


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


def test_hosted_model_metadata_is_default_and_run_preprocessing_is_overridable(tmp_path: Path):
    from amphilens.core import ProjectManifest
    from amphilens.models.hosted_models import get_hosted_model

    project = ProjectManifest.create(
        "custom classes",
        [tmp_path],
        ["toad", "mammal"],
        project_config=ProjectConfig(
            classes=["toad", "mammal"],
            preprocessing=PreprocessingConfig(short_side_dimension=512, clahe_enabled=True),
        ),
    )
    hosted = get_hosted_model("amphilens-rtdetr-l")

    inference = resolve_effective_configuration(
        project,
        hosted_model=hosted,
        use_source_classes=True,
    )
    training = resolve_effective_configuration(
        project,
        hosted_model=hosted,
        overrides={
            "preprocessing": PreprocessingConfig(
                short_side_dimension=384,
                grayscale_enabled=False,
                clahe_enabled=False,
            )
        },
    )

    assert inference.model_preset == "amphilens-rtdetr-l"
    assert inference.architecture == "rtdetr"
    assert inference.image_size == hosted.inference_image_size
    assert inference.classes == hosted.source_classes
    assert inference.preprocessing.to_dict() == hosted.preprocessing.to_dict()
    assert training.preprocessing.short_side_dimension == 384
    assert training.preprocessing.grayscale_enabled is False
    assert training.preprocessing.clahe_enabled is False
    assert training.image_size == hosted.inference_image_size
    assert inference.source == "huggingface"
    assert inference.hosted_model["revision"]
    assert training.architecture == "rtdetr"
    assert training.classes == tuple(project.classes)
    assert training.hosted_model["source_class_order"] == list(hosted.source_classes)
