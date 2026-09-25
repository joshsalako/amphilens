from pathlib import Path

import pytest

from amphilens.core import CheckpointManifest, ModelManifest, UnsupportedCheckpointError
from amphilens.registry import ModelRegistry


def test_registry_persists_model_and_validates_checkpoint_hash(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights-v1")
    checkpoint_manifest = CheckpointManifest.create(
        checkpoint,
        model_id="demo-yolo",
        architecture="yolo",
        classes=["toad"],
        preprocessing={"name": "none"},
    )
    model = ModelManifest(
        model_id="demo-yolo",
        architecture="yolo",
        classes=["toad"],
        training_domain="camera-trap",
        source="fixture",
        license="Apache-2.0",
        preprocessing={"name": "none"},
    )
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(model, checkpoint_manifest)

    loaded = registry.resolve(
        "demo-yolo", architecture="yolo", classes=["toad"], preprocessing={"name": "none"}
    )
    assert loaded.sha256 == checkpoint_manifest.sha256
    assert registry.list_models() == ["demo-yolo"]

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(UnsupportedCheckpointError, match="hash"):
        registry.resolve(
            "demo-yolo", architecture="yolo", classes=["toad"], preprocessing={"name": "none"}
        )

