from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from amphilens.core import DetectionRecord
from amphilens.models.hosted_models import (
    HuggingFaceAccessError,
    download_hosted_checkpoint,
    get_hosted_model,
    list_hosted_models,
    map_hosted_detections,
    resolve_class_mapping,
)


def test_hosted_catalog_contains_exact_architectures_classes_and_artifact_hashes():
    models = {model.model_id: model for model in list_hosted_models()}

    assert set(models) == {
        "amphilens-yolo26-m",
        "amphilens-rtdetr-l",
        "amphilens-faster-rcnn-resnet50",
    }
    assert models["amphilens-yolo26-m"].architecture == "yolo"
    assert models["amphilens-yolo26-m"].size == "medium"
    assert models["amphilens-rtdetr-l"].architecture == "rtdetr"
    assert models["amphilens-faster-rcnn-resnet50"].architecture == "faster_rcnn"
    assert models["amphilens-faster-rcnn-resnet50"].size == "resnet50"
    assert models["amphilens-yolo26-m"].source_classes == (
        "Other_Amphibian",
        "Small_Mammal",
        "Western_Leopard_Toad",
    )
    assert all(len(model.sha256) == 64 for model in models.values())


def test_class_mapping_prefills_exact_matches_and_requires_every_other_source_label():
    model = get_hosted_model("amphilens-yolo26-m")

    suggested = resolve_class_mapping(
        model.source_classes,
        ["Small_Mammal", "toad", "bird"],
        {"Other_Amphibian": None, "Western_Leopard_Toad": "toad"},
    )

    assert suggested == {
        "Other_Amphibian": None,
        "Small_Mammal": "Small_Mammal",
        "Western_Leopard_Toad": "toad",
    }
    with pytest.raises(ValueError, match="Other_Amphibian"):
        resolve_class_mapping(model.source_classes, ["toad"], {})
    with pytest.raises(ValueError, match="Unknown project class"):
        resolve_class_mapping(
            model.source_classes,
            ["toad"],
            {name: "missing" for name in model.source_classes},
        )


def test_class_mapping_can_ignore_detections_and_reindexes_project_classes():
    source = DetectionRecord(
        image_path="/images/one.jpg",
        image_id="one.jpg",
        class_id=2,
        class_name="Western_Leopard_Toad",
        confidence=0.9,
        bbox_xyxy=[1, 2, 3, 4],
        image_width=20,
        image_height=10,
        model_id="amphilens-yolo26-m",
        run_id="run-1",
        preprocessing="profile",
    )
    mapping = {
        "Other_Amphibian": None,
        "Small_Mammal": "mammal",
        "Western_Leopard_Toad": "toad",
    }

    mapped = map_hosted_detections([source], mapping, ["mammal", "toad"])

    assert len(mapped) == 1
    assert mapped[0].class_id == 1
    assert mapped[0].class_name == "toad"
    assert mapped[0].bbox_xyxy == source.bbox_xyxy


def test_download_uses_private_repo_login_pinned_revision_cache_and_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import builtins

    import amphilens.models.hosted_models as hosted

    artifact = tmp_path / "weights.pt"
    artifact.write_bytes(b"verified test checkpoint")
    captured = {}

    def fake_hf_hub_download(**kwargs):
        captured.update(kwargs)
        return str(artifact)

    monkeypatch.setattr(
        hosted, "_hub_runtime", lambda: (fake_hf_hub_download, lambda: "local-token")
    )
    model = get_hosted_model("amphilens-yolo26-m")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setattr(hosted, "get_hosted_model", lambda _: replace(model, sha256=digest))
    real_import = builtins.__import__

    def import_without_tqdm(name, *args, **kwargs):
        if name.split(".", 1)[0] == "tqdm":
            raise ModuleNotFoundError("No module named 'tqdm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tqdm)

    result = download_hosted_checkpoint(model.model_id)

    assert result == artifact
    assert captured["repo_id"] == "josh-salako/amphilens"
    assert captured["filename"] == "yolo_clahe.pt"
    assert captured["revision"] == hosted.HF_MODEL_REVISION
    assert captured["token"] is True
    assert captured["tqdm_class"] is None
    assert "cache_dir" not in captured


def test_download_preserves_the_snapshot_filename_for_model_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import amphilens.models.hosted_models as hosted
    from amphilens.models.backends import load_detector

    blob = tmp_path / "blobs" / "content-addressed-hash"
    blob.parent.mkdir()
    blob.write_bytes(b"verified test checkpoint")
    snapshot = tmp_path / "snapshots" / hosted.HF_MODEL_REVISION / "yolo_clahe.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)
    model = get_hosted_model("amphilens-yolo26-m")
    digest = hashlib.sha256(blob.read_bytes()).hexdigest()

    monkeypatch.setattr(hosted, "_hub_runtime", lambda: (lambda **_: str(snapshot), lambda: None))
    monkeypatch.setattr(hosted, "get_hosted_model", lambda _: replace(model, sha256=digest))

    result = download_hosted_checkpoint(model.model_id)

    assert result == snapshot
    assert result.suffix == ".pt"
    assert result.is_file()
    detector = load_detector(
        result,
        architecture="yolo",
        classes=list(model.source_classes),
    )
    assert detector.checkpoint == snapshot
    assert detector.checkpoint.suffix == ".pt"


def test_public_checkpoint_can_download_without_a_hugging_face_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import amphilens.models.hosted_models as hosted

    artifact = tmp_path / "public-weights.pt"
    artifact.write_bytes(b"public fixture")
    captured = {}

    def fake_hf_hub_download(**kwargs):
        captured.update(kwargs)
        return str(artifact)

    model = get_hosted_model("amphilens-yolo26-m")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setattr(hosted, "_hub_runtime", lambda: (fake_hf_hub_download, lambda: None))
    monkeypatch.setattr(hosted, "get_hosted_model", lambda _: replace(model, sha256=digest))

    assert download_hosted_checkpoint(model.model_id) == artifact
    assert captured["token"] is False


def test_private_repository_access_error_is_actionable_without_leaking_token(
    monkeypatch: pytest.MonkeyPatch,
):
    import sys
    from types import ModuleType

    import amphilens.models.hosted_models as hosted

    class FakeHfHubHTTPError(Exception):
        pass

    hub_module = ModuleType("huggingface_hub")
    hub_module.__path__ = []
    errors_module = ModuleType("huggingface_hub.errors")
    errors_module.HfHubHTTPError = FakeHfHubHTTPError
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors_module)

    def raise_access_error(**_kwargs):
        raise FakeHfHubHTTPError("HTTP 404: token=private-secret")

    monkeypatch.setattr(hosted, "_hub_runtime", lambda: (raise_access_error, lambda: "local-token"))

    with pytest.raises(HuggingFaceAccessError, match="public Hugging Face model") as error:
        download_hosted_checkpoint("amphilens-yolo26-m")
    assert "private-secret" not in str(error.value)
    assert "network connection" in str(error.value)
