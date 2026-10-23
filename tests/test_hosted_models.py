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
        "amphilens-yolo26-m-domain",
        "amphilens-rtdetr-l-domain",
        "amphilens-faster-rcnn-resnet50-domain",
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


def test_catalog_registers_only_the_two_verified_amphilens_yolo_ncnn_packages():
    models = {model.model_id: model for model in list_hosted_models()}
    expected = {
        "amphilens-yolo26-m": {
            "size": 1152,
            "files": {
                "ncnn/amphilens-yolo26-m/model.ncnn.param": (
                    "4cd78a2737891036898b469fc7ea50d8a5ca468ccec52567c4eff295c903c42b"
                ),
                "ncnn/amphilens-yolo26-m/model.ncnn.bin": (
                    "08b69cc94511475425ee4fe44b1f148c4b296d1bbc35fae3f43b97b932fce831"
                ),
            },
        },
        "amphilens-yolo26-m-domain": {
            "size": 640,
            "files": {
                "ncnn/amphilens-yolo26-m-domain/model.ncnn.param": (
                    "e98e804ba92c4412ae9bce92458fac7be619b0cf7182b7bf8705df825877f171"
                ),
                "ncnn/amphilens-yolo26-m-domain/model.ncnn.bin": (
                    "3ef28e8ef6d243acd845f4d2b683a9550cb469efb01cf618daeececed638f73c"
                ),
            },
        },
    }

    assert {model.model_id for model in models.values() if model.ncnn_available} == set(expected)
    for model_id, item in expected.items():
        artifact = models[model_id].ncnn
        assert artifact is not None
        assert artifact.status == "verified"
        assert artifact.adapter == "ultralytics-yolo-raw-detect-v1"
        assert artifact.precision == "fp16"
        assert (artifact.input_width, artifact.input_height) == (item["size"], item["size"])
        assert {file.filename: file.sha256 for file in artifact.files} == item["files"]


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


def test_hosted_manifest_accepts_verified_yolo_ncnn_pair_and_exposes_integrity_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import json

    import amphilens.models.hosted_models as hosted

    manifest = json.loads(hosted._MANIFEST_PATH.read_text(encoding="utf-8"))
    model = manifest["models"][0]
    model["ncnn"] = {
        "status": "verified",
        "adapter": "ultralytics-yolo-raw-detect-v1",
        "precision": "fp16",
        "input_width": 640,
        "input_height": 640,
        "input_name": "in0",
        "output_name": "out0",
        "files": [
            {"filename": "yolo_ncnn_model/model.ncnn.param", "sha256": "a" * 64},
            {"filename": "yolo_ncnn_model/model.ncnn.bin", "sha256": "b" * 64},
        ],
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(hosted, "_MANIFEST_PATH", path)

    listed = hosted.get_hosted_model("amphilens-yolo26-m")
    summary = listed.to_summary()

    assert listed.ncnn_available is True
    assert summary["ncnn_available"] is True
    assert summary["ncnn_artifact"]["precision"] == "fp16"
    assert summary["ncnn_artifact"]["files"][0]["sha256"] == "a" * 64
    assert summary["revision"] == hosted.HF_MODEL_REVISION


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: item.pop("input_name"),
        lambda item: item.update(files=[]),
        lambda item: item["files"][0].update(filename="../escape.param"),
        lambda item: item["files"][0].update(sha256="not-a-hash"),
    ],
    ids=["missing-descriptor-fields", "missing-pair", "unsafe-path", "invalid-hash"],
)
def test_hosted_ncnn_manifest_rejects_incomplete_or_unsafe_artifact_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate
):
    import json

    import amphilens.models.hosted_models as hosted

    manifest = json.loads(hosted._MANIFEST_PATH.read_text(encoding="utf-8"))
    model = manifest["models"][0]
    model["ncnn"] = {
        "status": "verified",
        "adapter": "ultralytics-yolo-raw-detect-v1",
        "precision": "fp16",
        "input_width": 640,
        "input_height": 640,
        "input_name": "in0",
        "output_name": "out0",
        "files": [
            {"filename": "model.ncnn.param", "sha256": "a" * 64},
            {"filename": "model.ncnn.bin", "sha256": "b" * 64},
        ],
    }
    mutate(model["ncnn"])
    path = tmp_path / "models.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(hosted, "_MANIFEST_PATH", path)

    with pytest.raises(ValueError):
        hosted.list_hosted_models()


def test_download_ncnn_pair_uses_pinned_revision_and_checks_each_file_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import amphilens.models.hosted_models as hosted

    model = get_hosted_model("amphilens-yolo26-m")
    param = tmp_path / "model.ncnn.param"
    binary = tmp_path / "model.ncnn.bin"
    param.write_bytes(b"param fixture")
    binary.write_bytes(b"binary fixture")
    descriptor = hosted.NCNNArtifactManifest(
        status="verified",
        adapter="ultralytics-yolo-raw-detect-v1",
        precision="fp16",
        input_width=640,
        input_height=640,
        input_name="in0",
        output_name="out0",
        files=(
            hosted.NCNNArtifactFile(
                "model.ncnn.param", hashlib.sha256(param.read_bytes()).hexdigest()
            ),
            hosted.NCNNArtifactFile(
                "model.ncnn.bin", hashlib.sha256(binary.read_bytes()).hexdigest()
            ),
        ),
    )
    model = replace(model, ncnn=descriptor)
    monkeypatch.setattr(hosted, "get_hosted_model", lambda _: model)
    captured = []

    def download(**kwargs):
        captured.append(kwargs)
        return str(param if kwargs["filename"].endswith(".param") else binary)

    monkeypatch.setattr(hosted, "_hub_runtime", lambda: (download, lambda: None))

    artifact = hosted.download_hosted_ncnn_artifact(model.model_id)

    assert artifact.param_path == param
    assert artifact.bin_path == binary
    assert artifact.revision == hosted.HF_MODEL_REVISION
    assert [item["filename"] for item in captured] == [
        "model.ncnn.param",
        "model.ncnn.bin",
    ]
    assert all(item["revision"] == hosted.HF_MODEL_REVISION for item in captured)


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
