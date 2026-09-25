from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from amphilens.core import CheckpointManifest, InferenceConfig, UnsupportedCheckpointError
from amphilens.models import UltralyticsDetector, load_detector
from amphilens.models.backends import OptionalDependencyError, _select_torch_device


class FakeArray:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, boxes, confidences, class_ids):
        self.xyxy = FakeArray(boxes)
        self.conf = FakeArray(confidences)
        self.cls = FakeArray(class_ids)


class FakeUltralyticsModel:
    names = {0: "toad"}

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=self.boxes, names=self.names)]


def _image(tmp_path: Path) -> Path:
    image = tmp_path / "camera.jpg"
    Image.new("RGB", (40, 30), color="black").save(image)
    return image


def test_ultralytics_adapter_preserves_dimensions_confidence_and_device(tmp_path: Path):
    image = _image(tmp_path)
    detector = UltralyticsDetector(tmp_path / "best.pt", "yolo", ["toad"])
    fake_model = FakeUltralyticsModel(FakeBoxes([[1, 2, 20, 25]], [0.75], [0]))
    detector._model = fake_model

    records = list(
        detector.predict(
            [image],
            InferenceConfig(model_id="fixture", confidence=0.6, device="cpu", run_id="run-1"),
        )
    )

    assert records[0].image_width == 40
    assert records[0].image_height == 30
    assert records[0].confidence == 0.75
    assert fake_model.calls[0]["conf"] == 0.6
    assert fake_model.calls[0]["device"] == "cpu"


def test_ultralytics_adapter_yields_no_records_for_empty_predictions(tmp_path: Path):
    image = _image(tmp_path)
    detector = UltralyticsDetector(tmp_path / "best.pt", "rtdetr", ["toad"])
    detector._model = FakeUltralyticsModel(None)

    assert list(detector.predict([image], InferenceConfig(model_id="fixture"))) == []


def test_load_detector_rejects_missing_and_stale_checkpoints(tmp_path: Path):
    missing = tmp_path / "missing.pt"
    with pytest.raises(UnsupportedCheckpointError, match="missing"):
        load_detector(missing, architecture="yolo", classes=["toad"])

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights-v1")
    manifest = CheckpointManifest.create(
        checkpoint,
        model_id="fixture",
        architecture="yolo",
        classes=["toad"],
        preprocessing={"name": "none"},
    )
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(UnsupportedCheckpointError, match="hash"):
        load_detector(
            checkpoint,
            architecture="yolo",
            classes=["toad"],
            checkpoint_manifest=manifest,
            preprocessing={"name": "none"},
        )


def test_load_detector_rejects_incompatible_checkpoint_manifest(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    manifest = CheckpointManifest.create(
        checkpoint,
        model_id="fixture",
        architecture="rtdetr",
        classes=["toad"],
        preprocessing={"name": "none"},
    )
    with pytest.raises(UnsupportedCheckpointError, match="architecture"):
        load_detector(
            checkpoint,
            architecture="yolo",
            classes=["toad"],
            checkpoint_manifest=manifest,
            preprocessing={"name": "none"},
        )


def test_torch_device_selection_fails_closed_for_unavailable_cuda():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda value: value,
    )
    assert _select_torch_device(fake_torch, "auto") == "cpu"
    with pytest.raises(OptionalDependencyError, match="CUDA"):
        _select_torch_device(fake_torch, "cuda")
