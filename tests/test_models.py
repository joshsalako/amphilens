from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from amphilens.core import CheckpointManifest, InferenceConfig, UnsupportedCheckpointError
from amphilens.models import FasterRCNNDetector, UltralyticsDetector, load_detector
from amphilens.models.backends import OptionalDependencyError, _select_torch_device
from amphilens.models.faster_rcnn_training import (
    FasterRCNNDatasetSpec,
    _selected_checkpoint_name,
    parse_yolo_label_lines,
)


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
        sources = kwargs["source"]
        if not isinstance(sources, list):
            sources = [sources]
        return [SimpleNamespace(boxes=self.boxes, names=self.names) for _ in sources]

    def train(self, **kwargs):
        self.train_calls = kwargs
        weight_name = "best.pt" if kwargs.get("val", True) else "last.pt"
        checkpoint = Path(kwargs["project"]) / kwargs["name"] / "weights" / weight_name
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"trained")
        return SimpleNamespace(save_dir=checkpoint.parent.parent)

    def _smart_load(self, key):
        assert key == "trainer"
        return FakeBaseTrainer


class FakeBaseTrainer:
    def __init__(self):
        self.metrics = {}
        self.fitness = None
        self.best_fitness = None
        self.last = Path("missing-last.pt")
        self.best = Path("missing-best.pt")

    def validate(self):
        raise AssertionError("Cloud training must skip validation")

    def final_eval(self):
        raise AssertionError("Cloud training must skip final evaluation")


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


def test_ultralytics_adapter_preprocesses_before_prediction_and_maps_boxes(tmp_path: Path):
    image = tmp_path / "wide.jpg"
    Image.new("RGB", (80, 40), color=(200, 10, 10)).save(image)
    detector = UltralyticsDetector(tmp_path / "best.pt", "yolo", ["toad"])
    fake_model = FakeUltralyticsModel(FakeBoxes([[0, 0, 20, 20]], [0.9], [0]))
    detector._model = fake_model

    records = list(
        detector.predict(
            [image],
            InferenceConfig(
                model_id="fixture",
                preprocessing={"max_dimension": 40},
                run_id="run-1",
            ),
        )
    )

    assert fake_model.calls[0]["source"][0].size == (40, 20)
    assert records[0].bbox_xyxy == [0.0, 0.0, 40.0, 40.0]
    assert (
        records[0].preprocessing
        == InferenceConfig(
            model_id="fixture", preprocessing={"max_dimension": 40}
        ).preprocessing_fingerprint
    )


def test_ultralytics_adapter_runs_a_batch_and_maps_each_result(tmp_path: Path):
    images = []
    for name in ("first.jpg", "second.jpg"):
        image = tmp_path / name
        Image.new("RGB", (40, 30), color="black").save(image)
        images.append(image)
    detector = UltralyticsDetector(tmp_path / "best.pt", "yolo", ["toad"])
    detector._model = FakeUltralyticsModel(FakeBoxes([[1, 2, 20, 25]], [0.75], [0]))

    records = list(
        detector.predict(
            images,
            InferenceConfig(model_id="fixture", batch_size=2, device="cuda", run_id="run-batch"),
        )
    )

    assert len(detector._model.calls) == 1
    assert len(detector._model.calls[0]["source"]) == 2
    assert detector._model.calls[0]["batch"] == 2
    assert detector._model.calls[0]["device"] == "cuda"
    assert [record.image_id for record in records] == ["first.jpg", "second.jpg"]


def test_ultralytics_adapter_passes_resume_checkpoint_to_training(tmp_path: Path):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    preprocessing = InferenceConfig(model_id="fixture").preprocessing_config.to_dict()
    manifest = CheckpointManifest.create(
        parent,
        model_id="fixture",
        architecture="yolo",
        classes=["toad"],
        preprocessing=preprocessing,
    )
    detector = UltralyticsDetector(parent, "yolo", ["toad"], model_id="fixture")
    model = FakeUltralyticsModel(None)
    detector._model = model

    result = detector.train(
        tmp_path / "dataset.yaml",
        tmp_path / "output",
        {"preprocessing": preprocessing, "epochs": 1},
        resume_from=manifest,
    )

    assert result.is_file()
    assert model.train_calls["resume"] == str(parent)
    assert model.train_calls["val"] is True


def test_ultralytics_cloud_training_uses_last_weights_without_validation(tmp_path: Path):
    detector = UltralyticsDetector(tmp_path / "base.pt", "yolo", ["toad"])
    model = FakeUltralyticsModel(None)
    detector._model = model
    provenance = {"provider": "modal"}
    config = {"epochs": 1, "evaluation": "not evaluated", "cloud": provenance}

    result = detector.train(tmp_path / "dataset.yaml", tmp_path / "output", config)

    assert result.name == "last.pt"
    assert model.train_calls["val"] is False
    assert provenance["checkpoint_selection"] == "last-no-validation"
    trainer = model.train_calls["trainer"]()
    assert trainer.validate() == ({}, 0.0)
    assert trainer.best_fitness == 0.0
    assert trainer.final_eval() is None


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


def test_torch_device_auto_uses_mps_when_cuda_is_unavailable():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        device=lambda value: value,
    )

    assert _select_torch_device(fake_torch, "auto") == "mps"
    assert _select_torch_device(fake_torch, "mps") == "mps"


def test_torch_device_selection_fails_closed_for_unavailable_mps():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        device=lambda value: value,
    )

    with pytest.raises(OptionalDependencyError, match="MPS"):
        _select_torch_device(fake_torch, "mps")


def test_faster_rcnn_dataset_spec_is_stable_for_yolo_layout(tmp_path: Path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    spec = FasterRCNNDatasetSpec.from_mapping(
        tmp_path / "dataset.yaml",
        {"path": str(tmp_path), "train": "images", "labels": "labels", "names": ["toad"]},
        expected_classes=["toad"],
    )
    assert spec.image_dir == image_dir.resolve()
    assert spec.label_dir == label_dir.resolve()


def test_faster_rcnn_label_parser_rejects_unknown_classes_and_bad_boxes(tmp_path: Path):
    labels = parse_yolo_label_lines(["0 0.5 0.5 0.4 0.6"], 40, 30, class_count=1)
    assert labels == [([12.0, 6.0, 28.0, 24.0], 1)]
    with pytest.raises(ValueError, match="class id"):
        parse_yolo_label_lines(["1 0.5 0.5 0.4 0.6"], 40, 30, class_count=1)
    with pytest.raises(ValueError, match="within the image"):
        parse_yolo_label_lines(["0 0.1 0.5 0.4 0.6"], 40, 30, class_count=1)


def test_faster_rcnn_cloud_training_selects_last_weights_without_validation():
    assert _selected_checkpoint_name({"cloud": {"provider": "modal"}}) == "last.pt"
    assert _selected_checkpoint_name({}) == "best.pt"


def test_faster_rcnn_training_rejects_missing_base_checkpoint(tmp_path: Path):
    detector = FasterRCNNDetector(tmp_path / "missing.pt", ["toad"])
    with pytest.raises(UnsupportedCheckpointError, match="missing"):
        detector.train(tmp_path / "dataset.yaml", tmp_path / "output", {"epochs": 1})
