from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from amphilens.core import CheckpointManifest, InferenceConfig, UnsupportedCheckpointError
from amphilens.models import FasterRCNNDetector, UltralyticsDetector, load_detector
from amphilens.models.backends import (
    OptionalDependencyError,
    _configure_faster_rcnn_transform,
    _select_torch_device,
)
from amphilens.models.faster_rcnn_training import (
    FasterRCNNDatasetSpec,
    _apply_faster_rcnn_augmentation,
    _selected_checkpoint_name,
    parse_yolo_label_lines,
)
from amphilens.preprocessing import PreprocessingConfig


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
    fake_model = FakeUltralyticsModel(FakeBoxes([[12, 6, 52, 26]], [0.9], [0]))
    detector._model = fake_model

    records = list(
        detector.predict(
            [image],
            InferenceConfig(
                model_id="fixture",
                preprocessing={"short_side_dimension": 20},
                run_id="run-1",
            ),
        )
    )

    assert fake_model.calls[0]["source"][0].size == (64, 32)
    assert fake_model.calls[0]["imgsz"] == (32, 64)
    assert records[0].bbox_xyxy == [0.0, 0.0, 80.0, 40.0]
    assert (
        records[0].preprocessing
        == InferenceConfig(
            model_id="fixture", preprocessing={"short_side_dimension": 20}
        ).preprocessing_fingerprint
    )


def test_ultralytics_prediction_clips_boxes_to_content_and_drops_padding_only_boxes(
    tmp_path: Path,
):
    image = tmp_path / "wide.jpg"
    Image.new("RGB", (60, 30), color="white").save(image)
    detector = UltralyticsDetector(tmp_path / "best.pt", "yolo", ["toad"])
    detector._model = FakeUltralyticsModel(
        FakeBoxes(
            [[8, 10, 20, 20], [0, 8, 10, 18], [20, 22, 60, 30]],
            [0.9, 0.8, 0.7],
            [0, 0, 0],
        )
    )

    records = list(
        detector.predict(
            [image],
            InferenceConfig(
                model_id="fixture",
                image_size=20,
                confidence=0.1,
                preprocessing=PreprocessingConfig(short_side_dimension=20),
                device="cpu",
            ),
        )
    )

    assert [record.bbox_xyxy for record in records] == [
        [0.0, 6.0, 12.0, 21.0],
        [12.0, 24.0, 60.0, 30.0],
    ]


@pytest.mark.parametrize("architecture", ["yolo", "rtdetr"])
def test_ultralytics_adapter_pads_mixed_aspect_batches_without_resizing_content(
    tmp_path: Path, architecture: str
):
    landscape = tmp_path / "landscape.png"
    portrait = tmp_path / "portrait.png"
    Image.new("RGB", (1200, 600), color=(40, 50, 60)).save(landscape)
    Image.new("RGB", (600, 1200), color=(70, 80, 90)).save(portrait)
    detector = UltralyticsDetector(tmp_path / "best.pt", architecture, ["toad"])
    fake_model = FakeUltralyticsModel(None)
    detector._model = fake_model

    list(
        detector.predict(
            [landscape, portrait],
            InferenceConfig(
                model_id="fixture",
                batch_size=2,
                device="cpu",
                preprocessing={"grayscale_enabled": False},
            ),
        )
    )

    call = fake_model.calls[0]
    assert call["imgsz"] == (1280, 1280)
    assert [image.size for image in call["source"]] == [(1280, 1280), (1280, 1280)]
    assert call["source"][0].getpixel((640, 319)) == (114, 114, 114)
    assert call["source"][0].getpixel((640, 320)) == (40, 50, 60)
    assert call["source"][1].getpixel((319, 640)) == (114, 114, 114)
    assert call["source"][1].getpixel((320, 640)) == (70, 80, 90)


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


def test_ultralytics_training_uses_prepared_dataset_canvas_without_resizing(tmp_path: Path):
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text('{"training_image_size": 1280}\n', encoding="utf-8")
    detector = UltralyticsDetector(tmp_path / "base.pt", "yolo", ["toad"])
    model = FakeUltralyticsModel(None)
    detector._model = model

    config = {"image_size": 640, "epochs": 1}
    detector.train(dataset_yaml, tmp_path / "output", config)

    assert model.train_calls["imgsz"] == 1280
    assert model.train_calls["close_mosaic"] == 0
    assert config["training_image_size"] == 1280
    assert model.train_calls["degrees"] == 15.0
    assert model.train_calls["translate"] == 0.1
    assert model.train_calls["scale"] == 0.5
    assert model.train_calls["shear"] == 2.0
    assert model.train_calls["perspective"] == 0.0001
    assert model.train_calls["flipud"] == 0.5
    assert model.train_calls["fliplr"] == 0.5
    assert model.train_calls["mosaic"] == 1.0
    assert model.train_calls["mixup"] == 0.15
    assert model.train_calls["copy_paste"] == 0.2


def test_faster_rcnn_augmentation_keeps_boxes_aligned_after_horizontal_flip():
    torch = pytest.importorskip("torch")
    from torchvision.transforms import v2

    image = torch.zeros((3, 20, 30), dtype=torch.uint8)
    target = {
        "boxes": torch.tensor([[2.0, 4.0, 10.0, 12.0]]),
        "labels": torch.tensor([1]),
    }

    augmented_image, augmented_target = _apply_faster_rcnn_augmentation(
        image, target, transform=v2.RandomHorizontalFlip(p=1.0)
    )

    assert augmented_image.shape == image.shape
    assert augmented_target["boxes"].tolist() == [[20.0, 4.0, 28.0, 12.0]]


@pytest.mark.parametrize(
    ("source_size", "processed_size"),
    [((1200, 600), (1280, 640)), ((600, 1200), (640, 1280)), ((800, 800), (640, 640))],
)
def test_faster_rcnn_prediction_keeps_the_preprocessed_short_side(
    tmp_path: Path, source_size: tuple[int, int], processed_size: tuple[int, int]
):
    torch = pytest.importorskip("torch")

    class FakeFasterRCNN:
        def __init__(self):
            self.transform = SimpleNamespace(min_size=(800,), max_size=1333)
            self.batch_shapes = None

        def to(self, device):
            return self

        def __call__(self, images):
            self.batch_shapes = [tuple(image.shape[-2:]) for image in images]
            empty = torch.empty((0, 4))
            return [
                {
                    "boxes": empty,
                    "scores": torch.empty(0),
                    "labels": torch.empty(0, dtype=torch.int64),
                }
                for _ in images
            ]

    image = tmp_path / "landscape.png"
    Image.new("RGB", source_size, color=(10, 20, 30)).save(image)
    detector = FasterRCNNDetector(tmp_path / "best.pt", ["toad"])
    model = FakeFasterRCNN()
    detector._model = model

    assert (
        list(
            detector.predict(
                [image],
                InferenceConfig(model_id="fixture", device="cpu"),
            )
        )
        == []
    )

    assert model.transform.min_size == (640,)
    assert model.transform.max_size > 1280
    assert model.batch_shapes == [(processed_size[1], processed_size[0])]


def test_ultralytics_adapter_passes_resume_checkpoint_to_training(tmp_path: Path):
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text('{"train":"images","val":"images"}\n', encoding="utf-8")
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
        dataset_yaml,
        tmp_path / "output",
        {"preprocessing": preprocessing, "epochs": 1},
        resume_from=manifest,
    )

    assert result.is_file()
    assert model.train_calls["resume"] == str(parent)
    assert model.train_calls["val"] is True


def test_ultralytics_cloud_training_monitors_training_data_without_test_split(tmp_path: Path):
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text('{"train":"images","val":"images"}\n', encoding="utf-8")
    detector = UltralyticsDetector(tmp_path / "base.pt", "yolo", ["toad"])
    model = FakeUltralyticsModel(None)
    detector._model = model
    provenance = {"provider": "modal"}
    config = {"epochs": 1, "evaluation": "not evaluated", "cloud": provenance}

    result = detector.train(dataset_yaml, tmp_path / "output", config)

    assert result.name == "best.pt"
    assert model.train_calls["val"] is True
    assert provenance["checkpoint_selection"] == "best-validation"
    assert config["evaluation"] == "not evaluated"
    assert [phase["phase"] for phase in config["training_phases"]] == [
        "backbone-frozen",
        "full-fine-tuning",
    ]


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
        {
            "path": str(tmp_path),
            "train": "images",
            "val": "images",
            "labels": "labels",
            "names": ["toad"],
            "short_side_dimension": 640,
            "training_image_size": 672,
        },
        expected_classes=["toad"],
    )
    assert spec.image_dir == image_dir.resolve()
    assert spec.label_dir == label_dir.resolve()
    assert spec.short_side_dimension == 640
    assert spec.training_image_size == 672


def test_faster_rcnn_transform_preserves_preprocessed_training_canvas():
    model = SimpleNamespace(transform=SimpleNamespace(min_size=(800,), max_size=1333))

    _configure_faster_rcnn_transform(model, 672)

    assert model.transform.min_size == (672,)
    assert model.transform.max_size > 672


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
