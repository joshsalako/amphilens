import pickle
import random

import pytest
from PIL import Image

from amphilens.models.augmentation import DetectionSample, compose_detection_samples


def test_two_panels_preserve_boxes_and_pixels_without_blending():
    red = Image.new("RGB", (80, 40), "red")
    blue = Image.new("RGB", (80, 40), "blue")
    samples = [
        DetectionSample(red, [[0, 0, 80, 40]], [1], "red"),
        DetectionSample(blue, [[0, 0, 80, 40]], [2], "blue"),
    ]
    result = compose_detection_samples(samples, (80, 80), angles=[0, 0])
    assert result.image.getpixel((40, 20)) == (255, 0, 0)
    assert result.image.getpixel((40, 60)) == (0, 0, 255)
    assert result.boxes == [[0, 0, 80, 40], [0, 40, 80, 80]]
    assert result.labels == [1, 2]
    assert all(item["minimum_box_visibility"] >= 0.5 for item in result.sources)
    assert red.size == (80, 40)


@pytest.mark.parametrize("angle", [-10, 10])
def test_rotation_keeps_entire_edge_boxes_inside_their_panel(angle):
    image = Image.new("RGB", (60, 40), "white")
    sample = DetectionSample(image, [[0, 0, 60, 40]], [1], "edge")
    result = compose_detection_samples([sample, sample], (100, 100), angles=[angle, angle])
    for index, box in enumerate(result.boxes):
        assert 0 <= box[0] < box[2] <= 100
        assert index * 50 <= box[1] < box[3] <= (index + 1) * 50
    assert all(item["minimum_box_visibility"] >= 0.99 for item in result.sources)


def test_negative_images_remain_negative_and_rotation_is_seeded():
    image = Image.new("RGB", (60, 40), "white")
    sample = DetectionSample(image, [], [], "negative")
    first = compose_detection_samples([sample], (64, 64), rng=random.Random(42))
    second = compose_detection_samples([sample], (64, 64), rng=random.Random(42))
    assert first.boxes == first.labels == []
    assert first.sources == second.sources
    assert first.image.tobytes() == second.image.tobytes()
    assert -10 <= first.sources[0]["rotation_degrees"] <= 10


def test_more_than_two_images_is_rejected():
    sample = DetectionSample(Image.new("RGB", (20, 20)), [], [])
    with pytest.raises(ValueError, match="one or two"):
        compose_detection_samples([sample] * 3, (40, 40))


def test_rotated_box_contains_the_object_pixels():
    from PIL import ImageDraw

    image = Image.new("RGB", (80, 60), "black")
    ImageDraw.Draw(image).rectangle((10, 20, 29, 39), fill="red")
    result = compose_detection_samples(
        [DetectionSample(image, [[10, 20, 30, 40]], [1])], (80, 80), angles=[10]
    )
    pixels = [
        (x, y)
        for y in range(80)
        for x in range(80)
        if (lambda p: p[0] > 240 and p[1] < 10 and p[2] < 10)(result.image.getpixel((x, y)))
    ]
    assert pixels
    x1, y1, x2, y2 = result.boxes[0]
    assert all(x1 - 1 <= x <= x2 + 1 and y1 - 1 <= y <= y2 + 1 for x, y in pixels)


def _training_images(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for name, color in (("red", "red"), ("blue", "blue")):
        Image.new("RGB", (64, 64), color).save(images / f"{name}.png")
        (labels / f"{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    return images, labels


@pytest.mark.parametrize("architecture", ["yolo", "rtdetr"])
def test_real_ultralytics_training_loader_uses_panels_and_validation_does_not(
    tmp_path, monkeypatch, architecture
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import get_cfg

    from amphilens.models.backends import _ULTRALYTICS_AUGMENTATION
    from amphilens.models.ultralytics_training import (
        AmphiLensDetectionTrainer,
        AmphiLensRTDETRTrainer,
        PanelTransform,
    )

    images, labels = _training_images(tmp_path)
    before = (labels / "red.txt").read_bytes()
    trainer_class = AmphiLensDetectionTrainer if architecture == "yolo" else AmphiLensRTDETRTrainer
    trainer = trainer_class.__new__(trainer_class)
    trainer.args = get_cfg(overrides={"imgsz": 64, "workers": 0, **_ULTRALYTICS_AUGMENTATION})
    trainer.data = {"names": {0: "animal"}, "nc": 1, "channels": 3}
    trainer.model = torch.nn.Module()
    trainer.model.stride = torch.tensor([32])
    dataset = trainer.build_dataset(str(images), mode="train", batch=2)
    assert isinstance(dataset.transforms.transforms[0], PanelTransform)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    sample = pickle.loads(pickle.dumps(dataset))[0]
    assert sample["img"].shape == (3, 64, 64)
    assert len(sample["cls"]) == 2
    assert torch.all((sample["bboxes"] >= 0) & (sample["bboxes"] <= 1))
    assert (labels / "red.txt").read_bytes() == before
    validation = trainer.build_dataset(str(images), mode="val", batch=2)
    assert not any(isinstance(t, PanelTransform) for t in validation.transforms.transforms)
    assert len(validation[0]["cls"]) == 1
    # Two negative training frames must not acquire foreground labels through pairing.
    for path in labels.glob("*.txt"):
        path.write_text("")
    negative_dataset = trainer.build_dataset(str(images), mode="train", batch=2)
    assert negative_dataset[0]["bboxes"].shape == (0, 4)
    assert negative_dataset[0]["cls"].shape == (0, 1)


def test_resuming_ultralytics_keeps_the_current_rotation_and_disables_default_rebuilds():
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import get_cfg

    from amphilens.models.ultralytics_training import AmphiLensDetectionTrainer

    trainer = AmphiLensDetectionTrainer.__new__(AmphiLensDetectionTrainer)
    # The legacy checkpoint's arguments could otherwise replace today's policy.
    trainer.args = get_cfg(overrides={"degrees": 15.0, "mosaic": 1.0, "close_mosaic": 10})
    trainer.check_resume({})
    assert trainer.args.degrees == 10.0
    assert trainer.args.mosaic == trainer.args.close_mosaic == 0


def test_faster_rcnn_training_loader_rotates_pairs_and_keeps_negative_validation(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from amphilens.models.faster_rcnn_training import FasterRCNNDataset, FasterRCNNDatasetSpec

    images, labels = _training_images(tmp_path)
    spec = FasterRCNNDatasetSpec(tmp_path / "dataset.yaml", tmp_path, images, labels, ("animal",))
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "uniform", lambda low, high: 10.0)
    dataset = FasterRCNNDataset(spec, augment=True)
    image, target = dataset[0]
    assert image.shape == (3, 64, 64)
    assert len(target["boxes"]) == len(target["labels"]) == 2
    assert len(dataset.last_augmentation_sources) == 2
    assert all(s["rotation_degrees"] == 10 for s in dataset.last_augmentation_sources)
    assert all(s["minimum_box_visibility"] >= 0.5 for s in dataset.last_augmentation_sources)
    (labels / "blue.txt").write_text("")
    validation = FasterRCNNDataset(spec, augment=False)
    _, negative = validation[0]
    assert negative["boxes"].shape == (0, 4)
    assert validation.last_augmentation_sources == []
    (labels / "red.txt").write_text("")
    _, negative_pair = dataset[0]
    assert negative_pair["boxes"].shape == (0, 4)
    assert negative_pair["labels"].shape == (0,)
