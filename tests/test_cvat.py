import json
from pathlib import Path

from PIL import Image

from amphilens.annotations.cvat import export_cvat, export_yolo, import_cvat, import_yolo
from amphilens.core import DetectionRecord


def test_cvat_round_trip_preserves_source_mapping_and_boxes(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "camera one.jpg"
    Image.new("RGB", (40, 30), color="black").save(image)
    record = DetectionRecord(
        image_path=str(image),
        image_id=image.name,
        class_id=0,
        class_name="toad",
        confidence=0.8,
        bbox_xyxy=[4, 5, 20, 25],
        image_width=40,
        image_height=30,
        model_id="fixture",
        run_id="run-1",
    )

    task = export_cvat([record], tmp_path / "cvat", classes=["toad"])
    coco = json.loads((task / "annotations.json").read_text())
    assert (task / "images").is_dir()
    assert coco["images"][0]["file_name"] == "camera_one.jpg"

    imported = import_cvat(task)
    assert imported[0].image_path == str(image)
    assert imported[0].bbox_xyxy == [4.0, 5.0, 20.0, 25.0]
    assert imported[0].class_name == "toad"


def test_yolo_exchange_preserves_normalized_coordinates(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "camera.jpg"
    Image.new("RGB", (40, 30), color="black").save(image)
    record = DetectionRecord(
        image_path=str(image),
        image_id=image.name,
        class_id=0,
        class_name="toad",
        confidence=None,
        bbox_xyxy=[4, 5, 20, 25],
        image_width=40,
        image_height=30,
        model_id="human-cvat",
        run_id="annotation-import",
    )

    task = export_yolo([record], tmp_path / "yolo", classes=["toad"])
    assert (task / "labels" / "camera.txt").read_text().strip() == "0 0.3 0.5 0.4 0.666667"
    imported = import_yolo(task)
    assert imported[0].bbox_xyxy == [4.0, 5.0, 20.0, 25.0]
