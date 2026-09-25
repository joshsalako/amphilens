import json
from pathlib import Path

import pytest
from PIL import Image

from amphilens.annotations.cvat import export_cvat, export_yolo, import_cvat, import_yolo
from amphilens.core import DetectionRecord, ValidationError


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

    task = export_yolo(iter([record]), tmp_path / "yolo", classes=["toad"])
    assert (task / "labels" / "camera.txt").read_text().strip() == "0 0.3 0.5 0.4 0.666667"
    imported = import_yolo(task)
    assert imported[0].bbox_xyxy == [4.0, 5.0, 20.0, 25.0]


def test_yolo_import_rejects_out_of_range_normalized_boxes(tmp_path: Path):
    task = tmp_path / "task"
    (task / "labels").mkdir(parents=True)
    (task / "classes.txt").write_text("toad\n")
    (task / "manifest.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "source_path": "/source/camera.jpg",
                        "file_name": "camera.jpg",
                        "width": 40,
                        "height": 30,
                    }
                ]
            }
        )
    )
    (task / "labels" / "camera.txt").write_text("0 0.5 0.5 1.2 0.2\n")

    with pytest.raises(ValidationError, match="between 0 and 1"):
        import_yolo(task)


def test_cvat_image_mapping_is_stable_when_record_order_changes(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    records = []
    for name in ("b.jpg", "a.jpg"):
        path = source / name
        Image.new("RGB", (20, 20), color="black").save(path)
        records.append(
            DetectionRecord(
                image_path=str(path),
                image_id=name,
                class_id=0,
                class_name="toad",
                confidence=None,
                bbox_xyxy=[1, 1, 5, 5],
                image_width=20,
                image_height=20,
                model_id="human-cvat",
                run_id="annotation-import",
            )
        )

    first = export_cvat(records, tmp_path / "first", classes=["toad"])
    second = export_cvat(reversed(records), tmp_path / "second", classes=["toad"])
    first_mapping = json.loads((first / "manifest.json").read_text())["images"]
    second_mapping = json.loads((second / "manifest.json").read_text())["images"]
    assert first_mapping == second_mapping


def test_cvat_import_rejects_unknown_classes_and_duplicate_annotation_ids(tmp_path: Path):
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
        model_id="fixture",
        run_id="run-1",
    )
    task = export_cvat([record], tmp_path / "task", classes=["toad"])

    coco = json.loads((task / "annotations.json").read_text())
    coco["categories"][0]["name"] = "unknown"
    (task / "annotations.json").write_text(json.dumps(coco))
    with pytest.raises(ValidationError, match="Unknown CVAT class"):
        import_cvat(task)

    coco["categories"][0]["name"] = "toad"
    coco["annotations"].append(dict(coco["annotations"][0]))
    (task / "annotations.json").write_text(json.dumps(coco))
    with pytest.raises(ValidationError, match="duplicate annotation id"):
        import_cvat(task)


def test_cvat_import_rejects_missing_images_dimension_mismatches_and_bad_boxes(tmp_path: Path):
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
        model_id="fixture",
        run_id="run-1",
    )

    missing_task = export_cvat([record], tmp_path / "missing", classes=["toad"])
    (missing_task / "images" / "camera.jpg").unlink()
    with pytest.raises(ValidationError, match="CVAT image is missing"):
        import_cvat(missing_task)

    mismatch_task = export_cvat([record], tmp_path / "mismatch", classes=["toad"])
    manifest = json.loads((mismatch_task / "manifest.json").read_text())
    manifest["images"][0]["width"] = 41
    (mismatch_task / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="dimensions do not match"):
        import_cvat(mismatch_task)

    bad_box_task = export_cvat([record], tmp_path / "bad-box", classes=["toad"])
    coco = json.loads((bad_box_task / "annotations.json").read_text())
    coco["annotations"][0]["bbox"] = [39, 5, 4, 4]
    (bad_box_task / "annotations.json").write_text(json.dumps(coco))
    with pytest.raises(ValidationError, match="outside image"):
        import_cvat(bad_box_task)


def test_cvat_export_rejects_duplicate_classes_and_unknown_record_class(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "camera.jpg"
    Image.new("RGB", (20, 20), color="black").save(image)
    record = DetectionRecord(
        image_path=str(image),
        image_id=image.name,
        class_id=1,
        class_name="frog",
        confidence=None,
        bbox_xyxy=[1, 1, 5, 5],
        image_width=20,
        image_height=20,
        model_id="fixture",
        run_id="run-1",
    )
    with pytest.raises(ValidationError, match="unique"):
        export_cvat([record], tmp_path / "duplicate", classes=["toad", "toad"])
    with pytest.raises(ValidationError, match="not present in classes"):
        export_cvat([record], tmp_path / "unknown", classes=["toad"])
