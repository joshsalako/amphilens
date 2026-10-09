from pathlib import Path

import pytest

from amphilens.core import DetectionRecord, InferenceConfig
from amphilens.inference import run_inference, write_predictions_csv


class FixtureDetector:
    model_id = "fixture"

    def predict(self, image_paths, config):
        for path in image_paths:
            yield DetectionRecord(
                image_path=str(path),
                image_id=Path(path).name,
                class_id=0,
                class_name="toad",
                confidence=0.9,
                bbox_xyxy=[1, 2, 11, 22],
                image_width=20,
                image_height=40,
                model_id=self.model_id,
                run_id=config.run_id,
            )


def test_inference_writes_stable_csv(tmp_path: Path):
    images = []
    for name in ("a.jpg", "b.jpg"):
        path = tmp_path / name
        path.write_bytes(b"fixture")
        images.append(path)

    config = InferenceConfig(model_id="fixture", image_size=640, confidence=0.2, run_id="run-1")
    records = list(run_inference(FixtureDetector(), images, config))
    output = tmp_path / "predictions.csv"
    write_predictions_csv(records, output)

    assert len(records) == 2
    text = output.read_text()
    assert "image_path" in text
    assert "bbox_xmin" in text
    assert "fixture" in text


def test_predictions_csv_records_the_run_name(tmp_path: Path):
    import csv

    image = tmp_path / "a.jpg"
    image.write_bytes(b"fixture")
    config = InferenceConfig(model_id="fixture", image_size=640, confidence=0.2, run_id="run-1")
    records = list(run_inference(FixtureDetector(), [image], config))
    output = tmp_path / "predictions.csv"

    write_predictions_csv(records, output, run_name="Spring survey")

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["run_name"] == "Spring survey"
    assert row["model_id"] == "fixture"


def test_collect_prediction_images_deduplicates_and_preserves_relative_paths(tmp_path: Path):
    import csv

    from amphilens.inference import collect_prediction_images

    image_root = tmp_path / "source"
    first = image_root / "camera-a" / "frame.jpg"
    second = image_root / "camera-b" / "frame.jpg"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path"])
        writer.writeheader()
        writer.writerows(
            [{"image_path": str(first)}, {"image_path": str(first)}, {"image_path": str(second)}]
        )
    destination = tmp_path / "project" / "runs" / "run-1" / "images"

    result = collect_prediction_images(predictions, image_root, destination)

    assert result.copied_count == 2
    assert result.already_present_count == 0
    assert result.missing_count == 0
    assert (destination / "camera-a" / "frame.jpg").read_bytes() == b"first"
    assert (destination / "camera-b" / "frame.jpg").read_bytes() == b"second"
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"

    repeated = collect_prediction_images(predictions, image_root, destination)
    assert repeated.copied_count == 0
    assert repeated.already_present_count == 2

    first.unlink()
    repeated_after_source_removed = collect_prediction_images(predictions, image_root, destination)
    assert repeated_after_source_removed.copied_count == 0
    assert repeated_after_source_removed.already_present_count == 2
    assert repeated_after_source_removed.missing_count == 0


def test_collect_prediction_images_reports_missing_source_images(tmp_path: Path):
    import csv

    from amphilens.inference import collect_prediction_images

    image_root = tmp_path / "source"
    image_root.mkdir()
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path"])
        writer.writeheader()
        writer.writerow({"image_path": str(image_root / "missing.jpg")})

    result = collect_prediction_images(
        predictions, image_root, tmp_path / "project" / "runs" / "run-1" / "images"
    )

    assert result.copied_count == 0
    assert result.missing_count == 1


def test_collect_prediction_images_rejects_sources_outside_the_selected_folder(tmp_path: Path):
    import csv

    from amphilens.inference import collect_prediction_images

    image_root = tmp_path / "source"
    outside_image = tmp_path / "private.jpg"
    image_root.mkdir()
    outside_image.write_bytes(b"private")
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path"])
        writer.writeheader()
        writer.writerow({"image_path": str(outside_image)})

    with pytest.raises(ValueError, match="outside the selected image folder"):
        collect_prediction_images(
            predictions, image_root, tmp_path / "project" / "runs" / "run-1" / "images"
        )


def test_collect_prediction_images_rejects_symlinked_destination_parents(tmp_path: Path):
    import csv

    from amphilens.inference import collect_prediction_images

    image_root = tmp_path / "source"
    image = image_root / "camera" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"source image")
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path"])
        writer.writeheader()
        writer.writerow({"image_path": str(image)})

    destination = tmp_path / "project" / "runs" / "run-1" / "images"
    destination.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "camera").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        collect_prediction_images(predictions, image_root, destination)

    assert not (outside / "frame.jpg").exists()
