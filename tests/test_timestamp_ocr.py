from __future__ import annotations

import csv
import sys

import cv2
import numpy as np
import pytest

from amphilens.timestamp_ocr import (
    OCRDependencyError,
    create_rapidocr_engine,
    parse_timestamp_text,
    process_prediction_csv,
    read_timestamp_from_image,
)


def _image(path, *, width=800, height=600):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return image


def test_parser_returns_iso_date_and_time_only_for_valid_pair():
    assert parse_timestamp_text("Date 09/10/2026 18:42:07") == ("2026-10-09", "18:42:07")
    assert parse_timestamp_text("2026-10-09 18:42") == ("2026-10-09", "18:42:00")
    assert parse_timestamp_text("31/02/2026 18:42:07") == ("", "")
    assert parse_timestamp_text("09/10/2026 no time") == ("", "")


def test_missing_optional_runtime_has_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    message = "uv sync --extra web --extra inference --extra ocr"
    with pytest.raises(OCRDependencyError, match=message):
        create_rapidocr_engine()


def test_ocr_receives_only_bottom_crops_and_tries_fallback_crops(tmp_path):
    image_path = tmp_path / "frame.jpg"
    original = _image(image_path)
    received = []

    def engine(crop):
        received.append(crop.shape[:2])
        if len(received) == 1:
            return "unreadable"
        return ([([0, 0], "O9/1O/2O26 l8:42:O7", 0.99)], (0.01, 0.02))

    assert read_timestamp_from_image(image_path, ocr_engine=engine) == (
        "2026-10-09",
        "18:42:07",
    )
    assert len(received) >= 2
    assert all(height < original.shape[0] for height, _ in received)
    assert all(width > height for height, width in received)
    assert original.shape[:2] not in received


def test_prediction_csv_processes_unique_images_and_repeats_values(tmp_path):
    image_root = tmp_path / "images"
    image_path = image_root / "camera" / "frame.jpg"
    _image(image_path)
    _image(image_root / "camera" / "unreadable.jpg")
    csv_path = tmp_path / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "class_name", "confidence"])
        writer.writeheader()
        writer.writerows(
            [
                {"image_path": "camera/frame.jpg", "class_name": "toad", "confidence": "0.9"},
                {"image_path": "camera/frame.jpg", "class_name": "frog", "confidence": "0.8"},
                {"image_path": "camera/unreadable.jpg", "class_name": "toad", "confidence": "0.75"},
                {"image_path": "camera/missing.jpg", "class_name": "toad", "confidence": "0.7"},
            ]
        )

    calls = []
    progress = []

    def engine(crop):
        calls.append(crop.shape)
        return "09/10/2026 18:42:07" if len(calls) == 1 else "unreadable timestamp"

    summary = process_prediction_csv(
        csv_path, image_root, ocr_engine=engine, progress_callback=progress.append
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert summary.image_count == 3
    assert summary.recognized_count == 1
    assert summary.unparsed_count == 2
    assert len(calls) == 4
    assert [(row["date"], row["time"]) for row in rows] == [
        ("2026-10-09", "18:42:07"),
        ("2026-10-09", "18:42:07"),
        ("", ""),
        ("", ""),
    ]
    assert [(row["class_name"], row["confidence"]) for row in rows] == [
        ("toad", "0.9"),
        ("frog", "0.8"),
        ("toad", "0.75"),
        ("toad", "0.7"),
    ]
    assert progress[-1]["completed"] == 3
    assert progress[-1]["recognized"] == 1
    assert progress[-1]["unparsed"] == 2

    replacement = process_prediction_csv(
        csv_path,
        image_root,
        ocr_engine=lambda crop: "10/10/2026 01:02:03",
    )
    assert replacement.recognized_count == 2
    with csv_path.open(newline="", encoding="utf-8") as handle:
        replaced_rows = list(csv.DictReader(handle))
    assert [(row["date"], row["time"]) for row in replaced_rows[:2]] == [
        ("2026-10-10", "01:02:03"),
        ("2026-10-10", "01:02:03"),
    ]


def test_failed_ocr_leaves_original_csv_untouched(tmp_path):
    image_root = tmp_path / "images"
    _image(image_root / "first.jpg")
    _image(image_root / "second.jpg")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "image_path,date,time,other\nfirst.jpg,old-date,old-time,keep\nsecond.jpg,,,also-keep\n",
        encoding="utf-8",
    )
    original = csv_path.read_bytes()
    call_count = 0

    def engine(crop):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("OCR engine failed")
        return "09/10/2026 18:42:07"

    with pytest.raises(RuntimeError, match="OCR engine failed"):
        process_prediction_csv(csv_path, image_root, ocr_engine=engine)

    assert csv_path.read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == ["images", "results.csv"]
