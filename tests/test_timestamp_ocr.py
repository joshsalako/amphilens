from __future__ import annotations

import csv
import sys

import cv2
import numpy as np
import pytest
from PIL import ExifTags, Image

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


def _image_with_exif(
    path,
    *,
    original=None,
    digitized=None,
    generic=None,
    width=800,
    height=600,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height))
    exif = Image.Exif()
    exif_ifd = {}
    if original is not None:
        exif_ifd[36867] = original
    if digitized is not None:
        exif_ifd[36868] = digitized
    if exif_ifd:
        exif[ExifTags.IFD.Exif] = exif_ifd
    if generic is not None:
        exif[306] = generic
    image.save(path, format="JPEG", exif=exif)


def test_parser_returns_iso_date_and_time_only_for_valid_pair():
    assert parse_timestamp_text("Date 09/10/2026 18:42:07") == ("2026-10-09", "18:42:07")
    assert parse_timestamp_text("2026-10-09 18:42") == ("2026-10-09", "18:42:00")
    assert parse_timestamp_text("31/02/2026 18:42:07") == ("", "")
    assert parse_timestamp_text("09/10/2026 no time") == ("", "")


def test_exif_timestamp_prefers_original_then_digitized_then_generic_datetime(tmp_path):
    original_path = tmp_path / "original.jpg"
    _image_with_exif(
        original_path,
        original="2025:10:07 03:07:16",
        digitized="2024:02:03 04:05:06",
        generic="2023:01:02 01:02:03",
    )
    digitized_path = tmp_path / "digitized.jpg"
    _image_with_exif(
        digitized_path,
        digitized="2024:02:03 04:05:06",
        generic="2023:01:02 01:02:03",
    )
    generic_path = tmp_path / "generic.jpg"
    _image_with_exif(generic_path, generic="2023:01:02 01:02:03")

    def unexpected_ocr(_crop):
        pytest.fail("RapidOCR should not run for usable EXIF")

    assert read_timestamp_from_image(original_path, ocr_engine=unexpected_ocr) == (
        "2025-10-07",
        "03:07:16",
    )
    assert read_timestamp_from_image(digitized_path, ocr_engine=unexpected_ocr) == (
        "2024-02-03",
        "04:05:06",
    )
    assert read_timestamp_from_image(generic_path, ocr_engine=unexpected_ocr) == (
        "2023-01-02",
        "01:02:03",
    )


def test_malformed_exif_timestamp_falls_back_to_ocr(tmp_path):
    image_path = tmp_path / "malformed.jpg"
    _image_with_exif(image_path, original="not a timestamp")

    result = read_timestamp_from_image(image_path, ocr_engine=lambda _crop: "09/10/2026 18:42:07")

    assert result == ("2026-10-09", "18:42:07")


def test_missing_optional_runtime_has_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    message = "uv sync --extra web --extra inference --extra ocr"
    with pytest.raises(OCRDependencyError, match=message):
        create_rapidocr_engine()


@pytest.mark.parametrize(
    ("device", "expected_engine", "device_option"),
    [("cuda", "TORCH", "use_cuda"), ("mps", "TORCH", "use_mps")],
)
def test_create_rapidocr_engine_uses_torch_for_local_accelerators(
    monkeypatch, device, expected_engine, device_option
):
    import types

    received = {}
    rapidocr = types.ModuleType("rapidocr")
    rapidocr.EngineType = types.SimpleNamespace(TORCH=expected_engine)

    def constructor(*, params):
        received.update(params)
        return lambda image: ""

    rapidocr.RapidOCR = constructor
    monkeypatch.setitem(sys.modules, "rapidocr", rapidocr)

    create_rapidocr_engine(device=device)

    assert received["Det.engine_type"] == expected_engine
    assert received["Cls.engine_type"] == expected_engine
    assert received["Rec.engine_type"] == expected_engine
    assert received[f"EngineConfig.torch.{device_option}"] is True


def test_create_rapidocr_engine_limits_cpu_threads(monkeypatch):
    import types

    received = {}
    rapidocr = types.ModuleType("rapidocr")
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))

    def constructor(*, params):
        received.update(params)
        return lambda image: ""

    rapidocr.RapidOCR = constructor
    monkeypatch.setitem(sys.modules, "rapidocr", rapidocr)

    create_rapidocr_engine(device="cpu", cpu_threads=2)

    assert received["EngineConfig.onnxruntime.intra_op_num_threads"] == 2


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


def test_ocr_builds_fallback_crop_only_when_needed(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.jpg"
    _image(image_path)
    equalize = cv2.equalizeHist
    processed_crops = []

    def count_preprocessing(image):
        processed_crops.append(image.shape)
        return equalize(image)

    monkeypatch.setattr(cv2, "equalizeHist", count_preprocessing)

    result = read_timestamp_from_image(
        image_path, ocr_engine=lambda _crop: "09/10/2026 18:42:07"
    )

    assert result == ("2026-10-09", "18:42:07")
    assert len(processed_crops) == 1


def test_timestamp_crop_export_contains_only_bottom_region(tmp_path):
    from amphilens.timestamp_ocr import save_timestamp_crop

    image_path = tmp_path / "frame.jpg"
    destination = tmp_path / "crop.jpg"
    _image(image_path, width=800, height=600)

    save_timestamp_crop(image_path, destination)

    crop = cv2.imread(str(destination), cv2.IMREAD_COLOR)
    assert crop.shape[:2] == (168, 800)
    assert destination.stat().st_size < image_path.stat().st_size


def test_large_local_ocr_crop_is_capped_before_ocr():
    from amphilens.timestamp_ocr import _bottom_crops

    large_image = np.zeros((1200, 6000, 3), dtype=np.uint8)

    crop = next(_bottom_crops(large_image))

    assert max(crop.shape[:2]) <= 4096


def test_cloud_timestamp_crop_caps_very_wide_images(tmp_path):
    from amphilens.timestamp_ocr import save_timestamp_crop

    image_path = tmp_path / "wide.jpg"
    destination = tmp_path / "crop.jpg"
    _image(image_path, width=6000, height=1200)

    save_timestamp_crop(image_path, destination)

    crop = cv2.imread(str(destination), cv2.IMREAD_COLOR)
    assert crop.shape[1] == 4096
    assert crop.shape[0] < int(1200 * 0.28)


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


def test_prediction_csv_uses_exif_and_ocr_only_for_unique_images_without_it(tmp_path, monkeypatch):
    import amphilens.timestamp_ocr as timestamp_ocr

    image_root = tmp_path / "images"
    _image_with_exif(
        image_root / "camera" / "metadata.jpg",
        original="2025:10:07 03:07:16",
    )
    _image(image_root / "camera" / "ocr.jpg")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "image_path\ncamera/metadata.jpg\ncamera/metadata.jpg\ncamera/ocr.jpg\n",
        encoding="utf-8",
    )
    ocr_calls = []
    engine_factories = []

    def engine(_crop):
        ocr_calls.append(True)
        return "09/10/2026 18:42:07"

    def create_engine(**kwargs):
        engine_factories.append(kwargs)
        return engine

    monkeypatch.setattr(
        timestamp_ocr,
        "create_rapidocr_engine",
        create_engine,
    )
    summary = process_prediction_csv(csv_path, image_root)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert summary.image_count == 2
    assert summary.recognized_count == 2
    assert summary.unparsed_count == 0
    assert engine_factories == [{"device": "cpu"}]
    assert len(ocr_calls) == 1
    assert [(row["date"], row["time"]) for row in rows] == [
        ("2025-10-07", "03:07:16"),
        ("2025-10-07", "03:07:16"),
        ("2026-10-09", "18:42:07"),
    ]


def test_prediction_csv_does_not_create_rapidocr_engine_when_all_images_have_exif(
    tmp_path, monkeypatch
):
    import amphilens.timestamp_ocr as timestamp_ocr

    image_root = tmp_path / "images"
    _image_with_exif(image_root / "metadata.jpg", original="2025:10:07 03:07:16")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("image_path\nmetadata.jpg\n", encoding="utf-8")
    monkeypatch.setattr(
        timestamp_ocr,
        "create_rapidocr_engine",
        lambda **_kwargs: pytest.fail("RapidOCR engine should not be created"),
    )

    summary = process_prediction_csv(csv_path, image_root)

    assert summary.recognized_count == 1


def test_prediction_csv_passes_local_device_to_ocr_engine(tmp_path, monkeypatch):
    import amphilens.timestamp_ocr as timestamp_ocr

    image_root = tmp_path / "images"
    image_path = image_root / "frame.jpg"
    _image(image_path)
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(f"image_path\n{image_path.name}\n", encoding="utf-8")
    selected_devices = []
    monkeypatch.setattr(
        timestamp_ocr,
        "create_rapidocr_engine",
        lambda *, device="cpu", cpu_threads=2: (
            selected_devices.append(device) or (lambda _crop: "09/10/2026 18:42:07")
        ),
    )

    process_prediction_csv(csv_path, image_root, device="mps")

    assert selected_devices == ["mps"]


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
