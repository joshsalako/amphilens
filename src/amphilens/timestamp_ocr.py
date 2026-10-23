"""Optional date and time extraction for camera-trap images, with OCR fallback."""

from __future__ import annotations

import csv
import os
import re
import struct
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date as calendar_date
from datetime import datetime as calendar_datetime
from datetime import time as clock_time
from pathlib import Path
from typing import Any

from .cancellation import check_cancellation
from .core import AmphiLensError

OCR_INSTALL_MESSAGE = (
    "Timestamp extraction is optional. For local extraction, add `--extra ocr` to your "
    "existing `uv sync` command. For a new CPU setup, use `uv sync --locked --python 3.11 "
    "--extra cli --extra web --extra inference --extra ocr`; replace `inference` with "
    "`local-cuda` or `local-mps` for those devices. For cloud OCR, add `--extra cloud-ocr` "
    "to your provider setup command, then restart AmphiLens."
)
_MAX_OCR_SIDE = 4096


class OCRDependencyError(AmphiLensError):
    """The optional OCR runtime is not installed."""


@dataclass(frozen=True, slots=True)
class TimestampOCRSummary:
    csv_path: Path
    image_count: int
    recognized_count: int
    unparsed_count: int


def create_rapidocr_engine(
    *, device: str = "cpu", cpu_threads: int = 2
) -> Callable[[Any], Any]:
    """Load RapidOCR with a bounded CPU or selected local accelerator backend."""
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("OCR device must be cpu, cuda, or mps")
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise OCRDependencyError(OCR_INSTALL_MESSAGE) from exc
    if device == "cpu":
        if cpu_threads < 1:
            raise ValueError("OCR CPU thread limit must be at least 1")
        try:
            import onnxruntime  # noqa: F401 - default CPU engine requires ONNX Runtime
        except ImportError as exc:
            raise OCRDependencyError(OCR_INSTALL_MESSAGE) from exc
        params = {
            "EngineConfig.onnxruntime.intra_op_num_threads": cpu_threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    else:
        try:
            from rapidocr import EngineType
        except ImportError as exc:
            raise OCRDependencyError(OCR_INSTALL_MESSAGE) from exc
        params = {
            "Det.engine_type": EngineType.TORCH,
            "Cls.engine_type": EngineType.TORCH,
            "Rec.engine_type": EngineType.TORCH,
            f"EngineConfig.torch.use_{device}": True,
        }
    try:
        return RapidOCR(params=params)
    except ImportError as exc:
        raise OCRDependencyError(OCR_INSTALL_MESSAGE) from exc


def _normalise_ocr_digits(value: str) -> str:
    substitutions = {"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"}
    return value.translate(str.maketrans(substitutions))


def parse_timestamp_text(text: str) -> tuple[str, str]:
    """Return an ISO date/time pair, or two blanks unless both values are valid."""
    cleaned = _normalise_ocr_digits(text)
    date_value: str | None = None
    date_pattern = re.compile(
        r"(?<!\d)(\d{1,4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{2,4})(?!\d)"
    )
    for match in date_pattern.finditer(cleaned):
        first, second, third = match.groups()
        try:
            if len(first) == 4:
                parsed_date = calendar_date(int(first), int(second), int(third))
            else:
                year = int(third)
                if year < 100:
                    year += 2000
                # Camera timestamps in this workflow use day/month/year ordering.
                parsed_date = calendar_date(year, int(second), int(first))
        except ValueError:
            continue
        date_value = parsed_date.isoformat()
        break

    time_value: str | None = None
    time_pattern = re.compile(r"(?<!\d)(\d{1,2})\s*:\s*(\d{2})(?:\s*:\s*(\d{2}))?(?!\d)")
    for match in time_pattern.finditer(cleaned):
        hour, minute, second = match.groups()
        try:
            parsed_time = clock_time(int(hour), int(minute), int(second or 0))
        except ValueError:
            continue
        time_value = parsed_time.isoformat(timespec="seconds")
        break

    if date_value is None or time_value is None:
        return "", ""
    return date_value, time_value


def _ocr_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    texts = getattr(result, "txts", None)
    if texts is not None:
        return " ".join(str(value) for value in texts if value)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
        # Older RapidOCR wrappers return (recognition rows, elapsed times).
        return _ocr_text(result[0])
    output: list[str] = []
    try:
        items = iter(result)
    except TypeError:
        return str(result)
    for item in items:
        if isinstance(item, str):
            output.append(item)
        elif isinstance(item, (tuple, list)) and len(item) > 1 and isinstance(item[1], str):
            output.append(item[1])
    return " ".join(output)


def _read_timestamp_from_exif(image_path: str | Path) -> tuple[str, str]:
    """Read a valid capture timestamp from standard EXIF date/time fields."""
    path = Path(image_path)
    if not path.is_file():
        return "", ""

    try:
        from PIL import ExifTags, Image
    except ImportError as exc:
        raise OCRDependencyError(OCR_INSTALL_MESSAGE) from exc

    try:
        with Image.open(path) as image:
            exif = image.getexif()
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
            candidates = (
                exif_ifd.get(ExifTags.Base.DateTimeOriginal),
                exif_ifd.get(ExifTags.Base.DateTimeDigitized),
                exif.get(ExifTags.Base.DateTime),
            )
            for value in candidates:
                if not isinstance(value, str):
                    continue
                try:
                    timestamp = calendar_datetime.strptime(
                        value.strip(" \t\r\n\x00"), "%Y:%m:%d %H:%M:%S"
                    )
                except ValueError:
                    continue
                return (
                    timestamp.date().isoformat(),
                    timestamp.time().isoformat(timespec="seconds"),
                )
    except (ImportError, OSError, SyntaxError, TypeError, ValueError, struct.error):
        return "", ""
    return "", ""


def _bottom_crops(image: Any, *, pre_cropped_region: bool = False) -> Iterator[Any]:
    """Yield progressively wider, preprocessed bottom crops only as needed."""
    try:
        import cv2
    except ImportError as exc:
        raise OCRDependencyError(OCR_INSTALL_MESSAGE) from exc

    height, width = image.shape[:2]
    if pre_cropped_region:
        # A local client uploads the union of the three regions: the bottom 28%.
        # Recreate the original fallback bounds relative to that reduced image.
        bounds = (
            (int(height * (3 / 7)), 0),
            (int(height * (3 / 14)), int(width * 0.35)),
            (0, int(width * 0.20)),
        )
    else:
        bounds = (
            (int(height * 0.84), 0),
            (int(height * 0.78), int(width * 0.35)),
            (int(height * 0.72), int(width * 0.20)),
        )
    for top, left in bounds:
        crop = image[top:height, left:width]
        scale = min(3.0, _MAX_OCR_SIDE / max(crop.shape[:2]))
        if scale < 1.0:
            crop = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
            scale = 1.0
        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Equalise contrast and binarise common white/dark date overlays.
        grayscale = cv2.equalizeHist(grayscale)
        _, processed = cv2.threshold(
            grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        enlarged = (
            cv2.resize(
                processed,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            if scale > 1.0
            else processed
        )
        yield enlarged


def read_timestamp_from_image(
    image_path: str | Path,
    *,
    ocr_engine: Callable[[Any], Any] | None = None,
    device: str = "cpu",
) -> tuple[str, str]:
    """Prefer embedded EXIF time, then read an overlay using preprocessed bottom crops."""
    timestamp = _read_timestamp_from_exif(image_path)
    if timestamp[0] and timestamp[1]:
        return timestamp

    timestamp, _ = _read_timestamp_with_ocr(image_path, ocr_engine=ocr_engine, device=device)
    return timestamp


def _read_timestamp_with_ocr(
    image_path: str | Path,
    *,
    ocr_engine: Callable[[Any], Any] | None = None,
    device: str = "cpu",
) -> tuple[tuple[str, str], Callable[[Any], Any] | None]:
    """Read an overlay and return the engine for reuse by the batch processor."""
    try:
        import cv2
    except ImportError as exc:
        raise OCRDependencyError(OCR_INSTALL_MESSAGE) from exc

    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return ("", ""), ocr_engine
    engine = ocr_engine or create_rapidocr_engine(device=device)
    for crop in _bottom_crops(image):
        date_value, time_value = parse_timestamp_text(_ocr_text(engine(crop)))
        if date_value and time_value:
            return (date_value, time_value), engine
    return ("", ""), engine


def read_timestamp_from_cropped_region(
    crop_path: str | Path,
    *,
    ocr_engine: Callable[[Any], Any],
) -> tuple[str, str]:
    """Read a timestamp from a client-uploaded bottom-28% crop."""
    try:
        import cv2
    except ImportError as exc:
        raise OCRDependencyError(OCR_INSTALL_MESSAGE) from exc
    image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if image is None:
        return "", ""
    for crop in _bottom_crops(image, pre_cropped_region=True):
        date_value, time_value = parse_timestamp_text(_ocr_text(ocr_engine(crop)))
        if date_value and time_value:
            return date_value, time_value
    return "", ""


def save_timestamp_crop(image_path: str | Path, destination: str | Path) -> bool:
    """Save only the bottom 28% of an image for remote timestamp extraction."""
    try:
        import cv2
    except ImportError as exc:
        raise OCRDependencyError(OCR_INSTALL_MESSAGE) from exc
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    height = image.shape[0]
    region = image[int(height * 0.72) :, :]
    scale = min(1.0, _MAX_OCR_SIDE / max(region.shape[:2]))
    if scale < 1.0:
        region = cv2.resize(
            region, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(destination), region, [cv2.IMWRITE_JPEG_QUALITY, 95]
    ):
        raise OSError("Could not save the timestamp crop")
    return True


def _resolve_prediction_image(value: str | None, image_root: Path) -> Path:
    if not value:
        raise ValueError("Predictions CSV contains an empty image_path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = image_root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(image_root):
        raise ValueError("Prediction image is outside the selected image folder")
    return resolved


def unique_prediction_images(
    predictions_csv: str | Path, image_root: str | Path
) -> list[Path]:
    """Return validated, de-duplicated source images referenced by a predictions CSV."""
    csv_path = Path(predictions_csv).expanduser().resolve()
    source_root = Path(image_root).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError("Predictions CSV was not found")
    if not source_root.is_dir():
        raise FileNotFoundError("Prediction image folder was not found")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError("Predictions CSV must contain an image_path column")
        unique_images: dict[Path, None] = {}
        for row in reader:
            unique_images.setdefault(
                _resolve_prediction_image(row.get("image_path"), source_root), None
            )
    return list(unique_images)


def apply_timestamp_results(
    predictions_csv: str | Path,
    image_root: str | Path,
    timestamps: dict[Path, tuple[str, str]],
) -> TimestampOCRSummary:
    """Atomically update the predictions CSV with results returned by an OCR worker."""
    csv_path = Path(predictions_csv).expanduser().resolve()
    source_root = Path(image_root).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError("Predictions CSV was not found")
    if not source_root.is_dir():
        raise FileNotFoundError("Prediction image folder was not found")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError("Predictions CSV must contain an image_path column")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    for column in ("date", "time"):
        if column not in fieldnames:
            fieldnames.append(column)
    row_images = [
        _resolve_prediction_image(row.get("image_path"), source_root) for row in rows
    ]
    unique_images = dict.fromkeys(row_images)
    complete_timestamps = {
        image: timestamps.get(image, ("", "")) for image in unique_images
    }
    recognized_count = 0
    for row, image in zip(rows, row_images, strict=True):
        row["date"], row["time"] = complete_timestamps[image]
    for date_value, time_value in complete_timestamps.values():
        if date_value and time_value:
            recognized_count += 1
    _write_prediction_rows(csv_path, fieldnames, rows)
    total = len(unique_images)
    return TimestampOCRSummary(
        csv_path=csv_path,
        image_count=total,
        recognized_count=recognized_count,
        unparsed_count=total - recognized_count,
    )


def _write_prediction_rows(csv_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=csv_path.parent,
            prefix=f".{csv_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, csv_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def process_prediction_csv(
    predictions_csv: str | Path,
    image_root: str | Path,
    *,
    ocr_engine: Callable[[Any], Any] | None = None,
    device: str = "cpu",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TimestampOCRSummary:
    """Extract EXIF timestamps first, OCR unresolved images, and update the CSV atomically."""
    csv_path = Path(predictions_csv).expanduser().resolve()
    source_root = Path(image_root).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError("Predictions CSV was not found")
    if not source_root.is_dir():
        raise FileNotFoundError("Prediction image folder was not found")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError("Predictions CSV must contain an image_path column")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    for column in ("date", "time"):
        if column not in fieldnames:
            fieldnames.append(column)

    unique_images: dict[Path, None] = {}
    row_images: list[Path] = []
    for row in rows:
        image = _resolve_prediction_image(row.get("image_path"), source_root)
        row_images.append(image)
        unique_images.setdefault(image, None)

    engine = ocr_engine
    timestamps: dict[Path, tuple[str, str]] = {}
    recognized_count = 0
    total = len(unique_images)
    for completed, image_path in enumerate(unique_images, start=1):
        check_cancellation(progress_callback, "Timestamp OCR canceled between images.")
        timestamp = _read_timestamp_from_exif(image_path)
        if not (timestamp[0] and timestamp[1]):
            timestamp, engine = _read_timestamp_with_ocr(
                image_path, ocr_engine=engine, device=device
            )
        timestamps[image_path] = timestamp
        if timestamp[0] and timestamp[1]:
            recognized_count += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "ocr",
                    "message": f"Read timestamps from {completed} of {total} unique images",
                    "completed": completed,
                    "total": total,
                    "remaining": total - completed,
                    "recognized": recognized_count,
                    "unparsed": completed - recognized_count,
                    "progress": completed / total if total else 1.0,
                }
            )

    check_cancellation(progress_callback, "Timestamp OCR canceled before updating the CSV.")
    for row, image_path in zip(rows, row_images, strict=True):
        row["date"], row["time"] = timestamps[image_path]

    _write_prediction_rows(csv_path, fieldnames, rows)

    return TimestampOCRSummary(
        csv_path=csv_path,
        image_count=total,
        recognized_count=recognized_count,
        unparsed_count=total - recognized_count,
    )
