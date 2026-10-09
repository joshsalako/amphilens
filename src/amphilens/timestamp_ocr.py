"""Optional local OCR for date and time overlays in camera-trap images."""

from __future__ import annotations

import csv
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date as calendar_date
from datetime import time as clock_time
from pathlib import Path
from typing import Any

from .core import AmphiLensError

OCR_INSTALL_MESSAGE = (
    "Timestamp OCR is optional. Install it with `uv sync --extra web --extra inference "
    "--extra ocr` or `pip install 'amphilens[ocr]'`, then restart AmphiLens."
)


class OCRDependencyError(AmphiLensError):
    """The optional OCR runtime is not installed."""


@dataclass(frozen=True, slots=True)
class TimestampOCRSummary:
    csv_path: Path
    image_count: int
    recognized_count: int
    unparsed_count: int


def create_rapidocr_engine() -> Callable[[Any], Any]:
    """Load RapidOCR only when a user explicitly starts timestamp extraction."""
    try:
        import onnxruntime  # noqa: F401 - provide a clear error if the selected backend is absent
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise OCRDependencyError(OCR_INSTALL_MESSAGE) from exc
    try:
        return RapidOCR()
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


def _bottom_crops(image: Any) -> list[Any]:
    """Build progressively wider crops from the bottom overlay area only."""
    try:
        import cv2
    except ImportError as exc:
        raise OCRDependencyError(OCR_INSTALL_MESSAGE) from exc

    height, width = image.shape[:2]
    bounds = (
        (int(height * 0.84), 0),
        (int(height * 0.78), int(width * 0.35)),
        (int(height * 0.72), int(width * 0.20)),
    )
    crops = []
    for top, left in bounds:
        grayscale = cv2.cvtColor(image[top:height, left:width], cv2.COLOR_BGR2GRAY)
        # Equalise contrast and binarise common white/dark date overlays.
        grayscale = cv2.equalizeHist(grayscale)
        _, processed = cv2.threshold(
            grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        enlarged = cv2.resize(
            processed,
            None,
            fx=3.0,
            fy=3.0,
            interpolation=cv2.INTER_CUBIC,
        )
        crops.append(enlarged)
    return crops


def read_timestamp_from_image(
    image_path: str | Path,
    *,
    ocr_engine: Callable[[Any], Any] | None = None,
) -> tuple[str, str]:
    """Read a date and time overlay, giving RapidOCR only preprocessed bottom crops."""
    try:
        import cv2
    except ImportError as exc:
        raise OCRDependencyError(OCR_INSTALL_MESSAGE) from exc

    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return "", ""
    engine = ocr_engine or create_rapidocr_engine()
    for crop in _bottom_crops(image):
        date_value, time_value = parse_timestamp_text(_ocr_text(engine(crop)))
        if date_value and time_value:
            return date_value, time_value
    return "", ""


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


def process_prediction_csv(
    predictions_csv: str | Path,
    image_root: str | Path,
    *,
    ocr_engine: Callable[[Any], Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TimestampOCRSummary:
    """OCR each unique prediction image, then atomically append/update date and time columns."""
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

    engine = ocr_engine or create_rapidocr_engine()
    timestamps: dict[Path, tuple[str, str]] = {}
    recognized_count = 0
    total = len(unique_images)
    for completed, image_path in enumerate(unique_images, start=1):
        timestamp = read_timestamp_from_image(image_path, ocr_engine=engine)
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

    for row, image_path in zip(rows, row_images, strict=True):
        row["date"], row["time"] = timestamps[image_path]

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

    return TimestampOCRSummary(
        csv_path=csv_path,
        image_count=total,
        recognized_count=recognized_count,
        unparsed_count=total - recognized_count,
    )
