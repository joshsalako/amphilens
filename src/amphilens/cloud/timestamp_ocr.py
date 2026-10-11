"""Local crop preparation and cloud timestamp OCR transport orchestration."""

from __future__ import annotations

import hashlib
import re
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from ..core import ValidationError
from ..timestamp_ocr import (
    TimestampOCRSummary,
    apply_timestamp_results,
    parse_timestamp_text,
    read_timestamp_from_cropped_region,
    save_timestamp_crop,
    unique_prediction_images,
)

_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,80}")
_SHA_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_cloud_timestamp_ocr(
    predictions_csv: str | Path,
    image_root: str | Path,
    *,
    transport: Any,
    gpu: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
    job_id_callback: Callable[[str], None] | None = None,
) -> TimestampOCRSummary:
    """Crop unique predicted images locally, upload only crops, then update the local CSV."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", str(gpu)):
        raise ValueError("Cloud OCR GPU selection is invalid")
    images = unique_prediction_images(predictions_csv, image_root)
    job_key = f"ocr-{uuid.uuid4().hex}"
    timestamps = {path: ("", "") for path in images}
    remote_attempted = False
    try:
        with tempfile.TemporaryDirectory(prefix="amphilens-ocr-crops-") as temporary_root:
            crop_root = Path(temporary_root)
            crops: dict[str, Path] = {}
            sources_by_id: dict[str, Path] = {}
            for image_path in images:
                if cancellation_requested and cancellation_requested():
                    from .prediction import PredictionCancelled

                    raise PredictionCancelled("Timestamp OCR canceled during local cropping.")
                image_id = "img-" + hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:24]
                crop_path = crop_root / f"{image_id}.jpg"
                if save_timestamp_crop(image_path, crop_path):
                    crops[image_id] = crop_path
                    sources_by_id[image_id] = image_path
            if not crops:
                return apply_timestamp_results(predictions_csv, image_root, timestamps)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "ocr_upload",
                        "message": f"Preparing bottom crops for {len(crops)} unique images",
                        "completed": 0,
                        "total": len(crops),
                        "remaining": len(crops),
                        "progress": 0.0,
                    }
                )
            remote_attempted = True
            result = transport.run_timestamp_ocr(
                job_key=job_key,
                gpu=str(gpu),
                crops=crops,
                progress_callback=progress_callback,
                cancellation_requested=cancellation_requested,
                job_id_callback=job_id_callback,
            )
            if not isinstance(result, dict) or result.get("state") != "finished":
                message = (
                    result.get("error", "Cloud OCR returned an invalid response")
                    if isinstance(result, dict)
                    else "Cloud OCR returned an invalid response"
                )
                raise RuntimeError(str(message))
            records = result.get("records")
            if not isinstance(records, list):
                raise RuntimeError("Cloud OCR returned an invalid timestamp list")
            seen: set[str] = set()
            for record in records:
                if not isinstance(record, dict):
                    raise RuntimeError("Cloud OCR returned an invalid timestamp record")
                image_id = str(record.get("image_id", ""))
                date_value = record.get("date", "")
                time_value = record.get("time", "")
                if (
                    image_id not in sources_by_id
                    or image_id in seen
                    or not isinstance(date_value, str)
                    or not isinstance(time_value, str)
                ):
                    raise RuntimeError("Cloud OCR returned an unknown or duplicate image")
                seen.add(image_id)
                parsed = parse_timestamp_text(f"{date_value} {time_value}")
                if date_value or time_value:
                    if parsed != (date_value, time_value):
                        raise RuntimeError("Cloud OCR returned an invalid date or time")
                timestamps[sources_by_id[image_id]] = parsed
            if seen != set(sources_by_id):
                raise RuntimeError("Cloud OCR did not return a result for every uploaded crop")
        return apply_timestamp_results(predictions_csv, image_root, timestamps)
    finally:
        cleanup = getattr(transport, "cleanup_timestamp_ocr", None)
        if remote_attempted and callable(cleanup):
            cleanup(job_key)


def run_remote_timestamp_ocr(
    payload: dict[str, Any],
    *,
    volume_root: str | Path,
    ocr_engine: Callable[[Any], Any],
    volume_reload: Callable[[], None] = lambda: None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Verify and OCR a single cloud job's cropped images sequentially."""
    root = Path(volume_root).resolve()
    job_key = str(payload.get("job_key", ""))
    images = payload.get("images")
    if not _ID_PATTERN.fullmatch(job_key):
        raise ValidationError("Timestamp OCR job identity is invalid")
    if not isinstance(images, list) or not images:
        raise ValidationError("Timestamp OCR job contains no image crops")
    job_root = (root / "prediction-jobs" / job_key / "timestamp-ocr").resolve()
    if not job_root.is_relative_to(root):
        raise ValidationError("Timestamp OCR job path is unsafe")
    started = time.monotonic()
    volume_reload()
    manifest_path = job_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValidationError("Timestamp OCR manifest is missing")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("images") != images:
        raise ValidationError("Timestamp OCR manifest does not match the request")
    records = []
    seen: set[str] = set()
    recognized = 0
    image_directory = (job_root / "images").resolve()
    for completed, item in enumerate(images, start=1):
        if not isinstance(item, dict):
            raise ValidationError("Timestamp OCR image manifest is invalid")
        image_id = str(item.get("image_id", ""))
        digest = str(item.get("sha256", ""))
        remote_path = str(item.get("remote_path", ""))
        relative = PurePosixPath(remote_path)
        if (
            not _ID_PATTERN.fullmatch(image_id)
            or image_id in seen
            or not _SHA_PATTERN.fullmatch(digest)
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.parent.as_posix()
            != f"prediction-jobs/{job_key}/timestamp-ocr/images"
            or relative.stem != image_id
            or relative.suffix.lower() != ".jpg"
        ):
            raise ValidationError("Timestamp OCR image identity or path is invalid")
        crop_path = (root / Path(*relative.parts)).resolve()
        if not crop_path.is_relative_to(image_directory) or not crop_path.is_file():
            raise ValidationError("A transferred timestamp crop is missing or unsafe")
        sentinel = Path(f"{crop_path}.sha256")
        if not sentinel.is_file() or sentinel.read_text(encoding="ascii").strip() != digest:
            raise ValidationError("Timestamp crop SHA-256 manifest is missing or mismatched")
        if _sha256_file(crop_path) != digest:
            raise ValidationError("Timestamp crop SHA-256 verification failed")
        date_value, time_value = read_timestamp_from_cropped_region(
            crop_path, ocr_engine=ocr_engine
        )
        records.append({"image_id": image_id, "date": date_value, "time": time_value})
        recognized += int(bool(date_value and time_value))
        seen.add(image_id)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "ocr",
                    "message": f"Read timestamps from {completed} of {len(images)} crops",
                    "completed": completed,
                    "total": len(images),
                    "remaining": len(images) - completed,
                    "recognized": recognized,
                    "unparsed": completed - recognized,
                    "progress": completed / len(images),
                }
            )
    return {
        "state": "finished",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "records": records,
        "recognized_count": recognized,
        "unparsed_count": len(records) - recognized,
    }
