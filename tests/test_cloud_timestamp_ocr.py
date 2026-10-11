from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from amphilens.cloud.timestamp_ocr import run_cloud_timestamp_ocr


def _image(path, *, width=800, height=600):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


def test_cloud_ocr_uploads_unique_bottom_crops_and_updates_local_csv(tmp_path):
    image_root = tmp_path / "images"
    first = image_root / "camera" / "first.jpg"
    second = image_root / "camera" / "second.jpg"
    _image(first)
    _image(second)
    original_bytes = {first: first.read_bytes(), second: second.read_bytes()}
    csv_path = tmp_path / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "class_name"])
        writer.writeheader()
        writer.writerows(
            [
                {"image_path": "camera/first.jpg", "class_name": "toad"},
                {"image_path": "camera/first.jpg", "class_name": "frog"},
                {"image_path": "camera/second.jpg", "class_name": "mouse"},
            ]
        )

    class FakeTransport:
        uploaded = None
        cleaned = []

        def run_timestamp_ocr(
            self,
            *,
            job_key,
            gpu,
            crops,
            progress_callback=None,
            cancellation_requested=None,
            job_id_callback=None,
        ):
            del cancellation_requested, job_id_callback
            self.uploaded = (job_key, gpu, dict(crops))
            for path in crops.values():
                crop = cv2.imread(str(path), cv2.IMREAD_COLOR)
                assert crop.shape[:2] == (168, 800)
            return {
                "state": "finished",
                "records": [
                    {
                        "image_id": image_id,
                        "date": "2026-10-09",
                        "time": "18:42:07",
                    }
                    for image_id in crops
                ],
                "recognized_count": 2,
                "unparsed_count": 0,
            }

        def cleanup_timestamp_ocr(self, job_key):
            self.cleaned.append(job_key)

    transport = FakeTransport()
    summary = run_cloud_timestamp_ocr(
        csv_path,
        image_root,
        transport=transport,
        gpu="L4",
    )

    job_key, gpu, crops = transport.uploaded
    assert job_key.startswith("ocr-")
    assert gpu == "L4"
    assert len(crops) == 2
    assert summary.image_count == 2
    assert summary.recognized_count == 2
    assert transport.cleaned == [job_key]
    assert {path: path.read_bytes() for path in original_bytes} == original_bytes
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["date"], row["time"]) for row in rows] == [
        ("2026-10-09", "18:42:07"),
        ("2026-10-09", "18:42:07"),
        ("2026-10-09", "18:42:07"),
    ]


def test_remote_timestamp_worker_verifies_crops_and_reports_progress(tmp_path):
    from amphilens.cloud.timestamp_ocr import run_remote_timestamp_ocr

    job_key = "ocr-test-job"
    image_id = "img-test-1"
    relative = Path("prediction-jobs") / job_key / "timestamp-ocr" / "images" / f"{image_id}.jpg"
    crop_path = tmp_path / relative
    _image(crop_path, width=800, height=168)
    digest = hashlib.sha256(crop_path.read_bytes()).hexdigest()
    (crop_path.parent / f"{crop_path.name}.sha256").write_text(digest, encoding="ascii")
    images = [
        {
            "image_id": image_id,
            "remote_path": relative.as_posix(),
            "sha256": digest,
        }
    ]
    manifest_path = crop_path.parents[1] / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "images": images}), encoding="utf-8"
    )
    payload = {"job_key": job_key, "images": images}
    processed_shapes = []
    progress = []

    result = run_remote_timestamp_ocr(
        payload,
        volume_root=tmp_path,
        ocr_engine=lambda crop: (
            processed_shapes.append(crop.shape[:2]) or "09/10/2026 18:42:07"
        ),
        progress_callback=progress.append,
    )

    assert result["state"] == "finished"
    assert result["records"] == [
        {"image_id": image_id, "date": "2026-10-09", "time": "18:42:07"}
    ]
    assert processed_shapes == [(288, 2400)]
    assert progress[-1]["completed"] == 1


def test_remote_timestamp_worker_rejects_crop_digest_mismatch(tmp_path):
    from amphilens.cloud.timestamp_ocr import run_remote_timestamp_ocr

    job_key = "ocr-test-job"
    image_id = "img-test-1"
    relative = Path("prediction-jobs") / job_key / "timestamp-ocr" / "images" / f"{image_id}.jpg"
    crop_path = tmp_path / relative
    _image(crop_path, width=800, height=168)
    digest = "0" * 64
    (crop_path.parent / f"{crop_path.name}.sha256").write_text(digest, encoding="ascii")
    images = [{"image_id": image_id, "remote_path": relative.as_posix(), "sha256": digest}]
    (crop_path.parents[1] / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "images": images}), encoding="utf-8"
    )

    try:
        run_remote_timestamp_ocr(
            {"job_key": job_key, "images": images},
            volume_root=tmp_path,
            ocr_engine=lambda _crop: "",
        )
    except Exception as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("a mismatched crop digest must fail")


def test_managed_cloud_transport_submits_crops_as_one_ocr_job(tmp_path):
    from amphilens.cloud.providers import _ManagedTransport

    crop_path = tmp_path / "img-one.jpg"
    _image(crop_path, width=800, height=168)

    class FakeTransport(_ManagedTransport):
        provider = "azure_ml"

        def __init__(self):
            super().__init__()
            self.objects = {}
            self.submitted = None
            self.remote_prefix = None

        @property
        def region(self):
            return "eastus"

        def _object_uri(self, key):
            return key

        def _put(self, source, key, sha256):
            self.objects[key] = source.read_bytes()
            return True

        def _get(self, key, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.objects[key])

        def _delete_prefix(self, prefix):
            return []

        def _delete_object(self, key):
            self.objects.pop(key, None)

        def _parse_remote_ref(self, remote_ref):
            return remote_ref

        def _submit_job(self, payload):
            self.submitted = payload
            return "remote-ocr-job"

        def _job_state(self, call_id):
            raise AssertionError("the test overrides poll")

        def _dashboard_url(self):
            return "https://example.invalid"

        def poll(self, call_id, *, remote_prefix):
            self.remote_prefix = remote_prefix
            return {
                "state": "finished",
                "records": [{"image_id": "img-one", "date": "", "time": ""}],
            }

    transport = FakeTransport()
    result = transport.run_timestamp_ocr(
        job_key="ocr-test-job",
        gpu="T4",
        crops={"img-one": crop_path},
    )

    assert result["state"] == "finished"
    assert transport.submitted["operation"] == "ocr"
    assert transport.submitted["gpu"] == "T4"
    assert transport.remote_prefix == "prediction-jobs/ocr-test-job/timestamp-ocr"
    uploaded_paths = set(transport.objects)
    assert "prediction-jobs/ocr-test-job/timestamp-ocr/images/img-one.jpg" in uploaded_paths
    assert all("first.jpg" not in path for path in uploaded_paths)
