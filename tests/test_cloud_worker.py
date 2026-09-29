from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from amphilens.cloud.worker import _TailBuffer, safe_extract_payload


def _zip(entries: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return stream.getvalue()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_worker_extracts_payload_and_preserves_expected_layout(tmp_path):
    payload = _zip(
        {
            "dataset/dataset.yaml": b"path: /mnt/amphilens/jobs/key/dataset\n",
            "dataset/images/frog.jpg": b"image-bytes",
        }
    )

    root = safe_extract_payload(payload, tmp_path, expected_sha256=_digest(payload))

    assert (root / "dataset" / "dataset.yaml").read_bytes().startswith(b"path:")
    assert (root / "dataset" / "images" / "frog.jpg").read_bytes() == b"image-bytes"


def test_worker_extracts_a_verified_archive_from_disk_without_loading_it_all(tmp_path):
    payload = _zip({"dataset/dataset.yaml": b"path: /mnt/amphilens/jobs/key/dataset\n"})
    archive_path = tmp_path / "payload.zip"
    archive_path.write_bytes(payload)

    root = safe_extract_payload(archive_path, tmp_path / "job", expected_sha256=_digest(payload))

    assert (root / "dataset" / "dataset.yaml").is_file()


def test_worker_log_buffer_keeps_only_the_tail():
    logs = _TailBuffer(8)
    assert logs.write("first line\n") == 11
    assert logs.write("last line\n") == 10
    assert logs.getvalue() == "st line\n"


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "dataset/../../escape.txt"])
def test_worker_rejects_payload_paths_that_escape_job_root(tmp_path, name):
    payload = _zip({name: b"nope"})

    with pytest.raises(ValueError, match="unsafe"):
        safe_extract_payload(payload, tmp_path, expected_sha256=_digest(payload))


def test_worker_rejects_payload_digest_mismatch(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        safe_extract_payload(
            _zip({"dataset/dataset.yaml": b"x"}), tmp_path, expected_sha256="0" * 64
        )
