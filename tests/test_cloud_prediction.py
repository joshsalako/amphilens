from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from amphilens.cloud.prediction import (
    ModalPredictionDetector,
    PredictionBudgetReached,
    download_verified_hosted_checkpoint,
    run_remote_prediction_batch,
)
from amphilens.core import DetectionRecord, InferenceConfig
from amphilens.runs import run_resumable_inference


class FakeTransport:
    def __init__(self, result=None):
        self.result = result or {"state": "finished", "records": []}
        self.uploaded = []
        self.cleaned = []
        self.payload = None

    def upload_prediction_batch(self, job_key, batch_id, sources):
        self.uploaded.append((job_key, batch_id, list(sources)))
        return [
            {
                "image_id": f"local-{index}",
                "remote_path": f"prediction-jobs/{job_key}/{batch_id}/images/{index}.jpg",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for index, path in enumerate(sources)
        ]

    def predict_batch(self, payload, *, cancellation_requested=None, progress_callback=None):
        del cancellation_requested, progress_callback
        self.payload = payload
        return self.result

    def cleanup_prediction_batch(self, job_key, batch_id):
        self.cleaned.append((job_key, batch_id))


def test_modal_prediction_detector_maps_local_ids_and_cleans_batch(tmp_path):
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    paths = [first, second]
    result = {
        "state": "finished",
        "records": [
            {
                "local_image_id": "local-1",
                "class_id": 0,
                "class_name": "toad",
                "confidence": 0.9,
                "bbox_xyxy": [1, 2, 3, 4],
                "image_width": 10,
                "image_height": 20,
            }
        ],
        "elapsed_seconds": 2.5,
        "setup_seconds": 1.0,
    }
    transport = FakeTransport(result)
    detector = ModalPredictionDetector(
        transport,
        job_key="job123",
        gpu="L4",
        model_spec={"source": "hosted", "model_id": "fine-tuned"},
        timeout_seconds=100,
        max_cost_usd=1.0,
        output_dir=tmp_path / "run",
    )
    config = InferenceConfig(model_id="fine-tuned", run_id="run-1", batch_size=2)

    records = list(detector.predict(paths, config))

    assert [record.image_path for record in records] == [str(second.resolve())]
    assert records[0].image_id == "b.jpg"
    assert records[0].run_id == "run-1"
    assert transport.payload["gpu"] == "L4"
    assert len(transport.payload["images"]) == 2
    assert "image_path" not in json.dumps(transport.payload)
    assert transport.cleaned == [("job123", transport.uploaded[0][1])]
    assert (tmp_path / "run" / "cloud-cost.json").is_file()


def test_remote_prediction_verifies_hash_maps_ids_and_removes_batch(tmp_path):
    image_root = tmp_path / "prediction-jobs" / "job123" / "batch1" / "images"
    image_root.mkdir(parents=True)
    image = image_root / "local-7.jpg"
    image.write_bytes(b"image")
    digest = hashlib.sha256(b"image").hexdigest()
    manifest = {
        "images": [
            {
                "image_id": "local-7",
                "remote_path": str(image.relative_to(tmp_path)),
                "sha256": digest,
            }
        ]
    }
    (image_root.parent / "manifest.json").write_text(
        json.dumps({"schema_version": 1, **manifest}), encoding="utf-8"
    )
    Path(f"{image}.sha256").write_text(digest + "\n", encoding="ascii")
    worker_payload = {
        "job_key": "job123",
        "batch_id": "batch1",
        "images": manifest["images"],
        "config": {
            "model_id": "model-a",
            "image_size": 640,
            "confidence": 0.3,
            "batch_size": 1,
            "device": "cuda",
            "run_id": "run1",
            "preprocessing": {
                "short_side_dimension": 512,
                "grayscale_enabled": True,
                "clahe_enabled": True,
            },
        },
    }

    class Detector:
        model_id = "model-a"

        def predict(self, paths, config):
            assert config.device == "cuda"
            assert config.preprocessing_config.short_side_dimension == 512
            assert config.preprocessing_config.clahe_enabled is True
            return [
                DetectionRecord(
                    str(paths[0]),
                    paths[0].stem,
                    0,
                    "toad",
                    0.8,
                    [0, 0, 2, 2],
                    3,
                    3,
                    "model-a",
                    "run1",
                )
            ]

    result = run_remote_prediction_batch(
        worker_payload,
        volume_root=tmp_path,
        model_cache_root=tmp_path / "model-cache",
        detector=Detector(),
        volume_reload=lambda: None,
        volume_commit=lambda: None,
    )

    assert result["state"] == "finished"
    assert result["records"][0]["local_image_id"] == "local-7"
    assert "image_path" not in result["records"][0]
    assert not image_root.exists()


def test_remote_prediction_rejects_checksum_mismatch_and_cleans(tmp_path):
    image_root = tmp_path / "prediction-jobs" / "job123" / "batch1" / "images"
    image_root.mkdir(parents=True)
    image = image_root / "local-7.jpg"
    image.write_bytes(b"wrong")
    manifest = {
        "schema_version": 1,
        "images": [
            {
                "image_id": "local-7",
                "remote_path": str(image.relative_to(tmp_path)),
                "sha256": "0" * 64,
            }
        ],
    }
    (image_root.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    Path(f"{image}.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    payload = {
        "job_key": "job123",
        "batch_id": "batch1",
        "images": [
            {
                "image_id": "local-7",
                "remote_path": str(image.relative_to(tmp_path)),
                "sha256": "0" * 64,
            }
        ],
        "config": {"model_id": "model-a", "run_id": "run1"},
    }

    result = run_remote_prediction_batch(
        payload,
        volume_root=tmp_path,
        model_cache_root=tmp_path / "model-cache",
        detector=object(),
        volume_reload=lambda: None,
        volume_commit=lambda: None,
    )

    assert result["state"] == "failed"
    assert "SHA-256" in result["error"]
    assert not image_root.exists()


def test_hosted_checkpoint_download_uses_pinned_public_revision_and_verified_cache(tmp_path):
    content = b"verified model bytes"
    digest = hashlib.sha256(content).hexdigest()
    calls = []
    commits = []

    class Hosted:
        model_id = "public-model"
        repo_id = "org/public-model"
        artifact = "model.pt"
        sha256 = digest

        def to_summary(self):
            return {"revision": "a" * 40}

    def download(**kwargs):
        calls.append(kwargs)
        source = tmp_path / "downloaded.pt"
        source.write_bytes(content)
        return source

    model = Hosted()
    cache_root = tmp_path / "cache"
    first = download_verified_hosted_checkpoint(
        model, cache_root, hf_hub_download=download, cache_commit=lambda: commits.append(True)
    )
    second = download_verified_hosted_checkpoint(
        model,
        cache_root,
        hf_hub_download=lambda **kwargs: pytest.fail("verified cache should be reused"),
    )

    assert first == second
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["token"] is False
    assert hashlib.sha256(first.read_bytes()).hexdigest() == digest
    assert commits == [True]


def test_hosted_checkpoint_checksum_failure_does_not_enter_verified_cache(tmp_path):
    class Hosted:
        model_id = "public-model"
        repo_id = "org/public-model"
        artifact = "model.pt"
        sha256 = "0" * 64

        def to_summary(self):
            return {"revision": "b" * 40}

    source = tmp_path / "bad.pt"
    source.write_bytes(b"wrong")
    commits = []
    with pytest.raises(RuntimeError, match="SHA-256 verification"):
        download_verified_hosted_checkpoint(
            Hosted(),
            tmp_path / "cache",
            hf_hub_download=lambda **kwargs: source,
            cache_commit=lambda: commits.append(True),
        )
    assert not list((tmp_path / "cache" / "verified").glob("*.pt"))
    assert not commits


def test_hosted_checkpoint_download_supports_hub_versions_without_progress_hook(tmp_path):
    content = b"public checkpoint"
    digest = hashlib.sha256(content).hexdigest()
    source = tmp_path / "downloaded.pt"
    source.write_bytes(content)
    calls = []

    class Hosted:
        model_id = "public-model"
        repo_id = "org/public-model"
        artifact = "model.pt"
        sha256 = digest

        def to_summary(self):
            return {"revision": "c" * 40}

    def older_hf_hub_download(
        repo_id, filename, *, repo_type, revision, token, cache_dir
    ):
        calls.append((repo_id, filename, repo_type, revision, token, cache_dir))
        return source

    progress = []
    verified = download_verified_hosted_checkpoint(
        Hosted(),
        tmp_path / "remote-cache",
        hf_hub_download=older_hf_hub_download,
        progress_callback=progress.append,
    )

    assert verified.read_bytes() == content
    assert calls == [
        ("org/public-model", "model.pt", "model", "c" * 40, False,
         str((tmp_path / "remote-cache" / "huggingface").resolve()))
    ]
    assert progress[0]["phase"] == "model_download"


def test_modal_prediction_spending_limit_stops_scheduling_after_observed_batch_cost(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    transport = FakeTransport(
        {
            "state": "finished",
            "records": [],
            "elapsed_seconds": 12,
            "setup_seconds": 0,
        }
    )
    detector = ModalPredictionDetector(
        transport,
        job_key="budget-job",
        gpu="L4",
        model_spec={"source": "preset", "model_id": "yolo26-l"},
        timeout_seconds=100,
        max_cost_usd=0.003,
        output_dir=tmp_path / "run",
    )
    config = InferenceConfig(model_id="yolo26-l", run_id="run-budget", batch_size=1)

    list(detector.predict([first], config))
    with pytest.raises(PredictionBudgetReached, match="spending limit"):
        list(detector.predict([second], config))

    assert len(transport.uploaded) == 1
    cost = json.loads((tmp_path / "run" / "cloud-cost.json").read_text())
    assert cost["estimated_cost_usd"] > cost["max_cost_usd"]


def test_modal_batches_are_sequential_and_resume_from_local_progress(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    paths = []
    for index in range(5):
        path = image_root / f"{index}.jpg"
        path.write_bytes(f"image-{index}".encode())
        paths.append(path)
    output = tmp_path / "run"

    class SequencedTransport(FakeTransport):
        def __init__(self):
            super().__init__({"state": "finished", "records": [], "elapsed_seconds": 0})
            self.active_calls = 0
            self.max_active_calls = 0

        def predict_batch(self, payload, *, cancellation_requested=None, progress_callback=None):
            del cancellation_requested, progress_callback
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            self.payload = payload
            self.active_calls -= 1
            return self.result

    first_transport = SequencedTransport()
    first_detector = ModalPredictionDetector(
        first_transport,
        job_key="resume1",
        gpu="L4",
        model_spec={"source": "preset", "model_id": "yolo26-l"},
        timeout_seconds=100,
        max_cost_usd=5,
        output_dir=output,
        cancellation_requested=lambda: len(first_transport.uploaded) >= 2,
    )
    config = InferenceConfig(model_id="yolo26-l", run_id="resume-run", batch_size=2, device="cuda")

    partial = run_resumable_inference(first_detector, paths, config, output)

    assert partial.completed_images == 4
    assert partial.interruption_reason
    assert first_transport.max_active_calls == 1
    assert [len(batch[2]) for batch in first_transport.uploaded] == [2, 2]

    resume_transport = SequencedTransport()
    resumed_detector = ModalPredictionDetector(
        resume_transport,
        job_key="resume2",
        gpu="L4",
        model_spec={"source": "preset", "model_id": "yolo26-l"},
        timeout_seconds=100,
        max_cost_usd=5,
        output_dir=output,
    )
    complete = run_resumable_inference(resumed_detector, paths, config, output)

    assert complete.completed_images == 5
    assert len(resume_transport.uploaded) == 1
    assert len(resume_transport.uploaded[0][2]) == 1
