import hashlib
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

from amphilens.cloud.models import CloudConsent
from amphilens.cloud.pricing import ProviderRate
from amphilens.cloud.service import CloudTrainingService
from amphilens.core import ProjectManifest, ProjectStore, atomic_write_json
from amphilens.dataset import DatasetImage, DatasetManifest


class FakeCloudTransport:
    def __init__(self):
        self.provider = "modal"
        self.uploaded = {}
        self.downloads = {}
        self.submissions = []
        self.canceled = []
        self.cleanup_calls = []
        self.poll_result = None
        self.poll_error = None
        self.cleanup_success = True
        self.cancel_error = None

    def describe(self):
        return {"provider": self.provider}

    def price_rate(self, gpu):
        return ProviderRate(
            provider=self.provider,
            gpu=gpu,
            region="us-central1",
            usd_per_hour=1.0,
            price_source="fake catalog",
            rate_checked_at="2026-10-09",
        )

    def upload(self, source, remote_path, sha256):
        self.uploaded[remote_path] = (Path(source).read_bytes(), sha256)
        return True

    def submit(self, payload):
        self.submissions.append(payload)
        return f"call-{len(self.submissions)}"

    def poll(self, call_id, *, remote_prefix):
        if self.poll_error is not None:
            raise self.poll_error
        return self.poll_result

    def cancel(self, call_id):
        if self.cancel_error is not None:
            raise self.cancel_error
        self.canceled.append(call_id)

    def download(self, remote_ref, destination):
        Path(destination).write_bytes(self.downloads[remote_ref])

    def cleanup(self, job_key):
        self.cleanup_calls.append(job_key)
        return {"success": self.cleanup_success, "removed": [job_key], "error": ""}

    def dashboard_url(self, call_id):
        return f"https://modal.com/apps/amphilens-cloud-training/calls/{call_id}"


def make_project(tmp_path: Path):
    image_root = tmp_path / "source-images"
    image_root.mkdir()
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("field-study", [image_root], ["frog"]))
    snapshot_root = store.root / "datasets" / "snapshot-001"
    (snapshot_root / "images").mkdir(parents=True)
    image_path = snapshot_root / "images" / "frog.jpg"
    Image.new("RGB", (16, 12), (10, 20, 30)).save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest = DatasetManifest(
        snapshot_id="snapshot-001",
        source_format="fixture",
        classes=["frog"],
        images=[
            DatasetImage(
                image_id="frog-001",
                relative_path="images/frog.jpg",
                source_path=str(image_root / "frog.jpg"),
                width=16,
                height=12,
                sha256=digest,
                reviewed=True,
                annotations=[],
            )
        ],
        source_archive="fixture.zip",
        source_archive_sha256="0" * 64,
    )
    atomic_write_json(snapshot_root / "manifest.json", manifest.to_dict())
    return store, snapshot_root


def config():
    return {
        "model_preset": "yolo26-l",
        "model_id": "yolo26-l",
        "architecture": "yolo",
        "classes": ["frog"],
        "preprocessing": {
            "max_dimension": 4096,
            "resize_enabled": False,
            "resize_interpolation": "lanczos",
            "grayscale_enabled": False,
            "clahe_enabled": False,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": [8, 8],
            "color_space": "rgb",
            "compatibility_mode": "max-dimension",
        },
        "image_size": 640,
        "confidence": 0.25,
        "device": "cuda",
        "epochs": 2,
        "batch_size": 1,
        "patience": 0,
        "seed": 42,
        "freeze_strategy": "none",
        "source": "project",
        "fingerprint": "effective-fingerprint",
    }


def consent():
    return CloudConsent(
        acknowledged=True,
        uploads_dataset=True,
        estimated_usd=0.04,
        max_cost_usd=5.0,
    )


def test_cloud_estimate_rejects_zero_epochs_instead_of_using_project_default(tmp_path: Path):
    store, snapshot = make_project(tmp_path)
    service = CloudTrainingService(store, FakeCloudTransport())

    with pytest.raises(ValueError, match="must be positive"):
        service.estimate(snapshot, epochs=0)


def test_cloud_estimate_uses_padded_training_canvas_size(tmp_path: Path):
    from amphilens.cloud.estimate import estimate_training_cost

    store, snapshot_path = make_project(tmp_path)
    service = CloudTrainingService(store, FakeCloudTransport())
    snapshot = service.estimate(snapshot_path, epochs=2000, image_size=640)
    image_bytes = sum(
        path.stat().st_size for path in (snapshot_path / "images").iterdir() if path.is_file()
    )
    expected = estimate_training_cost(
        image_count=1,
        dataset_bytes=image_bytes,
        epochs=2000,
        gpu="L4",
        max_cost_usd=5,
        image_size=864,
    )

    assert snapshot == expected


def test_cloud_submit_rejects_stale_consent_estimate_before_upload(tmp_path: Path):
    store, snapshot = make_project(tmp_path)
    transport = FakeCloudTransport()
    service = CloudTrainingService(store, transport)
    stale_consent = CloudConsent(
        acknowledged=True,
        uploads_dataset=True,
        estimated_usd=0,
        max_cost_usd=5,
    )

    with pytest.raises(ValueError, match="estimate is stale"):
        service.submit(
            snapshot_path=snapshot,
            effective_configuration=config(),
            training_config={"epochs": 2},
            consent=stale_consent,
        )

    assert transport.uploaded == {}


def complete_remote_run(transport: FakeCloudTransport, payload: dict):
    best = b"remote-best-weights"
    artifacts = {
        "best.pt": best,
        "last.pt": b"remote-last-weights",
        "metrics.json": b'{"evaluation":"not evaluated"}\n',
    }
    cloud_info = payload["echo"]
    checkpoint = {
        "checkpoint_path": "/mnt/amphilens/jobs/job-key/results/best.pt",
        "model_id": payload["effective_configuration"]["model_id"],
        "architecture": payload["effective_configuration"]["architecture"],
        "classes": payload["effective_configuration"]["classes"],
        "preprocessing": payload["effective_configuration"]["preprocessing"],
        "sha256": hashlib.sha256(best).hexdigest(),
        "created_at": "2026-09-29T00:00:00+00:00",
        "parent_checkpoint": None,
        "software": {"python": "3.11"},
        "training_config": {
            **payload["training_config"],
            "cloud": {**cloud_info, "code_version": "working-tree"},
        },
        "schema_version": 1,
    }
    artifacts["checkpoint.json"] = (json.dumps(checkpoint, sort_keys=True) + "\n").encode()
    remote_artifacts = {}
    for name, content in artifacts.items():
        remote_ref = f"remote/{name}"
        transport.downloads[remote_ref] = content
        remote_artifacts[name] = {
            "remote_ref": remote_ref,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    transport.poll_result = {
        "state": "finished",
        "remote_state": "success",
        "progress": 1.0,
        "echo": cloud_info,
        "artifacts": remote_artifacts,
        "environment": {"python": "3.11", "torch": "2.x"},
        "log_tail": "training complete",
    }


def test_cloud_submit_requires_explicit_consent_before_upload(tmp_path: Path):
    store, snapshot = make_project(tmp_path)
    transport = FakeCloudTransport()
    service = CloudTrainingService(store, transport)

    with pytest.raises(ValueError, match="consent"):
        service.submit(
            snapshot_path=snapshot,
            effective_configuration=config(),
            training_config={"epochs": 2},
            consent=None,
        )

    assert transport.uploaded == {}
    assert transport.submissions == []


def test_cloud_training_persists_the_selected_transport_provider(tmp_path: Path):
    store, snapshot = make_project(tmp_path)
    transport = FakeCloudTransport()
    transport.provider = "vertex_ai"
    service = CloudTrainingService(store, transport)

    record = service.submit(
        snapshot_path=snapshot,
        effective_configuration=config(),
        training_config={"epochs": 1, "image_size": 64},
        consent=consent(),
        gpu="T4",
        data_mode="training-monitor",
    )

    assert record.provider == "vertex_ai"
    assert record.region == "us-central1"
    assert transport.submissions[0]["provider"] == "vertex_ai"

    resumed = service.submit(
        snapshot_path=snapshot,
        effective_configuration=config(),
        training_config={"epochs": 1, "image_size": 64},
        consent=consent(),
        gpu="T4",
        data_mode="training-monitor",
    )
    assert resumed.run_id == record.run_id
    assert len(transport.submissions) == 1


def test_cloud_job_record_without_provider_defaults_to_modal(tmp_path: Path):
    from amphilens.cloud.models import CloudJobRecord

    store, snapshot = make_project(tmp_path)
    service = CloudTrainingService(store, FakeCloudTransport())
    record = service.submit(
        snapshot_path=snapshot,
        effective_configuration=config(),
        training_config={"epochs": 1, "image_size": 64},
        consent=consent(),
        gpu="T4",
        data_mode="training-monitor",
    )
    legacy_payload = record.to_dict()
    legacy_payload.pop("provider")

    assert CloudJobRecord.from_dict(legacy_payload).provider == "modal"


def test_cloud_training_blocks_before_upload_when_provider_price_is_unknown(tmp_path: Path):
    store, snapshot = make_project(tmp_path)
    transport = FakeCloudTransport()
    transport.provider = "azure_ml"

    def unknown_rate(_gpu):
        raise ValueError("current Azure GPU rate is unavailable; submission is blocked")

    transport.price_rate = unknown_rate
    service = CloudTrainingService(store, transport)

    with pytest.raises(ValueError, match="submission is blocked"):
        service.submit(
            snapshot_path=snapshot,
            effective_configuration=config(),
            training_config={"epochs": 1, "image_size": 64},
            consent=consent(),
            gpu="T4",
            data_mode="training-monitor",
        )

    assert transport.uploaded == {}
    assert transport.submissions == []


def test_cloud_job_record_redacts_remote_tokens_and_signed_urls(tmp_path: Path):
    from amphilens.cloud.provider_settings import AzureMLSettings
    from amphilens.cloud.providers import AzureMLTransport

    store, snapshot = make_project(tmp_path)
    transport = FakeCloudTransport()
    transport.redact = AzureMLTransport(
        AzureMLSettings("sub", "rg", "workspace", "eastus", "identity")
    ).redact
    service = CloudTrainingService(store, transport)
    record = service.submit(
        snapshot_path=snapshot,
        effective_configuration=config(),
        training_config={"epochs": 1, "image_size": 64},
        consent=consent(),
        gpu="T4",
        data_mode="training-monitor",
    )
    transport.poll_result = {
        "state": "finished",
        "progress_details": {
            "message": "Bearer access-secret",
            "log_tail": "https://storage.example/file?sig=signed-secret",
        },
    }

    refreshed = service.refresh(record.run_id)
    saved = (store.root / "runs" / record.run_id / "cloud-job.json").read_text(encoding="utf-8")

    assert refreshed.state == "finished"
    assert "access-secret" not in saved
    assert "signed-secret" not in saved
    assert "[redacted]" in saved


def test_cloud_training_yaml_uses_train_monitor_without_final_evaluation(tmp_path: Path):
    store, snapshot = make_project(tmp_path)
    transport = FakeCloudTransport()
    service = CloudTrainingService(store, transport)

    record = service.submit(
        snapshot_path=snapshot,
        effective_configuration=config(),
        training_config={"epochs": 1, "image_size": 64},
        consent=consent(),
        gpu="T4",
        data_mode="training-monitor",
    )

    payload_bytes, _ = next(
        value for path, value in transport.uploaded.items() if path.endswith("payload.zip")
    )
    with zipfile.ZipFile(io.BytesIO(payload_bytes)) as archive:
        dataset = json.loads(archive.read("dataset/dataset.yaml"))

    assert dataset["train"] == "images/train"
    assert dataset["val"] == "images/validation"
    assert record.training_config["val"] is True
    assert record.training_config["evaluation"] == "not evaluated"


def test_cloud_run_can_resume_after_restart_and_register_only_local_verified_bytes(
    tmp_path: Path,
):
    store, snapshot = make_project(tmp_path)
    transport = FakeCloudTransport()
    first_service = CloudTrainingService(store, transport)
    record = first_service.submit(
        snapshot_path=snapshot,
        effective_configuration=config(),
        training_config={"epochs": 2, "image_size": 640},
        consent=consent(),
        gpu="L4",
    )

    restarted_service = CloudTrainingService(store, transport)
    assert restarted_service.get_job(record.run_id).call_id == "call-1"
    assert restarted_service.list_jobs()[0].run_id == record.run_id
    complete_remote_run(transport, transport.submissions[0])
    assert restarted_service.refresh(record.run_id).state == "finished"

    verified = restarted_service.collect(record.run_id)
    best_path = store.root / "runs" / record.run_id / "results" / "best.pt"
    assert verified.state == "verified"
    assert verified.cleanup_succeeded is True
    assert verified.training_config["cloud"]["job_key"] == record.job_key
    assert verified.checkpoint_manifest.sha256 == hashlib.sha256(best_path.read_bytes()).hexdigest()
    indexed = {item.relative_path for item in store.load_artifact_index()}
    assert f"runs/{record.run_id}/results/best.pt" in indexed
    assert f"checkpoints/{record.run_id}/best.pt" in indexed


def test_cloud_collect_retries_a_transient_artifact_transfer_failure(tmp_path: Path):
    store, snapshot = make_project(tmp_path)
    transport = FakeCloudTransport()
    service = CloudTrainingService(store, transport)
    record = service.submit(
        snapshot_path=snapshot,
        effective_configuration=config(),
        training_config={"epochs": 2, "evaluation": "test set"},
        consent=consent(),
    )
    complete_remote_run(transport, transport.submissions[0])
    original_download = transport.download
    best_attempts = 0
    fail_last_once = True
    successful_best_downloads = 0

    def transient_download_failure(remote_ref, destination):
        nonlocal best_attempts, fail_last_once, successful_best_downloads
        if remote_ref == "remote/best.pt" and best_attempts == 0:
            best_attempts += 1
            Path(destination).write_bytes(b"transiently corrupted transfer")
            return
        if remote_ref == "remote/best.pt":
            successful_best_downloads += 1
        if remote_ref == "remote/last.pt" and fail_last_once:
            fail_last_once = False
            Path(destination).write_bytes(b"partial transfer")
            raise OSError("temporary storage DNS failure")
        original_download(remote_ref, destination)

    transport.download = transient_download_failure
    service.refresh(record.run_id)

    with pytest.raises(ValueError, match="hash mismatch"):
        service.collect(record.run_id)
    assert service.get_job(record.run_id).state == "incomplete"

    with pytest.raises(OSError, match="DNS failure"):
        service.collect(record.run_id)
    assert service.get_job(record.run_id).state == "incomplete"
    assert successful_best_downloads == 1

    verified = service.collect(record.run_id)

    assert verified.state == "verified"
    assert successful_best_downloads == 1
    assert not list((store.root / "runs" / record.run_id / "results").glob(".*.tmp"))
    assert verified.training_config["cloud"]["evaluation"] == "test set"
    assert verified.training_config["cloud"]["checkpoint_selection"] == "best-validation"


def test_identical_cloud_submit_reuses_job_and_detects_record_key_conflicts(tmp_path: Path):
    store, snapshot = make_project(tmp_path)
    transport = FakeCloudTransport()
    service = CloudTrainingService(store, transport)
    arguments = {
        "snapshot_path": snapshot,
        "effective_configuration": config(),
        "training_config": {"epochs": 2, "image_size": 640},
        "consent": consent(),
        "gpu": "L4",
    }
    first = service.submit(**arguments)
    second = service.submit(**arguments)
    assert second.run_id == first.run_id
    assert len(transport.submissions) == 1

    path = store.root / "runs" / first.run_id / "cloud-job.json"
    data = json.loads(path.read_text())
    data["snapshot_id"] = "different-snapshot"
    atomic_write_json(path, data)
    with pytest.raises(ValueError, match="does not match"):
        service.submit(**arguments)


def test_cloud_timeout_cancels_remote_call_after_server_deadline(tmp_path: Path):
    store, snapshot = make_project(tmp_path)
    transport = FakeCloudTransport()
    now = [datetime(2026, 9, 29, tzinfo=timezone.utc)]
    service = CloudTrainingService(store, transport, now=lambda: now[0])
    record = service.submit(
        snapshot_path=snapshot,
        effective_configuration=config(),
        training_config={"epochs": 2},
        consent=consent(),
    )
    now[0] += timedelta(seconds=record.timeout_seconds + 1)

    updated = service.refresh(record.run_id)

    assert updated.state == "timed_out"
    assert transport.canceled == [record.call_id]


def test_failed_deadline_cancellation_keeps_job_active_for_safe_retry(tmp_path: Path):
    store, snapshot = make_project(tmp_path)
    transport = FakeCloudTransport()
    now = [datetime(2026, 9, 29, tzinfo=timezone.utc)]
    service = CloudTrainingService(store, transport, now=lambda: now[0])
    record = service.submit(
        snapshot_path=snapshot,
        effective_configuration=config(),
        training_config={"epochs": 2},
        consent=consent(),
    )
    now[0] += timedelta(seconds=record.timeout_seconds + 1)
    transport.cancel_error = RuntimeError("provider unavailable")

    updated = service.refresh(record.run_id)

    assert updated.state == "submitted"
    assert updated.phase == "deadline-cancel-pending"
    assert updated.cleanup_succeeded is None
    assert "provider unavailable" in updated.error


def test_status_poll_failure_does_not_claim_the_remote_job_has_stopped(tmp_path: Path):
    store, snapshot = make_project(tmp_path)
    transport = FakeCloudTransport()
    service = CloudTrainingService(store, transport)
    record = service.submit(
        snapshot_path=snapshot,
        effective_configuration=config(),
        training_config={"epochs": 2},
        consent=consent(),
    )
    transport.poll_error = RuntimeError("provider temporarily unavailable")

    updated = service.refresh(record.run_id)

    assert updated.state == "submitted"
    assert updated.phase == "status-poll-failed"
    assert updated.cleanup_succeeded is None


@pytest.mark.parametrize("failure", ["missing-last", "bad-hash", "bad-echo"])
def test_unverified_cloud_results_are_incomplete_and_never_registered(tmp_path: Path, failure: str):
    store, snapshot = make_project(tmp_path)
    transport = FakeCloudTransport()
    service = CloudTrainingService(store, transport)
    record = service.submit(
        snapshot_path=snapshot,
        effective_configuration=config(),
        training_config={"epochs": 2},
        consent=consent(),
    )
    complete_remote_run(transport, transport.submissions[0])
    if failure == "missing-last":
        del transport.poll_result["artifacts"]["last.pt"]
    elif failure == "bad-hash":
        transport.poll_result["artifacts"]["best.pt"]["sha256"] = "f" * 64
    else:
        transport.poll_result["echo"]["effective_fingerprint"] = "wrong"
    service.refresh(record.run_id)

    with pytest.raises((ValueError, RuntimeError)):
        service.collect(record.run_id)

    assert service.get_job(record.run_id).state == "incomplete"
    assert store.load_artifact_index() == []
