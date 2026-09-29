import json
from pathlib import Path

from amphilens.execution import JobSpec, LocalExecutionBackend


def test_local_execution_backend_persists_reproducible_job_state(tmp_path: Path):
    job = JobSpec(
        operation="predict",
        project_dir=tmp_path / "project",
        config={"model_id": "fixture", "seed": 42},
        model_ref="fixture-yolo",
    )
    backend = LocalExecutionBackend(tmp_path / "jobs")
    handle = backend.submit(job)

    assert handle.job_id == job.job_id
    saved = json.loads((tmp_path / "jobs" / job.job_id / "job.json").read_text())
    assert saved["operation"] == "predict"
    assert saved["config"]["seed"] == 42
    assert backend.status(job.job_id).state == "queued"


def test_job_status_reads_legacy_and_future_status_files(tmp_path: Path):
    job = JobSpec(operation="train", project_dir=tmp_path, config={})
    backend = LocalExecutionBackend(tmp_path / "jobs")
    backend.submit(job)

    legacy = {"job_id": job.job_id, "state": "queued", "updated_at": "now"}
    (tmp_path / "jobs" / job.job_id / "status.json").write_text(json.dumps(legacy))
    assert backend.status(job.job_id).phase == ""

    future = {
        **legacy,
        "phase": "training",
        "progress": 0.25,
        "provider_extension": "ignored",
    }
    (tmp_path / "jobs" / job.job_id / "status.json").write_text(json.dumps(future))
    status = backend.status(job.job_id)
    assert status.phase == "training"
    assert status.progress == 0.25
    assert not hasattr(status, "provider_extension")


def test_local_execution_backend_can_cancel_a_job(tmp_path: Path):
    job = JobSpec(operation="train", project_dir=tmp_path, config={})
    backend = LocalExecutionBackend(tmp_path / "jobs")
    backend.submit(job)

    assert backend.cancel(job.job_id).state == "canceled"
