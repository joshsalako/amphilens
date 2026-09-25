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
