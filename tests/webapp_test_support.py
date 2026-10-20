from __future__ import annotations

import time

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient


def _wait_for_job(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["state"] in {"completed", "failed"}:
            return job
        time.sleep(0.01)
    pytest.fail("job did not finish before test deadline")
