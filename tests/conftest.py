from __future__ import annotations

import pytest


@pytest.fixture
def jobs():
    pytest.importorskip("starlette")
    from amphilens.webapp import JobManager

    manager = JobManager(max_workers=1, max_pending=3)
    try:
        yield manager
    finally:
        manager.shutdown()


@pytest.fixture
def client(tmp_path, jobs, monkeypatch):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient

    from amphilens.cloud.provider_settings import CloudProviderSettingsStore
    from amphilens.state import UserStateStore
    from amphilens.webapp import create_app

    monkeypatch.setattr(
        "amphilens.webapp.run_doctor",
        lambda *args, **kwargs: type("Report", (), {"to_dict": lambda self: {"ok": True}})(),
    )
    app = create_app(
        user_state_store=UserStateStore(tmp_path / "state.json"),
        cloud_provider_settings_store=CloudProviderSettingsStore(tmp_path / "cloud-providers.json"),
        jobs=jobs,
        source_checkout=tmp_path / "checkout",
    )
    return TestClient(app, client=("127.0.0.1", 50000))
