import json
from pathlib import Path

import pytest

from amphilens.state import UserStateError, UserStateStore, default_user_state_path


def test_default_user_state_path_uses_platform_configuration_directories(tmp_path: Path):
    assert default_user_state_path("darwin", home=tmp_path) == (
        tmp_path / "Library" / "Application Support" / "AmphiLens" / "state.json"
    )
    assert default_user_state_path("linux", home=tmp_path, environ={}) == (
        tmp_path / ".config" / "amphilens" / "state.json"
    )
    assert (
        default_user_state_path(
            "linux", home=tmp_path, environ={"XDG_CONFIG_HOME": str(tmp_path / "config")}
        )
        == tmp_path / "config" / "amphilens" / "state.json"
    )
    assert (
        default_user_state_path(
            "win32", home=tmp_path, environ={"APPDATA": str(tmp_path / "appdata")}
        )
        == tmp_path / "appdata" / "AmphiLens" / "state.json"
    )


def test_user_state_remembers_only_last_project_and_reloads_atomically(tmp_path: Path):
    state_path = tmp_path / "state.json"
    project = tmp_path / "project"
    project.mkdir()
    store = UserStateStore(state_path)

    store.remember_project(project)

    assert json.loads(state_path.read_text()) == {
        "schema_version": 1,
        "last_active_project": str(project.resolve()),
    }
    assert UserStateStore(state_path).last_active_project() == project.resolve()
    assert not list(tmp_path.glob("*.tmp"))


def test_user_state_clear_removes_project_without_leaving_credentials(tmp_path: Path):
    state_path = tmp_path / "state.json"
    store = UserStateStore(state_path)
    store.remember_project(tmp_path / "project")

    store.clear()

    assert UserStateStore(state_path).last_active_project() is None
    assert "CVAT_TOKEN" not in state_path.read_text()
    assert "project" not in state_path.read_text()


def test_user_state_rejects_unknown_schema_and_non_object_data(tmp_path: Path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"schema_version": 2}))

    with pytest.raises(UserStateError, match="schema version"):
        UserStateStore(state_path).last_active_project()

    state_path.write_text(json.dumps([]))
    with pytest.raises(UserStateError, match="JSON object"):
        UserStateStore(state_path).last_active_project()
