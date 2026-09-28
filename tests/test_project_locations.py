from pathlib import Path

import pytest

from amphilens.core import ProjectManifest, ProjectStore
from amphilens.locations import (
    ProjectLocationError,
    default_projects_root,
    open_project,
    relocate_project,
    validate_new_project_path,
)


def _create_project(root: Path, source: Path) -> ProjectStore:
    source.mkdir()
    (source / "image.jpg").write_bytes(b"fixture")
    store = ProjectStore(root)
    store.create(ProjectManifest.create("field study", [source], ["toad"]))
    (root / "artifacts" / "notes.txt").write_text("derived", encoding="utf-8")
    return store


def test_default_projects_root_is_user_data_not_current_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data"))

    result = default_projects_root(platform_name="linux")

    assert result == tmp_path / "user-data" / "AmphiLens" / "projects"
    assert result != Path.cwd() / "amphilens-project"


def test_new_project_path_rejects_source_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    with pytest.raises(ProjectLocationError, match="source checkout"):
        validate_new_project_path(checkout / "amphilens-project", source_checkout=checkout)


def test_new_project_path_accepts_external_location(tmp_path):
    checkout = tmp_path / "checkout"
    destination = tmp_path / "user-data" / "projects" / "study"
    checkout.mkdir()

    assert validate_new_project_path(destination, source_checkout=checkout) == destination


def test_open_project_requires_a_valid_manifest(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ProjectLocationError, match="manifest"):
        missing.mkdir()
        open_project(missing)


def test_relocate_project_verifies_copy_and_can_remove_source(tmp_path):
    source = tmp_path / "repo" / "amphilens-project"
    destination = tmp_path / "user-data" / "projects" / "amphilens-project"
    _create_project(source, tmp_path / "images")

    moved = relocate_project(source, destination, remove_source=True)

    assert moved == destination.resolve()
    assert not source.exists()
    assert (destination / "manifest.json").is_file()
    assert (destination / "artifacts" / "notes.txt").read_text(encoding="utf-8") == "derived"
    assert open_project(destination).load_manifest().name == "field study"
