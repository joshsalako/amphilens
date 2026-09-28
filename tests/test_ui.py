import runpy
from pathlib import Path

import pytest

from amphilens.annotations.managed import CVATProjectSummary, CVATTaskSummary
from amphilens.core import ProjectManifest, ProjectStore
from amphilens.ui import (
    active_project_path,
    choose_local_folder,
    cvat_project_choices,
    parse_class_mapping,
    parse_classes,
    preprocessing_from_controls,
    set_active_project,
)


def test_parse_classes_normalizes_and_rejects_duplicates():
    assert parse_classes("toad, frog\nother") == ["toad", "frog", "other"]
    with pytest.raises(ValueError, match="unique"):
        parse_classes("toad, toad")


def test_parse_classes_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        parse_classes("\n, ")


def test_ui_preprocessing_controls_build_reproducible_configuration():
    config = preprocessing_from_controls(max_dimension=320, grayscale=False, clahe=True)

    assert config.max_dimension == 320
    assert config.grayscale_enabled is False
    assert config.clahe_enabled is True


def test_parse_class_mapping_requires_object_json():
    assert parse_class_mapping('{"WLT": "toad"}') == {"WLT": "toad"}
    with pytest.raises(ValueError, match="JSON object"):
        parse_class_mapping('["toad"]')


def test_cvat_project_choices_are_readable_and_retain_project_ids():
    project = CVATProjectSummary(
        "17",
        "initial annotations",
        ["toad"],
        [CVATTaskSummary("23", "task one", 5, "completed")],
    )

    choices = cvat_project_choices([project])

    assert list(choices) == ["initial annotations (ID 17; 1 task)"]
    assert choices["initial annotations (ID 17; 1 task)"].project_id == "17"


def test_streamlit_script_can_load_ui_with_no_package_context():
    script = Path(__file__).parents[1] / "src" / "amphilens" / "ui.py"

    namespace = runpy.run_path(str(script), run_name="amphilens_ui_script")

    assert namespace["parse_classes"]("toad") == ["toad"]


def test_active_project_helpers_store_resolved_folder(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    (source / "image.jpg").write_bytes(b"fixture")
    project_dir = tmp_path / "project"
    ProjectStore(project_dir).create(ProjectManifest.create("study", [source], ["toad"]))
    state = {}

    set_active_project(state, project_dir)

    assert active_project_path(state) == project_dir.resolve()


def test_choose_local_folder_returns_normalized_selection(tmp_path):
    selected = choose_local_folder(ask_directory=lambda **_: str(tmp_path / "chosen"))

    assert selected == (tmp_path / "chosen").resolve()
