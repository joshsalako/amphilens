import runpy
from pathlib import Path

import pytest

from amphilens.annotations.managed import CVATProjectSummary, CVATTaskSummary
from amphilens.ui import (
    cvat_project_choices,
    parse_class_mapping,
    parse_classes,
    preprocessing_from_controls,
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
