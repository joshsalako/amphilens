import inspect
import runpy
from contextlib import nullcontext
from pathlib import Path

import pytest

import amphilens.ui as ui
from amphilens.annotations.managed import CVATProjectSummary, CVATTaskSummary
from amphilens.core import ProjectManifest, ProjectStore
from amphilens.preprocessing import PreprocessingConfig
from amphilens.state import UserStateStore
from amphilens.ui import (
    _folder_input,
    _render_cloud_credentials,
    _render_environment,
    active_project_path,
    clear_active_project,
    cvat_project_choices,
    parse_class_mapping,
    parse_classes,
    preprocessing_from_controls,
    project_widget_key,
    restore_active_project,
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


def test_ui_preprocessing_overrides_preserve_unexposed_project_settings():
    base = PreprocessingConfig(
        max_dimension=640,
        resize_interpolation="bilinear",
        grayscale_enabled=True,
        clahe_enabled=True,
        color_space="lab",
        compatibility_mode="shortest-side",
    )

    config = preprocessing_from_controls(
        max_dimension=320,
        grayscale=False,
        clahe=False,
        base=base,
    )

    assert config.max_dimension == 320
    assert config.grayscale_enabled is False
    assert config.clahe_enabled is False
    assert config.resize_interpolation == "bilinear"
    assert config.color_space == "lab"
    assert config.compatibility_mode == "shortest-side"


def test_project_widget_keys_are_namespaced_by_project_and_configuration(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert project_widget_key(first, "study", "train-gray") != project_widget_key(
        second, "study", "train-gray"
    )
    assert project_widget_key(first, "study", "train-gray") != project_widget_key(
        first, "changed-study", "train-gray"
    )


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

    set_active_project(state, project_dir, user_state=UserStateStore(tmp_path / "state.json"))

    assert active_project_path(state) == project_dir.resolve()


def test_active_project_helpers_restore_and_clear_persisted_folder(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    (source / "image.jpg").write_bytes(b"fixture")
    project_dir = tmp_path / "project"
    ProjectStore(project_dir).create(ProjectManifest.create("study", [source], ["toad"]))
    user_state = UserStateStore(tmp_path / "state.json")
    user_state.remember_project(project_dir)

    state = {}
    restored = restore_active_project(state, user_state=user_state)

    assert restored == project_dir.resolve()
    assert active_project_path(state) == project_dir.resolve()

    clear_active_project(state, user_state=user_state)

    assert active_project_path(state) is None
    assert user_state.last_active_project() is None


def test_restore_active_project_clears_missing_project(tmp_path):
    user_state = UserStateStore(tmp_path / "state.json")
    user_state.remember_project(tmp_path / "deleted-project")
    state = {}

    assert restore_active_project(state, user_state=user_state) is None
    assert active_project_path(state) is None
    assert user_state.last_active_project() is None


def test_folder_input_uses_a_path_field_without_a_native_dialog():
    class FakeStreamlit:
        session_state = {}

        def text_input(self, label, *, key):
            assert label == "Project folder path"
            return "/tmp/amphilens-project"

        def button(self, *args, **kwargs):
            raise AssertionError("path inputs must not render a native folder button")

    value = _folder_input(
        FakeStreamlit(),
        label="Project folder path",
        state_key="project-folder",
    )

    assert value == "/tmp/amphilens-project"


def test_cloud_credentials_form_saves_password_fields_outside_project(
    monkeypatch, tmp_path
):
    from amphilens.cloud import credentials as cloud_credentials

    credential_path = tmp_path / "user-state" / "credentials.json"
    monkeypatch.setattr(
        cloud_credentials,
        "default_credentials_path",
        lambda: credential_path,
    )
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    class FakeStreamlit:
        session_state = {}

        def __init__(self):
            self.labels = []
            self.captions = []

        def caption(self, value):
            self.captions.append(value)

        def form(self, key):
            assert key == "modal-cloud-credentials"
            return nullcontext()

        def text_input(self, label, *, type, key):
            self.labels.append((label, type, key))
            return "token-id-value" if "id" in key else "token-secret-value"

        def form_submit_button(self, label):
            assert label == "Save Modal credentials"
            return True

        def success(self, message):
            assert "owner-only" in message

        def rerun(self):
            return None

        def error(self, message):
            raise AssertionError(message)

    st = FakeStreamlit()
    result = _render_cloud_credentials(st)

    assert [(label, field_type) for label, field_type, _ in st.labels] == [
        ("Modal token ID", "password"),
        ("Modal token secret", "password"),
    ]
    assert "token-secret-value" not in " ".join(st.captions)
    assert result is not None
    assert credential_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("check_cloud", [False, True])
def test_environment_page_probes_modal_only_after_explicit_click(monkeypatch, check_cloud):
    from amphilens import doctor

    calls = []

    class Report:
        def to_dict(self):
            return {"torch_installed": True, "modal_connectivity": "not checked"}

    def fake_run_doctor(path, *, check_cloud=False):
        calls.append(check_cloud)
        return Report()

    class FakeStreamlit:
        def header(self, _label):
            return None

        def button(self, label):
            assert label == "Check Modal connectivity"
            return check_cloud

        def json(self, _value):
            return None

        def caption(self, _value):
            return None

    monkeypatch.setattr(doctor, "run_doctor", fake_run_doctor)
    _render_environment(FakeStreamlit())

    assert calls == [check_cloud]


def test_ui_contains_no_tkinter_or_native_window_code():
    source = inspect.getsource(ui)

    assert "tkinter" not in source
    assert "tk.Tk" not in source
