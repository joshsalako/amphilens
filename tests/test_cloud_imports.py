import sys

import pytest

from amphilens import cli, webapp


def test_importing_webapp_and_cli_does_not_load_modal_sdk_or_worker_entrypoint():
    assert cli is not None and webapp is not None
    assert "modal" not in sys.modules
    assert "amphilens.cloud.modal_app" not in sys.modules


def test_prediction_modal_class_uses_a_supported_parameter_annotation():
    pytest.importorskip("modal")
    from amphilens.cloud import modal_app

    assert modal_app.PredictionEngine(
        model_spec_json="{}", job_key="test-job", gpu_type="L4"
    ) is not None


def test_prediction_image_adds_local_python_source_after_build_steps():
    pytest.importorskip("modal")
    from amphilens.cloud import modal_app

    class FakeImage:
        def __init__(self, steps=()):
            self.steps = list(steps)

        def env(self, values):
            del values
            assert not self.steps or self.steps[-1] != "local-source"
            return FakeImage([*self.steps, "env"])

        def add_local_python_source(self, module):
            assert module == "amphilens"
            return FakeImage([*self.steps, "local-source"])

    image = modal_app._build_prediction_image(FakeImage(), {"HF_HOME": "/models/hf"})

    assert image.steps == ["env", "local-source"]


def test_modal_timestamp_ocr_adds_rapidocr_to_its_worker_image_only():
    pytest.importorskip("modal")
    from amphilens.cloud import modal_app

    class FakeImage:
        def __init__(self, steps=()):
            self.steps = list(steps)

        def uv_pip_install(self, *packages):
            return FakeImage([*self.steps, *packages])

    prediction_image = FakeImage(["torch", "ultralytics"])

    ocr_image = modal_app._build_timestamp_ocr_image(prediction_image)

    assert prediction_image.steps == ["torch", "ultralytics"]
    assert ocr_image.steps == ["torch", "ultralytics", "rapidocr==3.10.0"]
