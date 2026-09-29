import sys

from amphilens import cli, ui


def test_importing_ui_and_cli_does_not_load_modal_sdk_or_worker_entrypoint():
    assert cli is not None and ui is not None
    assert "modal" not in sys.modules
    assert "amphilens.cloud.modal_app" not in sys.modules
