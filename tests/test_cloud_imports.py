import sys

from amphilens import cli, webapp


def test_importing_webapp_and_cli_does_not_load_modal_sdk_or_worker_entrypoint():
    assert cli is not None and webapp is not None
    assert "modal" not in sys.modules
    assert "amphilens.cloud.modal_app" not in sys.modules
