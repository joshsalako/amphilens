import io
import sys
from contextlib import contextmanager
from types import SimpleNamespace

from amphilens.cloud.credentials import CloudCredentials
from amphilens.cloud.modal_transport import ModalTransport


class FakeVolume:
    def __init__(self):
        self.files = {}
        self.removed = []
        self.read_calls = []

    def read_file(self, path):
        self.read_calls.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        return iter([self.files[path]])

    def iterdir(self, path, recursive=False):
        del recursive
        return [SimpleNamespace(path=path)] if path in self.files else []

    @contextmanager
    def batch_upload(self, force=False):
        volume = self

        class Batch:
            def put_file(self, source, remote_path):
                if isinstance(source, io.BytesIO):
                    contents = source.getvalue()
                else:
                    from pathlib import Path

                    contents = Path(source).read_bytes()
                volume.files[remote_path.lstrip("/")] = contents

        yield Batch()

    def remove_file(self, path, recursive=False):
        self.removed.append((path, recursive))
        self.files.pop(path, None)


class FakeCall:
    def __init__(self, call_id="fc-123", result=None):
        self.object_id = call_id
        self.result = result
        self.canceled = []

    def get(self, timeout=None):
        if self.result is None:
            raise TimeoutError
        return self.result

    def cancel(self, terminate_containers=False):
        self.canceled.append(terminate_containers)


class FakeModalNotFoundError(Exception):
    pass


class FakeFunction:
    def __init__(self, call):
        self.call = call
        self.options = None
        self.payload = None

    def with_options(self, **kwargs):
        self.options = kwargs
        return self

    def spawn(self, payload):
        self.payload = payload
        return self.call


class FakeApp:
    def __init__(self):
        self.run_options = None

    @contextmanager
    def run(self, **kwargs):
        self.run_options = kwargs
        yield self


def make_transport(monkeypatch):
    credentials = CloudCredentials("token-id", "token-secret")
    transport = ModalTransport(credentials)
    call = FakeCall()
    function = FakeFunction(call)
    app = FakeApp()
    volume = FakeVolume()
    modal = SimpleNamespace(
        Client=SimpleNamespace(from_credentials=lambda token_id, secret: (token_id, secret)),
        exception=SimpleNamespace(NotFoundError=FakeModalNotFoundError),
        FunctionCall=SimpleNamespace(from_id=lambda call_id, client=None: call),
        Volume=SimpleNamespace(
            from_name=lambda name, create_if_missing=False, client=None: volume
        ),
    )
    app_module = SimpleNamespace(app=app, train=function)
    monkeypatch.setattr(transport, "_load_modal", lambda: modal)
    monkeypatch.setattr(transport, "_load_app_module", lambda: app_module)
    return transport, call, function, app, volume


def test_modal_transport_uses_detached_spawn_and_nonblocking_poll(monkeypatch):
    transport, call, function, app, volume = make_transport(monkeypatch)
    assert "token-secret" not in repr(transport)

    call_id = transport.submit(
        {
            "gpu": "L4",
            "timeout_seconds": 42,
            "run_id": "cloud-123",
        }
    )

    assert call_id == "fc-123"
    assert app.run_options["detach"] is True
    assert function.options["gpu"] == "L4"
    assert function.options["timeout"] == 42
    assert transport.poll(call_id, remote_prefix="jobs/key") is None
    assert "jobs/key/progress.json" not in volume.read_calls
    call.result = {"state": "finished"}
    assert transport.poll(call_id, remote_prefix="jobs/key") == {"state": "finished"}
    transport.cancel(call_id)
    assert call.canceled == [True]


def test_modal_upload_is_skipped_when_remote_payload_hash_matches(tmp_path, monkeypatch):
    transport, _, _, _, volume = make_transport(monkeypatch)
    source = tmp_path / "payload.zip"
    source.write_bytes(b"payload")

    assert transport.upload(source, "jobs/key/payload.zip", "digest") is True
    assert transport.upload(source, "jobs/key/payload.zip", "digest") is False
    assert volume.files["jobs/key/payload.zip"] == b"payload"
    assert volume.files["jobs/key/payload.zip.sha256"] == b"digest\n"


def test_modal_upload_replaces_stale_sentinel_without_reading_remote_payload(
    tmp_path, monkeypatch
):
    transport, _, _, _, volume = make_transport(monkeypatch)
    source = tmp_path / "payload.zip"
    source.write_bytes(b"replacement")
    volume.files["jobs/key/payload.zip"] = b"old" * 1_000_000
    volume.files["jobs/key/payload.zip.sha256"] = b"old-digest\n"

    assert transport.upload(source, "jobs/key/payload.zip", "new-digest") is True
    assert volume.files["jobs/key/payload.zip"] == b"replacement"


def test_modal_download_streams_volume_chunks_to_disk(tmp_path, monkeypatch):
    transport, _, _, _, volume = make_transport(monkeypatch)
    volume.files["results/best.pt"] = b"weights" * 100
    destination = tmp_path / "download.pt"

    transport.download("modal-volume://amphilens-cloud-training/results/best.pt", destination)

    assert destination.read_bytes() == b"weights" * 100


def test_modal_cleanup_treats_provider_not_found_as_already_cleaned(monkeypatch):
    transport, _, _, _, volume = make_transport(monkeypatch)
    transport._modal = SimpleNamespace(
        exception=SimpleNamespace(NotFoundError=FakeModalNotFoundError)
    )

    def list_missing(path, recursive=False):
        raise FakeModalNotFoundError("No such file or directory")

    def remove_unexpected(path, recursive=False):
        raise AssertionError("cleanup must not remove a path that is already absent")

    volume.iterdir = list_missing
    volume.remove_file = remove_unexpected
    result = transport.cleanup("missing-job")

    assert result == {"success": True, "removed": ["jobs/missing-job"], "error": ""}


def test_modal_import_is_deferred_until_first_provider_operation(monkeypatch):
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    credentials = CloudCredentials("token-id", "token-secret")
    transport = ModalTransport(credentials)
    assert "modal" not in sys.modules
    assert transport.describe()["provider"] == "modal"
