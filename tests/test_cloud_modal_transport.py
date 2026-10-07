import hashlib
import io
import json
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
        if recursive:
            matches = [
                name
                for name in self.files
                if name == path or name.startswith(path.rstrip("/") + "/")
            ]
            if not matches:
                raise FakeModalInvalidError("No such file or directory.")
            for name in matches:
                self.files.pop(name, None)
        else:
            self.files.pop(path, None)

    def commit(self):
        raise RuntimeError("commit() can only be called on a mounted volume inside a container")


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


class FakeModalInvalidError(Exception):
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
        exception=SimpleNamespace(
            NotFoundError=FakeModalNotFoundError,
            InvalidError=FakeModalInvalidError,
        ),
        FunctionCall=SimpleNamespace(from_id=lambda call_id, client=None: call),
        Volume=SimpleNamespace(from_name=lambda name, create_if_missing=False, client=None: volume),
    )
    transport._modal = modal
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


def test_modal_upload_replaces_stale_sentinel_without_reading_remote_payload(tmp_path, monkeypatch):
    transport, _, _, _, volume = make_transport(monkeypatch)
    source = tmp_path / "payload.zip"
    source.write_bytes(b"replacement")
    volume.files["jobs/key/payload.zip"] = b"old" * 1_000_000
    volume.files["jobs/key/payload.zip.sha256"] = b"old-digest\n"

    assert transport.upload(source, "jobs/key/payload.zip", "new-digest") is True
    assert volume.files["jobs/key/payload.zip"] == b"replacement"


def test_prediction_batch_upload_uses_opaque_ids_and_cleans_only_the_batch(tmp_path, monkeypatch):
    transport, _, _, _, volume = make_transport(monkeypatch)
    first = tmp_path / "local-camera-001.jpg"
    second = tmp_path / "local-camera-002.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    entries = transport.upload_prediction_batch("pred-abc", "batch-0001", [first, second])

    manifest_path = "prediction-jobs/pred-abc/batch-0001/manifest.json"
    manifest = json.loads(volume.files[manifest_path])
    assert len(entries) == 2
    assert manifest["images"] == entries
    assert str(first) not in volume.files[manifest_path].decode()
    for entry, source in zip(entries, [first, second], strict=True):
        assert volume.files[entry["remote_path"]] == source.read_bytes()
        assert volume.files[f"{entry['remote_path']}.sha256"] == (
            hashlib.sha256(source.read_bytes()).hexdigest().encode() + b"\n"
        )
    transport.cleanup_prediction_batch("pred-abc", "batch-0001")

    assert not any(name.startswith("prediction-jobs/pred-abc/batch-0001/") for name in volume.files)


def test_prediction_batch_cleanup_is_idempotent_after_worker_cleanup(monkeypatch):
    transport, _, _, _, _ = make_transport(monkeypatch)

    transport.cleanup_prediction_batch("pred-abc", "batch-already-removed")


def test_prediction_call_uses_selected_gpu_model_spec_and_one_container(monkeypatch):
    transport, _, _, app, _ = make_transport(monkeypatch)
    call = FakeCall(result={"state": "finished", "records": []})

    class FakePredictionMethod(FakeFunction):
        pass

    class FakeEngine:
        def __init__(self, model_spec_json):
            self.model_spec_json = model_spec_json
            self.predict_batch = FakePredictionMethod(call)

    class FakeEngineClass:
        options = None
        model_spec_json = None

        def with_options(self, **kwargs):
            self.options = kwargs
            return self

        def __call__(self, *, model_spec_json):
            self.model_spec_json = model_spec_json
            return FakeEngine(model_spec_json)

    engine_class = FakeEngineClass()
    module = SimpleNamespace(app=app, PredictionEngine=engine_class)
    monkeypatch.setattr(transport, "_load_app_module", lambda: module)
    payload = {
        "gpu": "A10",
        "timeout_seconds": 120,
        "model_spec": {"source": "hosted", "model_id": "public-model"},
        "images": [],
    }

    result = transport.predict_batch(payload)

    assert result["state"] == "finished"
    assert engine_class.options["gpu"] == "A10"
    assert engine_class.options["max_containers"] == 1
    assert json.loads(engine_class.model_spec_json) == payload["model_spec"]


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
