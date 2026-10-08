from pathlib import Path
from types import SimpleNamespace

import pytest

from amphilens.annotations.managed import CVATSdkTransport
from amphilens.core import ValidationError


class _Models:
    class ProjectWriteRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class TaskWriteRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class PatchedLabelRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


class _Project:
    def __init__(self, project_id, name, labels):
        self.id = project_id
        self.name = name
        self._labels = [SimpleNamespace(name=label) for label in labels]

    def get_labels(self):
        return self._labels


class _Task:
    def __init__(self, task_id, jobs):
        self.id = task_id
        self._jobs = jobs
        self.export_calls = []

    def get_jobs(self):
        return self._jobs

    def get_annotations(self):
        return SimpleNamespace(shapes=[1, 2], tracks=[], tags=[3])

    def export_dataset(self, format_name, filename, *, include_images, location):
        self.export_calls.append((format_name, Path(filename), include_images, location))
        Path(filename).write_bytes(b"zip fixture")


class _Client:
    def __init__(self):
        self.projects_created = []
        self.tasks_created = []
        self.projects = SimpleNamespace(list=lambda: [], create=self._create_project)
        self.tasks = SimpleNamespace(
            create_from_data=self._create_task,
            retrieve=lambda task_id: self.tasks_created[0],
        )

    def _create_project(self, request):
        project = _Project("7", request.name, [label.name for label in request.labels])
        self.projects_created.append(project)
        self.projects.list = lambda: list(self.projects_created)
        return project

    def _create_task(self, **kwargs):
        self.task_kwargs = kwargs
        task = _Task("42", [SimpleNamespace(id="99", state="completed")])
        self.tasks_created.append(task)
        return task


class _Context:
    def __init__(self, client):
        self.client = client

    def __enter__(self):
        return self.client

    def __exit__(self, *_args):
        return False


def _transport(client):
    transport = CVATSdkTransport("https://cvat.example/", token="secret")
    transport._load_sdk = lambda: (
        lambda host, access_token: _Context(client),
        _Models,
        SimpleNamespace(LOCAL="local"),
        SimpleNamespace(LOCAL="local"),
    )
    return transport


def test_sdk_transport_creates_reuses_project_and_exports_task(tmp_path: Path):
    client = _Client()
    transport = _transport(client)

    project_id = transport.ensure_project("study", ["toad", "frog"])
    assert project_id == "7"
    assert transport.ensure_project("study", ["toad", "frog"]) == "7"
    assert len(client.projects_created) == 1

    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    created = transport.create_task("7", "study-cycle-0", ["toad", "frog"], [first, second])
    assert created["task_id"] == "42"
    assert created["job_id"] == "99"
    assert client.task_kwargs["spec"].project_id == 7
    assert client.task_kwargs["resource_type"] == "local"
    assert client.task_kwargs["data_params"]["job_file_mapping"] == [["first.jpg", "second.jpg"]]

    status = transport.task_status("42")
    assert status == {"state": "completed", "annotation_count": 3, "job_states": ["completed"]}

    export = transport.export_task("42", tmp_path / "export.zip")
    assert export.read_bytes() == b"zip fixture"
    assert client.tasks_created[0].export_calls[0][0] == "CVAT for images 1.1"


def test_sdk_transport_rejects_project_label_drift(tmp_path: Path):
    client = _Client()
    existing = _Project("7", "study", ["toad"])
    client.projects.list = lambda: [existing]
    transport = _transport(client)

    with pytest.raises(ValidationError, match="label schema"):
        transport.ensure_project("study", ["toad", "frog"])


def test_sdk_transport_loads_credentials_from_dotenv(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CVAT_URL", raising=False)
    monkeypatch.delenv("CVAT_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "CVAT_URL=https://dotenv-cvat.example.org/\nCVAT_TOKEN=dotenv-fixture-token\n",
        encoding="utf-8",
    )

    transport = CVATSdkTransport()

    assert transport.server_url == "https://dotenv-cvat.example.org"
    assert transport.token == "dotenv-fixture-token"


def test_sdk_transport_environment_overrides_dotenv(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "CVAT_URL=https://dotenv-cvat.example.org\nCVAT_TOKEN=dotenv-fixture-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CVAT_URL", "https://process-cvat.example.org/")
    monkeypatch.setenv("CVAT_TOKEN", "process-fixture-token")

    transport = CVATSdkTransport()

    assert transport.server_url == "https://process-cvat.example.org"
    assert transport.token == "process-fixture-token"
