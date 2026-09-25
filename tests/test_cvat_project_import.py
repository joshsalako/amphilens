import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from amphilens.annotations.initial import CVATProjectImportService
from amphilens.annotations.managed import (
    CVATProjectSummary,
    CVATSdkTransport,
    CVATTaskSummary,
)
from amphilens.core import ProjectManifest, ProjectStore, ValidationError


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


class _Task:
    def __init__(self, task_id, name, size, status="annotation"):
        self.id = task_id
        self.name = name
        self.size = size
        self.status = status


class _Project:
    def __init__(self, project_id, name, labels, tasks):
        self.id = project_id
        self.name = name
        self._labels = [type("Label", (), {"name": label})() for label in labels]
        self._tasks = tasks
        self.export_calls = []

    def get_labels(self):
        return self._labels

    def get_tasks(self):
        return self._tasks

    def export_dataset(self, format_name, filename, *, include_images, location):
        self.export_calls.append((format_name, Path(filename), include_images, location))
        Path(filename).write_bytes(b"project export")


class _ClientContext:
    def __init__(self, client):
        self.client = client

    def __enter__(self):
        return self.client

    def __exit__(self, *_args):
        return False


class _ProjectClient:
    def __init__(self, project):
        self.project = project
        self.projects = type(
            "Projects",
            (),
            {
                "list": lambda inner: [self.project],
                "retrieve": lambda inner, project_id: self.project,
            },
        )()


def _sdk_transport(project):
    client = _ProjectClient(project)
    transport = CVATSdkTransport("https://cvat.example/", token="secret")
    transport._load_sdk = lambda: (
        lambda host, access_token: _ClientContext(client),
        _Models,
        type("ResourceType", (), {"LOCAL": "local"}),
        type("Location", (), {"LOCAL": "local"}),
    )
    return transport


def test_sdk_transport_lists_project_tasks_and_exports_complete_project(tmp_path: Path):
    project = _Project(
        7,
        "initial annotations",
        ["toad", "frog"],
        [_Task(11, "first task", 12, "completed"), _Task(12, "second task", 4)],
    )
    transport = _sdk_transport(project)

    projects = transport.list_projects()
    assert projects == [
        CVATProjectSummary(
            project_id="7",
            name="initial annotations",
            labels=["toad", "frog"],
            tasks=[
                CVATTaskSummary("11", "first task", 12, "completed"),
                CVATTaskSummary("12", "second task", 4, "annotation"),
            ],
        )
    ]
    assert transport.get_project("7") == projects[0]
    assert transport.list_project_tasks("7") == projects[0].tasks

    archive = transport.export_project("7", tmp_path / "initial.zip")
    assert archive.read_bytes() == b"project export"
    assert project.export_calls == [
        ("CVAT for images 1.1", tmp_path / "initial.zip", True, "local")
    ]


def test_sdk_transport_converts_project_access_failure_to_validation_error():
    class DeniedProjects:
        def list(self):
            raise RuntimeError("403 forbidden")

    client = type("Client", (), {"projects": DeniedProjects()})()
    transport = CVATSdkTransport("https://cvat.example/", token="secret")
    transport._load_sdk = lambda: (
        lambda host, access_token: _ClientContext(client),
        _Models,
        type("ResourceType", (), {"LOCAL": "local"}),
        type("Location", (), {"LOCAL": "local"}),
    )

    with pytest.raises(ValidationError, match="CVAT project listing failed"):
        transport.list_projects()


def _valid_project_archive(path: Path) -> Path:
    image = io.BytesIO()
    Image.new("RGB", (20, 10), color="black").save(image, format="JPEG")
    xml = """
      <annotations><meta><task><labels><label><name>toad</name></label></labels></task></meta>
      <image name="images/camera.jpg" width="20" height="10">
        <box label="toad" xtl="2" ytl="1" xbr="10" ybr="8"/>
      </image>
    </annotations>
    """
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("images/camera.jpg", image.getvalue())
        handle.writestr("annotations.xml", xml)
    return path


class _ImportTransport:
    server_url = "https://cvat.example"
    client_version = "cvat-sdk==2.76.0"

    def __init__(self, archive: Path, project: CVATProjectSummary):
        self.archive = archive
        self.project = project
        self.export_count = 0

    def get_project(self, project_id):
        assert str(project_id) == self.project.project_id
        return self.project

    def export_project(self, project_id, destination, *, include_images=True):
        assert str(project_id) == self.project.project_id
        assert include_images is True
        self.export_count += 1
        destination = Path(destination)
        destination.write_bytes(self.archive.read_bytes())
        return destination


def _store(tmp_path: Path) -> ProjectStore:
    (tmp_path / "images").mkdir()
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("study", [tmp_path / "images"], ["toad"]))
    return store


def test_project_import_downloads_validates_and_records_remote_provenance(tmp_path: Path):
    archive = _valid_project_archive(tmp_path / "export.zip")
    project = CVATProjectSummary(
        project_id="7",
        name="initial annotations",
        labels=["toad"],
        tasks=[CVATTaskSummary("11", "initial task", 1, "completed")],
    )
    transport = _ImportTransport(archive, project)
    snapshot = CVATProjectImportService(_store(tmp_path), transport).import_project("7")

    assert snapshot.manifest.source_format == "cvat-xml"
    assert snapshot.manifest.source_provenance == {
        "source_type": "cvat-project",
        "server_url": "https://cvat.example",
        "project_id": "7",
        "project_name": "initial annotations",
        "task_ids": ["11"],
        "export_format": "CVAT for images 1.1",
        "client_version": "cvat-sdk==2.76.0",
        "archive_filename": "cvat-project-7.zip",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    assert (snapshot.root / "source" / "cvat-project-7.zip").is_file()
    assert "secret" not in json.dumps(snapshot.manifest.to_dict())


def test_project_import_reuses_unchanged_snapshot_without_duplicate(tmp_path: Path):
    archive = _valid_project_archive(tmp_path / "export.zip")
    project = CVATProjectSummary(
        project_id="7",
        name="initial annotations",
        labels=["toad"],
        tasks=[CVATTaskSummary("11", "initial task", 1, "completed")],
    )
    transport = _ImportTransport(archive, project)
    service = CVATProjectImportService(_store(tmp_path), transport)

    first = service.import_project("7")
    second = service.import_project("7")

    assert second.root == first.root
    assert transport.export_count == 2
    assert len(list((first.root.parent).glob("snapshot-*"))) == 1


def test_project_import_rejects_empty_project_before_export(tmp_path: Path):
    project = CVATProjectSummary("7", "empty", ["toad"], [])
    transport = _ImportTransport(tmp_path / "unused.zip", project)

    with pytest.raises(ValidationError, match="no tasks or images"):
        CVATProjectImportService(_store(tmp_path), transport).import_project("7")
    assert transport.export_count == 0


def test_project_import_requires_explicit_mapping_for_unknown_cvat_label(tmp_path: Path):
    archive = _valid_project_archive(tmp_path / "export.zip")
    project = CVATProjectSummary(
        "7",
        "initial annotations",
        ["western leopard toad"],
        [CVATTaskSummary("11", "initial task", 1, "completed")],
    )
    transport = _ImportTransport(archive, project)

    with pytest.raises(ValidationError, match="explicit class mapping"):
        CVATProjectImportService(_store(tmp_path), transport).import_project("7")
