import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from amphilens.annotations.managed import ManagedCVATCycleService
from amphilens.core import ProjectManifest, ProjectStore, ValidationError


def _archive(path: Path) -> Path:
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


class FakeCVAT:
    def __init__(self, export: Path):
        self.export = export
        self.state = "annotation"
        self.created = []

    def ensure_project(self, name, labels):
        self.created.append(("project", name, labels))
        return "project-1"

    def create_task(self, project_id, name, labels, image_paths):
        self.created.append(("task", project_id, name, labels, list(image_paths)))
        return {"task_id": "task-1", "job_id": "job-1", "url": "http://cvat/task-1"}

    def task_status(self, task_id):
        return {"state": self.state, "annotation_count": 1}

    def export_task(self, task_id, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.export.read_bytes())
        return destination


def test_managed_cycle_is_idempotent_and_continues_into_merged_snapshot(tmp_path: Path):
    source = tmp_path / "images"
    source.mkdir()
    (source / "unlabelled.jpg").write_bytes(b"fixture")
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("study", [source], ["toad"]))
    initial = _archive(tmp_path / "initial.zip")
    parent = store.import_dataset(initial)
    fake = FakeCVAT(initial)
    service = ManagedCVATCycleService(store, fake)

    started = service.start(
        cycle=0, image_paths=[source / "unlabelled.jpg"], selection_hash="abc123"
    )
    repeated = service.start(
        cycle=0, image_paths=[source / "unlabelled.jpg"], selection_hash="abc123"
    )
    assert started == repeated
    assert fake.created[0] == ("project", "study", ["toad"])
    assert started.task_url == "http://cvat/task-1"

    with pytest.raises(ValidationError, match="not ready"):
        service.continue_cycle(0)

    fake.state = "completed"
    continued = service.continue_cycle(0)

    assert continued.manifest.source_format == "merged"
    saved = json.loads((store.root / "annotations" / "managed" / "cycle-0.json").read_text())
    assert saved["state"] == "continued"
    assert saved["task_id"] == "task-1"
    assert saved["selection_hash"] == "abc123"
    assert parent.root.is_dir()


def test_managed_cycle_rejects_selection_drift(tmp_path: Path):
    source = tmp_path / "images"
    source.mkdir()
    (source / "image.jpg").write_bytes(b"fixture")
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("study", [source], ["toad"]))
    fake = FakeCVAT(_archive(tmp_path / "export.zip"))
    service = ManagedCVATCycleService(store, fake)
    service.start(cycle=2, image_paths=[source / "image.jpg"], selection_hash="one")

    with pytest.raises(ValidationError, match="selection hash"):
        service.start(cycle=2, image_paths=[source / "image.jpg"], selection_hash="two")


def test_managed_cycle_without_parent_keeps_imported_snapshot(tmp_path: Path):
    source = tmp_path / "images"
    source.mkdir()
    selected = source / "image.jpg"
    selected.write_bytes(b"fixture")
    store = ProjectStore(tmp_path / "project")
    store.create(ProjectManifest.create("study", [source], ["toad"]))
    archive = _archive(tmp_path / "export.zip")
    fake = FakeCVAT(archive)
    service = ManagedCVATCycleService(store, fake)
    service.start(cycle=1, image_paths=[selected], selection_hash="one")
    fake.state = "completed"

    snapshot = service.continue_cycle(1)

    assert snapshot.root.is_dir()
    assert snapshot.root.is_relative_to(store.root / "datasets")
    assert snapshot.manifest.source_format == "cvat-xml"
    saved = json.loads((store.root / "annotations" / "managed" / "cycle-1.json").read_text())
    assert saved["merged_snapshot"].startswith("datasets/managed-cycle-1")
