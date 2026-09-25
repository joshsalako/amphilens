"""Initial dataset import from an existing CVAT project."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from ..core import ProjectStore, ValidationError
from ..dataset import DatasetSnapshot
from .managed import CVATProjectSummary, CVATTransport


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CVATProjectImportService:
    """List and import complete annotated CVAT projects into snapshots."""

    def __init__(self, store: ProjectStore, transport: CVATTransport):
        self.store = store
        self.transport = transport

    def list_projects(self) -> list[CVATProjectSummary]:
        return self.transport.list_projects()

    def get_project(self, project_id: str) -> CVATProjectSummary:
        return self.transport.get_project(str(project_id))

    def _validate_project(self, project: CVATProjectSummary, class_mapping: dict[str, str]) -> None:
        if not project.tasks:
            raise ValidationError(f"CVAT project {project.name!r} has no tasks or images")
        if not project.labels:
            raise ValidationError(f"CVAT project {project.name!r} has no labels")
        expected_classes = self.store.load_manifest().classes
        for source_label in project.labels:
            target_label = class_mapping.get(source_label, source_label)
            if target_label not in expected_classes:
                raise ValidationError(
                    f"CVAT project label {source_label!r} requires an explicit class mapping"
                )

    def _existing_snapshot(self, archive: Path) -> DatasetSnapshot | None:
        snapshot_path = self.store.root / "datasets" / f"snapshot-{_sha256(archive)[:12]}"
        if not snapshot_path.is_dir():
            return None
        return DatasetSnapshot.load(snapshot_path)

    def import_project(
        self,
        project_id: str,
        *,
        class_mapping: dict[str, str] | None = None,
    ) -> DatasetSnapshot:
        mapping = dict(class_mapping or {})
        project = self.get_project(project_id)
        self._validate_project(project, mapping)
        with tempfile.TemporaryDirectory(prefix="amphilens-cvat-project-") as temporary:
            archive = Path(temporary) / f"cvat-project-{project.project_id}.zip"
            exported = self.transport.export_project(
                project.project_id,
                archive,
                include_images=True,
            )
            exported = Path(exported).expanduser().resolve()
            if not exported.is_file():
                raise ValidationError(f"CVAT project export is missing: {exported}")
            existing = self._existing_snapshot(exported)
            if existing is not None:
                return existing
            provenance = {
                "source_type": "cvat-project",
                "server_url": str(getattr(self.transport, "server_url", "")),
                "project_id": project.project_id,
                "project_name": project.name,
                "task_ids": [task.task_id for task in project.tasks],
                "export_format": str(
                    getattr(self.transport, "export_format", "CVAT for images 1.1")
                ),
                "client_version": str(getattr(self.transport, "client_version", "unknown")),
                "archive_filename": exported.name,
                "archive_sha256": _sha256(exported),
            }
            return self.store.import_dataset(
                exported,
                class_mapping=mapping,
                source_provenance=provenance,
            )
