"""Initial dataset import from an existing CVAT project."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from ..core import ProjectStore, ValidationError
from ..dataset import DatasetImporter, DatasetSnapshot
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

    def import_project_tasks(
        self,
        project_id: str,
        *,
        class_mapping: dict[str, str] | None = None,
        display_name: str | None = None,
    ) -> list[DatasetSnapshot]:
        """Import each CVAT task as a separately selectable immutable snapshot."""
        mapping = dict(class_mapping or {})
        project = self.get_project(project_id)
        self._validate_project(project, mapping)
        empty_tasks = [task.name for task in project.tasks if task.size == 0]
        if empty_tasks:
            names = ", ".join(repr(name) for name in empty_tasks)
            raise ValidationError(f"CVAT task lists with no images cannot be imported: {names}")

        base_name = " ".join(str(display_name or project.name).split())
        importer = DatasetImporter()
        with tempfile.TemporaryDirectory(prefix="amphilens-cvat-tasks-") as temporary:
            temporary_root = Path(temporary)
            staged: list[tuple[DatasetSnapshot, str]] = []
            for task in project.tasks:
                task_identity = "\0".join(
                    (
                        str(getattr(self.transport, "server_url", "")),
                        project.project_id,
                        task.task_id,
                    )
                )
                task_key = hashlib.sha256(task_identity.encode("utf-8")).hexdigest()[:8]
                archive = temporary_root / f"task-{task_key}.zip"
                exported = Path(
                    self.transport.export_task(task.task_id, archive)
                ).expanduser().resolve()
                if not exported.is_file():
                    raise ValidationError(
                        f"CVAT task {task.name!r} did not produce an export archive"
                    )
                archive_hash = _sha256(exported)
                snapshot_id = f"snapshot-{archive_hash[:12]}-task-{task_key}"
                provenance = {
                    "source_type": "cvat-task",
                    "server_url": str(getattr(self.transport, "server_url", "")),
                    "project_id": project.project_id,
                    "project_name": project.name,
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "task_status": task.status,
                    "export_format": str(
                        getattr(self.transport, "export_format", "CVAT for images 1.1")
                    ),
                    "client_version": str(
                        getattr(self.transport, "client_version", "unknown")
                    ),
                    "archive_filename": exported.name,
                    "archive_sha256": archive_hash,
                }
                task_suffix = f" · {task.name or f'Task {task.task_id}'}"
                name_prefix = base_name[: max(0, 120 - len(task_suffix))].rstrip()
                task_display_name = f"{name_prefix}{task_suffix}".strip()[:120]
                staged_snapshot = importer.import_archive(
                    exported,
                    temporary_root / "snapshots" / "incoming",
                    classes=self.store.load_manifest().classes,
                    class_mapping=mapping,
                    source_provenance=provenance,
                    snapshot_id=snapshot_id,
                )
                staged.append((staged_snapshot, task_display_name))

            dataset_root = self.store.root / "datasets"
            dataset_root.mkdir(parents=True, exist_ok=True)
            snapshots: list[DatasetSnapshot] = []
            created_paths: list[Path] = []
            try:
                for staged_snapshot, task_display_name in staged:
                    destination = dataset_root / staged_snapshot.manifest.snapshot_id
                    if destination.is_dir():
                        snapshot = DatasetSnapshot.load(destination)
                        if (
                            snapshot.manifest.source_archive_sha256
                            != staged_snapshot.manifest.source_archive_sha256
                            or snapshot.manifest.source_provenance.get("project_id")
                            != staged_snapshot.manifest.source_provenance.get("project_id")
                            or snapshot.manifest.source_provenance.get("task_id")
                            != staged_snapshot.manifest.source_provenance.get("task_id")
                        ):
                            raise ValidationError(
                                "Existing snapshot identity does not match the CVAT task export"
                            )
                    else:
                        shutil.copytree(staged_snapshot.root, destination)
                        created_paths.append(destination)
                        snapshot = DatasetSnapshot.load(destination)
                    self.store.set_dataset_display_name(snapshot.root, task_display_name)
                    snapshots.append(snapshot)
            except Exception:
                for path in reversed(created_paths):
                    shutil.rmtree(path, ignore_errors=True)
                raise
        return snapshots
