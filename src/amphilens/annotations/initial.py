"""Initial dataset import from an existing CVAT project."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..cancellation import check_cancellation
from ..core import ProjectStore, ValidationError, atomic_write_json, read_json
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
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> list[DatasetSnapshot]:
        """Import each CVAT task as a separately selectable immutable snapshot."""
        check_cancellation(
            progress_callback,
            "CVAT import canceled before export started.",
        )
        mapping = dict(class_mapping or {})
        project = self.get_project(project_id)
        check_cancellation(
            progress_callback,
            "CVAT import canceled after loading project tasks.",
        )
        self._validate_project(project, mapping)
        empty_tasks = [task.name for task in project.tasks if task.size == 0]
        if empty_tasks:
            names = ", ".join(repr(name) for name in empty_tasks)
            raise ValidationError(f"CVAT task lists with no images cannot be imported: {names}")

        base_name = " ".join(str(display_name or project.name).split())
        importer = DatasetImporter()
        task_count = len(project.tasks)
        dataset_root = self.store.root / "datasets"
        dataset_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".amphilens-cvat-import-", dir=dataset_root
        ) as temporary:
            temporary_root = Path(temporary)
            archive_root = temporary_root / "archives"
            archive_root.mkdir()
            snapshot_parent = temporary_root / "snapshots"
            staged: list[tuple[DatasetSnapshot, str]] = []
            for task_index, task in enumerate(project.tasks, start=1):
                check_cancellation(
                    progress_callback,
                    f"CVAT import canceled before exporting task {task.name}.",
                )
                task_identity = "\0".join(
                    (
                        str(getattr(self.transport, "server_url", "")),
                        project.project_id,
                        task.task_id,
                    )
                )
                task_key = hashlib.sha256(task_identity.encode("utf-8")).hexdigest()[:8]
                archive = archive_root / f"task-{task_key}.zip"
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "cvat-export",
                            "message": (
                                f"Exporting CVAT task {task.name} "
                                f"({task_index} of {task_count}) with images"
                            ),
                            "completed": task_index - 1,
                            "total": task_count,
                            "progress": (task_index - 1) / task_count,
                        }
                    )
                exported = Path(
                    self.transport.export_task(task.task_id, archive)
                ).expanduser().resolve()
                check_cancellation(
                    progress_callback,
                    f"CVAT import canceled after exporting task {task.name}.",
                )
                if not exported.is_file():
                    raise ValidationError(
                        f"CVAT task {task.name!r} did not produce an export archive"
                    )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "cvat-export",
                            "message": (
                                f"Exported CVAT task {task.name} "
                                f"({task_index} of {task_count}); validating its images"
                            ),
                            "completed": task_index,
                            "total": task_count,
                            "progress": task_index / task_count,
                        }
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

                def report_task_progress(event: dict[str, object]) -> None:
                    if progress_callback is None:
                        return
                    progress = dict(event)
                    progress["task_index"] = task_index
                    progress["task_count"] = task_count
                    progress["task_name"] = task.name
                    progress["message"] = (
                        f"CVAT task {task.name} ({task_index} of {task_count}): "
                        f"{event.get('message', 'Importing images')}"
                    )
                    progress_callback(progress)

                report_task_progress.cancel_requested = getattr(
                    progress_callback, "cancel_requested", None
                )
                staged_snapshot = importer.import_archive(
                    exported,
                    snapshot_parent / "incoming",
                    classes=self.store.load_manifest().classes,
                    class_mapping=mapping,
                    source_provenance=provenance,
                    snapshot_id=snapshot_id,
                    progress_callback=report_task_progress,
                )
                staged.append((staged_snapshot, task_display_name))

            snapshots: list[DatasetSnapshot] = []
            created_paths: list[Path] = []
            catalog_path = dataset_root / "catalog.json"
            original_catalog = read_json(catalog_path) if catalog_path.is_file() else None
            try:
                for staged_snapshot, task_display_name in staged:
                    check_cancellation(
                        progress_callback,
                        "CVAT import canceled before publishing task snapshots.",
                    )
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
                        os.replace(staged_snapshot.root, destination)
                        created_paths.append(destination)
                        snapshot = DatasetSnapshot.load(destination)
                    self.store.set_dataset_display_name(snapshot.root, task_display_name)
                    snapshots.append(snapshot)
                    check_cancellation(
                        progress_callback,
                        "CVAT import canceled while publishing task snapshots.",
                    )
            except Exception:
                for path in reversed(created_paths):
                    shutil.rmtree(path, ignore_errors=True)
                if original_catalog is None:
                    catalog_path.unlink(missing_ok=True)
                else:
                    atomic_write_json(catalog_path, original_catalog)
                raise
        if progress_callback is not None:
            total_images = sum(len(snapshot.manifest.images) for snapshot in snapshots)
            progress_callback(
                {
                    "phase": "completed",
                    "message": (
                        f"Imported {len(snapshots)} CVAT task snapshots "
                        f"({total_images} images)"
                    ),
                    "completed": task_count,
                    "total": task_count,
                    "progress": 1.0,
                    "phase_progress": 1.0,
                }
            )
        return snapshots
