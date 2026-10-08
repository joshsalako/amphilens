"""Resumable managed CVAT cycle state over a small transport interface."""

from __future__ import annotations

import hashlib
import importlib.metadata
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..core import ProjectStore, ValidationError, atomic_write_json, read_json
from ..cvat_config import _configured_cvat_settings
from ..dataset import DatasetImporter, DatasetMerger, DatasetSnapshot


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class CVATTaskSummary:
    task_id: str
    name: str
    size: int | None = None
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CVATProjectSummary:
    project_id: str
    name: str
    labels: list[str]
    tasks: list[CVATTaskSummary] = field(default_factory=list)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "labels": list(self.labels),
            "task_count": self.task_count,
            "tasks": [task.to_dict() for task in self.tasks],
        }


class CVATTransport(Protocol):
    def list_projects(self) -> list[CVATProjectSummary]: ...

    def get_project(self, project_id: str) -> CVATProjectSummary: ...

    def list_project_tasks(self, project_id: str) -> list[CVATTaskSummary]: ...

    def export_project(
        self,
        project_id: str,
        destination: str | Path,
        *,
        include_images: bool = True,
    ) -> Path: ...

    def ensure_project(self, name: str, labels: list[str]) -> str: ...

    def create_task(
        self, project_id: str, name: str, labels: list[str], image_paths: list[Path]
    ) -> dict[str, Any]: ...

    def task_status(self, task_id: str) -> dict[str, Any]: ...

    def export_task(self, task_id: str, destination: str | Path) -> Path: ...


class CVATSdkTransport:
    """CVAT transport backed by the pinned official ``cvat-sdk`` package.

    The token is read at runtime and is never returned in a project manifest.
    The SDK is imported lazily so CPU-only installs can still use portable ZIP
    exchange and all non-CVAT workflows.
    """

    def __init__(
        self,
        server_url: str | None = None,
        token: str | None = None,
        *,
        export_format: str = "CVAT for images 1.1",
    ):
        configured_server_url, configured_token = _configured_cvat_settings()
        self.server_url = (server_url or "").strip().rstrip("/") or configured_server_url
        self.token = (token or "").strip() or configured_token or None
        self.export_format = export_format
        self.client_version = self._version()
        if not self.server_url:
            raise ValidationError("CVAT server URL is required (use --server-url or CVAT_URL)")
        if not self.token:
            raise ValidationError(
                "CVAT token is required through CVAT_TOKEN or an equivalent secret store"
            )

    def _load_sdk(self):
        try:
            from cvat_sdk import make_client, models
            from cvat_sdk.core.proxies.tasks import ResourceType
            from cvat_sdk.core.proxies.types import Location
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise ValidationError(
                "Managed CVAT requires the pinned CVAT extra: pip install 'amphilens[cvat]'"
            ) from exc
        return make_client, models, ResourceType, Location

    def _client(self):
        make_client, _, _, _ = self._load_sdk()
        return make_client(self.server_url, access_token=self.token)

    @staticmethod
    def _version() -> str:
        try:
            return f"cvat-sdk=={importlib.metadata.version('cvat-sdk')}"
        except importlib.metadata.PackageNotFoundError:
            return "cvat-sdk (version unavailable)"

    @staticmethod
    def _raise_api_failure(operation: str, exc: Exception):
        if isinstance(exc, ValidationError):
            raise exc
        raise ValidationError(f"CVAT {operation} failed: {exc}") from exc

    @staticmethod
    def _status_value(value: Any) -> str | None:
        if value is None:
            return None
        return str(getattr(value, "value", value)).lower()

    @classmethod
    def _task_summary(cls, task: Any) -> CVATTaskSummary:
        size = getattr(task, "size", None)
        try:
            size = int(size) if size is not None else None
        except (TypeError, ValueError):
            size = None
        return CVATTaskSummary(
            task_id=str(task.id),
            name=str(getattr(task, "name", task.id)),
            size=size,
            status=cls._status_value(getattr(task, "status", None)),
        )

    @classmethod
    def _project_summary(cls, project: Any) -> CVATProjectSummary:
        labels = [str(label.name) for label in project.get_labels()]
        tasks = [cls._task_summary(task) for task in project.get_tasks()]
        return CVATProjectSummary(
            project_id=str(project.id),
            name=str(project.name),
            labels=labels,
            tasks=tasks,
        )

    def list_projects(self) -> list[CVATProjectSummary]:
        try:
            with self._client() as client:
                return [self._project_summary(project) for project in client.projects.list()]
        except Exception as exc:  # noqa: BLE001 - normalize optional SDK failures
            self._raise_api_failure("project listing", exc)

    def get_project(self, project_id: str) -> CVATProjectSummary:
        try:
            resolved_id = int(project_id)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"CVAT project ID must be an integer: {project_id!r}") from exc
        try:
            with self._client() as client:
                return self._project_summary(client.projects.retrieve(resolved_id))
        except Exception as exc:  # noqa: BLE001 - normalize optional SDK failures
            self._raise_api_failure("project lookup", exc)

    def list_project_tasks(self, project_id: str) -> list[CVATTaskSummary]:
        return self.get_project(project_id).tasks

    def export_project(
        self,
        project_id: str,
        destination: str | Path,
        *,
        include_images: bool = True,
    ) -> Path:
        _, _, _, location = self._load_sdk()
        target = Path(destination).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            resolved_id = int(project_id)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"CVAT project ID must be an integer: {project_id!r}") from exc
        try:
            with self._client() as client:
                project = client.projects.retrieve(resolved_id)
                exported = project.export_dataset(
                    self.export_format,
                    target,
                    include_images=include_images,
                    location=location.LOCAL,
                )
        except Exception as exc:  # noqa: BLE001 - normalize optional SDK failures
            self._raise_api_failure("project export", exc)
        result = Path(exported).expanduser().resolve() if exported else target
        if not result.is_file():
            raise ValidationError(
                f"CVAT project export did not create the expected archive: {result}"
            )
        return result

    def ensure_project(self, name: str, labels: list[str]) -> str:
        _, models, _, _ = self._load_sdk()
        if not labels or len(labels) != len(set(labels)):
            raise ValidationError("CVAT project labels must be non-empty and unique")
        with self._client() as client:
            matches = [project for project in client.projects.list() if project.name == name]
            if matches:
                project = matches[0]
                existing = [label.name for label in project.get_labels()]
                if existing != labels:
                    raise ValidationError(
                        "CVAT project label schema does not match the AmphiLens project"
                    )
                return str(project.id)
            project = client.projects.create(
                models.ProjectWriteRequest(
                    name=name,
                    labels=[models.PatchedLabelRequest(name=label) for label in labels],
                )
            )
            return str(project.id)

    def create_task(
        self, project_id: str, name: str, labels: list[str], image_paths: list[Path]
    ) -> dict[str, Any]:
        _, models, resource_type, _ = self._load_sdk()
        if not image_paths:
            raise ValidationError("At least one image is required for a managed CVAT task")
        filenames = [path.name for path in image_paths]
        if len(filenames) != len(set(filenames)):
            raise ValidationError(
                "Selected CVAT images must have unique filenames; use portable exchange "
                "for duplicate basenames"
            )
        with self._client() as client:
            task = client.tasks.create_from_data(
                spec=models.TaskWriteRequest(name=name, project_id=int(project_id)),
                resource_type=resource_type.LOCAL,
                resources=image_paths,
                data_params={
                    "sorting_method": "predefined",
                    "job_file_mapping": [filenames],
                },
            )
            jobs = task.get_jobs()
            return {
                "task_id": str(task.id),
                "job_id": str(jobs[0].id) if jobs else None,
                "url": f"{self.server_url}/tasks/{task.id}",
                "client_version": self._version(),
            }

    def task_status(self, task_id: str) -> dict[str, Any]:
        with self._client() as client:
            task = client.tasks.retrieve(int(task_id))
            jobs = task.get_jobs()
            states = [str(job.state).lower() for job in jobs]
            state = (
                "completed"
                if states and all(item == "completed" for item in states)
                else "annotation"
            )
            return {
                "state": state,
                "annotation_count": self._annotation_count(task),
                "job_states": states,
            }

    def export_task(self, task_id: str, destination: str | Path) -> Path:
        _, _, _, location = self._load_sdk()
        target = Path(destination).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._client() as client:
            task = client.tasks.retrieve(int(task_id))
            task.export_dataset(
                self.export_format,
                target,
                include_images=True,
                location=location.LOCAL,
            )
        if not target.is_file():
            raise ValidationError(f"CVAT export did not create the expected archive: {target}")
        return target

    @staticmethod
    def _annotation_count(task: Any) -> int:
        getter = getattr(task, "get_annotations", None)
        if not callable(getter):
            return 0
        annotations = getter()
        if isinstance(annotations, dict):
            return sum(len(annotations.get(key, []) or []) for key in ("shapes", "tracks", "tags"))
        return sum(len(getattr(annotations, key, []) or []) for key in ("shapes", "tracks", "tags"))


@dataclass(slots=True)
class ManagedCVATCycleManifest:
    cycle: int
    project_name: str
    classes: list[str]
    server_url: str
    project_id: str
    task_id: str
    job_id: str | None
    task_url: str
    selection_hash: str
    state: str
    selected_images: list[str] = field(default_factory=list)
    annotation_count: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    export_path: str | None = None
    merged_snapshot: str | None = None
    client_version: str = "unknown"
    schema_version: int = 1

    def validate(self) -> None:
        if self.cycle < 0 or not self.project_name or not self.project_id or not self.task_id:
            raise ValidationError("Managed CVAT cycle identifiers are incomplete")
        if not self.classes or len(self.classes) != len(set(self.classes)):
            raise ValidationError("Managed CVAT cycle classes must be unique")
        if not self.selection_hash:
            raise ValidationError("Managed CVAT cycle selection hash is required")
        if self.state not in {"created", "annotation", "completed", "continued"}:
            raise ValidationError(f"Unsupported managed CVAT cycle state: {self.state}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManagedCVATCycleManifest:
        manifest = cls(**data)
        manifest.validate()
        return manifest


class ManagedCVATCycleService:
    """Create, resume, verify, export, and merge one CVAT task per cycle."""

    def __init__(self, store: ProjectStore, transport: CVATTransport, *, server_url: str = ""):
        self.store = store
        self.transport = transport
        self.server_url = server_url

    def _path(self, cycle: int) -> Path:
        return self.store.root / "annotations" / "managed" / f"cycle-{cycle}.json"

    def _load(self, cycle: int) -> ManagedCVATCycleManifest:
        path = self._path(cycle)
        if not path.is_file():
            raise ValidationError(f"Managed CVAT cycle does not exist: {cycle}")
        return ManagedCVATCycleManifest.from_dict(read_json(path))

    def _save(self, manifest: ManagedCVATCycleManifest) -> ManagedCVATCycleManifest:
        manifest.updated_at = _now()
        atomic_write_json(self._path(manifest.cycle), manifest.to_dict())
        return manifest

    def start(
        self,
        *,
        cycle: int,
        image_paths: list[str | Path],
        selection_hash: str,
    ) -> ManagedCVATCycleManifest:
        if not selection_hash:
            raise ValidationError("A deterministic selection hash is required before CVAT upload")
        existing_path = self._path(cycle)
        if existing_path.is_file():
            existing = self._load(cycle)
            if existing.selection_hash != selection_hash:
                raise ValidationError("Managed CVAT cycle selection hash does not match")
            return existing
        project = self.store.load_manifest()
        paths = [Path(path).expanduser().resolve() for path in image_paths]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ValidationError(f"Selected CVAT image does not exist: {missing[0]}")
        project_id = self.transport.ensure_project(project.name, list(project.classes))
        task = self.transport.create_task(
            project_id,
            f"{project.name}-cycle-{cycle}",
            list(project.classes),
            paths,
        )
        task_id = str(task.get("task_id", ""))
        if not task_id:
            raise ValidationError("CVAT transport did not return a task id")
        manifest = ManagedCVATCycleManifest(
            cycle=cycle,
            project_name=project.name,
            classes=list(project.classes),
            server_url=self.server_url,
            project_id=str(project_id),
            task_id=task_id,
            job_id=str(task["job_id"]) if task.get("job_id") is not None else None,
            task_url=str(task.get("url", "")),
            selection_hash=selection_hash,
            selected_images=[str(path) for path in paths],
            state="created",
            client_version=str(task.get("client_version", "unknown")),
        )
        self._path(cycle).parent.mkdir(parents=True, exist_ok=True)
        return self._save(manifest)

    def refresh(self, cycle: int) -> ManagedCVATCycleManifest:
        manifest = self._load(cycle)
        if manifest.state == "continued":
            return manifest
        status = self.transport.task_status(manifest.task_id)
        state = str(status.get("state", "annotation")).lower()
        if state in {"finished", "completed", "completed_with_errors"}:
            manifest.state = "completed"
        else:
            manifest.state = "annotation"
        manifest.annotation_count = int(status.get("annotation_count", manifest.annotation_count))
        return self._save(manifest)

    def continue_cycle(self, cycle: int) -> DatasetSnapshot:
        manifest = self.refresh(cycle)
        if manifest.state == "continued" and manifest.merged_snapshot:
            return DatasetSnapshot.load(self.store.root / manifest.merged_snapshot)
        if manifest.state != "completed":
            raise ValidationError(
                f"CVAT cycle {cycle} is not ready: task status must be completed before Continue"
            )
        export_dir = self.store.root / "annotations" / "managed" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = self.transport.export_task(
            manifest.task_id, export_dir / f"cycle-{cycle}.zip"
        )
        previous = self._latest_snapshot()
        with tempfile.TemporaryDirectory(prefix="amphilens-cvat-") as temporary:
            imported = DatasetImporter().import_archive(
                export_path,
                Path(temporary) / "incoming",
                classes=manifest.classes,
            )
            if previous is None:
                durable_root = self.store.root / "datasets" / f"managed-cycle-{cycle}"
                shutil.copytree(imported.root, durable_root)
                merged = DatasetSnapshot.load(durable_root)
                merged_relative = durable_root.relative_to(self.store.root).as_posix()
            else:
                merged = DatasetMerger().merge(
                    previous,
                    imported,
                    self.store.root / "datasets" / f"merged-cycle-{cycle}",
                )
                merged_relative = merged.root.relative_to(self.store.root).as_posix()
        manifest.state = "continued"
        manifest.export_path = export_path.relative_to(self.store.root).as_posix()
        manifest.merged_snapshot = merged_relative
        self._save(manifest)
        return merged

    def _latest_snapshot(self) -> DatasetSnapshot | None:
        candidates = sorted(
            path
            for path in (self.store.root / "datasets").glob("*")
            if path.is_dir() and (path / "manifest.json").is_file()
        )
        return DatasetSnapshot.load(candidates[-1]) if candidates else None


def selection_hash(image_paths: list[str | Path]) -> str:
    """Build a stable hash for a selected image list without storing credentials."""
    values = "\n".join(sorted(str(Path(path).expanduser().resolve()) for path in image_paths))
    return hashlib.sha256(values.encode("utf-8")).hexdigest()
