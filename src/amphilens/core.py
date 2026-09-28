"""Stable data contracts and project artifact storage for AmphiLens."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .preprocessing import PreprocessingConfig


class AmphiLensError(Exception):
    """Base error for user-facing AmphiLens failures."""


class ValidationError(AmphiLensError, ValueError):
    """Raised when a project or run contract is invalid."""


class SourceCollisionError(AmphiLensError):
    """Raised when an output could modify an input image tree."""


class UnsupportedCheckpointError(AmphiLensError):
    """Raised when a checkpoint is incompatible with a requested run."""


CURRENT_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_same_or_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_classes(classes: Iterable[str]) -> list[str]:
    values = [str(value).strip() for value in classes]
    if not values or any(not value for value in values):
        raise ValidationError("At least one non-empty class name is required")
    if len(values) != len(set(values)):
        raise ValidationError("Class names must be unique")
    return values


@dataclass(slots=True)
class ProjectConfig:
    """Explicit, serializable configuration for a reproducible project."""

    classes: list[str]
    model_preset: str = "yolo26-l"
    checkpoint_source: str = "official-general-purpose"
    checkpoint_identifier: str | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    image_size: int = 640
    confidence_threshold: float = 0.25
    device: str = "auto"
    epochs: int = 100
    batch_size: int = 16
    patience: int = 25
    random_seed: int = 42
    freeze_strategy: str = "none"
    active_learning: dict[str, Any] = field(default_factory=dict)
    evaluation: str = "not evaluated"
    software: dict[str, str] = field(default_factory=dict)
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.classes = _validate_classes(self.classes)
        self.preprocessing = PreprocessingConfig.from_any(self.preprocessing)
        if self.image_size <= 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValidationError("image_size, epochs, and batch_size must be positive")
        if self.patience < 0:
            raise ValidationError("patience cannot be negative")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValidationError("confidence_threshold must be between 0 and 1")
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValidationError(
                f"Unsupported project config schema version: {self.schema_version}"
            )

    @classmethod
    def paper_aligned(cls, classes: Iterable[str]) -> ProjectConfig:
        """Create the explicit research-oriented preset without claiming paper identity."""
        return cls(
            classes=list(classes),
            model_preset="yolo26-l",
            preprocessing=PreprocessingConfig(clahe_enabled=True),
            freeze_strategy="paper-phased",
        )

    def validate(self, expected_classes: Iterable[str] | None = None) -> None:
        _validate_classes(self.classes)
        if expected_classes is not None and self.classes != list(expected_classes):
            raise ValidationError("Project config classes do not match the manifest classes")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "classes": list(self.classes),
            "model_preset": self.model_preset,
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_identifier": self.checkpoint_identifier,
            "preprocessing": self.preprocessing.to_dict(),
            "image_size": self.image_size,
            "confidence_threshold": self.confidence_threshold,
            "device": self.device,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "random_seed": self.random_seed,
            "freeze_strategy": self.freeze_strategy,
            "active_learning": dict(self.active_learning),
            "evaluation": self.evaluation,
            "software": dict(self.software),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | None, *, fallback_classes: Iterable[str] | None = None
    ) -> ProjectConfig:
        values = dict(data or {})
        classes = values.pop("classes", None) or list(fallback_classes or [])
        values["classes"] = classes
        if "preprocessing" in values:
            values["preprocessing"] = PreprocessingConfig.from_any(values["preprocessing"])
        return cls(**values)


def _migrate_manifest_data(data: dict[str, Any], kind: str) -> dict[str, Any]:
    """Migrate pre-versioned manifests and reject versions we cannot read."""
    migrated = dict(data)
    version = migrated.get("schema_version", 0)
    if not isinstance(version, int):
        raise ValidationError(f"{kind} manifest schema version must be an integer")
    if version > CURRENT_SCHEMA_VERSION:
        raise ValidationError(
            f"{kind} manifest schema version {version} is newer than supported "
            f"version {CURRENT_SCHEMA_VERSION}"
        )
    if version == 0:
        if kind == "project" and "image_root" in migrated and "image_roots" not in migrated:
            migrated["image_roots"] = [migrated.pop("image_root")]
        if kind == "checkpoint" and "preprocess" in migrated and "preprocessing" not in migrated:
            migrated["preprocessing"] = migrated.pop("preprocess")
        migrated["schema_version"] = CURRENT_SCHEMA_VERSION
    return migrated


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Write JSON by replacement so an interrupted write keeps the old file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def read_json(path: str | Path) -> Any:
    """Read JSON and recover a valid interrupted temporary write if needed."""
    destination = Path(path)
    try:
        return json.loads(destination.read_text(encoding="utf-8"))
    except FileNotFoundError:
        candidates = sorted(
            destination.parent.glob(f".{destination.name}.*"),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                candidate.unlink(missing_ok=True)
                continue
            os.replace(candidate, destination)
            return value
        raise


def _validate_run_id(run_id: str) -> None:
    safe_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not run_id or any(character not in safe_characters for character in run_id):
        raise ValidationError("run_id must be a safe path component")


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    relative_path: str
    sha256: str
    artifact_type: str
    cycle: int | None = None
    producer_run: str | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        relative = Path(self.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValidationError("Artifact paths must be relative to the project")
        if not self.artifact_type.strip():
            raise ValidationError("Artifact type cannot be empty")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256.lower()
        ):
            raise ValidationError("Artifact sha256 must be a 64-character hexadecimal digest")
        if self.cycle is not None and self.cycle < 0:
            raise ValidationError("Artifact cycle cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactRecord:
        return cls(**data)


@dataclass(slots=True)
class ProjectManifest:
    name: str
    classes: list[str]
    image_roots: list[str]
    project_version: str = "1"
    schema_version: int = 1
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        image_roots: Iterable[str | Path],
        classes: Iterable[str],
        metadata: dict[str, Any] | None = None,
        project_config: ProjectConfig | dict[str, Any] | None = None,
    ) -> ProjectManifest:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValidationError("Project name cannot be empty")
        clean_classes = _validate_classes(classes)
        roots = [_resolve(root) for root in image_roots]
        if not roots:
            raise ValidationError("At least one image root is required")
        missing = [str(root) for root in roots if not root.exists()]
        if missing:
            raise ValidationError(f"Image root does not exist: {missing[0]}")
        return cls(
            name=clean_name,
            classes=clean_classes,
            image_roots=[str(root) for root in roots],
            metadata=dict(metadata or {}),
            config=(
                (
                    project_config.to_dict()
                    if isinstance(project_config, ProjectConfig)
                    else dict(project_config)
                )
                if project_config is not None
                else ProjectConfig(classes=clean_classes).to_dict()
            ),
        )

    def validate(self, require_existing_roots: bool = True) -> None:
        if not self.name.strip():
            raise ValidationError("Project name cannot be empty")
        _validate_classes(self.classes)
        if not self.image_roots:
            raise ValidationError("At least one image root is required")
        roots = [_resolve(root) for root in self.image_roots]
        if len(roots) != len(set(roots)):
            raise ValidationError("Image roots must be unique")
        if require_existing_roots:
            missing = [str(root) for root in roots if not root.exists()]
            if missing:
                raise ValidationError(f"Image root does not exist: {missing[0]}")
        self.project_config.validate(expected_classes=self.classes)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["config"] = self.project_config.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectManifest:
        migrated = _migrate_manifest_data(data, "project")
        if "project_config" in migrated and "config" not in migrated:
            migrated["config"] = migrated.pop("project_config")
        manifest = cls(**migrated)
        manifest.validate()
        return manifest

    @property
    def project_config(self) -> ProjectConfig:
        return ProjectConfig.from_dict(self.config, fallback_classes=self.classes)


@dataclass(slots=True)
class InferenceConfig:
    model_id: str
    image_size: int = 640
    confidence: float = 0.25
    preprocessing: PreprocessingConfig | dict[str, Any] | str = field(
        default_factory=PreprocessingConfig
    )
    batch_size: int = 1
    device: str = "auto"
    run_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValidationError("model_id cannot be empty")
        if self.image_size <= 0 or self.batch_size <= 0:
            raise ValidationError("image_size and batch_size must be positive")
        if not 0 <= self.confidence <= 1:
            raise ValidationError("confidence must be between 0 and 1")
        self.preprocessing = PreprocessingConfig.from_any(self.preprocessing)

    @property
    def preprocessing_config(self) -> PreprocessingConfig:
        return PreprocessingConfig.from_any(self.preprocessing)

    @property
    def preprocessing_fingerprint(self) -> str:
        return self.preprocessing_config.fingerprint

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["preprocessing"] = self.preprocessing_config.to_dict()
        return payload


@dataclass(slots=True)
class RunManifest:
    run_id: str
    kind: str
    started_at: str
    status: str
    config: dict[str, Any]
    software: dict[str, str]
    schema_version: int = CURRENT_SCHEMA_VERSION
    finished_at: str | None = None

    @classmethod
    def create(cls, kind: str, config: dict[str, Any]) -> RunManifest:
        return cls(
            run_id=f"run-{uuid.uuid4().hex[:12]}",
            kind=kind,
            started_at=utc_now(),
            status="running",
            config=config,
            software={"python": sys.version.split()[0], "platform": platform.platform()},
        )

    def validate(self) -> None:
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValidationError(f"Unsupported run manifest schema version: {self.schema_version}")
        _validate_run_id(self.run_id)
        if not self.kind.strip():
            raise ValidationError("Run kind cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        manifest = cls(**_migrate_manifest_data(data, "run"))
        manifest.validate()
        return manifest


@dataclass(slots=True)
class DetectionRecord:
    image_path: str
    image_id: str
    class_id: int
    class_name: str
    confidence: float | None
    bbox_xyxy: list[float]
    image_width: int
    image_height: int
    model_id: str
    run_id: str
    cycle: int | None = None
    preprocessing: str | None = None

    def __post_init__(self) -> None:
        if len(self.bbox_xyxy) != 4:
            raise ValidationError("bbox_xyxy must contain four coordinates")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValidationError("Image dimensions must be positive")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValidationError("confidence must be between 0 and 1")
        if self.bbox_xyxy[2] < self.bbox_xyxy[0] or self.bbox_xyxy[3] < self.bbox_xyxy[1]:
            raise ValidationError("Bounding box maximums must be >= minimums")

    @property
    def bbox_xyxyn(self) -> list[float]:
        width, height = self.image_width, self.image_height
        return [
            round(self.bbox_xyxy[0] / width, 6),
            round(self.bbox_xyxy[1] / height, 6),
            round(self.bbox_xyxy[2] / width, 6),
            round(self.bbox_xyxy[3] / height, 6),
        ]

    def to_row(self) -> dict[str, Any]:
        normalized = self.bbox_xyxyn
        return {
            "image_path": self.image_path,
            "image_id": self.image_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": "" if self.confidence is None else self.confidence,
            "bbox_xmin": float(self.bbox_xyxy[0]),
            "bbox_ymin": float(self.bbox_xyxy[1]),
            "bbox_xmax": float(self.bbox_xyxy[2]),
            "bbox_ymax": float(self.bbox_xyxy[3]),
            "bbox_xmin_norm": normalized[0],
            "bbox_ymin_norm": normalized[1],
            "bbox_xmax_norm": normalized[2],
            "bbox_ymax_norm": normalized[3],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "model_id": self.model_id,
            "run_id": self.run_id,
            "cycle": "" if self.cycle is None else self.cycle,
            "preprocessing": self.preprocessing or "",
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectionRecord:
        return cls(**data)


@dataclass(slots=True)
class ModelManifest:
    model_id: str
    architecture: str
    classes: list[str]
    training_domain: str
    source: str
    license: str
    preprocessing: dict[str, Any] = field(default_factory=dict)
    input_size: int = 640
    checkpoint_sha256: str | None = None
    model_card: str | None = None

    def validate(self) -> None:
        if not self.model_id.strip() or not self.architecture.strip():
            raise ValidationError("Model id and architecture are required")
        _validate_classes(self.classes)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelManifest:
        manifest = cls(**data)
        manifest.validate()
        return manifest


@dataclass(slots=True)
class CheckpointManifest:
    checkpoint_path: str
    model_id: str
    architecture: str
    classes: list[str]
    preprocessing: dict[str, Any]
    sha256: str
    created_at: str
    parent_checkpoint: str | None = None
    software: dict[str, str] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    schema_version: int = CURRENT_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        checkpoint_path: str | Path,
        *,
        model_id: str,
        architecture: str,
        classes: Iterable[str],
        preprocessing: dict[str, Any],
        parent_checkpoint: str | None = None,
        training_config: dict[str, Any] | None = None,
    ) -> CheckpointManifest:
        path = _resolve(checkpoint_path)
        if not path.is_file():
            raise ValidationError(f"Checkpoint does not exist: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(
            checkpoint_path=str(path),
            model_id=model_id,
            architecture=architecture,
            classes=_validate_classes(classes),
            preprocessing=dict(preprocessing),
            sha256=digest,
            created_at=utc_now(),
            parent_checkpoint=parent_checkpoint,
            software={"python": sys.version.split()[0]},
            training_config=dict(training_config or {}),
        )

    def validate(self) -> None:
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValidationError(
                f"Unsupported checkpoint manifest schema version: {self.schema_version}"
            )
        if not self.model_id.strip() or not self.architecture.strip():
            raise ValidationError("Checkpoint model_id and architecture are required")
        _validate_classes(self.classes)
        if len(self.sha256) != 64:
            raise ValidationError("Checkpoint sha256 must be a 64-character digest")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointManifest:
        manifest = cls(**_migrate_manifest_data(data, "checkpoint"))
        manifest.validate()
        return manifest

    def validate_compatibility(
        self,
        *,
        architecture: str,
        classes: Iterable[str],
        preprocessing: dict[str, Any],
    ) -> None:
        self.validate()
        if self.architecture != architecture:
            raise UnsupportedCheckpointError(
                f"Checkpoint architecture {self.architecture!r} does not match {architecture!r}"
            )
        if self.classes != list(classes):
            raise UnsupportedCheckpointError("Checkpoint classes do not match the project classes")
        if PreprocessingConfig.from_any(self.preprocessing).to_dict() != (
            PreprocessingConfig.from_any(preprocessing).to_dict()
        ):
            raise UnsupportedCheckpointError("Checkpoint preprocessing does not match the run")


class ProjectStore:
    """Filesystem-backed project state with immutable source-root checks."""

    def __init__(self, root: str | Path):
        self.root = _resolve(root)

    def create(self, manifest: ProjectManifest, output_dir: str | Path | None = None) -> None:
        manifest.validate()
        destination = _resolve(output_dir or self.root)
        roots = [_resolve(root) for root in manifest.image_roots]
        for root in roots:
            for candidate in (self.root, destination):
                if _is_same_or_inside(candidate, root) or _is_same_or_inside(root, candidate):
                    raise SourceCollisionError(
                        f"Project output {candidate} overlaps source image root {root}"
                    )
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in ("runs", "artifacts", "checkpoints", "annotations", "datasets"):
            (self.root / directory).mkdir(exist_ok=True)
        atomic_write_json(self.root / "manifest.json", manifest.to_dict())

    def load_manifest(self) -> ProjectManifest:
        path = self.root / "manifest.json"
        if not path.is_file():
            raise ValidationError(f"Project manifest not found: {path}")
        return ProjectManifest.from_dict(read_json(path))

    def start_run(self, config: InferenceConfig, kind: str = "inference") -> RunManifest:
        run = RunManifest.create(kind, config.to_dict())
        if config.run_id:
            _validate_run_id(config.run_id)
            run.run_id = config.run_id
        run.validate()
        run_dir = self.root / "runs" / run.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(run_dir / "run.json", run.to_dict())
        return run

    def complete_run(self, run_id: str, status: str = "completed") -> None:
        path = self.root / "runs" / run_id / "run.json"
        if not path.is_file():
            raise ValidationError(f"Run manifest not found: {run_id}")
        data = read_json(path)
        data["status"] = status
        data["finished_at"] = utc_now()
        atomic_write_json(path, data)

    def register_artifact(
        self,
        artifact_path: str | Path,
        *,
        artifact_type: str,
        cycle: int | None = None,
        producer_run: str | None = None,
    ) -> ArtifactRecord:
        path = _resolve(artifact_path)
        if not path.is_file():
            raise ValidationError(f"Artifact does not exist: {path}")
        if not _is_same_or_inside(path, self.root):
            raise ValidationError("Artifact must be inside the project root")
        relative = path.relative_to(self.root).as_posix()
        if relative == "artifacts/index.json":
            raise ValidationError("The artifact index cannot register itself")
        record = ArtifactRecord(
            relative_path=relative,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            artifact_type=artifact_type,
            cycle=cycle,
            producer_run=producer_run,
        )
        records = {item.relative_path: item for item in self.load_artifact_index()}
        records[record.relative_path] = record
        index_path = self.root / "artifacts" / "index.json"
        atomic_write_json(
            index_path,
            {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "artifacts": [
                    item.to_dict()
                    for item in sorted(records.values(), key=lambda value: value.relative_path)
                ],
            },
        )
        return record

    def load_artifact_index(self) -> list[ArtifactRecord]:
        path = self.root / "artifacts" / "index.json"
        if not path.is_file():
            return []
        data = read_json(path)
        version = data.get("schema_version", 0)
        if version > CURRENT_SCHEMA_VERSION:
            raise ValidationError(f"Unsupported artifact index schema version: {version}")
        records = [ArtifactRecord.from_dict(item) for item in data.get("artifacts", [])]
        if len({item.relative_path for item in records}) != len(records):
            raise ValidationError("Artifact index contains duplicate paths")
        return sorted(records, key=lambda item: item.relative_path)

    def import_dataset(
        self,
        archive_path: str | Path,
        *,
        class_mapping: dict[str, str] | None = None,
        source_provenance: dict[str, Any] | None = None,
    ):
        """Import an initial CVAT/COCO/YOLO archive under this project."""
        from .dataset import DatasetImporter

        manifest = self.load_manifest()
        return DatasetImporter().import_archive(
            archive_path,
            self.root / "datasets" / "incoming",
            classes=manifest.classes,
            class_mapping=class_mapping,
            source_provenance=source_provenance,
        )

    def import_cvat_project(
        self,
        project_id: str,
        *,
        class_mapping: dict[str, str] | None = None,
        server_url: str | None = None,
    ):
        """Import a complete existing CVAT project through the CVAT API."""
        from .annotations.initial import CVATProjectImportService
        from .annotations.managed import CVATSdkTransport

        transport = CVATSdkTransport(server_url=server_url)
        return CVATProjectImportService(self, transport).import_project(
            project_id,
            class_mapping=class_mapping,
        )

    def merge_dataset_snapshot(self, incoming, parent=None):
        """Merge an annotated snapshot into a new immutable project snapshot."""
        from .dataset import DatasetMerger, DatasetSnapshot

        incoming_snapshot = (
            incoming if isinstance(incoming, DatasetSnapshot) else DatasetSnapshot.load(incoming)
        )
        if parent is None:
            snapshots = sorted(
                path
                for path in (self.root / "datasets").glob("*")
                if path.is_dir() and (path / "manifest.json").is_file()
            )
            if not snapshots:
                raise ValidationError("No parent dataset snapshot is available")
            parent = DatasetSnapshot.load(snapshots[-1])
        parent_snapshot = (
            parent if isinstance(parent, DatasetSnapshot) else DatasetSnapshot.load(parent)
        )
        return DatasetMerger().merge(
            parent_snapshot,
            incoming_snapshot,
            self.root / "datasets" / "merged",
        )


def iter_images(roots: Iterable[str | Path]) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths: set[Path] = set()
    for root in roots:
        path = _resolve(root)
        if path.is_file() and path.suffix.lower() in extensions:
            paths.add(path)
        elif path.is_dir():
            paths.update(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in extensions
            )
    return sorted(paths)
