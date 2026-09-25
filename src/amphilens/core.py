"""Stable data contracts and project artifact storage for AmphiLens."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class AmphiLensError(Exception):
    """Base error for user-facing AmphiLens failures."""


class ValidationError(AmphiLensError, ValueError):
    """Raised when a project or run contract is invalid."""


class SourceCollisionError(AmphiLensError):
    """Raised when an output could modify an input image tree."""


class UnsupportedCheckpointError(AmphiLensError):
    """Raised when a checkpoint is incompatible with a requested run."""


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
class ProjectManifest:
    name: str
    classes: list[str]
    image_roots: list[str]
    project_version: str = "1"
    schema_version: int = 1
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        image_roots: Iterable[str | Path],
        classes: Iterable[str],
        metadata: dict[str, Any] | None = None,
    ) -> "ProjectManifest":
        clean_name = str(name).strip()
        if not clean_name:
            raise ValidationError("Project name cannot be empty")
        roots = [_resolve(root) for root in image_roots]
        if not roots:
            raise ValidationError("At least one image root is required")
        missing = [str(root) for root in roots if not root.exists()]
        if missing:
            raise ValidationError(f"Image root does not exist: {missing[0]}")
        return cls(
            name=clean_name,
            classes=_validate_classes(classes),
            image_roots=[str(root) for root in roots],
            metadata=dict(metadata or {}),
        )

    def validate(self) -> None:
        if not self.name.strip():
            raise ValidationError("Project name cannot be empty")
        _validate_classes(self.classes)
        if not self.image_roots:
            raise ValidationError("At least one image root is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectManifest":
        manifest = cls(**data)
        manifest.validate()
        return manifest


@dataclass(slots=True)
class InferenceConfig:
    model_id: str
    image_size: int = 640
    confidence: float = 0.25
    preprocessing: str = "none"
    batch_size: int = 1
    device: str = "auto"
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValidationError("model_id cannot be empty")
        if self.image_size <= 0 or self.batch_size <= 0:
            raise ValidationError("image_size and batch_size must be positive")
        if not 0 <= self.confidence <= 1:
            raise ValidationError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunManifest:
    run_id: str
    kind: str
    started_at: str
    status: str
    config: dict[str, Any]
    software: dict[str, str]

    @classmethod
    def create(cls, kind: str, config: dict[str, Any]) -> "RunManifest":
        return cls(
            run_id=f"run-{uuid.uuid4().hex[:12]}",
            kind=kind,
            started_at=utc_now(),
            status="running",
            config=config,
            software={"python": sys.version.split()[0], "platform": platform.platform()},
        )


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

    def validate(self) -> None:
        if not self.model_id.strip() or not self.architecture.strip():
            raise ValidationError("Model id and architecture are required")
        _validate_classes(self.classes)


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
    ) -> "CheckpointManifest":
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

    def validate_compatibility(
        self,
        *,
        architecture: str,
        classes: Iterable[str],
        preprocessing: dict[str, Any],
    ) -> None:
        if self.architecture != architecture:
            raise UnsupportedCheckpointError(
                f"Checkpoint architecture {self.architecture!r} does not match {architecture!r}"
            )
        if self.classes != list(classes):
            raise UnsupportedCheckpointError("Checkpoint classes do not match the project classes")
        if self.preprocessing != preprocessing:
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
            if _is_same_or_inside(destination, root) or _is_same_or_inside(root, destination):
                raise SourceCollisionError(
                    f"Project output {destination} overlaps source image root {root}"
                )
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in ("runs", "artifacts", "checkpoints", "annotations"):
            (self.root / directory).mkdir(exist_ok=True)
        (self.root / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    def load_manifest(self) -> ProjectManifest:
        path = self.root / "manifest.json"
        if not path.is_file():
            raise ValidationError(f"Project manifest not found: {path}")
        return ProjectManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def start_run(self, config: InferenceConfig, kind: str = "inference") -> RunManifest:
        run = RunManifest.create(kind, config.to_dict())
        if config.run_id:
            run.run_id = config.run_id
        run_dir = self.root / "runs" / run.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "run.json").write_text(json.dumps(asdict(run), indent=2) + "\n")
        return run

    def complete_run(self, run_id: str, status: str = "completed") -> None:
        path = self.root / "runs" / run_id / "run.json"
        if not path.is_file():
            raise ValidationError(f"Run manifest not found: {run_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["status"] = status
        data["finished_at"] = utc_now()
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def iter_images(roots: Iterable[str | Path]) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths: set[Path] = set()
    for root in roots:
        path = _resolve(root)
        if path.is_file() and path.suffix.lower() in extensions:
            paths.add(path)
        elif path.is_dir():
            paths.update(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions)
    return sorted(paths)

