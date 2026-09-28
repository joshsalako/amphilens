"""Safe filesystem locations for user-owned AmphiLens projects."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from .core import AmphiLensError, ProjectStore


class ProjectLocationError(AmphiLensError):
    """Raised when a project location cannot be safely used."""


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_same_or_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def default_projects_root(platform_name: str | None = None) -> Path:
    """Return the user Downloads directory for AmphiLens projects."""
    del platform_name  # Retained for compatibility with existing callers.
    return (Path.home() / "Downloads" / "AmphiLens" / "projects").expanduser()


def find_source_checkout(start: str | Path | None = None) -> Path | None:
    """Find the nearest Git checkout containing ``start``."""
    candidate = _resolve(start or Path.cwd())
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            return parent
    return None


def validate_new_project_path(
    project_dir: str | Path,
    *,
    source_checkout: str | Path | None = None,
) -> Path:
    """Validate a destination before creating a new project there."""
    destination = _resolve(project_dir)
    checkout = _resolve(source_checkout) if source_checkout else find_source_checkout()
    if checkout and _is_same_or_inside(destination, checkout):
        raise ProjectLocationError(
            f"Project location {destination} is inside the AmphiLens source checkout "
            f"{checkout}; choose a user-project folder outside the repository"
        )
    if destination.exists() and not destination.is_dir():
        raise ProjectLocationError(f"Project location is not a folder: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise ProjectLocationError(f"Project location is not empty: {destination}")
    return destination


def open_project(project_dir: str | Path) -> ProjectStore:
    """Open and validate a project folder containing ``manifest.json``."""
    destination = _resolve(project_dir)
    if not destination.is_dir():
        raise ProjectLocationError(f"Project folder does not exist: {destination}")
    if not (destination / "manifest.json").is_file():
        raise ProjectLocationError(f"Project manifest not found: {destination / 'manifest.json'}")
    store = ProjectStore(destination)
    try:
        store.load_manifest()
    except Exception as exc:  # noqa: BLE001 - normalize project-open failures
        raise ProjectLocationError(f"Project manifest is invalid: {exc}") from exc
    return store


def _file_digests(root: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digests[path.relative_to(root).as_posix()] = digest
    return digests


def relocate_project(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    remove_source: bool = False,
    source_checkout: str | Path | None = None,
) -> Path:
    """Copy, verify, and optionally remove a project at its old location."""
    source = _resolve(source_dir)
    destination = _resolve(destination_dir)
    if source == destination:
        raise ProjectLocationError("Project source and destination must be different")
    if not source.is_dir():
        raise ProjectLocationError(f"Project source does not exist: {source}")
    open_project(source)
    if _is_same_or_inside(destination, source) or _is_same_or_inside(source, destination):
        raise ProjectLocationError(
            "Project destination cannot contain or be contained by the source"
        )
    if destination.exists():
        raise ProjectLocationError(f"Project destination already exists: {destination}")
    validate_new_project_path(destination, source_checkout=source_checkout)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    staging = staging_parent / "project"
    try:
        shutil.copytree(source, staging)
        if _file_digests(source) != _file_digests(staging):
            raise ProjectLocationError("Project relocation verification failed: file hashes differ")
        staging.rename(destination)
        open_project(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        if staging_parent.exists() and not any(staging_parent.iterdir()):
            staging_parent.rmdir()

    if remove_source:
        shutil.rmtree(source)
    return destination
