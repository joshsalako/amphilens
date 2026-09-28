"""Small user-level state for restoring the last active AmphiLens project."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core import atomic_write_json, read_json


USER_STATE_SCHEMA_VERSION = 1


class UserStateError(ValueError):
    """Raised when the user-level AmphiLens state cannot be trusted."""


def default_user_state_path(
    platform_name: str | None = None,
    *,
    home: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the platform-standard path for the AmphiLens user state."""
    platform_name = platform_name or sys.platform
    home_path = Path(home).expanduser() if home is not None else Path.home()
    environment = os.environ if environ is None else environ

    if platform_name.startswith("win"):
        base = Path(environment.get("APPDATA", home_path / "AppData" / "Roaming"))
        return base / "AmphiLens" / "state.json"
    if platform_name == "darwin":
        return home_path / "Library" / "Application Support" / "AmphiLens" / "state.json"
    base = Path(environment.get("XDG_CONFIG_HOME", home_path / ".config"))
    return base / "amphilens" / "state.json"


class UserStateStore:
    """Persist only the last active project path in an atomic JSON file."""

    def __init__(self, path: str | Path | None = None):
        self.path = (
            Path(path).expanduser().resolve()
            if path is not None
            else default_user_state_path()
        )

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": USER_STATE_SCHEMA_VERSION}
        try:
            value = read_json(self.path)
        except Exception as exc:  # noqa: BLE001 - normalize local state failures
            raise UserStateError(f"Cannot read AmphiLens user state: {self.path}") from exc
        if not isinstance(value, dict):
            raise UserStateError("AmphiLens user state must be a JSON object")
        if value.get("schema_version") != USER_STATE_SCHEMA_VERSION:
            raise UserStateError(
                "Unsupported AmphiLens user state schema version: "
                f"{value.get('schema_version')!r}"
            )
        project = value.get("last_active_project")
        if project is not None and not isinstance(project, str):
            raise UserStateError("last_active_project must be a path string or null")
        return value

    def last_active_project(self) -> Path | None:
        """Return the remembered project path without validating the project itself."""
        value = self._read().get("last_active_project")
        return Path(value).expanduser().resolve() if value else None

    def remember_project(self, project_dir: str | Path) -> None:
        """Atomically remember one resolved project path."""
        project = Path(project_dir).expanduser().resolve()
        atomic_write_json(
            self.path,
            {
                "schema_version": USER_STATE_SCHEMA_VERSION,
                "last_active_project": str(project),
            },
        )

    def clear(self) -> None:
        """Forget the active project while retaining a valid empty state file."""
        atomic_write_json(self.path, {"schema_version": USER_STATE_SCHEMA_VERSION})
