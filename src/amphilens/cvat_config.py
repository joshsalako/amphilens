"""Local CVAT connection settings from process environment or a nearby .env file."""

from __future__ import annotations

import os


def _dotenv_values() -> dict[str, str]:
    """Read CVAT settings without mutating the process environment."""
    try:
        from dotenv import dotenv_values, find_dotenv
    except ImportError:
        return {}
    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        return {}
    values = dotenv_values(dotenv_path)
    return {
        key: value.strip()
        for key in ("CVAT_URL", "CVAT_TOKEN")
        if isinstance((value := values.get(key)), str) and value.strip()
    }


def _configured_cvat_settings() -> tuple[str, str]:
    values = _dotenv_values()
    server_url = os.environ.get("CVAT_URL") or values.get("CVAT_URL") or ""
    token = os.environ.get("CVAT_TOKEN") or values.get("CVAT_TOKEN") or ""
    return server_url.strip().rstrip("/"), token.strip()


def configured_cvat_server_url() -> str:
    """Return the configured CVAT URL for display; never expose the CVAT token."""
    return _configured_cvat_settings()[0]
