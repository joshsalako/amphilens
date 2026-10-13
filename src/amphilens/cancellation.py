"""Cooperative cancellation helpers for long-running workflows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class WorkflowCancelled(RuntimeError):
    """Raised when a workflow reaches a safe point after cancellation was requested."""


def cancellation_requested(
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> bool:
    check = getattr(progress_callback, "cancel_requested", None)
    value = check() if callable(check) else False
    return value if isinstance(value, bool) else False


def check_cancellation(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    message: str = "Workflow canceled at a safe point.",
) -> None:
    if cancellation_requested(progress_callback):
        raise WorkflowCancelled(message)
