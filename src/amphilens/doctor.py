"""Environment diagnostics for the local application."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(slots=True)
class DoctorReport:
    python: str
    platform: str
    cpu_count: int
    free_disk_gb: float
    torch_installed: bool
    cuda_available: bool
    ultralytics_installed: bool
    streamlit_installed: bool
    cvat_exchange: bool = True

    def to_dict(self):
        return asdict(self)


def run_doctor(path: str | Path = ".") -> DoctorReport:
    usage = shutil.disk_usage(Path(path).expanduser().resolve())
    torch_installed = importlib.util.find_spec("torch") is not None
    cuda_available = False
    if torch_installed:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    return DoctorReport(
        python=sys.version.split()[0],
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 1,
        free_disk_gb=round(usage.free / (1024**3), 2),
        torch_installed=torch_installed,
        cuda_available=cuda_available,
        ultralytics_installed=importlib.util.find_spec("ultralytics") is not None,
        streamlit_installed=importlib.util.find_spec("streamlit") is not None,
    )

