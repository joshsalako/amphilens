"""Environment diagnostics for the local application."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
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
    browser_app_installed: bool
    cvat_exchange: bool = True
    pillow_installed: bool = False
    numpy_installed: bool = False
    opencv_installed: bool = False
    yaml_installed: bool = False
    cvat_sdk_installed: bool = False
    modal_token_present: bool = False
    modal_connectivity: str = "not checked"

    def to_dict(self):
        return asdict(self)


def run_doctor(path: str | Path = ".", *, check_cloud: bool = False) -> DoctorReport:
    usage = shutil.disk_usage(Path(path).expanduser().resolve())
    torch_installed = importlib.util.find_spec("torch") is not None
    cuda_available = False
    if torch_installed:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    try:
        from .cloud.credentials import CloudCredentialsStore

        modal_credentials = CloudCredentialsStore().resolve()
    except Exception:
        modal_credentials = None
    modal_connectivity = "not checked"
    if check_cloud:
        if modal_credentials is None:
            modal_connectivity = "not configured"
        else:
            try:
                from .cloud.modal_transport import ModalTransport

                ModalTransport(modal_credentials).probe()
                modal_connectivity = "connected"
            except Exception:
                modal_connectivity = "failed; run `amphilens cloud diagnose` for details"
    return DoctorReport(
        python=sys.version.split()[0],
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 1,
        free_disk_gb=round(usage.free / (1024**3), 2),
        torch_installed=torch_installed,
        cuda_available=cuda_available,
        ultralytics_installed=importlib.util.find_spec("ultralytics") is not None,
        browser_app_installed=(
            importlib.util.find_spec("starlette") is not None
            and importlib.util.find_spec("uvicorn") is not None
        ),
        pillow_installed=importlib.util.find_spec("PIL") is not None,
        numpy_installed=importlib.util.find_spec("numpy") is not None,
        opencv_installed=importlib.util.find_spec("cv2") is not None,
        yaml_installed=importlib.util.find_spec("yaml") is not None,
        cvat_sdk_installed=importlib.util.find_spec("cvat_sdk") is not None,
        modal_token_present=modal_credentials is not None,
        modal_connectivity=modal_connectivity,
    )
