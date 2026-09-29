"""The only module that imports Modal and declares the remote training function."""

from __future__ import annotations

from pathlib import Path

import modal

from .constants import IMAGE_PINS, MODAL_APP_NAME, VOLUME_MOUNT, VOLUME_NAME
from .estimate import MAX_FUNCTION_TIMEOUT_SECONDS
from .worker import run_remote_training

training_image = (
    modal.Image.debian_slim(python_version=IMAGE_PINS["python"])
    .uv_pip_install(
        f"torch=={IMAGE_PINS['torch']}",
        f"torchvision=={IMAGE_PINS['torchvision']}",
        f"ultralytics=={IMAGE_PINS['ultralytics']}",
        f"PyYAML=={IMAGE_PINS['pyyaml']}",
        f"Pillow=={IMAGE_PINS['pillow']}",
        f"numpy=={IMAGE_PINS['numpy']}",
        f"opencv-python-headless=={IMAGE_PINS['opencv-python-headless']}",
    )
    .add_local_python_source("amphilens")
)

app = modal.App(MODAL_APP_NAME)
training_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=training_image,
    gpu="L4",
    timeout=MAX_FUNCTION_TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
    scaledown_window=10,
    volumes={VOLUME_MOUNT: training_volume},
)
def train(payload: dict) -> dict:
    return run_remote_training(
        payload,
        volume_root=Path(VOLUME_MOUNT),
        volume_commit=training_volume.commit,
    )
