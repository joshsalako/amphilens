"""Versioned remote container dependencies shared by the service and Modal worker."""

from __future__ import annotations

IMAGE_PINS = {
    "python": "3.11",
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "ultralytics": "8.4.163",
    "pyyaml": "6.0.3",
    "pillow": "12.3.0",
    "numpy": "2.3.5",
    "opencv-python-headless": "5.0.0.93",
}

IMAGE_APT_PACKAGES = ("libgl1", "libglib2.0-0")

VOLUME_NAME = "amphilens-cloud-training"
VOLUME_MOUNT = "/mnt/amphilens"
MODAL_APP_NAME = "amphilens-cloud-training"
MODAL_FUNCTION_NAME = "train"
