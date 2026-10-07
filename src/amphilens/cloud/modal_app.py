"""The only module that imports Modal and declares the remote training function."""

from pathlib import Path

import modal

from .constants import (
    IMAGE_APT_PACKAGES,
    IMAGE_PINS,
    MODAL_APP_NAME,
    MODEL_CACHE_MOUNT,
    MODEL_CACHE_VOLUME_NAME,
    PREDICTION_MOUNT,
    PREDICTION_VOLUME_NAME,
    VOLUME_MOUNT,
    VOLUME_NAME,
)
from .estimate import MAX_FUNCTION_TIMEOUT_SECONDS
from .worker import run_remote_training

base_image = (
    modal.Image.debian_slim(python_version=IMAGE_PINS["python"])
    .apt_install(*IMAGE_APT_PACKAGES)
    .uv_pip_install(
        f"torch=={IMAGE_PINS['torch']}",
        f"torchvision=={IMAGE_PINS['torchvision']}",
        f"ultralytics=={IMAGE_PINS['ultralytics']}",
        f"PyYAML=={IMAGE_PINS['pyyaml']}",
        f"Pillow=={IMAGE_PINS['pillow']}",
        f"numpy=={IMAGE_PINS['numpy']}",
        f"opencv-python-headless=={IMAGE_PINS['opencv-python-headless']}",
        f"huggingface-hub=={IMAGE_PINS['huggingface-hub']}",
    )
)
training_image = base_image.add_local_python_source("amphilens")

app = modal.App(MODAL_APP_NAME)
training_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
prediction_volume = modal.Volume.from_name(PREDICTION_VOLUME_NAME, create_if_missing=True)
model_cache_volume = modal.Volume.from_name(MODEL_CACHE_VOLUME_NAME, create_if_missing=True)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _build_prediction_image(image, environment):
    return image.env(environment).add_local_python_source("amphilens")


prediction_image = _build_prediction_image(
    base_image,
    {
        "YOLO_CONFIG_DIR": f"{MODEL_CACHE_MOUNT}/ultralytics",
        "TORCH_HOME": f"{MODEL_CACHE_MOUNT}/torch",
        "HF_HOME": f"{MODEL_CACHE_MOUNT}/huggingface",
    },
)


@app.cls(
    image=prediction_image,
    timeout=MAX_FUNCTION_TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
    scaledown_window=30,
    volumes={
        PREDICTION_MOUNT: prediction_volume,
        MODEL_CACHE_MOUNT: model_cache_volume,
    },
)
class PredictionEngine:
    """One warm container loads a selected model once and serves its image batches."""

    model_spec_json: str = modal.parameter()
    job_key: str = modal.parameter()
    gpu_type: str = modal.parameter()

    def _report_progress(self, values: dict) -> None:
        import re
        from datetime import datetime, timezone

        from ..core import atomic_write_json

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.job_key):
            return
        job_root = Path(PREDICTION_MOUNT).resolve() / "prediction-jobs" / self.job_key
        job_root.mkdir(parents=True, exist_ok=True)
        safe_values = dict(values)
        safe_values.setdefault("gpu", self.gpu_type)
        atomic_write_json(
            job_root / "progress.json",
            {
                "state": "running",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **safe_values,
            },
        )
        prediction_volume.commit()

    @modal.enter()
    def load_model(self):
        import json
        import os
        import time
        from pathlib import Path

        from ..models import ModelCatalog, load_detector, load_preset_detector
        from ..models.hosted_models import (
            HostedClassMappedDetector,
            get_hosted_model,
            resolve_class_mapping,
        )

        started = time.monotonic()
        prediction_volume.reload()
        model_cache_volume.reload()
        self._report_progress(
            {
                "phase": "model_setup",
                "message": "Starting model setup on the Modal GPU",
                "gpu": self.gpu_type,
            }
        )
        spec = json.loads(self.model_spec_json)
        source = str(spec["source"])
        classes = list(spec["classes"])
        cache_root = Path(MODEL_CACHE_MOUNT).resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(cache_root / "ultralytics")
        os.environ["TORCH_HOME"] = str(cache_root / "torch")
        os.environ["HF_HOME"] = str(cache_root / "huggingface")

        if source == "hosted":
            hosted = get_hosted_model(str(spec["hosted_model_id"]))
            from .prediction import download_verified_hosted_checkpoint

            verified = download_verified_hosted_checkpoint(
                hosted,
                cache_root,
                cache_commit=model_cache_volume.commit,
                progress_callback=self._report_progress,
            )
            detector = load_detector(
                verified,
                architecture=hosted.architecture,
                classes=list(hosted.source_classes),
                model_id=hosted.model_id,
                preprocessing=hosted.preprocessing.to_dict(),
            )
            mapping = resolve_class_mapping(
                hosted.source_classes,
                classes,
                spec.get("class_mapping"),
            )
            detector = HostedClassMappedDetector(detector, mapping, classes)
        elif source == "checkpoint":
            relative = Path(str(spec["checkpoint_remote_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("The uploaded checkpoint path is unsafe")
            checkpoint = (cache_root / relative).resolve()
            if not checkpoint.is_relative_to(cache_root) or not checkpoint.is_file():
                raise RuntimeError("The uploaded project checkpoint is missing")
            if _sha256_file(checkpoint) != str(spec["checkpoint_sha256"]):
                raise RuntimeError("The uploaded project checkpoint failed SHA-256 verification")
            detector = load_detector(
                checkpoint,
                architecture=str(spec["architecture"]),
                classes=classes,
                model_id=str(spec["model_id"]),
                preprocessing=spec.get("preprocessing"),
            )
        elif source == "preset":
            self._report_progress(
                {
                    "phase": "model_download",
                    "message": "Loading pretrained weights; first use may download them",
                }
            )
            detector = load_preset_detector(
                ModelCatalog().get(str(spec["model_id"])), classes=classes
            )
        else:
            raise RuntimeError("Unsupported remote prediction model source")
        self.detector = detector
        self._setup_seconds = round(time.monotonic() - started, 3)
        try:
            import torch

            self._gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            self._gpu_name = "GPU"
        self._report_progress(
            {
                "phase": "model_setup",
                "message": "Model is ready on the Modal GPU",
                "phase_progress": 1.0,
                "device": self._gpu_name,
            }
        )

    @modal.method()
    def predict_batch(self, payload: dict) -> dict:
        from .prediction import run_remote_prediction_batch

        result = run_remote_prediction_batch(
            payload,
            volume_root=PREDICTION_MOUNT,
            model_cache_root=MODEL_CACHE_MOUNT,
            detector=self.detector,
            volume_reload=prediction_volume.reload,
            volume_commit=prediction_volume.commit,
            progress_callback=self._report_progress,
        )
        result["setup_seconds"] = self._setup_seconds
        result["gpu_name"] = self._gpu_name
        self._setup_seconds = 0.0
        return result
