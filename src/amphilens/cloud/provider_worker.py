"""Entry point executed inside Azure ML and Vertex AI managed GPU jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from ..core import ValidationError


class _JobStorage:
    def __init__(self):
        self.provider = os.environ.get("AMPHILENS_PROVIDER", "")
        if self.provider == "azure_ml":
            try:
                from azure.identity import DefaultAzureCredential
                from azure.storage.blob import BlobServiceClient
            except ImportError as exc:
                raise RuntimeError("Azure ML worker image is missing its storage SDK") from exc
            account = os.environ["AMPHILENS_STORAGE_ACCOUNT"]
            container = os.environ["AMPHILENS_STORAGE_CONTAINER"]
            credential = DefaultAzureCredential()
            service = BlobServiceClient(
                account_url=f"https://{account}.blob.core.windows.net", credential=credential
            )
            self.container = service.get_container_client(container)
        elif self.provider == "vertex_ai":
            try:
                from google.cloud import storage
            except ImportError as exc:
                raise RuntimeError("Vertex AI worker image is missing its storage SDK") from exc
            project = os.environ["AMPHILENS_GCP_PROJECT"]
            bucket = os.environ["AMPHILENS_GCS_BUCKET"]
            self.container = storage.Client(project=project).bucket(bucket)
        else:
            raise ValueError("Managed cloud worker provider is missing or unsupported")

    @staticmethod
    def _key(key: str) -> str:
        normalized = str(key).strip().strip("/")
        path = PurePosixPath(normalized)
        if not normalized or path.is_absolute() or ".." in path.parts or "\\" in normalized:
            raise ValueError("Managed cloud object path is unsafe")
        return normalized

    def download(self, key: str, destination: Path) -> None:
        key = self._key(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.provider == "azure_ml":
            data = self.container.download_blob(key).readall()
            destination.write_bytes(data)
        else:
            self.container.blob(key).download_to_filename(str(destination))

    def upload(self, source: Path, key: str) -> None:
        key = self._key(key)
        digest = _sha256(source)
        if self.provider == "azure_ml":
            blob = self.container.get_blob_client(key)
            with source.open("rb") as handle:
                blob.upload_blob(handle, overwrite=True, metadata={"sha256": digest})
        else:
            blob = self.container.blob(key)
            blob.metadata = {"sha256": digest}
            blob.upload_from_filename(str(source))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        value = re.sub(r"(?i)(https?://[^\s?#]+)\?[^\s\"'<>)]*", r"\1?[redacted]", value)
        return re.sub(
            r"(?i)((?:token|secret|password|api[_-]?key|access[_-]?token|client[_-]?secret|sig|signature)\b\s*[:=]\s*)[^\s,;&]+",
            r"\1[redacted]",
            value,
        )
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _write_json(storage: _JobStorage, destination: Path, key: str, value: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_redact_value(value), ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    storage.upload(destination, key)


def _load_prediction_detector(model_spec: dict[str, Any], cache_root: Path, report) -> Any:
    import os

    from ..models import ModelCatalog, load_detector, load_preset_detector

    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache_root / "ultralytics")
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    classes = list(model_spec["classes"])
    source = str(model_spec["source"])
    if source == "hosted":
        from ..models.hosted_models import (
            HostedClassMappedDetector,
            get_hosted_model,
            resolve_class_mapping,
        )
        from .prediction import download_verified_hosted_checkpoint

        hosted = get_hosted_model(str(model_spec["hosted_model_id"]))
        checkpoint = download_verified_hosted_checkpoint(
            hosted, cache_root, progress_callback=report
        )
        detector = load_detector(
            checkpoint,
            architecture=hosted.architecture,
            classes=list(hosted.source_classes),
            model_id=hosted.model_id,
            preprocessing=model_spec.get("preprocessing", hosted.preprocessing.to_dict()),
        )
        mapping = resolve_class_mapping(
            hosted.source_classes, classes, model_spec.get("class_mapping")
        )
        return HostedClassMappedDetector(detector, mapping, classes)
    if source == "checkpoint":
        relative = Path(str(model_spec["checkpoint_remote_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValidationError("The uploaded project checkpoint path is unsafe")
        checkpoint = (cache_root / relative).resolve()
        if not checkpoint.is_relative_to(cache_root) or not checkpoint.is_file():
            raise ValidationError("The uploaded project checkpoint is missing")
        if _sha256(checkpoint) != str(model_spec["checkpoint_sha256"]):
            raise ValidationError("The uploaded project checkpoint failed SHA-256 verification")
        return load_detector(
            checkpoint,
            architecture=str(model_spec["architecture"]),
            classes=classes,
            model_id=str(model_spec["model_id"]),
            preprocessing=model_spec.get("preprocessing"),
        )
    if source == "preset":
        report({"phase": "model_setup", "message": "Loading pretrained model weights"})
        return load_preset_detector(
            ModelCatalog().get(str(model_spec["model_id"])), classes=classes
        )
    raise ValidationError("Unsupported managed cloud prediction model source")


def _run_training(storage: _JobStorage, request_key: str, result_key: str) -> dict[str, Any]:
    from .constants import VOLUME_NAME
    from .worker import run_remote_training

    root = Path(tempfile.mkdtemp(prefix="amphilens-managed-training-"))
    managed_mount = None
    try:
        request_path = root / request_key
        storage.download(request_key, request_path)
        payload = {
            "provider": storage.provider,
            "payload_remote_path": request_key,
            "volume_name": VOLUME_NAME,
        }
        # Training requests are stored as JSON metadata beside the immutable ZIP payload.
        metadata_key = request_key.removesuffix("payload.zip") + "request.json"
        metadata_path = root / metadata_key
        storage.download(metadata_key, metadata_path)
        payload.update(json.loads(metadata_path.read_text(encoding="utf-8")))
        payload["provider"] = storage.provider
        payload["payload_remote_path"] = f"jobs/{payload['job_key']}/payload.zip"
        prefix = f"jobs/{payload['job_key']}"
        managed_mount = Path(str(payload["dataset_mount"])).resolve()
        if storage.provider not in {"azure_ml", "vertex_ai"} or not managed_mount.is_relative_to(
            Path("/tmp/amphilens/jobs").resolve()
        ):
            raise ValidationError(
                "Managed training dataset mount is outside its safe job directory"
            )
        managed_mount.parent.mkdir(parents=True, exist_ok=True)
        dataset_root = (root / prefix / "dataset").resolve()
        if managed_mount.exists() or managed_mount.is_symlink():
            raise ValidationError("Managed training dataset mount already exists")
        managed_mount.symlink_to(dataset_root, target_is_directory=True)
        local_progress = root / prefix / "progress.json"

        def sync_progress():
            if local_progress.is_file():
                details = json.loads(local_progress.read_text(encoding="utf-8"))
                _write_json(storage, local_progress, f"{prefix}/progress.json", details)

        result = run_remote_training(
            payload,
            volume_root=root,
            volume_commit=sync_progress,
            model_cache_root=root / "model-cache",
        )
        if result.get("state") in {"finished", "success", "completed"}:
            for name in ("best.pt", "last.pt", "metrics.json", "checkpoint.json"):
                artifact = root / prefix / "results" / name
                if not artifact.is_file():
                    raise ValidationError(
                        f"Managed training did not produce required artifact: {name}"
                    )
                storage.upload(artifact, f"{prefix}/results/{name}")
                metadata = result.get("artifacts", {}).get(name, {})
                metadata["remote_ref"] = _artifact_uri(storage.provider, f"{prefix}/results/{name}")
                result["artifacts"][name] = metadata
        result_path = root / "result.json"
        _write_json(storage, result_path, result_key, result)
        return result
    except Exception as exc:
        result = {"state": "failed", "error": f"{type(exc).__name__}: {str(exc)[:1500]}"}
        _write_json(storage, root / "result.json", result_key, result)
        return result
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if managed_mount is not None:
            try:
                managed_mount.unlink(missing_ok=True)
                managed_mount.parent.rmdir()
                managed_mount.parent.parent.rmdir()
            except OSError:
                pass


def _artifact_uri(provider: str, key: str) -> str:
    if provider == "vertex_ai":
        return f"gs://{os.environ['AMPHILENS_GCS_BUCKET']}/{key}"
    container = os.environ["AMPHILENS_STORAGE_CONTAINER"]
    return f"azureml-blob://{container}/{key}"


def _run_prediction(storage: _JobStorage, request_key: str, result_key: str) -> dict[str, Any]:
    from .prediction import run_remote_prediction_batch

    root = Path(tempfile.mkdtemp(prefix="amphilens-managed-prediction-"))
    try:
        request_path = root / "request.json"
        storage.download(request_key, request_path)
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        job_key = str(payload["job_key"])
        batch_id = str(payload["batch_id"])
        prefix = f"prediction-jobs/{job_key}/{batch_id}"
        batch_root = root / prefix
        batch_root.mkdir(parents=True, exist_ok=True)
        storage.download(
            f"prediction-jobs/{job_key}/{batch_id}/manifest.json", batch_root / "manifest.json"
        )
        for image in payload.get("images", []):
            remote = str(image["remote_path"])
            image_path = root / remote
            storage.download(remote, image_path)
            sentinel = Path(f"{image_path}.sha256")
            sentinel.write_text(str(image["sha256"]), encoding="ascii")

        model_spec = payload["model_spec"]
        cache_root = root / "model-cache"
        if model_spec.get("source") == "checkpoint":
            model_key = str(model_spec["checkpoint_remote_path"])
            if Path(model_key).is_absolute() or ".." in Path(model_key).parts:
                raise ValidationError("Uploaded checkpoint path is unsafe")
            storage.download(model_key, cache_root / model_key)
        progress_key = f"prediction-jobs/{job_key}/progress.json"

        def report(values):
            progress = root / "progress.json"
            _write_json(storage, progress, progress_key, dict(values))

        detector = _load_prediction_detector(model_spec, cache_root, report)
        result = run_remote_prediction_batch(
            payload,
            volume_root=root,
            model_cache_root=cache_root,
            detector=detector,
            progress_callback=report,
        )
        try:
            import torch

            result["gpu_name"] = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
            )
        except Exception:
            result["gpu_name"] = "GPU"
        _write_json(storage, root / "result.json", result_key, result)
        return result
    except Exception as exc:
        result = {"state": "failed", "error": f"{type(exc).__name__}: {str(exc)[:1500]}"}
        _write_json(storage, root / "result.json", result_key, result)
        return result
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices={"train", "predict"})
    parser.add_argument("--request-key", required=True)
    parser.add_argument("--result-key", required=True)
    arguments = parser.parse_args(argv)
    storage = _JobStorage()
    if arguments.operation == "train":
        _run_training(storage, arguments.request_key, arguments.result_key)
    else:
        _run_prediction(storage, arguments.request_key, arguments.result_key)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised in managed provider jobs
    raise SystemExit(main())
