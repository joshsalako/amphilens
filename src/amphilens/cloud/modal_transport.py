"""Lazy Modal SDK adapter for asynchronous training and volume file operations."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import re
import time
from pathlib import Path
from typing import Any

from ..models.backends import OptionalDependencyError
from .constants import (
    MODAL_APP_NAME,
    MODEL_CACHE_VOLUME_NAME,
    PREDICTION_VOLUME_NAME,
    VOLUME_NAME,
)
from .credentials import CloudCredentials


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModalTransport:
    """Keep Modal SDK objects and credentials inside this adapter."""

    def __init__(
        self,
        credentials: CloudCredentials | None,
        *,
        app_name: str = MODAL_APP_NAME,
        volume_name: str = VOLUME_NAME,
    ):
        self._credentials = credentials
        self.app_name = app_name
        self.volume_name = volume_name
        self._modal = None
        self._client = None

    def __repr__(self) -> str:
        return f"ModalTransport(app_name={self.app_name!r}, volume_name={self.volume_name!r})"

    def describe(self) -> dict[str, str]:
        return {"provider": "modal", "app_name": self.app_name, "volume_name": self.volume_name}

    def _load_modal(self):
        if self._modal is None:
            try:
                self._modal = importlib.import_module("modal")
            except ImportError as exc:
                raise OptionalDependencyError(
                    "Cloud training requires the 'cloud' extra: pip install 'amphilens[cloud]'"
                ) from exc
        return self._modal

    def _load_app_module(self):
        return importlib.import_module("amphilens.cloud.modal_app")

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._credentials is None:
            raise ValueError("Modal credentials are not configured; run `amphilens cloud login`")
        modal = self._load_modal()
        try:
            self._client = modal.Client.from_credentials(
                self._credentials.token_id,
                self._credentials.token_secret,
            )
        except Exception as exc:
            raise RuntimeError(self._redact(str(exc))) from exc
        return self._client

    def redact(self, message: str) -> str:
        return self._redact(message)

    def probe(self) -> dict[str, str]:
        """Check credentials and connectivity without creating provider resources."""
        client = self._get_client()
        try:
            client.hello()
        except Exception as exc:
            raise RuntimeError(self._redact(str(exc))) from exc
        return {"provider": "modal", "connectivity": "connected"}

    def _volume(self, *, create_if_missing: bool = False, volume_name: str | None = None):
        modal = self._load_modal()
        return modal.Volume.from_name(
            volume_name or self.volume_name,
            create_if_missing=create_if_missing,
            client=self._get_client(),
        )

    @staticmethod
    def _chunks(volume, remote_path: str):
        data = volume.read_file(remote_path)
        if isinstance(data, bytes):
            yield data
        else:
            yield from data

    @classmethod
    def _read_file(cls, volume, remote_path: str) -> bytes:
        return b"".join(cls._chunks(volume, remote_path))

    @staticmethod
    def _file_exists(volume, remote_path: str) -> bool:
        try:
            return next(iter(volume.iterdir(remote_path, recursive=False)), None) is not None
        except FileNotFoundError:
            return False

    def _is_not_found_error(self, error: Exception) -> bool:
        if isinstance(error, FileNotFoundError):
            return True
        exceptions = getattr(self._modal, "exception", None)
        provider_error = getattr(exceptions, "NotFoundError", None)
        if isinstance(provider_error, type) and isinstance(error, provider_error):
            return True
        invalid_error = getattr(exceptions, "InvalidError", None)
        return (
            isinstance(invalid_error, type)
            and isinstance(error, invalid_error)
            and "no such file or directory" in str(error).lower()
        )

    def _redact(self, value: str) -> str:
        if self._credentials is None:
            return value
        return value.replace(self._credentials.token_secret, "[redacted]").replace(
            self._credentials.token_id, "[redacted]"
        )

    def upload(
        self,
        source: Path,
        remote_path: str,
        sha256: str,
        *,
        volume_name: str | None = None,
    ) -> bool:
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            volume = self._volume(create_if_missing=True, volume_name=volume_name)
            sentinel_path = f"{remote_path}.sha256"
            try:
                sentinel = self._read_file(volume, sentinel_path).decode("ascii").strip()
            except Exception:
                sentinel = ""
            if sentinel == sha256 and self._file_exists(volume, remote_path):
                return False
            with volume.batch_upload(force=True) as batch:
                batch.put_file(str(source), remote_path)
                batch.put_file(io.BytesIO(f"{sha256}\n".encode("ascii")), sentinel_path)
            return True
        except Exception as exc:
            raise RuntimeError(self._redact(str(exc))) from exc

    def upload_prediction_batch(
        self, job_key: str, batch_id: str, sources: list[Path]
    ) -> list[dict[str, str]]:
        """Upload only this batch under opaque IDs and publish its manifest atomically."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_key) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,80}", batch_id
        ):
            raise ValueError("Invalid prediction batch identity")
        if not sources or len(sources) > 32:
            raise ValueError("Prediction uploads must contain between 1 and 32 images")
        prefix = f"prediction-jobs/{job_key}/{batch_id}"
        manifest_images = []
        try:
            volume = self._volume(create_if_missing=True, volume_name=PREDICTION_VOLUME_NAME)
            with volume.batch_upload(force=True) as batch:
                for source in sources:
                    path = Path(source).expanduser().resolve()
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    extension = path.suffix.lower()
                    if extension not in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".tif",
                        ".tiff",
                        ".webp",
                    }:
                        raise ValueError(
                            f"Unsupported image type for cloud prediction: {extension}"
                        )
                    digest = _sha256_file(path)
                    image_id = "img-" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:20]
                    remote_path = f"{prefix}/images/{image_id}{extension}"
                    batch.put_file(str(path), remote_path)
                    batch.put_file(
                        io.BytesIO(f"{digest}\n".encode("ascii")), f"{remote_path}.sha256"
                    )
                    manifest_images.append(
                        {"image_id": image_id, "remote_path": remote_path, "sha256": digest}
                    )
                manifest = json.dumps(
                    {"schema_version": 1, "images": manifest_images}, sort_keys=True
                ).encode("utf-8")
                batch.put_file(io.BytesIO(manifest), f"{prefix}/manifest.json")
            return manifest_images
        except Exception as exc:
            raise RuntimeError(self._redact(str(exc))) from exc

    def predict_batch(
        self,
        payload: dict[str, Any],
        *,
        cancellation_requested=None,
        progress_callback=None,
    ) -> dict[str, Any]:
        """Submit one staged batch to a single dynamically selected Modal GPU worker."""
        from .prediction import PredictionCancelled

        self._load_modal()
        client = self._get_client()
        module = self._load_app_module()
        gpu = str(payload["gpu"])
        timeout = max(1, int(payload["timeout_seconds"]))
        model_spec_json = json.dumps(payload["model_spec"], ensure_ascii=False, sort_keys=True)
        job_key = str(payload["job_key"])
        try:
            with module.app.run(detach=True, client=client):
                engine_class = module.PredictionEngine.with_options(
                    gpu=gpu,
                    timeout=timeout,
                    retries=0,
                    max_containers=1,
                )
                engine = engine_class(
                    model_spec_json=model_spec_json,
                    job_key=job_key,
                    gpu_type=gpu,
                )
                call = engine.predict_batch.spawn(payload)
                deadline = time.monotonic() + timeout + 120
                while True:
                    try:
                        result = call.get(timeout=1)
                        break
                    except TimeoutError:
                        progress = self._prediction_progress(job_key)
                        if progress and progress_callback is not None:
                            progress_callback(progress)
                        if cancellation_requested and cancellation_requested():
                            call.cancel(terminate_containers=True)
                            raise PredictionCancelled(
                                "Prediction canceled. The active Modal call was stopped."
                            )
                        if time.monotonic() >= deadline:
                            call.cancel(terminate_containers=True)
                            raise TimeoutError("Modal prediction exceeded its local wait deadline")
            if not isinstance(result, dict):
                raise RuntimeError("Modal prediction worker returned an invalid response")
            return result
        except Exception as exc:
            raise RuntimeError(self._redact(str(exc))) from exc

    def cleanup_prediction_batch(self, job_key: str, batch_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_key) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,80}", batch_id
        ):
            raise ValueError("Invalid prediction batch identity")
        path = f"prediction-jobs/{job_key}/{batch_id}"
        try:
            volume = self._volume(volume_name=PREDICTION_VOLUME_NAME)
            volume.remove_file(path, recursive=True)
        except FileNotFoundError:
            return
        except Exception as exc:
            if self._is_not_found_error(exc):
                return
            raise RuntimeError(self._redact(str(exc))) from exc

    def cleanup_prediction_progress(self, job_key: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_key):
            raise ValueError("Invalid prediction job identity")
        path = f"prediction-jobs/{job_key}/progress.json"
        try:
            volume = self._volume(volume_name=PREDICTION_VOLUME_NAME)
            volume.remove_file(path, recursive=False)
        except FileNotFoundError:
            return
        except Exception as exc:
            if self._is_not_found_error(exc):
                return
            raise RuntimeError(self._redact(str(exc))) from exc

    def _prediction_progress(self, job_key: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_key):
            return {}
        try:
            volume = self._volume(volume_name=PREDICTION_VOLUME_NAME)
            volume.reload()
            path = f"prediction-jobs/{job_key}/progress.json"
            if not self._file_exists(volume, path):
                return {}
            payload = json.loads(self._read_file(volume, path))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def upload_model_checkpoint(self, source: Path, sha256: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("Invalid model checkpoint SHA-256")
        source = Path(source).expanduser().resolve()
        if _sha256_file(source) != sha256:
            raise ValueError("Project checkpoint SHA-256 does not match the selected file")
        remote_path = f"checkpoints/{sha256}.pt"
        self.upload(
            source,
            remote_path,
            sha256,
            volume_name=MODEL_CACHE_VOLUME_NAME,
        )
        return remote_path

    def submit(self, payload: dict[str, Any]) -> str:
        self._load_modal()
        client = self._get_client()
        module = self._load_app_module()
        options = {
            "gpu": str(payload["gpu"]),
            "timeout": max(1, int(payload["timeout_seconds"])),
            "retries": 0,
            "max_containers": 1,
        }
        try:
            with module.app.run(detach=True, client=client):
                call = module.train.with_options(**options).spawn(payload)
        except Exception as exc:
            raise RuntimeError(self._redact(str(exc))) from exc
        return str(call.object_id)

    def poll(self, call_id: str, *, remote_prefix: str) -> dict[str, Any] | None:
        modal = self._load_modal()
        call = modal.FunctionCall.from_id(call_id, client=self._get_client())
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            result = None
        except Exception as exc:
            raise RuntimeError(self._redact(str(exc))) from exc
        progress = self._progress(remote_prefix)
        if result is None:
            if not progress:
                return None
            return {
                "state": "running",
                "progress": progress.get("progress"),
                "progress_details": progress,
            }
        if not isinstance(result, dict):
            raise RuntimeError("Modal training function returned an invalid result")
        response = dict(result)
        for key in ("error", "log_tail"):
            if isinstance(response.get(key), str):
                response[key] = self._redact(response[key])
        if progress:
            response.setdefault("progress", progress.get("progress"))
            response["progress_details"] = progress
        return response

    def _progress(self, remote_prefix: str) -> dict[str, Any]:
        try:
            volume = self._volume()
            progress_path = f"{remote_prefix}/progress.json"
            if not self._file_exists(volume, progress_path):
                return {}
            data = self._read_file(volume, progress_path)
            import json

            value = json.loads(data)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def cancel(self, call_id: str) -> None:
        modal = self._load_modal()
        call = modal.FunctionCall.from_id(call_id, client=self._get_client())
        call.cancel(terminate_containers=True)

    def download(self, remote_ref: str, destination: Path) -> None:
        volume_name, remote_path = self._parse_remote_ref(remote_ref)
        try:
            modal = self._load_modal()
            volume = modal.Volume.from_name(volume_name, client=self._get_client())
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as output:
                for chunk in self._chunks(volume, remote_path):
                    output.write(chunk)
        except Exception as exc:
            raise RuntimeError(self._redact(str(exc))) from exc

    def cleanup(self, job_key: str) -> dict[str, Any]:
        if not job_key or "/" in job_key or ".." in job_key:
            raise ValueError("Invalid cloud job key")
        volume = self._volume()
        prefix = f"jobs/{job_key}"
        removed, errors = [], []
        try:
            next(iter(volume.iterdir(prefix, recursive=True)))
        except StopIteration:
            removed.append(prefix)
        except Exception as exc:
            if self._is_not_found_error(exc):
                removed.append(prefix)
            else:
                errors.append(f"{prefix}: {self._redact(str(exc))}")
        else:
            try:
                volume.remove_file(prefix, recursive=True)
                removed.append(prefix)
            except FileNotFoundError:
                removed.append(prefix)
            except Exception as exc:
                if self._is_not_found_error(exc):
                    removed.append(prefix)
                else:
                    errors.append(f"{prefix}: {self._redact(str(exc))}")
        return {
            "success": not errors,
            "removed": removed,
            "error": "; ".join(errors),
        }

    def dashboard_url(self, call_id: str) -> str:
        del call_id
        return "https://modal.com/apps"

    @staticmethod
    def _parse_remote_ref(remote_ref: str) -> tuple[str, str]:
        prefix = "modal-volume://"
        if not remote_ref.startswith(prefix):
            raise ValueError("Unsupported Modal artifact reference")
        value = remote_ref[len(prefix) :]
        volume_name, separator, remote_path = value.partition("/")
        if not separator or not volume_name or not remote_path or ".." in Path(remote_path).parts:
            raise ValueError("Invalid Modal artifact reference")
        return volume_name, remote_path
