"""Lazy Modal SDK adapter for asynchronous training and volume file operations."""

from __future__ import annotations

import importlib
import io
from pathlib import Path
from typing import Any

from ..models.backends import OptionalDependencyError
from .constants import MODAL_APP_NAME, VOLUME_NAME
from .credentials import CloudCredentials


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

    def _volume(self, *, create_if_missing: bool = False):
        modal = self._load_modal()
        return modal.Volume.from_name(
            self.volume_name,
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
        provider_error = getattr(getattr(self._modal, "exception", None), "NotFoundError", None)
        return isinstance(provider_error, type) and isinstance(error, provider_error)

    def _redact(self, value: str) -> str:
        if self._credentials is None:
            return value
        return value.replace(self._credentials.token_secret, "[redacted]").replace(
            self._credentials.token_id, "[redacted]"
        )

    def upload(self, source: Path, remote_path: str, sha256: str) -> bool:
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            volume = self._volume(create_if_missing=True)
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
            return {"state": "running", **progress} if progress else None
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
