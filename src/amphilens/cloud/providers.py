"""Lazy Azure ML and Vertex AI adapters for managed cloud jobs.

Provider SDK objects, credentials, and job handles stay inside these adapters. Only
resource identifiers and checksummed artifact references cross the application boundary.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from ..models.backends import OptionalDependencyError
from .provider_settings import AzureMLSettings, VertexAISettings

_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,400}$")
_GPU_NAMES = {"T4", "A10G", "A100", "L4"}
_PROVIDER_GPUS = {
    "azure_ml": {"T4", "A10G", "A100"},
    "vertex_ai": {"T4", "L4", "A100"},
}
_SECRET_FIELD = re.compile(
    r"(?:^|[_-])(?:access[_-]?token|refresh[_-]?token|token|secret|password|api[_-]?key|credential)(?:$|[_-])",
    re.I,
)
_SIGNED_URL = re.compile(r"(?i)https?://[^\s?#]+\?[^\s]*?(?:sig|signature|token|x-amz-signature)=")
_AZURE_VM_TYPES = {
    "T4": "Standard_NC4as_T4_v3",
    "A10G": "Standard_NV36ads_A10_v5",
    "A100": "Standard_NC24ads_A100_v4",
}
_VERTEX_MACHINE_TYPES = {
    "T4": "n1-standard-4",
    "L4": "g2-standard-4",
    "A100": "a2-highgpu-1g",
}
_VERTEX_ACCELERATORS = {
    "T4": "NVIDIA_TESLA_T4",
    "L4": "NVIDIA_L4",
    "A100": "NVIDIA_TESLA_A100",
}


def _safe_key(value: str) -> str:
    key = str(value).strip().strip("/")
    if not _KEY.fullmatch(key) or ".." in key.split("/"):
        raise ValueError("Cloud object path is invalid")
    return key


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_no_secret_values(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _SECRET_FIELD.search(str(key)):
                raise ValueError("Cloud job payload cannot contain credentials or tokens")
            _assert_no_secret_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_secret_values(item)
    elif isinstance(value, str) and _SIGNED_URL.search(value):
        raise ValueError("Cloud job payload cannot contain signed URLs")


class _ManagedTransport:
    provider = ""

    def __init__(self):
        self._storage = None
        self._jobs = None
        self._sdk = None

    def describe(self) -> dict[str, str]:
        return {"provider": self.provider, "region": self.region}

    def price_rate(self, gpu: str):
        from .pricing import azure_vm_rate, vertex_vm_rate

        if self.provider == "azure_ml":
            return azure_vm_rate(gpu, self.region)
        return vertex_vm_rate(gpu, self.region)

    @property
    def region(self) -> str:
        raise NotImplementedError

    def redact(self, message: str) -> str:
        # SDKs can echo credentials in errors. Drop URL query strings because SAS and
        # signed GCS URLs carry reusable credentials there.
        value = re.sub(r"(Bearer\s+)[^\s]+", r"\1[redacted]", str(message), flags=re.I)
        value = re.sub(r"(?i)(https?://[^\s?#]+)\?[^\s\"'<>)]*", r"\1?[redacted]", value)
        return re.sub(
            r"(?i)((?:token|secret|password|api[_-]?key|access[_-]?token|client[_-]?secret|sig|signature)\b\s*[:=]\s*)[^\s,;&]+",
            r"\1[redacted]",
            value,
        )

    def _object_uri(self, key: str) -> str:
        raise NotImplementedError

    def _put(self, source: Path, key: str, sha256: str) -> bool:
        raise NotImplementedError

    def _get(self, key: str, destination: Path) -> None:
        raise NotImplementedError

    def _delete_prefix(self, prefix: str) -> list[str]:
        raise NotImplementedError

    def _delete_object(self, key: str) -> None:
        raise NotImplementedError

    def upload(self, source: Path, remote_path: str, sha256: str) -> bool:
        source = Path(source).expanduser().resolve()
        if not source.is_file() or _sha256(source) != sha256:
            raise ValueError("Cloud upload source is missing or its SHA-256 does not match")
        return self._put(source, _safe_key(remote_path), sha256)

    def upload_model_checkpoint(
        self, source: Path, sha256: str, *, job_key: str | None = None
    ) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("Invalid model checkpoint SHA-256")
        if job_key is None or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_key):
            raise ValueError("A valid prediction job identity is required for checkpoint staging")
        key = f"prediction-jobs/{job_key}/checkpoints/{sha256}.pt"
        self.upload(source, key, sha256)
        return key

    def upload_prediction_batch(
        self, job_key: str, batch_id: str, sources: list[Path]
    ) -> list[dict[str, str]]:
        from .modal_transport import _sha256_file

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_key) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,80}", batch_id
        ):
            raise ValueError("Invalid prediction batch identity")
        if not sources or len(sources) > 32:
            raise ValueError("Prediction uploads must contain between 1 and 32 images")
        images = []
        for source in sources:
            path = Path(source).expanduser().resolve()
            suffix = path.suffix.lower()
            if not path.is_file() or suffix not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff",
                ".webp",
            }:
                raise ValueError("Cloud prediction image is missing or has an unsupported type")
            digest = _sha256_file(path)
            image_id = "img-" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:20]
            remote_path = f"prediction-jobs/{job_key}/{batch_id}/images/{image_id}{suffix}"
            self.upload(path, remote_path, digest)
            images.append({"image_id": image_id, "remote_path": remote_path, "sha256": digest})
        manifest = json.dumps({"schema_version": 1, "images": images}, sort_keys=True)
        self._put_bytes(
            manifest.encode("utf-8"),
            f"prediction-jobs/{job_key}/{batch_id}/manifest.json",
            hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
        )
        return images

    def _put_bytes(self, content: bytes, key: str, sha256: str) -> None:
        with tempfile.NamedTemporaryFile(prefix="amphilens-cloud-") as handle:
            handle.write(content)
            handle.flush()
            self._put(Path(handle.name), _safe_key(key), sha256)

    def download(self, remote_ref: str, destination: Path) -> None:
        key = self._parse_remote_ref(remote_ref)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._get(key, destination)

    def _parse_remote_ref(self, remote_ref: str) -> str:
        raise NotImplementedError

    def cleanup(self, job_key: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_key):
            raise ValueError("Invalid cloud job key")
        try:
            removed = self._delete_prefix(f"jobs/{job_key}")
            return {"success": True, "removed": removed, "error": ""}
        except Exception as exc:
            return {"success": False, "removed": [], "error": self.redact(str(exc))[:500]}

    def cleanup_prediction_batch(self, job_key: str, batch_id: str) -> None:
        self._validate_prediction_identity(job_key, batch_id)
        self._delete_prefix(f"prediction-jobs/{job_key}/{batch_id}")

    def cleanup_prediction_progress(self, job_key: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_key):
            raise ValueError("Invalid prediction job identity")
        self._delete_object(f"prediction-jobs/{job_key}/progress.json")
        self._delete_prefix(f"prediction-jobs/{job_key}/checkpoints")

    @staticmethod
    def _validate_prediction_identity(job_key: str, batch_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_key) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,80}", batch_id
        ):
            raise ValueError("Invalid prediction batch identity")

    def _job_state(self, call_id: str) -> tuple[Any, str]:
        raise NotImplementedError

    def _submit_job(self, payload: dict[str, Any]) -> str:
        raise NotImplementedError

    def submit(self, payload: dict[str, Any]) -> str:
        _assert_no_secret_values(payload)
        operation = str(payload.get("operation", "train"))
        if operation not in {"train", "predict"}:
            raise ValueError("Unsupported managed cloud job operation")
        gpu = str(payload.get("gpu", ""))
        if gpu not in _GPU_NAMES or gpu not in _PROVIDER_GPUS[self.provider]:
            raise ValueError(f"Unsupported {self.provider} GPU selection: {gpu}")
        if operation == "train":
            remote_prefix = _safe_key(str(payload.get("remote_prefix", "")))
            metadata = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self._put_bytes(
                metadata,
                f"{remote_prefix}/request.json",
                hashlib.sha256(metadata).hexdigest(),
            )
        return self._submit_job(payload)

    def poll(self, call_id: str, *, remote_prefix: str) -> dict[str, Any] | None:
        job, state = self._job_state(call_id)
        result_prefix = _safe_key(remote_prefix)
        progress_prefix = result_prefix
        if result_prefix.startswith("prediction-jobs/"):
            parts = result_prefix.split("/")
            if len(parts) >= 3:
                progress_prefix = "/".join(parts[:2])
        if state == "running":
            progress = self._read_json(f"{progress_prefix}/progress.json")
            if not progress:
                return None
            return {
                "state": "running",
                "remote_state": str(getattr(job, "status", "running")),
                "progress_details": progress,
                "progress": progress.get("progress"),
            }
        result = self._read_json(f"{result_prefix}/result.json")
        if result:
            result.setdefault("remote_state", str(getattr(job, "status", state)))
            return result
        if state == "canceled":
            return {"state": "canceled", "remote_state": str(getattr(job, "status", state))}
        if state == "failed":
            return {
                "state": "failed",
                "remote_state": str(getattr(job, "status", state)),
                "error": "Managed cloud job failed before publishing a verified result",
            }
        return None

    def _read_json(self, key: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="amphilens-cloud-") as temp:
            path = Path(temp) / "value.json"
            try:
                self._get(key, path)
            except Exception:
                return {}
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
            return value if isinstance(value, dict) else {}

    def cancel(self, call_id: str) -> None:
        job, _ = self._job_state(call_id)
        cancel = getattr(job, "cancel", None)
        if not callable(cancel):
            raise RuntimeError("The managed cloud SDK does not support job cancellation")
        cancel()

    def dashboard_url(self, call_id: str) -> str:
        del call_id
        return self._dashboard_url()

    def _dashboard_url(self) -> str:
        raise NotImplementedError

    def predict_batch(
        self,
        payload: dict[str, Any],
        *,
        cancellation_requested=None,
        progress_callback=None,
        job_id_callback=None,
    ) -> dict[str, Any]:
        from .prediction import PredictionCancelled

        job_key = str(payload.get("job_key", ""))
        batch_id = str(payload.get("batch_id", ""))
        self._validate_prediction_identity(job_key, batch_id)
        prefix = f"prediction-jobs/{job_key}/{batch_id}"
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self._put_bytes(encoded, f"{prefix}/request.json", hashlib.sha256(encoded).hexdigest())
        call_id = self.submit(
            {
                "operation": "predict",
                "job_key": job_key,
                "batch_id": batch_id,
                "gpu": payload["gpu"],
                "timeout_seconds": payload["timeout_seconds"],
                "remote_prefix": f"prediction-jobs/{job_key}",
                "request_remote_path": f"{prefix}/request.json",
                "result_remote_path": f"{prefix}/result.json",
            }
        )
        if job_id_callback is not None:
            job_id_callback(call_id)
        deadline = time.monotonic() + int(payload.get("timeout_seconds", 3600)) + 120
        while time.monotonic() < deadline:
            if cancellation_requested and cancellation_requested():
                self.cancel(call_id)
                raise PredictionCancelled("Prediction canceled. The active cloud job was stopped.")
            result = self.poll(call_id, remote_prefix=prefix)
            if result is not None and result.get("state") != "running":
                result.setdefault("remote_job_id", call_id)
                return result
            if result and progress_callback:
                progress_callback(dict(result.get("progress_details", {})))
            time.sleep(2)
        self.cancel(call_id)
        raise TimeoutError("Managed cloud prediction exceeded its local wait deadline")


class AzureMLTransport(_ManagedTransport):
    """Azure ML serverless command jobs backed by the workspace default datastore."""

    provider = "azure_ml"

    def __init__(self, settings: AzureMLSettings):
        super().__init__()
        if not isinstance(settings, AzureMLSettings):
            raise ValueError("Azure ML transport requires Azure ML settings")
        self.settings = settings
        self._credential = None
        self._datastore = None

    @property
    def region(self) -> str:
        return self.settings.location

    def _load_sdk(self):
        if self._sdk is None:
            try:
                identity = importlib.import_module("azure.identity")
                ml = importlib.import_module("azure.ai.ml")
                entities = importlib.import_module("azure.ai.ml.entities")
                storage = importlib.import_module("azure.storage.blob")
            except ImportError as exc:
                raise OptionalDependencyError(
                    "Azure ML support requires `pip install 'amphilens[cloud-azure]'`"
                ) from exc
            self._sdk = (identity, ml, entities, storage)
        return self._sdk

    def _clients(self):
        if self._jobs is not None:
            return self._jobs
        identity, ml, _, storage = self._load_sdk()
        try:
            self._credential = identity.DefaultAzureCredential()
            client = ml.MLClient(
                self._credential,
                subscription_id=self.settings.subscription_id,
                resource_group_name=self.settings.resource_group,
                workspace_name=self.settings.workspace_name,
            )
            self._datastore = client.datastores.get("workspaceblobstore")
            account = str(getattr(self._datastore, "account_name", ""))
            container = str(getattr(self._datastore, "container_name", ""))
            if not account or not container:
                raise RuntimeError(
                    "Azure ML workspaceblobstore has no readable storage coordinates"
                )
            service = storage.BlobServiceClient(
                account_url=f"https://{account}.blob.core.windows.net",
                credential=self._credential,
            )
            self._storage = service.get_container_client(container)
            self._jobs = client
        except Exception as exc:
            raise RuntimeError(self.redact(f"Azure ML connection failed: {exc}")) from exc
        return self._jobs

    def probe(self) -> dict[str, str]:
        client = self._clients()
        try:
            self._credential.get_token("https://management.azure.com/.default")
            workspace = client.workspaces.get(self.settings.workspace_name)
            if not workspace:
                raise RuntimeError("Azure ML workspace was not found")
            identity = getattr(workspace, "identity", None)
            assigned = getattr(identity, "user_assigned_identities", None) or {}
            if not assigned:
                raise RuntimeError("Configure a user-assigned managed identity on the workspace")
            if not any(
                str(self.settings.managed_identity_client_id).lower()
                in str(
                    identity_data.get("client_id", "")
                    if isinstance(identity_data, dict)
                    else getattr(identity_data, "client_id", "")
                ).lower()
                for resource_id, identity_data in assigned.items()
            ):
                raise RuntimeError(
                    "The configured managed identity is not attached to the Azure ML workspace"
                )
            self._storage.get_container_properties()
            return {"provider": self.provider, "connectivity": "connected", "region": self.region}
        except Exception as exc:
            raise RuntimeError(self.redact(f"Azure ML connection check failed: {exc}")) from exc

    def _object_uri(self, key: str) -> str:
        self._clients()
        container = str(getattr(self._datastore, "container_name"))
        return f"azureml-blob://{container}/{_safe_key(key)}"

    def _put(self, source: Path, key: str, sha256: str) -> bool:
        self._clients()
        blob = self._storage.get_blob_client(_safe_key(key))
        try:
            props = blob.get_blob_properties()
            if (getattr(props, "metadata", None) or {}).get("sha256") == sha256:
                return False
        except Exception:
            pass
        with source.open("rb") as handle:
            blob.upload_blob(handle, overwrite=True, metadata={"sha256": sha256})
        return True

    def _get(self, key: str, destination: Path) -> None:
        self._clients()
        blob = self._storage.get_blob_client(_safe_key(key))
        with destination.open("wb") as output:
            output.write(blob.download_blob().readall())

    def _delete_prefix(self, prefix: str) -> list[str]:
        self._clients()
        removed = []
        for item in self._storage.list_blobs(name_starts_with=_safe_key(prefix).rstrip("/") + "/"):
            self._storage.delete_blob(item.name)
            removed.append(item.name)
        return removed

    def _delete_object(self, key: str) -> None:
        self._clients()
        try:
            self._storage.delete_blob(_safe_key(key))
        except Exception as exc:
            if "not found" not in str(exc).lower() and "404" not in str(exc):
                raise

    def _parse_remote_ref(self, remote_ref: str) -> str:
        self._clients()
        prefix = f"azureml-blob://{getattr(self._datastore, 'container_name', '')}/"
        if not remote_ref.startswith(prefix):
            raise ValueError("Unsupported Azure ML artifact reference")
        return _safe_key(remote_ref[len(prefix) :])

    def _submit_job(self, payload: dict[str, Any]) -> str:
        client = self._clients()
        _, ml, entities, _ = self._load_sdk()
        operation = str(payload.get("operation", "train"))
        job_key = _safe_key(str(payload.get("job_key", "")))
        remote_prefix = _safe_key(str(payload.get("remote_prefix", f"jobs/{job_key}")))
        if operation == "train":
            request_path = f"{remote_prefix}/payload.zip"
            result_path = f"{remote_prefix}/result.json"
        else:
            request_path = str(payload["request_remote_path"])
            result_path = str(payload["result_remote_path"])
        command = self._worker_command(operation, request_path, result_path)
        env = entities.Environment(image="pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")
        command_job = ml.command(
            code=str(Path(__file__).resolve().parents[2]),
            command=command,
            environment=env,
            environment_variables={
                "AMPHILENS_PROVIDER": self.provider,
                "AMPHILENS_JOB_KEY": job_key,
                "AMPHILENS_STORAGE_ACCOUNT": str(getattr(self._datastore, "account_name")),
                "AMPHILENS_STORAGE_CONTAINER": str(getattr(self._datastore, "container_name")),
                "AZURE_CLIENT_ID": self.settings.managed_identity_client_id,
            },
            resources=entities.JobResourceConfiguration(
                instance_count=1,
                instance_type=_AZURE_VM_TYPES[str(payload["gpu"])],
            ),
            identity=entities.ManagedIdentityConfiguration(
                client_id=self.settings.managed_identity_client_id
            ),
            limits=entities.CommandJobLimits(timeout=int(payload["timeout_seconds"])),
        )
        try:
            job = client.jobs.create_or_update(command_job)
            return str(job.name)
        except Exception as exc:
            raise RuntimeError(self.redact(f"Azure ML job submission failed: {exc}")) from exc

    @staticmethod
    def _worker_command(operation: str, request_path: str, result_path: str) -> str:
        # Provider credentials come from the job identity; no tokens are put in the payload.
        return (
            "python -m pip install 'ultralytics==8.4.163' 'Pillow==12.3.0' 'PyYAML==6.0.3' "
            "'numpy==2.3.5' 'opencv-python-headless==5.0.0.93' 'huggingface-hub==0.35.3' "
            "'azure-identity>=1.17' 'azure-storage-blob>=12.20' && "
            f"python -m amphilens.cloud.provider_worker {operation} "
            f"--request-key '{request_path}' --result-key '{result_path}'"
        )

    def cancel(self, call_id: str) -> None:
        client = self._clients()
        try:
            client.jobs.begin_cancel(call_id)
        except Exception as exc:
            raise RuntimeError(self.redact(f"Azure ML job cancellation failed: {exc}")) from exc

    def _job_state(self, call_id: str) -> tuple[Any, str]:
        client = self._clients()
        try:
            job = client.jobs.get(call_id)
        except Exception as exc:
            raise RuntimeError(self.redact(f"Azure ML job lookup failed: {exc}")) from exc
        raw = str(getattr(job, "status", "")).lower()
        if raw in {"completed"}:
            state = "finished"
        elif raw in {"failed"}:
            state = "failed"
        elif raw in {"canceled", "cancelled"}:
            state = "canceled"
        else:
            state = "running"
        return job, state

    def _dashboard_url(self) -> str:
        return "https://ml.azure.com/"


class VertexAITransport(_ManagedTransport):
    """Vertex AI CustomJob and GCS adapter using Application Default Credentials."""

    provider = "vertex_ai"

    def __init__(self, settings: VertexAISettings):
        super().__init__()
        if not isinstance(settings, VertexAISettings):
            raise ValueError("Vertex AI transport requires Vertex AI settings")
        self.settings = settings
        self._aiplatform = None

    @property
    def region(self) -> str:
        return self.settings.location

    def _clients(self):
        if self._jobs is not None:
            return self._jobs
        try:
            self._aiplatform = importlib.import_module("google.cloud.aiplatform")
            storage = importlib.import_module("google.cloud.storage")
        except ImportError as exc:
            raise OptionalDependencyError(
                "Vertex AI support requires `pip install 'amphilens[cloud-gcp]'`"
            ) from exc
        try:
            self._aiplatform.init(
                project=self.settings.project_id,
                location=self.settings.location,
                staging_bucket=f"gs://{self.settings.staging_bucket}",
            )
            self._storage = storage.Client(project=self.settings.project_id).bucket(
                self.settings.staging_bucket
            )
            self._jobs = self._aiplatform
        except Exception as exc:
            raise RuntimeError(self.redact(f"Vertex AI connection failed: {exc}")) from exc
        return self._jobs

    def probe(self) -> dict[str, str]:
        self._clients()
        try:
            if not self._storage.exists():
                raise RuntimeError("Configured Vertex AI staging bucket is not accessible")
            client = self._aiplatform.gapic.JobServiceClient(
                client_options={
                    "api_endpoint": f"{self.settings.location}-aiplatform.googleapis.com"
                }
            )
            parent = f"projects/{self.settings.project_id}/locations/{self.settings.location}"
            next(iter(client.list_custom_jobs(parent=parent, page_size=1)), None)
            return {"provider": self.provider, "connectivity": "connected", "region": self.region}
        except Exception as exc:
            raise RuntimeError(self.redact(f"Vertex AI connection check failed: {exc}")) from exc

    def _object_uri(self, key: str) -> str:
        self._clients()
        return f"gs://{self.settings.staging_bucket}/{_safe_key(key)}"

    def _put(self, source: Path, key: str, sha256: str) -> bool:
        self._clients()
        blob = self._storage.blob(_safe_key(key))
        if blob.exists() and (blob.metadata or {}).get("sha256") == sha256:
            return False
        blob.metadata = {"sha256": sha256}
        blob.upload_from_filename(str(source))
        return True

    def _get(self, key: str, destination: Path) -> None:
        self._clients()
        self._storage.blob(_safe_key(key)).download_to_filename(str(destination))

    def _delete_prefix(self, prefix: str) -> list[str]:
        self._clients()
        removed = []
        for blob in self._storage.list_blobs(prefix=_safe_key(prefix).rstrip("/") + "/"):
            name = str(blob.name)
            blob.delete()
            removed.append(name)
        return removed

    def _delete_object(self, key: str) -> None:
        self._clients()
        blob = self._storage.blob(_safe_key(key))
        if blob.exists():
            blob.delete()

    def _parse_remote_ref(self, remote_ref: str) -> str:
        prefix = f"gs://{self.settings.staging_bucket}/"
        if not remote_ref.startswith(prefix):
            raise ValueError("Unsupported Vertex AI artifact reference")
        return _safe_key(remote_ref[len(prefix) :])

    def _submit_job(self, payload: dict[str, Any]) -> str:
        self._clients()
        operation = str(payload.get("operation", "train"))
        job_key = _safe_key(str(payload.get("job_key", "")))
        prefix = _safe_key(str(payload.get("remote_prefix", f"jobs/{job_key}")))
        request_key = (
            f"{prefix}/payload.zip" if operation == "train" else str(payload["request_remote_path"])
        )
        result_key = (
            f"{prefix}/result.json" if operation == "train" else str(payload["result_remote_path"])
        )
        source_key = f"jobs/{job_key}/source.zip"
        source_path = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="amphilens-source-") as temp:
            archive = Path(temp) / "source.zip"
            import zipfile

            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                for item in sorted(source_path.rglob("*")):
                    if not item.is_file() or "__pycache__" in item.parts:
                        continue
                    output.write(item, item.relative_to(source_path.parent).as_posix())
            self._put(archive, source_key, _sha256(archive))
        gpu = str(payload["gpu"])
        image = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"
        script = (
            "set -e; python -m pip install 'ultralytics==8.4.163' 'Pillow==12.3.0' "
            "'PyYAML==6.0.3' 'numpy==2.3.5' 'opencv-python-headless==5.0.0.93' "
            "'huggingface-hub==0.35.3' 'google-cloud-storage>=2.18'; "
            "mkdir -p /tmp/amphilens-source; "
            "python -c 'import os; from google.cloud import storage; "
            'storage.Client(project=os.environ["AMPHILENS_GCP_PROJECT"]).bucket('
            'os.environ["AMPHILENS_GCS_BUCKET"]).blob(os.environ["AMPHILENS_SOURCE_KEY"])'
            '.download_to_filename("/tmp/amphilens-source/source.zip")\'; '
            "cd /tmp/amphilens-source; "
            "python -m zipfile -e source.zip .; "
            "export PYTHONPATH=/tmp/amphilens-source/src:$PYTHONPATH; "
            f"python -m amphilens.cloud.provider_worker {operation} --request-key '{request_key}' "
            f"--result-key '{result_key}'"
        )
        worker_pool_specs = [
            {
                "machine_spec": {
                    "machine_type": _VERTEX_MACHINE_TYPES[gpu],
                    "accelerator_type": _VERTEX_ACCELERATORS[gpu],
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": image,
                    "command": ["bash", "-c"],
                    "args": [script],
                    "env": [
                        {"name": "AMPHILENS_PROVIDER", "value": self.provider},
                        {"name": "AMPHILENS_GCP_PROJECT", "value": self.settings.project_id},
                        {"name": "AMPHILENS_GCS_BUCKET", "value": self.settings.staging_bucket},
                        {"name": "AMPHILENS_SOURCE_KEY", "value": source_key},
                        {"name": "AMPHILENS_JOB_KEY", "value": job_key},
                    ],
                },
            }
        ]
        try:
            job = self._aiplatform.CustomJob(
                display_name=f"amphilens-{operation}-{job_key[:16]}",
                worker_pool_specs=worker_pool_specs,
                staging_bucket=f"gs://{self.settings.staging_bucket}",
                base_output_dir=self._object_uri(f"jobs/{job_key}/vertex-output"),
            )
            job.run(
                service_account=self.settings.service_account,
                sync=False,
                timeout=int(payload["timeout_seconds"]),
            )
            return str(getattr(job, "resource_name", "") or getattr(job, "name", ""))
        except Exception as exc:
            raise RuntimeError(self.redact(f"Vertex AI job submission failed: {exc}")) from exc

    def _job_state(self, call_id: str) -> tuple[Any, str]:
        self._clients()
        try:
            job = self._aiplatform.CustomJob.get(call_id)
        except Exception as exc:
            raise RuntimeError(self.redact(f"Vertex AI job lookup failed: {exc}")) from exc
        raw = str(getattr(job, "state", "")).lower()
        if raw.endswith("succeeded") or raw.endswith("job_state_succeeded"):
            state = "finished"
        elif raw.endswith("failed") or raw.endswith("job_state_failed"):
            state = "failed"
        elif raw.endswith("cancelled") or raw.endswith("canceled"):
            state = "canceled"
        else:
            state = "running"
        return job, state

    def _dashboard_url(self) -> str:
        return (
            "https://console.cloud.google.com/vertex-ai/training/training-pipelines?"
            f"project={self.settings.project_id}&region={self.settings.location}"
        )

    def cleanup_prediction_progress(self, job_key: str) -> None:
        super().cleanup_prediction_progress(job_key)
        self._delete_prefix(f"jobs/{_safe_key(job_key)}")


def create_cloud_transport(provider: str, settings: Any):
    """Construct a provider adapter without importing its optional SDK."""
    if provider == "azure_ml" and isinstance(settings, AzureMLSettings):
        return AzureMLTransport(settings)
    if provider == "vertex_ai" and isinstance(settings, VertexAISettings):
        return VertexAITransport(settings)
    if provider not in {"azure_ml", "vertex_ai"}:
        raise ValueError(f"Unsupported managed cloud provider: {provider}")
    raise ValueError(f"Invalid settings for cloud provider {provider}")
