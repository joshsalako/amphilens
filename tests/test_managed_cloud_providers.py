from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from amphilens.cloud.pricing import azure_vm_rate, vertex_vm_rate
from amphilens.cloud.provider_settings import AzureMLSettings, VertexAISettings


def test_provider_transport_factory_is_lazy_and_provider_neutral():
    from amphilens.cloud.providers import create_cloud_transport

    azure = create_cloud_transport(
        "azure_ml",
        AzureMLSettings("sub", "rg", "workspace", "eastus", "identity"),
    )
    vertex = create_cloud_transport(
        "vertex_ai",
        VertexAISettings(
            "project", "us-central1", "staging", "runner@example.iam.gserviceaccount.com"
        ),
    )

    assert azure.describe() == {"provider": "azure_ml", "region": "eastus"}
    assert vertex.describe() == {"provider": "vertex_ai", "region": "us-central1"}
    assert "azure.ai.ml" not in sys.modules
    assert "google.cloud.aiplatform" not in sys.modules


def test_azure_price_lookup_returns_full_vm_price_and_source():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {
                    "Items": [
                        {
                            "serviceName": "Virtual Machines",
                            "armSkuName": "Standard_NC4as_T4_v3",
                            "armRegionName": "eastus",
                            "unitOfMeasure": "1 Hour",
                            "retailPrice": 1.25,
                        }
                    ]
                }
            ).encode()

    rate = azure_vm_rate("T4", "eastus", opener=lambda *_args, **_kwargs: Response())

    assert rate.usd_per_hour == 1.25
    assert rate.price_source == "Azure Retail Prices API"


def test_vertex_price_lookup_requires_gpu_cpu_and_ram_rates():
    class Response:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            return None

        def json(self):
            return self.value

    skus = [
        _vertex_sku("NVIDIA Tesla T4 GPU running in us-central1", 0.3),
        _vertex_sku("N1 Predefined Instance Core running in us-central1", 0.05),
        _vertex_sku("N1 Predefined Instance Ram running in us-central1", 0.01),
    ]

    rate = vertex_vm_rate(
        "T4",
        "us-central1",
        session=type("Session", (), {"get": lambda *_args, **_kwargs: Response({"skus": skus})})(),
    )

    assert rate.usd_per_hour == pytest.approx(0.65)
    assert "GPU, vCPU, and RAM" in rate.price_source


def test_missing_price_data_blocks_managed_cloud_submission():
    with pytest.raises(ValueError, match="submission is blocked"):
        azure_vm_rate(
            "T4", "eastus", opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError())
        )


def test_provider_transport_rejects_settings_for_another_provider():
    from amphilens.cloud.providers import create_cloud_transport

    with pytest.raises(ValueError, match="settings"):
        create_cloud_transport(
            "azure_ml",
            VertexAISettings(
                "project", "us-central1", "staging", "runner@example.iam.gserviceaccount.com"
            ),
        )


@pytest.mark.parametrize(
    "secret_url",
    [
        "https://storage.example/object?sig=secret-value&se=2030-01-01",
        "https://storage.example/object?X-Goog-Signature=secret-value",
    ],
)
def test_managed_provider_redaction_removes_bearer_tokens_and_signed_url_queries(secret_url):
    from amphilens.cloud.providers import create_cloud_transport

    transport = create_cloud_transport(
        "azure_ml", AzureMLSettings("sub", "rg", "workspace", "eastus", "identity")
    )
    redacted = transport.redact(f"Bearer bearer-secret {secret_url}")

    assert "bearer-secret" not in redacted
    assert "secret-value" not in redacted
    assert "[redacted]" in redacted


def test_azure_adapter_checks_identity_storage_and_cancels_with_sdk(monkeypatch):
    from amphilens.cloud.providers import AzureMLTransport

    transport = AzureMLTransport(
        AzureMLSettings("sub", "rg", "workspace", "eastus", "identity-client")
    )
    calls = []
    transport._credential = SimpleNamespace(get_token=lambda scope: calls.append(scope))
    workspace = SimpleNamespace(
        identity=SimpleNamespace(
            user_assigned_identities={"/identity": {"client_id": "identity-client"}}
        )
    )
    client = SimpleNamespace(
        workspaces=SimpleNamespace(get=lambda name: workspace),
        jobs=SimpleNamespace(begin_cancel=lambda name: calls.append(name)),
    )
    transport._jobs = client
    transport._storage = SimpleNamespace(get_container_properties=lambda: calls.append("storage"))

    status = transport.probe()
    transport.cancel("remote-job")

    assert status == {"provider": "azure_ml", "connectivity": "connected", "region": "eastus"}
    assert calls == ["https://management.azure.com/.default", "storage", "remote-job"]


def test_azure_adapter_redacts_authentication_failures():
    from amphilens.cloud.providers import AzureMLTransport

    transport = AzureMLTransport(
        AzureMLSettings("sub", "rg", "workspace", "eastus", "identity-client")
    )
    transport._credential = SimpleNamespace(
        get_token=lambda _scope: (_ for _ in ()).throw(RuntimeError("Bearer token-secret"))
    )
    transport._jobs = SimpleNamespace(workspaces=SimpleNamespace(get=lambda _name: object()))
    transport._storage = SimpleNamespace()

    with pytest.raises(RuntimeError) as error:
        transport.probe()

    assert "token-secret" not in str(error.value)


def test_vertex_adapter_checks_bucket_and_job_api():
    from amphilens.cloud.providers import VertexAITransport

    transport = VertexAITransport(
        VertexAISettings("project", "us-central1", "staging", "runner@example.com")
    )
    calls = []

    class JobServiceClient:
        def __init__(self, *, client_options):
            calls.append(client_options)

        def list_custom_jobs(self, *, parent, page_size):
            calls.append((parent, page_size))
            return iter(())

    transport._jobs = object()
    transport._storage = SimpleNamespace(exists=lambda: True)
    transport._aiplatform = SimpleNamespace(
        gapic=SimpleNamespace(JobServiceClient=JobServiceClient)
    )

    assert transport.probe() == {
        "provider": "vertex_ai",
        "connectivity": "connected",
        "region": "us-central1",
    }
    assert calls[-1] == ("projects/project/locations/us-central1", 1)


@pytest.mark.parametrize("operation", ["train", "predict"])
def test_vertex_adapter_submits_custom_job_with_least_privilege_identity(operation):
    from amphilens.cloud.providers import VertexAITransport

    captured = {}

    class CustomJob:
        resource_name = "projects/p/locations/us-central1/customJobs/123"

        def __init__(self, **kwargs):
            captured["job"] = kwargs

        def run(self, **kwargs):
            captured["run"] = kwargs

    class FakeTransport(VertexAITransport):
        def _clients(self):
            self._aiplatform = SimpleNamespace(CustomJob=CustomJob)
            self._jobs = self._aiplatform
            return self._jobs

        def _put(self, source, key, sha256):
            captured["source_key"] = key
            captured["source_sha256"] = sha256
            captured["source_size"] = Path(source).stat().st_size
            return True

        def _object_uri(self, key):
            return f"gs://staging/{key}"

    transport = FakeTransport(
        VertexAISettings("project", "us-central1", "staging", "runner@example.com")
    )
    payload = {
        "operation": operation,
        "job_key": "job-1",
        "gpu": "T4",
        "timeout_seconds": 60,
    }
    if operation == "predict":
        payload.update(
            request_remote_path="prediction-jobs/pred-1/batch-1/request.json",
            result_remote_path="prediction-jobs/pred-1/batch-1/result.json",
        )
    remote_id = transport._submit_job(payload)

    assert remote_id.endswith("customJobs/123")
    assert captured["run"] == {
        "service_account": "runner@example.com",
        "sync": False,
        "timeout": 60,
    }
    assert captured["source_key"] == "jobs/job-1/source.zip"
    env = captured["job"]["worker_pool_specs"][0]["container_spec"]["env"]
    assert {item["name"]: item["value"] for item in env}["AMPHILENS_SOURCE_KEY"] == captured[
        "source_key"
    ]


def test_azure_adapter_submits_serverless_command_with_gpu_and_managed_identity():
    from amphilens.cloud.providers import AzureMLTransport

    captured = {}

    class FakeML:
        @staticmethod
        def command(**kwargs):
            captured["command"] = kwargs
            return kwargs

    class FakeClient:
        class jobs:
            @staticmethod
            def create_or_update(job):
                captured["submitted"] = job
                return SimpleNamespace(name="azure-job-123")

    class FakeEntities:
        class Environment:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class JobResourceConfiguration:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class ManagedIdentityConfiguration:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class CommandJobLimits:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    transport = AzureMLTransport(
        AzureMLSettings("sub", "rg", "workspace", "eastus", "identity-client")
    )
    transport._jobs = FakeClient()
    transport._sdk = (None, FakeML, FakeEntities, None)
    transport._datastore = SimpleNamespace(
        account_name="storageaccount", container_name="workspace-container"
    )

    remote_id = transport._submit_job(
        {
            "operation": "train",
            "job_key": "training-1",
            "gpu": "T4",
            "timeout_seconds": 600,
        }
    )

    command = captured["command"]
    assert remote_id == "azure-job-123"
    assert "provider_worker train" in command["command"]
    assert "--request-key 'jobs/training-1/payload.zip'" in command["command"]
    assert command["resources"].kwargs == {
        "instance_count": 1,
        "instance_type": "Standard_NC4as_T4_v3",
    }
    assert command["identity"].kwargs == {"client_id": "identity-client"}
    assert command["environment_variables"]["AMPHILENS_JOB_KEY"] == "training-1"


@pytest.mark.parametrize("provider", ["azure_ml", "vertex_ai"])
def test_managed_transport_job_storage_lifecycle_contract(tmp_path, provider, monkeypatch):
    from amphilens.cloud.providers import AzureMLTransport, VertexAITransport

    base = AzureMLTransport if provider == "azure_ml" else VertexAITransport
    settings = (
        AzureMLSettings("sub", "rg", "workspace", "eastus", "identity")
        if provider == "azure_ml"
        else VertexAISettings("project", "us-central1", "staging", "runner@example.com")
    )

    class FakeManagedTransport(base):
        def __init__(self, selected_settings):
            super().__init__(selected_settings)
            self.objects = {}
            self.jobs = {}
            self.cancelled = []
            self._datastore = SimpleNamespace(container_name="container")
            self._jobs = SimpleNamespace(
                jobs=SimpleNamespace(begin_cancel=lambda call_id: self.cancelled.append(call_id))
            )

        def _put(self, source, key, sha256):
            self.objects[key] = (Path(source).read_bytes(), sha256)
            return True

        def _get(self, key, destination):
            content, expected = self.objects[key]
            assert hashlib.sha256(content).hexdigest() == expected
            Path(destination).write_bytes(content)

        def _delete_prefix(self, prefix):
            removed = [key for key in self.objects if key.startswith(prefix.rstrip("/") + "/")]
            for key in removed:
                del self.objects[key]
            return removed

        def _delete_object(self, key):
            self.objects.pop(key, None)

        def _job_state(self, call_id):
            return SimpleNamespace(
                status="Completed", cancel=lambda: self.cancelled.append(call_id)
            ), "finished"

        def _submit_job(self, payload):
            call_id = f"remote-{len(self.jobs) + 1}"
            self.jobs[call_id] = payload
            if payload.get("operation") == "predict":
                result = json.dumps({"state": "finished", "records": []}).encode()
                key = payload["result_remote_path"]
                self.objects[key] = (result, hashlib.sha256(result).hexdigest())
            return call_id

        def _dashboard_url(self):
            return "https://cloud.example/jobs"

    transport = FakeManagedTransport(settings)
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"verified artifact")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    transport.upload(source, "jobs/train-1/input.bin", digest)
    with pytest.raises(ValueError, match="SHA-256"):
        transport.upload(source, "jobs/train-1/bad.bin", "0" * 64)

    with pytest.raises(ValueError, match="cannot contain credentials"):
        transport.submit(
            {
                "operation": "train",
                "job_key": "train-1",
                "remote_prefix": "jobs/train-1",
                "gpu": "T4",
                "timeout_seconds": 60,
                "access_token": "never persist this",
            }
        )
    call_id = transport.submit(
        {
            "operation": "train",
            "job_key": "train-1",
            "remote_prefix": "jobs/train-1",
            "gpu": "T4",
            "timeout_seconds": 60,
        }
    )
    request_bytes, _ = transport.objects["jobs/train-1/request.json"]
    assert b"access_token" not in request_bytes
    result = json.dumps({"state": "finished"}).encode()
    transport.objects["jobs/train-1/result.json"] = (result, hashlib.sha256(result).hexdigest())
    assert transport.poll(call_id, remote_prefix="jobs/train-1")["state"] == "finished"
    downloaded = tmp_path / "downloaded.bin"
    remote_ref = (
        "azureml-blob://container/jobs/train-1/input.bin"
        if provider == "azure_ml"
        else "gs://staging/jobs/train-1/input.bin"
    )
    transport.download(remote_ref, downloaded)
    assert downloaded.read_bytes() == source.read_bytes()

    transport.cancel(call_id)
    assert transport.cancelled == [call_id]
    callbacks = []
    response = transport.predict_batch(
        {
            "job_key": "prediction-1",
            "batch_id": "batch-1",
            "gpu": "T4",
            "timeout_seconds": 60,
        },
        job_id_callback=callbacks.append,
    )
    assert response["state"] == "finished"
    assert response["remote_job_id"] == callbacks[0] == "remote-2"
    assert transport.cleanup("train-1")["success"] is True
    transport.cleanup_prediction_batch("prediction-1", "batch-1")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"temporary prediction checkpoint")
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    staged_checkpoint = transport.upload_model_checkpoint(
        checkpoint, checkpoint_digest, job_key="prediction-1"
    )
    assert staged_checkpoint == (f"prediction-jobs/prediction-1/checkpoints/{checkpoint_digest}.pt")
    transport.cleanup_prediction_progress("prediction-1")
    assert staged_checkpoint not in transport.objects


def _vertex_sku(description: str, hourly_price: float) -> dict:
    units = int(hourly_price)
    nanos = int(round((hourly_price - units) * 1_000_000_000))
    return {
        "description": description,
        "serviceRegions": ["us-central1"],
        "pricingInfo": [
            {
                "effectiveTime": "2026-10-09T00:00:00Z",
                "pricingExpression": {
                    "usageUnitDescription": "hour",
                    "tieredRates": [
                        {"unitPrice": {"currencyCode": "USD", "units": units, "nanos": nanos}}
                    ],
                },
            }
        ],
    }
