"""Cloud training service adapters and job result helpers."""

from __future__ import annotations

import math
from typing import Any

from .core import ProjectStore
from .webapp_jobs import redact_sensitive_text, safe_error


def _cloud_service(
    store: ProjectStore,
    credentials_store,
    provider_settings_store=None,
    provider: str = "modal",
):
    from .cloud.service import CloudTrainingService

    if provider == "modal":
        from .cloud.modal_transport import ModalTransport

        credentials = credentials_store.resolve()
        return CloudTrainingService(store, ModalTransport(credentials)), credentials
    if provider_settings_store is None:
        raise ValueError("Cloud provider settings are unavailable")
    from .cloud.provider_settings import AzureMLSettings, VertexAISettings
    from .cloud.providers import create_cloud_transport

    values = provider_settings_store.load().get(provider, {})
    settings_type = {"azure_ml": AzureMLSettings, "vertex_ai": VertexAISettings}.get(provider)
    if settings_type is None:
        raise ValueError(f"Unsupported cloud provider: {provider}")
    settings = settings_type.from_mapping(values)
    transport = create_cloud_transport(provider, settings)
    return CloudTrainingService(store, transport), None


def _cloud_job_dict(job) -> dict[str, Any]:
    payload = job.to_dict()
    # Job logs can contain echoed provider output; redact values before the browser sees them.
    if payload.get("error"):
        payload["error"] = safe_error(RuntimeError(payload["error"])).split(": ", 1)[-1]
    if payload.get("log_tail"):
        payload["log_tail"] = redact_sensitive_text(str(payload["log_tail"]))[-8000:]
    for key in ("progress_details",):
        details = payload.get(key)
        if isinstance(details, dict):
            payload[key] = dict(details)
            for text_key in ("phase", "message", "error"):
                if isinstance(payload[key].get(text_key), str):
                    payload[key][text_key] = redact_sensitive_text(payload[key][text_key])[:1000]
            metrics = payload[key].get("metrics")
            if isinstance(metrics, dict):
                payload[key]["metrics"] = {
                    str(metric_key)[:80]: (
                        redact_sensitive_text(metric_value)[:240]
                        if isinstance(metric_value, str)
                        else metric_value
                    )
                    for metric_key, metric_value in list(metrics.items())[:24]
                    if isinstance(metric_value, (str, int, bool))
                    or (isinstance(metric_value, float) and math.isfinite(metric_value))
                }
    events = payload.get("progress_events")
    if isinstance(events, list):
        safe_events = []
        for event in events[-100:]:
            if not isinstance(event, dict):
                continue
            safe_event = dict(event)
            for text_key in ("phase", "message", "error"):
                if isinstance(safe_event.get(text_key), str):
                    safe_event[text_key] = redact_sensitive_text(safe_event[text_key])[:1000]
            metrics = safe_event.get("metrics")
            if isinstance(metrics, dict):
                safe_event["metrics"] = {
                    str(metric_key)[:80]: (
                        redact_sensitive_text(metric_value)[:240]
                        if isinstance(metric_value, str)
                        else metric_value
                    )
                    for metric_key, metric_value in list(metrics.items())[:24]
                    if isinstance(metric_value, (str, int, bool))
                    or (isinstance(metric_value, float) and math.isfinite(metric_value))
                }
            safe_events.append(safe_event)
        payload["progress_events"] = safe_events
    return payload


def _create_cloud_action_impl(
    store: ProjectStore,
    credentials_store,
    run_id: str,
    action: str,
    provider_settings_store=None,
    *,
    cloud_service,
):
    from .cloud.models import CloudJobRecord
    from .core import ValidationError, read_json

    record_path = store.root / "runs" / run_id / "cloud-job.json"
    if not record_path.is_file():
        raise ValidationError(f"Cloud training run not found: {run_id}")
    record = CloudJobRecord.from_dict(read_json(record_path))
    service, credentials = cloud_service(
        store,
        credentials_store,
        provider_settings_store,
        provider=record.provider,
    )
    if record.provider == "modal" and credentials is None:
        raise ValueError(
            "Modal credentials are not configured. Save credentials or set both "
            "Modal environment variables."
        )
    if action == "refresh":
        record = service.refresh(run_id)
    elif action == "cancel":
        record = service.cancel(run_id)
    elif action == "cleanup":
        record = service.retry_cleanup(run_id)
    else:
        raise ValidationError(f"Unsupported cloud job action: {action}")
    return _cloud_job_dict(record)
