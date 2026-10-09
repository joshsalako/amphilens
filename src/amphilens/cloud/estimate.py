"""Planning estimates for Modal training cost and upload time."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date

from ..core import ValidationError

RATE_CHECKED_AT = "2026-09-29"
MODAL_CPU_RATE_PER_CORE_SECOND = 0.0000131
MODAL_MEMORY_RATE_PER_GIB_SECOND = 0.00000222
CPU_CORES = 8
MEMORY_GIB = 32
MAX_FUNCTION_TIMEOUT_SECONDS = 24 * 60 * 60
MAX_REGION_MULTIPLIER = 1.75
IMAGE_EPOCHS_REFERENCE = 200_000
REFERENCE_RUNTIME_LOW_SECONDS = 90 * 60
REFERENCE_RUNTIME_HIGH_SECONDS = 120 * 60
UPLOAD_FAST_BYTES_PER_SECOND = 50 * 1024 * 1024
UPLOAD_SLOW_BYTES_PER_SECOND = 10 * 1024 * 1024

# USD/second from Modal's published GPU rate table, checked on RATE_CHECKED_AT.
GPU_RATES_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "RTX-PRO-6000": 0.000842,
    "H100": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
    "B300": 0.001972,
}


@dataclass(frozen=True, slots=True)
class CostEstimate:
    gpu: str
    image_count: int
    epochs: int
    low_usd: float
    high_usd: float
    runtime_low_seconds: int
    runtime_high_seconds: int
    time_limit_seconds: int
    upload_time_low_seconds: int
    upload_time_high_seconds: int
    rate_checked_at: str
    rates_stale: bool
    disclaimer: str
    provider: str = "modal"
    region: str = ""
    price_source: str = "Modal GPU price table"

    def to_dict(self) -> dict:
        return asdict(self)


def _cost_rate(gpu: str, rate_per_second: float | None = None) -> float:
    if rate_per_second is not None:
        if not math.isfinite(rate_per_second) or rate_per_second <= 0:
            raise ValidationError("Cloud VM price rate must be a positive finite value")
        return rate_per_second
    try:
        gpu_rate = GPU_RATES_PER_SECOND[gpu]
    except KeyError as exc:
        raise ValidationError(f"Unsupported Modal GPU type: {gpu}") from exc
    return (
        gpu_rate
        + MODAL_CPU_RATE_PER_CORE_SECOND * CPU_CORES
        + MODAL_MEMORY_RATE_PER_GIB_SECOND * MEMORY_GIB
    )


def estimate_training_cost(
    *,
    image_count: int,
    dataset_bytes: int,
    epochs: int,
    gpu: str = "L4",
    max_cost_usd: float = 5.0,
    image_size: int = 640,
    rate_per_second: float | None = None,
    provider: str = "modal",
    region: str = "",
    price_source: str = "Modal GPU price table",
    rate_checked_at: str | None = None,
) -> CostEstimate:
    """Return a rough runtime/cost range and derive a server-side time limit.

    The runtime heuristic is anchored to the unverified 2,000-image, 100-epoch,
    90–120 minute L4 planning example in the feature plan. It is not a benchmark.
    """
    if image_count <= 0 or epochs <= 0 or dataset_bytes < 0 or image_size <= 0:
        raise ValidationError("image_count, epochs, and image_size must be positive")
    if not math.isfinite(max_cost_usd) or max_cost_usd <= 0:
        raise ValidationError("max_cost_usd must be a positive finite value")
    rate = _cost_rate(gpu, rate_per_second)
    work_scale = image_count * epochs / IMAGE_EPOCHS_REFERENCE
    resolution_scale = (image_size / 640) ** 2
    runtime_low = max(60, math.ceil(REFERENCE_RUNTIME_LOW_SECONDS * work_scale * resolution_scale))
    runtime_high = max(
        runtime_low, math.ceil(REFERENCE_RUNTIME_HIGH_SECONDS * work_scale * resolution_scale)
    )
    low_usd = runtime_low * rate
    high_usd = runtime_high * rate * MAX_REGION_MULTIPLIER
    if low_usd > max_cost_usd:
        raise ValidationError(
            "The cheapest plausible run exceeds your cost ceiling; reduce epochs, choose a "
            "smaller dataset, or raise the budget"
        )
    timeout_seconds = min(
        MAX_FUNCTION_TIMEOUT_SECONDS,
        max(1, int(max_cost_usd / (rate * MAX_REGION_MULTIPLIER))),
    )
    upload_low = math.ceil(dataset_bytes / UPLOAD_FAST_BYTES_PER_SECOND)
    upload_high = math.ceil(dataset_bytes / UPLOAD_SLOW_BYTES_PER_SECOND)
    checked_at = rate_checked_at or RATE_CHECKED_AT
    try:
        age_days = (date.today() - date.fromisoformat(checked_at)).days
    except ValueError as exc:
        raise ValidationError("Cloud price source returned an invalid checked date") from exc
    if provider != "modal" and rate_per_second is None:
        raise ValidationError("Managed cloud rates must be resolved before estimating a job")
    return CostEstimate(
        gpu=gpu,
        image_count=image_count,
        epochs=epochs,
        low_usd=round(low_usd, 2),
        high_usd=round(high_usd, 2),
        runtime_low_seconds=runtime_low,
        runtime_high_seconds=runtime_high,
        time_limit_seconds=timeout_seconds,
        upload_time_low_seconds=upload_low,
        upload_time_high_seconds=upload_high,
        rate_checked_at=checked_at,
        rates_stale=age_days > 90,
        disclaimer=(
            "Planning estimate based on an advisory VM rate and unverified training-time "
            "assumptions. Startup, preemption retries, storage, and provider billing can vary; "
            "this is not a guaranteed billing cap."
        ),
        provider=provider,
        region=region,
        price_source=price_source,
    )
