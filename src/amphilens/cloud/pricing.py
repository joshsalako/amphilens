"""Advisory hourly VM rates fetched from each provider's public catalog."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from ..core import ValidationError

AZURE_RETAIL_PRICES_URL = "https://prices.azure.com/api/retail/prices"
GCP_COMPUTE_SERVICE = "6F81-5844-456A"

_AZURE_SKUS = {
    "T4": "Standard_NC4as_T4_v3",
    "A10G": "Standard_NV36ads_A10_v5",
    "A100": "Standard_NC24ads_A100_v4",
}
_VERTEX_RESOURCES = {
    "T4": {
        "machine_prefix": "N1 Predefined Instance",
        "cpu": 4,
        "ram_gib": 15,
        "gpu_terms": ("NVIDIA Tesla T4 GPU",),
    },
    "L4": {
        "machine_prefix": "G2 Instance",
        "cpu": 4,
        "ram_gib": 16,
        "gpu_terms": ("NVIDIA L4 GPU",),
    },
    "A100": {
        "machine_prefix": "A2 Instance",
        "cpu": 12,
        "ram_gib": 85,
        "gpu_terms": ("NVIDIA A100 40GB GPU",),
    },
}


@dataclass(frozen=True, slots=True)
class ProviderRate:
    provider: str
    gpu: str
    region: str
    usd_per_hour: float
    price_source: str
    rate_checked_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unknown(provider: str, gpu: str, region: str) -> ValidationError:
    return ValidationError(
        f"A current {provider} GPU VM rate could not be determined for {gpu} in {region}; "
        "cloud submission is blocked until pricing is available"
    )


def _decode_json(response) -> dict[str, Any]:
    raw = response.read()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Pricing response was not a JSON object")
    return value


def azure_vm_rate(
    gpu: str,
    region: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> ProviderRate:
    """Return the full Azure Linux VM hourly retail price or fail closed."""
    sku = _AZURE_SKUS.get(gpu)
    if not sku:
        raise _unknown("Azure ML", gpu, region)
    filter_value = (
        "serviceName eq 'Virtual Machines' and "
        f"armRegionName eq '{region}' and armSkuName eq '{sku}' and priceType eq 'Consumption'"
    )
    url = (
        f"{AZURE_RETAIL_PRICES_URL}?api-version=2023-01-01-preview&"
        f"{urllib.parse.urlencode({'$filter': filter_value, 'currencyCode': 'USD'})}"
    )
    try:
        items = []
        next_url = url
        for _ in range(20):
            with opener(next_url, timeout=15) as response:
                value = _decode_json(response)
            page_items = value.get("Items")
            if not isinstance(page_items, list):
                raise ValueError("Missing Azure price items")
            items.extend(page_items)
            next_url = value.get("NextPageLink")
            if not next_url:
                break
        else:
            raise ValueError("Azure price catalog returned too many pages")
        matches = [
            item
            for item in items
            if isinstance(item, dict)
            and item.get("serviceName") == "Virtual Machines"
            and str(item.get("armSkuName", "")).casefold() == sku.casefold()
            and str(item.get("armRegionName", "")).casefold() == region.casefold()
            and str(item.get("unitOfMeasure", "")).casefold() in {"1 hour", "hour"}
            and not item.get("reservationTerm")
            and float(item.get("retailPrice", 0)) > 0
        ]
        if not matches:
            raise ValueError("Azure GPU VM SKU was not returned")
        # The maximum matching price is conservative when Linux/Windows meters are both listed.
        rate = max(float(item["retailPrice"]) for item in matches)
        return ProviderRate(
            "azure_ml",
            gpu,
            region,
            rate,
            "Azure Retail Prices API",
            date.today().isoformat(),
        )
    except Exception as exc:
        if isinstance(exc, ValidationError):
            raise
        raise _unknown("Azure ML", gpu, region) from exc


def _money(rate: dict[str, Any]) -> float:
    price = rate.get("unitPrice", {})
    if not isinstance(price, dict) or price.get("currencyCode", "USD") != "USD":
        raise ValueError("SKU price is not denominated in USD")
    return float(price.get("units", 0)) + float(price.get("nanos", 0)) / 1_000_000_000


def _hourly_price(sku: dict[str, Any]) -> float:
    info = sku.get("pricingInfo")
    if not isinstance(info, list) or not info:
        raise ValueError("SKU has no active price")
    expression = info[-1].get("pricingExpression", {})
    if not isinstance(expression, dict):
        raise ValueError("SKU pricing expression is invalid")
    unit = str(expression.get("usageUnitDescription", "")).casefold()
    if "hour" not in unit:
        raise ValueError("GPU VM SKU does not have an hourly price")
    tiers = expression.get("tieredRates")
    if not isinstance(tiers, list) or not tiers:
        raise ValueError("SKU has no rate tiers")
    return _money(tiers[0])


def _select_sku(
    skus: list[dict[str, Any]], region: str, description_terms: tuple[str, ...]
) -> dict[str, Any]:
    candidates = []
    for sku in skus:
        if region not in sku.get("serviceRegions", []):
            continue
        description = str(sku.get("description", "")).casefold()
        if all(term.casefold() in description for term in description_terms):
            candidates.append(sku)
    if not candidates:
        raise ValueError("No regional matching SKU")
    return max(candidates, key=_hourly_price)


def vertex_vm_rate(
    gpu: str,
    region: str,
    *,
    session=None,
    credentials=None,
) -> ProviderRate:
    """Resolve GPU plus machine CPU/RAM rates from the Cloud Billing SKU catalog."""
    requirements = _VERTEX_RESOURCES.get(gpu)
    if requirements is None:
        raise _unknown("Vertex AI", gpu, region)
    try:
        if session is None:
            try:
                import google.auth
                from google.auth.transport.requests import AuthorizedSession
            except ImportError as exc:
                raise RuntimeError("Google authentication libraries are unavailable") from exc
            if credentials is None:
                credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-billing.readonly"]
                )
            session = AuthorizedSession(credentials)
        skus = []
        page_token = None
        while True:
            query = {"pageSize": 5000, "currencyCode": "USD"}
            if page_token:
                query["pageToken"] = page_token
            url = (
                f"https://cloudbilling.googleapis.com/v1/services/{GCP_COMPUTE_SERVICE}/skus?"
                f"{urllib.parse.urlencode(query)}"
            )
            response = session.get(url, timeout=20)
            response.raise_for_status()
            value = response.json()
            skus.extend(value.get("skus", []))
            page_token = value.get("nextPageToken")
            if not page_token:
                break
        machine = requirements["machine_prefix"]
        gpu_sku = _select_sku(skus, region, requirements["gpu_terms"])
        cpu_sku = _select_sku(skus, region, (machine, "Core"))
        ram_sku = _select_sku(skus, region, (machine, "Ram"))
        total = (
            _hourly_price(gpu_sku)
            + requirements["cpu"] * _hourly_price(cpu_sku)
            + requirements["ram_gib"] * _hourly_price(ram_sku)
        )
        if total <= 0:
            raise ValueError("Computed Vertex VM rate is not positive")
        effective = str(gpu_sku.get("pricingInfo", [{}])[-1].get("effectiveTime", ""))
        checked = effective[:10] if len(effective) >= 10 else date.today().isoformat()
        return ProviderRate(
            "vertex_ai",
            gpu,
            region,
            total,
            "Google Cloud Billing SKU catalog (GPU, vCPU, and RAM)",
            checked,
        )
    except Exception as exc:
        if isinstance(exc, ValidationError):
            raise
        raise _unknown("Vertex AI", gpu, region) from exc
