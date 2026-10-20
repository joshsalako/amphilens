"""Request models and validation for the local browser API."""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from starlette.requests import Request

from .core import ProjectStore


@dataclass(slots=True)
class ProjectOpenRequest:
    path: str


@dataclass(slots=True)
class ProjectCreateRequest:
    name: str
    path: str
    classes: list[str]
    image_root: str | None = None
    model_preset: str = "yolo26-l"
    image_dimension: int = 640
    grayscale: bool = True
    clahe: bool = False


@dataclass(slots=True)
class PredictionsRequest:
    image_root: str | None = None
    checkpoint: str | None = None
    output_dir: str | None = None
    confidence: float | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] | None = None
    execution: Literal["local", "modal", "azure_ml", "vertex_ai"] = "local"
    batch_size: int | None = None
    gpu: str = "L4"
    max_cost_usd: float = 5.0
    acknowledged: bool = False
    uploads_dataset: bool = False
    expected_image_count: int | None = None
    expected_total_bytes: int | None = None
    expected_rate_usd_per_hour: float | None = None
    model_preset: str | None = None
    hosted_model_id: str | None = None
    preprocessing: dict[str, Any] | None = None
    class_mapping: dict[str, str | None] | None = None
    run_name: str = ""


@dataclass(slots=True)
class TimestampOCRRequest:
    acknowledged: bool = False


@dataclass(slots=True)
class DatasetImportRequest:
    archive: str
    class_mapping: dict[str, str] | None = None
    display_name: str | None = None


@dataclass(slots=True)
class CvatProjectListRequest:
    server_url: str | None = None


@dataclass(slots=True)
class CvatProjectDetailsRequest:
    project_id: str
    server_url: str | None = None


@dataclass(slots=True)
class CvatProjectImportRequest:
    project_id: str
    class_mapping: dict[str, str] | None = None
    server_url: str | None = None
    display_name: str | None = None


@dataclass(slots=True)
class DatasetRenameRequest:
    snapshot_path: str
    display_name: str


@dataclass(slots=True)
class DatasetDeleteRequest:
    snapshot_path: str


@dataclass(slots=True)
class TrainingRequest:
    snapshot_path: str
    checkpoint: str | None = None
    output_dir: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] | None = None
    training_source: str | None = None
    model_preset: str | None = None
    hosted_model_id: str | None = None
    preprocessing: dict[str, Any] | None = None
    data_mode: Literal["auto-split", "training-monitor", "separate-snapshots"] = (
        "separate-snapshots"
    )
    validation_snapshot_path: str | None = None
    test_snapshot_path: str | None = None
    patience: int | None = None
    allow_image_level_split: bool = False


@dataclass(slots=True)
class ActiveLearningRequest:
    predictions: str
    calibration: str
    features: str
    output: str
    budget: int = 100
    seed: int = 42
    pool_multiplier: int = 200


@dataclass(slots=True)
class CvatCycleRequest:
    action: Literal["start", "refresh", "continue"]
    cycle: int
    queue: str | None = None
    server_url: str | None = None


@dataclass(slots=True)
class CloudCredentialsRequest:
    token_id: str
    token_secret: str


@dataclass(slots=True)
class CloudEstimateRequest:
    snapshot_path: str
    provider: Literal["modal", "azure_ml", "vertex_ai"] = "modal"
    gpu: str = "L4"
    epochs: int | None = None
    max_cost_usd: float = 5.0
    image_size: int = 640
    training_source: str | None = None
    model_preset: str | None = None
    checkpoint: str | None = None
    hosted_model_id: str | None = None
    preprocessing: dict[str, Any] | None = None
    data_mode: Literal["auto-split", "training-monitor", "separate-snapshots"] = (
        "separate-snapshots"
    )
    validation_snapshot_path: str | None = None
    test_snapshot_path: str | None = None
    patience: int | None = None
    allow_image_level_split: bool = False


@dataclass(slots=True)
class CloudTrainingRequest(CloudEstimateRequest):
    batch_size: int | None = None
    device: Literal["auto", "cpu", "cuda"] | None = None
    estimated_usd: float = 0.0
    acknowledged: bool = False
    uploads_dataset: bool = False


@dataclass(slots=True)
class CloudJobActionRequest:
    action: Literal["refresh", "cancel", "collect", "cleanup"]


def _class_names(values: list[str]) -> list[str]:
    classes = [str(value).strip() for value in values]
    if not classes or any(not value for value in classes):
        raise ValueError("At least one non-empty class name is required")
    if len(classes) != len(set(classes)):
        raise ValueError("Class names must be unique")
    return classes


async def _read_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise ValueError("Request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return payload


def _string(data: dict[str, Any], key: str, *, required: bool = True, default: str = "") -> str:
    value = data.get(key, default)
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    if required and not value.strip():
        raise ValueError(f"{key} is required")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    return _string(data, key)


def _prediction_run_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("run_name must be text")
    name = value.strip()
    if not name:
        raise ValueError("run_name is required")
    if len(name) > 120:
        raise ValueError("run_name must be 120 characters or fewer")
    invalid_characters = '<>:/\\|?*"'
    reserved_windows_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
    if (
        name in {".", ".."}
        or name.endswith((".", " "))
        or any(character in invalid_characters or ord(character) < 32 for character in name)
        or name.split(".", 1)[0].upper() in reserved_windows_names
    ):
        raise ValueError("run_name must be a valid folder name")
    return name


def _prediction_run_name_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _ensure_prediction_run_name_is_unique(store: ProjectStore, run_name: str) -> None:
    """Reject prediction names already used in this project's run history."""
    key = _prediction_run_name_key(_prediction_run_name(run_name))
    runs_root = store.root / "runs"
    if not runs_root.is_dir():
        return
    try:
        run_directories = sorted(path for path in runs_root.iterdir() if path.is_dir())
    except OSError as exc:
        raise ValueError("Could not check whether this prediction run name is unique") from exc

    for run_directory in run_directories:
        if _prediction_run_name_key(run_directory.name) == key:
            raise ValueError(
                f"Prediction run name '{run_name}' has already been used in this project. "
                "Choose a unique name."
            )
        manifest_path = run_directory / "run.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or manifest.get("kind") != "inference":
            continue
        config = manifest.get("config")
        metadata = config.get("metadata") if isinstance(config, dict) else None
        existing_name = metadata.get("run_name") if isinstance(metadata, dict) else None
        if isinstance(existing_name, str) and _prediction_run_name_key(existing_name) == key:
            raise ValueError(
                f"Prediction run name '{run_name}' has already been used in this project. "
                "Choose a unique name."
            )


def _integer(
    data: dict[str, Any], key: str, default: int | None = None, *, minimum: int | None = None
) -> int | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


def _number(
    data: dict[str, Any], key: str, default: float | None = None, *, minimum: float | None = None
) -> float | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    selected = float(value)
    if minimum is not None and selected < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return selected


def _boolean(data: dict[str, Any], key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def _mapping(data: dict[str, Any], key: str) -> dict[str, str | None] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict) or any(
        not isinstance(source, str) or (target is not None and not isinstance(target, str))
        for source, target in value.items()
    ):
        raise ValueError(f"{key} must map source labels to project labels or null for Ignore")
    return value


def _preprocessing_override(data: dict[str, Any]) -> dict[str, Any] | None:
    value = data.get("preprocessing")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("preprocessing must be an object")
    from .preprocessing import PreprocessingConfig

    if set(value) == {"name"}:
        PreprocessingConfig.from_any(value)
        return dict(value)
    fallback = PreprocessingConfig(clahe_enabled=True).to_dict(include_fingerprint=False)
    try:
        PreprocessingConfig.from_any({**fallback, **value})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid preprocessing settings: {exc}") from exc
    return dict(value)


def _cloud_gpus(provider: str) -> set[str]:
    if provider == "modal":
        from .cloud.estimate import GPU_RATES_PER_SECOND

        return set(GPU_RATES_PER_SECOND)
    return {
        "azure_ml": {"T4", "A10G", "A100"},
        "vertex_ai": {"T4", "L4", "A100"},
    }.get(provider, set())


def _request(cls, payload: dict[str, Any]):
    """Parse the small local API contract without adding a schema dependency."""
    if cls is ProjectOpenRequest:
        return cls(path=_string(payload, "path"))
    if cls is ProjectCreateRequest:
        classes = payload.get("classes")
        if not isinstance(classes, list) or any(not isinstance(value, str) for value in classes):
            raise ValueError("classes must be a list of names")
        dimension_key = (
            "short_side_dimension" if "short_side_dimension" in payload else "max_dimension"
        )
        request = cls(
            name=_string(payload, "name"),
            path=_string(payload, "path"),
            classes=classes,
            image_root=_optional_string(payload, "image_root"),
            model_preset=_string(payload, "model_preset", required=False, default="yolo26-l"),
            image_dimension=_integer(payload, dimension_key, 640, minimum=32),
            grayscale=_boolean(payload, "grayscale", True),
            clahe=_boolean(payload, "clahe", False),
        )
        return request
    if cls is PredictionsRequest:
        run_name = _prediction_run_name(_string(payload, "run_name"))
        confidence = _number(payload, "confidence")
        if confidence is not None and confidence > 1:
            raise ValueError("confidence must be between 0 and 1")
        device = _optional_string(payload, "device")
        if device is not None and device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        execution = _string(payload, "execution", required=False, default="local")
        if execution not in {"local", "modal", "azure_ml", "vertex_ai"}:
            raise ValueError("execution must be local, modal, azure_ml, or vertex_ai")
        raw_batch_size = payload.get("batch_size", "auto")
        batch_size = None
        if raw_batch_size != "auto":
            batch_size = _integer(payload, "batch_size", minimum=1)
            if batch_size is not None and batch_size > 32:
                raise ValueError("batch_size must be between 1 and 32")
        gpu = _string(payload, "gpu", required=False, default="L4")
        if execution != "local" and gpu not in _cloud_gpus(execution):
            if execution == "modal":
                raise ValueError(f"Unsupported Modal GPU: {gpu}")
            raise ValueError(f"Unsupported GPU for {execution}: {gpu}")
        max_cost_usd = _number(payload, "max_cost_usd", 5.0, minimum=0.01)
        if execution != "local" and "max_cost_usd" not in payload:
            raise ValueError("Set a spending limit before choosing cloud prediction")
        if max_cost_usd is None or not math.isfinite(max_cost_usd):
            raise ValueError("max_cost_usd must be a positive finite number")
        return cls(
            run_name=run_name,
            image_root=_optional_string(payload, "image_root"),
            checkpoint=_optional_string(payload, "checkpoint"),
            output_dir=_optional_string(payload, "output_dir"),
            confidence=confidence,
            device=device,
            execution=execution,
            batch_size=batch_size,
            gpu=gpu,
            max_cost_usd=max_cost_usd,
            acknowledged=_boolean(payload, "acknowledged", False),
            uploads_dataset=_boolean(payload, "uploads_dataset", False),
            expected_image_count=_integer(payload, "expected_image_count", minimum=1),
            expected_total_bytes=_integer(payload, "expected_total_bytes", minimum=0),
            expected_rate_usd_per_hour=_number(
                payload, "expected_rate_usd_per_hour", minimum=0.000001
            ),
            model_preset=_optional_string(payload, "model_preset"),
            hosted_model_id=_optional_string(payload, "hosted_model_id"),
            preprocessing=_preprocessing_override(payload),
            class_mapping=_mapping(payload, "class_mapping"),
        )
    if cls is TimestampOCRRequest:
        return cls(acknowledged=_boolean(payload, "acknowledged", False))
    if cls is DatasetImportRequest:
        return cls(
            archive=_string(payload, "archive"),
            class_mapping=_mapping(payload, "class_mapping"),
            display_name=_optional_string(payload, "display_name"),
        )
    if cls is CvatProjectListRequest:
        return cls(server_url=_optional_string(payload, "server_url"))
    if cls is CvatProjectDetailsRequest:
        return cls(
            project_id=_string(payload, "project_id"),
            server_url=_optional_string(payload, "server_url"),
        )
    if cls is CvatProjectImportRequest:
        return cls(
            project_id=_string(payload, "project_id"),
            class_mapping=_mapping(payload, "class_mapping"),
            server_url=_optional_string(payload, "server_url"),
            display_name=_optional_string(payload, "display_name"),
        )
    if cls is DatasetRenameRequest:
        return cls(
            snapshot_path=_string(payload, "snapshot_path"),
            display_name=_string(payload, "display_name"),
        )
    if cls is DatasetDeleteRequest:
        return cls(snapshot_path=_string(payload, "snapshot_path"))
    if cls is TrainingRequest:
        epochs = _integer(payload, "epochs", minimum=1)
        batch_size = _integer(payload, "batch_size", minimum=1)
        patience = _integer(payload, "patience", 25, minimum=0)
        device = _optional_string(payload, "device")
        if device is not None and device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        data_mode = _string(payload, "data_mode", required=False, default="separate-snapshots")
        if data_mode not in {"auto-split", "training-monitor", "separate-snapshots"}:
            raise ValueError(
                "data_mode must be auto-split, training-monitor, or separate-snapshots"
            )
        return cls(
            snapshot_path=_string(payload, "snapshot_path"),
            checkpoint=_optional_string(payload, "checkpoint"),
            output_dir=_optional_string(payload, "output_dir"),
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            training_source=_optional_string(payload, "training_source"),
            model_preset=_optional_string(payload, "model_preset"),
            hosted_model_id=_optional_string(payload, "hosted_model_id"),
            preprocessing=_preprocessing_override(payload),
            data_mode=data_mode,
            validation_snapshot_path=_optional_string(payload, "validation_snapshot_path"),
            test_snapshot_path=_optional_string(payload, "test_snapshot_path"),
            patience=patience,
            allow_image_level_split=_boolean(payload, "allow_image_level_split", False),
        )
    if cls is ActiveLearningRequest:
        return cls(
            predictions=_string(payload, "predictions"),
            calibration=_string(payload, "calibration"),
            features=_string(payload, "features"),
            output=_string(payload, "output"),
            budget=_integer(payload, "budget", 100, minimum=1),
            seed=_integer(payload, "seed", 42, minimum=0),
            pool_multiplier=_integer(payload, "pool_multiplier", 200, minimum=1),
        )
    if cls is CvatCycleRequest:
        action = _string(payload, "action")
        if action not in {"start", "refresh", "continue"}:
            raise ValueError("action must be start, refresh, or continue")
        cycle = _integer(payload, "cycle", minimum=0)
        if cycle is None:
            raise ValueError("cycle is required")
        return cls(
            action=action,
            cycle=cycle,
            queue=_optional_string(payload, "queue"),
            server_url=_optional_string(payload, "server_url"),
        )
    if cls is CloudCredentialsRequest:
        return cls(
            token_id=_string(payload, "token_id"), token_secret=_string(payload, "token_secret")
        )
    if cls is CloudEstimateRequest:
        provider = _string(payload, "provider", required=False, default="modal")
        if provider not in {"modal", "azure_ml", "vertex_ai"}:
            raise ValueError("provider must be modal, azure_ml, or vertex_ai")
        gpu = _string(payload, "gpu", required=False, default="L4")
        if gpu not in _cloud_gpus(provider):
            raise ValueError(f"Unsupported GPU for {provider}: {gpu}")
        data_mode = _string(payload, "data_mode", required=False, default="separate-snapshots")
        if data_mode not in {"auto-split", "training-monitor", "separate-snapshots"}:
            raise ValueError(
                "data_mode must be auto-split, training-monitor, or separate-snapshots"
            )
        return cls(
            snapshot_path=_string(payload, "snapshot_path"),
            provider=provider,
            gpu=gpu,
            epochs=_integer(payload, "epochs", minimum=1),
            max_cost_usd=_number(payload, "max_cost_usd", 5.0, minimum=0.001),
            image_size=_integer(payload, "image_size", 640, minimum=1),
            training_source=_optional_string(payload, "training_source"),
            model_preset=_optional_string(payload, "model_preset"),
            checkpoint=_optional_string(payload, "checkpoint"),
            hosted_model_id=_optional_string(payload, "hosted_model_id"),
            preprocessing=_preprocessing_override(payload),
            data_mode=data_mode,
            validation_snapshot_path=_optional_string(payload, "validation_snapshot_path"),
            test_snapshot_path=_optional_string(payload, "test_snapshot_path"),
            patience=_integer(payload, "patience", 25, minimum=0),
            allow_image_level_split=_boolean(payload, "allow_image_level_split", False),
        )
    if cls is CloudTrainingRequest:
        base = _request(CloudEstimateRequest, payload)
        batch_size = _integer(payload, "batch_size", minimum=1)
        device = _optional_string(payload, "device")
        if device is not None and device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        estimated_usd = _number(payload, "estimated_usd", minimum=0)
        if estimated_usd is None:
            raise ValueError("estimated_usd is required")
        return cls(
            snapshot_path=base.snapshot_path,
            provider=base.provider,
            gpu=base.gpu,
            epochs=base.epochs,
            max_cost_usd=base.max_cost_usd,
            image_size=base.image_size,
            training_source=base.training_source,
            model_preset=base.model_preset,
            checkpoint=base.checkpoint,
            hosted_model_id=base.hosted_model_id,
            preprocessing=base.preprocessing,
            data_mode=base.data_mode,
            validation_snapshot_path=base.validation_snapshot_path,
            test_snapshot_path=base.test_snapshot_path,
            patience=base.patience,
            allow_image_level_split=base.allow_image_level_split,
            batch_size=batch_size,
            device=device,
            estimated_usd=estimated_usd,
            acknowledged=_boolean(payload, "acknowledged", False),
            uploads_dataset=_boolean(payload, "uploads_dataset", False),
        )
    raise TypeError(f"Unsupported request model: {cls.__name__}")
