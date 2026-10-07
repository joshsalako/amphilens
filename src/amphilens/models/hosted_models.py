"""Private Hugging Face checkpoints distributed with AmphiLens."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import DetectionRecord
from ..preprocessing import PreprocessingConfig

# Pin weights independently of the mutable Hub branch. Updated after the private upload.
HF_MODEL_REVISION = "0fbcf6b047e0ef88d3f6eb3933f061d316d4fe9b"
_MANIFEST_PATH = Path(__file__).with_name("amphilens-models.json")


class HuggingFaceAccessError(ValueError):
    """A private Hub model could not be read with the current local login."""


@dataclass(frozen=True, slots=True)
class HostedModel:
    model_id: str
    name: str
    repo_id: str
    architecture: str
    size: str
    artifact: str
    sha256: str
    source_classes: tuple[str, ...]
    training_provenance: dict[str, Any]
    inference_image_size: int
    preprocessing: PreprocessingConfig
    license: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "repo_id": self.repo_id,
            "revision": HF_MODEL_REVISION,
            "artifact": self.artifact,
            "sha256": self.sha256,
            "architecture": self.architecture,
            "size": self.size,
            "source_class_order": list(self.source_classes),
            "inference_image_size": self.inference_image_size,
            "preprocessing": self.preprocessing.to_dict(),
            "training_provenance": dict(self.training_provenance),
            "license": self.license,
        }


def _read_manifest() -> dict[str, Any]:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def list_hosted_models() -> list[HostedModel]:
    manifest = _read_manifest()
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported AmphiLens model manifest version")
    repo_id = str(manifest["repo_id"])
    models = []
    for value in manifest["models"]:
        classes = tuple(str(item) for item in value["source_class_order"])
        if len(classes) != len(set(classes)) or not classes:
            raise ValueError(f"Invalid source class order for {value['model_id']}")
        digest = str(value["sha256"])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Invalid SHA-256 for {value['model_id']}")
        models.append(
            HostedModel(
                model_id=str(value["model_id"]),
                name=str(value["name"]),
                repo_id=repo_id,
                architecture=str(value["architecture"]),
                size=str(value["size"]),
                artifact=str(value["artifact"]),
                sha256=digest,
                source_classes=classes,
                training_provenance=dict(value["training_provenance"]),
                inference_image_size=int(value["inference_image_size"]),
                preprocessing=PreprocessingConfig.from_dict(value["preprocessing"]),
                license=str(value["license"]),
            )
        )
    return models


def get_hosted_model(model_id: str) -> HostedModel:
    for model in list_hosted_models():
        if model.model_id == model_id:
            return model
    raise ValueError(f"Unknown AmphiLens pretrained model: {model_id}")


def resolve_class_mapping(
    source_classes: tuple[str, ...] | list[str],
    project_classes: list[str],
    requested: dict[str, str | None] | None,
) -> dict[str, str | None]:
    requested = requested or {}
    source_names = set(source_classes)
    unknown = sorted(set(requested) - source_names)
    if unknown:
        raise ValueError(f"Unknown source class mapping: {', '.join(unknown)}")
    mapping: dict[str, str | None] = {}
    missing = []
    for source in source_classes:
        if source in requested:
            target = requested[source]
        elif source in project_classes:
            target = source
        else:
            missing.append(source)
            continue
        if target is not None and target not in project_classes:
            raise ValueError(f"Unknown project class for {source}: {target}")
        mapping[source] = target
    if missing:
        raise ValueError(
            "Map each source class to a project class or Ignore: " + ", ".join(missing)
        )
    return mapping


def map_hosted_detections(
    records: list[DetectionRecord],
    mapping: dict[str, str | None],
    project_classes: list[str],
) -> list[DetectionRecord]:
    from dataclasses import replace

    target_ids = {name: index for index, name in enumerate(project_classes)}
    output = []
    for record in records:
        target = mapping.get(record.class_name)
        if target is None:
            continue
        if target not in target_ids:
            raise ValueError(f"Unknown project class mapping: {target}")
        output.append(replace(record, class_id=target_ids[target], class_name=target))
    return output


class HostedClassMappedDetector:
    """Translate a hosted model's source labels into a project's class IDs."""

    def __init__(self, detector, mapping: dict[str, str | None], project_classes: list[str]):
        self.detector = detector
        self.mapping = dict(mapping)
        self.project_classes = list(project_classes)
        self.model_id = detector.model_id

    def predict(self, image_paths, config):
        for record in self.detector.predict(image_paths, config):
            yield from map_hosted_detections([record], self.mapping, self.project_classes)


def _hub_runtime():
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import get_token
    except ImportError as exc:
        raise RuntimeError(
            "AmphiLens pretrained models need the inference extra. Install with "
            "`pip install 'amphilens[inference]'`."
        ) from exc
    return hf_hub_download, get_token


def _progress_tqdm_class(callback: Callable[[dict[str, Any]], None] | None, model_name: str):
    if callback is None:
        return None

    from tqdm.auto import tqdm

    class ReportingTqdm(tqdm):
        def update(self, n: int = 1):
            result = super().update(n)
            total = self.total
            callback(
                {
                    "message": f"Downloading {model_name}",
                    "completed": self.n,
                    "total": total,
                    "progress": self.n / total if total else None,
                }
            )
            return result

    return ReportingTqdm


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_hosted_checkpoint(
    model_id: str,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    model = get_hosted_model(model_id)
    hf_hub_download, get_token = _hub_runtime()
    authenticated = bool(get_token())
    if progress_callback is not None:
        progress_callback({"message": f"Checking access to {model.name}…", "progress": None})
    try:
        path = Path(
            hf_hub_download(
                repo_id=model.repo_id,
                filename=model.artifact,
                repo_type="model",
                revision=HF_MODEL_REVISION,
                token=True if authenticated else False,
                tqdm_class=_progress_tqdm_class(progress_callback, model.name),
            )
        )
    except Exception as exc:
        try:
            from huggingface_hub.errors import HfHubHTTPError
        except ImportError:
            HfHubHTTPError = ()

        if HfHubHTTPError and isinstance(exc, HfHubHTTPError):
            raise HuggingFaceAccessError(
                f"Could not read {model.repo_id}. If the repository is private, run "
                "`hf auth login` and confirm that your Hugging Face account has read "
                "access to the AmphiLens model repository."
            ) from exc
        raise
    if not path.is_file():
        raise FileNotFoundError(f"Hugging Face did not return a checkpoint file for {model.name}")
    if _sha256_file(path) != model.sha256:
        raise ValueError(
            f"The cached {model.name} checkpoint failed its SHA-256 integrity check. "
            "Clear that model from the Hugging Face cache and download it again."
        )
    if progress_callback is not None:
        progress_callback({"message": f"Verified {model.name} checkpoint", "progress": 1.0})
    # Keep the snapshot alias: resolving its symlink can erase the `.pt` suffix
    # that model loaders use to identify PyTorch checkpoints.
    return path
