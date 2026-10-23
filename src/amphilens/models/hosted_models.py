"""Public Hugging Face checkpoints distributed with AmphiLens."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..core import DetectionRecord
from ..preprocessing import PreprocessingConfig

# Pin weights independently of the mutable Hub branch. Update after model uploads.
HF_MODEL_REVISION = "99a18b1d6ba5e91fca86b5e131b44223c60369ae"
_MANIFEST_PATH = Path(__file__).with_name("amphilens-models.json")
_NCNN_ADAPTERS = {"yolo": "ultralytics-yolo-raw-detect-v1"}


class HuggingFaceAccessError(ValueError):
    """A public Hub checkpoint could not be downloaded from Hugging Face."""


@dataclass(frozen=True, slots=True)
class NCNNArtifactFile:
    filename: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"filename": self.filename, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class NCNNArtifactManifest:
    status: str
    adapter: str
    precision: str
    input_width: int
    input_height: int
    input_name: str
    output_name: str
    files: tuple[NCNNArtifactFile, ...]

    def is_compatible(self, architecture: str) -> bool:
        return self.status == "verified" and _NCNN_ADAPTERS.get(architecture) == self.adapter

    def to_dict(self, *, available: bool) -> dict[str, Any]:
        return {
            "status": self.status,
            "adapter": self.adapter,
            "precision": self.precision,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "files": [item.to_dict() for item in self.files],
            "available": available,
        }


@dataclass(frozen=True, slots=True)
class DownloadedNCNNArtifact:
    param_path: Path
    bin_path: Path
    revision: str
    manifest: NCNNArtifactManifest


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
    ncnn: NCNNArtifactManifest | None = None

    @property
    def ncnn_available(self) -> bool:
        return self.ncnn is not None and self.ncnn.is_compatible(self.architecture)

    @property
    def ncnn_unavailable_reason(self) -> str:
        if self.ncnn is None:
            return "No NCNN package has been published for this model."
        if self.ncnn.status != "verified":
            return "An NCNN package is listed, but its inference adapter has not been verified."
        if not self.ncnn.is_compatible(self.architecture):
            return f"NCNN inference is not verified for the {self.architecture} model family."
        return ""

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
            "ncnn_artifact": self.ncnn.to_dict(available=self.ncnn_available)
            if self.ncnn is not None
            else None,
            "ncnn_available": self.ncnn_available,
            "ncnn_unavailable_reason": self.ncnn_unavailable_reason,
        }


def _read_manifest() -> dict[str, Any]:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def _manifest_sha256(value: Any, context: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"Invalid SHA-256 for {context}")
    return digest


def _ncnn_manifest(value: Any, model_id: str) -> NCNNArtifactManifest | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Invalid NCNN artifact metadata for {model_id}")
    status = str(value.get("status", ""))
    if status not in {"uploaded", "verified"}:
        raise ValueError(f"Invalid NCNN artifact status for {model_id}")
    adapter = str(value.get("adapter", ""))
    if not adapter:
        raise ValueError(f"NCNN output adapter is required for {model_id}")
    precision = str(value.get("precision", ""))
    if precision not in {"fp32", "fp16", "int8"}:
        raise ValueError(f"Invalid NCNN precision for {model_id}")
    input_width = value.get("input_width")
    input_height = value.get("input_height")
    if (
        isinstance(input_width, bool)
        or not isinstance(input_width, int)
        or input_width < 32
        or input_width > 8192
        or isinstance(input_height, bool)
        or not isinstance(input_height, int)
        or input_height < 32
        or input_height > 8192
    ):
        raise ValueError(f"Invalid NCNN input dimensions for {model_id}")
    input_name = str(value.get("input_name", ""))
    output_name = str(value.get("output_name", ""))
    if not input_name or not output_name:
        raise ValueError(f"NCNN input and output names are required for {model_id}")
    raw_files = value.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != 2:
        raise ValueError(f"NCNN artifacts must contain one .param and one .bin file for {model_id}")
    files = []
    for raw_file in raw_files:
        if not isinstance(raw_file, dict):
            raise ValueError(f"Invalid NCNN file metadata for {model_id}")
        filename = str(raw_file.get("filename", ""))
        path = PurePosixPath(filename)
        if (
            not filename
            or path.is_absolute()
            or "\\" in filename
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.suffix not in {".param", ".bin"}
        ):
            raise ValueError(f"Unsafe NCNN artifact filename for {model_id}")
        files.append(NCNNArtifactFile(filename, _manifest_sha256(raw_file.get("sha256"), filename)))
    if {PurePosixPath(item.filename).suffix for item in files} != {".param", ".bin"}:
        raise ValueError(f"NCNN artifacts must contain one .param and one .bin file for {model_id}")
    if PurePosixPath(files[0].filename).parent != PurePosixPath(files[1].filename).parent:
        raise ValueError(f"NCNN .param and .bin files must share a directory for {model_id}")
    return NCNNArtifactManifest(
        status=status,
        adapter=adapter,
        precision=precision,
        input_width=input_width,
        input_height=input_height,
        input_name=input_name,
        output_name=output_name,
        files=tuple(files),
    )


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
        digest = _manifest_sha256(value["sha256"], str(value["model_id"]))
        architecture = str(value["architecture"])
        models.append(
            HostedModel(
                model_id=str(value["model_id"]),
                name=str(value["name"]),
                repo_id=repo_id,
                architecture=architecture,
                size=str(value["size"]),
                artifact=str(value["artifact"]),
                sha256=digest,
                source_classes=classes,
                training_provenance=dict(value["training_provenance"]),
                inference_image_size=int(value["inference_image_size"]),
                preprocessing=PreprocessingConfig.from_dict(value["preprocessing"]),
                license=str(value["license"]),
                ncnn=_ncnn_manifest(value.get("ncnn"), str(value["model_id"])),
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


def _supports_tqdm_class(hf_hub_download) -> bool:
    """Check whether the installed Hub client supports its optional progress hook."""
    try:
        parameters = inspect.signature(hf_hub_download).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "tqdm_class" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


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
        download_options = {
            "repo_id": model.repo_id,
            "filename": model.artifact,
            "repo_type": "model",
            "revision": HF_MODEL_REVISION,
            "token": True if authenticated else False,
        }
        if _supports_tqdm_class(hf_hub_download):
            download_options["tqdm_class"] = _progress_tqdm_class(progress_callback, model.name)
        path = Path(hf_hub_download(**download_options))
    except Exception as exc:
        try:
            from huggingface_hub.errors import HfHubHTTPError
        except ImportError:
            HfHubHTTPError = ()

        if HfHubHTTPError and isinstance(exc, HfHubHTTPError):
            raise HuggingFaceAccessError(
                f"Could not download the public Hugging Face model {model.repo_id}. "
                "Check the network connection and confirm the repository is available."
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


def download_hosted_ncnn_artifact(
    model_id: str,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> DownloadedNCNNArtifact:
    """Download a pinned NCNN pair and verify each file before inference."""
    model = get_hosted_model(model_id)
    if not model.ncnn_available or model.ncnn is None:
        raise ValueError(model.ncnn_unavailable_reason or "No verified NCNN artifact is available.")
    hf_hub_download, get_token = _hub_runtime()
    authenticated = bool(get_token())
    if progress_callback is not None:
        progress_callback(
            {"message": f"Checking access to {model.name} NCNN model…", "progress": None}
        )
    downloaded: dict[str, Path] = {}
    for file in model.ncnn.files:
        try:
            options = {
                "repo_id": model.repo_id,
                "filename": file.filename,
                "repo_type": "model",
                "revision": HF_MODEL_REVISION,
                "token": True if authenticated else False,
            }
            if _supports_tqdm_class(hf_hub_download):
                options["tqdm_class"] = _progress_tqdm_class(progress_callback, model.name)
            path = Path(hf_hub_download(**options))
        except Exception as exc:
            try:
                from huggingface_hub.errors import HfHubHTTPError
            except ImportError:
                HfHubHTTPError = ()
            if HfHubHTTPError and isinstance(exc, HfHubHTTPError):
                raise HuggingFaceAccessError(
                    f"Could not download the NCNN files for {model.name} from {model.repo_id}. "
                    "Check the network connection and confirm the files exist at the "
                    "pinned revision."
                ) from exc
            raise
        if not path.is_file():
            raise FileNotFoundError(f"Hugging Face did not return {file.filename} for {model.name}")
        if _sha256_file(path) != file.sha256:
            raise ValueError(
                f"The {file.filename} file for {model.name} failed its SHA-256 integrity check. "
                "Clear the model from the Hugging Face cache and download it again."
            )
        downloaded[file.filename] = path
    param_file = next(item for item in model.ncnn.files if item.filename.endswith(".param"))
    bin_file = next(item for item in model.ncnn.files if item.filename.endswith(".bin"))
    param_path = downloaded[param_file.filename]
    bin_path = downloaded[bin_file.filename]
    if param_path.parent != bin_path.parent:
        raise ValueError("Downloaded NCNN .param and .bin files are not in the same Hub snapshot")
    if progress_callback is not None:
        progress_callback({"message": f"Verified {model.name} NCNN files", "progress": 1.0})
    return DownloadedNCNNArtifact(param_path, bin_path, HF_MODEL_REVISION, model.ncnn)
