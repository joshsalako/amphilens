"""Privacy-preserving, deterministic packaging for a remote training run."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..core import ValidationError


@dataclass(frozen=True, slots=True)
class PreparedPayload:
    path: Path
    sha256: str
    size_bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100600 << 16
    return info


def _clean_image(source: Path, destination: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Cloud payload preparation requires Pillow") from exc
    try:
        with Image.open(source) as opened:
            rgb = opened.convert("RGB")
            cleaned = Image.new("RGB", rgb.size)
            cleaned.paste(rgb)
            image_format = Image.registered_extensions().get(source.suffix.lower())
            if image_format is None:
                raise ValidationError(f"Unsupported prepared image format: {source.suffix}")
            save_options = {"quality": 95} if image_format == "JPEG" else {}
            cleaned.save(destination, format=image_format, **save_options)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Prepared image cannot be safely re-encoded: {source.name}") from exc


def pack_training_payload(
    prepared_dataset: str | Path,
    output_zip: str | Path,
    *,
    container_mount: str,
    base_checkpoint: str | Path | None = None,
    base_manifest: dict | None = None,
) -> PreparedPayload:
    """Re-encode dataset images, rewrite its root, and write a reproducible zip."""
    source_root = Path(prepared_dataset).expanduser().resolve()
    destination = Path(output_zip).expanduser().resolve()
    dataset_yaml = source_root / "dataset.yaml"
    if not source_root.is_dir() or not dataset_yaml.is_file():
        raise ValidationError("Prepared dataset must contain dataset.yaml")
    mount = PurePosixPath(container_mount)
    if not mount.is_absolute() or ".." in mount.parts:
        raise ValidationError("Container dataset path must be an absolute safe path")
    try:
        dataset_config = json.loads(dataset_yaml.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("Prepared dataset.yaml must contain a JSON/YAML mapping") from exc
    if not isinstance(dataset_config, dict):
        raise ValidationError("Prepared dataset.yaml must contain a mapping")
    dataset_config["path"] = str(mount)
    rewritten_yaml = json.dumps(dataset_config, ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    checkpoint_path = Path(base_checkpoint).expanduser().resolve() if base_checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise ValidationError(f"Base checkpoint does not exist: {checkpoint_path}")
    if base_manifest is not None and checkpoint_path is None:
        raise ValidationError("A base checkpoint manifest requires a base checkpoint")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as out:
            entries: list[tuple[str, Path | bytes, bool]] = [
                ("dataset/dataset.yaml", rewritten_yaml.encode("utf-8"), False)
            ]
            for source in sorted(item for item in source_root.rglob("*") if item.is_file()):
                relative = source.relative_to(source_root).as_posix()
                if relative == "dataset.yaml":
                    continue
                if relative.startswith("/") or ".." in PurePosixPath(relative).parts:
                    raise ValidationError("Prepared dataset contains an unsafe relative path")
                entries.append((f"dataset/{relative}", source, relative.startswith("images/")))
            if checkpoint_path is not None:
                entries.append(("model/base.pt", checkpoint_path, False))
                if base_manifest is not None:
                    manifest_data = {
                        "checkpoint_path": str(mount.parent / "model" / "base.pt"),
                        "model_id": base_manifest["model_id"],
                        "architecture": base_manifest["architecture"],
                        "classes": list(base_manifest["classes"]),
                        "preprocessing": dict(base_manifest["preprocessing"]),
                        "sha256": base_manifest["sha256"],
                        "created_at": base_manifest["created_at"],
                        "software": {},
                        "training_config": {},
                        "schema_version": base_manifest.get("schema_version", 1),
                    }
                    entries.append(
                        (
                            "model/checkpoint.json",
                            (json.dumps(manifest_data, ensure_ascii=False, sort_keys=True, indent=2)
                             + "\n").encode("utf-8"),
                            False,
                        )
                    )
            for name, value, image in sorted(entries, key=lambda entry: entry[0]):
                if isinstance(value, bytes):
                    out.writestr(_zip_info(name), value)
                elif image:
                    with tempfile.TemporaryDirectory(dir=destination.parent) as temp_dir:
                        encoded = Path(temp_dir) / value.name
                        _clean_image(value, encoded)
                        out.writestr(_zip_info(name), encoded.read_bytes())
                else:
                    out.writestr(_zip_info(name), value.read_bytes())
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return PreparedPayload(destination, _sha256(destination), destination.stat().st_size)
