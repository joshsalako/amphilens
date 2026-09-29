import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image

from amphilens.cloud.constants import IMAGE_APT_PACKAGES, IMAGE_PINS
from amphilens.cloud.credentials import CloudCredentialsStore
from amphilens.cloud.estimate import RATE_CHECKED_AT, estimate_training_cost
from amphilens.cloud.pack import pack_training_payload


def test_modal_training_image_numpy_pin_supports_its_python_311_base():
    assert IMAGE_PINS["python"] == "3.11"
    assert IMAGE_PINS["numpy"] == "2.3.5"


def test_modal_training_image_includes_opencv_runtime_libraries():
    assert {"libgl1", "libglib2.0-0"}.issubset(IMAGE_APT_PACKAGES)


def test_cloud_credentials_are_private_and_environment_wins(tmp_path: Path):
    path = tmp_path / "credentials.json"
    store = CloudCredentialsStore(path)
    store.save("stored-id", "stored-secret")

    assert path.stat().st_mode & 0o777 == 0o600
    stored = store.resolve(environ={})
    assert stored.token_id == "stored-id"
    assert stored.token_secret == "stored-secret"
    assert "stored-secret" not in repr(stored)
    assert "stored-secret" not in json.dumps(stored.describe())

    environment = store.resolve(
        environ={"MODAL_TOKEN_ID": "env-id", "MODAL_TOKEN_SECRET": "env-secret"}
    )
    assert environment.source == "environment"
    assert environment.token_id == "env-id"
    assert environment.token_secret == "env-secret"


def test_cloud_credentials_read_dotenv_when_process_environment_is_unset(
    tmp_path: Path, monkeypatch
):
    (tmp_path / ".env").write_text(
        'MODAL_TOKEN_ID="dotenv-id"\nMODAL_TOKEN_SECRET="dotenv-secret"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    credentials = CloudCredentialsStore(tmp_path / "credentials.json").resolve()

    assert credentials.source == "dotenv"
    assert credentials.token_id == "dotenv-id"
    assert credentials.token_secret == "dotenv-secret"


def test_cloud_cost_estimate_is_a_range_and_derives_a_bounded_timeout():
    estimate = estimate_training_cost(
        image_count=2000,
        dataset_bytes=500_000_000,
        epochs=100,
        gpu="L4",
        max_cost_usd=5.0,
    )

    assert estimate.low_usd < estimate.high_usd
    assert estimate.rate_checked_at == RATE_CHECKED_AT
    assert 0 < estimate.time_limit_seconds <= 24 * 60 * 60
    assert estimate.upload_time_low_seconds <= estimate.upload_time_high_seconds
    assert "not a guaranteed billing cap" in estimate.disclaimer.lower()


def test_cloud_estimate_rejects_budget_below_its_low_range():
    import pytest

    from amphilens.core import ValidationError

    with pytest.raises(ValidationError, match="cheapest plausible run"):
        estimate_training_cost(
            image_count=2000,
            dataset_bytes=500_000_000,
            epochs=100,
            gpu="L4",
            max_cost_usd=0.01,
        )


def test_training_payload_is_private_deterministic_and_remote_path_safe(tmp_path: Path):
    prepared = tmp_path / "prepared"
    image_dir = prepared / "images"
    image_dir.mkdir(parents=True)
    (prepared / "labels").mkdir()
    source_image = Image.new("RGB", (8, 8), (20, 30, 40))
    exif = source_image.getexif()
    exif[271] = "private-camera-model"
    source_image.save(image_dir / "frog.jpg", exif=exif)
    (prepared / "labels" / "frog.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (prepared / "dataset.yaml").write_text(
        json.dumps(
            {
                "path": str(tmp_path / "user-home" / "project"),
                "train": "images",
                "labels": "labels",
                "names": {"0": "frog"},
            }
        )
    )

    first = pack_training_payload(
        prepared,
        tmp_path / "first.zip",
        container_mount="/mnt/amphilens/jobs/job-key/dataset",
    )
    second = pack_training_payload(
        prepared,
        tmp_path / "second.zip",
        container_mount="/mnt/amphilens/jobs/job-key/dataset",
    )

    assert first.sha256 == second.sha256
    assert first.sha256 == hashlib.sha256((tmp_path / "first.zip").read_bytes()).hexdigest()
    with zipfile.ZipFile(tmp_path / "first.zip") as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert not any("user-home" in name for name in names)
        dataset = json.loads(archive.read("dataset/dataset.yaml"))
        assert dataset["path"] == "/mnt/amphilens/jobs/job-key/dataset"
        assert dataset["train"] == "images"
        image_path = tmp_path / "stripped.jpg"
        image_path.write_bytes(archive.read("dataset/images/frog.jpg"))
    with Image.open(image_path) as uploaded_image:
        assert uploaded_image.getexif() == {}
        assert "private-camera-model" not in str(uploaded_image.info)


def test_cloud_payload_redacts_unneeded_paths_from_checkpoint_manifest(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "dataset.yaml").write_text(
        json.dumps({"path": str(prepared), "train": "images", "names": {"0": "frog"}})
    )
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base weights")
    private_path = "/Users/example/private-field-site/original-checkpoint.pt"
    base_manifest = {
        "checkpoint_path": private_path,
        "model_id": "yolo26-l",
        "architecture": "yolo",
        "classes": ["frog"],
        "preprocessing": {"compatibility_mode": "max-dimension"},
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "created_at": "2026-09-29T00:00:00+00:00",
        "parent_checkpoint": private_path,
        "software": {"custom": private_path},
        "training_config": {"metadata": {"source_path": private_path}},
        "schema_version": 1,
    }

    payload = pack_training_payload(
        prepared,
        tmp_path / "payload.zip",
        container_mount="/mnt/amphilens/jobs/key/dataset",
        base_checkpoint=checkpoint,
        base_manifest=base_manifest,
    )

    with zipfile.ZipFile(payload.path) as archive:
        manifest = archive.read("model/checkpoint.json").decode()
    assert private_path not in manifest
    assert "/mnt/amphilens/jobs/key/model/base.pt" in manifest
