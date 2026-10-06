import csv
import json
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

typer = pytest.importorskip("typer")
from typer.testing import CliRunner

import amphilens.cli as cli_module
from amphilens.annotations.managed import CVATProjectSummary, CVATTaskSummary
from amphilens.cli import app
from amphilens.core import DetectionRecord, ProjectConfig, ProjectManifest, ProjectStore
from amphilens.inference import write_predictions_csv
from amphilens.preprocessing import PreprocessingConfig


def test_project_create_and_inspect_commands(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    project = tmp_path / "project"
    runner = CliRunner()

    created = runner.invoke(
        app,
        [
            "project",
            "create",
            str(project),
            "--image-root",
            str(images),
            "--class-name",
            "toad",
            "--name",
            "demo",
        ],
    )
    assert created.exit_code == 0, created.stdout

    inspected = runner.invoke(app, ["project", "inspect", str(project)])
    assert inspected.exit_code == 0, inspected.stdout
    assert '"name": "demo"' in inspected.stdout


def test_project_default_location_command_uses_user_data():
    from amphilens.locations import default_projects_root

    result = CliRunner().invoke(app, ["project", "default-location"])

    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip() == str(default_projects_root())


def test_app_starts_local_browser_server_by_default(monkeypatch):
    calls = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "uvicorn",
        SimpleNamespace(run=lambda app, **kwargs: calls.append((app, kwargs))),
    )

    result = CliRunner().invoke(app, ["app"])

    assert result.exit_code == 0, result.stdout
    assert calls == [
        ("amphilens.webapp:app", {"host": "127.0.0.1", "port": 8501, "log_level": "info"})
    ]


def test_project_move_command_verifies_and_removes_source(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    source = tmp_path / "old-project"
    destination = tmp_path / "projects" / "study"
    runner = CliRunner()
    created = runner.invoke(
        app,
        [
            "project",
            "create",
            str(source),
            "--image-root",
            str(images),
            "--class-name",
            "toad",
        ],
    )
    assert created.exit_code == 0, created.stdout

    result = runner.invoke(
        app,
        ["project", "move", str(source), str(destination), "--remove-source"],
    )

    assert result.exit_code == 0, result.stdout
    assert not source.exists()
    assert (destination / "manifest.json").is_file()


def test_project_create_rejects_checkout_path_even_when_launched_elsewhere(
    monkeypatch, tmp_path: Path
):
    images = tmp_path / "images"
    images.mkdir()
    checkout_project = Path(__file__).parents[1] / ".amphilens" / "test-project-location"
    monkeypatch.chdir(tmp_path)

    try:
        result = CliRunner().invoke(
            app,
            [
                "project",
                "create",
                str(checkout_project),
                "--image-root",
                str(images),
                "--class-name",
                "toad",
            ],
        )

        assert result.exit_code != 0
        assert "source checkout" in str(result.exception)
        assert not checkout_project.exists()
    finally:
        shutil.rmtree(checkout_project, ignore_errors=True)


def test_dataset_import_command_creates_an_immutable_snapshot(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    project = tmp_path / "project"
    runner = CliRunner()
    assert (
        runner.invoke(
            app,
            [
                "project",
                "create",
                str(project),
                "--image-root",
                str(images),
                "--class-name",
                "toad",
            ],
        ).exit_code
        == 0
    )
    archive = tmp_path / "initial.zip"
    image_bytes = BytesIO()
    Image.new("RGB", (20, 10), color="black").save(image_bytes, format="JPEG")
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("images/camera.jpg", image_bytes.getvalue())
        handle.writestr("classes.txt", "toad\n")
        handle.writestr("labels/camera.txt", "")

    result = runner.invoke(app, ["dataset", "import", str(project), str(archive)])

    assert result.exit_code == 0, result.stdout
    assert "snapshot_id" in result.stdout


def test_cvat_projects_command_lists_project_metadata(monkeypatch):
    class FakeTransport:
        def __init__(self, server_url=None, token=None):
            assert server_url == "https://cvat.example"
            assert token is None

        def list_projects(self):
            return [
                CVATProjectSummary(
                    "17",
                    "initial annotations",
                    ["toad"],
                    [CVATTaskSummary("23", "task one", 5, "completed")],
                )
            ]

    monkeypatch.setattr(cli_module, "CVATSdkTransport", FakeTransport)
    result = CliRunner().invoke(
        app,
        ["cvat", "projects", "--server-url", "https://cvat.example"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload[0]["project_id"] == "17"
    assert payload[0]["task_count"] == 1
    assert payload[0]["tasks"][0]["name"] == "task one"


def test_cvat_project_import_command_uses_project_id_and_mapping(monkeypatch, tmp_path: Path):
    observed = {}

    def fake_import(self, project_id, *, class_mapping=None, server_url=None):
        observed.update(
            project_id=project_id,
            class_mapping=class_mapping,
            server_url=server_url,
        )
        return SimpleNamespace(
            manifest=SimpleNamespace(to_dict=lambda: {"snapshot_id": "snapshot-abc"})
        )

    monkeypatch.setattr(cli_module.ProjectStore, "import_cvat_project", fake_import)
    result = CliRunner().invoke(
        app,
        [
            "dataset",
            "import-cvat",
            str(tmp_path / "project"),
            "--project-id",
            "17",
            "--server-url",
            "https://cvat.example",
            "--class-mapping",
            '{"western leopard toad":"toad"}',
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert observed == {
        "project_id": "17",
        "class_mapping": {"western leopard toad": "toad"},
        "server_url": "https://cvat.example",
    }


def test_train_command_is_blocked_until_a_labelled_snapshot_exists(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    project = tmp_path / "project"
    assert (
        CliRunner()
        .invoke(
            app,
            [
                "project",
                "create",
                str(project),
                "--image-root",
                str(images),
                "--class-name",
                "toad",
            ],
        )
        .exit_code
        == 0
    )

    result = CliRunner().invoke(app, ["train", str(project), "--output-dir", str(tmp_path / "out")])

    assert result.exit_code != 0
    assert "Import an annotated dataset" in str(result.exception)


def test_train_command_passes_model_and_preprocessing_choices_to_engine(
    monkeypatch, tmp_path: Path
):
    images = tmp_path / "images"
    images.mkdir()
    project = tmp_path / "project"
    runner = CliRunner()
    assert (
        runner.invoke(
            app,
            [
                "project",
                "create",
                str(project),
                "--image-root",
                str(images),
                "--class-name",
                "toad",
            ],
        ).exit_code
        == 0
    )
    archive = tmp_path / "initial.zip"
    image_bytes = BytesIO()
    Image.new("RGB", (20, 10), color="black").save(image_bytes, format="JPEG")
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("images/camera.jpg", image_bytes.getvalue())
        handle.writestr("classes.txt", "toad\n")
        handle.writestr("labels/camera.txt", "")
    assert runner.invoke(app, ["dataset", "import", str(project), str(archive)]).exit_code == 0

    observed = {}

    class Detector:
        pass

    def fake_load(preset, **_):
        observed["preset"] = preset
        return Detector()

    monkeypatch.setattr(cli_module, "load_preset_detector", fake_load)

    def fake_train(detector, **kwargs):
        observed.update(kwargs)
        checkpoint = tmp_path / "out" / "best.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"weights")
        return SimpleNamespace(checkpoint=checkpoint)

    monkeypatch.setattr(cli_module, "train_snapshot_and_register", fake_train)
    result = runner.invoke(
        app,
        [
            "train",
            str(project),
            "--output-dir",
            str(tmp_path / "out"),
            "--model-preset",
            "rtdetr-l",
            "--max-dimension",
            "320",
            "--no-grayscale",
            "--clahe",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert observed["preset"].model_id == "rtdetr-l"
    assert observed["preprocessing"].max_dimension == 320
    assert observed["preprocessing"].grayscale_enabled is False
    assert observed["preprocessing"].clahe_enabled is True


def test_train_command_uses_saved_project_configuration_when_options_are_omitted(
    monkeypatch, tmp_path: Path
):
    images = tmp_path / "images"
    images.mkdir()
    project = tmp_path / "project"
    config = ProjectConfig(
        classes=["toad"],
        model_preset="rtdetr-l",
        preprocessing=PreprocessingConfig(max_dimension=320, clahe_enabled=True),
        image_size=320,
        epochs=3,
        batch_size=2,
        device="cpu",
    )
    ProjectStore(project).create(
        ProjectManifest.create("configured", [images], ["toad"], project_config=config)
    )
    archive = tmp_path / "initial.zip"
    image_bytes = BytesIO()
    Image.new("RGB", (20, 10), color="black").save(image_bytes, format="JPEG")
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("images/camera.jpg", image_bytes.getvalue())
        handle.writestr("classes.txt", "toad\n")
        handle.writestr("labels/camera.txt", "")
    assert CliRunner().invoke(app, ["dataset", "import", str(project), str(archive)]).exit_code == 0

    observed = {}

    class Detector:
        pass

    monkeypatch.setattr(
        cli_module,
        "load_preset_detector",
        lambda preset, **_: observed.update(preset=preset) or Detector(),
    )

    def fake_train(detector, **kwargs):
        observed.update(kwargs)
        checkpoint = tmp_path / "out" / "best.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"weights")
        return SimpleNamespace(checkpoint=checkpoint)

    monkeypatch.setattr(cli_module, "train_snapshot_and_register", fake_train)
    result = CliRunner().invoke(app, ["train", str(project), "--output-dir", str(tmp_path / "out")])

    assert result.exit_code == 0, result.stdout
    assert observed["preset"].model_id == "rtdetr-l"
    assert observed["preprocessing"].max_dimension == 320
    assert observed["preprocessing"].clahe_enabled is True
    assert observed["config"].epochs == 3
    assert observed["config"].batch_size == 2


def test_predict_command_uses_saved_project_configuration_when_options_are_omitted(
    monkeypatch, tmp_path: Path
):
    images = tmp_path / "images"
    images.mkdir()
    (images / "camera.jpg").write_bytes(b"fixture")
    project = tmp_path / "project"
    config = ProjectConfig(
        classes=["toad"],
        model_preset="rtdetr-l",
        preprocessing=PreprocessingConfig(max_dimension=320, clahe_enabled=True),
        image_size=320,
        confidence_threshold=0.61,
        device="cpu",
    )
    ProjectStore(project).create(
        ProjectManifest.create("configured", [images], ["toad"], project_config=config)
    )
    observed = {}

    class Detector:
        model_id = "external"

        def predict(self, image_paths, config):
            observed["config"] = config
            for path in image_paths:
                yield DetectionRecord(
                    image_path=str(path),
                    image_id=path.name,
                    class_id=0,
                    class_name="toad",
                    confidence=0.9,
                    bbox_xyxy=[1, 1, 5, 5],
                    image_width=10,
                    image_height=10,
                    model_id="external",
                    run_id=config.run_id,
                )

    def fake_load_detector(path, **kwargs):
        observed["kwargs"] = kwargs
        return Detector()

    monkeypatch.setattr(cli_module, "load_detector", fake_load_detector)
    checkpoint = tmp_path / "external.pt"
    checkpoint.write_bytes(b"weights")
    result = CliRunner().invoke(
        app,
        ["predict", str(project), str(checkpoint), "--output-dir", str(tmp_path / "output")],
    )

    assert result.exit_code == 0, result.stdout
    assert observed["kwargs"]["architecture"] == "rtdetr"
    assert observed["config"].image_size == 320
    assert observed["config"].confidence == 0.61
    assert observed["config"].preprocessing_config.clahe_enabled is True


def test_cvat_cli_exports_and_imports_prediction_csv(tmp_path: Path):
    image = tmp_path / "camera.jpg"
    Image.new("RGB", (20, 40), color="black").save(image)
    predictions = tmp_path / "predictions.csv"
    write_predictions_csv(
        [
            DetectionRecord(
                image_path=str(image),
                image_id=image.name,
                class_id=0,
                class_name="toad",
                confidence=0.9,
                bbox_xyxy=[1, 2, 11, 22],
                image_width=20,
                image_height=40,
                model_id="fixture",
                run_id="run-1",
            )
        ],
        predictions,
    )
    runner = CliRunner()
    exported = runner.invoke(
        app,
        ["cvat", "export", str(predictions), str(tmp_path / "task"), "--class-name", "toad"],
    )
    assert exported.exit_code == 0, exported.stdout

    imported_path = tmp_path / "imported.csv"
    imported = runner.invoke(app, ["cvat", "import", str(tmp_path / "task"), str(imported_path)])
    assert imported.exit_code == 0, imported.stdout
    with imported_path.open(newline="") as handle:
        assert list(csv.DictReader(handle))[0]["class_name"] == "toad"


def test_predict_command_runs_project_images(monkeypatch, tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "camera.jpg").write_bytes(b"fixture")
    project = tmp_path / "project"
    runner = CliRunner()
    created = runner.invoke(
        app,
        ["project", "create", str(project), "--image-root", str(images), "--class-name", "toad"],
    )
    assert created.exit_code == 0, created.stdout

    class Detector:
        model_id = "fixture"

        def predict(self, image_paths, config):
            for path in image_paths:
                yield DetectionRecord(
                    image_path=str(path),
                    image_id=path.name,
                    class_id=0,
                    class_name="toad",
                    confidence=0.9,
                    bbox_xyxy=[1, 1, 5, 5],
                    image_width=10,
                    image_height=10,
                    model_id="fixture",
                    run_id=config.run_id,
                )

    monkeypatch.setattr(cli_module, "load_detector", lambda *args, **kwargs: Detector())
    result = runner.invoke(
        app,
        [
            "predict",
            str(project),
            str(tmp_path / "model.pt"),
            "--architecture",
            "yolo",
            "--output-dir",
            str(tmp_path / "artifacts"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "artifacts" / "predictions.csv").is_file()


def test_predict_indexes_artifacts_when_output_is_inside_project(monkeypatch, tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "camera.jpg").write_bytes(b"fixture")
    project = tmp_path / "project"
    runner = CliRunner()
    created = runner.invoke(
        app,
        ["project", "create", str(project), "--image-root", str(images), "--class-name", "toad"],
    )
    assert created.exit_code == 0, created.stdout

    class Detector:
        model_id = "fixture"

        def predict(self, image_paths, config):
            for path in image_paths:
                yield DetectionRecord(
                    image_path=str(path),
                    image_id=path.name,
                    class_id=0,
                    class_name="toad",
                    confidence=0.9,
                    bbox_xyxy=[1, 1, 5, 5],
                    image_width=10,
                    image_height=10,
                    model_id="fixture",
                    run_id=config.run_id,
                )

    monkeypatch.setattr(cli_module, "load_detector", lambda *args, **kwargs: Detector())
    output = project / "artifacts" / "predict"
    result = runner.invoke(
        app,
        [
            "predict",
            str(project),
            str(tmp_path / "model.pt"),
            "--architecture",
            "yolo",
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    index = json.loads((project / "artifacts" / "index.json").read_text())
    assert {item["artifact_type"] for item in index["artifacts"]} >= {
        "run-manifest",
        "predictions-csv",
        "run-summary",
    }


def test_active_learn_command_writes_ppal_queue(tmp_path: Path):
    predictions = tmp_path / "predictions.csv"
    records = []
    for name, confidence in (("a.jpg", 0.51), ("b.jpg", 0.9)):
        records.append(
            DetectionRecord(
                image_path=f"/pool/{name}",
                image_id=name,
                class_id=0,
                class_name="toad",
                confidence=confidence,
                bbox_xyxy=[1, 1, 5, 5],
                image_width=10,
                image_height=10,
                model_id="fixture",
                run_id="run-1",
            )
        )
    write_predictions_csv(records, predictions)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "classes": ["toad"],
                "difficulties": {"toad": 0.4},
                "weights": {"toad": 1.1},
                "xi": 0.5,
                "alpha": 1.0,
                "beta": 2.0,
                "source": "validation",
            }
        )
    )
    features = tmp_path / "features.json"
    features.write_text(json.dumps({"/pool/a.jpg": [1, 0], "/pool/b.jpg": [0, 1]}))

    result = CliRunner().invoke(
        app,
        [
            "active-learn",
            str(predictions),
            str(calibration),
            str(features),
            str(tmp_path / "cycle"),
            "--budget",
            "1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "cycle" / "selection_queue.csv").is_file()


def test_report_command_writes_summary_and_markdown(tmp_path: Path):
    predictions = tmp_path / "predictions.csv"
    write_predictions_csv(
        [
            DetectionRecord(
                image_path="/pool/a.jpg",
                image_id="a.jpg",
                class_id=0,
                class_name="toad",
                confidence=0.8,
                bbox_xyxy=[1, 1, 5, 5],
                image_width=10,
                image_height=10,
                model_id="fixture",
                run_id="run-1",
            )
        ],
        predictions,
    )
    result = CliRunner().invoke(app, ["report", str(predictions), str(tmp_path / "report")])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "report" / "report.md").is_file()


def test_checkpoint_registry_commands_persist_compatibility_metadata(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture-weights")
    registry = tmp_path / "registry"
    runner = CliRunner()
    registered = runner.invoke(
        app,
        [
            "checkpoint",
            "register",
            str(registry),
            str(checkpoint),
            "--model-id",
            "fixture-yolo",
            "--architecture",
            "yolo",
            "--class-name",
            "toad",
            "--training-domain",
            "camera-trap",
            "--source",
            "fixture",
            "--license",
            "Apache-2.0",
            "--preprocessing",
            '{"name":"none"}',
        ],
    )
    assert registered.exit_code == 0, registered.stdout

    listed = runner.invoke(app, ["checkpoint", "list", str(registry)])
    assert listed.exit_code == 0, listed.stdout
    assert "fixture-yolo" in listed.stdout

    inspected = runner.invoke(app, ["checkpoint", "inspect", str(registry), "fixture-yolo"])
    assert inspected.exit_code == 0, inspected.stdout
    assert '"architecture": "yolo"' in inspected.stdout


def test_predict_can_enforce_registry_compatibility(monkeypatch, tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "camera.jpg").write_bytes(b"fixture")
    project = tmp_path / "project"
    registry = tmp_path / "registry"
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture-weights")
    runner = CliRunner()
    assert (
        runner.invoke(
            app,
            [
                "project",
                "create",
                str(project),
                "--image-root",
                str(images),
                "--class-name",
                "toad",
            ],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app,
            [
                "checkpoint",
                "register",
                str(registry),
                str(checkpoint),
                "--model-id",
                "fixture",
                "--architecture",
                "yolo",
                "--class-name",
                "toad",
                "--preprocessing",
                '{"name":"none"}',
            ],
        ).exit_code
        == 0
    )

    class Detector:
        model_id = "fixture"

        def predict(self, image_paths, config):
            for path in image_paths:
                yield DetectionRecord(
                    image_path=str(path),
                    image_id=path.name,
                    class_id=0,
                    class_name="toad",
                    confidence=0.9,
                    bbox_xyxy=[1, 1, 5, 5],
                    image_width=10,
                    image_height=10,
                    model_id="fixture",
                    run_id=config.run_id,
                )

    observed = {}

    def fake_load_detector(path, **kwargs):
        observed.update(kwargs)
        return Detector()

    monkeypatch.setattr(cli_module, "load_detector", fake_load_detector)
    result = runner.invoke(
        app,
        [
            "predict",
            str(project),
            str(checkpoint),
            "--architecture",
            "yolo",
            "--output-dir",
            str(tmp_path / "output"),
            "--model-id",
            "fixture",
            "--registry-dir",
            str(registry),
            "--preprocessing",
            '{"name":"none"}',
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert observed["checkpoint_manifest"].model_id == "fixture"
