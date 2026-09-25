import csv
import json
from pathlib import Path

import pytest
from PIL import Image

typer = pytest.importorskip("typer")
from typer.testing import CliRunner

import amphilens.cli as cli_module
from amphilens.cli import app
from amphilens.core import DetectionRecord
from amphilens.inference import write_predictions_csv


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
