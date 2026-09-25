from pathlib import Path
import csv

import pytest

typer = pytest.importorskip("typer")
from typer.testing import CliRunner

from amphilens.cli import app
import amphilens.cli as cli_module
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
    image.write_bytes(b"fixture")
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
                    image_path=str(path), image_id=path.name, class_id=0, class_name="toad",
                    confidence=0.9, bbox_xyxy=[1, 1, 5, 5], image_width=10, image_height=10,
                    model_id="fixture", run_id=config.run_id,
                )

    monkeypatch.setattr(cli_module, "load_detector", lambda *args, **kwargs: Detector())
    result = runner.invoke(
        app,
        [
            "predict", str(project), str(tmp_path / "model.pt"),
            "--architecture", "yolo", "--output-dir", str(tmp_path / "artifacts"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "artifacts" / "predictions.csv").is_file()
