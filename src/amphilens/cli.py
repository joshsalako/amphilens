"""Typer CLI for AmphiLens."""

from __future__ import annotations

import json
import csv
import subprocess
import sys
from pathlib import Path

try:
    import typer
except ImportError:  # pragma: no cover - exercised only in minimal installs
    typer = None

from .core import ProjectManifest, ProjectStore, iter_images
from .doctor import run_doctor
from .annotations.cvat import export_cvat, import_cvat
from .active_learning import HybridPPALConfig, HybridPPALStrategy, PPALCalibration
from .core import DetectionRecord
from .core import InferenceConfig
from .curation import write_selection_artifacts
from .inference import write_predictions_csv
from .models import load_detector
from .runs import run_resumable_inference


def _require_typer():
    if typer is None:
        raise RuntimeError("The CLI requires the 'cli' extra: pip install 'amphilens[cli]'")


if typer is not None:
    app = typer.Typer(help="Reproducible wildlife camera-trap detection.")
    project_app = typer.Typer(help="Create and inspect portable projects.")
    cvat_app = typer.Typer(help="Exchange annotations with CVAT and compatible tools.")
    app.add_typer(project_app, name="project")
    app.add_typer(cvat_app, name="cvat")

    @app.command()
    def doctor(path: str = "."):
        """Report local CPU, disk, ML dependency, and CUDA status."""
        typer.echo(json.dumps(run_doctor(path).to_dict(), indent=2))

    @project_app.command("create")
    def project_create(
        project_dir: Path,
        image_root: list[Path] = typer.Option(..., "--image-root"),
        class_name: list[str] = typer.Option(..., "--class-name"),
        name: str = typer.Option("amphilens-project", "--name"),
    ):
        """Create a portable AmphiLens project directory."""
        manifest = ProjectManifest.create(name, image_root, class_name)
        ProjectStore(project_dir).create(manifest)
        typer.echo(f"Created project at {project_dir.resolve()}")

    @project_app.command("inspect")
    def project_inspect(project_dir: Path):
        """Print a project's validated manifest."""
        typer.echo(json.dumps(ProjectStore(project_dir).load_manifest().to_dict(), indent=2))

    @app.command("project-create")
    def project_create_legacy(
        project_dir: Path,
        image_root: list[Path] = typer.Option(..., "--image-root"),
        class_name: list[str] = typer.Option(..., "--class-name"),
        name: str = typer.Option("amphilens-project", "--name"),
    ):
        """Backward-compatible alias for `project create`."""
        project_create(project_dir, image_root, class_name, name)

    @app.command("app")
    def app_ui():
        """Launch the local browser application."""
        ui_path = Path(__file__).with_name("ui.py")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)], check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("The UI requires the 'ui' extra: pip install 'amphilens[ui]'") from exc

    @app.command()
    def images(project_dir: Path):
        """List image files recorded by a project manifest."""
        manifest = ProjectStore(project_dir).load_manifest()
        for path in iter_images(manifest.image_roots):
            typer.echo(path)

    @app.command()
    def predict(
        project_dir: Path,
        checkpoint: Path,
        architecture: str = typer.Option(..., "--architecture", help="yolo, rtdetr, or faster_rcnn"),
        output_dir: Path = typer.Option(..., "--output-dir"),
        confidence: float = typer.Option(0.25, "--confidence"),
        image_size: int = typer.Option(640, "--image-size"),
        device: str = typer.Option("auto", "--device"),
        model_id: str | None = typer.Option(None, "--model-id"),
        run_id: str | None = typer.Option(None, "--run-id"),
    ):
        """Run a compatible detector and persist resumable prediction artifacts."""
        manifest = ProjectStore(project_dir).load_manifest()
        resolved_model_id = model_id or checkpoint.stem
        detector = load_detector(
            checkpoint,
            architecture=architecture,
            classes=manifest.classes,
            model_id=resolved_model_id,
        )
        config = InferenceConfig(
            model_id=resolved_model_id,
            image_size=image_size,
            confidence=confidence,
            device=device,
            run_id=run_id or f"predict-{resolved_model_id}",
        )
        summary = run_resumable_inference(
            detector, iter_images(manifest.image_roots), config, output_dir
        )
        typer.echo(json.dumps({
            "run_id": summary.run_id,
            "completed_images": summary.completed_images,
            "failed_images": summary.failed_images,
            "detection_count": summary.detection_count,
            "predictions_csv": str(summary.predictions_csv),
        }, indent=2))

    @app.command("active-learn")
    def active_learn(
        predictions_csv: Path,
        calibration_json: Path,
        features_json: Path,
        output_dir: Path,
        budget: int = typer.Option(100, "--budget"),
        seed: int = typer.Option(42, "--seed"),
    ):
        """Select a reproducible Hybrid PPAL annotation queue."""
        predictions = _records_from_csv(predictions_csv)
        calibration = PPALCalibration(**json.loads(calibration_json.read_text()))
        features = json.loads(features_json.read_text())
        config = HybridPPALConfig(budget=budget, seed=seed)
        selected = HybridPPALStrategy(config).select(
            predictions, calibration, features=features
        )
        artifacts = write_selection_artifacts(selected, calibration, config, output_dir)
        typer.echo(json.dumps({name: str(path) for name, path in artifacts.items()}, indent=2))

    def _records_from_csv(path: Path) -> list[DetectionRecord]:
        records = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                records.append(
                    DetectionRecord(
                        image_path=row["image_path"],
                        image_id=row["image_id"],
                        class_id=int(row["class_id"]),
                        class_name=row["class_name"],
                        confidence=float(row["confidence"]) if row.get("confidence") else None,
                        bbox_xyxy=[
                            float(row["bbox_xmin"]),
                            float(row["bbox_ymin"]),
                            float(row["bbox_xmax"]),
                            float(row["bbox_ymax"]),
                        ],
                        image_width=int(row["image_width"]),
                        image_height=int(row["image_height"]),
                        model_id=row.get("model_id", "csv-import"),
                        run_id=row.get("run_id", "csv-import"),
                    )
                )
        return records

    @cvat_app.command("export")
    def cvat_export(
        predictions_csv: Path,
        output_dir: Path,
        class_name: list[str] = typer.Option(..., "--class-name"),
    ):
        """Export prediction rows as a portable CVAT/COCO task."""
        task = export_cvat(_records_from_csv(predictions_csv), output_dir, classes=class_name)
        typer.echo(f"Exported CVAT task to {task}")

    @cvat_app.command("import")
    def cvat_import(task_dir: Path, output_csv: Path):
        """Import a CVAT task into the stable AmphiLens CSV schema."""
        write_predictions_csv(import_cvat(task_dir), output_csv)
        typer.echo(f"Imported annotations to {output_csv.resolve()}")

    def main():
        app()
else:

    def main():
        _require_typer()
