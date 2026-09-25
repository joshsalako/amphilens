"""Typer CLI for AmphiLens."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import typer
except ImportError:  # pragma: no cover - exercised only in minimal installs
    typer = None

from .active_learning import HybridPPALConfig, HybridPPALStrategy, PPALCalibration
from .annotations.cvat import export_cvat, import_cvat
from .annotations.managed import CVATSdkTransport, ManagedCVATCycleService, selection_hash
from .core import (
    CheckpointManifest,
    DetectionRecord,
    InferenceConfig,
    ModelManifest,
    ProjectManifest,
    ProjectStore,
    iter_images,
)
from .curation import write_selection_artifacts
from .dataset import DatasetSnapshot
from .doctor import run_doctor
from .inference import read_predictions_csv, write_predictions_csv
from .models import ModelCatalog, load_detector, load_preset_detector
from .preprocessing import PreprocessingConfig
from .registry import ModelRegistry
from .reporting import write_report
from .runs import run_resumable_inference
from .training import TrainingConfig, load_checkpoint_manifest, train_snapshot_and_register


def _require_typer():
    if typer is None:
        raise RuntimeError("The CLI requires the 'cli' extra: pip install 'amphilens[cli]'")


def _register_run_artifacts(store: ProjectStore, summary, output_dir: Path) -> None:
    """Index run outputs when the caller placed them inside the project."""
    output = output_dir.expanduser().resolve()
    if not (output == store.root or output.is_relative_to(store.root)):
        return
    artifacts = (
        (summary.run_manifest, "run-manifest"),
        (summary.predictions_csv, "predictions-csv"),
        (summary.summary_json, "run-summary"),
        (output / "progress.json", "run-progress"),
        (output / "predictions.jsonl", "predictions-jsonl"),
    )
    for path, artifact_type in artifacts:
        if path.is_file():
            store.register_artifact(
                path,
                artifact_type=artifact_type,
                producer_run=summary.run_id,
            )


if typer is not None:
    app = typer.Typer(help="Reproducible wildlife camera-trap detection.")
    project_app = typer.Typer(help="Create and inspect portable projects.")
    cvat_app = typer.Typer(help="Exchange annotations with CVAT and compatible tools.")
    dataset_app = typer.Typer(help="Import and merge immutable annotated datasets.")
    checkpoint_app = typer.Typer(help="Register and inspect reusable checkpoints.")
    app.add_typer(project_app, name="project")
    app.add_typer(cvat_app, name="cvat")
    app.add_typer(dataset_app, name="dataset")
    app.add_typer(checkpoint_app, name="checkpoint")

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

    @dataset_app.command("import")
    def dataset_import(
        project_dir: Path,
        archive: Path,
        class_mapping: str = typer.Option("{}", "--class-mapping"),
    ):
        """Import a CVAT, COCO, or YOLO ZIP into an immutable snapshot."""
        try:
            mapping = json.loads(class_mapping)
        except json.JSONDecodeError as exc:
            raise ValueError("--class-mapping must be a JSON object") from exc
        if not isinstance(mapping, dict):
            raise ValueError("--class-mapping must be a JSON object")
        snapshot = ProjectStore(project_dir).import_dataset(archive, class_mapping=mapping)
        typer.echo(json.dumps(snapshot.manifest.to_dict(), indent=2))

    @app.command()
    def train(
        project_dir: Path,
        output_dir: Path = typer.Option(..., "--output-dir"),
        snapshot: Path | None = typer.Option(None, "--snapshot"),
        model_preset: str = typer.Option("yolo26-l", "--model-preset"),
        checkpoint: Path | None = typer.Option(None, "--checkpoint"),
        epochs: int = typer.Option(100, "--epochs"),
        batch_size: int = typer.Option(16, "--batch-size"),
        max_dimension: int = typer.Option(640, "--max-dimension"),
        grayscale: bool = typer.Option(True, "--grayscale/--no-grayscale"),
        clahe: bool = typer.Option(False, "--clahe/--no-clahe"),
        device: str = typer.Option("auto", "--device"),
        resume_from: Path | None = typer.Option(None, "--resume-from"),
    ):
        """Fine-tune a catalog model from an imported immutable dataset snapshot."""
        store = ProjectStore(project_dir)
        project = store.load_manifest()
        selected_snapshot = DatasetSnapshot.load(snapshot) if snapshot else _latest_snapshot(store)
        preset = ModelCatalog().get(model_preset)
        preprocessing = PreprocessingConfig(
            max_dimension=max_dimension,
            grayscale_enabled=grayscale,
            clahe_enabled=clahe,
        )
        parent_manifest = load_checkpoint_manifest(resume_from) if resume_from else None
        if parent_manifest is not None:
            parent_checkpoint = Path(parent_manifest.checkpoint_path)
            if checkpoint is not None and checkpoint.expanduser().resolve() != parent_checkpoint:
                raise ValueError("--checkpoint and --resume-from must refer to the same checkpoint")
            checkpoint = parent_checkpoint
        detector = (
            load_detector(
                checkpoint,
                architecture=preset.architecture,
                classes=project.classes,
                model_id=parent_manifest.model_id if parent_manifest else preset.model_id,
                checkpoint_manifest=parent_manifest,
                preprocessing=preprocessing.to_dict(),
            )
            if checkpoint is not None
            else load_preset_detector(preset, classes=project.classes)
        )
        result = train_snapshot_and_register(
            detector,
            snapshot=selected_snapshot,
            output_dir=output_dir,
            config=TrainingConfig(
                epochs=epochs,
                batch_size=batch_size,
                image_size=project.project_config.image_size,
                patience=project.project_config.patience,
                seed=project.project_config.random_seed,
                device=device,
                preprocessing=preprocessing,
            ),
            preprocessing=preprocessing,
            resume_from=parent_manifest,
        )
        typer.echo(
            json.dumps(
                {
                    "checkpoint": str(result.checkpoint),
                    "checkpoint_manifest": str(Path(output_dir) / "checkpoint.json"),
                    "evaluation": "not evaluated",
                },
                indent=2,
            )
        )

    @app.command("project-create")
    def project_create_legacy(
        project_dir: Path,
        image_root: list[Path] = typer.Option(..., "--image-root"),
        class_name: list[str] = typer.Option(..., "--class-name"),
        name: str = typer.Option("amphilens-project", "--name"),
    ):
        """Backward-compatible alias for `project create`."""
        project_create(project_dir, image_root, class_name, name)

    @checkpoint_app.command("register")
    def checkpoint_register(
        registry_dir: Path,
        checkpoint: Path,
        model_id: str = typer.Option(..., "--model-id"),
        architecture: str = typer.Option(..., "--architecture"),
        class_name: list[str] = typer.Option(..., "--class-name"),
        training_domain: str = typer.Option("unknown", "--training-domain"),
        source: str = typer.Option("user", "--source"),
        license_name: str = typer.Option("unknown", "--license"),
        preprocessing: str = typer.Option("{}", "--preprocessing"),
        model_card: str | None = typer.Option(None, "--model-card"),
    ):
        """Register a checkpoint with hash and compatibility metadata."""
        try:
            preprocessing_config = json.loads(preprocessing)
        except json.JSONDecodeError as exc:
            raise ValueError("--preprocessing must be a JSON object") from exc
        if not isinstance(preprocessing_config, dict):
            raise ValueError("--preprocessing must be a JSON object")
        checkpoint_manifest = CheckpointManifest.create(
            checkpoint,
            model_id=model_id,
            architecture=architecture,
            classes=class_name,
            preprocessing=preprocessing_config,
        )
        model = ModelManifest(
            model_id=model_id,
            architecture=architecture,
            classes=class_name,
            training_domain=training_domain,
            source=source,
            license=license_name,
            preprocessing=preprocessing_config,
            checkpoint_sha256=checkpoint_manifest.sha256,
            model_card=model_card,
        )
        ModelRegistry(registry_dir).register(model, checkpoint_manifest)
        typer.echo(json.dumps(model.to_dict(), indent=2))

    @checkpoint_app.command("list")
    def checkpoint_list(registry_dir: Path):
        """List registered model identifiers."""
        typer.echo(json.dumps(ModelRegistry(registry_dir).list_models(), indent=2))

    @checkpoint_app.command("inspect")
    def checkpoint_inspect(registry_dir: Path, model_id: str):
        """Print registered model and checkpoint metadata."""
        model, checkpoint = ModelRegistry(registry_dir).get(model_id)
        typer.echo(
            json.dumps({"model": model.to_dict(), "checkpoint": checkpoint.to_dict()}, indent=2)
        )

    @app.command("app")
    def app_ui():
        """Launch the local browser application."""
        ui_path = Path(__file__).with_name("ui.py")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)], check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "The UI requires the 'ui' extra: pip install 'amphilens[ui]'"
            ) from exc

    @app.command()
    def images(project_dir: Path):
        """List image files recorded by a project manifest."""
        store = ProjectStore(project_dir)
        manifest = store.load_manifest()
        for path in iter_images(manifest.image_roots):
            typer.echo(path)

    @app.command()
    def predict(
        project_dir: Path,
        checkpoint: Path,
        architecture: str = typer.Option(
            ..., "--architecture", help="yolo, rtdetr, or faster_rcnn"
        ),
        output_dir: Path = typer.Option(..., "--output-dir"),
        confidence: float = typer.Option(0.25, "--confidence"),
        image_size: int = typer.Option(640, "--image-size"),
        device: str = typer.Option("auto", "--device"),
        model_id: str | None = typer.Option(None, "--model-id"),
        run_id: str | None = typer.Option(None, "--run-id"),
        registry_dir: Path | None = typer.Option(None, "--registry-dir"),
        preprocessing: str = typer.Option("{}", "--preprocessing"),
        max_dimension: int = typer.Option(640, "--max-dimension"),
        grayscale: bool = typer.Option(True, "--grayscale/--no-grayscale"),
        clahe: bool = typer.Option(False, "--clahe/--no-clahe"),
    ):
        """Run a compatible detector and persist resumable prediction artifacts."""
        store = ProjectStore(project_dir)
        manifest = store.load_manifest()
        resolved_model_id = model_id or checkpoint.stem
        try:
            preprocessing_config = json.loads(preprocessing)
        except json.JSONDecodeError as exc:
            raise ValueError("--preprocessing must be a JSON object") from exc
        if not isinstance(preprocessing_config, dict):
            raise ValueError("--preprocessing must be a JSON object")
        if not preprocessing_config:
            preprocessing_config = PreprocessingConfig(
                max_dimension=max_dimension,
                grayscale_enabled=grayscale,
                clahe_enabled=clahe,
            ).to_dict()
        checkpoint_manifest = None
        if registry_dir is not None:
            checkpoint_manifest = ModelRegistry(registry_dir).resolve(
                resolved_model_id,
                architecture=architecture,
                classes=manifest.classes,
                preprocessing=preprocessing_config,
            )
        detector = load_detector(
            checkpoint,
            architecture=architecture,
            classes=manifest.classes,
            model_id=resolved_model_id,
            checkpoint_manifest=checkpoint_manifest,
            preprocessing=preprocessing_config,
        )
        config = InferenceConfig(
            model_id=resolved_model_id,
            image_size=image_size,
            confidence=confidence,
            device=device,
            preprocessing=preprocessing_config,
            run_id=run_id or f"predict-{resolved_model_id}",
        )
        summary = run_resumable_inference(
            detector, iter_images(manifest.image_roots), config, output_dir
        )
        _register_run_artifacts(store, summary, output_dir)
        typer.echo(
            json.dumps(
                {
                    "run_id": summary.run_id,
                    "completed_images": summary.completed_images,
                    "failed_images": summary.failed_images,
                    "detection_count": summary.detection_count,
                    "predictions_csv": str(summary.predictions_csv),
                },
                indent=2,
            )
        )

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
        selected = HybridPPALStrategy(config).select(predictions, calibration, features=features)
        artifacts = write_selection_artifacts(selected, calibration, config, output_dir)
        typer.echo(json.dumps({name: str(path) for name, path in artifacts.items()}, indent=2))

    @app.command()
    def report(predictions_csv: Path, output_dir: Path):
        """Write JSON and Markdown summaries for a prediction CSV."""
        artifacts = write_report(predictions_csv, output_dir)
        typer.echo(json.dumps({name: str(path) for name, path in artifacts.items()}, indent=2))

    def _records_from_csv(path: Path) -> list[DetectionRecord]:
        return read_predictions_csv(path)

    def _latest_snapshot(store: ProjectStore) -> DatasetSnapshot:
        candidates = sorted(
            path
            for path in (store.root / "datasets").glob("*")
            if path.is_dir() and (path / "manifest.json").is_file()
        )
        if not candidates:
            raise ValueError("Import an annotated dataset before training")
        return DatasetSnapshot.load(candidates[-1])

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

    def _managed_service(project_dir: Path, server_url: str | None):
        """Build a managed CVAT service without placing the token in arguments or output."""
        transport = CVATSdkTransport(server_url=server_url, token=os.environ.get("CVAT_TOKEN"))
        return ManagedCVATCycleService(
            ProjectStore(project_dir), transport, server_url=transport.server_url
        )

    @cvat_app.command("managed-start")
    def cvat_managed_start(
        project_dir: Path,
        cycle: int = typer.Option(..., "--cycle"),
        image: list[Path] = typer.Option(..., "--image"),
        server_url: str | None = typer.Option(None, "--server-url"),
    ):
        """Create or resume one correctly labelled CVAT task for a queue."""
        paths = [item.expanduser().resolve() for item in image]
        service = _managed_service(project_dir, server_url)
        manifest = service.start(
            cycle=cycle,
            image_paths=paths,
            selection_hash=selection_hash(paths),
        )
        typer.echo(json.dumps(manifest.to_dict(), indent=2))

    @cvat_app.command("managed-status")
    def cvat_managed_status(
        project_dir: Path,
        cycle: int = typer.Option(..., "--cycle"),
        server_url: str | None = typer.Option(None, "--server-url"),
    ):
        """Refresh and print the managed CVAT task status."""
        manifest = _managed_service(project_dir, server_url).refresh(cycle)
        typer.echo(json.dumps(manifest.to_dict(), indent=2))

    @cvat_app.command("managed-continue")
    def cvat_managed_continue(
        project_dir: Path,
        cycle: int = typer.Option(..., "--cycle"),
        server_url: str | None = typer.Option(None, "--server-url"),
    ):
        """Export a completed CVAT task and merge it into a new dataset snapshot."""
        service = _managed_service(project_dir, server_url)
        snapshot = service.continue_cycle(cycle)
        typer.echo(
            json.dumps(
                {
                    "snapshot": str(snapshot.root),
                    "manifest": snapshot.manifest.to_dict(),
                },
                indent=2,
            )
        )

    def main():
        app()
else:

    def main():
        _require_typer()
