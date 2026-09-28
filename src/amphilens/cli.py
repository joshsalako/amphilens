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
from .configuration import discover_checkpoint_manifest, resolve_effective_configuration
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
from .locations import (
    default_projects_root,
    find_source_checkout,
    relocate_project,
    validate_new_project_path,
)
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
        destination = validate_new_project_path(
            project_dir, source_checkout=find_source_checkout(Path(__file__))
        )
        ProjectStore(destination).create(manifest)
        typer.echo(f"Created project at {destination}")

    @project_app.command("default-location")
    def project_default_location():
        """Print the default user directory for AmphiLens projects."""
        typer.echo(default_projects_root())

    @project_app.command("move")
    def project_move(
        source_dir: Path,
        destination_dir: Path,
        remove_source: bool = typer.Option(
            False,
            "--remove-source",
            help="Remove the original only after the copied project is verified.",
        ),
    ):
        """Verify and relocate a portable AmphiLens project."""
        moved = relocate_project(
            source_dir,
            destination_dir,
            remove_source=remove_source,
            source_checkout=find_source_checkout(Path(__file__)),
        )
        typer.echo(f"Moved project to {moved}")

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

    @dataset_app.command("import-cvat")
    def dataset_import_cvat(
        project_dir: Path,
        project_id: str = typer.Option(..., "--project-id"),
        class_mapping: str = typer.Option("{}", "--class-mapping"),
        server_url: str | None = typer.Option(None, "--server-url"),
    ):
        """Import a complete existing CVAT project through the CVAT API."""
        try:
            mapping = json.loads(class_mapping)
        except json.JSONDecodeError as exc:
            raise ValueError("--class-mapping must be a JSON object") from exc
        if not isinstance(mapping, dict):
            raise ValueError("--class-mapping must be a JSON object")
        snapshot = ProjectStore(project_dir).import_cvat_project(
            project_id,
            class_mapping=mapping,
            server_url=server_url,
        )
        typer.echo(json.dumps(snapshot.manifest.to_dict(), indent=2))

    @app.command()
    def train(
        project_dir: Path,
        output_dir: Path = typer.Option(..., "--output-dir"),
        snapshot: Path | None = typer.Option(None, "--snapshot"),
        model_preset: str | None = typer.Option(None, "--model-preset"),
        checkpoint: Path | None = typer.Option(None, "--checkpoint"),
        epochs: int | None = typer.Option(None, "--epochs"),
        batch_size: int | None = typer.Option(None, "--batch-size"),
        max_dimension: int | None = typer.Option(None, "--max-dimension"),
        grayscale: bool | None = typer.Option(None, "--grayscale/--no-grayscale"),
        clahe: bool | None = typer.Option(None, "--clahe/--no-clahe"),
        device: str | None = typer.Option(None, "--device"),
        resume_from: Path | None = typer.Option(None, "--resume-from"),
    ):
        """Fine-tune a catalog model from an imported immutable dataset snapshot."""
        store = ProjectStore(project_dir)
        project = store.load_manifest()
        selected_snapshot = DatasetSnapshot.load(snapshot) if snapshot else _latest_snapshot(store)
        parent_manifest = load_checkpoint_manifest(resume_from) if resume_from else None
        if parent_manifest is not None:
            parent_checkpoint = Path(parent_manifest.checkpoint_path)
            if checkpoint is not None and checkpoint.expanduser().resolve() != parent_checkpoint:
                raise ValueError("--checkpoint and --resume-from must refer to the same checkpoint")
            checkpoint = parent_checkpoint
        checkpoint_manifest = parent_manifest or (
            discover_checkpoint_manifest(checkpoint) if checkpoint is not None else None
        )
        overrides = {}
        if model_preset is not None:
            overrides["model_preset"] = model_preset
        if epochs is not None:
            overrides["epochs"] = epochs
        if batch_size is not None:
            overrides["batch_size"] = batch_size
        if device is not None:
            overrides["device"] = device
        if max_dimension is not None or grayscale is not None or clahe is not None:
            project_preprocessing = project.project_config.preprocessing
            overrides["preprocessing"] = PreprocessingConfig(
                max_dimension=(
                    max_dimension
                    if max_dimension is not None
                    else project_preprocessing.max_dimension
                ),
                resize_enabled=project_preprocessing.resize_enabled,
                resize_interpolation=project_preprocessing.resize_interpolation,
                grayscale_enabled=(
                    grayscale
                    if grayscale is not None
                    else project_preprocessing.grayscale_enabled
                ),
                clahe_enabled=(
                    clahe if clahe is not None else project_preprocessing.clahe_enabled
                ),
                clahe_clip_limit=project_preprocessing.clahe_clip_limit,
                clahe_tile_grid_size=project_preprocessing.clahe_tile_grid_size,
                color_space=project_preprocessing.color_space,
                compatibility_mode=project_preprocessing.compatibility_mode,
            )
        effective = resolve_effective_configuration(
            project,
            checkpoint_manifest=checkpoint_manifest,
            checkpoint_path=checkpoint,
            overrides=overrides,
        )
        detector = (
            load_detector(
                checkpoint,
                architecture=effective.architecture,
                classes=list(effective.classes),
                model_id=effective.model_id,
                checkpoint_manifest=checkpoint_manifest,
                preprocessing=effective.preprocessing.to_dict(),
            )
            if checkpoint is not None
            else load_preset_detector(
                ModelCatalog().get(effective.model_preset), classes=list(effective.classes)
            )
        )
        result = train_snapshot_and_register(
            detector,
            snapshot=selected_snapshot,
            output_dir=output_dir,
            config=TrainingConfig(
                epochs=effective.epochs,
                batch_size=effective.batch_size,
                image_size=effective.image_size,
                patience=effective.patience,
                seed=effective.seed,
                device=effective.device,
                freeze_strategy=effective.freeze_strategy,
                preprocessing=effective.preprocessing,
                metadata={"effective_configuration": effective.to_dict()},
            ),
            preprocessing=effective.preprocessing,
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
        architecture: str | None = typer.Option(
            None, "--architecture", help="yolo, rtdetr, or faster_rcnn"
        ),
        output_dir: Path = typer.Option(..., "--output-dir"),
        confidence: float | None = typer.Option(None, "--confidence"),
        image_size: int | None = typer.Option(None, "--image-size"),
        device: str | None = typer.Option(None, "--device"),
        model_id: str | None = typer.Option(None, "--model-id"),
        run_id: str | None = typer.Option(None, "--run-id"),
        registry_dir: Path | None = typer.Option(None, "--registry-dir"),
        preprocessing: str | None = typer.Option(None, "--preprocessing"),
        max_dimension: int | None = typer.Option(None, "--max-dimension"),
        grayscale: bool | None = typer.Option(None, "--grayscale/--no-grayscale"),
        clahe: bool | None = typer.Option(None, "--clahe/--no-clahe"),
    ):
        """Run a compatible detector and persist resumable prediction artifacts."""
        store = ProjectStore(project_dir)
        manifest = store.load_manifest()
        try:
            preprocessing_override = json.loads(preprocessing) if preprocessing else None
        except json.JSONDecodeError as exc:
            raise ValueError("--preprocessing must be a JSON object") from exc
        if preprocessing_override is not None and not isinstance(preprocessing_override, dict):
            raise ValueError("--preprocessing must be a JSON object")
        overrides = {}
        if architecture is not None:
            overrides["architecture"] = architecture
        if model_id is not None:
            overrides["model_id"] = model_id
        if confidence is not None:
            overrides["confidence"] = confidence
        if image_size is not None:
            overrides["image_size"] = image_size
        if device is not None:
            overrides["device"] = device
        if preprocessing_override is not None:
            overrides["preprocessing"] = preprocessing_override
        elif max_dimension is not None or grayscale is not None or clahe is not None:
            project_preprocessing = manifest.project_config.preprocessing
            overrides["preprocessing"] = PreprocessingConfig(
                max_dimension=(
                    max_dimension
                    if max_dimension is not None
                    else project_preprocessing.max_dimension
                ),
                resize_enabled=project_preprocessing.resize_enabled,
                resize_interpolation=project_preprocessing.resize_interpolation,
                grayscale_enabled=(
                    grayscale
                    if grayscale is not None
                    else project_preprocessing.grayscale_enabled
                ),
                clahe_enabled=(
                    clahe if clahe is not None else project_preprocessing.clahe_enabled
                ),
                clahe_clip_limit=project_preprocessing.clahe_clip_limit,
                clahe_tile_grid_size=project_preprocessing.clahe_tile_grid_size,
                color_space=project_preprocessing.color_space,
                compatibility_mode=project_preprocessing.compatibility_mode,
            )
        checkpoint_manifest = discover_checkpoint_manifest(checkpoint)
        effective = resolve_effective_configuration(
            manifest,
            checkpoint_manifest=checkpoint_manifest,
            checkpoint_path=checkpoint,
            overrides=overrides,
        )
        resolved_model_id = effective.model_id
        if registry_dir is not None:
            checkpoint_manifest = ModelRegistry(registry_dir).resolve(
                resolved_model_id,
                architecture=effective.architecture,
                classes=manifest.classes,
                preprocessing=effective.preprocessing.to_dict(),
            )
            effective = resolve_effective_configuration(
                manifest,
                checkpoint_manifest=checkpoint_manifest,
                checkpoint_path=checkpoint,
                overrides=overrides,
            )
        detector = load_detector(
            checkpoint,
            architecture=effective.architecture,
            classes=list(effective.classes),
            model_id=resolved_model_id,
            checkpoint_manifest=checkpoint_manifest,
            preprocessing=effective.preprocessing.to_dict(),
        )
        config = InferenceConfig(
            model_id=resolved_model_id,
            image_size=effective.image_size,
            confidence=effective.confidence,
            device=effective.device,
            preprocessing=effective.preprocessing,
            run_id=run_id or f"predict-{resolved_model_id}",
            metadata={"effective_configuration": effective.to_dict()},
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

    @cvat_app.command("projects")
    def cvat_projects(server_url: str | None = typer.Option(None, "--server-url")):
        """List accessible CVAT projects and their task metadata."""
        transport = CVATSdkTransport(server_url=server_url, token=os.environ.get("CVAT_TOKEN"))
        typer.echo(
            json.dumps([project.to_dict() for project in transport.list_projects()], indent=2)
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
