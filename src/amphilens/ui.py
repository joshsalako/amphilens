"""Guided local Streamlit workflow over the shared AmphiLens services."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import ModelCatalog
from .preprocessing import PreprocessingConfig


def parse_classes(value: str) -> list[str]:
    classes = [item.strip() for item in re.split(r"[,\n]", value) if item.strip()]
    if not classes:
        raise ValueError("Enter at least one class")
    if len(classes) != len(set(classes)):
        raise ValueError("Class names must be unique")
    return classes


def parse_class_mapping(value: str) -> dict[str, str]:
    try:
        mapping = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Class mapping must be a JSON object") from exc
    if not isinstance(mapping, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in mapping.items()
    ):
        raise ValueError("Class mapping must be a JSON object")
    return mapping


def preprocessing_from_controls(
    *, max_dimension: int, grayscale: bool, clahe: bool
) -> PreprocessingConfig:
    return PreprocessingConfig(
        max_dimension=int(max_dimension),
        grayscale_enabled=bool(grayscale),
        clahe_enabled=bool(clahe),
    )


def _snapshot_paths(project_dir: str | Path) -> list[Path]:
    root = Path(project_dir).expanduser().resolve() / "datasets"
    return sorted(
        path for path in root.glob("*") if path.is_dir() and (path / "manifest.json").is_file()
    )


def _render_environment(st):
    from .doctor import run_doctor

    st.header("Environment")
    report = run_doctor(".").to_dict()
    st.json(report)
    st.caption(
        "CPU inference is supported. A CUDA GPU is recommended for fine-tuning "
        "and large image pools."
    )
    if not report["torch_installed"]:
        st.warning(
            "PyTorch is not installed. Install the inference or training package extra first."
        )


def _render_create_project(st):
    from .core import ProjectConfig, ProjectManifest, ProjectStore

    st.header("1. Create or open a project")
    st.write(
        "Choose the folder with your camera-trap images and describe the animals you want to find."
    )
    name = st.text_input("Project name", value="amphilens-project")
    image_root = st.text_input("Unlabelled image folder")
    class_text = st.text_area("Classes to detect", value="toad\nother_amphibian")
    project_dir = st.text_input("AmphiLens project folder", value="./amphilens-project")
    catalog = ModelCatalog()
    preset_id = st.selectbox("Default model", [item.model_id for item in catalog.list()], index=0)
    max_dimension = st.number_input("Maximum image dimension", min_value=32, value=640, step=32)
    grayscale = st.checkbox("Convert images to grayscale", value=True)
    clahe = st.checkbox("Improve uneven lighting with CLAHE", value=False)
    if st.button("Create project", type="primary"):
        try:
            classes = parse_classes(class_text)
            preprocessing = preprocessing_from_controls(
                max_dimension=int(max_dimension), grayscale=grayscale, clahe=clahe
            )
            config = ProjectConfig(
                classes=classes,
                model_preset=preset_id,
                preprocessing=preprocessing,
            )
            manifest = ProjectManifest.create(name, [image_root], classes, project_config=config)
            ProjectStore(project_dir).create(manifest)
            st.success(f"Created {Path(project_dir).resolve()}")
            st.json(manifest.to_dict())
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))


def _render_import_dataset(st):
    from .core import ProjectStore

    st.header("2. Import your initial annotations")
    st.write(
        "Export a CVAT project with its images, or provide a YOLO ZIP. "
        "AmphiLens keeps the original "
        "archive and creates a new immutable snapshot."
    )
    project_dir = st.text_input("Project folder", value="./amphilens-project", key="import-project")
    archive = st.text_input("CVAT or YOLO ZIP file")
    mapping = st.text_area("Optional class mapping (JSON)", value="{}")
    if st.button("Import initial dataset", type="primary"):
        try:
            snapshot = ProjectStore(project_dir).import_dataset(
                archive, class_mapping=parse_class_mapping(mapping)
            )
            st.success(f"Imported {len(snapshot.manifest.images)} reviewed images")
            st.json(snapshot.manifest.to_dict())
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))


def _render_train(st):
    from .core import ProjectStore
    from .dataset import DatasetSnapshot
    from .models import load_detector, load_preset_detector
    from .training import TrainingConfig, train_snapshot_and_register

    st.header("3. Train or continue a model")
    project_dir = st.text_input("Project folder", value="./amphilens-project", key="train-project")
    try:
        project = ProjectStore(project_dir).load_manifest()
        snapshot_paths = _snapshot_paths(project_dir)
    except Exception as exc:  # noqa: BLE001 - the form remains usable while paths are edited
        st.info(f"Project not loaded: {exc}")
        return
    if not snapshot_paths:
        st.warning("Import an initial CVAT or YOLO dataset before training.")
        return
    snapshot_path = st.selectbox(
        "Labelled dataset snapshot", [str(path) for path in snapshot_paths]
    )
    catalog = ModelCatalog()
    preset = catalog.get(st.selectbox("Model", [item.model_id for item in catalog.list()], index=0))
    checkpoint = st.text_input(
        "Optional local checkpoint (leave blank for official general-purpose weights)"
    )
    output_dir = st.text_input(
        "Training output folder", value=str(Path(project_dir) / "checkpoints" / "cycle-0")
    )
    epochs = st.number_input("Training epochs", min_value=1, value=100, step=1)
    batch_size = st.number_input("Batch size", min_value=1, value=16, step=1)
    max_dimension = st.number_input("Maximum image dimension", min_value=32, value=640, step=32)
    grayscale = st.checkbox("Convert images to grayscale", value=True, key="train-gray")
    clahe = st.checkbox("Use CLAHE", value=False, key="train-clahe")
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], key="train-device")
    if st.button("Train model", type="primary"):
        try:
            preprocessing = preprocessing_from_controls(
                max_dimension=int(max_dimension), grayscale=grayscale, clahe=clahe
            )
            detector = (
                load_detector(
                    checkpoint,
                    architecture=preset.architecture,
                    classes=project.classes,
                    model_id=preset.model_id,
                )
                if checkpoint.strip()
                else load_preset_detector(preset, classes=project.classes)
            )
            result = train_snapshot_and_register(
                detector,
                snapshot=DatasetSnapshot.load(snapshot_path),
                output_dir=output_dir,
                config=TrainingConfig(
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    image_size=project.project_config.image_size,
                    patience=project.project_config.patience,
                    seed=project.project_config.random_seed,
                    device=device,
                    preprocessing=preprocessing,
                ),
                preprocessing=preprocessing,
            )
            st.success(f"Training finished: {result.checkpoint}")
            st.json({"checkpoint": str(result.checkpoint), "evaluation": "not evaluated"})
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))


def _render_predict(st):
    from .core import InferenceConfig, ProjectStore, iter_images
    from .models import load_detector, load_preset_detector
    from .reporting import write_report
    from .runs import run_resumable_inference

    st.header("4. Find animals and download results")
    project_dir = st.text_input(
        "Project folder", value="./amphilens-project", key="predict-project"
    )
    try:
        project = ProjectStore(project_dir).load_manifest()
    except Exception as exc:  # noqa: BLE001 - the form remains usable while paths are edited
        st.info(f"Project not loaded: {exc}")
        return
    image_root = st.text_input("Image folder to scan", value=project.image_roots[0])
    catalog = ModelCatalog()
    preset = catalog.get(
        st.selectbox(
            "Model",
            [item.model_id for item in catalog.list()],
            index=0,
            key="predict-model",
        )
    )
    checkpoint = st.text_input("Checkpoint path (optional for official weights)")
    output_dir = st.text_input(
        "Results folder", value=str(Path(project_dir) / "artifacts" / "prediction")
    )
    confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.25, 0.01)
    image_size = st.number_input("Detector input size", min_value=32, value=640, step=32)
    max_dimension = st.number_input("Maximum image dimension", min_value=32, value=640, step=32)
    grayscale = st.checkbox("Convert images to grayscale", value=True, key="predict-gray")
    clahe = st.checkbox("Use CLAHE", value=False, key="predict-clahe")
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], key="predict-device")
    if st.button("Run detection", type="primary"):
        try:
            preprocessing = preprocessing_from_controls(
                max_dimension=int(max_dimension), grayscale=grayscale, clahe=clahe
            )
            detector = (
                load_detector(
                    checkpoint,
                    architecture=preset.architecture,
                    classes=project.classes,
                    model_id=preset.model_id,
                )
                if checkpoint.strip()
                else load_preset_detector(preset, classes=project.classes)
            )
            config = InferenceConfig(
                model_id=preset.model_id,
                image_size=int(image_size),
                confidence=confidence,
                device=device,
                preprocessing=preprocessing,
                run_id=f"predict-{preset.model_id}",
            )
            summary = run_resumable_inference(
                detector, iter_images([image_root]), config, output_dir
            )
            report = write_report(summary.predictions_csv, Path(output_dir) / "report")
            st.success(
                f"Processed {summary.completed_images} images and "
                f"{summary.detection_count} detections"
            )
            st.json({"csv": str(summary.predictions_csv), "report": str(report["markdown"])})
            st.download_button(
                "Download predictions CSV",
                summary.predictions_csv.read_bytes(),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))


def _render_active_learning(st):
    from .active_learning import HybridPPALConfig, HybridPPALStrategy, PPALCalibration
    from .curation import write_selection_artifacts
    from .inference import read_predictions_csv

    st.header("5. Active learning queue")
    st.write("AmphiLens samples difficult and diverse images for the next annotation cycle.")
    predictions = st.text_input("Prediction CSV")
    calibration = st.text_input("Calibration JSON")
    features = st.text_input("Feature JSON")
    output = st.text_input("Queue output folder", value="./amphilens-project/annotations/cycle-0")
    budget = st.number_input("Images to annotate", min_value=1, value=100, step=1)
    seed = st.number_input("Selection seed", min_value=0, value=42, step=1)
    pool_multiplier = st.number_input("Candidate pool multiplier", min_value=1, value=200, step=1)
    uncertain_ratio = st.slider("Uncertain ratio", 0.0, 1.0, 0.4, 0.05)
    certain_ratio = st.slider("Certain ratio", 0.0, 1.0, 0.5, 0.05)
    random_ratio = st.slider("Random ratio", 0.0, 1.0, 0.1, 0.05)
    priority_class = st.text_input("Priority class (optional)")
    priority_weight = st.number_input("Priority-class weight", min_value=0.0, value=1.0, step=0.1)
    if st.button("Select annotation queue", type="primary"):
        try:
            selected_calibration = PPALCalibration(**json.loads(Path(calibration).read_text()))
            selected_config = HybridPPALConfig(
                budget=int(budget),
                pool_multiplier=int(pool_multiplier),
                uncertain_ratio=uncertain_ratio,
                certain_ratio=certain_ratio,
                random_ratio=random_ratio,
                seed=int(seed),
                priority_class=priority_class.strip() or None,
                priority_weight=priority_weight,
            )
            selected = HybridPPALStrategy(selected_config).select(
                read_predictions_csv(Path(predictions)),
                selected_calibration,
                features=json.loads(Path(features).read_text()),
            )
            artifacts = write_selection_artifacts(
                selected, selected_calibration, selected_config, output
            )
            st.success(f"Selected {len(selected)} images")
            st.json({key: str(value) for key, value in artifacts.items()})
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))


def main():
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("The UI requires the 'ui' extra: pip install 'amphilens[ui]'") from exc

    st.set_page_config(page_title="AmphiLens", page_icon="🐸", layout="wide")
    st.title("AmphiLens")
    st.caption("Find amphibians and other wildlife in camera-trap images")
    page = st.sidebar.radio(
        "Workflow",
        [
            "Environment",
            "Create project",
            "Import initial dataset",
            "Train model",
            "Find animals",
            "Active learning queue",
        ],
    )
    if page == "Environment":
        _render_environment(st)
    elif page == "Create project":
        _render_create_project(st)
    elif page == "Import initial dataset":
        _render_import_dataset(st)
    elif page == "Train model":
        _render_train(st)
    elif page == "Find animals":
        _render_predict(st)
    else:
        _render_active_learning(st)


if __name__ == "__main__":
    main()
