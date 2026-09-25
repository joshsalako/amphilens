"""Guided local Streamlit workflow over the shared AmphiLens services."""

from __future__ import annotations

import json
import re
from pathlib import Path


def parse_classes(value: str) -> list[str]:
    classes = [item.strip() for item in re.split(r"[,\n]", value) if item.strip()]
    if not classes:
        raise ValueError("Enter at least one class")
    if len(classes) != len(set(classes)):
        raise ValueError("Class names must be unique")
    return classes


def _render_environment(st):
    from .doctor import run_doctor

    st.header("Environment")
    st.json(run_doctor(".").to_dict())
    st.caption(
        "CPU inference is supported. A CUDA GPU is recommended for fine-tuning and large pools."
    )


def _render_create_project(st):
    from .core import ProjectManifest, ProjectStore

    st.header("Create project")
    name = st.text_input("Project name", value="amphilens-project")
    image_root = st.text_input("Image folder")
    class_text = st.text_area("Classes", value="toad\nother_amphibian")
    project_dir = st.text_input("Project output folder", value="./amphilens-project")
    if st.button("Create project", type="primary"):
        try:
            classes = parse_classes(class_text)
            manifest = ProjectManifest.create(name, [image_root], classes)
            ProjectStore(project_dir).create(manifest)
            st.success(f"Created {Path(project_dir).resolve()}")
            st.json(manifest.to_dict())
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))


def _render_project(st):
    from .core import InferenceConfig, ProjectStore, iter_images
    from .models import load_detector
    from .reporting import write_report
    from .runs import run_resumable_inference

    st.header("Run prediction")
    project_dir = st.text_input("Project folder", value="./amphilens-project")
    try:
        manifest = ProjectStore(project_dir).load_manifest()
    except Exception as exc:  # noqa: BLE001 - the form remains usable while paths are edited
        st.info(f"Project not loaded: {exc}")
        return
    st.write(f"Classes: {', '.join(manifest.classes)}")
    checkpoint = st.text_input("Checkpoint path")
    architecture = st.selectbox("Architecture", ["yolo", "rtdetr", "faster_rcnn"])
    output_dir = st.text_input(
        "Output folder", value=str(Path(project_dir) / "artifacts" / "prediction")
    )
    confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    image_size = st.number_input("Image size", min_value=32, value=640, step=32)
    device = st.selectbox("Device", ["auto", "cpu", "cuda"])
    preprocessing = st.text_input("Preprocessing label", value="none")
    run_id = st.text_input("Run ID", value=f"predict-{Path(checkpoint).stem or 'model'}")
    if st.button("Run prediction", type="primary"):
        try:
            model_id = Path(checkpoint).stem
            detector = load_detector(
                checkpoint, architecture=architecture, classes=manifest.classes, model_id=model_id
            )
            config = InferenceConfig(
                model_id=model_id,
                image_size=int(image_size),
                confidence=confidence,
                preprocessing=preprocessing.strip() or "none",
                device=device,
                run_id=run_id.strip() or f"predict-{model_id}",
            )
            summary = run_resumable_inference(
                detector, iter_images(manifest.image_roots), config, output_dir
            )
            report = write_report(summary.predictions_csv, Path(output_dir) / "report")
            st.success(
                f"Processed {summary.completed_images} images and "
                f"{summary.detection_count} detections"
            )
            st.json({"summary": str(summary.summary_json), "report": str(report["markdown"])})
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))


def _render_active_learning(st):
    from .active_learning import HybridPPALConfig, HybridPPALStrategy, PPALCalibration
    from .curation import write_selection_artifacts
    from .inference import read_predictions_csv

    st.header("Hybrid PPAL queue")
    predictions = st.text_input("Prediction CSV")
    calibration = st.text_input("Calibration JSON")
    features = st.text_input("Feature JSON")
    output = st.text_input("Queue output folder", value="./amphilens-project/annotations/cycle-0")
    budget = st.number_input("Annotation budget", min_value=1, value=100, step=1)
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
    st.caption("Local-first, reproducible amphibian and wildlife camera-trap detection")
    page = st.sidebar.radio(
        "Workflow", ["Environment", "Create project", "Run prediction", "Hybrid PPAL queue"]
    )
    if page == "Environment":
        _render_environment(st)
    elif page == "Create project":
        _render_create_project(st)
    elif page == "Run prediction":
        _render_project(st)
    else:
        _render_active_learning(st)


if __name__ == "__main__":
    main()
