"""Guided local Streamlit workflow over the shared AmphiLens services."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Streamlit executes the file passed to ``streamlit run`` as a script. Ensure
# relative imports still resolve when the CLI points at this source file.
if __package__ in {None, ""}:  # pragma: no cover - exercised by Streamlit
    _source_root = Path(__file__).resolve().parents[1]
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))
    __package__ = "amphilens"

from .configuration import discover_checkpoint_manifest, resolve_effective_configuration
from .locations import (
    ProjectLocationError,
    default_projects_root,
    find_source_checkout,
    open_project,
    relocate_project,
    validate_new_project_path,
)
from .models import ModelCatalog
from .preprocessing import PreprocessingConfig
from .state import UserStateError, UserStateStore


def cvat_project_choices(projects):
    """Build readable, stable Streamlit labels for CVAT project summaries."""
    choices = {}
    for project in projects:
        noun = "task" if project.task_count == 1 else "tasks"
        label = f"{project.name} (ID {project.project_id}; {project.task_count} {noun})"
        choices[label] = project
    return choices


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
    *,
    max_dimension: int,
    grayscale: bool,
    clahe: bool,
    base: PreprocessingConfig | None = None,
) -> PreprocessingConfig:
    reference = base or PreprocessingConfig()
    return PreprocessingConfig(
        resize_enabled=reference.resize_enabled,
        max_dimension=int(max_dimension),
        resize_interpolation=reference.resize_interpolation,
        grayscale_enabled=bool(grayscale),
        clahe_enabled=bool(clahe),
        clahe_clip_limit=reference.clahe_clip_limit,
        clahe_tile_grid_size=reference.clahe_tile_grid_size,
        color_space=reference.color_space,
        compatibility_mode=reference.compatibility_mode,
    )


def project_widget_key(project_dir: str | Path, project_identity: str, widget_name: str) -> str:
    """Return a stable widget key isolated to one project/configuration identity."""
    payload = json.dumps(
        {
            "project_dir": str(Path(project_dir).expanduser().resolve()),
            "project_identity": project_identity,
            "widget": widget_name,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{widget_name}-{digest}"


def _workflow_identity(store, project) -> str:
    return f"{project.name}:{resolve_effective_configuration(project).fingerprint}"


def _workflow_key(store, project, widget_name: str) -> str:
    return project_widget_key(store.root, _workflow_identity(store, project), widget_name)


def _show_effective_configuration(st, effective) -> None:
    st.caption(f"Configuration source: {effective.source}; fingerprint: {effective.fingerprint}")
    for warning in effective.warnings:
        st.warning(warning)
    st.json(effective.to_dict())


def _preprocessing_overrides(st, effective, key_prefix: str) -> dict:
    return {
        "preprocessing": preprocessing_from_controls(
            max_dimension=int(
                st.number_input(
                    "Maximum image dimension",
                    min_value=32,
                    value=effective.preprocessing.max_dimension,
                    step=32,
                    key=f"{key_prefix}-max-dimension",
                )
            ),
            grayscale=st.checkbox(
                "Convert images to grayscale",
                value=effective.preprocessing.grayscale_enabled,
                key=f"{key_prefix}-grayscale",
            ),
            clahe=st.checkbox(
                "Use CLAHE",
                value=effective.preprocessing.clahe_enabled,
                key=f"{key_prefix}-clahe",
            ),
            base=effective.preprocessing,
        )
    }


ACTIVE_PROJECT_KEY = "amphilens-active-project"
USER_STATE_STORE_KEY = "amphilens-user-state-store"
PROJECT_RESTORE_MESSAGE_KEY = "amphilens-project-restore-message"
OPEN_PROJECT_INPUT_KEY = "amphilens-open-project-input"
CREATE_PROJECT_INPUT_KEY = "amphilens-create-project-input"


def active_project_path(state) -> Path | None:
    """Return the resolved project folder stored in a Streamlit-like state mapping."""
    value = state.get(ACTIVE_PROJECT_KEY)
    return Path(value).expanduser().resolve() if value else None


def _user_state_store(state, user_state: UserStateStore | None = None) -> UserStateStore:
    if user_state is not None:
        state[USER_STATE_STORE_KEY] = user_state
        return user_state
    existing = state.get(USER_STATE_STORE_KEY)
    if isinstance(existing, UserStateStore):
        return existing
    created = UserStateStore()
    state[USER_STATE_STORE_KEY] = created
    return created


def restore_active_project(
    state, *, user_state: UserStateStore | None = None
) -> Path | None:
    """Restore and validate the last project once for a Streamlit session."""
    if ACTIVE_PROJECT_KEY in state:
        return active_project_path(state)
    store = _user_state_store(state, user_state)
    try:
        remembered = store.last_active_project()
    except UserStateError as exc:
        store.clear()
        state[PROJECT_RESTORE_MESSAGE_KEY] = str(exc)
        return None
    if remembered is None:
        return None
    try:
        project = open_project(remembered)
    except ProjectLocationError as exc:
        store.clear()
        state[PROJECT_RESTORE_MESSAGE_KEY] = (
            f"The remembered project could not be opened and was forgotten: {exc}"
        )
        return None
    state[ACTIVE_PROJECT_KEY] = str(project.root)
    return project.root


def set_active_project(
    state, project_dir: str | Path, *, user_state: UserStateStore | None = None
):
    """Validate and store the active project, returning its filesystem store."""
    store = open_project(project_dir)
    state[ACTIVE_PROJECT_KEY] = str(store.root)
    _user_state_store(state, user_state).remember_project(store.root)
    state.pop(PROJECT_RESTORE_MESSAGE_KEY, None)
    return store


def clear_active_project(state, *, user_state: UserStateStore | None = None) -> None:
    _user_state_store(state, user_state).clear()
    state.pop(ACTIVE_PROJECT_KEY, None)
    state.pop(PROJECT_RESTORE_MESSAGE_KEY, None)


def _folder_input(st, *, label: str, state_key: str, default: str = "") -> str:
    if state_key not in st.session_state:
        st.session_state[state_key] = default
    return st.text_input(label, key=state_key)


def _active_store(st):
    project_dir = active_project_path(st.session_state)
    if project_dir is None:
        st.info("Create or open a project before using this workflow.")
        return None
    try:
        return open_project(project_dir)
    except ProjectLocationError as exc:
        st.error(str(exc))
        return None


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

    st.header("1. Create a project")
    st.write(
        "Enter your camera-trap image path and describe the animals you want to find. "
        "Project data is stored outside the AmphiLens source folder."
    )
    name = st.text_input("Project name", value="amphilens-project")
    image_root = st.text_input("Unlabelled image folder path")
    class_text = st.text_area("Classes to detect", value="toad\nother_amphibian")
    project_dir = _folder_input(
        st,
        label="Project folder path",
        state_key=CREATE_PROJECT_INPUT_KEY,
        default=str(default_projects_root() / "amphilens-project"),
    )
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
            destination = validate_new_project_path(
                project_dir, source_checkout=find_source_checkout(Path(__file__))
            )
            ProjectStore(destination).create(manifest)
            set_active_project(st.session_state, destination)
            st.success(f"Created {destination}")
            st.json(manifest.to_dict())
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))


def _render_open_project(st):
    st.header("1. Open a project")
    st.write("Enter the path to a folder containing an AmphiLens `manifest.json` file.")
    project_dir = _folder_input(
        st,
        label="Existing project folder path",
        state_key=OPEN_PROJECT_INPUT_KEY,
    )
    if st.button("Open project", type="primary", disabled=not project_dir.strip()):
        try:
            store = set_active_project(st.session_state, project_dir)
            st.success(f"Opened {store.root}")
            st.json(store.load_manifest().to_dict())
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))

    if not project_dir.strip():
        return
    try:
        candidate = open_project(project_dir)
    except ProjectLocationError:
        return
    checkout = find_source_checkout(Path(__file__))
    inside_checkout = bool(
        checkout and (candidate.root == checkout or checkout in candidate.root.parents)
    )
    if not inside_checkout:
        return

    st.warning(
        "This project is inside the AmphiLens source checkout and may appear as Git changes. "
        "Move it to the user-project folder."
    )
    destination = _folder_input(
        st,
        label="Safe destination folder path",
        state_key="amphilens-migrate-project-input",
        default=str(default_projects_root() / candidate.root.name),
    )
    confirm = st.checkbox(
        "I understand that the verified move will remove the original folder.",
        key="amphilens-migrate-project-confirm",
    )
    if st.button("Move and remove original", disabled=not confirm):
        try:
            moved = relocate_project(
                candidate.root,
                destination,
                remove_source=True,
                source_checkout=checkout,
            )
            set_active_project(st.session_state, moved)
            st.success(f"Moved project to {moved}")
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))


def _render_import_dataset(st):
    from .annotations.initial import CVATProjectImportService
    from .annotations.managed import CVATSdkTransport

    st.header("2. Import your initial annotations")
    st.write(
        "Select an existing CVAT project through its API, or provide a local "
        "CVAT/COCO/YOLO ZIP. AmphiLens keeps the source and creates an immutable snapshot."
    )
    store = _active_store(st)
    if store is None:
        return
    source = st.radio(
        "Annotation source",
        ["CVAT project", "Local archive"],
        horizontal=True,
        key="initial-dataset-source",
    )
    mapping = st.text_area("Optional class mapping (JSON)", value="{}")
    if source == "Local archive":
        archive = st.text_input("CVAT, COCO, or YOLO ZIP file")
        if st.button("Import initial dataset", type="primary"):
            try:
                snapshot = store.import_dataset(
                    archive, class_mapping=parse_class_mapping(mapping)
                )
                st.success(f"Imported {len(snapshot.manifest.images)} reviewed images")
                st.json(snapshot.manifest.to_dict())
            except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
                st.error(str(exc))
        return

    server_url = st.text_input(
        "CVAT server URL", value=os.environ.get("CVAT_URL", "http://localhost:8080")
    )
    st.caption("Set CVAT_TOKEN before starting AmphiLens; it is never saved in the project.")
    if "amphilens-initial-cvat-projects" not in st.session_state:
        st.session_state["amphilens-initial-cvat-projects"] = []
    if st.button("Connect to CVAT"):
        try:
            transport = CVATSdkTransport(server_url=server_url or None)
            projects = CVATProjectImportService(
                store, transport
            ).list_projects()
            st.session_state["amphilens-initial-cvat-projects"] = projects
            st.success(f"Found {len(projects)} CVAT projects")
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))

    projects = st.session_state["amphilens-initial-cvat-projects"]
    if not projects:
        st.info("Connect to CVAT to choose the project containing your initial annotations.")
        return
    choices = cvat_project_choices(projects)
    selected = choices[st.selectbox("CVAT project", list(choices))]
    st.write(f"Labels: {', '.join(selected.labels)}")
    st.write(f"Tasks: {selected.task_count}")
    if st.button("Import project", type="primary"):
        try:
            transport = CVATSdkTransport(server_url=server_url or None)
            snapshot = CVATProjectImportService(
                store, transport
            ).import_project(
                selected.project_id,
                class_mapping=parse_class_mapping(mapping),
            )
            st.success(f"Imported {len(snapshot.manifest.images)} reviewed images")
            st.json(snapshot.manifest.to_dict())
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))


def _render_train(st):
    from .dataset import DatasetSnapshot
    from .models import load_detector, load_preset_detector
    from .training import (
        TrainingConfig,
        load_checkpoint_manifest,
        train_snapshot_and_register,
    )

    st.header("3. Train or continue a model")
    store = _active_store(st)
    if store is None:
        return
    project = store.load_manifest()
    snapshot_paths = _snapshot_paths(store.root)
    if not snapshot_paths:
        st.warning("Import an initial CVAT or YOLO dataset before training.")
        return
    identity = _workflow_identity(store, project)
    snapshot_path = st.selectbox(
        "Labelled dataset snapshot",
        [str(path) for path in snapshot_paths],
        key=project_widget_key(store.root, identity, "train-snapshot"),
    )
    checkpoint = st.text_input(
        "Optional local checkpoint (leave blank for official general-purpose weights)",
        key=project_widget_key(store.root, identity, "train-checkpoint"),
    )
    resume_manifest_path = st.text_input(
        "Optional parent checkpoint manifest (for continuing a cycle)",
        value="",
        key=project_widget_key(store.root, identity, "train-resume-manifest"),
    )
    output_dir = st.text_input(
        "Training output folder",
        value=str(store.root / "checkpoints" / "cycle-0"),
        key=project_widget_key(store.root, identity, "train-output"),
    )
    parent_manifest = None
    if resume_manifest_path.strip():
        try:
            parent_manifest = load_checkpoint_manifest(resume_manifest_path)
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))
            return
    selected_checkpoint = checkpoint.strip() or (
        parent_manifest.checkpoint_path if parent_manifest else ""
    )
    try:
        checkpoint_manifest = parent_manifest or (
            discover_checkpoint_manifest(selected_checkpoint) if selected_checkpoint else None
        )
        base_effective = resolve_effective_configuration(
            project,
            checkpoint_manifest=checkpoint_manifest,
            checkpoint_path=selected_checkpoint or None,
        )
    except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
        st.error(str(exc))
        return

    advanced = st.checkbox(
        "Advanced run overrides",
        value=False,
        key=project_widget_key(store.root, identity, "train-advanced"),
    )
    overrides = {}
    if advanced:
        if checkpoint_manifest is None:
            catalog = ModelCatalog()
            model_options = [item.model_id for item in catalog.list()]
            model_default = (
                model_options.index(base_effective.model_preset)
                if base_effective.model_preset in model_options
                else 0
            )
            overrides["model_preset"] = st.selectbox(
                "Model",
                model_options,
                index=model_default,
                key=project_widget_key(store.root, identity, "train-model"),
            )
        else:
            st.info(
                f"Checkpoint settings are locked: {base_effective.model_id} / "
                f"{base_effective.architecture}"
            )
        if checkpoint_manifest is None:
            overrides.update(_preprocessing_overrides(st, base_effective, "train"))
            overrides["image_size"] = st.number_input(
                "Detector input size",
                min_value=32,
                value=base_effective.image_size,
                step=32,
                key=project_widget_key(store.root, identity, "train-image-size"),
            )
        else:
            st.write(
                "Preprocessing and detector input size are locked to the checkpoint: "
                f"{base_effective.preprocessing.fingerprint}, {base_effective.image_size}px"
            )
        overrides["epochs"] = st.number_input(
            "Training epochs",
            min_value=1,
            value=base_effective.epochs,
            step=1,
            key=project_widget_key(store.root, identity, "train-epochs"),
        )
        overrides["batch_size"] = st.number_input(
            "Batch size",
            min_value=1,
            value=base_effective.batch_size,
            step=1,
            key=project_widget_key(store.root, identity, "train-batch-size"),
        )
        overrides["device"] = st.selectbox(
            "Device",
            ["auto", "cpu", "cuda"],
            index=["auto", "cpu", "cuda"].index(base_effective.device)
            if base_effective.device in {"auto", "cpu", "cuda"}
            else 0,
            key=project_widget_key(store.root, identity, "train-device"),
        )
    try:
        effective = resolve_effective_configuration(
            project,
            checkpoint_manifest=checkpoint_manifest,
            checkpoint_path=selected_checkpoint or None,
            overrides=overrides,
        )
    except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
        st.error(str(exc))
        return
    _show_effective_configuration(st, effective)
    if st.button("Train model", type="primary"):
        try:
            detector = (
                load_detector(
                    selected_checkpoint,
                    architecture=effective.architecture,
                    classes=list(effective.classes),
                    model_id=effective.model_id,
                    checkpoint_manifest=checkpoint_manifest,
                    preprocessing=effective.preprocessing.to_dict(),
                )
                if selected_checkpoint
                else load_preset_detector(
                    ModelCatalog().get(effective.model_preset), classes=list(effective.classes)
                )
            )
            result = train_snapshot_and_register(
                detector,
                snapshot=DatasetSnapshot.load(snapshot_path),
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
            st.success(f"Training finished: {result.checkpoint}")
            st.json({"checkpoint": str(result.checkpoint), "evaluation": "not evaluated"})
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))


def _render_predict(st):
    from .core import InferenceConfig, iter_images
    from .models import load_detector, load_preset_detector
    from .reporting import write_report
    from .runs import run_resumable_inference

    st.header("4. Find animals and download results")
    store = _active_store(st)
    if store is None:
        return
    project = store.load_manifest()
    identity = _workflow_identity(store, project)
    image_root = st.text_input(
        "Image folder to scan",
        value=project.image_roots[0],
        key=project_widget_key(store.root, identity, "predict-image-root"),
    )
    checkpoint = st.text_input(
        "Checkpoint path (optional for official weights)",
        key=project_widget_key(store.root, identity, "predict-checkpoint"),
    )
    output_dir = st.text_input(
        "Results folder",
        value=str(store.root / "artifacts" / "prediction"),
        key=project_widget_key(store.root, identity, "predict-output"),
    )
    try:
        checkpoint_manifest = (
            discover_checkpoint_manifest(checkpoint) if checkpoint.strip() else None
        )
        base_effective = resolve_effective_configuration(
            project,
            checkpoint_manifest=checkpoint_manifest,
            checkpoint_path=checkpoint.strip() or None,
        )
    except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
        st.error(str(exc))
        return
    advanced = st.checkbox(
        "Advanced run overrides",
        value=False,
        key=project_widget_key(store.root, identity, "predict-advanced"),
    )
    overrides = {}
    if advanced:
        if checkpoint_manifest is None:
            catalog = ModelCatalog()
            model_options = [item.model_id for item in catalog.list()]
            model_default = (
                model_options.index(base_effective.model_preset)
                if base_effective.model_preset in model_options
                else 0
            )
            overrides["model_preset"] = st.selectbox(
                "Model",
                model_options,
                index=model_default,
                key=project_widget_key(store.root, identity, "predict-model"),
            )
        else:
            st.info(
                f"Checkpoint settings are locked: {base_effective.model_id} / "
                f"{base_effective.architecture}"
            )
        if checkpoint_manifest is None:
            overrides.update(_preprocessing_overrides(st, base_effective, "predict"))
            overrides["image_size"] = st.number_input(
                "Detector input size",
                min_value=32,
                value=base_effective.image_size,
                step=32,
                key=project_widget_key(store.root, identity, "predict-image-size"),
            )
        else:
            st.write(
                "Preprocessing and detector input size are locked to the checkpoint: "
                f"{base_effective.preprocessing.fingerprint}, {base_effective.image_size}px"
            )
        overrides["confidence"] = st.slider(
            "Minimum confidence",
            0.0,
            1.0,
            base_effective.confidence,
            0.01,
            key=project_widget_key(store.root, identity, "predict-confidence"),
        )
        overrides["device"] = st.selectbox(
            "Device",
            ["auto", "cpu", "cuda"],
            index=["auto", "cpu", "cuda"].index(base_effective.device)
            if base_effective.device in {"auto", "cpu", "cuda"}
            else 0,
            key=project_widget_key(store.root, identity, "predict-device"),
        )
    try:
        effective = resolve_effective_configuration(
            project,
            checkpoint_manifest=checkpoint_manifest,
            checkpoint_path=checkpoint.strip() or None,
            overrides=overrides,
        )
    except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
        st.error(str(exc))
        return
    _show_effective_configuration(st, effective)
    if st.button("Run detection", type="primary"):
        try:
            detector = (
                load_detector(
                    checkpoint,
                    architecture=effective.architecture,
                    classes=list(effective.classes),
                    model_id=effective.model_id,
                    checkpoint_manifest=checkpoint_manifest,
                    preprocessing=effective.preprocessing.to_dict(),
                )
                if checkpoint.strip()
                else load_preset_detector(
                    ModelCatalog().get(effective.model_preset), classes=list(effective.classes)
                )
            )
            config = InferenceConfig(
                model_id=effective.model_id,
                image_size=effective.image_size,
                confidence=effective.confidence,
                device=effective.device,
                preprocessing=effective.preprocessing,
                run_id=f"predict-{effective.model_id}",
                metadata={"effective_configuration": effective.to_dict()},
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
    store = _active_store(st)
    if store is None:
        return
    predictions = st.text_input("Prediction CSV")
    calibration = st.text_input("Calibration JSON")
    features = st.text_input("Feature JSON")
    output = st.text_input(
        "Queue output folder path", value=str(store.root / "annotations" / "cycle-0")
    )
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


def _read_queue_paths(queue_or_folder: str) -> list[Path]:
    source = Path(queue_or_folder).expanduser().resolve()
    if source.is_dir():
        from .core import iter_images

        return iter_images([source])
    if source.is_file() and source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            if not rows.fieldnames or "image_path" not in rows.fieldnames:
                raise ValueError("Selection CSV must contain an image_path column")
            return [Path(row["image_path"]).expanduser().resolve() for row in rows]
    raise ValueError("Choose an image folder or selection_queue.csv")


def _render_managed_cvat(st):
    from .annotations.managed import CVATSdkTransport, ManagedCVATCycleService, selection_hash

    st.header("6. Send a queue to CVAT")
    st.write(
        "AmphiLens can create the CVAT project and task for you. Annotate there, "
        "save your work, then return here and press Continue."
    )
    store = _active_store(st)
    if store is None:
        return
    server_url = st.text_input(
        "CVAT server URL", value=os.environ.get("CVAT_URL", "http://localhost:8080")
    )
    st.caption(
        "Set CVAT_TOKEN in the terminal before starting the app; it is never saved in the project."
    )
    cycle = st.number_input("Active-learning cycle", min_value=0, value=0, step=1)
    queue = st.text_input(
        "Selection queue CSV or folder",
        value=str(store.root / "annotations" / "cycle-0" / "selection_queue.csv"),
    )
    try:
        paths = _read_queue_paths(queue)
        st.info(f"Selected images: {len(paths)}")
    except Exception as exc:  # noqa: BLE001 - shown as an actionable UI message
        paths = []
        st.warning(str(exc))

    def service():
        transport = CVATSdkTransport(server_url=server_url or None)
        return ManagedCVATCycleService(
            store, transport, server_url=transport.server_url
        )

    start, refresh, continue_button = st.columns(3)
    if start.button("Send to CVAT", type="primary", disabled=not paths):
        try:
            manifest = service().start(
                cycle=int(cycle), image_paths=paths, selection_hash=selection_hash(paths)
            )
            st.success("CVAT task is ready for annotation")
            st.json(manifest.to_dict())
            if manifest.task_url:
                st.link_button("Open CVAT", manifest.task_url)
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))
    if refresh.button("Refresh status"):
        try:
            manifest = service().refresh(int(cycle))
            st.info(f"Status: {manifest.state}; annotations: {manifest.annotation_count}")
            st.write(f"Selected images: {len(manifest.selected_images)}")
            if manifest.task_url:
                st.link_button("Open CVAT", manifest.task_url)
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))
    if continue_button.button("Continue cycle"):
        try:
            snapshot = service().continue_cycle(int(cycle))
            st.success("CVAT annotations imported into a new immutable dataset snapshot")
            st.json(snapshot.manifest.to_dict())
        except Exception as exc:  # noqa: BLE001 - shown as an actionable UI error
            st.error(str(exc))


def _render_active_project_sidebar(st):
    restore_message = st.session_state.pop(PROJECT_RESTORE_MESSAGE_KEY, None)
    if restore_message:
        st.sidebar.warning(restore_message)
    project_dir = active_project_path(st.session_state)
    st.sidebar.subheader("Active project")
    if project_dir is None:
        st.sidebar.info("Create or open a project to begin.")
        return
    st.sidebar.code(str(project_dir), language=None)
    if st.sidebar.button("Close active project"):
        clear_active_project(st.session_state)
        st.sidebar.success("Project closed")


def main():
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("The UI requires the 'ui' extra: pip install 'amphilens[ui]'") from exc

    st.set_page_config(page_title="AmphiLens", page_icon="🐸", layout="wide")
    st.title("AmphiLens")
    st.caption("Find amphibians and other wildlife in camera-trap images")
    restore_active_project(st.session_state)
    _render_active_project_sidebar(st)
    page = st.sidebar.radio(
        "Workflow",
        [
            "Environment",
            "Create project",
            "Open project",
            "Import initial dataset",
            "Train model",
            "Find animals",
            "Active learning queue",
            "CVAT cycle",
        ],
    )
    if page == "Environment":
        _render_environment(st)
    elif page == "Create project":
        _render_create_project(st)
    elif page == "Open project":
        _render_open_project(st)
    elif page == "Import initial dataset":
        _render_import_dataset(st)
    elif page == "Train model":
        _render_train(st)
    elif page == "Find animals":
        _render_predict(st)
    elif page == "Active learning queue":
        _render_active_learning(st)
    else:
        _render_managed_cvat(st)


if __name__ == "__main__":
    main()
