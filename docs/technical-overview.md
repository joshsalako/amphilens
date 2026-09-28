# AmphiLens technical overview

This page contains implementation and reproducibility details that are not
needed for a first-time user. Start with the [README](../README.md) if you
only want to run the browser app.

## 1. Current architecture

AmphiLens is a local-first Python package with a Streamlit browser interface
and a Typer command-line interface. Both use the same services:

The complete user behavior is defined in the
[canonical workflow specification](superpowers/specs/2026-09-28-amphilens-user-workflows.md).

- `amphilens.core` stores project, run, model, checkpoint, and detection
  manifests.
- `amphilens.models` provides explicit YOLO26-L, RT-DETR-L, and Faster R-CNN
  ResNet-50 presets plus detector adapters.
- `amphilens.dataset` validates CVAT XML, COCO, and YOLO archives and creates
  immutable snapshots that can be prepared for training.
- `amphilens.preprocessing` owns the shared resize, grayscale, CLAHE, cache,
  and coordinate-mapping behavior.
- `amphilens.inference` runs predictions, writes stable CSV files, and creates
  visual evidence.
- `amphilens.active_learning` implements the default Hybrid PPAL selection
  workflow.
- `amphilens.annotations` provides portable CVAT/COCO/YOLO exchange and the
  optional managed CVAT SDK transport.

The package keeps source images untouched. Derived files are written to the
project's artifact directories and retain mappings back to their source files.

## 2. Project location and lifecycle

`ProjectStore` treats a project folder as a portable unit containing
`manifest.json`, immutable dataset snapshots, run records, artifacts, and
checkpoints. The browser app does not place this unit in the source checkout:
new projects default to the platform user-data directory, and a new location
inside the checkout is rejected.

The Streamlit session stores one resolved active project path. **Create
project** creates it, **Open project** validates it from `manifest.json`, and
all import, training, prediction, and CVAT pages use that active store. The
CLI remains path-explicit for reproducibility and provides `project move` for
verified relocation.

Relocation copies the complete project to a new destination, compares file
hashes, reloads the copied manifest, and only removes the original when the
user explicitly requests it. Absolute source-image paths and provenance are
preserved because source images are not moved.

## 3. Dataset import and preprocessing

An initial labelled dataset is optional for prediction but required for
fine-tuning. Supported archives must include image files. The importer checks
archive safety, image readability, dimensions, duplicate content, classes, and
bounding boxes, and retains zero-box reviewed images as negatives.

The preprocessing contract is:

```text
source image -> max width/height cap -> grayscale (optional) -> CLAHE (optional)
             -> three-channel model input
```

The source image is never overwritten. Detector boxes are mapped back to the
original image dimensions before they are written to CSV. The exact structured
configuration and fingerprint are recorded in project, run, dataset, and
checkpoint metadata.

## 4. Active-learning method

The default sequence is:

```text
validation calibration -> PPAL class difficulty -> DCUS uncertainty -> CCMS diversity
```

Calibration evidence is required by default. The selected classes are covered
before the remaining budget is filled with uncertainty and feature-based
diversity. Advanced settings include the annotation budget, seed, confidence
threshold, image size, priority class, and sampling ratios.

The implementation follows the method in the
[AmphiLens research reference](https://openreview.net/pdf?id=0YnE65NGna),
while keeping paths, classes, checkpoints, and configuration independent of a
single wildlife species or dataset.

## 5. Command-line workflows

The browser app is the recommended interface. The CLI is useful for repeatable
or unattended runs:

```bash
amphilens doctor
amphilens app
amphilens project default-location
amphilens project create /path/to/my-project \
  --image-root /path/to/camera-trap-images \
  --class-name toad \
  --name tunnel-study
amphilens project inspect /path/to/my-project
amphilens images /path/to/my-project
```

Run prediction with a registered compatible checkpoint:

```bash
amphilens predict ./my-project /path/to/model.pt \
  --architecture yolo \
  --output-dir ./my-project/artifacts/predict-yolo \
  --registry-dir ./model-registry \
  --model-id wlt-yolo-v1 \
  --preprocessing '{"name":"none"}'
```

Export and import a portable CVAT review bundle:

```bash
amphilens cvat export \
  ./my-project/artifacts/predict-yolo/predictions.csv \
  ./my-project/annotations/cycle-0 \
  --class-name toad
amphilens cvat import \
  ./my-project/annotations/cycle-0 \
  ./my-project/artifacts/cycle-0-annotations.csv
```

Create a prediction report:

```bash
amphilens report \
  ./my-project/artifacts/predict-yolo/predictions.csv \
  ./my-project/artifacts/predict-yolo/report
```

See the CLI help for the complete command set.

Import an initial annotated CVAT project through the API and train from the
newest snapshot:

```bash
export CVAT_URL=http://localhost:8080
export CVAT_TOKEN='read-from-your-secret-store'
amphilens cvat projects
amphilens dataset import-cvat ./my-project --project-id 17
amphilens train ./my-project --output-dir ./my-project/checkpoints/cycle-0 \
  --model-preset yolo26-l --max-dimension 640
```

The local archive fallback remains available:

```bash
amphilens dataset import ./my-project /path/to/initial-cvat-or-yolo.zip
```

## 6. CVAT exchange and managed integration

The current portable exchange writes:

- a COCO `annotations.json` file;
- review images with deterministic flat filenames;
- a manifest mapping each review image to its original source path;
- `classes.txt`; and
- YOLO labels and `dataset.yaml` when YOLO exchange is requested.

The initial CVAT import workflow uses the pinned SDK to list accessible
projects, preview project labels and tasks, and export the selected complete
project with images. The export is passed through `DatasetImporter` and becomes
an immutable snapshot. Its provenance records the CVAT server URL, project ID
and name, task IDs, export format, SDK version, and archive hash. The token is
never recorded. Whole-project import is intentional in v1; selecting one task
or reusing local images without downloading is not supported yet.

The managed workflow uses the official `cvat-sdk==2.76.0` profile. It creates
or reuses one CVAT project per AmphiLens project, creates one task per
active-learning cycle, uploads the selected local images with the exact ordered
project label schema, and persists the CVAT IDs, task URL, selected source
paths, selection hash, and client version. Repeating **Send to CVAT** for the
same cycle is idempotent; a different selection is rejected rather than
creating a second task.

Set `CVAT_URL` and `CVAT_TOKEN` in the process environment. The token is never
written to a project manifest, log, URL, or command output. **Refresh status**
only makes **Continue cycle** available after every CVAT job is completed. An
empty box list is valid for an image that was reviewed as a negative, but an
uncompleted or partial job is blocked. Continue exports a `CVAT for images 1.1`
ZIP, validates it through the same importer as initial data, and creates a new
immutable merged snapshot. The Train model page can then resume from the
parent checkpoint using that newest snapshot.

The CLI equivalent is:

```bash
export CVAT_URL=http://localhost:8080
export CVAT_TOKEN='read-from-your-secret-store'
amphilens cvat managed-start ./my-project --cycle 0 \
  --image /data/queue/camera_001.jpg --image /data/queue/camera_002.jpg
amphilens cvat managed-status ./my-project --cycle 0
amphilens cvat managed-continue ./my-project --cycle 0
```

The portable exchange remains the recovery path when CVAT is unavailable or
credentials cannot be used. AmphiLens never deletes CVAT projects or tasks
automatically.

## 7. Project outputs

Projects contain a manifest, run records, annotations, artifacts, an artifact
index, and checkpoints. Prediction runs can include:

- stable `predictions.csv` output;
- per-image JSONL and progress records for resumable batches;
- failure records for corrupt or unreadable images;
- summary and Markdown/JSON reports; and
- optional overlay images.

Run and checkpoint metadata records the model, architecture, class order,
preprocessing, thresholds, image size, seed, software version, parent
checkpoint, and output paths. Checkpoints are reusable only when their
compatibility metadata matches the new project.

## 8. Python API

```python
from amphilens import InferenceConfig, ProjectManifest, ProjectStore

manifest = ProjectManifest.create(
    name="tunnel-study",
    image_roots=["/data/camera-traps"],
    classes=["toad", "other_amphibian"],
)
store = ProjectStore("./my-project")
store.create(manifest)
run = store.start_run(InferenceConfig(model_id="wlt-yolo", confidence=0.25))
```

## 9. Development and verification

Install the optional development dependencies and run the test suite:

```bash
python -m pip install -e ".[cli,inference,dev]"
PYTHONPATH=src uv run --with pytest --with pillow --with typer --no-project pytest -q
```

Run Ruff when available and inspect `git diff --check` before committing.
Do not commit datasets, model weights, generated outputs, caches, virtual
environments, credentials, or machine-specific paths.

For deployment details, see [deployment.md](deployment.md). For training
details, see [training.md](training.md). For release checks, see
[release-checklist.md](release-checklist.md).
