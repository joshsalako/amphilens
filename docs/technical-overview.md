# AmphiLens technical overview

This page contains implementation and reproducibility details that are not
needed for a first-time user. Start with the [README](../README.md) if you
only want to run the browser app.

## Current architecture

AmphiLens is a local-first Python package with a Streamlit browser interface
and a Typer command-line interface. Both use the same services:

- `amphilens.core` stores project, run, model, checkpoint, and detection
  manifests.
- `amphilens.models` provides YOLO, RT-DETR, and Faster R-CNN adapters.
- `amphilens.inference` runs predictions, writes stable CSV files, and creates
  visual evidence.
- `amphilens.active_learning` implements the default Hybrid PPAL selection
  workflow.
- `amphilens.annotations` provides portable CVAT/COCO/YOLO exchange and is the
  boundary for the planned managed CVAT adapter.

The package keeps source images untouched. Derived files are written to the
project's artifact directories and retain mappings back to their source files.

## Active-learning method

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

## Command-line workflows

The browser app is the recommended interface. The CLI is useful for repeatable
or unattended runs:

```bash
amphilens doctor
amphilens app
amphilens project-create ./my-project \
  --image-root /path/to/camera-trap-images \
  --class-name toad \
  --name tunnel-study
amphilens images ./my-project
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

## CVAT exchange and managed integration

The current portable exchange writes:

- a COCO `annotations.json` file;
- review images with deterministic flat filenames;
- a manifest mapping each review image to its original source path;
- `classes.txt`; and
- YOLO labels and `dataset.yaml` when YOLO exchange is requested.

The planned managed workflow will use the official CVAT SDK with a pinned
compatibility profile. It will create one CVAT project per AmphiLens project
and one task per active-learning cycle, then persist the CVAT IDs and selection
hash. A Continue action will check readiness, export annotations, validate
classes/dimensions/boxes, merge a new immutable dataset snapshot, and resume
training. Credentials will stay outside project and run manifests.

## Project outputs

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

## Python API

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

## Development and verification

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
