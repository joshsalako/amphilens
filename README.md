# AmphiLens

AmphiLens is a local-first Python package for detecting amphibians and other wildlife in camera-trap imagery. It is designed for conservation teams and ecology researchers who need a repeatable workflow:

1. run a compatible base detector on a large image pool;
2. select informative images with Hybrid PPAL active learning;
3. annotate the selected sample in CVAT or another compatible tool;
4. fine-tune and save a reusable checkpoint; and
5. export predictions and visual evidence with complete provenance.

The working product name is **AmphiLens**. The project is intentionally broader than Western Leopard Toads, although the accepted WLT paper and its implementation are the initial scientific reference.

## Current status

This repository contains the first dependency-light foundation:

- portable project and run manifests;
- source/output collision protection;
- detection records and stable CSV export;
- overlay generation;
- checkpoint manifests with hashes and compatibility checks;
- YOLO, RT-DETR, and Faster R-CNN detector adapters with lazy ML imports;
- paper-compatible PPAL calibration and Hybrid PPAL selection contracts;
- COCO/CVAT and YOLO annotation exchange with deterministic source mappings;
- a Typer CLI, Streamlit shell, and environment diagnostics.

The UI orchestration, remote execution, and full production training workflows are being built incrementally. Faster R-CNN inference is supported through the optional ML adapter; its training path remains explicitly guarded until the dataset-specific trainer contract is completed.

## Requirements

- Python 3.10 or newer;
- CPU: supported for inference;
- CUDA GPU: recommended for large-pool inference and fine-tuning;
- disk space for image pools, checkpoints, and derived evidence;
- CVAT for external annotation workflows.

GPU support is optional. AmphiLens will not upload camera-trap data in local mode.

## Install from source

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cli,ui,inference]"
```

For a CPU-only core/CLI installation:

```bash
python -m pip install -e ".[cli]"
```

Use `amphilens doctor` to inspect the runtime, disk, optional ML libraries, and CUDA availability.

## Quick start

Create a project without editing a configuration file:

```bash
amphilens project-create ./my-project \
  --image-root /path/to/camera-trap-images \
  --class-name toad \
  --class-name other_amphibian \
  --name tunnel-study
```

Inspect the environment and recorded images:

```bash
amphilens doctor
amphilens images ./my-project
amphilens app
```

The project directory contains a manifest, run records, annotations, artifacts, and checkpoints. Original image roots remain outside the project and are never copied unless an annotation exchange explicitly requires a review subset.

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

Detector backends are selected explicitly from checkpoint metadata. A bare checkpoint without compatible architecture, classes, preprocessing, and checksum information must be registered before reuse.

## Active learning

The default strategy is Hybrid PPAL:

```text
validation calibration -> PPAL class difficulty -> DCUS uncertainty -> CCMS diversity
```

The default requires calibration evidence for every selected class. Advanced users may configure budgets, ratios, seeds, priority-class weighting, image size, confidence thresholds, preprocessing, and architecture. Uncalibrated fallback behavior is never silent.

## Annotation exchange

The CVAT adapter produces:

- a COCO `annotations.json` file;
- copied review images with deterministic flat filenames;
- a manifest mapping every exported file to its original absolute path;
- `classes.txt`; and
- YOLO labels and `dataset.yaml` when YOLO exchange is requested.

Import always resolves annotations through the mapping manifest so a renamed review image cannot silently point to the wrong source.

## Reproducibility and outputs

Each run records the model identifier, architecture, classes, preprocessing, thresholds, image size, seed, software version, parent checkpoint, and output paths. Prediction CSV rows include source image identity, class, confidence, pixel and normalized boxes, model, run, cycle, and preprocessing.

Checkpoints are stored with a JSON manifest containing SHA-256, compatibility metadata, training configuration, and parent checkpoint. `best` and `last` checkpoint retention will be added to the training orchestration layer.

## Future deployment

The engine is designed around an execution-backend interface. The roadmap is:

1. local process execution;
2. SSH and Slurm GPU execution;
3. Docker CPU/GPU images and a Docker Compose single-machine deployment;
4. provider-neutral remote workers with object storage and resumable jobs; and
5. optional hosted multi-user deployment.

The project format and artifact contracts remain stable across these modes.

## Development

Run the tests with the dependency-light isolated command:

```bash
PYTHONPATH=src uv run --with pytest --with pillow --no-project pytest -q
```

Do not add datasets, model weights, generated artifacts, credentials, or machine-specific paths to source control. See [`AGENTS.md`](AGENTS.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Scientific reference

The starting method is based on:

> Joshua Salako, Kim Gordon, and Lorène Jeantet. *Annotation-Efficient Object Detection of Endangered Western Leopard Toads in Camera Trap Imagery for Assessing Wildlife Tunnel Use.*

The reproduction repository is maintained separately at `/Users/joshua/Downloads/wtl-detection` in this workspace.
