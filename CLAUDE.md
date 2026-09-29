# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

`wlt-app` is the AmphiLens product codebase. `/Users/joshua/Downloads/wtl-detection` is the
paper/reproduction reference — read it for method questions, never edit it as part of product work.

`AGENTS.md` holds the authoritative engineering standards and is not repeated in full here. Read it.

## Commands

```bash
# Dependency-light suite (the verification command in AGENTS.md)
PYTHONPATH=src uv run --with pytest --with pillow --with typer --no-project pytest -q
```

```bash
# Single test file, or a single test
PYTHONPATH=src uv run --with pytest --with pillow --with typer --no-project pytest tests/test_ppal.py -q
```

```bash
# Lint (line-length 100, rules E,F,I,UP; target py310)
ruff check src tests
```

```bash
# Install all extras, then build a wheel + sdist
uv sync --locked --python 3.11 --extra cli --extra ui --extra inference --extra training
python -m build
```

`pyproject.toml` sets `pythonpath = ["src"]` and `testpaths = ["tests"]`, so a plain `pytest` also
works inside an installed environment. There is no `conftest.py`; tests use `tmp_path` and fakes.

## Running the product

```bash
uv run --locked amphilens doctor      # CPU / disk / ML dependency / CUDA report
uv run --locked amphilens app         # launches Streamlit on :8501
uv run --locked amphilens --help      # CLI command set
```

Environment variables: `CVAT_URL` and `CVAT_TOKEN` for managed CVAT; `AMPHILENS_PROJECT_ROOT` and
`AMPHILENS_MODEL_ROOT` for the Docker/Compose layouts. Optional cloud training accepts
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` or a user-scoped credential file created by
`amphilens cloud login`.

## Architecture

**Contracts first.** `src/amphilens/core.py` is the spine: every dataclass contract
(`ProjectConfig`, `ProjectManifest`, `InferenceConfig`, `RunManifest`, `DetectionRecord`,
`ModelManifest`, `CheckpointManifest`, `ArtifactRecord`) plus `ProjectStore`, which owns the
on-disk project layout (`manifest.json`, `runs/`, `artifacts/`, `checkpoints/`, `annotations/`,
`datasets/`). A new field is a schema change: bump `CURRENT_SCHEMA_VERSION` and extend
`_migrate_manifest_data`, which also rejects manifests newer than the running code.

**Every write goes through `atomic_write_json` / `read_json`.** They replace-by-rename and recover an
interrupted temporary write. Do not call `json.dump` onto a manifest directly.

**UI and CLI are thin adapters.** `cli.py` (Typer) and `ui.py` (Streamlit) both call the same
services, and the canonical workflow spec requires it stay that way: *"Do not add a second CVAT
parser or a separate UI-only import path."* New user-facing behavior goes in a service first, then
gets wired into both surfaces.

**Immutability is enforced, not documented.** `ProjectStore.create` raises `SourceCollisionError`
when an output directory overlaps an image root; `DatasetSnapshot.to_yolo_dataset` refuses an
existing destination; `DatasetMerger` never rewrites its inputs. Source images and prior snapshots
are read-only.

**Preprocessing identity travels with the data.** `PreprocessingConfig.fingerprint` (sha256, first 16
chars) is recorded in project, run, dataset, and checkpoint metadata. Training and inference must use
the same config; `CheckpointManifest.validate_compatibility` compares architecture, ordered classes,
and the full preprocessing dict before a checkpoint may be reused, and `load_detector` additionally
re-checks the checkpoint's sha256.

**Optional ML/UI dependencies load lazily, inside functions.** `UltralyticsDetector._load()` and
`FasterRCNNDetector._load()` import torch/ultralytics on first use; `CVATSdkTransport._load_sdk()`
imports `cvat_sdk` similarly; `ui.main()` imports streamlit. Missing extras raise
`OptionalDependencyError` / `RuntimeError` with the extra name to install. The core package,
manifests, CSV export, CVAT exchange, and diagnostics must stay importable without CUDA — CI asserts
this with a bare `import amphilens`.

**Detector contract:** `predict(image_paths, config)` is a generator yielding `DetectionRecord`.
Boxes come back through `PreprocessedImage.map_box_to_original`, so CSV boxes are always in source
image coordinates. Adding a model family means a new adapter in `models/backends.py` plus a
`ModelPreset` in `models/catalog.py` (`ModelCatalog` is deliberately small; keep it that way).

**Training path:** `DatasetSnapshot.to_yolo_dataset` materializes a prepared copy →
`train_and_register` → `best.pt`, `last.pt`, `metrics.json`, `checkpoint.json`. Faster R-CNN trains
through its own `models/faster_rcnn_training.py`. Resume is only allowed with a compatible parent
manifest. Without a holdout dataset the output must report `evaluation: not evaluated` — never
invent a validation split.

**Active learning:** `active_learning.HybridPPALStrategy` runs DCUS-style uncertainty then
CCMS-style diversity. `calibrate_ppal` raises `CalibrationRequiredError` /
`MissingCalibrationClassError` when validation evidence is absent, and `select` raises
`FeatureRequiredError` when an image lacks features. Per AGENTS.md, missing evidence is reported —
never substituted with hard-coded difficulty values.

**CVAT access is behind the `CVATTransport` Protocol** in `annotations/managed.py`, with
`CVATSdkTransport` as the only real implementation and fake transports in tests. That is why the
suite never touches the network. Managed cycles are keyed by `selection_hash` and idempotent: the
same selection resumes the same task, a changed selection is rejected. `CVAT_TOKEN` is read from the
environment only and must never reach a manifest, log, CSV, URL, or commit.

**Cloud training is opt-in and provider-isolated.** `CloudTrainingService` persists consented
jobs and verifies every downloaded artifact before registration. Modal imports stay inside
`cloud/modal_app.py` and `ModalTransport` operations. Tests must not use real Modal credentials
or launch paid work; exercise the worker and transport with local fixtures and fakes.

## Working rules

- Write a focused failing test before production behavior, then run the full suite.
- Do not hard-code machine-specific paths, camera names, WLT-only classes, or credentials.
- Do not commit datasets, weights, caches, `.venv`, generated predictions, or credentials
  (`.gitignore` covers most of this, including `.env`, `.agents/`, `.claude/`).
- The normative behavior reference is
  `docs/superpowers/specs/2026-09-28-amphilens-user-workflows.md`; update it when behavior changes.
- `README.md` is user-facing prose for non-Python users; `docs/technical-overview.md` and
  `docs/training.md` are the developer-facing counterparts.
