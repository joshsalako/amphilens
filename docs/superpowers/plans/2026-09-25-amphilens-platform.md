# AmphiLens Wildlife Detection Platform Implementation Plan

> Living execution plan. Update the status checkboxes and checkpoint notes after each meaningful slice so another session can resume without reconstructing decisions from chat.

**Goal:** Build AmphiLens as a local-first Python package and browser application for reproducible wildlife camera-trap inference, Hybrid PPAL active learning, CVAT annotation exchange, checkpoint reuse, and future local/remote execution.

**Canonical project:** `/Users/joshua/Downloads/wlt-app`

**Scientific reference:** `/Users/joshua/Downloads/wtl-detection` (read-only reference for this product task)

**Working name:** AmphiLens; verify package and trademark availability before publication.

**Architecture:** A dependency-light Python core owns project/run/checkpoint contracts and artifacts. Detector, annotation, active-learning, execution, CLI, and UI adapters sit above those contracts. Local execution is first; SSH/Slurm/Docker/cloud workers reuse the same job and artifact formats later.

**Tech stack:** Python 3.10+, `src/` package layout, Typer CLI, Streamlit local UI, optional Pillow/NumPy/PyTorch/torchvision/Ultralytics, Pytest, Ruff, COCO/YOLO exchange.

## Global constraints

- Keep `wtl-detection` unchanged unless explicitly requested.
- Never overwrite source image roots or source annotations.
- Never hard-code `/srv`, home-directory paths, camera names, credentials, or WLT-only classes.
- Hybrid PPAL is the default: validation calibration → PPAL difficulty → DCUS uncertainty → CCMS diversity.
- Missing calibration evidence fails clearly; an uncalibrated fallback is explicit advanced behavior and is recorded in run metadata.
- Checkpoints are reusable only with compatible architecture, ordered classes, preprocessing, input size, and recorded provenance.
- Core imports must work without CUDA, PyTorch, Ultralytics, Streamlit, or CVAT installed.
- Do not commit datasets, weights, generated predictions, caches, virtual environments, or credentials.
- Use focused tests before behavior changes and run the complete suite at every checkpoint.

## Current repository state

`wlt-app` was initially empty and required an elevated local Git initialization because the managed workspace rejected `.git` writes. Git is now initialized on `main`, the requested `origin` is configured, and the first checkpoint is pushed.

GitHub CLI was switched to `joshsalako` on 2026-09-25. Fresh `gh auth status`
confirmed that account as the active HTTPS identity with `repo` and `workflow`
scopes; `origin` is `https://github.com/joshsalako/amphilens.git`.

Checkpoint history currently pushed to `origin/main`:

- `9195577` foundation, contracts, docs, and tests
- `7df9928` model registry and checkpoint validation
- `2cce326` resumable inference and class-aware PPAL
- `2e09aa0` CVAT and project CLI workflows
- `baee771` execution contracts and Docker deployment foundations
- `1ae363a` auditable Hybrid PPAL queues
- `338ab26` prediction reports
- `d46a250` guided local Streamlit workflow

## Completed implementation slices

- [x] Create `pyproject.toml`, `.gitignore`, `src/amphilens`, and `tests/`.
- [x] Add `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, model-card template, and design specification.
- [x] Implement `ProjectManifest`, `RunManifest`, `ModelManifest`, `InferenceConfig`, `DetectionRecord`, and `CheckpointManifest`.
- [x] Implement `ProjectStore` with source/output collision checks, run state, and completion state.
- [x] Implement deterministic image discovery and stable prediction CSV fields.
- [x] Implement model-independent inference orchestration and optional Pillow overlay generation.
- [x] Implement lazy YOLO/RT-DETR and Faster R-CNN inference adapters.
- [x] Implement PPAL calibration, class-aware Hybrid PPAL selection, uncertainty scoring, and feature-based diversity selection contracts.
- [x] Implement COCO/CVAT and YOLO exchange with deterministic flat filenames and source mappings.
- [x] Implement checkpoint-aware training orchestration and checkpoint manifests.
- [x] Implement a filesystem model registry with atomic metadata writes and checkpoint hash validation.
- [x] Implement resumable per-image inference artifacts, failure records, stable CSV output, and run summaries.
- [x] Make Hybrid PPAL class-aware before filling the remaining budget with diversity selection.
- [x] Add `project create/inspect`, `predict`, and CVAT export/import CLI workflows over shared services.
- [x] Add `active-learn` CLI workflow with PPAL calibration/feature inputs and auditable queue artifacts.
- [x] Add guided Streamlit workflows for environment, project creation, prediction, and Hybrid PPAL queue selection.
- [x] Add JSON and Markdown prediction reports and the `report` CLI command.
- [x] Add serializable `JobSpec`/`JobStatus` contracts and a local execution backend.
- [x] Add CPU/CUDA Dockerfiles, Docker Compose reference deployment, and deployment documentation.
- [x] Implement environment diagnostics, Typer CLI shell, and Streamlit shell.
- [x] Verify the current suite: 33 tests passing with CLI dependencies, including PPAL artifacts, reports, UI helpers, deployment-contract coverage, manifest/artifact hardening, and CVAT integrity checks.
- [x] Add Apache-2.0 license, changelog, security/privacy note, code of conduct, and GitHub Actions CI for supported Python versions.

Current verification command (the local equivalent of the CI dependency-light job):

```bash
PYTHONPATH=src UV_CACHE_DIR=/private/tmp/amphilens-uv-cache \
  uv run --with pytest --with pillow --with typer --no-project pytest -q
```

## Remaining tasks

### Task 0: Establish Git checkpointing and GitHub remote — complete

- [x] Obtain writable Git metadata for `wlt-app` with approved elevation.
- [x] Run `git init -b main`.
- [x] Confirm the working tree excludes `.venv`, caches, weights, and generated artifacts.
- [x] Add the requested remote:

```bash
git remote add origin https://github.com/joshsalako/amphilens.git
git branch -M main
```

- [x] Switch GitHub CLI to `joshsalako`; `gh auth status` confirms a valid active token with `repo` and `workflow` scopes.
- [x] Create the first checkpoint commit containing the foundation, docs, tests, and this plan: `9195577`.
- [x] Push after local verification:

```bash
git push -u origin main
```

Remote verification: `origin` is `https://github.com/joshsalako/amphilens.git`; `main` tracks `origin/main`.

### Task 1: Harden core contracts and artifact storage — in progress

- [x] Add schema version migrations for project, run, and checkpoint manifests.
- [x] Add atomic JSON writes and recovery for interrupted writes.
- [ ] Add remaining manifest validation for duplicate class IDs and all output-path forms; missing roots, unsafe run IDs, nested collisions, and stale checkpoint hashes are covered.
- [x] Add an explicit artifact index containing relative artifact paths, SHA-256, type, cycle, and producer run.
- [x] Add tests for interrupted writes, stale hashes, nested source/output collisions, and manifest round trips.

### Task 2: Complete model registry and detector contracts

- [ ] Add `ModelRegistry` and `ModelManifest` persistence with model-card links, license, domain, class order, preprocessing, and checkpoint hash.
- [ ] Add `load_detector()` compatibility checks before loading any checkpoint.
- [ ] Add backend contract tests covering empty predictions, confidence filtering, image dimensions, device selection, and malformed checkpoints.
- [ ] Complete the Faster R-CNN training adapter using a stable dataset/trainer interface rather than importing hard-coded research paths.
- [ ] Preserve YOLO and RT-DETR training configs while making epochs, freeze schedule, batch size, seed, preprocessing, and image size explicit.
- [ ] Add `best.pt`, `last.pt`, checkpoint manifest, training log, and metrics artifact handling for every cycle.

### Task 3: Finish production inference and reporting — in progress

- [x] Add per-image inference with persisted progress, failure records, and resume support.
- [ ] Add corrupt-image handling that records failures instead of producing fake detections.
- [x] Add summary reports containing image counts, detection counts, class counts, models, runs, and confidence summary.
- [x] Add JSON and Markdown report outputs; CSV, JSONL, and overlays remain separate artifacts.
- [ ] Add tests proving reruns do not duplicate rows or overwrite prior run artifacts.
- [ ] Add representative CPU benchmark and GPU benchmark scripts; publish measured hardware tiers rather than guessed requirements.

### Task 4: Make Hybrid PPAL paper-faithful and generic — in progress

- [ ] Compare `ppal_instance_difficulty`, class-weight calculation, DCUS ratios, and CCMS selection against controlled reference values from the paper/research repository.
- [x] Add validation-match ingestion from labeled prediction/IoU evidence.
- [x] Require calibration evidence for every selected class in default mode.
- [ ] Add advanced controls for `xi`, `alpha`, `beta`, pool multiplier, uncertain/certain/random ratios, target class, priority weight, seed, and diversity feature backend.
- [x] Add reproducible selection queue and calibration artifacts containing selected rows, reasons, seed, and calibration source.
- [x] Add tests for class coverage, missing calibration, missing features, and budgeted selection.

### Task 5: Complete CVAT workflow — in progress

- [x] Add task-level manifest validation before export/import.
- [x] Add stable image IDs independent of filenames and preserve relative plus absolute source references where possible.
- [x] Add import validation for unknown classes, duplicate annotation IDs, invalid boxes, image dimension mismatches, and missing images.
- [ ] Add optional CVAT REST integration only after file-based exchange is stable; credentials must never enter project manifests.
- [ ] Add a user guide with the exact CVAT export, annotation, export, and import steps.

### Task 6: Build the guided CLI and Streamlit workflow — in progress

- [x] Add CLI commands: `app`, `doctor`, `project create`, `project inspect`, `predict`, `active-learn`, `cvat export`, `cvat import`, and `report`.
- [x] Keep CLI and UI on shared services; no duplicate business logic.
- [x] Add a guided wizard for image roots, classes, base model, output location, prediction, and PPAL queue steps.
- [ ] Add advanced panels for all documented model, preprocessing, threshold, PPAL, and training controls.
- [ ] Show actionable failures for missing GPU, missing calibration, incompatible checkpoint, invalid annotation, and insufficient disk.
- [ ] Add UI smoke tests for project creation, diagnostics, configuration validation, and artifact download.

### Task 7: Package and release quality

- [ ] Add lockable development environments for supported Python versions.
- [x] Add CI with Ruff linting, Pytest coverage, import-without-ML-dependencies check, and package build verification.
- [x] Add a changelog, security/privacy note, code of conduct, and license file.
- [x] Add a release checklist.
- [x] Execute clean wheel/sdist install verification in isolated environments; both artifacts imported as version `0.1.0`.
- [ ] Complete model cards for every published base checkpoint.
- [ ] Verify PyPI name availability and package/trademark naming before publishing.

### Task 8: Docker and remote execution roadmap — in progress

- [x] Define `JobSpec`, `JobHandle`, `JobStatus`, and `ExecutionBackend` interfaces without changing local project contracts.
- [x] Implement a local backend as the reference implementation.
- [x] Add Docker CPU inference image and CUDA training image foundations.
- [x] Add Docker Compose single-machine deployment with local volumes and documented privacy boundaries.
- [ ] Add SSH and Slurm adapters for institutional GPUs.
- [ ] Add provider-neutral remote worker protocol with resumable uploads/downloads, logs, checkpoint artifacts, and job cancellation.
- [ ] Only then evaluate hosted deployment with API, queue, workers, metadata database, object storage, authentication, and multi-user isolation.

## Public interfaces to preserve

```text
ProjectManifest
RunManifest
ModelManifest
CheckpointManifest
DetectionRecord
ProjectStore
DetectorBackend.predict(images, config)
DetectorBackend.train(dataset, output_dir, config, resume_from)
ActiveLearningStrategy.select(predictions, calibration, features)
ExecutionBackend.submit(job_spec)
```

Changes to these interfaces require updating the design specification, migration notes, and contract tests together.

## Acceptance criteria

- A clean CPU environment can install the core/CLI and run `amphilens doctor`.
- A user can create a project without editing Python code or machine-specific configuration.
- A compatible checkpoint can run inference and produce stable CSV plus provenance/evidence artifacts.
- A user can export a review subset to CVAT, annotate it, import it, and preserve source mappings.
- Default Hybrid PPAL stops when calibration evidence is missing and produces reproducible selections when evidence is present.
- Checkpoints can be resumed/reused only when compatibility metadata matches.
- The three detector backends share one tested interface; unsupported runtime dependencies fail with actionable messages.
- Interrupted work resumes without duplicate detections or overwritten source artifacts.
- Docker and future remote execution consume the same project/job/artifact contracts.

## Checkpoint protocol

Do not wait until the entire platform is complete to commit. Create a checkpoint after each coherent slice:

1. foundation + docs + tests;
2. hardened contracts and registry;
3. inference/reporting;
4. Hybrid PPAL;
5. CVAT workflow;
6. CLI/UI workflow;
7. package/CI/release quality;
8. Docker/remote execution.

Before every checkpoint: run the complete test command, inspect `git diff --check`, inspect the changed-file list, and record any deferred limitations in this plan.

## Decisions and rulings

- **Ruling:** Keep `wlt-app` separate from `wtl-detection` — this preserves paper reproducibility while allowing product interfaces and paths to evolve.
- **Ruling:** Use local browser UI first — it protects camera-trap privacy and avoids making cloud storage/GPU billing a v1 prerequisite.
- **Ruling:** Use CVAT file exchange before REST integration — it keeps the first annotation contract portable and credential-free.
- **Ruling:** Require calibration in default Hybrid PPAL — silent AP/hard-coded fallbacks are not dependable for new taxa or domains.
- **Ruling:** Support all three detector families behind adapters — this preserves the paper’s architecture comparison while keeping the engine model-independent.
- **Ruling:** Repository publication is performed only through the verified `joshsalako` identity and the requested `origin` remote.

## Next immediate work

1. Harden manifest schemas, atomic artifact indexing, and recovery for interrupted writes.
2. Add deeper CVAT validation for unknown classes, duplicate IDs, dimensions, and missing images.
3. Complete detector backend contract coverage and decide the stable Faster R-CNN training dataset interface.
4. Add advanced Streamlit controls and explicit failure guidance for calibration, checkpoints, GPU, and disk.
5. Keep this plan and the checkpoint history current after each coherent implementation slice.
