# AmphiLens Wildlife Detection Platform Design

## Purpose

AmphiLens is a local-first Python package and browser application for detecting amphibians and other organisms in camera-trap imagery. It brings the accepted WLT active-learning research workflow into a reusable, documented, and reproducible product for Nature Connect and the broader ecology community.

The normative end-to-end user journeys are maintained in
[`2026-09-28-amphilens-user-workflows.md`](2026-09-28-amphilens-user-workflows.md).

## Product contract

- CPU inference is supported; GPU is recommended for large pools and fine-tuning.
- A guided wizard is the normal path; advanced mode exposes model, preprocessing, threshold, image-size, cycle, budget, sampling, and training controls.
- The base model registry describes supported classes, domain assumptions, licenses, preprocessing, and checkpoint provenance.
- User-defined classes require labeled examples and fine-tuning; a typed class name does not create zero-shot capability.
- CVAT is the first external annotation integration. File exchange remains portable, while the managed workflow creates one CVAT project per AmphiLens project and one task per active-learning cycle. Exchange bundles retain deterministic mappings to original image paths.
- Results include stable CSV predictions, run metadata, checkpoints, logs, and optional visual evidence.

## Managed CVAT active-learning cycle

The target user experience is a resumable three-action cycle:

1. AmphiLens selects a review budget, creates or reuses a CVAT project with the exact ordered classes from `ProjectManifest`, creates a task for the cycle, uploads the selected images, and persists the CVAT URL, project/task/job IDs, selection hash, and compatibility metadata.
2. The user opens the task in CVAT, annotates and saves it, then returns to AmphiLens and clicks **Continue**.
3. AmphiLens verifies the task/job is ready, exports the annotations, validates classes, dimensions, boxes, source mappings, and completeness policy, merges the validated human labels into a new immutable dataset snapshot, and continues fine-tuning from the cycle checkpoint.

The Continue action is authoritative: CVAT does not provide a universal per-image “the user finished reviewing this frame” signal, so the default policy requires the CVAT job/task to be completed and rejects an empty or partial export. An explicit advanced override may import a partial task, but it must be recorded in run metadata and cannot silently start a normal cycle. An empty annotation on a reviewed image is valid; an unreviewed image is not inferred from the absence of boxes.

The integration uses the official pinned `cvat-sdk` API for task creation, upload, status, and export. The official `cvat-cli` remains available for diagnostics and manual recovery, but AmphiLens does not parse an interactive subprocess as its primary control plane. A crash after task creation is recovered by the persisted task ID and selection hash rather than creating a duplicate task. AmphiLens never deletes CVAT projects or tasks automatically.

The v1 managed profile is deliberately narrow: Python 3.11, CVAT Community `v2.76.0`, `cvat-sdk==2.76.0`, and `cvat-cli==2.76.0`, with the remaining AmphiLens runtime dependencies resolved into a committed lockfile and verified in the same CPU/GPU smoke-test images. A project may connect to an existing compatible CVAT server, but a local pinned CVAT Compose deployment is the reference environment. Tokens are supplied through environment/keychain configuration and never stored in project manifests.

## Architecture

`amphilens.core` owns manifests, typed records, compatibility checks, and filesystem artifacts. `amphilens.models` provides YOLO, RT-DETR, and Faster R-CNN adapters. `amphilens.inference` owns model-independent prediction and export. `amphilens.active_learning` owns the default Hybrid PPAL strategy. `amphilens.annotations` owns portable CVAT/COCO/YOLO exchange and the optional managed CVAT adapter. The browser app and Typer CLI call these services rather than duplicating pipeline logic.

Project state is portable and source-preserving. Each run records configuration, code/runtime information, model identity, and output paths. Derived files never overwrite image roots or previous cycle artifacts.

## Active learning

The default is the paper-compatible Hybrid PPAL sequence:

1. calibration from labeled validation evidence;
2. PPAL class difficulty calculation;
3. Difficulty Calibrated Uncertainty Sampling;
4. Category Conditioned Matching Similarity diversity selection; and
5. annotation export and cycle pause.

All selected classes participate by default. A priority class and weight are available in advanced mode. Missing calibration evidence fails with a user-actionable error. Any uncalibrated fallback must be explicitly selected and recorded in the run manifest.

## Checkpoint and deployment policy

Every checkpoint has a SHA-256 digest and a manifest containing architecture, class order, preprocessing, input size, training configuration, seed, software, parent checkpoint, and validation results. Checkpoints are reusable only when these compatibility fields match.

The first execution backend is local. Future backends consume the same job and artifact contracts: SSH, Slurm, Docker, cloud GPU workers, and finally an optional hosted multi-user service. Docker Compose is the first cloud-adjacent deployment target; provider-specific orchestration is deliberately deferred.

## Quality bar

The package uses typed modular interfaces, dependency-light core imports, lazy optional ML/UI dependencies, deterministic tests, model cards, contributor guidance, and plain-language documentation. It fails closed on missing images, invalid boxes, class mismatches, unsupported checkpoints, missing calibration evidence, and source/output collisions.

## First usable app contracts

The v1 user workflow accepts an unlabeled image folder and an optional initial
annotated archive. CVAT for Images 1.1 XML, COCO 1.0, and YOLO ZIP archives
must include image files. AmphiLens validates dimensions, boxes, classes,
duplicate image content, corrupt files, and archive paths, then stores an
immutable snapshot with the original archive hash and reviewed negative images.

The shared preprocessing order is maximum-dimension resize, optional grayscale,
and optional CLAHE, followed by three-channel model input. Resizing preserves
aspect ratio and never enlarges images. Grayscale defaults on; CLAHE defaults
off for generic projects and uses the paper-aligned `2.0`/`(8, 8)` settings
when enabled. The same structured configuration and fingerprint are used by
snapshot preparation and detector inference, and prediction boxes are mapped
back to original dimensions.

The first explicit product presets are YOLO26-L (`yolo26l.pt`), RT-DETR-L
(`rtdetr-l.pt`), and Faster R-CNN ResNet-50 FPN v2. Official general-purpose
weights are distinct from any future paper-specific checkpoint; the catalog
does not claim equivalence without a recorded checkpoint source.
