# AmphiLens Wildlife Detection Platform Design

## Purpose

AmphiLens is a local-first Python package and browser application for detecting amphibians and other organisms in camera-trap imagery. It brings the accepted WLT active-learning research workflow into a reusable, documented, and reproducible product for Nature Connect and the broader ecology community.

## Product contract

- CPU inference is supported; GPU is recommended for large pools and fine-tuning.
- A guided wizard is the normal path; advanced mode exposes model, preprocessing, threshold, image-size, cycle, budget, sampling, and training controls.
- The base model registry describes supported classes, domain assumptions, licenses, preprocessing, and checkpoint provenance.
- User-defined classes require labeled examples and fine-tuning; a typed class name does not create zero-shot capability.
- CVAT is the first external annotation integration. Exchange bundles retain deterministic mappings to original image paths.
- Results include stable CSV predictions, run metadata, checkpoints, logs, and optional visual evidence.

## Architecture

`amphilens.core` owns manifests, typed records, compatibility checks, and filesystem artifacts. `amphilens.models` provides YOLO, RT-DETR, and Faster R-CNN adapters. `amphilens.inference` owns model-independent prediction and export. `amphilens.active_learning` owns the default Hybrid PPAL strategy. `amphilens.annotations` owns CVAT/COCO/YOLO exchange. The Typer CLI and Streamlit app call these services rather than duplicating pipeline logic.

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

