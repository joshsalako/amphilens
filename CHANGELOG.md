# Changelog

## 0.1.0a1

- Add the first guided local workflow for annotated dataset import, model
  selection, shared preprocessing, training, prediction, and active learning.
- Add immutable CVAT/COCO/YOLO dataset snapshots and reproducible project
  configuration metadata.
- Add managed CVAT cycles with idempotent task creation, completion checks,
  validated annotation import, immutable merging, and resumable training
  lineage.
- Publish `0.1.0a1` to TestPyPI; clean Python 3.11 installation verified with
  CLI, Streamlit, and CVAT extras.

All notable changes to AmphiLens are recorded here.

The project is currently pre-release. Version `0.1.0` is a foundation
release for local development and scientific review, not a claim that every
planned detector-training and remote-execution feature is production ready.

## [Unreleased]

- Add optional Modal cloud GPU training with explicit upload/cost consent,
  durable job records, progress and cancellation, and checksum-verified local
  checkpoint registration.
- Continue hardening manifests, annotation validation, detector contracts,
  training workflows, and release automation.
- Add direct CVAT project listing and whole-project initial dataset import
  through the pinned CVAT SDK, with immutable snapshots and provenance.
- Add a canonical two-process workflow specification for prediction-only and
  active-learning use.

## [0.1.0] - 2026-09-25

- Added dependency-light project, run, model, checkpoint, prediction, and
  active-learning contracts.
- Added optional YOLO, RT-DETR, Faster R-CNN inference adapters.
- Added resumable local inference, CSV/JSONL outputs, reports, and evidence
  overlays.
- Added Hybrid PPAL calibration, uncertainty, diversity, and class-aware
  selection artifacts.
- Added file-based CVAT/COCO and YOLO import/export with source mappings.
- Added Typer CLI, Streamlit guided workflow, diagnostics, and local execution
  contracts.
- Added Docker CPU/CUDA foundations and Compose documentation.
