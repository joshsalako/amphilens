# Changelog

## 0.1.0a1

- Add the first guided local workflow for annotated dataset import, model
  selection, shared preprocessing, training, prediction, and active learning.
- Add immutable CVAT/COCO/YOLO dataset snapshots and reproducible project
  configuration metadata.

All notable changes to AmphiLens are recorded here.

The project is currently pre-release. Version `0.1.0` is a foundation
release for local development and scientific review, not a claim that every
planned detector-training and remote-execution feature is production ready.

## [Unreleased]

- Continue hardening manifests, annotation validation, detector contracts,
  training workflows, and release automation.

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
