# Automatic Cloud Training Output Collection

## Goal

When a cloud training job finishes, automatically transfer and verify every regular file produced inside the model trainer's run-output directory. Preserve the directory structure under the job's project run folder, and keep the selected best checkpoint registered for reuse.

## Scope

Include all regular files recursively written under the trainer's output directory, including files such as per-phase `results.csv`, plots, configuration files, metrics, and phase checkpoints. Preserve paths relative to that directory. Do not include captured console output, datasets, model caches, or unrelated temporary files.

For the already-collected `cloud-094a7c80ca3d76c7` run, the local result folder and remote artifact metadata contain only `best.pt`, `last.pt`, `metrics.json`, and `checkpoint.json`; its remote cleanup succeeded. This change cannot backfill files the old worker did not transfer.

## Design

Use a recursive per-file manifest rather than a single archive. Each entry contains a safe relative path, SHA-256 digest, size, and provider-specific remote reference. This keeps partial collection resumable: already-present files are reused only after their size and digest match.

The worker copies the complete trainer output tree into remote `results/`, updates the top-level checkpoint manifest to reference the canonical remote `best.pt`, and returns metadata for every regular file. Managed provider workers upload every listed file while preserving nested paths. The Modal worker references the equivalent files already written to its volume.

When normal job refresh first observes provider success, the service automatically collects the outputs. It downloads each file into `<project>/runs/<run-id>/results/<relative-path>`, validates its path, size, and hash, validates the required top-level checkpoint files and manifest, and registers every collected file in the project artifact index. It copies the selected `best.pt` and a local checkpoint manifest to `<project>/checkpoints/<run-id>/` so the model remains available for later training and inference.

Collection records a `collecting-results` phase while transferring. A transfer or validation error leaves the job incomplete with a visible error; a later normal Refresh retries collection and skips files already verified. The browser has no separate collect action or button. The CLI collect command remains available as a manual recovery option.

The cloud job row shows the run output directory and an expandable list of collected relative paths under the matching run. Remove the global collection result panel so output details cannot appear detached at the bottom of the job list. Keep ordinary Refresh, Cancel, and provider-job actions.

## Failure and safety behavior

- Require `best.pt`, `last.pt`, `metrics.json`, and `checkpoint.json` at the top level for a successful collected run.
- Reject absolute paths, parent traversal, backslashes, empty paths, and destinations that resolve outside the run's `results/` directory.
- Verify each file's byte count and SHA-256 before accepting it.
- On retries, reuse only files whose local size and SHA-256 match the remote manifest.
- Do not mark a job verified or clean up remote files until all outputs and checkpoint compatibility checks pass.
- Keep failed output metadata and errors associated with the same run ID in the UI.

## Verification plan

Review the worker/provider/service path end to end with a nested fixture containing the four required checkpoint files plus a CSV and a plot. Confirm that all relative paths and hashes survive transfer, automatic refresh reaches the verified state without a collect action, and a retry resumes after one failed file transfer. Inspect the rendered job row to ensure the output list belongs to its run. Do not perform a live cloud job as part of local verification.
