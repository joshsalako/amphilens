# Cloud Training Output Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically collect and verify every regular file emitted in a cloud trainer output directory into the matching project run's `results/` folder, while preserving checkpoint registration and removing the manual collection UI.

**Architecture:** The trainer worker recursively inventories regular output files and reports safe relative paths, SHA-256, size, and remote references. Provider workers transfer each listed file. The cloud service collects these files automatically when refresh observes remote success, validates all files before verification/cleanup, supports verified-file reuse on retry, and registers the full output set. The UI presents per-run artifacts inline and has no collect button or detached result panel.

**Tech Stack:** Python 3.10+, FastAPI service layer, JavaScript web client, existing provider storage adapters.

### Task 1: Preserve complete trainer outputs in remote artifacts

**Files:** `src/amphilens/cloud/worker.py`

- Recursively enumerate regular files under the trainer output directory only; exclude symlinks and anything outside that directory.
- Preserve relative paths under remote `results/`, hash and size each file, and keep required checkpoint files at the result root.
- Update the checkpoint manifest's canonical best-checkpoint reference and return a relative-path keyed artifact manifest.
- Keep console capture, uploaded datasets, caches, and temporary staging files out of the result manifest.

### Task 2: Transfer nested outputs through managed providers

**Files:** `src/amphilens/cloud/provider_worker.py`

- Upload each manifest-listed file to its matching nested result key.
- Preserve nested relative paths and attach provider-specific remote references.
- Reject unsafe paths or missing files before reporting success.

### Task 3: Auto-collect, verify, and register outputs

**Files:** `src/amphilens/cloud/service.py`, `src/amphilens/cloud/models.py` (only if the record contract needs a field)

- On normal refresh, automatically collect when the provider first reports success; also retry saved `finished` jobs after a process restart.
- Validate safe relative paths, sizes, and hashes; retry by reusing local files only when their contents still match.
- Preserve directory structure under `runs/<run-id>/results/`, require the canonical checkpoint files, register every collected file, and retain reusable `best.pt` plus manifest in `checkpoints/<run-id>/`.
- Mark the job verified and clean up remote resources only after full validation; expose collection errors on that run and allow Refresh to retry.
- Remove the manual browser collection action; keep the CLI collection command as a retry path for incomplete jobs.

### Task 4: Keep result information attached to its run

**Files:** `src/amphilens/webapp.py`, `src/amphilens/web/app.js`, `src/amphilens/web/app.css`

- Remove the Collect checkpoint/artifacts action and endpoint.
- Render the run's saved `results/` path and an expandable list of collected files directly in that cloud job row.
- Remove the global bottom collection-result panel so notifications and output details cannot be detached from their run.

### Task 5: Review and static verification

- Review worker, provider, service, and UI contracts together, including unsafe paths, nested files, partial transfers, refresh retries, and failed cleanup.
- Run non-test syntax/static checks available in the environment and `git diff --check`.
- Do not add or run tests: the governing task instruction for this session prohibits it unless the user explicitly asks for tests or verification.

## Review Focus

- Ensure only files rooted inside the trainer output directory enter the collected run outputs.
- Ensure nested result paths cannot escape the run directory and are not flattened or overwritten.
- Ensure normal refresh can collect once and retry partial failures without allowing a false verified state.
- Ensure every artifact path and error remains attached to its corresponding run row.
