# AmphiLens project guidance

## Scope

This file applies to the entire `wlt-app` project. `wlt-app` is the product codebase; `/Users/joshua/Downloads/wtl-detection` is the paper/reproduction reference and must not be edited as part of product work unless explicitly requested.

## Engineering standards

- Use the `src/amphilens` layout and Python 3.10+.
- Write a focused failing test before production behavior, then run the full suite after each meaningful change.
- Keep optional ML and UI dependencies lazy-loaded. The core project, manifests, CSV export, CVAT exchange, and diagnostics must remain importable without CUDA.
- Keep the Modal SDK lazy-loaded; only `amphilens.cloud.modal_app` imports Modal at module scope. Unit tests use fake transports and must not authenticate or start remote work.
- Cloud uploads require explicit consent, local content verification, redacted provider errors, and user-visible cost limits. Estimates and timeouts are not guaranteed billing caps.
- Preserve source images and source annotations. Every derived file must have an explicit output path, provenance, and stable mapping back to the source.
- Keep user-owned project folders outside the source checkout by default. The UI and CLI must use the project-location safety contract, and relocation must verify copied hashes before any explicit source removal.
- Do not hard-code machine-specific paths, camera names, WLT-only classes, or credentials.
- Hybrid PPAL is the default active-learning strategy. Missing calibration evidence must be reported clearly; do not silently substitute hard-coded difficulty values.
- Checkpoints are reusable only with compatible architecture, classes, preprocessing, and recorded provenance.
- Prefer small, typed, modular interfaces over large orchestration scripts. Document user behavior and failure modes in plain language.
- Do not commit datasets, model weights, caches, `.venv`, generated predictions, or credentials.

## Verification

Run the dependency-light suite with:

```bash
PYTHONPATH=src uv run --with pytest --with pillow --with typer --no-project pytest -q
```

Run formatting/linting with Ruff when available. Before claiming completion, inspect the complete test output and verify the requested files and repository state.
