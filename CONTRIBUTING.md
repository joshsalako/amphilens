# Contributing to AmphiLens

## Before changing code

- Read `AGENTS.md`.
- Keep product changes in `wlt-app`; treat `wtl-detection` as a reference repository.
- Add or update a focused test before implementing behavior.
- Keep optional dependencies lazy and preserve source-image immutability.

## Test command

```bash
PYTHONPATH=src uv run --with pytest --with pillow --no-project pytest -q
```

## Pull requests

Describe the user-visible behavior, data/provenance implications, hardware assumptions, and verification command. Do not include datasets, model weights, credentials, generated predictions, or local absolute paths.

## Documentation

User documentation must explain the workflow and failure modes without assuming Python knowledge. Developer documentation should state interfaces, dependency boundaries, and compatibility rules. Avoid generated filler and unsupported performance claims.

