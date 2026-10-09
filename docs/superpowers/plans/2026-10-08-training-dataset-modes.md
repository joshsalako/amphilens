# Implementation plan: training data modes and snapshot names

## Task 1: Dataset roles and project snapshot catalog

- Add deterministic grouped 80/10/10 partitioning with an explicit image-level fallback.
- Add shared materialization for train/validation/test directories with preprocessing and YOLO label transforms.
- Add a project-local display-name catalog, import defaults, rename validation, and summary serialization.

## Task 2: Local training contract and phases

- Extend request parsing and validation for three data modes, role snapshots, patience, and epochs per phase.
- Resolve partitions before model work, prepare train/val/test YAML, and record snapshot and split provenance.
- Implement freeze-then-unfreeze training with validation patience and test-only final evaluation in Ultralytics and Faster R-CNN adapters.

## Task 3: Modal training path

- Include selected snapshots and split provenance in cloud estimates, identity, prepared payload, and remote worker configuration.
- Permit remote validation for early stopping, preserve final test-only evaluation, and return the registered best checkpoint and metrics.
- Keep existing explicit upload consent, cost limits, progress reporting, payload verification, and cleanup semantics.

## Task 4: Web interface

- Add the three-mode selector and conditional snapshot selectors; default to separate snapshots.
- Expose patience and maximum epochs per phase with defaults 25 and 100.
- Add editable snapshot names on archive/CVAT import and rename controls in the snapshot list.
- Carry all selected role paths and training settings into local, cloud estimate, and Modal submissions.

## Task 5: Documentation and static verification

- Update training and cloud-training docs with mode behavior, split semantics, phase settings, and evaluation rules.
- Run syntax and lint checks only; automated tests are outside the explicit request for this turn.
- Inspect the final diff and report any backend limitation that remains.
