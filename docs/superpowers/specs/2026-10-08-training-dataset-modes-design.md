# Training data modes and snapshot names

## Goal

Let users choose how a training run gets its training, validation, and test data, expose patience in the training form, and identify imported dataset snapshots with names users can edit. The same split contract must work for local and Modal training and be recorded with the resulting checkpoint.

## Data modes

The training form defaults to **Separate snapshots** and offers:

1. **Automatic split**: deterministically assign 80% of a combined snapshot to training, 10% to validation, and 10% to final testing. Keep images from the same source folder together. If source grouping cannot produce all three partitions, require the user to explicitly allow image-level splitting before submission.
2. **Use training data for validation**: use the selected training snapshot for early-stopping validation and do not run final evaluation.
3. **Separate snapshots**: training is required; validation and test snapshots are optional. With no validation snapshot, use training data for early stopping. With no test snapshot, skip final evaluation.

Test data never participates in early stopping. Validate class order, snapshot ownership, non-empty partitions, and cross-role image duplication before training or cloud upload.

## Training behavior

Expose patience (default 25) and maximum epochs per phase (default 100). Use the paper-aligned schedule: freeze the backbone for phase one, then unfreeze all layers for phase two, each with its own early stopping. Apply the schedule to YOLO, RT-DETR, Faster R-CNN, and Modal. Record data mode, snapshot IDs, phase settings, preprocessing, and evaluation results in run/checkpoint metadata. Use validation for early stopping and only the selected test set for final evaluation.

## Snapshot naming

Keep snapshot IDs, image content, annotations, and manifests immutable. Store editable display names in a project-local dataset catalog keyed by snapshot ID. Ask for a name during archive and CVAT imports, derive a readable default from the source when omitted, and allow later renaming. Show display names and image counts in training selectors.

## User interface and API

The local import and CVAT import requests accept an optional display name. Add a project-local rename endpoint. Training and cloud-estimate requests carry the selected data mode, role snapshot paths, patience, phase epoch limit, seed, and the explicit image-level fallback choice. The UI conditionally displays one combined snapshot selector or three role selectors based on mode. Cloud estimates account for the selected role datasets; existing upload and cost consent remains required.

## Failure handling

Reject missing training data, incompatible classes, duplicate content across independently selected roles, invalid patience/epoch values, or impossible automatic splits before starting a run. If grouped splitting is infeasible and image-level fallback was not approved, return a concise actionable error. Never silently substitute training data for a requested test snapshot.

## Verification

Verify deterministic partitioning, no group leakage, role-specific YAML paths, metadata round-trip, local and cloud request parsing, patience forwarding, phase transitions, and snapshot rename behavior. Do not include datasets, model weights, or cloud credentials in repository artifacts.
