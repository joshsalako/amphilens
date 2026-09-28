# AmphiLens User Workflows

## Status and purpose

This is the normative end-to-end workflow reference for AmphiLens. Future
models and engineers should use this document when changing the Streamlit UI,
CLI, CVAT integration, dataset handling, inference, or training behavior.

AmphiLens has two distinct user processes:

1. **Prediction-only:** use an existing pretrained model or compatible
   checkpoint to find organisms in an image folder.
2. **Active learning:** find useful images, annotate a small selected queue in
   CVAT, merge the annotations into an immutable dataset, and train a better
   project-specific model.

The prediction-only process does not require CVAT or labelled data. The active-
learning process uses CVAT when human annotation is required.

## Shared concepts

- **AmphiLens project:** records image roots, ordered classes, model settings,
  preprocessing, and reproducibility metadata.
- **Base model:** an official pretrained model, a paper-specific checkpoint, or
  a compatible local checkpoint used before project-specific training.
- **Initial dataset:** the first labelled images used for fine-tuning. It may
  come from a CVAT project selected through the API or from a local archive.
- **Dataset snapshot:** an immutable, validated copy of images and annotations.
- **Prediction run:** a resumable scan of images that produces predictions,
  reports, and optional visual evidence.
- **Active-learning cycle:** one selection, annotation, merge, and training
  iteration.
- **Checkpoint:** a reusable model plus compatibility metadata and lineage.

## Process 1: prediction-only

### Purpose

Use this process when the user wants predictions immediately and does not want
to train a project-specific model.

### Flow

```text
camera-trap image folder
        ↓
create or open AmphiLens project
        ↓
choose pretrained model and settings
        ↓
run Find animals
        ↓
predictions CSV, reports, and evidence images
```

### User steps

1. Start AmphiLens and open **Create project** or an existing project.
2. Select the folder containing the camera-trap images.
3. Define the ordered organism classes, for example `toad`, `frog`, or
   `salamander`.
4. Choose a pretrained model or compatible local checkpoint:
   - YOLO26-L;
   - RT-DETR-L;
   - Faster R-CNN with ResNet-50 FPN v2; or
   - a compatible user-supplied checkpoint.
5. Configure maximum image dimension, grayscale, CLAHE, confidence threshold,
   and CPU or CUDA execution.
6. Open **Find animals** and press **Run detection**.
7. Download or inspect the outputs.

### Outputs

The run may produce:

- `predictions.csv` with image paths, classes, confidence scores, and boxes;
- resumable per-image progress and JSONL records;
- a summary report;
- optional overlay/evidence images; and
- run metadata containing model, preprocessing, thresholds, and software.

Boxes in exported results must be mapped back to the original image dimensions.
Source images must never be modified.

### CVAT boundary

CVAT is not required for prediction-only work. If the user wants to manually
review detections, AmphiLens may create a review queue for CVAT, but sending a
review queue to CVAT is an optional review action and does not automatically
change the model.

## Process 2: active learning

### Purpose

Use this process when the base model needs to adapt to a new camera location,
species mix, season, lighting condition, camera setup, or other domain shift.

The model should not ask the user to annotate the entire image collection. It
should select a small, informative queue for annotation.

### Starting options

The user can begin in either of these ways:

#### Option A: existing initial annotations

The user already has a small annotated CVAT project.

1. Create an AmphiLens project with the image folder and ordered classes.
2. Set `CVAT_URL` and `CVAT_TOKEN` before starting the app.
3. Open **Import initial dataset** and choose **CVAT project**.
4. Press **Connect to CVAT**.
5. Select the existing CVAT project containing the initial annotations.
6. Review its labels and provide an explicit class mapping if names differ.
7. Press **Import project**.

AmphiLens uses the CVAT API to export the complete selected project, including
all tasks and images. The user does not manually download a ZIP. The export is
validated through `DatasetImporter` and becomes an immutable dataset snapshot.

The project snapshot records the CVAT server URL, project ID and name, task
IDs, export format, SDK version, archive filename, and archive SHA-256. The
CVAT token is never stored.

8. Open **Train model**.
9. Select the imported snapshot and the model preset.
10. Train the initial project-specific checkpoint.

#### Option B: no initial annotations

If no annotated sample exists:

1. Create an AmphiLens project with the image folder and ordered classes.
2. Select a pretrained base model.
3. Run inference on the image folder.
4. Continue to the active-learning queue below.

The first completed CVAT queue becomes the first labelled dataset snapshot for
training.

### Active-learning cycle

```text
model checkpoint
        ↓
inference on image pool
        ↓
Hybrid PPAL queue selection
        ↓
AmphiLens creates or reuses CVAT project and cycle task
        ↓
human annotation in CVAT
        ↓
status refresh and completed-job check
        ↓
CVAT export with images
        ↓
validation and immutable dataset merge
        ↓
resume training from parent checkpoint
        ↺ repeat when useful
```

### User steps for each cycle

1. Run inference using the current base or project checkpoint.
2. Open **Active learning queue**.
3. Provide the prediction, calibration, and feature artifacts.
4. Choose the annotation budget and any advanced sampling settings.
5. Press **Select annotation queue**.
6. Open **CVAT cycle** and select the generated `selection_queue.csv`.
7. Press **Send to CVAT**.

In managed mode, AmphiLens creates or reuses one CVAT project for the
AmphiLens project and creates one task for the cycle. The CVAT project uses the
ordered class schema from the AmphiLens project. Repeating the same send action
for the same cycle is idempotent; a changed selection is rejected.

8. Press **Open CVAT**.
9. Annotate every selected image.
10. Save the CVAT task.

An image with no target organism may have an empty annotation list, but it must
still be reviewed. An incomplete or partially annotated task cannot continue.

11. Return to AmphiLens and press **Refresh status**.
12. Wait until every CVAT job is completed.
13. Press **Continue cycle**.

AmphiLens then:

1. exports the completed CVAT task with images through the API;
2. validates the export, labels, dimensions, image files, and boxes;
3. creates a new immutable incoming snapshot;
4. merges it with the previous snapshot without modifying either snapshot; and
5. records the managed CVAT cycle and merged snapshot lineage.

14. Open **Train model**.
15. Select the new merged dataset snapshot.
16. Select the compatible model architecture and preprocessing configuration.
17. Provide the parent checkpoint manifest.
18. Press **Train model**.

The resulting checkpoint records its parent checkpoint, dataset snapshot,
classes, preprocessing, model architecture, configuration, and software.
Repeat the cycle when additional images would improve the model.

## UI and CLI mapping

The UI and CLI must use the same underlying services. They must not implement
different dataset or CVAT behavior.

| User action | Streamlit | CLI |
|---|---|---|
| Check environment | **Environment** | `amphilens doctor` |
| Create project | **Create project** | `amphilens project create` |
| List CVAT projects | **Connect to CVAT** | `amphilens cvat projects` |
| Import an existing CVAT project | **Import project** | `amphilens dataset import-cvat PROJECT --project-id ID` |
| Import local annotations | **Local archive** | `amphilens dataset import PROJECT ARCHIVE.zip` |
| Train | **Train model** | `amphilens train` |
| Predict | **Find animals** | `amphilens predict` |
| Select an annotation queue | **Active learning queue** | `amphilens active-learn` |
| Start managed CVAT cycle | **Send to CVAT** | `amphilens cvat managed-start` |
| Check annotation completion | **Refresh status** | `amphilens cvat managed-status` |
| Merge completed annotations | **Continue cycle** | `amphilens cvat managed-continue` |

## Non-negotiable behavior

- Prediction-only inference must work without CVAT or labelled data.
- Fine-tuning must be blocked unless a valid labelled snapshot exists.
- Initial CVAT import means the complete selected project in v1; task-level
  selection is not supported.
- CVAT project import must include image files, not annotation-only exports.
- Class mismatches require explicit mappings; positional or silent mappings are
  forbidden.
- Dataset snapshots, source archives, and checkpoints are immutable.
- Existing source images and previous snapshots are never overwritten.
- The same preprocessing configuration must be used for training and inference
  and recorded in metadata.
- Missing calibration evidence must fail clearly in default active learning.
- A CVAT cycle cannot continue until every job is completed.
- Empty annotations are valid only for reviewed negative images.
- CVAT tokens must never enter manifests, logs, CSV files, URLs, or commits.
- The local archive workflow remains available when CVAT is unavailable.
- Evaluation must say `evaluation: not evaluated` when no holdout dataset was
  supplied.

## Failure handling

The UI and CLI should show actionable errors for:

- missing or invalid CVAT credentials;
- inaccessible CVAT projects;
- projects with no tasks, labels, or images;
- annotation-only or corrupt exports;
- invalid or out-of-bounds boxes;
- duplicate image content or unsafe archive paths;
- unknown classes or missing class mappings;
- missing calibration or feature evidence;
- incomplete CVAT jobs; and
- incompatible parent checkpoints.

Failures must not delete CVAT projects, CVAT tasks, source images, prior
snapshots, or prior checkpoints.

## Implementation anchors

Use these services and contracts when implementing or changing behavior:

- `CVATSdkTransport` for authenticated CVAT API access;
- `CVATProjectImportService` for initial whole-project imports;
- `ManagedCVATCycleService` for active-learning task lifecycle;
- `ProjectStore.import_cvat_project(...)` for project-level API imports;
- `DatasetImporter` for archive validation and snapshot creation;
- `DatasetMerger` for immutable active-learning dataset merges; and
- shared preprocessing, inference, training, and checkpoint services.

Do not add a second CVAT parser or a separate UI-only import path. Extend the
shared services and update this workflow specification when behavior changes.
