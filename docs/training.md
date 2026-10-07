# Training and active learning

This guide describes the normal workflow after the browser app is installed.
The normative two-process reference is the
[canonical user workflows](superpowers/specs/2026-09-28-amphilens-user-workflows.md).

## 1. Prepare the project

1. Open **Create project** for a new project, or **Open project** for an
   existing folder containing `manifest.json`.
2. Accept the safe user-project folder or enter another path outside the
   AmphiLens source checkout.
3. Set the image folder and the classes you want to detect.
4. Import an initial annotated dataset from an existing CVAT project or a
   local CVAT, COCO, or YOLO archive if labelled images are available.
5. Confirm that the imported snapshot contains the expected images and classes.

The project selected in **Open project** remains active for the rest of the
browser session. If an old project is inside the source checkout, open it and
use **Move and remove original**; AmphiLens verifies the copy before deleting
the old project folder and does not move the source image folder.

Initial imported images are all used for training. AmphiLens does not invent a
validation split. Until a separate validation dataset is supplied, results say
`evaluation: not evaluated`.

## 2. Import the initial dataset

### Import from an existing CVAT project

1. Install the `cvat` extra and set `CVAT_URL` and `CVAT_TOKEN`.
2. Open **Import images**.
3. Choose **CVAT project** and press **Connect to CVAT**.
4. Select the project containing the initial annotations.
5. Confirm the labels or provide an explicit class mapping.
6. Press **Import project**.

AmphiLens exports the complete project through the CVAT API, including all
tasks and images, then validates it through the same importer used for local
archives. The dataset snapshot records the CVAT project ID, task IDs, server
URL, export format, SDK version, and archive hash. Tokens are not recorded.

### Import a local archive

Initial local import accepts:

1. CVAT for Images 1.1 ZIP;
2. COCO 1.0 ZIP; or
3. YOLO ZIP with images, labels, and `classes.txt` or `dataset.yaml`.

The archive must include the image files. Images with empty labels are kept as
reviewed negatives. Archives are hashed and copied into immutable snapshots;
original images and previous snapshots are not modified.

Example YOLO layout:

```text
cycle-0/
  dataset.yaml
  images/
    camera_001.jpg
  labels/
    camera_001.txt
  classes.txt
```

## 3. Select a model

The first supported model presets are:

- YOLO26-L (`yolo26l.pt`), the default;
- RT-DETR-L (`rtdetr-l.pt`); and
- Faster R-CNN with ResNet-50 FPN v2.

Official general-purpose weights are used when no local checkpoint is selected.
This is not automatically claimed to be the exact paper checkpoint.

## 4. Configure preprocessing

The same preprocessing configuration is used when preparing training data and
when running inference:

```text
load image -> configured-side resize -> optional grayscale -> optional CLAHE
```

New browser-created projects use these defaults:

1. short-side dimension: `640` pixels;
2. no upscaling of smaller images;
3. grayscale: enabled and replicated to three channels; and
4. CLAHE: disabled for generic projects.

Existing projects retain their saved resize mode and dimensions.

When CLAHE is enabled, the paper-aligned settings are `clip_limit=2.0` and an
`(8, 8)` tile grid. Resizing happens before CLAHE to reduce processing time. A
cache uses the source-image hash and preprocessing fingerprint.

## 5. Configuration precedence

AmphiLens stores the selected model and preprocessing settings in the project
`manifest.json`. Refreshing the browser does not reset these values.

1. A checkpoint with `checkpoint.json` supplies the authoritative model ID,
   architecture, classes, preprocessing, and recorded detector input size.
2. Otherwise, the active project configuration supplies the defaults.
3. **Advanced run overrides** apply only to the current run and are recorded in
   its metadata.
4. Incompatible checkpoint overrides are rejected before output files are
   created.

Device and confidence threshold are runtime choices and can be changed for a
run. A local checkpoint without a manifest remains usable with the project
configuration, but AmphiLens warns that its metadata could not be verified.

## 6. Train the initial model

In the browser app:

1. Open **Train a model**.
2. Select the labelled dataset snapshot.
3. Select the model preset.
4. Set epochs, batch size, image settings, and device.
5. Press **Train model**.

The trainer writes `best.pt`, `last.pt`, `metrics.json`, and a checkpoint
manifest. The manifest records classes, preprocessing, configuration, and
software information.

For a Modal cloud GPU, install the optional `cloud` extra and choose **Modal
cloud GPU** from the training location control. Cloud training requires an
explicit dataset-upload and cost consent, uploads a prepared copy of the
selected snapshot, and registers results only after checksum and compatibility
checks. The local training workflow remains available. Follow the
[cloud training guide](cloud-training.md) for credentials, estimates, billing
limits, progress, cleanup, and the CLI commands.

The CLI equivalent is:

```bash
amphilens project default-location
amphilens project create /path/to/my-project \
  --image-root /path/to/unlabelled-images \
  --class-name toad
amphilens cvat projects
amphilens dataset import-cvat /path/to/my-project --project-id 17
# Local archive fallback:
# amphilens dataset import /path/to/my-project /path/to/initial-cvat-or-yolo.zip
amphilens train /path/to/my-project \
  --output-dir /path/to/my-project/checkpoints/cycle-0 \
  --model-preset yolo26-l
```

## 7. Run inference

1. Open **Find wildlife**.
2. Select the image folder to scan.
3. Select the model or trained checkpoint.
4. Set confidence, image size, preprocessing, and device.
5. Press **Run detection**.

The output includes predictions CSV, reports, and optional visual evidence.
Detection boxes are mapped back to the original image dimensions.

## 8. Create an active-learning queue

1. Run inference on the unlabeled image folder.
2. Open **Choose images to review**.
3. Provide the prediction, calibration, and feature files.
4. Choose the annotation budget and advanced sampling settings.
5. Press **Select annotation queue**.

The default method is Hybrid PPAL. Missing calibration evidence is reported as
an error rather than silently replaced with guessed values.

## 9. Annotate a queue in CVAT

For managed CVAT:

1. Open **Annotation cycle**.
2. Select the generated `selection_queue.csv`.
3. Choose **Send a queue to CVAT** and press **Start cycle**.
4. Press **Open CVAT** and annotate every selected image.
5. Save the CVAT task.
6. Return to AmphiLens and press **Refresh status**.
7. Press **Continue cycle** only after all CVAT jobs are complete.

AmphiLens downloads a CVAT-for-Images archive, validates it, and creates a new
immutable merged dataset snapshot. An empty box list is valid for a reviewed
negative image; an incomplete task is blocked.

Set `CVAT_URL` and `CVAT_TOKEN` before starting the app. The token is not
stored in project files or manifests.

## 10. Continue training

1. Open **Train a model**.
2. Select the new merged snapshot.
3. Select the same compatible model architecture.
4. Provide the parent `checkpoint.json` as the parent checkpoint manifest.
5. Press **Train model**.

The new checkpoint records the parent checkpoint and preserves the previous
snapshot. Incompatible classes, architectures, or preprocessing settings fail
before training starts.

## 11. Evaluate honestly

Training metrics currently describe training loss and explicitly report
`evaluation: not evaluated` when no holdout dataset is supplied. Use a separate
validation dataset for scientific evaluation, and record the dataset, model,
preprocessing, device, and software versions with the result.
