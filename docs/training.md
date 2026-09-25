# Training and active learning

This guide describes the normal workflow after the browser app is installed.

## 1. Prepare the project

1. Create or open a project in the browser app.
2. Set the image folder and the classes you want to detect.
3. Import an initial CVAT, COCO, or YOLO archive if labelled images are
   available.
4. Confirm that the imported snapshot contains the expected images and classes.

Initial imported images are all used for training. AmphiLens does not invent a
validation split. Until a separate validation dataset is supplied, results say
`evaluation: not evaluated`.

## 2. Supported annotation archives

Initial import accepts:

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
load image -> maximum-dimension resize -> optional grayscale -> optional CLAHE
```

The defaults are:

1. maximum dimension: `640`;
2. no upscaling of smaller images;
3. grayscale: enabled and replicated to three channels; and
4. CLAHE: disabled for generic projects.

When CLAHE is enabled, the paper-aligned settings are `clip_limit=2.0` and an
`(8, 8)` tile grid. Resizing happens before CLAHE to reduce processing time. A
cache uses the source-image hash and preprocessing fingerprint.

## 5. Train the initial model

In the browser app:

1. Open **Train model**.
2. Select the labelled dataset snapshot.
3. Select the model preset.
4. Set epochs, batch size, image settings, and device.
5. Press **Train model**.

The trainer writes `best.pt`, `last.pt`, `metrics.json`, and a checkpoint
manifest. The manifest records classes, preprocessing, configuration, and
software information.

The CLI equivalent is:

```bash
amphilens project create ./my-project \
  --image-root /path/to/unlabelled-images \
  --class-name toad
amphilens dataset import ./my-project /path/to/initial-cvat-or-yolo.zip
amphilens train ./my-project \
  --output-dir ./my-project/checkpoints/cycle-0 \
  --model-preset yolo26-l
```

## 6. Run inference

1. Open **Find animals**.
2. Select the image folder to scan.
3. Select the model or trained checkpoint.
4. Set confidence, image size, preprocessing, and device.
5. Press **Run detection**.

The output includes predictions CSV, reports, and optional visual evidence.
Detection boxes are mapped back to the original image dimensions.

## 7. Create an active-learning queue

1. Run inference on the unlabeled image folder.
2. Open **Active learning queue**.
3. Provide the prediction, calibration, and feature files.
4. Choose the annotation budget and advanced sampling settings.
5. Press **Select annotation queue**.

The default method is Hybrid PPAL. Missing calibration evidence is reported as
an error rather than silently replaced with guessed values.

## 8. Annotate a queue in CVAT

For managed CVAT:

1. Open **CVAT cycle**.
2. Select the generated `selection_queue.csv`.
3. Press **Send to CVAT**.
4. Press **Open CVAT** and annotate every selected image.
5. Save the CVAT task.
6. Return to AmphiLens and press **Refresh status**.
7. Press **Continue cycle** only after all CVAT jobs are complete.

AmphiLens downloads a CVAT-for-Images archive, validates it, and creates a new
immutable merged dataset snapshot. An empty box list is valid for a reviewed
negative image; an incomplete task is blocked.

Set `CVAT_URL` and `CVAT_TOKEN` before starting the app. The token is not
stored in project files or manifests.

## 9. Continue training

1. Open **Train model**.
2. Select the new merged snapshot.
3. Select the same compatible model architecture.
4. Provide the parent `checkpoint.json` as the parent checkpoint manifest.
5. Press **Train model**.

The new checkpoint records the parent checkpoint and preserves the previous
snapshot. Incompatible classes, architectures, or preprocessing settings fail
before training starts.

## 10. Evaluate honestly

Training metrics currently describe training loss and explicitly report
`evaluation: not evaluated` when no holdout dataset is supplied. Use a separate
validation dataset for scientific evaluation, and record the dataset, model,
preprocessing, device, and software versions with the result.
