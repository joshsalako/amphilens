# Training and active learning

The normal path is the browser app. Create a project, import a labelled CVAT or
YOLO ZIP, choose a model, and press **Train model**. The initial imported images
are all used for training. AmphiLens does not invent a validation split; until
you provide a separate validation dataset, reports say `evaluation: not evaluated`.

Install the training runtime:

```bash
python -m pip install -e ".[cli,inference,training]"
```

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

The first supported model presets are YOLO26-L (`yolo26l.pt`), RT-DETR-L
(`rtdetr-l.pt`), and Faster R-CNN ResNet-50 FPN v2. Official general-purpose
weights are used when no local checkpoint is selected. A preset is not claimed
to be the exact paper checkpoint until that checkpoint is supplied and recorded.

## Initial dataset formats

The import action accepts CVAT for Images 1.1 ZIP, COCO 1.0 ZIP, or YOLO ZIP
with `images/`, labels, and `classes.txt` or `dataset.yaml`. Images with empty
labels are retained as reviewed negatives. Archives are hashed, validated, and
copied into immutable snapshots; original images and previous snapshots are
never modified.

The expected dataset bundle contains:

```text
cycle-0/
  dataset.yaml
  images/
    camera_001.jpg
  labels/
    camera_001.txt
  classes.txt
```

`dataset.yaml` must identify one training image directory and class order. The
AmphiLens snapshot preparation writes the required fields:

```yaml
path: /absolute/path/to/cycle-0
train: images
labels: labels
names:
  0: toad
  1: other_amphibian
```

The stable Python interface is:

```python
from amphilens.models import FasterRCNNDetector

detector = FasterRCNNDetector(
    "base-faster-rcnn.pt",
    classes=["toad", "other_amphibian"],
)
best = detector.train(
    "cycle-0/dataset.yaml",
    "checkpoints/cycle-0",
    {
        "epochs": 10,
        "batch_size": 2,
        "image_size": 640,
        "device": "auto",
        "seed": 42,
        "preprocessing": {"name": "none"},
    },
)
print(best)
```

## Preprocessing

The same configuration is used when preparing training data and when running
inference:

```text
load image -> maximum-dimension resize -> optional grayscale -> optional CLAHE
```

The maximum dimension defaults to 640 and never enlarges a smaller image.
Grayscale is on by default and is replicated into three channels. CLAHE is off
for generic projects and uses `clip_limit=2.0` and an `(8, 8)` tile grid when
enabled. Resizing happens first so CLAHE processes fewer pixels. A cache is
keyed by the source hash and preprocessing fingerprint.

The trainer writes `best.pt`, `last.pt`, and `metrics.json`. Metrics currently
report training loss and explicitly record `evaluation: not evaluated`; a
holdout split and representative CPU/GPU benchmarks are required before a
checkpoint is described as scientifically evaluated.

An explicit `device: cuda` request fails if CUDA is unavailable. `device:
auto` uses CUDA when available and otherwise uses CPU, which is useful for
small smoke tests but not a substitute for production training benchmarks.

## Managed CVAT cycle

For a queue produced by active learning, open **CVAT cycle** in the browser
app, select the `selection_queue.csv`, and press **Send to CVAT**. AmphiLens
creates the correctly labelled task, shows **Open CVAT**, and records the task
ID so a browser refresh does not create a duplicate. Set `CVAT_URL` and
`CVAT_TOKEN` before starting the app; the token is not stored.

Annotate and save every image in CVAT. Return to AmphiLens, press **Refresh
status**, and then **Continue cycle** once the task is completed. AmphiLens
downloads the CVAT-for-Images archive, validates classes, dimensions, image
files, and boxes, and creates a new immutable merged snapshot. Select that
snapshot on **Train model** and resume from the parent checkpoint. The
portable ZIP import/export workflow remains available if the managed server is
not reachable.
