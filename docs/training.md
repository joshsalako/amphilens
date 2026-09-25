# Fine-tuning a Faster R-CNN checkpoint

AmphiLens can fine-tune Faster R-CNN from the portable YOLO bundle produced by
the CVAT exchange. This path requires the optional `training` extra because it
uses PyTorch, torchvision, NumPy, Pillow, and PyYAML.

Install the training runtime:

```bash
python -m pip install -e ".[cli,inference,training]"
```

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
CVAT/YOLO export already writes the required fields:

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

The trainer writes `best.pt`, `last.pt`, and `metrics.json`. Metrics currently
report training loss and explicitly record `evaluation: not evaluated`; a
holdout split and representative CPU/GPU benchmarks are required before a
checkpoint is described as scientifically evaluated.

An explicit `device: cuda` request fails if CUDA is unavailable. `device:
auto` uses CUDA when available and otherwise uses CPU, which is useful for
small smoke tests but not a substitute for production training benchmarks.
