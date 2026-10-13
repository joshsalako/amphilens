"""Optional Ultralytics trainers with AmphiLens's constrained panel augmentation.

This module is loaded only when Ultralytics training or previews are requested.
Top-level trainer classes also support Ultralytics's distributed trainer loader.
"""

from __future__ import annotations

import random

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer

from .augmentation import PANEL_AUGMENTATION, DetectionSample, compose_detection_samples


class PanelTransform:
    """Compose at most two raw dataset samples before formatting training tensors."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.last_sources = []

    @staticmethod
    def _sample(labels):
        import numpy as np
        from PIL import Image

        instances = labels["instances"]
        instances.convert_bbox(format="xyxy")
        height, width = labels["img"].shape[:2]
        instances.denormalize(width, height)
        return DetectionSample(
            Image.fromarray(labels["img"]),
            instances.bboxes.tolist(),
            np.asarray(labels["cls"]).reshape(-1).astype(int).tolist(),
            labels["im_file"],
        )

    def __call__(self, labels):
        import numpy as np
        from ultralytics.utils.instance import Instances

        samples = [self._sample(labels)]
        if len(self.dataset) > 1 and random.random() < PANEL_AUGMENTATION["pair_probability"]:
            anchor = self.dataset.im_files.index(labels["im_file"])
            partner = random.randrange(len(self.dataset) - 1)
            partner += partner >= anchor
            samples.append(self._sample(self.dataset.get_image_and_label(partner)))
        size = int(self.dataset.imgsz)
        result = compose_detection_samples(samples, (size, size))
        self.last_sources = result.sources
        labels["img"] = np.asarray(result.image).copy()
        labels["cls"] = np.asarray(result.labels, dtype=np.float32).reshape(-1, 1)
        labels["instances"] = Instances(
            np.asarray(result.boxes, dtype=np.float32).reshape(-1, 4),
            bbox_format="xyxy",
            normalized=False,
        )
        labels["resized_shape"] = (size, size)
        labels.pop("ratio_pad", None)
        return labels


def configure_training_dataset(dataset):
    """Use the same pipeline in trainers and previews; leave validation untouched."""
    if not dataset.augment:
        return dataset
    from ultralytics.data.augment import Compose, Format, RandomFlip, RandomHSV

    dataset.transforms = Compose(
        [
            PanelTransform(dataset),
            RandomHSV(hgain=0.0, sgain=0.0, vgain=0.1),
            RandomFlip(p=0.5, direction="vertical"),
            RandomFlip(p=0.5, direction="horizontal"),
            Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0),
        ]
    )
    return dataset


class _PanelTrainerMixin:
    def check_resume(self, overrides):
        super().check_resume(overrides)
        from .backends import _ULTRALYTICS_AUGMENTATION

        # Checkpoints may contain the old mosaic/mixup policy. Keep the current
        # policy for resumed runs before arguments are saved or loaders built.
        for name, value in _ULTRALYTICS_AUGMENTATION.items():
            setattr(self.args, name, value)

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        return configure_training_dataset(dataset) if mode == "train" else dataset


class AmphiLensDetectionTrainer(_PanelTrainerMixin, DetectionTrainer):
    """YOLO detection trainer with opaque two-panel composition."""


class AmphiLensRTDETRTrainer(_PanelTrainerMixin, RTDETRTrainer):
    """RT-DETR trainer with the same visibility guarantees as YOLO."""
