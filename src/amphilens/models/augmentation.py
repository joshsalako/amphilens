"""Shared detection augmentation with opaque panels and preserved box visibility."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

PANEL_AUGMENTATION = {
    "layout": "top-bottom",
    "max_images": 2,
    "pair_probability": 0.5,
    "rotation_degrees": 10.0,
    "rotation_probability": 1.0,
    "minimum_box_visibility": 0.5,
    "fit": "contain",
    "fill": 114,
    "blur": False,
}


@dataclass
class DetectionSample:
    image: Any
    boxes: list[list[float]]
    labels: list[int]
    source: str = ""


@dataclass
class ComposedSample(DetectionSample):
    sources: list[dict[str, Any]] = field(default_factory=list)


def compose_detection_samples(
    samples: list[DetectionSample],
    output_size: tuple[int, int],
    *,
    angles: list[float] | None = None,
    rng=None,
) -> ComposedSample:
    """Rotate whole frames on expanded canvases, then fit them into separate panels.

    Fitting preserves all frame content. Every returned box must retain at least
    50% of its transformed area; labels are never removed from visible objects.
    PIL is imported only when local training or preview generation calls this.
    """
    from PIL import Image

    if len(samples) not in (1, 2):
        raise ValueError("Detection composition requires one or two images")
    width, height = output_size
    if width <= 0 or height < len(samples):
        raise ValueError("Composition canvas must have positive panel dimensions")
    if angles is not None and len(angles) != len(samples):
        raise ValueError("Each source image needs one rotation angle")
    generator = rng if rng is not None else random
    fill = (PANEL_AUGMENTATION["fill"],) * 3
    canvas = Image.new("RGB", (width, height), fill)
    output_boxes, output_labels, source_records = [], [], []
    for index, sample in enumerate(samples):
        image = sample.image.convert("RGB")
        if len(sample.boxes) != len(sample.labels):
            raise ValueError("Every bounding box needs a class label")
        angle = angles[index] if angles is not None else generator.uniform(-10.0, 10.0)
        if not math.isfinite(angle) or abs(angle) > PANEL_AUGMENTATION["rotation_degrees"]:
            raise ValueError("Rotation must remain between -10 and 10 degrees")
        rotated = image.rotate(
            angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=fill
        )
        top = index * height // len(samples)
        bottom = (index + 1) * height // len(samples)
        panel_height = bottom - top
        ratio = min(width / rotated.width, panel_height / rotated.height)
        fit_width = min(width, max(1, round(rotated.width * ratio)))
        fit_height = min(panel_height, max(1, round(rotated.height * ratio)))
        left = (width - fit_width) // 2
        offset_y = top + (panel_height - fit_height) // 2
        fitted = rotated.resize((fit_width, fit_height), Image.Resampling.BILINEAR)
        canvas.paste(fitted, (left, offset_y))
        cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        fractions = []
        for box, label in zip(sample.boxes, sample.labels):
            x1, y1, x2, y2 = box
            if not (
                all(math.isfinite(value) for value in box)
                and 0 <= x1 < x2 <= image.width
                and 0 <= y1 < y2 <= image.height
            ):
                raise ValueError("Source boxes must lie inside their image")
            corners = []
            for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                dx, dy = x - image.width / 2, y - image.height / 2
                rx = cosine * dx + sine * dy + rotated.width / 2
                ry = -sine * dx + cosine * dy + rotated.height / 2
                corners.append(
                    (
                        left + rx * fit_width / rotated.width,
                        offset_y + ry * fit_height / rotated.height,
                    )
                )
            raw = [
                min(x for x, _ in corners),
                min(y for _, y in corners),
                max(x for x, _ in corners),
                max(y for _, y in corners),
            ]
            clipped = [max(0, raw[0]), max(top, raw[1]), min(width, raw[2]), min(bottom, raw[3])]
            raw_area = (raw[2] - raw[0]) * (raw[3] - raw[1])
            visible_area = max(0, clipped[2] - clipped[0]) * max(0, clipped[3] - clipped[1])
            visibility = visible_area / raw_area if raw_area > 0 else 0
            if visibility < PANEL_AUGMENTATION["minimum_box_visibility"]:
                raise ValueError("Composition would retain less than 50% of a bounding box")
            output_boxes.append(clipped)
            output_labels.append(label)
            fractions.append(visibility)
        source_records.append(
            {
                "source": sample.source,
                "rotation_degrees": angle,
                "panel": [0, top, width, bottom],
                "boxes": len(sample.boxes),
                "minimum_box_visibility": min(fractions, default=1.0),
            }
        )
    return ComposedSample(canvas, output_boxes, output_labels, sources=source_records)
