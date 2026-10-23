"""Direct CPU inference for verified NCNN model artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..core import DetectionRecord, InferenceConfig, ValidationError
from .backends import OptionalDependencyError, _image_size, _inference_image


def _ncnn_yolo_detections(output, class_count: int, confidence_threshold: float):
    """Decode Ultralytics' raw YOLO detect tensor and apply class-aware NMS."""
    import numpy as np

    values = np.asarray(output, dtype=np.float32).squeeze()
    if values.ndim != 2:
        raise RuntimeError(f"NCNN YOLO output must be a 2D prediction matrix, got {values.shape}")
    channels = class_count + 4
    if values.shape[0] == channels:
        values = values.T
    elif values.shape[1] != channels:
        raise RuntimeError(
            f"NCNN YOLO output has {values.shape} values; expected one dimension to be {channels}"
        )
    if values.shape[0] == 0:
        return []
    coordinates = values[:, :4]
    scores = values[:, 4:]
    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]
    valid = np.isfinite(values).all(axis=1) & (confidences >= confidence_threshold)
    coordinates = coordinates[valid]
    confidences = confidences[valid]
    class_ids = class_ids[valid]
    if not len(coordinates):
        return []

    boxes = np.empty_like(coordinates)
    boxes[:, 0] = coordinates[:, 0] - coordinates[:, 2] / 2
    boxes[:, 1] = coordinates[:, 1] - coordinates[:, 3] / 2
    boxes[:, 2] = coordinates[:, 0] + coordinates[:, 2] / 2
    boxes[:, 3] = coordinates[:, 1] + coordinates[:, 3] / 2
    selected = []
    for class_id in np.unique(class_ids):
        class_indices = np.flatnonzero(class_ids == class_id)
        order = class_indices[np.argsort(confidences[class_indices])[::-1]]
        while len(order) and len(selected) < 300:
            current = int(order[0])
            selected.append(current)
            if len(order) == 1:
                break
            others = order[1:]
            left = np.maximum(boxes[current, 0], boxes[others, 0])
            top = np.maximum(boxes[current, 1], boxes[others, 1])
            right = np.minimum(boxes[current, 2], boxes[others, 2])
            bottom = np.minimum(boxes[current, 3], boxes[others, 3])
            intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
            current_area = np.maximum(0, boxes[current, 2] - boxes[current, 0]) * np.maximum(
                0, boxes[current, 3] - boxes[current, 1]
            )
            other_area = np.maximum(0, boxes[others, 2] - boxes[others, 0]) * np.maximum(
                0, boxes[others, 3] - boxes[others, 1]
            )
            union = current_area + other_area - intersection
            overlap = intersection / np.maximum(union, 1e-12)
            order = others[overlap <= 0.7]
    selected.sort(key=lambda index: float(confidences[index]), reverse=True)
    return [
        (int(class_ids[index]), float(confidences[index]), boxes[index].tolist())
        for index in selected
    ]


class NCNNYoloDetector:
    """Direct NCNN CPU adapter for verified raw Ultralytics YOLO detection tensors."""

    def __init__(self, artifact, classes: list[str], model_id: str | None = None):
        self.artifact = artifact
        self.classes = list(classes)
        self.model_id = model_id or artifact.param_path.parent.name
        self.architecture = "yolo"
        self._ncnn = None
        self._net = None

    def _load(self):
        if self._net is not None:
            return self._net
        try:
            import ncnn
        except ImportError as exc:
            raise OptionalDependencyError(
                "NCNN inference requires the ARM runtime profile. Install "
                "`amphilens[edge-ncnn]` on ARM Linux."
            ) from exc
        net = ncnn.Net()
        net.opt.use_vulkan_compute = False
        if net.load_param(str(self.artifact.param_path)) != 0:
            raise RuntimeError(f"NCNN could not load model parameters: {self.artifact.param_path}")
        if net.load_model(str(self.artifact.bin_path)) != 0:
            raise RuntimeError(f"NCNN could not load model weights: {self.artifact.bin_path}")
        input_names = list(net.input_names())
        output_names = list(net.output_names())
        descriptor = self.artifact.manifest
        if descriptor.input_name not in input_names:
            raise RuntimeError(f"NCNN model is missing input blob {descriptor.input_name!r}")
        if descriptor.output_name not in output_names:
            raise RuntimeError(f"NCNN model is missing output blob {descriptor.output_name!r}")
        self._ncnn = ncnn
        self._net = net
        return net

    def predict(self, image_paths: Iterable[Path], config: InferenceConfig):
        import numpy as np
        from PIL import Image

        net = self._load()
        descriptor = self.artifact.manifest
        target_width, target_height = descriptor.input_width, descriptor.input_height
        for path in image_paths:
            original_width, original_height = _image_size(path)
            transformed = _inference_image(path, config)
            image_width, image_height = transformed.processed_size
            scale = min(target_width / image_width, target_height / image_height)
            resized_width = max(1, min(target_width, round(image_width * scale)))
            resized_height = max(1, min(target_height, round(image_height * scale)))
            resized = transformed.image.resize(
                (resized_width, resized_height), Image.Resampling.BILINEAR
            )
            left = (target_width - resized_width) // 2
            top = (target_height - resized_height) // 2
            canvas = Image.new("RGB", (target_width, target_height), (114, 114, 114))
            canvas.paste(resized, (left, top))
            input_array = np.ascontiguousarray(
                np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
            )
            extractor = net.create_extractor()
            input_result = extractor.input(descriptor.input_name, self._ncnn.Mat(input_array))
            if input_result != 0:
                raise RuntimeError("NCNN rejected the prepared input image")
            output_result, output = extractor.extract(descriptor.output_name)
            if output_result != 0:
                raise RuntimeError("NCNN failed to run the selected model output")
            detections = _ncnn_yolo_detections(output, len(self.classes), config.confidence)
            for class_id, confidence, network_box in detections:
                x1 = (network_box[0] - left) / scale
                y1 = (network_box[1] - top) / scale
                x2 = (network_box[2] - left) / scale
                y2 = (network_box[3] - top) / scale
                box = [
                    max(0.0, min(float(image_width), x1)),
                    max(0.0, min(float(image_height), y1)),
                    max(0.0, min(float(image_width), x2)),
                    max(0.0, min(float(image_height), y2)),
                ]
                box = transformed.clip_box_to_content(box)
                if box is None:
                    continue
                yield DetectionRecord(
                    image_path=str(path),
                    image_id=path.name,
                    class_id=class_id,
                    class_name=self.classes[class_id],
                    confidence=confidence,
                    bbox_xyxy=transformed.map_box_to_original(box),
                    image_width=original_width,
                    image_height=original_height,
                    model_id=self.model_id,
                    run_id=config.run_id,
                    preprocessing=config.preprocessing_fingerprint,
                )


def load_ncnn_detector(artifact, *, architecture: str, classes: list[str], model_id: str):
    if architecture != "yolo" or artifact.manifest.adapter != "ultralytics-yolo-raw-detect-v1":
        raise ValidationError(f"NCNN inference is not verified for the {architecture} architecture")
    if not classes or len(classes) != len(set(classes)):
        raise ValidationError("Detector classes must be non-empty and unique")
    return NCNNYoloDetector(artifact, classes, model_id)
