import json
import threading
from pathlib import Path

from amphilens.core import DetectionRecord, InferenceConfig
from amphilens.runs import run_resumable_inference


class RecoverableDetector:
    model_id = "recoverable"

    def __init__(self, fail_name=None):
        self.fail_name = fail_name

    def predict(self, image_paths, config):
        for path in image_paths:
            if path.name == self.fail_name:
                raise RuntimeError("temporary image failure")
            yield DetectionRecord(
                image_path=str(path),
                image_id=path.name,
                class_id=0,
                class_name="toad",
                confidence=0.9,
                bbox_xyxy=[1, 2, 11, 22],
                image_width=20,
                image_height=40,
                model_id=self.model_id,
                run_id=config.run_id,
            )


class BatchRecordingDetector:
    model_id = "batch-recording"

    def __init__(self, *, fail_above=None):
        self.calls = []
        self.fail_above = fail_above

    def predict(self, image_paths, config):
        paths = list(image_paths)
        self.calls.append([path.name for path in paths])
        if self.fail_above is not None and len(paths) > self.fail_above:
            raise RuntimeError("CUDA out of memory")
        for path in paths:
            yield DetectionRecord(
                image_path=str(path),
                image_id=path.name,
                class_id=0,
                class_name="toad",
                confidence=0.9,
                bbox_xyxy=[1, 2, 11, 22],
                image_width=20,
                image_height=40,
                model_id=self.model_id,
                run_id=config.run_id,
            )


def _images(tmp_path: Path, count: int):
    paths = []
    for index in range(count):
        path = tmp_path / f"{index:02}.jpg"
        path.write_bytes(b"fixture")
        paths.append(path)
    return paths


def test_resumable_inference_records_failures_and_resumes(tmp_path: Path):
    images = []
    for name in ("a.jpg", "b.jpg"):
        path = tmp_path / name
        path.write_bytes(b"fixture")
        images.append(path)

    config = InferenceConfig(model_id="recoverable", run_id="run-resume")
    first = run_resumable_inference(
        RecoverableDetector(fail_name="b.jpg"), images, config, tmp_path / "artifacts"
    )
    assert first.completed_images == 1
    assert first.failed_images == [str(images[1].resolve())]
    assert json.loads(first.run_manifest.read_text())["status"] == "completed_with_failures"

    second = run_resumable_inference(RecoverableDetector(), images, config, tmp_path / "artifacts")
    assert second.completed_images == 2
    assert second.failed_images == []
    rows = (tmp_path / "artifacts" / "predictions.csv").read_text().splitlines()
    assert len(rows) == 3
    progress = json.loads((tmp_path / "artifacts" / "progress.json").read_text())
    assert len(progress["completed_images"]) == 2
    assert json.loads(second.run_manifest.read_text())["status"] == "completed"


def test_resumable_inference_sends_configured_batches_and_preserves_order(tmp_path: Path):
    images = _images(tmp_path, 5)
    detector = BatchRecordingDetector()

    result = run_resumable_inference(
        detector,
        images,
        InferenceConfig(model_id="batch-recording", batch_size=2, run_id="run-batches"),
        tmp_path / "batch-artifacts",
    )

    assert detector.calls == [["00.jpg", "01.jpg"], ["02.jpg", "03.jpg"], ["04.jpg"]]
    assert result.completed_images == 5
    recorded = [
        json.loads(line)["image_id"]
        for line in (result.predictions_csv.parent / "predictions.jsonl").read_text().splitlines()
    ]
    assert recorded == [path.name for path in images]


def test_resumable_inference_halves_gpu_batch_after_out_of_memory(tmp_path: Path):
    images = _images(tmp_path, 4)
    detector = BatchRecordingDetector(fail_above=2)

    result = run_resumable_inference(
        detector,
        images,
        InferenceConfig(model_id="batch-recording", batch_size=4, device="cuda", run_id="run-oom"),
        tmp_path / "oom-artifacts",
    )

    assert [len(call) for call in detector.calls] == [4, 2, 2]
    assert result.completed_images == 4
    assert result.failed_images == []


def test_prediction_progress_streams_batch_counts_before_run_finishes(tmp_path: Path):
    images = _images(tmp_path, 4)
    second_batch_started = threading.Event()
    finish_second_batch = threading.Event()
    first_batch_reported = threading.Event()
    progress = []

    class DelayedSecondBatchDetector(BatchRecordingDetector):
        def predict(self, image_paths, config):
            paths = list(image_paths)
            self.calls.append([path.name for path in paths])
            if paths[0].name == "02.jpg":
                second_batch_started.set()
                finish_second_batch.wait(timeout=3)
            for path in paths:
                yield DetectionRecord(
                    image_path=str(path),
                    image_id=path.name,
                    class_id=0,
                    class_name="toad",
                    confidence=0.9,
                    bbox_xyxy=[1, 2, 11, 22],
                    image_width=20,
                    image_height=40,
                    model_id=self.model_id,
                    run_id=config.run_id,
                )

    def report(event):
        progress.append(event)
        if event.get("phase") == "prediction" and event.get("completed") == 2:
            first_batch_reported.set()

    detector = DelayedSecondBatchDetector()
    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            run_resumable_inference(
                detector,
                images,
                InferenceConfig(
                    model_id="batch-recording",
                    batch_size=2,
                    device="mps",
                    run_id="run-streaming",
                ),
                tmp_path / "streaming-artifacts",
                progress_callback=report,
            )
        )
    )
    worker.start()
    try:
        assert first_batch_reported.wait(timeout=2)
        assert second_batch_started.wait(timeout=2)
        assert worker.is_alive()
        update = next(
            event
            for event in reversed(progress)
            if event.get("phase") == "prediction" and event.get("completed") == 2
        )
        assert update["total"] == 4
        assert update["remaining"] == 2
        assert update["message"] == "Predicted 2 of 4 images"
        assert all(image.name not in repr(update) for image in images)
    finally:
        finish_second_batch.set()
        worker.join(timeout=3)

    assert not worker.is_alive()
    assert result[0].completed_images == 4
