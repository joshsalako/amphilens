import json
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

    second = run_resumable_inference(
        RecoverableDetector(), images, config, tmp_path / "artifacts"
    )
    assert second.completed_images == 2
    assert second.failed_images == []
    rows = (tmp_path / "artifacts" / "predictions.csv").read_text().splitlines()
    assert len(rows) == 3
    progress = json.loads((tmp_path / "artifacts" / "progress.json").read_text())
    assert len(progress["completed_images"]) == 2

