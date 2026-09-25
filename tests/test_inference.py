from pathlib import Path

from amphilens.core import DetectionRecord, InferenceConfig
from amphilens.inference import run_inference, write_predictions_csv


class FixtureDetector:
    model_id = "fixture"

    def predict(self, image_paths, config):
        for path in image_paths:
            yield DetectionRecord(
                image_path=str(path),
                image_id=Path(path).name,
                class_id=0,
                class_name="toad",
                confidence=0.9,
                bbox_xyxy=[1, 2, 11, 22],
                image_width=20,
                image_height=40,
                model_id=self.model_id,
                run_id=config.run_id,
            )


def test_inference_writes_stable_csv(tmp_path: Path):
    images = []
    for name in ("a.jpg", "b.jpg"):
        path = tmp_path / name
        path.write_bytes(b"fixture")
        images.append(path)

    config = InferenceConfig(model_id="fixture", image_size=640, confidence=0.2, run_id="run-1")
    records = list(run_inference(FixtureDetector(), images, config))
    output = tmp_path / "predictions.csv"
    write_predictions_csv(records, output)

    assert len(records) == 2
    text = output.read_text()
    assert "image_path" in text
    assert "bbox_xmin" in text
    assert "fixture" in text

