import json
from pathlib import Path

from amphilens.core import DetectionRecord
from amphilens.inference import write_predictions_csv
from amphilens.reporting import write_report


def test_report_contains_counts_and_provenance(tmp_path: Path):
    csv_path = tmp_path / "predictions.csv"
    write_predictions_csv(
        [
            DetectionRecord(
                image_path="/pool/a.jpg", image_id="a.jpg", class_id=0, class_name="toad",
                confidence=0.8, bbox_xyxy=[1, 1, 5, 5], image_width=10, image_height=10,
                model_id="fixture", run_id="run-1",
            ),
            DetectionRecord(
                image_path="/pool/b.jpg", image_id="b.jpg", class_id=1, class_name="frog",
                confidence=0.7, bbox_xyxy=[2, 2, 6, 6], image_width=10, image_height=10,
                model_id="fixture", run_id="run-1",
            ),
        ],
        csv_path,
    )

    outputs = write_report(csv_path, tmp_path / "report")
    summary = json.loads(outputs["summary_json"].read_text())
    assert summary["image_count"] == 2
    assert summary["detection_count"] == 2
    assert summary["class_counts"] == {"frog": 1, "toad": 1}
    assert "fixture" in outputs["markdown"].read_text()

