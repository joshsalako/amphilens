import json
from pathlib import Path

from amphilens.active_learning import HybridPPALConfig, SelectedImage, calibrate_ppal
from amphilens.curation import write_selection_artifacts


def test_selection_artifacts_are_auditable(tmp_path: Path):
    calibration = calibrate_ppal(
        [{"class_name": "toad", "difficulty": 0.4}], classes=["toad"]
    )
    selected = [SelectedImage("/source/a.jpg", "toad", 0.72, "hybrid_ppal:dcus_uncertainty")]

    artifacts = write_selection_artifacts(
        selected, calibration, HybridPPALConfig(budget=1), tmp_path / "cycle-0"
    )

    assert artifacts["queue_csv"].is_file()
    assert "a.jpg" in artifacts["queue_csv"].read_text()
    metadata = json.loads(artifacts["selection_json"].read_text())
    assert metadata["config"]["budget"] == 1
    assert metadata["calibration"]["source"] == "validation"

