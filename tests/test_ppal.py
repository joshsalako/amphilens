import pytest

from amphilens.active_learning import (
    CalibrationRequiredError,
    HybridPPALConfig,
    HybridPPALStrategy,
    calibrate_ppal,
)
from amphilens.core import DetectionRecord


def _record(image: str, confidence: float, class_name: str = "toad", class_id: int = 0):
    return DetectionRecord(
        image_path=f"/pool/{image}.jpg",
        image_id=f"{image}.jpg",
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=[0, 0, 10, 10],
        image_width=20,
        image_height=20,
        model_id="fixture",
        run_id="run-1",
    )


def test_default_ppal_requires_calibration_evidence():
    with pytest.raises(CalibrationRequiredError):
        calibrate_ppal([], classes=["toad"])


def test_hybrid_ppal_returns_budgeted_diverse_images():
    predictions = [_record("a", 0.51), _record("b", 0.52), _record("c", 0.95)]
    calibration = calibrate_ppal(
        [{"class_name": "toad", "difficulty": 0.4}], classes=["toad"]
    )
    strategy = HybridPPALStrategy(HybridPPALConfig(budget=2, seed=7))
    selected = strategy.select(
        predictions,
        calibration,
        features={
            "/pool/a.jpg": [1.0, 0.0],
            "/pool/b.jpg": [0.0, 1.0],
            "/pool/c.jpg": [1.0, 1.0],
        },
    )

    assert len(selected) == 2
    assert len({item.image_path for item in selected}) == 2
    assert all(item.curation_reason for item in selected)


def test_hybrid_ppal_covers_available_classes_before_filling_budget():
    predictions = [
        _record("toad-1", 0.51, "toad", 0),
        _record("toad-2", 0.52, "toad", 0),
        _record("frog-1", 0.99, "frog", 1),
    ]
    calibration = calibrate_ppal(
        [
            {"class_name": "toad", "difficulty": 0.4},
            {"class_name": "frog", "difficulty": 0.4},
        ],
        classes=["toad", "frog"],
    )
    selected = HybridPPALStrategy(HybridPPALConfig(budget=2)).select(
        predictions,
        calibration,
        features={
            "/pool/toad-1.jpg": [1.0, 0.0],
            "/pool/toad-2.jpg": [0.0, 1.0],
            "/pool/frog-1.jpg": [0.9, 0.1],
        },
    )

    assert {item.class_name for item in selected} == {"toad", "frog"}
