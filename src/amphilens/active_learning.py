"""Paper-compatible Hybrid PPAL active-learning selection."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .core import DetectionRecord, ValidationError


class CalibrationRequiredError(ValidationError):
    """Raised when default PPAL calibration evidence is absent."""


class FeatureRequiredError(ValidationError):
    """Raised when CCMS cannot perform diversity selection."""


class MissingCalibrationClassError(CalibrationRequiredError):
    """Raised when one of the selected classes lacks calibration evidence."""


@dataclass(slots=True)
class PPALCalibration:
    classes: list[str]
    difficulties: dict[str, float]
    weights: dict[str, float]
    xi: float
    alpha: float
    beta: float
    source: str = "validation"


def ppal_instance_difficulty(confidence: float, iou: float, xi: float = 0.5) -> float:
    if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
        raise ValidationError("confidence and IoU must be between 0 and 1")
    if not 0 <= xi <= 1:
        raise ValidationError("xi must be between 0 and 1")
    return 1.0 - confidence**xi * iou ** (1.0 - xi)


def calibrate_ppal(
    evidence: Iterable[dict],
    *,
    classes: Iterable[str],
    xi: float = 0.5,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> PPALCalibration:
    selected_classes = list(classes)
    rows = list(evidence)
    if not rows:
        raise CalibrationRequiredError(
            "Hybrid PPAL requires validation evidence; provide matched "
            "validation difficulty records"
        )
    by_class: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        name = str(row.get("class_name", "")).strip()
        if name in selected_classes:
            value = float(row["difficulty"])
            if not 0 <= value <= 1:
                raise ValidationError("Calibration difficulty must be between 0 and 1")
            by_class[name].append(value)
    missing = [name for name in selected_classes if not by_class.get(name)]
    if missing:
        raise MissingCalibrationClassError(
            "Missing PPAL calibration evidence for: " + ", ".join(missing)
        )
    difficulties = {name: sum(values) / len(values) for name, values in by_class.items()}
    gamma = math.exp(1.0 / alpha) - 1.0
    weights = {
        name: 1.0 + (alpha / beta) * math.log(1.0 + gamma * difficulty)
        for name, difficulty in difficulties.items()
    }
    return PPALCalibration(selected_classes, difficulties, weights, xi, alpha, beta)


def calibrate_ppal_from_matches(
    matches: Iterable[dict],
    *,
    classes: Iterable[str],
    xi: float = 0.5,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> PPALCalibration:
    """Convert matched validation predictions into the paper's PPAL evidence."""
    evidence = []
    for match in matches:
        evidence.append(
            {
                "class_name": match["class_name"],
                "difficulty": ppal_instance_difficulty(
                    float(match["confidence"]), float(match["iou"]), xi
                ),
            }
        )
    return calibrate_ppal(evidence, classes=classes, xi=xi, alpha=alpha, beta=beta)


@dataclass(slots=True)
class HybridPPALConfig:
    budget: int = 100
    pool_multiplier: int = 200
    uncertain_ratio: float = 0.4
    certain_ratio: float = 0.5
    random_ratio: float = 0.1
    seed: int = 42
    priority_class: str | None = None
    priority_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.budget <= 0 or self.pool_multiplier <= 0:
            raise ValidationError("budget and pool_multiplier must be positive")
        if abs(self.uncertain_ratio + self.certain_ratio + self.random_ratio - 1.0) > 1e-6:
            raise ValidationError("PPAL pool ratios must sum to 1")


@dataclass(slots=True)
class SelectedImage:
    image_path: str
    class_name: str
    score: float
    curation_reason: str


def _entropy(confidence: float, classes: int) -> float:
    if classes <= 1:
        return 0.0
    p = min(max(confidence, 1e-6), 1 - 1e-6)
    other = min(max((1.0 - p) / (classes - 1), 1e-6), 1 - 1e-6)
    return -p * math.log(p) - (classes - 1) * other * math.log(other)


def _cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 1.0
    return 1.0 - dot / (left_norm * right_norm)


class HybridPPALStrategy:
    """DCUS-style uncertainty followed by CCMS-style diversity selection."""

    def __init__(self, config: HybridPPALConfig | None = None):
        self.config = config or HybridPPALConfig()

    def select(
        self,
        predictions: Iterable[DetectionRecord],
        calibration: PPALCalibration,
        *,
        features: dict[str, list[float]],
    ) -> list[SelectedImage]:
        records = list(predictions)
        if not records:
            return []
        grouped: dict[str, list[DetectionRecord]] = defaultdict(list)
        for record in records:
            grouped[record.image_path].append(record)
        missing_features = [path for path in grouped if path not in features]
        if missing_features:
            raise FeatureRequiredError(f"Missing CCMS features for {missing_features[0]}")

        image_scores: list[tuple[str, str, float]] = []
        for image_path, image_records in grouped.items():
            best = max(image_records, key=lambda item: item.confidence or 0.0)
            score = sum(
                calibration.weights.get(record.class_name, 1.0)
                * (
                    self.config.priority_weight
                    if record.class_name == self.config.priority_class
                    else 1.0
                )
                * _entropy(record.confidence or 0.0, len(calibration.classes))
                for record in image_records
            )
            image_scores.append((image_path, best.class_name, score))
        image_scores.sort(key=lambda row: (-row[2], row[0]))
        pool = image_scores[
            : min(len(image_scores), self.config.budget * self.config.pool_multiplier)
        ]
        if not pool:
            return []

        chosen: list[tuple[str, str, float, str]] = []
        remaining = list(pool)
        available_by_class: dict[str, tuple[str, str, float]] = {}
        for candidate in pool:
            available_by_class.setdefault(candidate[1], candidate)
        for candidate in sorted(
            available_by_class.values(), key=lambda row: (-row[2], row[1], row[0])
        ):
            if len(chosen) >= min(self.config.budget, len(pool)):
                break
            chosen.append((*candidate, "hybrid_ppal:class_coverage+dcus_uncertainty"))
            remaining.remove(candidate)
        while remaining and len(chosen) < min(self.config.budget, len(pool)):
            best_candidate = max(
                remaining,
                key=lambda candidate: (
                    min(
                        _cosine_distance(features[candidate[0]], features[item[0]])
                        for item in chosen
                    ),
                    candidate[2],
                    candidate[0],
                ),
            )
            remaining.remove(best_candidate)
            chosen.append((*best_candidate, "hybrid_ppal:dcus_uncertainty+ccms_diversity"))
        return [SelectedImage(*item) for item in chosen]
