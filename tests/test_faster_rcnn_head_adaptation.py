from __future__ import annotations

from types import SimpleNamespace

import pytest

from amphilens.models.faster_rcnn_training import adapt_faster_rcnn_state_dict


def _tensor(shape):
    return SimpleNamespace(shape=shape)


def test_faster_rcnn_head_adaptation_keeps_compatible_backbone_and_reinitializes_labels():
    source_state = {
        "backbone.body.conv1.weight": _tensor((64, 3, 7, 7)),
        "roi_heads.box_predictor.cls_score.weight": _tensor((4, 1024)),
        "roi_heads.box_predictor.cls_score.bias": _tensor((4,)),
        "roi_heads.box_predictor.bbox_pred.weight": _tensor((16, 1024)),
        "roi_heads.box_predictor.bbox_pred.bias": _tensor((16,)),
    }
    target_state = {
        "backbone.body.conv1.weight": _tensor((64, 3, 7, 7)),
        "roi_heads.box_predictor.cls_score.weight": _tensor((3, 1024)),
        "roi_heads.box_predictor.cls_score.bias": _tensor((3,)),
        "roi_heads.box_predictor.bbox_pred.weight": _tensor((12, 1024)),
        "roi_heads.box_predictor.bbox_pred.bias": _tensor((12,)),
    }

    compatible = adapt_faster_rcnn_state_dict(
        source_state,
        target_state,
        source_classes=["frog", "mammal", "toad"],
        target_classes=["toad", "mammal"],
    )

    assert set(compatible) == {"backbone.body.conv1.weight"}
    assert compatible["backbone.body.conv1.weight"] is source_state["backbone.body.conv1.weight"]


def test_faster_rcnn_head_adaptation_reinitializes_head_when_class_order_changes():
    source = {
        "backbone.body.conv1.weight": _tensor((64, 3, 7, 7)),
        "roi_heads.box_predictor.cls_score.weight": _tensor((4, 1024)),
        "roi_heads.box_predictor.cls_score.bias": _tensor((4,)),
        "roi_heads.box_predictor.bbox_pred.weight": _tensor((16, 1024)),
        "roi_heads.box_predictor.bbox_pred.bias": _tensor((16,)),
    }

    compatible = adapt_faster_rcnn_state_dict(
        source,
        source,
        source_classes=["frog", "mammal", "toad"],
        target_classes=["toad", "mammal", "frog"],
    )

    assert set(compatible) == {"backbone.body.conv1.weight"}


def test_faster_rcnn_head_adaptation_rejects_non_head_shape_mismatch():
    with pytest.raises(ValueError, match="backbone.body.conv1.weight"):
        adapt_faster_rcnn_state_dict(
            {"backbone.body.conv1.weight": _tensor((32, 3, 7, 7))},
            {"backbone.body.conv1.weight": _tensor((64, 3, 7, 7))},
            source_classes=["toad"],
            target_classes=["toad"],
        )
