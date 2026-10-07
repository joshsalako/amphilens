from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


class _ConvHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = torch.nn.Conv2d(4, 4, 1)
        self.cv3 = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(4, 2, 1))])
        self.one2one_cv3 = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(4, 2, 1))])


class _RTDETRHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_score_head = torch.nn.Linear(4, 2)
        self.dec_score_head = torch.nn.ModuleList([torch.nn.Linear(4, 2)])
        self.denoising_class_embed = torch.nn.Embedding(2, 4)
        self.dec_bbox_head = torch.nn.ModuleList([torch.nn.Linear(4, 4)])


@pytest.mark.parametrize("architecture", ["yolo", "rtdetr"])
def test_ultralytics_head_adaptation_resets_class_outputs_but_keeps_features(architecture):
    from amphilens.models.backends import reset_ultralytics_classification_head

    head = _ConvHead() if architecture == "yolo" else _RTDETRHead()
    if architecture == "yolo":
        classifier = head.cv3[0][0]
        feature = head.feature
    else:
        classifier = head.enc_score_head
        feature = head.dec_bbox_head[0]
    classifier.weight.data.fill_(1)
    classifier_bias = getattr(classifier, "bias", None)
    if classifier_bias is not None:
        classifier_bias.data.fill_(1)
    feature_before = feature.weight.detach().clone()
    expected_head = SimpleNamespace(
        model=SimpleNamespace(model=[head]),
        names={0: "old_a", 1: "old_b"},
    )

    reset = reset_ultralytics_classification_head(
        expected_head,
        architecture=architecture,
        source_classes=["old_a", "old_b"],
        target_classes=["new_a", "new_b"],
    )

    assert reset is True
    assert not torch.all(classifier.weight == 1)
    assert torch.equal(feature.weight, feature_before)


def test_ultralytics_head_is_preserved_when_class_order_matches():
    from amphilens.models.backends import reset_ultralytics_classification_head

    head = _ConvHead()
    head.cv3[0][0].weight.data.fill_(1)
    detector = SimpleNamespace(model=SimpleNamespace(model=[head]))

    reset = reset_ultralytics_classification_head(
        detector,
        architecture="yolo",
        source_classes=["toad", "mammal"],
        target_classes=["toad", "mammal"],
    )

    assert reset is False
    assert torch.all(head.cv3[0][0].weight == 1)
