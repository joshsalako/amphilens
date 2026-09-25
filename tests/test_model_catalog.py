import pytest

from amphilens.core import ValidationError
from amphilens.models import load_preset_detector
from amphilens.models.catalog import ModelCatalog, ModelPreset, ModelSource


def test_catalog_exposes_supported_large_presets_and_product_default():
    catalog = ModelCatalog()

    assert catalog.default.model_id == "yolo26-l"
    assert catalog.get("yolo26-l").checkpoint_reference == "yolo26l.pt"
    assert catalog.get("rtdetr-l").checkpoint_reference == "rtdetr-l.pt"
    assert catalog.get("faster-rcnn-resnet50").architecture == "faster_rcnn"
    assert all(item.checkpoint_source == "official-general-purpose" for item in catalog.list())


def test_model_source_records_paper_status_without_claiming_equivalence():
    source = ModelSource(
        kind="paper-specific",
        identifier="pending-user-supplied-url",
        is_paper_specific=True,
    )

    assert source.is_paper_specific is True
    assert source.identifier.startswith("pending-")


def test_catalog_rejects_unknown_or_unsupported_presets():
    catalog = ModelCatalog()
    with pytest.raises(ValidationError, match="Unknown model preset"):
        catalog.get("faster-rcnn-large")

    with pytest.raises(ValidationError, match="architecture"):
        ModelPreset(
            model_id="bad",
            architecture="unknown",
            size="large",
            checkpoint_reference="bad.pt",
            checkpoint_source="official-general-purpose",
        )


def test_preset_factory_keeps_official_weights_lazy_and_records_model_id():
    detector = load_preset_detector(ModelCatalog().get("yolo26-l"), classes=["toad"])

    assert detector.model_id == "yolo26-l"
    assert detector.architecture == "yolo"
