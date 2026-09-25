import pytest

from amphilens.ui import parse_classes


def test_parse_classes_normalizes_and_rejects_duplicates():
    assert parse_classes("toad, frog\nother") == ["toad", "frog", "other"]
    with pytest.raises(ValueError, match="unique"):
        parse_classes("toad, toad")


def test_parse_classes_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        parse_classes("\n, ")
