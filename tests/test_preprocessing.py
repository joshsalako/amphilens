from pathlib import Path

from PIL import Image

from amphilens.preprocessing import PreprocessingConfig, PreprocessingService


def test_max_dimension_resize_preserves_aspect_ratio_without_upscaling(tmp_path: Path):
    source = tmp_path / "wide.png"
    Image.new("RGB", (1200, 600), color=(10, 20, 30)).save(source)

    service = PreprocessingService(PreprocessingConfig(max_dimension=640))
    result = service.transform(source)

    assert result.original_size == (1200, 600)
    assert result.image.size == (640, 320)
    assert result.scale == 640 / 1200

    small = tmp_path / "small.png"
    Image.new("RGB", (320, 200), color=(10, 20, 30)).save(small)
    assert service.transform(small).image.size == (320, 200)


def test_grayscale_is_replicated_to_three_channels(tmp_path: Path):
    source = tmp_path / "colour.png"
    Image.new("RGB", (12, 8), color=(200, 10, 10)).save(source)

    result = PreprocessingService(PreprocessingConfig()).transform(source)

    assert result.image.mode == "RGB"
    assert result.image.getpixel((0, 0))[0] == result.image.getpixel((0, 0))[1]
    assert result.image.getpixel((0, 0))[1] == result.image.getpixel((0, 0))[2]


def test_resize_precedes_clahe_and_fingerprint_is_stable(tmp_path: Path):
    import pytest

    pytest.importorskip("cv2")
    source = tmp_path / "trap.png"
    Image.new("RGB", (1200, 600), color=(80, 80, 80)).save(source)
    config = PreprocessingConfig(max_dimension=640, clahe_enabled=True)

    result = PreprocessingService(config).transform(source)

    assert result.image.size == (640, 320)
    assert config.fingerprint == PreprocessingConfig.from_dict(config.to_dict()).fingerprint
    assert config.to_dict()["clahe_clip_limit"] == 2.0
    assert config.to_dict()["clahe_tile_grid_size"] == [8, 8]


def test_preprocessed_cache_uses_source_hash_and_config_fingerprint(tmp_path: Path):
    source = tmp_path / "trap.png"
    Image.new("RGB", (20, 10), color="black").save(source)
    cache = tmp_path / "cache"
    service = PreprocessingService(PreprocessingConfig(max_dimension=10), cache_dir=cache)

    first = service.materialize(source)
    second = service.materialize(source)

    assert first == second
    assert first.is_file()
    assert source.read_bytes() != first.read_bytes()


def test_research_short_side_profile_rounds_dimensions_and_maps_each_axis(tmp_path: Path):
    import pytest

    pytest.importorskip("cv2")
    source = tmp_path / "wide.png"
    Image.new("RGB", (1000, 501), color=(20, 30, 40)).save(source)
    config = PreprocessingConfig(
        max_dimension=640,
        resize_interpolation="opencv-linear",
        grayscale_enabled=True,
        clahe_enabled=True,
        compatibility_mode="shortest-side",
        round_to_multiple=32,
        allow_upscale=True,
    )

    result = PreprocessingService(config).transform(source)

    assert result.image.size == (1280, 640)
    assert result.map_box_to_original([0, 0, 1280, 640]) == [0, 0, 1000, 501]


def test_research_max_dimension_profile_can_upscale_and_round_short_axis(tmp_path: Path):
    import pytest

    pytest.importorskip("cv2")
    source = tmp_path / "portrait.png"
    Image.new("RGB", (501, 1000), color=(20, 30, 40)).save(source)
    config = PreprocessingConfig(
        max_dimension=640,
        resize_interpolation="opencv-linear",
        grayscale_enabled=True,
        clahe_enabled=True,
        compatibility_mode="max-dimension",
        round_to_multiple=32,
        allow_upscale=True,
    )

    result = PreprocessingService(config).transform(source)

    assert result.image.size == (352, 640)
