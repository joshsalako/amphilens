from pathlib import Path

from PIL import Image

from amphilens.preprocessing import PreprocessingConfig, PreprocessingService


def test_default_resize_makes_short_side_640_and_preserves_aspect_ratio(tmp_path: Path):
    expected = {
        (1200, 600): (1280, 640),
        (600, 1200): (640, 1280),
        (600, 600): (640, 640),
        (320, 200): (1024, 640),
    }
    service = PreprocessingService()

    for index, (source_size, expected_size) in enumerate(expected.items()):
        source = tmp_path / f"image-{index}.png"
        Image.new("RGB", source_size, color=(10, 20, 30)).save(source)

        result = service.transform(source)

        assert result.image.size == expected_size
        assert min(result.image.size) == 640
        assert (
            abs(result.image.width / result.image.height - source_size[0] / source_size[1]) < 0.002
        )


def test_preprocessing_config_serializes_canonical_short_side_field():
    config = PreprocessingConfig(short_side_dimension=512)

    assert config.to_dict()["short_side_dimension"] == 512
    assert "max_dimension" not in config.to_dict()
    assert PreprocessingConfig.from_dict(config.to_dict()).short_side_dimension == 512


def test_legacy_preprocessing_config_migrates_to_short_side_resize():
    config = PreprocessingConfig.from_dict(
        {
            "max_dimension": 640,
            "compatibility_mode": "max-dimension",
            "round_to_multiple": 32,
            "allow_upscale": False,
        }
    )

    assert config.short_side_dimension == 640
    assert "compatibility_mode" not in config.to_dict()
    assert "round_to_multiple" not in config.to_dict()
    assert "allow_upscale" not in config.to_dict()


def test_padding_tracks_offsets_for_prediction_box_mapping(tmp_path: Path):
    source = tmp_path / "wide.png"
    Image.new("RGB", (1200, 600), color=(10, 20, 30)).save(source)
    transformed = PreprocessingService().transform(source)

    padded = transformed.pad_to(1312, 672, centered=True)

    assert padded.image.size == (1312, 672)
    assert padded.map_box_to_original([16, 16, 656, 336]) == [0, 0, 600, 300]
    assert padded.map_box_to_processed([0, 0, 600, 300]) == [16, 16, 656, 336]


def test_short_side_resize_preserves_aspect_ratio_and_upscales(tmp_path: Path):
    source = tmp_path / "wide.png"
    Image.new("RGB", (1200, 600), color=(10, 20, 30)).save(source)

    service = PreprocessingService(PreprocessingConfig(short_side_dimension=640))
    result = service.transform(source)

    assert result.original_size == (1200, 600)
    assert result.image.size == (1280, 640)
    assert result.scale == 640 / 600

    small = tmp_path / "small.png"
    Image.new("RGB", (320, 200), color=(10, 20, 30)).save(small)
    assert service.transform(small).image.size == (1024, 640)


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

    assert result.image.size == (1280, 640)
    assert config.fingerprint == PreprocessingConfig.from_dict(config.to_dict()).fingerprint
    assert config.to_dict()["clahe_clip_limit"] == 2.0
    assert config.to_dict()["clahe_tile_grid_size"] == [8, 8]


def test_preprocessed_cache_uses_source_hash_and_config_fingerprint(tmp_path: Path):
    source = tmp_path / "trap.png"
    Image.new("RGB", (20, 10), color="black").save(source)
    cache = tmp_path / "cache"
    service = PreprocessingService(PreprocessingConfig(short_side_dimension=5), cache_dir=cache)

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
        short_side_dimension=640,
        resize_interpolation="opencv-linear",
        grayscale_enabled=True,
        clahe_enabled=True,
        compatibility_mode="shortest-side",
        round_to_multiple=32,
        allow_upscale=True,
    )

    result = PreprocessingService(config).transform(source)

    assert result.image.size == (1277, 640)
    assert result.map_box_to_original([0, 0, 1277, 640]) == [0, 0, 1000, 501]


def test_short_side_profile_resizes_portrait_images_without_dimension_rounding(tmp_path: Path):
    import pytest

    pytest.importorskip("cv2")
    source = tmp_path / "portrait.png"
    Image.new("RGB", (501, 1000), color=(20, 30, 40)).save(source)
    config = PreprocessingConfig(
        short_side_dimension=640,
        resize_interpolation="opencv-linear",
        grayscale_enabled=True,
        clahe_enabled=True,
        compatibility_mode="max-dimension",
        round_to_multiple=32,
        allow_upscale=True,
    )

    result = PreprocessingService(config).transform(source)

    assert result.image.size == (640, 1277)
