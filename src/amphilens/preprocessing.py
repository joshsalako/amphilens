"""Shared, reproducible image preprocessing for training and inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class PreprocessingDependencyError(RuntimeError):
    """Raised when an enabled preprocessing operation needs an unavailable library."""


@dataclass(frozen=True, slots=True, init=False)
class PreprocessingConfig:
    """Versioned preprocessing choices shared by every model backend."""

    resize_enabled: bool = True
    short_side_dimension: int = 640
    resize_interpolation: str = "lanczos"
    grayscale_enabled: bool = True
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    color_space: str = "rgb"

    def __init__(
        self,
        short_side_dimension: int = 640,
        resize_enabled: bool = True,
        resize_interpolation: str = "lanczos",
        grayscale_enabled: bool = True,
        clahe_enabled: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
        color_space: str = "rgb",
        *,
        max_dimension: int | None = None,
        compatibility_mode: str | None = None,
        round_to_multiple: int | None = None,
        allow_upscale: bool | None = None,
    ) -> None:
        """Build the canonical short-side config, accepting legacy keyword aliases."""
        del resize_enabled, compatibility_mode, round_to_multiple, allow_upscale
        if max_dimension is not None:
            if short_side_dimension != 640 and short_side_dimension != max_dimension:
                raise ValueError("Specify only one short-side dimension")
            short_side_dimension = max_dimension
        object.__setattr__(self, "resize_enabled", True)
        object.__setattr__(self, "short_side_dimension", short_side_dimension)
        object.__setattr__(self, "resize_interpolation", resize_interpolation)
        object.__setattr__(self, "grayscale_enabled", grayscale_enabled)
        object.__setattr__(self, "clahe_enabled", clahe_enabled)
        object.__setattr__(self, "clahe_clip_limit", clahe_clip_limit)
        object.__setattr__(self, "clahe_tile_grid_size", tuple(clahe_tile_grid_size))
        object.__setattr__(self, "color_space", color_space)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.short_side_dimension <= 0:
            raise ValueError("short_side_dimension must be positive")
        if self.resize_interpolation not in {
            "nearest",
            "bilinear",
            "bicubic",
            "lanczos",
            "opencv-linear",
        }:
            raise ValueError(
                "resize_interpolation must be nearest, bilinear, bicubic, lanczos, or opencv-linear"
            )
        if self.clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive")
        if len(self.clahe_tile_grid_size) != 2 or any(
            value <= 0 for value in self.clahe_tile_grid_size
        ):
            raise ValueError("clahe_tile_grid_size must contain two positive integers")
        if self.color_space not in {"rgb", "lab"}:
            raise ValueError("color_space must be rgb or lab")

    @property
    def max_dimension(self) -> int:
        """Deprecated compatibility alias for the target short side."""
        return self.short_side_dimension

    @property
    def compatibility_mode(self) -> str:
        return "shortest-side"

    @property
    def round_to_multiple(self) -> None:
        return None

    @property
    def allow_upscale(self) -> bool:
        return True

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(include_fingerprint=False), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict:
        data = {
            "resize_enabled": self.resize_enabled,
            "short_side_dimension": self.short_side_dimension,
            "resize_interpolation": self.resize_interpolation,
            "grayscale_enabled": self.grayscale_enabled,
            "clahe_enabled": self.clahe_enabled,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_grid_size": list(self.clahe_tile_grid_size),
            "color_space": self.color_space,
        }
        if include_fingerprint:
            data["fingerprint"] = self.fingerprint
        return data

    @classmethod
    def from_dict(cls, data: dict | None) -> PreprocessingConfig:
        if not data:
            return cls()
        values = dict(data)
        values.pop("fingerprint", None)
        if "short_side_dimension" not in values and "max_dimension" in values:
            values["short_side_dimension"] = values.pop("max_dimension")
        else:
            values.pop("max_dimension", None)
        values.pop("compatibility_mode", None)
        values.pop("round_to_multiple", None)
        values.pop("allow_upscale", None)
        # The previous app stored a label. Preserve the useful named behavior.
        if set(values) == {"name"}:
            return cls(clahe_enabled=values["name"].strip().lower() == "clahe")
        if "clahe_tile_grid_size" in values:
            values["clahe_tile_grid_size"] = tuple(values["clahe_tile_grid_size"])
        return cls(**values)

    @classmethod
    def from_any(cls, value: PreprocessingConfig | dict | str | None) -> PreprocessingConfig:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(clahe_enabled=value.strip().lower() == "clahe")
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    image: object
    original_size: tuple[int, int]
    scale: float
    scale_x: float | None = None
    scale_y: float | None = None
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def processed_size(self) -> tuple[int, int]:
        return tuple(self.image.size)

    def clip_box_to_content(self, box_xyxy: list[float] | tuple[float, ...]) -> list[float] | None:
        """Clip model boxes to resized image content, excluding any padding bars."""
        if len(box_xyxy) != 4:
            raise ValueError("box_xyxy must contain four coordinates")
        width, height = self.processed_size
        left, top, right, bottom = self.padding
        clipped = [
            max(float(box_xyxy[0]), left),
            max(float(box_xyxy[1]), top),
            min(float(box_xyxy[2]), width - right),
            min(float(box_xyxy[3]), height - bottom),
        ]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        return clipped

    def map_box_to_original(self, box_xyxy: list[float] | tuple[float, ...]) -> list[float]:
        if len(box_xyxy) != 4:
            raise ValueError("box_xyxy must contain four coordinates")
        scale_x = self.scale if self.scale_x is None else self.scale_x
        scale_y = self.scale if self.scale_y is None else self.scale_y
        left, top, _, _ = self.padding
        if scale_x == 1 and scale_y == 1 and left == 0 and top == 0:
            return [float(value) for value in box_xyxy]
        return [
            round((float(box_xyxy[0]) - left) / scale_x, 6),
            round((float(box_xyxy[1]) - top) / scale_y, 6),
            round((float(box_xyxy[2]) - left) / scale_x, 6),
            round((float(box_xyxy[3]) - top) / scale_y, 6),
        ]

    def map_box_to_processed(self, box_xyxy: list[float] | tuple[float, ...]) -> list[float]:
        if len(box_xyxy) != 4:
            raise ValueError("box_xyxy must contain four coordinates")
        scale_x = self.scale if self.scale_x is None else self.scale_x
        scale_y = self.scale if self.scale_y is None else self.scale_y
        left, top, _, _ = self.padding
        return [
            round(float(box_xyxy[0]) * scale_x + left, 6),
            round(float(box_xyxy[1]) * scale_y + top, 6),
            round(float(box_xyxy[2]) * scale_x + left, 6),
            round(float(box_xyxy[3]) * scale_y + top, 6),
        ]

    def pad_to(
        self,
        width: int,
        height: int,
        *,
        fill: tuple[int, int, int] = (114, 114, 114),
        centered: bool = True,
    ) -> PreprocessedImage:
        """Pad without scaling and retain offsets for mapping model predictions."""
        current_width, current_height = self.processed_size
        if width < current_width or height < current_height:
            raise ValueError("Padding size cannot be smaller than the processed image")
        left = (width - current_width) // 2 if centered else 0
        top = (height - current_height) // 2 if centered else 0
        right = width - current_width - left
        bottom = height - current_height - top
        from PIL import Image

        canvas = Image.new("RGB", (width, height), fill)
        canvas.paste(self.image, (left, top))
        return PreprocessedImage(
            image=canvas,
            original_size=self.original_size,
            scale=self.scale,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            padding=(left, top, right, bottom),
        )


class PreprocessingService:
    """Apply the shared resize -> grayscale -> CLAHE pipeline without mutating sources."""

    def __init__(
        self, config: PreprocessingConfig | None = None, cache_dir: str | Path | None = None
    ):
        self.config = config or PreprocessingConfig()
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None

    def transform(self, source: str | Path) -> PreprocessedImage:
        image = self._load(source)
        original_size = image.size
        scale = self._resize_scale(image.size)
        target_size = self._target_size(image.size, scale)
        if target_size != image.size:
            image = self._resize(image, target_size)
        image = self._to_three_channel(image)
        if self.config.clahe_enabled:
            image = self._apply_clahe(image)
        return PreprocessedImage(
            image=image,
            original_size=original_size,
            scale=scale,
            scale_x=image.width / original_size[0],
            scale_y=image.height / original_size[1],
        )

    def materialize(self, source: str | Path) -> Path:
        source_path = Path(source).expanduser().resolve()
        if self.cache_dir is None:
            raise ValueError("cache_dir is required to materialize preprocessed images")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:16]
        destination = self.cache_dir / f"{digest}-{self.config.fingerprint}.png"
        if not destination.is_file():
            temporary = destination.with_suffix(".tmp.png")
            self.transform(source_path).image.save(temporary, format="PNG")
            temporary.replace(destination)
        return destination

    @staticmethod
    def _load(source: str | Path):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - exercised in minimal installs
            raise PreprocessingDependencyError(
                "Image preprocessing requires Pillow; install 'amphilens[inference]'"
            ) from exc
        try:
            return Image.open(source).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - normalize image decoder errors
            raise ValueError(f"Cannot read image: {source}") from exc

    def _resize_scale(self, size: tuple[int, int]) -> float:
        if not self.config.resize_enabled:
            return 1.0
        width, height = size
        return self.config.short_side_dimension / min(width, height)

    def _target_size(self, size: tuple[int, int], scale: float) -> tuple[int, int]:
        if not self.config.resize_enabled:
            return size
        width, height = size
        if width <= height:
            return self.config.short_side_dimension, max(1, round(height * scale))
        return max(1, round(width * scale)), self.config.short_side_dimension

    def _resize(self, image, target_size: tuple[int, int]):
        if self.config.resize_interpolation == "opencv-linear":
            try:
                import cv2
                import numpy as np
            except ImportError as exc:  # pragma: no cover - inference extra supplies OpenCV
                raise PreprocessingDependencyError(
                    "OpenCV linear resize requires 'opencv-python-headless'; "
                    "install 'amphilens[inference]'"
                ) from exc
            resized = cv2.resize(np.asarray(image), target_size, interpolation=cv2.INTER_LINEAR)
            return self._pil_image(resized)
        return image.resize(target_size, self._resampling())

    def _resampling(self):
        from PIL import Image

        return {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }[self.config.resize_interpolation]

    def _to_three_channel(self, image):
        if self.config.grayscale_enabled:
            return image.convert("L").convert("RGB")
        return image.convert("RGB")

    def _apply_clahe(self, image):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency is in inference extra
            raise PreprocessingDependencyError(
                "CLAHE requires OpenCV; install the 'inference' extra"
            ) from exc

        array = np.asarray(image)
        if self.config.grayscale_enabled:
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid_size,
            )
            enhanced = clahe.apply(gray)
            return self._pil_image(np.stack([enhanced] * 3, axis=-1))
        lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return self._pil_image(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

    @staticmethod
    def _pil_image(array):
        from PIL import Image

        return Image.fromarray(array, mode="RGB")
