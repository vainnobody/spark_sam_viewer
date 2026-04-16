from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .config import SETTINGS


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _local_sam3_project_root() -> Path:
    return _workspace_root() / "sam3"


def _import_sam3_builder() -> Callable[..., object]:
    module = importlib.import_module("sam3")
    builder = getattr(module, "build_sam3_image_model", None)
    if builder is None:
        module_file = getattr(module, "__file__", None)
        module_paths = ", ".join(str(path) for path in getattr(module, "__path__", []))
        raise ImportError(
            "Imported 'sam3' but it did not expose build_sam3_image_model. "
            f"module_file={module_file!r}, module_paths={module_paths or '<none>'}"
        )
    return builder


def _resolve_sam3_builder() -> Callable[..., object]:
    try:
        return _import_sam3_builder()
    except ModuleNotFoundError as exc:
        if exc.name != "sam3":
            raise RuntimeError(
                "SAM3 import failed because a required dependency is missing: "
                f"{exc.name!r}. Install the backend requirements in the server environment."
            ) from exc
        initial_error: Exception = exc
    except ImportError as exc:
        initial_error = exc

    local_root = _local_sam3_project_root()
    local_init = local_root / "sam3" / "__init__.py"
    if not local_init.exists():
        raise RuntimeError(
            "The Python package 'sam3' is not importable in the backend environment, and no local "
            f"SAM3 checkout was found at {local_root}."
        ) from initial_error

    local_root_str = str(local_root)
    if local_root_str not in sys.path:
        sys.path.insert(0, local_root_str)
    sys.modules.pop("sam3", None)
    importlib.invalidate_caches()

    try:
        return _import_sam3_builder()
    except ModuleNotFoundError as exc:
        if exc.name == "sam3":
            raise RuntimeError(
                "Found a local SAM3 checkout at "
                f"{local_root}, but it still was not importable after adding it to sys.path."
            ) from exc
        raise RuntimeError(
            "Found a local SAM3 checkout at "
            f"{local_root}, but importing it failed because dependency {exc.name!r} is missing. "
            "Install the backend requirements in the server environment."
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "Found a local SAM3 checkout at "
            f"{local_root}, but importing it failed: {exc}"
        ) from exc


class SamPredictor:
    def __init__(self) -> None:
        self._predictor = None

    @property
    def ready(self) -> bool:
        return self._predictor is not None

    def load(self) -> None:
        if self._predictor is not None:
            return
        checkpoint = SETTINGS.sam3_checkpoint
        if checkpoint is None:
            raise RuntimeError(
                "SAM3_CHECKPOINT is not set. Point it to a local SAM3 checkpoint before using preview."
            )
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise RuntimeError(f"SAM3 checkpoint does not exist: {checkpoint_path}")

        build_sam3_image_model = _resolve_sam3_builder()
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False,
            device=SETTINGS.device,
            enable_inst_interactivity=True,
        )
        if model.inst_interactive_predictor is None:
            raise RuntimeError("SAM3 image model did not expose an interactive predictor.")
        self._predictor = model.inst_interactive_predictor

    def predict_mask(
        self,
        image_rgb: np.ndarray,
        prompt_pixels: np.ndarray,
        prompt_labels: np.ndarray,
    ) -> np.ndarray:
        self.load()
        pil_image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        self._predictor.set_image(pil_image)
        multimask = prompt_pixels.shape[0] <= 1
        masks, scores, _ = self._predictor.predict(
            point_coords=prompt_pixels.astype(np.float32),
            point_labels=(prompt_labels > 0).astype(np.int32),
            multimask_output=multimask,
            normalize_coords=False,
        )
        best_index = int(np.argmax(scores))
        return masks[best_index].astype(np.uint8)
