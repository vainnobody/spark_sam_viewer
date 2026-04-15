from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import bootstrap  # noqa: F401
from .config import SETTINGS
from sam3 import build_sam3_image_model


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
