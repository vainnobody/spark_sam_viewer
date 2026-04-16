from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.prompt_pixels import resolve_sam_prompt_pixels


def test_resolve_sam_prompt_pixels_prefers_clicked_pixels_when_close():
    projected = np.asarray([[120, 80]], dtype=np.int32)
    valid = np.asarray([True])
    clicked = np.asarray([[126.2, 84.8]], dtype=np.float32)

    pixels, indices = resolve_sam_prompt_pixels(projected, valid, clicked, 400, 300)

    assert indices.tolist() == [0]
    assert pixels.tolist() == [[126, 85]]


def test_resolve_sam_prompt_pixels_falls_back_to_projection_when_far():
    projected = np.asarray([[120, 80]], dtype=np.int32)
    valid = np.asarray([True])
    clicked = np.asarray([[260.0, 190.0]], dtype=np.float32)

    pixels, indices = resolve_sam_prompt_pixels(projected, valid, clicked, 400, 300)

    assert indices.tolist() == [0]
    assert pixels.tolist() == [[120, 80]]


def test_resolve_sam_prompt_pixels_skips_invalid_points():
    projected = np.asarray([[120, 80], [40, 22]], dtype=np.int32)
    valid = np.asarray([False, True])

    pixels, indices = resolve_sam_prompt_pixels(projected, valid, None, 400, 300)

    assert indices.tolist() == [1]
    assert pixels.tolist() == [[40, 22]]
