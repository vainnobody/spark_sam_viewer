from __future__ import annotations

import numpy as np


def resolve_sam_prompt_pixels(
    projected_pixels: np.ndarray,
    projected_valid: np.ndarray,
    clicked_pixels: np.ndarray | None,
    width: int,
    height: int,
    reprojection_tolerance: float = 64.0,
) -> tuple[np.ndarray, np.ndarray]:
    selected_pixels: list[np.ndarray] = []
    selected_indices: list[int] = []

    for index, projected_pixel in enumerate(projected_pixels):
        if not projected_valid[index]:
            continue

        chosen_pixel = projected_pixel.astype(np.int32, copy=False)
        if clicked_pixels is not None:
            clicked_pixel = clicked_pixels[index]
            clicked_inside = (
                np.isfinite(clicked_pixel).all()
                and 0 <= clicked_pixel[0] < width
                and 0 <= clicked_pixel[1] < height
            )
            if clicked_inside:
                reprojection_error = np.linalg.norm(
                    clicked_pixel - projected_pixel.astype(np.float32)
                )
                if reprojection_error <= reprojection_tolerance:
                    chosen_pixel = np.round(clicked_pixel).astype(np.int32)

        selected_pixels.append(chosen_pixel)
        selected_indices.append(index)

    if not selected_pixels:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    return np.stack(selected_pixels, axis=0), np.asarray(selected_indices, dtype=np.int32)
