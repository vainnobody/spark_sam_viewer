from __future__ import annotations

import numpy as np

from .schemas import CameraPayload


def _as_matrix(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def project_world_to_pixels(
    payload: CameraPayload,
    points_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_world.size == 0:
        empty_pixels = np.zeros((0, 2), dtype=np.int32)
        empty_depth = np.zeros((0,), dtype=np.float32)
        empty_inside = np.zeros((0,), dtype=bool)
        return empty_pixels, empty_depth, empty_inside

    view = _as_matrix(payload.viewMatrix)
    projection = _as_matrix(payload.projectionMatrix)
    homogeneous = np.concatenate(
        [points_world.astype(np.float32), np.ones((points_world.shape[0], 1), dtype=np.float32)],
        axis=1,
    )

    view_space = (view @ homogeneous.T).T
    clip = (projection @ view_space.T).T
    w = clip[:, 3]
    valid_w = np.abs(w) > 1e-6

    ndc = np.zeros((points_world.shape[0], 3), dtype=np.float32)
    ndc[valid_w] = clip[valid_w, :3] / w[valid_w, None]

    x = ndc[:, 0]
    y = ndc[:, 1]
    z = ndc[:, 2]

    u = np.round((x * 0.5 + 0.5) * (payload.width - 1)).astype(np.int32)
    v = np.round((1.0 - (y * 0.5 + 0.5)) * (payload.height - 1)).astype(np.int32)

    inside = (
        valid_w
        & (x >= -1.0)
        & (x <= 1.0)
        & (y >= -1.0)
        & (y <= 1.0)
        & (z >= -1.0)
        & (z <= 1.0)
        & (u >= 0)
        & (u < payload.width)
        & (v >= 0)
        & (v < payload.height)
    )
    pixels = np.stack([u, v], axis=1)
    return pixels, z.astype(np.float32), inside


def build_depth_buffer(
    payload: CameraPayload,
    points_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pixels, depth, inside = project_world_to_pixels(payload, points_world)
    depth_buffer = np.full((payload.height, payload.width), np.inf, dtype=np.float32)

    if np.any(inside):
        inside_pixels = pixels[inside]
        np.minimum.at(depth_buffer, (inside_pixels[:, 1], inside_pixels[:, 0]), depth[inside])

    return pixels, depth, inside, depth_buffer


def compute_visible_mask(
    pixels: np.ndarray,
    depth: np.ndarray,
    inside: np.ndarray,
    depth_buffer: np.ndarray,
    tolerance: float = 1e-4,
) -> np.ndarray:
    visible = inside.copy()
    if np.any(inside):
        inside_idx = np.nonzero(inside)[0]
        pixel_depth = depth_buffer[pixels[inside, 1], pixels[inside, 0]]
        visible[inside_idx] = depth[inside] <= (pixel_depth + tolerance)
    return visible
