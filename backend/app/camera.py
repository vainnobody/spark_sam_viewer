from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

from . import bootstrap  # noqa: F401
from .schemas import CameraPayload
from scene.camera import MiniCam
from utils.graphics_utils import getProjectionMatrix


PIPELINE = SimpleNamespace(
    compute_cov3d_python=False,
    convert_shs_python=False,
    debug=False,
)

# Three/Spark uses an OpenGL-style camera where forward is -Z and up is +Y.
# semantic-gaussians expects a COLMAP/OpenCV-style camera where forward is +Z
# and image-space Y grows downward.
OPENGL_TO_COLMAP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _as_tensor(matrix: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(matrix, dtype=torch.float32, device=device).transpose(0, 1)


def get_colmap_world_to_camera(payload: CameraPayload) -> np.ndarray:
    view_gl = np.asarray(payload.viewMatrix, dtype=np.float32)
    return OPENGL_TO_COLMAP @ view_gl


def build_minicam(payload: CameraPayload, device: str) -> MiniCam:
    fovx = 2.0 * math.atan(math.tan(payload.fovY * 0.5) * payload.aspect)
    world_view_transform = _as_tensor(get_colmap_world_to_camera(payload), device)
    projection_matrix = (
        getProjectionMatrix(
            znear=payload.near,
            zfar=payload.far,
            fovX=fovx,
            fovY=payload.fovY,
        )
        .transpose(0, 1)
        .to(device)
    )
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    return MiniCam(
        payload.width,
        payload.height,
        payload.fovY,
        fovx,
        payload.near,
        payload.far,
        world_view_transform,
        full_proj_transform,
    )


def project_points_to_screen(
    payload: CameraPayload,
    points_world: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if points_world.size == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

    view = get_colmap_world_to_camera(payload)
    homogeneous = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1)
    cam_points = (view @ homogeneous.T).T

    z = cam_points[:, 2]
    fovx = 2.0 * math.atan(math.tan(payload.fovY * 0.5) * payload.aspect)
    fx = payload.width / (2.0 * math.tan(fovx * 0.5))
    fy = payload.height / (2.0 * math.tan(payload.fovY * 0.5))
    cx = payload.width * 0.5
    cy = payload.height * 0.5

    u = np.round((cam_points[:, 0] * fx / np.clip(z, 1e-6, None)) + cx).astype(np.int32)
    v = np.round((cam_points[:, 1] * fy / np.clip(z, 1e-6, None)) + cy).astype(np.int32)

    inside = (
        (z > 0.01)
        & (u >= 0)
        & (v >= 0)
        & (u < payload.width)
        & (v < payload.height)
    )

    if np.any(inside):
        sampled_depth = depth[v[inside], u[inside]]
        depth_ok = z[inside] < np.maximum(sampled_depth, 1e-6) * 1.1
        inside_indices = np.nonzero(inside)[0]
        inside[inside_indices] = depth_ok

    return np.stack([u, v], axis=1), inside
