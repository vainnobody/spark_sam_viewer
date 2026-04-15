from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import cv2
import numpy as np
import torch

from . import bootstrap  # noqa: F401
from .bitset import encode_mask
from .camera import PIPELINE, build_minicam, project_points_to_screen
from .config import SETTINGS
from .sam_predictor import SamPredictor
from .schemas import PreviewRequest
from dataset.fusion_utils import PointCloudToImageMapper
from model import GaussianModel, render


@dataclass
class SessionState:
    session_id: str
    upload_path: Path
    gaussians: GaussianModel
    current_visible_mask: torch.Tensor
    accumulated_mask: torch.Tensor
    preview_mask: torch.Tensor | None = None


def _infer_sh_degree(file_bytes: bytes) -> int:
    header = file_bytes.split(b"end_header", 1)[0].decode("utf-8", errors="ignore")
    rest_props = sum(1 for line in header.splitlines() if "f_rest_" in line)
    if rest_props <= 0:
        return 0
    if rest_props % 3 != 0:
        raise ValueError(
            f"Unsupported PLY SH layout: found {rest_props} f_rest properties, which is not divisible by 3."
        )

    coeff_count = (rest_props + 3) // 3
    root = int(round(coeff_count ** 0.5))
    if root * root != coeff_count:
        raise ValueError(
            f"Unsupported PLY SH layout: found {rest_props} f_rest properties, which is not a valid 3DGS degree."
        )
    return root - 1


def _set_model_requires_grad(model: GaussianModel, enabled: bool) -> None:
    for attr_name in (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_opacity",
        "_scaling",
        "_rotation",
        "_features_semantic",
        "_times",
    ):
        value = getattr(model, attr_name, None)
        if isinstance(value, torch.Tensor):
            value.requires_grad_(enabled)


class SessionStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[str, SessionState] = {}
        self._sam = SamPredictor()
        self._render_device = "cuda" if torch.cuda.is_available() else SETTINGS.device
        bg = [0.0, 0.0, 0.0]
        self._background = torch.tensor(bg, dtype=torch.float32, device=self._render_device)

    def _ensure_runtime_ready(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "semantic-gaussians rendering requires CUDA, but no CUDA device is available in this environment."
            )
        if not self._render_device.startswith("cuda"):
            raise RuntimeError(
                f"semantic-gaussians rendering requires a CUDA device, got SPARK_SAM_VIEWER_DEVICE={SETTINGS.device!r}."
            )

    def create_session(self, file_name: str, file_bytes: bytes) -> SessionState:
        self._ensure_runtime_ready()
        session_id = uuid.uuid4().hex
        session_dir = SETTINGS.session_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        upload_path = session_dir / file_name
        upload_path.write_bytes(file_bytes)

        gaussians = GaussianModel(_infer_sh_degree(file_bytes))
        gaussians.load_ply(str(upload_path))
        _set_model_requires_grad(gaussians, False)

        visible_mask = torch.ones(
            gaussians.get_xyz.shape[0],
            dtype=torch.bool,
            device=gaussians.get_xyz.device,
        )
        state = SessionState(
            session_id=session_id,
            upload_path=upload_path,
            gaussians=gaussians,
            current_visible_mask=visible_mask.clone(),
            accumulated_mask=torch.zeros_like(visible_mask),
        )
        with self._lock:
            self._sessions[session_id] = state
        return state

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(session_id)
        return state

    def build_bbox(self, state: SessionState) -> dict[str, list[float]]:
        xyz = state.gaussians.get_xyz.detach().cpu().numpy()
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        return {
            "min": xyz_min.tolist(),
            "max": xyz_max.tolist(),
            "center": ((xyz_min + xyz_max) * 0.5).tolist(),
        }

    def preview(self, session_id: str, payload: PreviewRequest) -> dict[str, object]:
        state = self.get(session_id)
        if not payload.points:
            raise ValueError("At least one prompt point is required.")

        with self._lock:
            camera = build_minicam(payload.camera, self._render_device)
            with torch.no_grad():
                render_output = render(
                    viewpoint_camera=camera,
                    pc=state.gaussians,
                    pipe=PIPELINE,
                    bg_color=self._background,
                    foreground=state.current_visible_mask,
                    override_shape=(payload.camera.width, payload.camera.height),
                )
                rgb = (
                    render_output["render"]
                    .detach()
                    .clamp(0, 1)
                    .mul(255)
                    .byte()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                )
                depth = render_output["depth"].detach().cpu().numpy().squeeze(0)

            prompt_world = np.asarray([point.world for point in payload.points], dtype=np.float32)
            prompt_labels = np.asarray([point.label for point in payload.points], dtype=np.int32)
            prompt_pixels, inside_mask = project_points_to_screen(payload.camera, prompt_world, depth)
            prompt_pixels = prompt_pixels[inside_mask]
            prompt_labels = prompt_labels[inside_mask]
            if prompt_pixels.shape[0] == 0:
                raise ValueError("No prompt points are visible in the current view.")

            mask_2d = self._sam.predict_mask(rgb, prompt_pixels, prompt_labels)
            preview_mask = self._map_mask_to_gaussians(
                state.gaussians,
                state.current_visible_mask,
                camera,
                depth,
                mask_2d,
                payload.camera.width,
                payload.camera.height,
            )
            state.preview_mask = preview_mask

            preview_image = self._render_preview_image(rgb, mask_2d, prompt_pixels, prompt_labels)
            return {
                "previewImage": preview_image,
                "previewCount": int(preview_mask.sum().item()),
                "previewMaskBitset": encode_mask(preview_mask),
                "visibleCount": int(state.current_visible_mask.sum().item()),
                "visibleMaskBitset": encode_mask(state.current_visible_mask),
            }

    def commit(self, session_id: str, op: str) -> dict[str, object]:
        state = self.get(session_id)
        with self._lock:
            if op == "reset":
                state.current_visible_mask = torch.ones_like(state.current_visible_mask)
                state.accumulated_mask = torch.zeros_like(state.accumulated_mask)
                state.preview_mask = None
            else:
                if state.preview_mask is None:
                    raise ValueError("No preview mask available. Run preview first.")
                preview = state.preview_mask & state.current_visible_mask
                if op == "union":
                    state.accumulated_mask = state.accumulated_mask | preview
                    state.current_visible_mask = state.accumulated_mask.clone()
                elif op == "invert":
                    state.current_visible_mask = state.current_visible_mask & (~preview)
                    state.accumulated_mask = state.accumulated_mask & state.current_visible_mask
                else:
                    raise ValueError(f"Unsupported commit op: {op}")
                state.preview_mask = None

            return {
                "visibleCount": int(state.current_visible_mask.sum().item()),
                "visibleMaskBitset": encode_mask(state.current_visible_mask),
            }

    def export_mask_bytes(self, session_id: str) -> bytes:
        state = self.get(session_id)
        with io.BytesIO() as buffer:
            np.save(buffer, state.current_visible_mask.detach().cpu().numpy().astype(np.uint8))
            return buffer.getvalue()

    def export_ply_bytes(self, session_id: str) -> bytes:
        state = self.get(session_id)
        with self._lock:
            mask = state.current_visible_mask
            if not torch.any(mask):
                raise ValueError("Current mask is empty.")
            export_model = GaussianModel(state.gaussians.max_sh_degree)
            export_model.active_sh_degree = state.gaussians.active_sh_degree
            export_model._xyz = state.gaussians._xyz[mask].detach().clone()
            export_model._features_dc = state.gaussians._features_dc[mask].detach().clone()
            export_model._features_rest = state.gaussians._features_rest[mask].detach().clone()
            export_model._opacity = state.gaussians._opacity[mask].detach().clone()
            export_model._scaling = state.gaussians._scaling[mask].detach().clone()
            export_model._rotation = state.gaussians._rotation[mask].detach().clone()

            tmp_path = state.upload_path.parent / "export.ply"
            export_model.save_ply(str(tmp_path))
            try:
                return tmp_path.read_bytes()
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

    def _map_mask_to_gaussians(
        self,
        gaussians: GaussianModel,
        visible_mask: torch.Tensor,
        camera,
        depth: np.ndarray,
        mask_2d: np.ndarray,
        width: int,
        height: int,
    ) -> torch.Tensor:
        fx = (width / 2.0) / np.tan(camera.FoVx / 2.0)
        fy = (height / 2.0) / np.tan(camera.FoVy / 2.0)
        cx = width / 2.0
        cy = height / 2.0
        mapper = PointCloudToImageMapper(
            image_dim=[width, height],
            visibility_threshold=9999.0,
            cut_bound=4,
            intrinsics=np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]],
                dtype=np.float32,
            ),
        )

        visible_indices = torch.where(visible_mask)[0]
        points = gaussians.get_xyz[visible_mask].detach().cpu().numpy()
        mapping, _ = mapper.compute_mapping(
            camera.world_view_transform.detach().cpu().numpy(),
            points,
            depth,
        )
        mapping_t = torch.from_numpy(mapping).long().to(gaussians.get_xyz.device)
        projected = mapping_t[:, 2] > 0

        preview_mask = torch.zeros_like(visible_mask)
        if projected.any():
            v = torch.clamp(mapping_t[projected, 0], 0, height - 1)
            u = torch.clamp(mapping_t[projected, 1], 0, width - 1)
            mask_t = torch.as_tensor(mask_2d, dtype=torch.bool, device=gaussians.get_xyz.device)
            hits = mask_t[v, u]
            preview_mask[visible_indices[projected]] = hits
        return preview_mask

    def _render_preview_image(
        self,
        rgb: np.ndarray,
        mask_2d: np.ndarray,
        prompt_pixels: np.ndarray,
        prompt_labels: np.ndarray,
    ) -> str:
        image = rgb.copy()
        overlay = image.copy()
        overlay[mask_2d > 0] = np.array([255, 106, 61], dtype=np.uint8)
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        for pixel, label in zip(prompt_pixels, prompt_labels):
            color = (255, 106, 61) if label > 0 else (45, 198, 255)
            cv2.circle(blended, tuple(int(v) for v in pixel), 6, color, -1)
            cv2.circle(blended, tuple(int(v) for v in pixel), 10, (255, 255, 255), 1)

        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not success:
            raise RuntimeError("Failed to encode preview image.")
        import base64

        return f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


STORE = SessionStore()
