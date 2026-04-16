from __future__ import annotations

import base64
import io
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import cv2
import numpy as np
from PIL import Image

from .bitset import encode_mask
from .camera import build_depth_buffer, compute_visible_mask, project_world_to_pixels
from .config import SETTINGS
from .gaussian_cloud import GaussianCloud
from .prompt_pixels import resolve_sam_prompt_pixels
from .sam_predictor import SamPredictor
from .schemas import PreviewRequest


@dataclass
class SessionState:
    session_id: str
    upload_path: Path
    gaussians: GaussianCloud
    current_visible_mask: np.ndarray
    preview_mask: np.ndarray | None = None


def _decode_image_data_url(data_url: str) -> np.ndarray:
    if "," not in data_url:
        raise ValueError("Preview imageDataUrl is malformed.")
    _, payload = data_url.split(",", 1)
    image_bytes = base64.b64decode(payload.encode("ascii"))
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)

class SessionStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[str, SessionState] = {}
        self._sam = SamPredictor()

    def create_session(self, file_name: str, file_bytes: bytes) -> SessionState:
        session_id = uuid.uuid4().hex
        session_dir = SETTINGS.session_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        upload_path = session_dir / file_name
        upload_path.write_bytes(file_bytes)

        gaussians = GaussianCloud.load_ply(upload_path)
        visible_mask = np.ones(gaussians.count, dtype=np.bool_)

        state = SessionState(
            session_id=session_id,
            upload_path=upload_path,
            gaussians=gaussians,
            current_visible_mask=visible_mask.copy(),
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
        return state.gaussians.bbox()

    def preview(self, session_id: str, payload: PreviewRequest) -> dict[str, object]:
        state = self.get(session_id)
        if not payload.points:
            raise ValueError("At least one prompt point is required.")

        with self._lock:
            rgb = _decode_image_data_url(payload.imageDataUrl)
            if rgb.shape[1] != payload.camera.width or rgb.shape[0] != payload.camera.height:
                raise ValueError(
                    "Preview image dimensions do not match the current camera payload. "
                    "Refresh the view and retry."
                )

            visible_points = state.gaussians.xyz[state.current_visible_mask]
            pixels, depth, inside, depth_buffer = build_depth_buffer(payload.camera, visible_points)
            visible_on_screen = compute_visible_mask(pixels, depth, inside, depth_buffer)

            prompt_world = np.asarray([point.world for point in payload.points], dtype=np.float32)
            prompt_labels = np.asarray([point.label for point in payload.points], dtype=np.int32)
            prompt_clicked_pixels = None
            if any(point.screen is not None for point in payload.points):
                prompt_clicked_pixels = np.asarray(
                    [
                        point.screen if point.screen is not None else [np.nan, np.nan]
                        for point in payload.points
                    ],
                    dtype=np.float32,
                )

            prompt_projected_pixels, prompt_depth, prompt_inside = project_world_to_pixels(
                payload.camera, prompt_world
            )
            prompt_visible = compute_visible_mask(
                prompt_projected_pixels, prompt_depth, prompt_inside, depth_buffer, tolerance=5e-3
            )
            prompt_valid = prompt_inside & prompt_visible

            prompt_pixels, prompt_indices = resolve_sam_prompt_pixels(
                prompt_projected_pixels,
                prompt_valid,
                prompt_clicked_pixels,
                payload.camera.width,
                payload.camera.height,
            )
            prompt_labels = prompt_labels[prompt_indices]
            if prompt_pixels.shape[0] == 0:
                raise ValueError("No prompt points are visible in the current view.")

            mask_2d = self._sam.predict_mask(rgb, prompt_pixels, prompt_labels)
            preview_mask = self._map_mask_to_gaussians(
                state,
                pixels,
                visible_on_screen,
                mask_2d,
            )
            state.preview_mask = preview_mask

            preview_image = self._render_preview_image(rgb, mask_2d, prompt_pixels, prompt_labels)
            return {
                "previewImage": preview_image,
                "previewCount": int(preview_mask.sum()),
                "previewMaskBitset": encode_mask(preview_mask),
                "visibleCount": int(state.current_visible_mask.sum()),
                "visibleMaskBitset": encode_mask(state.current_visible_mask),
            }

    def commit(self, session_id: str, op: str) -> dict[str, object]:
        state = self.get(session_id)
        with self._lock:
            if op == "reset":
                state.current_visible_mask = np.ones_like(state.current_visible_mask)
                state.preview_mask = None
            else:
                if state.preview_mask is None:
                    raise ValueError("No preview mask available. Run preview first.")
                preview = state.preview_mask.copy()
                if op == "isolate":
                    state.current_visible_mask = preview
                elif op == "invert":
                    state.current_visible_mask = state.current_visible_mask & (~preview)
                else:
                    raise ValueError(f"Unsupported commit op: {op}")
                state.preview_mask = None

            return {
                "visibleCount": int(state.current_visible_mask.sum()),
                "visibleMaskBitset": encode_mask(state.current_visible_mask),
            }

    def export_mask_bytes(self, session_id: str) -> bytes:
        state = self.get(session_id)
        with io.BytesIO() as buffer:
            np.save(buffer, state.current_visible_mask.astype(np.uint8))
            return buffer.getvalue()

    def export_ply_bytes(self, session_id: str) -> bytes:
        state = self.get(session_id)
        with self._lock:
            mask = state.current_visible_mask
            if not np.any(mask):
                raise ValueError("Current mask is empty.")
            return state.gaussians.subset(mask).to_ply_bytes()

    def _map_mask_to_gaussians(
        self,
        state: SessionState,
        visible_pixels: np.ndarray,
        visible_on_screen: np.ndarray,
        mask_2d: np.ndarray,
    ) -> np.ndarray:
        visible_indices = np.nonzero(state.current_visible_mask)[0]
        preview_mask = np.zeros_like(state.current_visible_mask)
        candidate = visible_on_screen.copy()
        if np.any(candidate):
            u = visible_pixels[candidate, 0]
            v = visible_pixels[candidate, 1]
            hits = mask_2d[v, u] > 0
            preview_mask[visible_indices[np.nonzero(candidate)[0][hits]]] = True
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
            cv2.circle(blended, tuple(int(value) for value in pixel), 6, color, -1)
            cv2.circle(blended, tuple(int(value) for value in pixel), 10, (255, 255, 255), 1)

        success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 82],
        )
        if not success:
            raise RuntimeError("Failed to encode preview image.")
        return f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


STORE = SessionStore()
