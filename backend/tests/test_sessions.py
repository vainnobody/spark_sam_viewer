from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.bitset import decode_mask
from app.sessions import SessionState, SessionStore


def make_store_with_state(
    current_visible_mask: np.ndarray,
    preview_mask: np.ndarray | None = None,
) -> tuple[SessionStore, SessionState]:
    store = SessionStore()
    session_id = "test-session"
    state = SessionState(
        session_id=session_id,
        upload_path=Path("dummy.ply"),
        gaussians=SimpleNamespace(
            xyz=np.zeros((current_visible_mask.size, 3), dtype=np.float32),
            count=int(current_visible_mask.size),
        ),
        current_visible_mask=current_visible_mask.copy(),
        preview_mask=None if preview_mask is None else preview_mask.copy(),
    )
    store._sessions[session_id] = state
    return store, state


def test_commit_isolate_replaces_visible_mask_with_preview():
    store, _state = make_store_with_state(
        np.asarray([True, False, True, True]),
        np.asarray([False, True, False, True]),
    )

    response = store.commit("test-session", "isolate")

    decoded = decode_mask(response["visibleMaskBitset"], 4)
    assert response["visibleCount"] == 2
    assert decoded.tolist() == [False, True, False, True]


def test_commit_invert_removes_preview_from_current_visible_mask():
    store, _state = make_store_with_state(
        np.asarray([True, True, False, True]),
        np.asarray([False, True, True, False]),
    )

    response = store.commit("test-session", "invert")

    decoded = decode_mask(response["visibleMaskBitset"], 4)
    assert response["visibleCount"] == 2
    assert decoded.tolist() == [True, False, False, True]


def test_commit_reset_restores_full_visibility():
    store, state = make_store_with_state(
        np.asarray([True, False, False, True]),
        np.asarray([False, True, False, False]),
    )

    response = store.commit("test-session", "reset")

    decoded = decode_mask(response["visibleMaskBitset"], 4)
    assert response["visibleCount"] == 4
    assert decoded.tolist() == [True, True, True, True]
    assert state.preview_mask is None


def test_commit_requires_preview_for_isolate_and_invert():
    store, _state = make_store_with_state(np.asarray([True, True, True]))

    with pytest.raises(ValueError, match="Run preview first"):
        store.commit("test-session", "isolate")

    with pytest.raises(ValueError, match="Run preview first"):
        store.commit("test-session", "invert")


def test_map_mask_to_gaussians_only_targets_current_visible_indices():
    store, state = make_store_with_state(np.asarray([True, False, True]))

    preview_mask = store._map_mask_to_gaussians(
        state,
        visible_pixels=np.asarray([[0, 0], [1, 0]], dtype=np.int32),
        visible_on_screen=np.asarray([True, True]),
        mask_2d=np.asarray(
            [
                [0, 1],
                [0, 0],
            ],
            dtype=np.uint8,
        ),
    )

    assert preview_mask.tolist() == [False, False, True]
