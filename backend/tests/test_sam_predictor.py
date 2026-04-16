from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.sam_predictor import SamPredictor


def test_prepare_interactive_predictor_reuses_main_backbone():
    predictor_model = SimpleNamespace(backbone=None)
    predictor = SimpleNamespace(model=predictor_model)
    main_backbone = object()
    model = SimpleNamespace(
        inst_interactive_predictor=predictor,
        backbone=main_backbone,
    )

    prepared = SamPredictor()._prepare_interactive_predictor(model)

    assert prepared is predictor
    assert predictor_model.backbone is main_backbone


def test_prepare_interactive_predictor_fails_without_any_backbone():
    predictor = SimpleNamespace(model=SimpleNamespace(backbone=None))
    model = SimpleNamespace(
        inst_interactive_predictor=predictor,
        backbone=None,
    )

    with pytest.raises(RuntimeError, match="without a backbone"):
        SamPredictor()._prepare_interactive_predictor(model)


def test_predict_mask_uses_image_relative_point_normalization():
    predictor = SimpleNamespace()
    predictor.set_image = lambda image: None
    predictor.predict_calls = []

    def fake_predict(**kwargs):
        predictor.predict_calls.append(kwargs)
        return (
            np.asarray([[[True, False], [False, False]]], dtype=bool),
            np.asarray([0.9], dtype=np.float32),
            np.asarray([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
        )

    predictor.predict = fake_predict

    sam_predictor = SamPredictor()
    sam_predictor._predictor = predictor

    mask = sam_predictor.predict_mask(
        np.zeros((32, 48, 3), dtype=np.uint8),
        np.asarray([[12, 8]], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
    )

    assert mask.shape == (2, 2)
    assert len(predictor.predict_calls) == 1
    assert predictor.predict_calls[0]["normalize_coords"] is True
