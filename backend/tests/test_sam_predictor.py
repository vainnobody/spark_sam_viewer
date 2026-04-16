from pathlib import Path
from types import SimpleNamespace
import sys

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
