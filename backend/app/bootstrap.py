from __future__ import annotations

import sys
from pathlib import Path

from .config import SETTINGS


def _prepend_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


_prepend_path(SETTINGS.semantic_gaussians_root)
_prepend_path(SETTINGS.sam3_root)
