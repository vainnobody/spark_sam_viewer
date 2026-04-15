from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

@dataclass(frozen=True)
class Settings:
    session_root: Path
    device: str
    sam3_checkpoint: str | None
    cors_origin: str


def load_settings() -> Settings:
    session_root = Path(
        os.environ.get(
            "SPARK_SAM_VIEWER_SESSION_ROOT",
            Path(__file__).resolve().parents[2] / ".sessions",
        )
    )
    session_root.mkdir(parents=True, exist_ok=True)

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    return Settings(
        session_root=session_root,
        device=os.environ.get("SPARK_SAM_VIEWER_DEVICE", default_device),
        sam3_checkpoint=os.environ.get("SAM3_CHECKPOINT"),
        cors_origin=os.environ.get("SPARK_SAM_VIEWER_CORS_ORIGIN", "*"),
    )


SETTINGS = load_settings()
