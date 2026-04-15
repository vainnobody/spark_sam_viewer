from __future__ import annotations

import base64

import numpy as np


def encode_mask(mask: np.ndarray) -> str:
    mask_np = np.asarray(mask, dtype=np.bool_).astype(np.uint8)
    packed = np.packbits(mask_np, bitorder="little")
    return base64.b64encode(packed.tobytes()).decode("ascii")


def decode_mask(encoded: str, count: int) -> np.ndarray:
    raw = base64.b64decode(encoded.encode("ascii"))
    packed = np.frombuffer(raw, dtype=np.uint8)
    unpacked = np.unpackbits(packed, bitorder="little")[:count]
    return unpacked.astype(np.bool_)
