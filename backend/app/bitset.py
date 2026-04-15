from __future__ import annotations

import base64

import numpy as np
import torch


def encode_mask(mask: torch.Tensor) -> str:
    mask_np = mask.detach().to(dtype=torch.bool).cpu().numpy().astype(np.uint8)
    packed = np.packbits(mask_np, bitorder="little")
    return base64.b64encode(packed.tobytes()).decode("ascii")


def decode_mask(encoded: str, count: int, device: str = "cpu") -> torch.Tensor:
    raw = base64.b64decode(encoded.encode("ascii"))
    packed = np.frombuffer(raw, dtype=np.uint8)
    unpacked = np.unpackbits(packed, bitorder="little")[:count]
    return torch.from_numpy(unpacked.astype(np.bool_)).to(device=device)
