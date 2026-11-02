"""Lossless skin streams shared by archive readers, meshes and GPU uploads."""
from __future__ import annotations

import numpy as np

MAX_INFLUENCES = 8
DEFAULT_INFLUENCES = 4


def validate_streams(indices: np.ndarray, weights: np.ndarray, count: int) -> None:
    """Reject malformed or unsupported skin streams before any mesh mutation."""
    if (weights.ndim != 2 or indices.shape != weights.shape or len(weights) != count
            or not 1 <= weights.shape[1] <= MAX_INFLUENCES):
        raise ValueError(f"skin requires matching streams with 1..{MAX_INFLUENCES} influences")
    if (not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1)
            or not np.isfinite(indices).all() or np.any(indices != np.floor(indices))
            or np.any(indices < 0)):
        raise ValueError("skin contains invalid bone indices or influence weights")


def pad_stream(values: np.ndarray, width: int) -> np.ndarray:
    """Add only inactive zero slots, preserving every authored value and dtype."""
    if values.ndim != 2 or not values.shape[1] <= width <= MAX_INFLUENCES:
        raise ValueError("unsupported skin influence padding")
    return np.pad(values, ((0, 0), (0, width - values.shape[1])))


def split_archive_skin(skin: np.ndarray, *, bone_count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Decode archived [weights, indices] rows without truncation or renormalization."""
    skin = np.asarray(skin)
    if skin.ndim != 2 or skin.shape[1] % 2:
        raise ValueError("invalid archived skin influence layout")
    width = skin.shape[1] // 2
    weights, indices = skin[:, :width], skin[:, width:]
    validate_streams(indices, weights, len(skin))
    if bone_count is not None and np.any(indices >= bone_count):
        raise ValueError("archived skin bone index is outside its palette")
    return weights, indices.astype(np.int64)
