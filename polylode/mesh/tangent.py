"""Preserve authored tangent directions and their separately stored handedness."""
from __future__ import annotations

import numpy as np


def validate(tangents: np.ndarray, vertices: int) -> None:
    """Reject partial or nonfinite authored streams; an absent stream stays absent."""
    if tangents.shape not in ((0, 4), (vertices, 4)) or not np.isfinite(tangents).all():
        raise ValueError('authored tangents require a finite complete Nx4 stream or no stream')


def directions(vectors: np.ndarray, linear: np.ndarray) -> np.ndarray:
    """Transform every direction without orthogonalizing or replacing zero vectors."""
    moved = (np.einsum('vij,vj->vi', linear, vectors) if linear.ndim == 3 else vectors @ linear.T)
    length = np.linalg.norm(moved, axis=1, keepdims=True)
    return np.divide(moved, length, out=np.zeros_like(moved), where=length != 0)


def affine(tangents: np.ndarray, linear: np.ndarray) -> np.ndarray:
    """Apply an invertible object transform, including its reflection handedness."""
    determinant = float(np.linalg.det(linear))
    if determinant == 0 or not np.isfinite(determinant):
        raise ValueError('singular object transform cannot preserve an authored tangent frame')
    moved = directions(tangents[:, :3].astype(np.float64), linear)
    return np.c_[moved, tangents[:, 3] * np.sign(determinant)].astype(np.float32)
