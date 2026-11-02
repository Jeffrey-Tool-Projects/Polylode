"""Region deformers used to sculpt the lofted body from slider values."""
from __future__ import annotations

import numpy as np

from .mesh import Mesh


def _region_weight(p: np.ndarray, center, radius, falloff: float = 1.0) -> np.ndarray:
    """Smooth 0..1 falloff inside an axis-aligned ellipsoid."""
    c = np.asarray(center, dtype=np.float64)
    r = np.asarray(radius if not np.isscalar(radius) else (radius,) * 3, dtype=np.float64)
    d = (p - c) / np.maximum(r, 1e-9)
    dist = np.linalg.norm(d, axis=1)
    t = np.clip(1.0 - dist, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return t ** max(falloff, 1e-3)


def push_region(mesh: Mesh, center, radius, direction=(0, 0, 1), amount: float = 0.0,
                falloff: float = 1.0, radial: float = 0.0, mask=None,
                recompute: bool = True) -> Mesh:
    """Translate vertices inside an ellipsoidal region.

    ``radial`` blends the push direction toward "away from centre", which turns
    a flat push into a rounded bulge (breasts, cheeks, glutes).
    """
    if abs(amount) < 1e-7:
        return mesh
    p = mesh.positions.astype(np.float64)
    w = _region_weight(p, center, radius, falloff)
    if mask is not None:
        w = w * np.asarray(mask, dtype=np.float64)
    if not np.any(w > 1e-6):
        return mesh
    dirv = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(dirv)
    dirv = dirv / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    if radial > 0.0:
        c = np.asarray(center, dtype=np.float64)
        rd = p - c
        ln = np.linalg.norm(rd, axis=1, keepdims=True)
        rd = rd / np.maximum(ln, 1e-9)
        blend = dirv[None, :] * (1.0 - radial) + rd * radial
        bl = np.linalg.norm(blend, axis=1, keepdims=True)
        dirs = blend / np.maximum(bl, 1e-9)
    else:
        dirs = np.broadcast_to(dirv, p.shape)
    mesh.positions = (p + dirs * (w * amount)[:, None]).astype(np.float32)
    if recompute:
        mesh.compute_normals()
    return mesh


def band_scale(mesh: Mesh, y_lo: float, y_hi: float, sx: float = 1.0, sz: float = 1.0,
               center_x: float = 0.0, center_z: float = 0.0, feather: float = 0.05,
               recompute: bool = True) -> Mesh:
    """Scale a horizontal band of the mesh in X/Z (waist, hips, chest...)."""
    if abs(sx - 1.0) < 1e-6 and abs(sz - 1.0) < 1e-6:
        return mesh
    p = mesh.positions.astype(np.float64)
    y = p[:, 1]
    w = np.clip((y - (y_lo - feather)) / max(feather, 1e-6), 0.0, 1.0)
    w *= np.clip(((y_hi + feather) - y) / max(feather, 1e-6), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    p[:, 0] = center_x + (p[:, 0] - center_x) * (1.0 + (sx - 1.0) * w)
    p[:, 2] = center_z + (p[:, 2] - center_z) * (1.0 + (sz - 1.0) * w)
    mesh.positions = p.astype(np.float32)
    if recompute:
        mesh.compute_normals()
    return mesh


def bend_y(mesh: Mesh, y_lo: float, y_hi: float, dx: float = 0.0, dz: float = 0.0,
           curve: float = 2.0, recompute: bool = True) -> Mesh:
    """Progressive lateral offset along Y — posture / back curvature."""
    if abs(dx) < 1e-7 and abs(dz) < 1e-7:
        return mesh
    p = mesh.positions.astype(np.float64)
    t = np.clip((p[:, 1] - y_lo) / max(y_hi - y_lo, 1e-9), 0.0, 1.0) ** curve
    p[:, 0] += dx * t
    p[:, 2] += dz * t
    mesh.positions = p.astype(np.float32)
    if recompute:
        mesh.compute_normals()
    return mesh


def relax_region(mesh: Mesh, center, radius, iterations: int = 2, factor: float = 0.5,
                 falloff: float = 1.0) -> Mesh:
    """Laplacian smoothing restricted to a region (cleans up joint blends)."""
    if mesh.tri_count == 0:
        return mesh
    tri = mesh.indices.reshape(-1, 3)
    e0 = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
    e1 = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
    src = np.concatenate([e0, e1])
    dst = np.concatenate([e1, e0])
    counts = np.bincount(src, minlength=mesh.vertex_count).astype(np.float64)
    counts[counts == 0] = 1.0
    p = mesh.positions.astype(np.float64)
    w = _region_weight(p, center, radius, falloff) * factor
    for _ in range(iterations):
        acc = np.zeros_like(p)
        np.add.at(acc, src, p[dst])
        avg = acc / counts[:, None]
        p = p + (avg - p) * w[:, None]
    mesh.positions = p.astype(np.float32)
    mesh.compute_normals()
    return mesh
