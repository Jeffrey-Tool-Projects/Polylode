"""Pose-dependent convex coverings made only from their declared bound source points."""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from ..mesh import skinning
from ..mesh.mesh import Mesh


def hull_triangles(points: np.ndarray) -> np.ndarray:
    """Outward triangles of the source-point convex hull, without adding points."""
    hull = ConvexHull(np.asarray(points, dtype=np.float64))
    triangles = hull.simplices.copy()
    corners = points[triangles].astype(np.float64)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    reverse = np.einsum('ij,ij->i', normals, hull.equations[:, :3]) < 0
    triangles[reverse] = triangles[reverse, ::-1]
    return triangles.astype(np.uint32)


class ConvexSurface:
    """Rebuild a covering's topology and normals from the same points in each pose."""

    def __init__(self, part, rest: Mesh, *, smooth: bool) -> None:
        self.part, self.rest, self.smooth = part, rest.copy(), smooth
        self.used = np.unique(rest.bone_idx[rest.bone_wt > 0])
        self.last = None
        self.result = None

    def update(self, matrices: np.ndarray, rests: np.ndarray) -> None:
        """Refresh only when its source bones change; replace mesh so GPU indices reload."""
        changed = matrices[self.used]
        if (self.last is not None and np.array_equal(changed, self.last)
                and self.part.mesh is self.result):
            return
        self.last = changed.copy()
        share = self.rest.skin_dual if len(self.rest.skin_dual) else None
        points, deform = skinning.frames(self.rest.positions, self.rest.bone_idx, self.rest.bone_wt,
                                        matrices, self.part.skinning, rests, share)
        result = self.rest.copy()
        result.indices = hull_triangles(points).ravel()
        posed = Mesh.from_arrays(points, result.indices).compute_normals(smooth=self.smooth)
        if not self.smooth:
            selected = result.indices.copy()
            result.split_faceted()
            deform = deform[selected]
        # The renderer applies the skin cofactor. Its inverse leaves the
        # freshly constructed posed normal exactly where the hull faces point.
        a, b, c = deform[:, :, 0], deform[:, :, 1], deform[:, :, 2]
        cofactor = np.stack([np.cross(b, c), np.cross(c, a), np.cross(a, b)], axis=2)
        normals = np.linalg.solve(cofactor, posed.normals[..., None])[..., 0]
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        result.normals = normals.astype(np.float32)
        self.part.mesh, self.part.stale = result, True
        self.result = result
