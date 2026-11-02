"""Declared authored normals transported by physical incident-face rotations."""
from __future__ import annotations

import numpy as np

from ..mesh import skinning


def load_source(descriptor, root, skeleton, vertex_count):
    """Read a pinned, bounded source dependency; remap bones by declared names."""
    from .body_archive import file_digest
    from ..mesh.skin import validate_streams
    from ..mesh.pose_corrective import PoseCorrective
    if descriptor.get("version") != 1 or descriptor.get("kind") != "physical_source_rotation_988":
        raise ValueError("unsupported declared source normal transport")
    path = (root / descriptor["file"]).resolve()
    if (not path.is_relative_to(root.resolve()) or not path.is_file()
            or file_digest(path) != descriptor.get("sha256")):
        raise ValueError("declared source normal dependency is changed or outside its store")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    n = len(data["positions"])
    vertices, labels = data["vertices"], data["labels"]
    c = len(labels)
    corners = np.asarray(descriptor["corners"], dtype=np.int64)
    if (data["positions"].shape != (n, 3) or vertices.shape != (c, 3)
            or data["inverse"].shape != (c, 3, 3) or data["authored"].shape != (c, 3)
            or labels.ndim != 1 or corners.shape != (vertex_count,)
            or vertices.dtype.kind not in "iu" or labels.dtype.kind not in "iu"
            or np.any(vertices < 0) or np.any(vertices >= n)
            or np.any(corners < 0) or np.any(corners >= c)
            or not all(np.isfinite(value).all() for value in data.values())):
        raise ValueError("invalid declared physical source normal arrays")
    mapping = np.array([skeleton.index[name] for name in descriptor["bone_names"]], np.int32)
    validate_streams(data["bone_idx"], data["bone_wt"], n)
    if np.any(data["bone_idx"] >= len(mapping)):
        raise ValueError("declared source normal bone index is outside its palette")
    data["bone_idx"] = mapping[data["bone_idx"]]
    if descriptor["skinning"] not in skinning.MODES:
        raise ValueError("unsupported source normal skin blend")
    if descriptor["skinning"] == skinning.MIXED:
        skinning.dual_share(data.get("dual_share"), n)
    corrective = None
    if descriptor.get("corrective") is not None:
        found = descriptor["corrective"]
        corrective = PoseCorrective([skeleton.index[name] for name in found["drivers"]],
                                    found["terms"], data["corrective_vertices"],
                                    data["corrective_deltas"])
    return data, corners, corrective


def transported(points, vertices, labels, inverse, authored):
    """The source graph's uniform corner accumulation and seven polar steps."""
    corners = points[vertices]
    edge = corners[:, 1] - corners[:, 0]
    last = corners[:, 2] - corners[:, 0]
    length = np.linalg.norm(edge, axis=1)
    cross = np.cross(edge, last)
    area = np.linalg.norm(cross, axis=1)
    if np.any(length <= 0) or np.any(area <= 0):
        raise ValueError("declared normal transport has a degenerate posed source face")
    tangent, normal = edge / length[:, None], cross / area[:, None]
    basis = np.stack([tangent, np.cross(normal, tangent), normal], axis=2)
    rotation = basis @ inverse
    _, group = np.unique(labels, return_inverse=True)
    total = np.zeros((group.max() + 1, 3, 3), np.float64)
    np.add.at(total, group, rotation)
    total /= np.bincount(group)[:, None, None]
    if not np.isfinite(total).all():
        raise ValueError("declared physical source rotation is not finite")
    folded = np.linalg.det(total) <= 0
    if folded.all():
        raise ValueError("declared physical source rotation is nonpositive")
    for _ in range(7):
        total[~folded] = (total[~folded] + np.linalg.inv(total[~folded]).transpose(0, 2, 1)) * 0.5
    turn = total[group]
    # A group whose faces fold over in this pose has no mean turn (its sum is singular: Si's
    # shoulders standing, two groups of six faces, §7.1089).  Each of its corners takes its own
    # face's turn, the same step for a group of one face; the rest of the surface is unchanged.
    alone = folded[group]
    turn[alone] = rotation[alone]
    result = np.einsum("vij,vj->vi", turn, authored)
    length = np.linalg.norm(result, axis=1)
    if not np.isfinite(result).all() or np.any(length <= 0):
        raise ValueError("declared authored normal transport is invalid")
    return result / length[:, None]


class SourceNormals:
    """Refresh normal inputs after pose corrections; keep GPU position skinning."""
    after_correctives = True

    def __init__(self, part, data, corners, mode, corrective=None, parents=None):
        self.part, self.data, self.corners, self.mode = part, data, corners, mode
        self.corrective, self.parents = corrective, parents
        self.last = None
        self.last_mesh = None

    def remap_bones(self, index) -> None:
        """Say the source skin's bone indices as another skeleton's (*index*: old bone → new; ``converted_rig.Rebound``).

        Only where each new bone skins as the old one did: a pose corrective
        reads its drivers' turns in the old skeleton's rest frames, and a
        declared source program its own bone table — ``ValueError`` for either.
        """
        if self.corrective is not None or hasattr(self, "source_motion"):
            raise ValueError("a source normal transport with a pose corrective or a source program "
                             "is not carried onto another skeleton")
        self.data["bone_idx"] = np.asarray(index, dtype=np.int64)[self.data["bone_idx"]].astype(np.int32)
        self.last = None

    def update(self, matrices, rests):
        if (self.last is not None and np.array_equal(self.last, matrices)
                and self.part.mesh is self.last_mesh):
            return
        d = self.data
        rest = d["positions"]
        if self.corrective is not None:
            from ..mesh import pose_corrective
            vectors = pose_corrective.rotation_vectors(matrices, rests, self.parents,
                                                       self.corrective.drivers)
            rest = self.corrective.corrected(rest, vectors)
        if hasattr(self, 'source_motion'):
            points = self.source_motion.points(matrices)
        else:
            points, _ = skinning.frames(rest, d["bone_idx"], d["bone_wt"], matrices,
                                       self.mode, rests, d.get("dual_share"))
        wanted = transported(points, d["vertices"], d["labels"], d["inverse"],
                             d["authored"])[self.corners]
        mesh = self.part.mesh
        if not self.part.skinned:
            mesh.normals = wanted.astype(np.float32)
            self.part.stale = True
            self.last, self.last_mesh = matrices.copy(), mesh
            return
        _, deform = skinning.frames(mesh.positions, mesh.bone_idx, mesh.bone_wt, matrices,
                                    self.part.skinning, rests,
                                    mesh.skin_dual if len(mesh.skin_dual) else None)
        a, b, c = deform[:, :, 0], deform[:, :, 1], deform[:, :, 2]
        cofactor = np.stack([np.cross(b, c), np.cross(c, a), np.cross(a, b)], axis=2)
        normals = np.linalg.solve(cofactor, wanted[..., None])[..., 0]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        if not np.isfinite(normals).all():
            raise ValueError("source normal input cannot pass through the declared skin blend")
        mesh.normals = normals.astype(np.float32)
        self.part.stale = True
        self.last = matrices.copy()
        self.last_mesh = mesh

def joined_surface_normals(positions, triangles):
    """Average incident face areas at exact shared positions, including UV splits.

    Return normals and a used-vertex mask. Unused vertices have zero normals;
    callers must preserve their existing normals rather than normalize zeros.
    Coincident positions are joined only within the explicitly supplied surface.
    """
    points = np.asarray(positions, dtype=np.float64)
    faces = np.asarray(triangles)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('positions must be finite XYZ vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError('triangles must contain integer vertex indices')
    if not faces.size or faces.min() < 0 or faces.max() >= len(points):
        raise ValueError('triangles must reference the supplied surface')
    unique, weld = np.unique(points, axis=0, return_inverse=True)
    vectors = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                       points[faces[:, 2]] - points[faces[:, 0]])
    sums = np.zeros_like(unique)
    np.add.at(sums, weld[faces].ravel(), np.repeat(vectors, 3, axis=0))
    used = np.zeros(len(unique), dtype=bool)
    used[weld[faces].ravel()] = True
    lengths = np.linalg.norm(sums, axis=1)
    if np.any(lengths[used] == 0):
        raise ValueError('rendered surface has undefined incident normals')
    sums[used] /= lengths[used, None]
    return sums[weld], used[weld]
