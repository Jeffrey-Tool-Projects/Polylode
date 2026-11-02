"""The averaged normal an outline hull is pushed out along (``_OutlineAverageNormal``, PROGRESS #171).

``HGRP/CharacterNPR*``'s outline vertex program reads, when the material's
``_OutlineAverageNormal`` is set (nearly every outlined skin, cloth and hair
material), a normal baked into ``TEXCOORD2`` instead of the shading normal.
That normal is the surface's own at the vertex's position, whatever seams the
mesh has there: measured against the face normals of every face at the same
position (welded at 1e-5), weighted by the face's angle at the vertex, it
differs by a median 0.15–0.25° and a p95 0.4–0.6°; the shading normal by a
p95 52–70° (§7.1272).  Pushed along the shading normal, a hull splits open at
every hard edge and every UV seam, and the split shows as a line.

The store keeps ``uv2`` but not the tangents it is decoded with, so the
normal is made again here, the measured way.  :func:`averaged` does it for
meshes that were one surface in the game (a converted body's lent skin, its
own skin and the collar given to its head were ``body_01``): a position that
is on more than one of them takes every face round it, and the hulls on both
sides of the weld meet (PROGRESS #161).
"""
from __future__ import annotations

import numpy as np

#: Points closer than this (m) are one position: §7.1272's weld.
WELD = 1e-5
#: The same across meshes sewn together where the game had one: their seam vertices are sewn to within
#: hundredths of a millimetre, not exactly (Typhoea's collar on the lent chest: 14 of its 20 within
#: 0.01 mm, all 20 within 0.1 mm; Wulfa's and EndminF's own skin on the lent body: all within 0.01 mm)
SEAM_WELD = 1e-4


def _groups(points: np.ndarray, weld: float) -> np.ndarray:
    """A group number per point: points within *weld* of one another, chained, share one."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(points).query_pairs(weld, output_type="ndarray")
    n = len(points)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) \
        else coo_matrix((n, n))
    return connected_components(graph, directed=False)[1]


def _corner_angles(corners: np.ndarray) -> np.ndarray:
    """``(F, 3)``: each triangle's angle at each of its corners (radians)."""
    out = np.empty(corners.shape[:2])
    for k in range(3):
        a = corners[:, (k + 1) % 3] - corners[:, k]
        b = corners[:, (k + 2) % 3] - corners[:, k]
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        cos = np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-30)
        out[:, k] = np.arccos(np.clip(cos, -1.0, 1.0))
    return out


def _face_normals(mesh, points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normals of *faces*, each turned the way its corners' shading normals face.

    Winding is not one way over every mesh (a converted woman's own skin is
    wound inward, §7.1103); the shading normals are, since she is lit by them.
    """
    corners = points[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    if len(mesh.normals) == len(points):
        shaded = np.asarray(mesh.normals, dtype=np.float64)[faces].sum(axis=1)
        normal *= np.where(np.einsum("ij,ij->i", normal, shaded) < 0.0, -1.0, 1.0)[:, None]
    return normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)


def averaged(meshes: list, weld: float = WELD, faces: list | None = None) -> list[np.ndarray]:
    """Per mesh of *meshes* (at rest, in one frame), each vertex's averaged normal ``(V, 3)``, float32.

    The angle-weighted sum of the normals of every face of any of *meshes*
    that has a corner at the vertex's position (within *weld*), made unit.  A
    vertex no face uses, or whose faces cancel, keeps its shading normal.
    *faces*: per mesh the ``(F, 3)`` faces that are its surface, or None for
    all of its triangles.
    """
    points = [np.asarray(m.positions, dtype=np.float64) for m in meshes]
    counts = [len(p) for p in points]
    if not sum(counts):
        return [np.zeros((0, 3), np.float32) for _ in meshes]
    group = _groups(np.concatenate(points), weld)
    total = np.zeros((int(group.max()) + 1, 3))
    start = 0
    for k, (mesh, at, n) in enumerate(zip(meshes, points, counts)):
        tris = faces[k] if faces is not None and faces[k] is not None else mesh.indices
        tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        if len(tris):
            weighted = _face_normals(mesh, at, tris)[:, None, :] * _corner_angles(at[tris])[:, :, None]
            np.add.at(total, group[start + tris.ravel()], weighted.reshape(-1, 3))
        start += n
    length = np.linalg.norm(total, axis=1)
    out, start = [], 0
    for mesh, n in zip(meshes, counts):
        mine = group[start:start + n]
        normal = total[mine] / np.maximum(length[mine], 1e-30)[:, None]
        flat = length[mine] <= 1e-12
        if flat.any() and len(mesh.normals) == n:
            normal[flat] = np.asarray(mesh.normals, dtype=np.float64)[flat]
        out.append(normal.astype(np.float32))
        start += n
    return out


def fill(meshes: list, weld: float = WELD, faces: list | None = None) -> None:
    """Give each of *meshes* its :func:`averaged` normals as ``Mesh.outline_normals`` (one surface)."""
    for mesh, normal in zip(meshes, averaged(meshes, weld, faces)):
        mesh.outline_normals = normal


def fill_parts(parts: list, weld: float = WELD) -> None:
    """``Mesh.outline_normals`` for every part of *parts* whose material asks for them (``outline_average``).

    Each mesh is its own surface, as each of the game's meshes has its own
    ``TEXCOORD2``; a mesh shared by several parts is filled once.
    """
    done = set()
    for part in parts:
        mesh = part.mesh
        if id(mesh) in done or not getattr(part.material, "outline_average", False):
            continue
        done.add(id(mesh))
        fill([mesh], weld)


def hull_faces(mesh) -> np.ndarray:
    """The faces of *mesh* its outline hull is drawn with: ``(F, 3)``, all but the folded ones.

    The hull is the mesh drawn again with its front faces culled, so a face
    turned against the surface it is part of -- wound the other way from the
    mesh's own winding, against its corners' shading normals -- is a back face
    on the front of the body and draws as a line.  The surface pass culls the
    same face, so it is never seen there.  Such folds are where a converted
    body's lent skin was sewn onto its own skin (Typhoea's lent chest along
    the collar: 10 of the 34 lent faces within 6 mm of the collar's edge,
    the V line of PROGRESS #161); the game's ``body_01`` is one mesh with no
    fold there.  "The other way" is against the mesh's own majority: a
    converted woman's own skin is wound inward throughout (§7.1103).
    """
    faces = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
    if not len(faces) or len(mesh.normals) != mesh.vertex_count:
        return faces
    points = np.asarray(mesh.positions, dtype=np.float64)
    winding = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    along = np.einsum("ij,ij->i", winding, np.asarray(mesh.normals, dtype=np.float64)[faces].sum(axis=1))
    inward = along < 0.0
    folded = ~inward if inward.sum() * 2 > len(faces) else inward
    return faces[~folded]
