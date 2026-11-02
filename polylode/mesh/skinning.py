"""The ways a skinned vertex follows its bones, on the CPU.

*Linear blend* weights the bones' matrices; it is what the game engines the
converted characters come from use.  *Dual quaternion* blend weights the
bones' rigid motions instead, so a twisted joint keeps its volume; it is what
a mesh deformed with an armature's "preserve volume" follows, and a surface
built against that deformation only keeps its shape under the same one.

The dual-quaternion path is transcribed from Blender's armature deform
(``mat4_to_dquat``, ``add_weighted_dq_dq_pivot``, ``normalize_dq``,
``mul_v3m3_dq``), step for step, because "the same deformation" has to mean
the same numbers: ``tools/import_endfield_reconstruction.py --scene`` measures
this module against that host's evaluated vertices.  What that takes:

* a bone's motion is split into a rotation and a scale **in the bone's own
  rest frame**, keeping the bone's Y axis (an armature's bone axis) — so the
  palette needs each bone's rest frame, not only its skinning matrix;
* a scaled bone's scale is not blended as a matrix: for each vertex it is
  turned into the translation it gives that vertex and folded into the bone's
  dual part;
* quaternions are summed towards the running sum's hemisphere, the sum is
  divided by the total weight, and the unit length is restored only when the
  vertex is transformed;
* the surface around a vertex is carried by the vertex's *deform matrix* —
  the blended rotation times the weighted sum of the bones' scale matrices,
  an unscaled bone counting as identity.  A tangent goes through the matrix
  and a normal through its cofactor (the inverse transpose times the
  determinant's sign): the normal of the deformed triangle as its corners now
  wind, which under a mirroring bone points the other way.  Measured on the
  probe's vertices under scaled bones, against the host's own normals: turning
  the normal alone is a median 4.0° off under uneven scale and 63° under a
  mirror, the inverse transpose 0.07° and 178°, the cofactor 0.07° and 1.8°.

A surface can also be *mixed*: each vertex takes a share of the
dual-quaternion blend and the rest of the linear one, from the same weights
(``Mesh.skin_dual``).  That is what two armature modifiers on one mesh give
when the second is told to work from the same input as the first and mix by a
vertex group — a surface that is the game's own skin in one place, a
volume-preserving body in another, and runs from one into the other between.
Its deform matrix is the same mix of the two blends' matrices.

``gl/shaders.py`` holds the same arithmetic for the GPU, fed by
:func:`palette_texels`; ``tools/check_skin_influences.py`` holds the two equal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: ``Part.skinning`` values
LINEAR, DUAL_QUATERNION, MIXED = "linear", "dual_quaternion", "mixed"
MODES = (LINEAR, DUAL_QUATERNION, MIXED)
#: texels per bone in the bone texture: the matrix alone, and with the palette
MATRIX_TEXELS, PALETTE_TEXELS = 4, 10
#: squared length of (column sizes − 1) above which a bone counts as scaled,
#: and how far from orthonormal its matrix may be (the host's own thresholds)
_SCALE_EPSILON, _ORTHONORMAL_EPSILON = 1e-8, 1e-5


@dataclass
class DualPalette:
    """Per bone: the rigid motion as a dual quaternion, and the scale beside it."""
    quat: np.ndarray       # (N, 4) w, x, y, z
    trans: np.ndarray      # (N, 4) the dual part
    scale: np.ndarray      # (N, 4, 4) identity for an unscaled bone
    scaled: np.ndarray     # (N,) bool


def linear_frames(positions: np.ndarray, bone_idx: np.ndarray, bone_wt: np.ndarray,
                  matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(positions, 3×3 per vertex)`` under the weighted sum of the bones' *matrices*."""
    blend = np.einsum("vk,vkij->vij", bone_wt.astype(np.float64), matrices[bone_idx])
    points = np.c_[positions.astype(np.float64), np.ones(len(positions))]
    return np.einsum("vij,vj->vi", blend, points)[:, :3], blend[:, :3, :3]


def linear_blend(positions: np.ndarray, bone_idx: np.ndarray, bone_wt: np.ndarray,
                 matrices: np.ndarray) -> np.ndarray:
    """Vertices under the weighted sum of their bones' 4×4 *matrices*, float64."""
    return linear_frames(positions, bone_idx, bone_wt, matrices)[0]


def rest_frames(bind_worlds: np.ndarray) -> np.ndarray:
    """Each bone's rest world with its scale taken out (the polar rotation)."""
    worlds = np.asarray(bind_worlds, dtype=np.float64)
    u, _s, vt = np.linalg.svd(worlds[:, :3, :3])
    flip = np.linalg.det(u @ vt) < 0
    u[flip, :, -1] *= -1.0
    out = np.tile(np.eye(4), (len(worlds), 1, 1))
    out[:, :3, :3] = u @ vt
    out[:, :3, 3] = worlds[:, :3, 3]
    return out


def _quaternions(rotations: np.ndarray) -> np.ndarray:
    """Unit quaternions (w, x, y, z) of (N, 3, 3) rotations; either sign."""
    m = rotations
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    candidates = np.stack([
        np.stack([1.0 + trace, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0],
                  m[:, 1, 0] - m[:, 0, 1]], axis=1),
        np.stack([m[:, 2, 1] - m[:, 1, 2], 1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
                  m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0]], axis=1),
        np.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 0, 1] + m[:, 1, 0],
                  1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], m[:, 1, 2] + m[:, 2, 1]], axis=1),
        np.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1],
                  1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], axis=1)], axis=1)
    # the candidate with the longest vector is the well-conditioned one
    best = np.argmax(np.einsum("ncj,ncj->nc", candidates, candidates), axis=1)
    chosen = candidates[np.arange(len(m)), best]
    return chosen / np.linalg.norm(chosen, axis=1, keepdims=True)


def _unit(vectors: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(length > 0.0, length, 1.0)


def dual_palette(matrices: np.ndarray, rests: np.ndarray) -> DualPalette:
    """The dual-quaternion palette of skinning *matrices* (``posed · rest⁻¹``).

    *rests* are the bones' rest frames (:func:`rest_frames`).  An unscaled
    bone's matrix is its rigid motion as it is.  A scaled one is split in the
    posed bone's frame: the rotation keeps the bone's Y axis, and what is left
    is the scale, carried back into the space the vertices are in.
    """
    matrices = np.asarray(matrices, dtype=np.float64)
    rests = np.asarray(rests, dtype=np.float64)
    posed = matrices @ rests
    columns = posed[:, :3, :3]
    size = np.linalg.norm(columns, axis=1)
    linear = matrices[:, :3, :3]
    gram = np.einsum("nki,nkj->nij", linear, linear)
    scaled = ((np.abs(gram - np.eye(3)).max(axis=(1, 2)) > _ORTHONORMAL_EPSILON)
              | (np.linalg.det(linear) < 0.0)
              | (((size - 1.0) ** 2).sum(axis=1) > _SCALE_EPSILON))
    rigid = matrices.copy()
    scale = np.tile(np.eye(4), (len(matrices), 1, 1))
    if scaled.any():
        chosen = posed[scaled]
        y = _unit(chosen[:, :3, 1])
        z = _unit(np.cross(chosen[:, :3, 0], chosen[:, :3, 1]))
        turned = np.tile(np.eye(4), (len(chosen), 1, 1))
        turned[:, :3, 0], turned[:, :3, 1], turned[:, :3, 2] = np.cross(y, z), y, z
        turned[:, :3, 3] = chosen[:, :3, 3]
        inverse = np.linalg.inv(rests[scaled])
        rigid[scaled] = turned @ inverse
        scale[scaled] = rests[scaled] @ (np.linalg.inv(turned) @ chosen) @ inverse
    q = _quaternions(rigid[:, :3, :3])
    t = rigid[:, :3, 3]
    trans = 0.5 * np.stack([
        -(t[:, 0] * q[:, 1] + t[:, 1] * q[:, 2] + t[:, 2] * q[:, 3]),
        t[:, 0] * q[:, 0] + t[:, 1] * q[:, 3] - t[:, 2] * q[:, 2],
        -t[:, 0] * q[:, 3] + t[:, 1] * q[:, 0] + t[:, 2] * q[:, 1],
        t[:, 0] * q[:, 2] - t[:, 1] * q[:, 1] + t[:, 2] * q[:, 0]], axis=1)
    return DualPalette(q, trans, scale, scaled)


def _blended(positions: np.ndarray, bone_idx: np.ndarray, bone_wt: np.ndarray,
             palette: DualPalette) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each vertex's summed dual quaternion and scale: ``(real, dual, 3×3)``.

    All three divided by the vertex's total weight.  The scale is the weighted
    sum of the scaled bones' matrices with identity for the rest; a flipped
    quaternion does not flip its bone's share of it.
    """
    points = positions.astype(np.float64)
    weights = bone_wt.astype(np.float64)
    real = np.zeros((len(points), 4))
    dual = np.zeros((len(points), 4))
    scale = np.zeros((len(points), 3, 3))
    scale_weight = np.zeros(len(points))
    for slot in range(bone_idx.shape[1]):
        bone, weight = bone_idx[:, slot], weights[:, slot]
        q, t = palette.quat[bone], palette.trans[bone].copy()
        matrix = palette.scale[bone]
        # what the bone's scale moves this vertex by, as part of its translation
        moved = np.einsum("vij,vj->vi", matrix[:, :3, :3], points) + matrix[:, :3, 3] - points
        moved[~palette.scaled[bone]] = 0.0
        t[:, 0] -= 0.5 * (q[:, 1] * moved[:, 0] + q[:, 2] * moved[:, 1] + q[:, 3] * moved[:, 2])
        t[:, 1] += 0.5 * (q[:, 0] * moved[:, 0] + q[:, 2] * moved[:, 2] - q[:, 3] * moved[:, 1])
        t[:, 2] += 0.5 * (q[:, 0] * moved[:, 1] + q[:, 3] * moved[:, 0] - q[:, 1] * moved[:, 2])
        t[:, 3] += 0.5 * (q[:, 0] * moved[:, 2] + q[:, 1] * moved[:, 1] - q[:, 2] * moved[:, 0])
        signed = np.where(np.einsum("vj,vj->v", q, real) < 0.0, -weight, weight)
        real += signed[:, None] * q
        dual += signed[:, None] * t
        share = np.where(palette.scaled[bone], weight, 0.0)
        scale += share[:, None, None] * matrix[:, :3, :3]
        scale_weight += share
    total = weights.sum(axis=1)
    total = np.where(total > 0.0, total, 1.0)
    scale += (total - scale_weight)[:, None, None] * np.eye(3)
    return real / total[:, None], dual / total[:, None], scale / total[:, None, None]


def dual_rotations(real: np.ndarray) -> np.ndarray:
    """(V, 3, 3) rotation of each summed quaternion, unit length restored."""
    w, x, y, z = real.T
    rows = np.stack([
        np.stack([w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), w * w + y * y - x * x - z * z, 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), w * w + z * z - x * x - y * y], axis=1)],
        axis=1)
    length = np.einsum("vj,vj->v", real, real)
    return rows / np.where(length > 0.0, length, 1.0)[:, None, None]


def dual_quaternion_frames(positions: np.ndarray, bone_idx: np.ndarray, bone_wt: np.ndarray,
                           palette: DualPalette) -> tuple[np.ndarray, np.ndarray]:
    """``(positions, deform matrix per vertex)`` under the dual-quaternion blend."""
    real, dual, scale = _blended(positions, bone_idx, bone_wt, palette)
    w, x, y, z = real.T
    t0, t1, t2, t3 = dual.T
    shift = 2.0 * np.stack([-t0 * x + w * t1 - t2 * z + y * t3,
                            -t0 * y + t1 * z - x * t3 + w * t2,
                            -t0 * z + x * t2 + w * t3 - t1 * y], axis=1)
    length = np.einsum("vj,vj->v", real, real)
    shift /= np.where(length > 0.0, length, 1.0)[:, None]
    rotation = dual_rotations(real)
    moved = np.einsum("vij,vj->vi", rotation, positions.astype(np.float64)) + shift
    return moved, rotation @ scale


def dual_quaternion_blend(positions: np.ndarray, bone_idx: np.ndarray, bone_wt: np.ndarray,
                          palette: DualPalette) -> np.ndarray:
    """Vertices under the weighted blend of their bones' rigid motions, float64."""
    return dual_quaternion_frames(positions, bone_idx, bone_wt, palette)[0]


def dual_share(share: np.ndarray | None, count: int) -> np.ndarray:
    """``(count,)`` float64: each vertex's share of the dual-quaternion blend.

    Refuses a missing stream, one of another length and a share outside 0‥1:
    a mixed part without its shares has no deformation to fall back on.
    """
    if share is None or len(share) != count:
        raise ValueError(f"mixed skinning needs a dual-quaternion share per vertex ({count})")
    out = np.asarray(share, dtype=np.float64).reshape(count)
    if not np.all(np.isfinite(out)) or out.min() < 0.0 or out.max() > 1.0:
        raise ValueError("a dual-quaternion share is outside 0..1")
    return out


def mixed_frames(positions: np.ndarray, bone_idx: np.ndarray, bone_wt: np.ndarray,
                 matrices: np.ndarray, palette: DualPalette,
                 share: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(positions, deform matrix per vertex)``: *share* of the dual blend, the rest linear."""
    share = dual_share(share, len(positions))
    points, linear = linear_frames(positions, bone_idx, bone_wt, matrices)
    turned, dual = dual_quaternion_frames(positions, bone_idx, bone_wt, palette)
    return (points + share[:, None] * (turned - points),
            linear + share[:, None, None] * (dual - linear))


def frames(positions: np.ndarray, bone_idx: np.ndarray, bone_wt: np.ndarray,
           matrices: np.ndarray, mode: str = LINEAR, rests: np.ndarray | None = None,
           share: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(positions, 3×3 per vertex)`` posed by *matrices* the way *mode* says.

    *rests* are the bones' rest frames, needed by every mode but the linear
    one; *share* the vertices' dual-quaternion shares of a mixed surface.
    """
    if mode == LINEAR:
        return linear_frames(positions, bone_idx, bone_wt, matrices)
    if mode not in MODES or rests is None:
        raise ValueError(f"unsupported skinning {mode!r}, or no rest frames for it")
    palette = dual_palette(matrices, rests)
    if mode == MIXED:
        return mixed_frames(positions, bone_idx, bone_wt, matrices, palette, share)
    return dual_quaternion_frames(positions, bone_idx, bone_wt, palette)


def skinned(positions: np.ndarray, bone_idx: np.ndarray, bone_wt: np.ndarray,
            matrices: np.ndarray, mode: str = LINEAR, rests: np.ndarray | None = None,
            share: np.ndarray | None = None) -> np.ndarray:
    """Vertices posed by *matrices* the way *mode* says."""
    return frames(positions, bone_idx, bone_wt, matrices, mode, rests, share)[0]


def posed_normals(deform: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Unit normals carried by the vertices' 3×3 *deform* (:func:`frames`).

    Through the cofactor — the inverse transpose, turned round where the
    matrix mirrors — which keeps a normal square to a surface a bone scales
    unevenly and on the side the deformed triangle faces.  The same for every
    blend, and what every path of ``gl/shaders.py`` does.  The cofactor's
    columns are the cross products of the matrix's, so a flattened (singular)
    matrix still gives the one direction left.
    """
    a, b, c = deform[:, :, 0], deform[:, :, 1], deform[:, :, 2]
    cofactor = np.stack([np.cross(b, c), np.cross(c, a), np.cross(a, b)], axis=2)
    return _unit(np.einsum("vij,vj->vi", cofactor, normals.astype(np.float64)))


def palette_texels(matrices: np.ndarray, palette: DualPalette | None = None) -> np.ndarray:
    """The bone texture's rows: ``(N, 4 or 10, 4)`` float32.

    Texels 0‥3 are the matrix's columns.  With a *palette*: 4 the quaternion,
    5 the dual part, 6‥8 the scale's first three rows, 9 ``(scaled, 0, 0, 0)``.
    """
    matrices = np.asarray(matrices, dtype=np.float32).reshape(-1, 4, 4)
    columns = np.stack([matrices[:, :, c] for c in range(4)], axis=1)
    if palette is None:
        return np.ascontiguousarray(columns, dtype=np.float32)
    flags = np.zeros((len(matrices), 1, 4), dtype=np.float32)
    flags[:, 0, 0] = palette.scaled
    return np.ascontiguousarray(np.concatenate([
        columns, palette.quat[:, None, :], palette.trans[:, None, :],
        palette.scale[:, :3, :], flags], axis=1), dtype=np.float32)
