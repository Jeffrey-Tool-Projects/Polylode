"""Corrections to a skinned surface that are driven by how its bones are turned.

Bone weights make a vertex move as a blend of its bones' motions.  A surface
that was built some other way — a node graph solving a transition between two
skins — also moves with the pose, but not as any such blend: turn one bone one
way and the other and part of the vertex's shift is the *same* both times.  No
weight can say that (``mesh/skin_fit.py`` leaves it as the residual).

What can is a small offset added to the vertex's rest position before it is
skinned, as a polynomial in the rotation vectors of a few *driver* bones::

    offset(v) = Σ terms  delta[term, v] · Π factors θ[bone][axis]

A term is a product of components of the drivers' local rotations (relative to
their rest, as rotation vectors in the bone's own rest axes): ``θneck.x``,
``θneck.x·θneck.y``, ``θneck.x·θhead.z`` … — one bone at a time first, then
bones together.  The deltas are fitted order by order (:func:`fit`): a term of
several bones is fitted to what the terms of fewer bones leave over, on poses
that turn exactly those bones.  Offsets ride through the skinning with the
vertex, so they turn with it.

Stored as blend shapes whose names spell the term (:func:`as_shapes`), which
is what the archive already keeps per piece.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

#: prefix of a blend shape that is a pose-driven term, not an authored shape
PREFIX = "pose:"
AXES = "xyz"
#: highest total degree of a term, by how many bones it multiplies
DEGREE = {1: 3, 2: 2, 3: 3}
#: a group of bones is fitted only with this many times more poses than terms
POSES_PER_TERM = 2.0


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """The rotation vector (axis · angle) of a 3×3 rotation."""
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    skew = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    if angle < 1e-9:
        return 0.5 * skew
    if np.pi - angle < 1e-6:
        # near a half turn the skew part vanishes: the axis is in R + I
        column = (rotation + np.eye(3))[:, int(np.argmax(np.diag(rotation)))]
        return column / np.linalg.norm(column) * angle
    return skew * (angle / (2.0 * np.sin(angle)))


def rotation_vectors(skin: np.ndarray, frames: np.ndarray, parents: np.ndarray,
                     bones: list) -> np.ndarray:
    """``(len(bones), 3)``: each bone's local turn from rest, in its own rest axes.

    *skin* are the skinning matrices (``posed · rest⁻¹``), *frames* the bones'
    rest frames without scale (``Skeleton.rest_frames``), *parents* the parent
    index of every bone (−1: none).  A bone's posed frame is its skinning
    matrix on its rest frame; its local turn is that frame seen from its
    parent's, less the same at rest.
    """
    out = np.zeros((len(bones), 3))
    for row, bone in enumerate(bones):
        posed = (skin[bone] @ frames[bone])[:3, :3]
        rest = frames[bone][:3, :3]
        parent = int(parents[bone])
        if parent >= 0:
            posed = (skin[parent] @ frames[parent])[:3, :3].T @ posed
            rest = frames[parent][:3, :3].T @ rest
        out[row] = _rotation_vector(rest.T @ posed)
    return out


def group_terms(members: tuple, total: int) -> list:
    """Terms that multiply a component of *every* member, of total degree ≤ *total*.

    A term is a sorted tuple of ``(driver slot, axis)`` factors.
    """
    out = []
    for degrees in itertools.product(range(1, total + 1), repeat=len(members)):
        if sum(degrees) > total:
            continue
        choices = [[tuple((member, axis) for axis in combo)
                    for combo in itertools.combinations_with_replacement(range(3), degree)]
                   for member, degree in zip(members, degrees)]
        out += [tuple(sorted(itertools.chain(*combo))) for combo in itertools.product(*choices)]
    return out


def term_values(vectors: np.ndarray, terms: list) -> np.ndarray:
    """``(T,)``: each term's product of rotation-vector components."""
    return np.array([np.prod([vectors[slot, axis] for slot, axis in term]) for term in terms])


@dataclass
class PoseCorrective:
    """Per-vertex rest offsets as a polynomial in driver bones' rotation vectors."""
    drivers: list              # skeleton bone indices, by driver slot
    terms: list                # per term: ((slot, axis), …)
    vertices: np.ndarray       # (S,) indices of the vertices that are corrected
    deltas: np.ndarray         # (T, S, 3)

    def offsets(self, vectors: np.ndarray) -> np.ndarray:
        """``(S, 3)`` for the drivers' rotation vectors ``(D, 3)``."""
        return np.einsum("t,tvi->vi", term_values(vectors, self.terms), self.deltas)

    def corrected(self, positions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """*positions* (rest) with the offsets added; a copy."""
        out = np.array(positions, dtype=np.float64)
        out[self.vertices] += self.offsets(vectors)
        return out


def fit(vectors: np.ndarray, missed: np.ndarray, moved: list, drivers: list,
        floor: float) -> PoseCorrective | None:
    """Fit the terms' deltas, one group of bones after another.

    *vectors* ``(P, D, 3)`` the drivers' rotation vectors in each pose;
    *missed* ``(P, V, 3)`` what the weights leave of each vertex's seen
    motion, in rest space; *moved* per pose the set of driver slots the pose
    turns (poses that turn any other bone are not passed).  Groups of one
    bone, then two, then three: each is fitted on the poses that turn exactly
    its bones, to what the smaller groups leave, and only when there are
    :data:`POSES_PER_TERM` times more such poses than it has terms.  Vertices
    no term moves by more than *floor* are left out; ``None`` when none is.
    """
    terms, deltas = [], []
    left = np.array(missed, dtype=np.float64)
    for order in sorted(DEGREE):
        for members in sorted({tuple(sorted(group)) for group in moved if len(group) == order}):
            rows = [pose for pose, group in enumerate(moved) if tuple(sorted(group)) == members]
            own = group_terms(members, DEGREE[order])
            if len(rows) < POSES_PER_TERM * len(own):
                continue
            design = np.stack([term_values(vectors[pose], own) for pose in rows])
            solved = np.einsum("tn,nvi->tvi", np.linalg.pinv(design), left[rows])
            # every pose that turns these bones (with others or not) carries the term
            for pose, group in enumerate(moved):
                if set(members) <= set(group):
                    left[pose] -= np.einsum("t,tvi->vi", term_values(vectors[pose], own), solved)
            terms += own
            deltas.append(solved)
    if not terms:
        return None
    stacked = np.concatenate(deltas)
    # what the terms take up, pose by pose: a vertex they never move by *floor* is not theirs
    taken = np.linalg.norm(np.asarray(missed, dtype=np.float64) - left, axis=2).max(axis=0)
    kept = np.flatnonzero(taken > floor)
    if not len(kept):
        return None
    return PoseCorrective(list(drivers), terms, kept.astype(np.int32),
                          stacked[:, kept].astype(np.float32))


def as_shapes(corrective: PoseCorrective, names: list, rows: np.ndarray) -> list:
    """The corrective as ``(name, indices, deltas)`` blend shapes of one mesh.

    *names* are the driver bones' names by slot; *rows* maps each of the
    mesh's vertices to the corrected surface's vertex it is (a mesh split into
    render vertices repeats them).  Shapes that touch none of the mesh's
    vertices are still written, empty: the terms are one list.
    """
    slot = {int(vertex): k for k, vertex in enumerate(corrective.vertices)}
    picked = [(row, slot[int(vertex)]) for row, vertex in enumerate(rows) if int(vertex) in slot]
    index = np.array([row for row, _ in picked], dtype=np.int32)
    source = np.array([k for _, k in picked], dtype=np.int64)
    out = []
    for term, delta in zip(corrective.terms, corrective.deltas):
        label = PREFIX + "*".join(f"{names[s]}.{AXES[axis]}" for s, axis in term)
        out.append((label, index, delta[source].astype(np.float32)))
    return out


def from_shapes(shapes: list | None, bone_index: dict) -> PoseCorrective | None:
    """The corrective a piece's shapes spell, or ``None`` when it has none.

    *bone_index* maps a bone name to its skeleton index.  Authored shapes
    (without :data:`PREFIX`) are not touched.
    """
    own = [shape for shape in shapes or [] if str(shape[0]).startswith(PREFIX)]
    if not own:
        return None
    drivers, terms = [], []
    for shape in own:
        term = []
        for factor in str(shape[0])[len(PREFIX):].split("*"):
            bone, _, axis = factor.rpartition(".")
            if bone not in bone_index or axis not in AXES:
                raise ValueError(f"pose term {shape[0]!r} names no bone and axis of this rig")
            if bone_index[bone] not in drivers:
                drivers.append(bone_index[bone])
            term.append((drivers.index(bone_index[bone]), AXES.index(axis)))
        terms.append(tuple(term))
    vertices = np.asarray(own[0][1], dtype=np.int32)
    if any(not np.array_equal(np.asarray(shape[1]), vertices) for shape in own):
        raise ValueError("the pose terms of one piece do not share their vertices")
    return PoseCorrective(drivers, terms, vertices, np.stack([np.asarray(s[2]) for s in own]))
