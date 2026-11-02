"""Take a set of faces out of a surface along the mesh's own lines.

A set of faces picked by a measurement (which lie over something, which show
a colour) has a rim that zigzags by the triangle: half of a quad in, half
out; a single face sticking out or in.  A seam is then asked to follow that
rim, and its bridge folds on every tooth.  :func:`tidy` moves the set's rim
onto the lines the mesh was modelled with.
"""
from __future__ import annotations

import numpy as np


def _welded(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    _, welded = np.unique(np.round(np.asarray(positions, dtype=np.float64) / 1e-6).astype(np.int64), axis=0,
                          return_inverse=True)
    return welded.reshape(-1)[triangles]


def neighbours(corners: np.ndarray) -> np.ndarray:
    """``(F, 3)``: the face across each edge of every face (edge ``k`` is corners ``k`` and ``k + 1``), -1 for none."""
    edges = np.sort(np.stack([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]], axis=1), axis=2).reshape(-1, 2)
    owner = np.repeat(np.arange(len(corners)), 3)
    slot = np.tile(np.arange(3), len(corners))
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner, slot = edges[order], owner[order], slot[order]
    out = np.full((len(corners), 3), -1, dtype=np.int64)
    pair = np.flatnonzero(np.all(edges[1:] == edges[:-1], axis=1))
    out[owner[pair], slot[pair]] = owner[pair + 1]
    out[owner[pair + 1], slot[pair + 1]] = owner[pair]
    return out


def quad_mates(positions: np.ndarray, corners: np.ndarray, across: np.ndarray) -> np.ndarray:
    """``(F,)``: the face each face makes a quad with — the one across its longest edge — or -1.

    Two triangles are a quad's halves when each is across the other's longest
    edge (the quad's diagonal).
    """
    points = np.asarray(positions, dtype=np.float64)
    length = np.stack([np.linalg.norm(points[corners[:, k]] - points[corners[:, (k + 1) % 3]], axis=1) for k in range(3)], axis=1)
    longest = np.argmax(length, axis=1)
    mate = across[np.arange(len(corners)), longest]
    mutual = (mate >= 0) & (np.where(mate >= 0, mate[np.maximum(mate, 0)], -1) == np.arange(len(corners)))
    return np.where(mutual, mate, -1)


def tidy(positions: np.ndarray, triangles: np.ndarray, faces: np.ndarray, picked: np.ndarray, grow: int = 0) -> np.ndarray:
    """*picked* (a bool per row of *faces*) with its rim moved onto the mesh's quads and its teeth taken off.

    *faces* are rows of *triangles* that make one surface; the result is a
    bool per row of *faces* again.  The unit is a quad (both halves, picked
    when either was) or a triangle that is half of none.  Then, until nothing
    changes: a unit of the rest with more than half of its outer edges on
    picked units is picked too (a tooth of the rest sticking into the set),
    and a picked unit with more than half of its outer edges on the rest is
    given back (a tooth of the set).  The surface's own border counts for
    neither, but is one of the edges.  *grow* widens the set by that many
    rings of units first.
    """
    faces = np.asarray(faces, dtype=np.int64)
    corners = _welded(positions, np.asarray(triangles, dtype=np.int64))[faces]
    position = np.zeros((int(corners.max()) + 1, 3))
    position[corners] = np.asarray(positions, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)[faces]]
    across = neighbours(corners)
    mate = quad_mates(position, corners, across)
    index = np.arange(len(faces))
    unit = np.where(mate >= 0, np.minimum(index, mate), index)            # a quad is named by its lower half
    outer = across != mate[:, None]                                       # every edge but the quad's diagonal
    there = outer & (across >= 0)
    other = unit[np.maximum(across, 0)]                                   # the unit across each edge
    edges = np.bincount(unit, weights=outer.sum(axis=1), minlength=len(faces))
    out = np.asarray(picked, dtype=bool).copy()
    out = np.bincount(unit, weights=out, minlength=len(faces))[unit] > 0  # a quad goes whole

    def beside(state: np.ndarray) -> np.ndarray:
        """Per face, how many of its unit's outer edges lie on a unit in *state*."""
        return np.bincount(unit, weights=(there & state[other]).sum(axis=1), minlength=len(faces))[unit]

    for _ in range(grow):
        touched = np.zeros(int(corners.max()) + 1, dtype=bool)
        touched[corners[out]] = True
        out = np.bincount(unit, weights=touched[corners].any(axis=1), minlength=len(faces))[unit] > 0
    for _ in range(len(faces)):
        fill = ~out & (2 * beside(out) > edges[unit])
        if fill.any():
            out = out | fill
            continue
        back = out & (2 * beside(~out) > edges[unit])
        if not back.any():
            break
        out = out & ~back
    return out
