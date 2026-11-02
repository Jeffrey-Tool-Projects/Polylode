"""An outfit taken apart: the pieces of each mesh, and which of them cover which body face.

The game ships an outfit as a few meshes with many separate things in each — a
coat, its buckles, a pouch, a strap.  What can be told apart in the data is
connectivity: a *piece* is a connected run of triangles of one part (vertices
welded where they sit at the very same place, which is how a mesh is cut along
its UV seams).  Nothing here names a piece or guesses what it is.

Taking one piece off raises the question the whole-outfit switch never had to
answer: which of the body's faces was *that piece* covering?  The converted
data holds one list for the whole outfit (``composite``'s ``covered_faces``).
This module measures the same thing piece by piece, by the rule that list was
made with (``sample_blender.coverage``): from a body face's centre, a ray
outward along its normal and one inward, each no longer than half the body's
own thickness there; a face is covered by every piece either ray meets.  A
face is then left out while any piece that covers it is on — so with all on,
the faces left out are the ones the one list names, and with a piece off, what
only it covered comes back while the rest stays covered (less the rim of each
covered patch, :func:`inner`).

All of it is at rest, in the model's space, on triangles as the parts hold
them.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

#: a triangle whose bounding sphere is larger than this many times the median
#: is tested against every ray instead of being looked up by its centre
_LARGE = 8.0


def islands(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """``(T,)`` the piece each triangle is in, numbered from 0 by first triangle.

    Vertices at the same stored place are one: a mesh splits a vertex wherever
    its UVs or normals break, and those copies are the same point.
    """
    triangles = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    if not len(triangles):
        return np.zeros(0, dtype=np.int32)
    _, welded = np.unique(np.ascontiguousarray(positions, dtype=np.float32), axis=0,
                          return_inverse=True)
    corners = welded.reshape(-1)[triangles]
    count = int(corners.max()) + 1
    rows = np.concatenate([corners[:, 0], corners[:, 1]])
    cols = np.concatenate([corners[:, 1], corners[:, 2]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(count, count))
    _, label = connected_components(graph, directed=False)
    of_triangle = label[corners[:, 0]]
    _, first, inverse = np.unique(of_triangle, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))                 # numbered by first appearance
    return order[inverse].astype(np.int32)


def _hits(origins: np.ndarray, directions: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """``(R, T)`` distance along each unit ray to each triangle, ``inf`` where it misses.

    Möller–Trumbore, both faces of a triangle counting.
    """
    a = corners[:, 0][None]
    e1 = (corners[:, 1] - corners[:, 0])[None]
    e2 = (corners[:, 2] - corners[:, 0])[None]
    d = directions[:, None, :]
    p = np.cross(d, e2)
    det = np.einsum("rtk,rtk->rt", e1 * np.ones_like(p), p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origins[:, None, :] - a
    u = np.einsum("rtk,rtk->rt", s, p) * inv
    q = np.cross(s, e1)
    v = np.einsum("rtk,rtk->rt", d * np.ones_like(q), q) * inv
    t = np.einsum("rtk,rtk->rt", e2 * np.ones_like(q), q) * inv
    inside = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 0.0)
    return np.where(inside, t, np.inf)


#: How many (ray, triangle) pairs are tested in one pass (:meth:`Triangles.met_many`): memory, not a rule.
_PAIRS = 400_000
#: A ray is looked along in stretches this many times the widest triangle of the size being looked for.
_STRETCH = 4.0
#: Rays :func:`closed_in` hands :meth:`Triangles.met_many` at a time (memory, not results: each ray is answered alone).
RAY_CHUNK = 20_000


def _pair_hits(origins: np.ndarray, directions: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """``(P,)`` distance along unit ray ``k`` to triangle ``k``, ``inf`` where it misses (:func:`_hits`, pairwise)."""
    a = corners[:, 0]
    e1 = corners[:, 1] - a
    e2 = corners[:, 2] - a
    p = np.cross(directions, e2)
    det = np.einsum("pk,pk->p", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origins - a
    u = np.einsum("pk,pk->p", s, p) * inv
    q = np.cross(s, e1)
    v = np.einsum("pk,pk->p", directions, q) * inv
    t = np.einsum("pk,pk->p", e2, q) * inv
    inside = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 0.0)
    return np.where(inside, t, np.inf)


class Triangles:
    """Triangles to cast rays against, looked up by where they are.

    The small ones go in a tree by their centres and are found by a ball
    around a stretch of the ray; the few much larger than the rest are tested
    against every ray.
    """

    def __init__(self, corners: np.ndarray):
        corners = np.asarray(corners, dtype=np.float64).reshape(-1, 3, 3)
        middle = corners.mean(axis=1) if len(corners) else np.zeros((0, 3))
        spread = (np.linalg.norm(corners - middle[:, None], axis=2).max(axis=1)
                  if len(corners) else np.zeros(0))
        cut = max(float(np.median(spread)) * _LARGE, 1e-9) if len(corners) else 0.0
        self.corners = corners
        self.small = np.flatnonzero(spread <= cut)
        self.large = np.flatnonzero(spread > cut)
        self.radius = float(spread[self.small].max()) if len(self.small) else 0.0
        self.tree = cKDTree(middle[self.small]) if len(self.small) else None
        #: the small ones again, by size: (their indices, the largest spread among them, a tree of
        #: their centres).  A ball has to be as wide as the widest triangle it may hold, and one
        #: tree for all of them makes every ball as wide as the widest of all — on an outfit of
        #: 43,127 triangles, 7 cm round every ray for triangles mostly a few millimetres across.
        self.sizes = []
        if len(self.small):
            step = max(float(np.median(spread[self.small])), 1e-9)
            band = np.ceil(np.log2(np.maximum(spread[self.small] / step, 1.0))).astype(np.int64)
            for which in np.unique(band):
                rows = self.small[band == which]
                self.sizes.append((rows, float(spread[rows].max()), cKDTree(middle[rows])))

    def _near(self, start: np.ndarray, direction: np.ndarray, length: float) -> np.ndarray:
        """The triangles a stretch of one ray could meet."""
        found = self.large
        if self.tree is not None:
            middle = start + direction * (length * 0.5)
            near = self.tree.query_ball_point(middle, length * 0.5 + self.radius)
            if near:
                found = np.concatenate([self.small[np.asarray(near, dtype=np.int64)], found])
        return found

    def met(self, origin: np.ndarray, direction: np.ndarray, length: float) -> tuple:
        """``(triangle indices, distances)`` of everything one unit ray meets within *length*."""
        candidates = self._near(origin, direction, length)
        if not len(candidates):
            return candidates, np.zeros(0)
        reach = _hits(origin[None], direction[None], self.corners[candidates])[0]
        keep = reach <= length
        return candidates[keep], reach[keep]

    def met_many(self, origins: np.ndarray, directions: np.ndarray, lengths) -> tuple:
        """:meth:`met` for many unit rays at once: ``(ray, triangle, distance)`` of every meeting.

        *lengths* is one length or one per ray.  The same triangles are looked
        at as :meth:`met` looks at ray by ray, in one pass over all the pairs
        (:data:`_PAIRS` at a time) instead of a pass per ray.
        """
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        lengths = np.broadcast_to(np.asarray(lengths, dtype=np.float64), (len(origins),))
        none = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0))
        if not len(origins) or not len(self.corners):
            return none
        rays = [np.repeat(np.arange(len(origins)), len(self.large))]
        triangles = [np.tile(self.large, len(origins))]
        for rows, widest, tree in self.sizes:
            # a long ray is looked along in stretches: one ball round all of it holds everything
            # within half its length of its middle, most of which is nowhere near it
            stretch = max(_STRETCH * widest, 1e-9)
            pieces_of = np.maximum(np.ceil(lengths / stretch), 1.0).astype(np.int64)
            owner = np.repeat(np.arange(len(origins)), pieces_of)
            which = np.arange(len(owner)) - np.repeat(np.cumsum(pieces_of) - pieces_of, pieces_of)
            part = (lengths / pieces_of)[owner]
            middles = origins[owner] + directions[owner] * ((which + 0.5) * part)[:, None]
            near = tree.query_ball_point(middles, part * 0.5 + widest, return_sorted=False)
            count = np.fromiter((len(found) for found in near), dtype=np.int64, count=len(near))
            if not count.sum():
                continue
            # each (ray, triangle) once.  By sorting: tens of millions of pairs, and the hash table
            # ``np.unique`` builds for them took more than half of a cover's measurement (§7.1009)
            pair = np.sort(np.repeat(owner, count) * len(self.corners)
                           + rows[np.concatenate([np.asarray(found, dtype=np.int64) for found in near if found])])
            pair = pair[np.concatenate(([True], pair[1:] != pair[:-1]))]
            rays.append(pair // len(self.corners))
            triangles.append(pair % len(self.corners))
        ray, triangle = np.concatenate(rays), np.concatenate(triangles)
        kept = []
        for start in range(0, len(ray), _PAIRS):
            r, t = ray[start:start + _PAIRS], triangle[start:start + _PAIRS]
            distance = _pair_hits(origins[r], directions[r], self.corners[t])
            keep = distance <= lengths[r]
            kept.append((r[keep], t[keep], distance[keep]))
        if not kept:
            return none
        return tuple(np.concatenate(column) for column in zip(*kept))

    def first_many(self, origins: np.ndarray, directions: np.ndarray, reach: float, stride: float) -> np.ndarray:
        """:meth:`first` for many unit rays at once: the distance each first meets something at, ``inf`` if never."""
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        out = np.full(len(origins), np.inf)
        active = np.arange(len(origins))
        start = 0.0
        while start < reach and len(active):
            length = min(stride, reach - start)
            ray, _triangle, distance = self.met_many(origins[active] + directions[active] * start,
                                                     directions[active], length)
            nearest_hit = np.full(len(active), np.inf)
            np.minimum.at(nearest_hit, ray, distance)
            found = np.isfinite(nearest_hit)
            out[active[found]] = start + nearest_hit[found]
            active = active[~found]
            start += length
        return out

    def first(self, origin: np.ndarray, direction: np.ndarray, reach: float,
              stride: float) -> float:
        """Distance to the nearest triangle one unit ray meets within *reach*; ``inf`` if none.

        Marched in stretches of *stride*, so a ray that meets something close
        never looks at what is far.
        """
        start = 0.0
        while start < reach:
            length = min(stride, reach - start)
            _found, distance = self.met(origin + direction * start, direction, length)
            if len(distance):
                return start + float(distance.min())
            start += length
        return float("inf")


#: How far inside a body face a worn surface is looked for (:func:`through`), and how far a worn
#: face has to be turned the way the body face is (cosine) to count as lying over it.
THROUGH_REACH, THROUGH_SAME_WAY = 0.03, 0.3


def through(body: np.ndarray, faces: np.ndarray, worn: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Per row of *faces* (body triangles on *body*, posed), how far it lies outside a worn surface (0: it does not).

    A face is outside when a ray from its middle inward meets, within
    :data:`THROUGH_REACH` and before the body's own far wall, a worn face
    turned the way it is (*worn*: the posed ``(T, 3, 3)`` corners of what is
    worn).  Depths under *floor* count as 0.  ``tools/ef_body_poke.py``'s
    rule, all rays at once.
    """
    depth = np.zeros(len(faces))
    if not len(worn) or not len(faces):
        return depth
    faces = np.asarray(faces, dtype=np.int64)
    corners = body[faces]
    centre = corners.mean(axis=1)
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    start = centre - normal * 1e-5
    turned = np.cross(worn[:, 1] - worn[:, 0], worn[:, 2] - worn[:, 0])
    turned /= np.maximum(np.linalg.norm(turned, axis=1, keepdims=True), 1e-12)
    ray, hit, distance = Triangles(worn).met_many(start, -normal, THROUGH_REACH)
    aligned = np.einsum("ij,ij->i", turned[hit], normal[ray]) > THROUGH_SAME_WAY
    nearest = np.full(len(faces), np.inf)
    np.minimum.at(nearest, ray[aligned], distance[aligned])
    candidate = np.flatnonzero(np.isfinite(nearest) & (nearest >= floor))
    if not len(candidate):
        return depth
    # the body's own far wall first: another limb in front of the garment
    mine, wall, gap = Triangles(corners).met_many(start[candidate], -normal[candidate], nearest[candidate])
    own = wall != candidate[mine]
    blocked = np.zeros(len(candidate), dtype=bool)
    blocked[mine[own & (gap < nearest[candidate][mine])]] = True
    keep = candidate[~blocked]
    depth[keep] = nearest[keep]
    return depth


#: the stored whole-outfit lists were first made with the two half-thickness rays alone
RULE_HALF = "bidirectional garment rays bounded by local body half thickness"
#: … then, for a day, with the inward ray going on to the body's far wall
RULE_FAR_WALL = ("bidirectional garment rays bounded by local body half thickness; "
                 "inward to the far body wall for garment faces turned the way the body face is")
#: … and this adds a garment sunk into the body: its near wall lying deeper
#: under the face than that, whatever folds of the body lie between
RULE_NEAR_WALL = ("bidirectional garment rays bounded by local body half thickness; "
                  "inward through whole folds of the body to a garment face turned the way "
                  "the body face is")


def coverage(body_positions: np.ndarray, body_indices: np.ndarray,
             garment_corners: np.ndarray, garment_owner: np.ndarray,
             near_wall: bool = True) -> list:
    """Per body face: the owners (pieces) that cover it; an empty set for a bare face.

    *garment_corners* ``(M, 3, 3)`` are the outfit's triangles and
    *garment_owner* ``(M,)`` whatever identifies the piece each belongs to.
    The rule is the one at the head of this module.  With *near_wall* the
    inward ray also looks for a garment sunk into the body
    (:data:`RULE_NEAR_WALL`): where the body is fuller than what is worn — an
    ankle in tight hose, a fold of skin against close shorts — the garment
    lies under the face deeper than half the body's thickness there, and the
    body stands out through it.  "Thickness there" is no help in a fold, where
    the ray meets the fold's own other side a millimetre in.  So the garment
    face counts when the stretch of the ray up to it is *inside the body*: it
    has crossed the body's surface an even number of times (out of a fold and
    into the next is two), within a typical thickness of this body (the median
    over its faces).  Only a garment face turned the way the body face is
    counts; one turned the other way is the far side's, which covers the far
    side.
    """
    half, whole = coverage_by_rule(body_positions, body_indices, garment_corners, garment_owner)
    return whole if near_wall else half


def coverage_by_rule(body_positions: np.ndarray, body_indices: np.ndarray,
                     garment_corners: np.ndarray, garment_owner: np.ndarray) -> tuple[list, list]:
    """:func:`coverage` under both rules in one pass: ``(half-thickness rays alone, with the near wall)``."""
    positions = np.asarray(body_positions, dtype=np.float64)
    faces = np.asarray(body_indices, dtype=np.int64).reshape(-1, 3)
    corners = positions[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = np.linalg.norm(normal, axis=1)
    flat = length <= 0.0
    normal = normal / np.where(flat, 1.0, length)[:, None]
    centre = corners.mean(axis=1)
    half: list = [frozenset()] * len(faces)
    whole: list = [frozenset()] * len(faces)
    garment_owner = np.asarray(garment_owner)
    if not len(faces) or not len(garment_owner):
        return half, whole
    extent = max(float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))), 1e-6)
    epsilon = extent * 1e-6
    body, garment = Triangles(corners), Triangles(garment_corners)
    worn = garment.corners
    turned = np.cross(worn[:, 1] - worn[:, 0], worn[:, 2] - worn[:, 0])
    stride = max(body.radius * 4.0, extent * 0.02)
    thick = np.full(len(faces), np.inf)
    solid = np.flatnonzero(~flat)
    thick[solid] = body.first_many(centre[solid] - normal[solid] * epsilon, -normal[solid], extent, stride)
    measured = np.isfinite(thick)
    if not measured.any():
        return half, whole
    reach = float(np.median(thick[measured]))
    # a face with no wall of the body behind it — the far side is the character's own skin, cut
    # away from the lent body — is as thick as the body is in the middle: dropped, Wulfa's lent
    # chest stayed drawn 17 mm in front of the top she wears over it (§7.1065)
    thick[solid[~measured[solid]]] = reach
    rows = np.flatnonzero(np.isfinite(thick))
    out, inward = normal[rows], -normal[rows]
    near: list = [set() for _ in rows]
    # the two half-thickness rays, out of the face and into the body
    for direction in (out, inward):
        ray, found, _distance = garment.met_many(centre[rows] + direction * epsilon, direction, thick[rows] * 0.5)
        for k, owner in zip(ray.tolist(), garment_owner[found].tolist()):
            near[k].add(owner)
    # the inward ray on to a garment sunk into the body: whole folds of the body between
    start = centre[rows] + inward * epsilon
    ray, found, distance = garment.met_many(start, inward, np.maximum(thick[rows], reach))
    same = np.einsum("pk,pk->p", turned[found], normal[rows][ray]) > 0.0
    ray, found, distance = ray[same], found[same], distance[same]
    sunk: list = [set() for _ in rows]
    if len(ray):
        wall_ray, _wall, crossed = body.met_many(start, inward, reach)
        # how many of its ray's crossings of the body lie before each garment face
        span = 2.0 * max(reach, float(distance.max(initial=0.0)), 1.0) + 1.0
        walls = np.sort(wall_ray * span + crossed)
        folds = (np.searchsorted(walls, ray * span + distance, side="left")
                 - np.searchsorted(walls, ray * span, side="left"))
        keep = (distance > thick[rows][ray] * 0.5) & (folds % 2 == 0) & ((folds == 0) | (distance <= reach))
        for k, owner in zip(ray[keep].tolist(), garment_owner[found[keep]].tolist()):
            sunk[k].add(owner)
    for k, face in enumerate(rows.tolist()):
        if near[k]:
            half[face] = frozenset(near[k])
        if near[k] or sunk[k]:
            whole[face] = frozenset(near[k] | sunk[k])
    return half, whole


#: … and, for the whole outfit, what is left in patches the outfit closes in
RULE_CLOSED_IN = RULE_NEAR_WALL + "; less the patches of the rest that the outfit closes in"
#: … and a hand or a foot more than half of which is left out or closed in goes whole (:func:`extremities`)
RULE_EXTREMITIES = RULE_CLOSED_IN + "; and the rest of a hand or foot more than half of which is left out or closed in"
#: the regions (``part_roles.REGIONS``) :func:`extremities` treats whole
EXTREMITIES = ("LeftHand", "RightHand", "LeftFoot", "RightFoot")


def extremities(left_out: np.ndarray, regions, body_positions=None, body_indices=None,
                garment_corners=None, reach: float | None = None, lent=()) -> np.ndarray:
    """*left_out* ``(F,)`` with each hand or foot mostly left out or closed in, left out whole.

    A lent foot is fitted to the character's foot bones, which stand in her
    shoe; where it is longer or thicker than the shoe its toes come out
    through the sole (Texas: 154 and 161 points, up to 28.5 mm under her
    boots' soles, §7.1440), skin no garment is over or round.  A shoe that
    hides most of a foot stands for the foot, and the rest of it goes too;
    with the shoe off the foot is not mostly hidden, and all of it is back.
    Asked of the final mask, not of the rays alone: the inside of a roomy
    boot is hidden as a patch the boot closes in, not by a ray meeting it.
    Given the body and the *garment_corners* worn, a face left drawn that
    the garments close in (:func:`closed_in`, a majority of its directions
    within *reach*) counts toward the majority too: a lent hand a little
    larger than her own, which is part of the outfit, sits inside it with
    its fingers through (Texas's right hand: 562 of its 1508 faces left out,
    453 more closed in).  A hand or foot the item *lent* (its table's
    ``lent_regions``) is the lent body's to show, fingers out of a fingerless
    glove included, and is left as it is.  This asks the mask; a foot whose
    shoe is a piece of its own is also left out by the shoe that covers a
    fifth of it (``ActorController._shod_feet``, for a foot mostly drawn
    still).  *regions*: per face, its index in ``part_roles.REGIONS``
    (``part_roles.face_regions``).
    """
    from .part_roles import REGIONS
    out = np.array(left_out, dtype=bool)
    regions = np.asarray(regions)
    for name in EXTREMITIES:
        if name in lent:
            continue
        rows = regions == REGIONS.index(name)
        if not rows.any() or out[rows].all():
            continue
        hidden = int(out[rows].sum())
        drawn = np.flatnonzero(rows & ~out)
        if garment_corners is not None and len(garment_corners) and hidden * 2 <= rows.sum():
            hidden += int((closed_in(body_positions, body_indices, garment_corners, drawn, reach,
                                     majority=True) > 0.5).sum())
        if hidden * 2 > rows.sum():
            out[rows] = True
    return out


#: the directions a face looks out along: its normal, and two rings round it (degrees off, count)
_LOOKS = ((35.0, 6), (65.0, 6))


def _looks(normal: np.ndarray) -> np.ndarray:
    """``(13, 3)`` unit directions over the hemisphere a face looks out into."""
    aside = np.cross(normal, (0.0, 1.0, 0.0) if abs(normal[1]) < 0.9 else (1.0, 0.0, 0.0))
    aside = aside / np.linalg.norm(aside)
    other = np.cross(normal, aside)
    out = [normal]
    for ring, (tilt, count) in enumerate(_LOOKS):
        lean = np.radians(tilt)
        for step in range(count):
            turn = 2.0 * np.pi * (step + 0.5 * ring) / count
            out.append(np.cos(lean) * normal
                       + np.sin(lean) * (np.cos(turn) * aside + np.sin(turn) * other))
    return np.array(out)


def _face_frames(body_positions, body_indices) -> tuple:
    positions = np.asarray(body_positions, dtype=np.float64)
    faces = np.asarray(body_indices, dtype=np.int64).reshape(-1, 3)
    corners = positions[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = np.linalg.norm(normal, axis=1)
    normal = normal / np.where(length > 0.0, length, 1.0)[:, None]
    extent = max(float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))), 1e-6)
    return corners, normal, length, corners.mean(axis=1), extent


def thickness(body_positions: np.ndarray, body_indices: np.ndarray) -> float:
    """A typical thickness of a body: the median over its faces of the way through along the normal.

    The reach of :func:`closed_in`.  It is the body's alone, so one that asks
    again and again (a piece taken off, then another) measures it once.
    """
    corners, normal, length, centre, extent = _face_frames(body_positions, body_indices)
    body = Triangles(corners)
    stride = max(body.radius * 4.0, extent * 0.02)
    epsilon = extent * 1e-6
    solid = np.flatnonzero(length > 0.0)
    thick = body.first_many(centre[solid] - normal[solid] * epsilon, -normal[solid], extent, stride)
    return float(np.median(thick[np.isfinite(thick)])) if np.isfinite(thick).any() else 0.0


def closed_in(body_positions: np.ndarray, body_indices: np.ndarray, garment_corners: np.ndarray,
              rows: np.ndarray, reach: float | None = None, majority: bool = False) -> np.ndarray:
    """For body faces *rows*: the share of the directions each looks out along that the outfit closes.

    A direction is closed when it meets the outfit within a typical thickness
    of this body (:func:`thickness`; measured here unless given as *reach*).
    Skin an outfit leaves open looks out into the open in most directions; a
    sliver seen through a slit, or skin inside a boot, does not.  With
    *majority* a face is only asked whether more than half are closed (1.0)
    or not (0.0).
    """
    _corners, normal, length, centre, extent = _face_frames(body_positions, body_indices)
    epsilon = extent * 1e-6
    garment = Triangles(garment_corners)
    if reach is None:
        reach = thickness(body_positions, body_indices)
    rows = np.asarray(rows, dtype=np.int64)
    out = np.zeros(len(rows))
    solid = np.flatnonzero(length[rows] > 0.0)
    if not len(solid):
        return out
    looks = np.stack([_looks(normal[face]) for face in rows[solid]])            # (S, L, 3)
    per = looks.shape[1]
    starts = np.repeat(centre[rows[solid]] + normal[rows[solid]] * epsilon, per, axis=0)
    directions = looks.reshape(-1, 3)
    closed = np.zeros(len(starts), dtype=bool)
    # the rays a stretch at a time: each is answered on its own, and an outfit of fine chain (tens of
    # thousands of triangles) paired with every ray at once asked for gigabytes
    for begin in range(0, len(starts), RAY_CHUNK):
        ray, _found, _distance = garment.met_many(starts[begin:begin + RAY_CHUNK],
                                                  directions[begin:begin + RAY_CHUNK], reach)
        closed[begin + ray] = True
    share = closed.reshape(-1, per).mean(axis=1)
    out[solid] = (share > 0.5).astype(np.float64) if majority else share
    return out


def patches(positions: np.ndarray, indices: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """``(len(rows),)`` the connected patch each of faces *rows* is in, among those faces alone."""
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return np.zeros(0, dtype=np.int32)
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)[rows]
    _, welded = np.unique(np.ascontiguousarray(positions, dtype=np.float32), axis=0,
                          return_inverse=True)
    corners = welded.reshape(-1)[faces]
    count = int(corners.max()) + 1
    first = np.concatenate([corners[:, 0], corners[:, 1]])
    second = np.concatenate([corners[:, 1], corners[:, 2]])
    graph = coo_matrix((np.ones(len(first), dtype=np.int8), (first, second)), shape=(count, count))
    _, label = connected_components(graph, directed=False)
    return np.unique(label[corners[:, 0]], return_inverse=True)[1].astype(np.int32)


def closed_patches(body_positions, body_indices, garment_corners, covered: np.ndarray,
                   never: np.ndarray | None = None, reach: float | None = None,
                   bare: np.ndarray | None = None) -> np.ndarray:
    """``(F,)`` bool: the uncovered faces in patches the outfit closes in.

    Of the faces *covered* leaves drawn (less those *never* drawn anyway), a
    connected patch goes when more than half its faces are closed in in more
    than half the directions they look out along (:func:`closed_in`), or when
    it has no inside at all.  Whole patches, so no crumbs are left of one.

    *bare* marks skin a piece that is now off covered.  That is skin a
    garment taken off has uncovered, not a gap between garments, and the
    majority within a body's thickness is the wrong question for it: a belly
    between a top and shorts is closed in that way, and leaving it out was a
    hole through the body (§7.1032).  It is asked instead whether what is
    still worn closes **every** direction it looks out along, within the
    body's size (:func:`enclosed_patches`): hips under a long skirt are, a belly is not
    (§7.1033).  The gaps beside it are asked as before, each a patch of its
    own: kept with the bare skin they touch, the gaps under the shorts and the
    top came through them.
    """
    covered = np.asarray(covered, dtype=bool)
    left = ~covered if never is None else ~(covered | np.asarray(never, dtype=bool))
    out = np.zeros(len(covered), dtype=bool)
    shown_bare = np.zeros(len(covered), dtype=bool)
    if bare is not None:
        bare = np.asarray(bare, dtype=bool) & left
        left &= ~bare
        out |= enclosed_patches(body_positions, body_indices, garment_corners, np.flatnonzero(bare))
        shown_bare = bare & ~out
    rows = np.flatnonzero(left)
    if not len(rows):
        return out
    label = patches(body_positions, body_indices, rows)
    shut = closed_in(body_positions, body_indices, garment_corners, rows, reach, majority=True) > 0.5
    share = np.bincount(label, weights=shut) / np.bincount(label)
    # a patch with no face inside it — every face of it borders skin that is
    # covered — is a seam's width of skin showing through a slit, not an area;
    # one that touches bare skin left drawn is that skin's edge, not a slit
    open_skin = left | shown_bare
    within = (inner(open_skin, body_positions, body_indices) & left)[rows]
    has_inside = np.bincount(label, weights=within) > 0
    out[rows[(share[label] > 0.5) | ~has_inside[label]]] = True
    return out


def enclosed_patches(body_positions, body_indices, garment_corners, rows) -> np.ndarray:
    """``(F,)`` bool: of the faces *rows*, the patches whose faces a garment closes on every side.

    A face is enclosed when every direction it looks out along
    (:func:`closed_in`) meets the garment within the body's own size, not its
    thickness — a skirt hangs farther off the hips than a body is thick.  A connected patch of *rows*
    goes when more than half its faces are; whole patches, so no crumbs.
    Measured on Zhuangfy with the shorts off: 848 of the 976 faces they had
    covered are enclosed by the long skirt; with the main garment off instead,
    none of the 6310 it had covered is (§7.1033).
    """
    rows = np.asarray(rows, dtype=np.int64)
    out = np.zeros(len(np.asarray(body_indices).reshape(-1, 3)), dtype=bool)
    if not len(rows) or not len(garment_corners):
        return out
    positions = np.asarray(body_positions, dtype=np.float64)
    extent = max(float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))), 1e-6)
    whole = closed_in(positions, body_indices, garment_corners, rows, extent) >= 1.0
    label = patches(body_positions, body_indices, rows)
    share = np.bincount(label, weights=whole) / np.bincount(label)
    out[rows[share[label] > 0.5]] = True
    return out


def stands_outside(positions, indices, garment_corners, reach: float) -> np.ndarray:
    """``(F,)`` bool: the skin faces standing outside a garment that should be over them.

    For skin that is **drawn under a whole outfit** and is not a closed body
    (a reconstruction's dressed skin: a shell cut to what its scene left in
    view, so it has no thickness of its own to measure by).  Looking back
    into the body from a face, along the directions round its normal
    (:func:`_looks`), a garment face turned the way the skin face is, within
    *reach* (a typical thickness of the body), is a garment the skin has come
    through: it is worn over this skin, and the skin is in front of it.  A
    garment on the far side of the limb is turned the other way and does not
    count, and neither does one farther off than a body is thick.
    """
    _corners, normal, length, centre, extent = _face_frames(positions, indices)
    out = np.zeros(len(normal), dtype=bool)
    rows = np.flatnonzero(length > 0.0)
    worn = np.asarray(garment_corners, dtype=np.float64)
    if not len(rows) or not len(worn) or not reach > 0.0:
        return out
    turned = np.cross(worn[:, 1] - worn[:, 0], worn[:, 2] - worn[:, 0])
    looks = np.stack([_looks(normal[face]) for face in rows])
    per = looks.shape[1]
    starts = np.repeat(centre[rows] - normal[rows] * extent * 1e-6, per, axis=0)
    ray, found, _distance = Triangles(worn).met_many(starts, -looks.reshape(-1, 3), reach)
    same = np.einsum("pk,pk->p", turned[found], np.repeat(normal[rows], per, axis=0)[ray]) > 0.0
    out[rows[np.unique(ray[same] // per)]] = True
    return out


def in_front_of_worn(positions, indices, garment_corners, rows, reach: float) -> np.ndarray:
    """``(F,)`` bool: of the body's faces *rows*, those lying in front of a worn face.

    Straight in from a face's centre, a garment face turned the way the skin
    face is, met before the body's own far wall and within *reach* (a typical
    thickness of the body: an open shell has no far wall): the garment is worn
    over this skin and the skin is outside it.  The far wall is one of *rows*,
    the faces that are drawn: a lent body passes through itself at the hips,
    and a sheet of it left out 3 mm under the skin hid the shorts 20 mm in
    from the ray, not from the eye.  So it is asked again of what is still
    drawn until nothing more goes: a face whose wall was a face that went
    lies in front of the garment too.  Leaving such a face out never opens a
    hole — the garment is right behind it.  The rim :func:`inner` keeps, a
    gap no ray covered, skin a piece now off bared: under a garment as tight
    as a shell (shorts painted on the skin), the lent body is fuller and each
    of them stood up to 30 mm through it (§7.1032).  The same test as
    ``tools/ef_body_poke.py``, without its millimetre floor.
    """
    corners, normal, length, centre, extent = _face_frames(positions, indices)
    out = np.zeros(len(normal), dtype=bool)
    rows = np.asarray(rows, dtype=np.int64)
    worn = np.asarray(garment_corners, dtype=np.float64)
    if not len(worn) or not reach > 0.0:
        return out
    garment = Triangles(worn)
    turned = np.cross(worn[:, 1] - worn[:, 0], worn[:, 2] - worn[:, 0])
    drawn = rows
    while True:
        asked = drawn[length[drawn] > 0.0]
        if not len(asked):
            return out
        start = centre[asked] - normal[asked] * extent * 1e-6
        inward = -normal[asked]
        body = Triangles(corners[drawn])
        wall = body.first_many(start, inward, reach, max(body.radius * 4.0, extent * 0.02))
        ray, found, _distance = garment.met_many(start, inward, np.minimum(wall, reach))
        same = np.einsum("pk,pk->p", turned[found], normal[asked][ray]) > 0.0
        gone = asked[np.unique(ray[same])]
        if not len(gone):
            return out
        out[gone] = True
        drawn = drawn[~out[drawn]]


def dressed_left_out(positions, indices, garment_corners, reach: float) -> np.ndarray:
    """``(F,)`` bool: the faces of a dressed skin not to draw under its whole outfit.

    What stands outside a garment (:func:`stands_outside`), and what is then
    left in patches the outfit closes in or that have no inside
    (:func:`closed_patches`) — the crumbs between the faces taken.  Skin in a
    real opening of the outfit looks out into the open and stays.
    """
    outside = stands_outside(positions, indices, garment_corners, reach)
    return outside | closed_patches(positions, indices, garment_corners, outside, None, reach)


def nearest(origin: np.ndarray, direction: np.ndarray, corners: np.ndarray) -> tuple[int, float]:
    """``(triangle, distance)`` of the first of *corners*' triangles a ray meets, ``(-1, inf)`` if none."""
    if not len(corners):
        return -1, float("inf")
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    reach = _hits(np.asarray(origin, dtype=np.float64)[None], direction[None],
                  np.asarray(corners, dtype=np.float64))[0]
    best = int(np.argmin(reach))
    return (best, float(reach[best])) if np.isfinite(reach[best]) else (-1, float("inf"))


@dataclass
class Outfit:
    """The pieces of an item's outfit parts, and what is off.

    ``parts`` are the model's parts that are worn, by index in the model;
    ``labels[i]`` the piece of each triangle of ``parts[i]`` as first loaded.
    A piece is ``(part index, island)``.
    """
    parts: dict                       # part index → Part
    meshes: dict = field(default_factory=dict)    # part index → the part's whole mesh
    labels: dict = field(default_factory=dict)    # part index → (T,) island per triangle
    off: set = field(default_factory=set)         # {(part index, island)}

    @classmethod
    def of(cls, parts: dict) -> Outfit:
        out = cls(dict(parts))
        for index, part in parts.items():
            out.meshes[index] = part.mesh
            out.labels[index] = islands(part.mesh.positions, part.mesh.indices)
        return out

    def pieces(self) -> list:
        """``(part index, island, triangles, shown)`` of every piece, parts in model order."""
        out = []
        for index in sorted(self.parts):
            counts = np.bincount(self.labels[index])
            out += [(index, island, int(count), (index, island) not in self.off)
                    for island, count in enumerate(counts)]
        return out

    def shown_owners(self) -> set:
        return {(index, island) for index, island, _count, shown in self.pieces() if shown}

    def apply(self) -> None:
        """Give every worn part the mesh of its pieces that are on (a copy; the whole one is kept)."""
        for index, part in self.parts.items():
            hidden = [island for owner, island in self.off if owner == index]
            whole = self.meshes[index]
            if not hidden:
                part.mesh = whole
                continue
            keep = ~np.isin(self.labels[index], hidden)
            mesh = whole.copy()
            mesh.indices = whole.indices.reshape(-1, 3)[keep].ravel().copy()
            part.mesh = mesh

    def corners(self) -> tuple[np.ndarray, np.ndarray]:
        """``((M, 3, 3) triangles, (M, 2) owner of each)`` of the whole outfit at rest."""
        triangles, owners = [], []
        for index in sorted(self.parts):
            mesh = self.meshes[index]
            faces = mesh.indices.reshape(-1, 3).astype(np.int64)
            triangles.append(mesh.positions.astype(np.float64)[faces])
            owners.append(np.stack([np.full(len(faces), index), self.labels[index]], axis=1))
        if not triangles:
            return np.zeros((0, 3, 3)), np.zeros((0, 2), dtype=np.int64)
        return np.concatenate(triangles), np.concatenate(owners)


#: What a piece of an outfit is, by what it does to the body under it (:func:`kinds`).
GARMENT, TRIM, ACCESSORY = "garment", "trim", "accessory"


def kinds(outfit: Outfit, coverers: list, body: np.ndarray) -> dict:
    """``{(part index, island): GARMENT | TRIM | ACCESSORY}`` for every piece of *outfit*.

    The game's data does not say which piece is a garment and which an
    accessory; where the piece is does.  A piece that covers a face of the
    body (*coverers*, :func:`face_coverers`) clothes it: a **garment**.  A
    piece that covers none goes with whatever it is closest to: nearer to a
    garment than to the body (*body*: the vertices of the character itself —
    the surfaces the cover is measured on, the face and the hair, and only
    the vertices those parts draw), it is that garment's **trim** — a button,
    a strip, something hanging off a hem — and comes off with it, or it would
    be left in the air; trim on trim counts.  Nearer to the body, it is worn
    on its own: an **accessory**.

    Both distances are between vertices, not surfaces: on meshes as fine as
    these (edges around a centimetre) that is the same answer, on a sheet of
    two triangles it is not.
    """
    from scipy.spatial import cKDTree
    covering = {tuple(owner) for owned in coverers for owner in owned}
    points, out = {}, {}
    for index in sorted(outfit.parts):
        mesh = outfit.meshes[index]
        faces = mesh.indices.reshape(-1, 3).astype(np.int64)
        position = mesh.positions.astype(np.float64)
        labels = outfit.labels[index]
        for island in range(int(labels.max()) + 1 if len(labels) else 0):
            key = (index, island)
            points[key] = position[np.unique(faces[labels == island])]
            out[key] = GARMENT if key in covering else ACCESSORY
    from_body = {}
    if len(body):
        skin = cKDTree(np.asarray(body, dtype=np.float64))
        from_body = {key: float(skin.query(points[key])[0].min()) for key, kind in out.items()
                     if kind == ACCESSORY}
    while True:
        worn = [points[key] for key, kind in out.items() if kind != ACCESSORY]
        if not worn:
            return out
        tree = cKDTree(np.concatenate(worn))
        joined = [key for key, kind in out.items() if kind == ACCESSORY
                  and float(tree.query(points[key])[0].min()) < from_body.get(key, np.inf)]
        if not joined:
            return out
        for key in joined:
            out[key] = TRIM


#: How close a garment lies on another for the one to be worn on the other (:func:`worn_on`).  Measured
#: on Typhoea and Wulfa (§7.1085): straps, laces and belts on the garment under them 2.4-9.7 mm (median
#: over the piece), harness pieces on a bodysuit 12-15 mm; a skirt's frills over a bodysuit 45-51 mm,
#: a cape 23-43 mm.
ON_GARMENT = 0.02


def worn_on(outfit: Outfit, kinds: dict, coverers: list = (), body: np.ndarray | None = None) -> dict:
    """``{piece: the garment it sits on}`` for the pieces that come off with another garment.

    A trim (:func:`kinds`) sits on the garment nearest it, and that is often
    another part's: Typhoea's thigh straps and ankle laces are pieces of
    three other parts on her stockings (``cloth_01``), and taking the
    stockings off left them in the air (§7.1085).  Some of those straps
    cover a few faces of the body of their own and are garments; a garment
    is worn on another part's garment (all of that part's garment pieces
    together: a stocking is a dozen) when all the skin it covers that covers
    too (and more), it lies on it (median under :data:`ON_GARMENT`), and
    nearer to it than to the body (*body*: :func:`kinds`'s).  It goes with
    the piece of that garment nearest it.  What a piece sits
    on is followed down to a garment that sits on nothing.  Nearest by
    vertices, as :func:`kinds` measures.
    """
    from scipy.spatial import cKDTree
    garments = [key for key, kind in kinds.items() if kind == GARMENT]
    if not garments:
        return {}
    points = {}
    for key in kinds:
        mesh = outfit.meshes[key[0]]
        faces = mesh.indices.reshape(-1, 3).astype(np.int64)
        points[key] = mesh.positions.astype(np.float64)[np.unique(faces[outfit.labels[key[0]] == key[1]])]
    trees: dict = {}

    def tree(key):
        if key not in trees:
            trees[key] = cKDTree(points[key])
        return trees[key]

    covers: dict = {}
    for face, owned in enumerate(coverers):
        for owner in owned:
            covers.setdefault(tuple(owner), set()).add(face)
    skin = cKDTree(np.asarray(body, dtype=np.float64)) if body is not None and len(body) else None
    # a garment is many pieces (a stocking is a dozen), and what lies on it lies across several
    by_part: dict = {}
    for key in garments:
        by_part.setdefault(key[0], []).append(key)
    sits: dict = {}
    for piece in garments:
        mine = covers.get(piece, set())
        if not mine or skin is None:
            continue
        to_body = float(np.median(skin.query(points[piece])[0]))
        best = None
        for part, keys in by_part.items():
            if part == piece[0]:
                continue                # its own part: listed and taken off with it already
            others = keys
            under = set().union(*(covers.get(key, set()) for key in others)) if others else set()
            if len(under) <= len(mine) or not mine <= under:
                continue
            apart = {key: tree(key).query(points[piece])[0] for key in others}
            gap = float(np.median(np.min(np.stack(list(apart.values())), axis=0)))
            if gap < min(ON_GARMENT, to_body) and (best is None or gap < best[0]):
                best = (gap, min(apart, key=lambda key: float(apart[key].min())))
        if best is not None:
            sits[piece] = best[1]
    trims = [key for key, kind in kinds.items() if kind == TRIM]
    if trims:
        everything = cKDTree(np.concatenate([points[key] for key in garments]))
        owner = np.concatenate([np.full(len(points[key]), number) for number, key in enumerate(garments)])
        for key in trims:
            distance, nearest = everything.query(points[key])
            sits[key] = garments[int(owner[nearest[int(np.argmin(distance))]])]
    out = {}
    for piece in sits:
        seen, at = {piece}, sits[piece]
        while at in sits and sits[at] not in seen:      # down to what sits on nothing
            seen.add(at)
            at = sits[at]
        out[piece] = at
    return out


#: What a garment is called (:func:`garments`), by what of the body it covers.
GLOVES, SHOES, SOCKS, STOCKINGS, SLEEVES = "gloves", "shoes", "socks", "stockings", "sleeves"
PANTS, SHORTS, UNDERWEAR, SKIRT = "pants", "shorts", "underwear", "skirt"
TOP, INNER_TOP, BODYSUIT, OTHER = "top", "inner top", "bodysuit", "other"
#: The regions (``part_roles.REGIONS``) a name is read off.
_TORSO = {"Neck", "UpperChest", "Chest", "Spine", "LeftShoulder", "RightShoulder"}
_LEGS = {"LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg", "LeftFoot", "RightFoot"}
_THIGHS = {"LeftUpperLeg", "RightUpperLeg"}
_ARMS = {"LeftUpperArm", "LeftLowerArm", "RightUpperArm", "RightLowerArm", "LeftHand", "RightHand"}
#: Pieces whose vertices come this close (m) are one garment: a garment's islands part at seams
#: and share the seam's positions there.
SEAM = 0.001
#: Most of what a garment covers, and a fair share of it (naming only: the regions of every
#: garment of the six women in Free H fall well to one side, §7.1104).
MOST, SHARE = 0.5, 0.25
#: How far off the body (median, m) a garment over the hips hangs to be a skirt: shorts and
#: underwear lie 6-25 mm off, skirts 63-74 mm (§7.1104).
LOOSE = 0.04


def garment_name(regions: list, inner: bool, off_body: float) -> str:
    """The name of a garment that covers the body faces of *regions* (a region name per face).

    *inner*: most of it is covered by a garment farther out; *off_body*: its
    median distance from the body (m).
    """
    if not regions:
        return OTHER
    count = len(regions)

    def share(names: set) -> float:
        return sum(region in names for region in regions) / count
    hips, torso, legs, thighs, arms = (share({"Hips"}), share(_TORSO), share(_LEGS), share(_THIGHS),
                                       share(_ARMS))
    if share({"LeftHand", "RightHand"}) >= MOST:
        return GLOVES
    if share({"LeftFoot", "RightFoot"}) >= MOST:
        return SHOES
    if legs >= MOST and hips < SHARE:
        return STOCKINGS if thighs >= SHARE else SOCKS
    if hips + thighs >= MOST and hips >= SHARE:
        if off_body >= LOOSE:
            return SKIRT
        if legs >= SHARE:
            return PANTS
        return UNDERWEAR if inner else SHORTS
    if torso >= MOST and hips < SHARE:
        return INNER_TOP if inner else TOP
    if torso >= SHARE and hips >= SHARE:
        return BODYSUIT
    if arms >= MOST:
        return SLEEVES
    return TOP if torso >= MOST else OTHER


def garments(outfit: Outfit, kinds: dict, garment_of: dict, coverers: list, regions: list,
             body: np.ndarray) -> dict:
    """``{(part index, name): {pieces}}``: each worn part split into the garments it is made of.

    A part is often more than one garment: Si's dress carries shorts close
    over her hips, Typhoea's bodysuit both gloves' cuffs, and taking the part
    off was the only switch (owner, 2026-10-16).  Its garment pieces
    (*kinds*) that meet at a seam (:data:`SEAM`) are one garment, with what
    is worn on them (*garment_of*, ``worn_on``).  A garment that hides no
    skin of its own and lies on one it overlaps (as :func:`worn_on` reads
    "lies on") is a decoration on that one and goes with it.  Each is named by the body
    regions it covers (*coverers*, *regions*: a region name per body face;
    :func:`garment_name`), and a part's garments of one name are one entry:
    a pair of gloves is one switch.  *body*: the vertices :func:`kinds`
    measures from.
    """
    from scipy.spatial import cKDTree
    roots = sorted(key for key, kind in kinds.items() if kind == GARMENT and garment_of.get(key, key) == key)
    if not roots:
        return {}
    points = {}
    for key in roots:
        mesh = outfit.meshes[key[0]]
        faces = mesh.indices.reshape(-1, 3).astype(np.int64)
        points[key] = mesh.positions.astype(np.float64)[np.unique(faces[outfit.labels[key[0]] == key[1]])]
    parent = {key: key for key in roots}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    trees = {key: cKDTree(points[key]) for key in roots}
    for a in roots:
        for b in roots:
            if a < b and a[0] == b[0] and find(a) != find(b)                     and np.isfinite(trees[a].query(points[b], distance_upper_bound=SEAM)[0]).any():
                parent[find(a)] = find(b)
    covers: dict = {}
    for face, owned in enumerate(coverers):
        for owner in owned:
            covers.setdefault(tuple(owner), set()).add(face)
    skin = cKDTree(np.asarray(body, dtype=np.float64)) if len(body) else None
    groups: dict = {}
    for key in roots:
        groups.setdefault(find(key), []).append(key)
    faces_of = {root: set().union(*(covers.get(key, set()) for key in members)) for root, members in groups.items()}
    off_body = {root: float(np.median(skin.query(np.concatenate([points[key] for key in members]))[0]))
                if skin is not None else 0.0 for root, members in groups.items()}
    every = {root: np.concatenate([points[key] for key in members]) for root, members in groups.items()}

    def lies_on(root, other) -> bool:
        # as :func:`worn_on` has it: on the other (median under ON_GARMENT) and nearer it than the body;
        # a skirt over shorts hangs off them and is a garment of its own
        gap = float(np.median(cKDTree(every[other]).query(every[root])[0]))
        return gap < min(ON_GARMENT, off_body[root])

    # a decoration goes with the garment under it that shares the most of its faces
    into = {}
    for root in sorted(groups, key=lambda root: len(faces_of[root])):
        others = [other for other in groups if other != root and other not in into]
        alone = faces_of[root] - set().union(*(faces_of[other] for other in others)) if others else faces_of[root]
        under = [other for other in others if off_body[other] < off_body[root] and faces_of[root] & faces_of[other]
                 and lies_on(root, other)]
        if faces_of[root] and not alone and under:
            into[root] = max(under, key=lambda other: len(faces_of[root] & faces_of[other]))
    for root in into:
        while into.get(root) in into:
            into[root] = into[into[root]]
    named: dict = {}
    for root, members in groups.items():
        if root in into:
            continue
        mine = faces_of[root]
        outer = set().union(*(faces_of[other] for other in groups
                              if other != root and other not in into and off_body[other] > off_body[root]))
        inner = bool(mine) and len(mine & outer) >= MOST * len(mine)
        name = garment_name([regions[face] for face in mine], inner, off_body[root])
        taken = {key for key in members}
        taken |= {key for other, target in into.items() if target == root for key in groups[other]}
        taken |= {piece for piece, sits in garment_of.items() if sits in taken}
        named.setdefault((root[0], name), set()).update(taken)
    return named


def covered_now(coverers: list, shown: set) -> np.ndarray:
    """``(F,)`` bool: the body faces some piece that is on covers."""
    return np.array([any(owner in shown for owner in owned) for owned in coverers], dtype=bool)


def hiding(outfit: Outfit, shown: set) -> set:
    """The pieces of *shown* that hide the skin under them: not those of a see-through part.

    A part drawn blended (``Material.blend``: the game's own ``_SrcBlend``／
    ``_DstBlend`` or transparent surface type) shows what is under it.
    Zhuangfy's top is one; the skin under it was left out as covered and
    the inside of the body showed through the fabric (§7.1033).  Skin under
    it is still not bare (:func:`bared_now` asks what is worn, not this).
    """
    return {piece for piece in shown if not getattr(outfit.parts[piece[0]].material, "blend", False)}


def bared_now(coverers: list, shown: set) -> np.ndarray:
    """``(F,)`` bool: the body faces some piece covers and none that is on — skin a garment taken off uncovered."""
    return np.array([bool(owned) and not any(owner in shown for owner in owned) for owned in coverers],
                    dtype=bool)


#: How many faces deep a covered patch is behind its rim for the rim to go with it (:func:`inner`).
#: Two would take the rim of a strap three faces wide, and leave a hole of skin beside it.
WIDE = 3


def inner(covered: np.ndarray, positions: np.ndarray, indices: np.ndarray,
          beyond: np.ndarray | None = None) -> np.ndarray:
    """``(F,)`` bool: the covered faces every face around them is covered too.

    A face is called covered when a ray from its centre meets a piece — which
    a strap an inch wide does, without hiding the face.  Under a whole outfit
    that never shows: the faces round it are covered as well.  Under part of
    one it is a hole in the skin beside the strap.  So only a face whose every
    corner is shared with covered faces alone is left out; the rim of each
    covered patch stays drawn, under the piece that covers it.  *beyond*
    marks faces that are never drawn anyway, which count as covered here.
    Corners at the same stored place are one corner.

    That rim is for the strap.  Round a patch that is wide — a trouser leg,
    a sleeve — it is skin drawn under cloth for no reason, and where the body
    is fuller than the garment (an ankle in a narrow hem) it comes through
    it.  So the rim is left out as well wherever the patch behind it is at
    least :data:`WIDE` faces deep: the patch is worn back that many times,
    and what is left of it (its core) is grown back as far, within the patch.
    """
    covered = np.asarray(covered, dtype=bool)
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    if not covered.any():
        return covered.copy()
    _, welded = np.unique(np.ascontiguousarray(positions, dtype=np.float32), axis=0,
                          return_inverse=True)
    corners = welded.reshape(-1)[faces]
    open_face = ~covered if beyond is None else ~(covered | np.asarray(beyond, dtype=bool))
    count = int(corners.max()) + 1

    def worn_back(kept: np.ndarray, outside: np.ndarray) -> np.ndarray:
        """*kept* less its faces that share a corner with a face *outside*."""
        touched = np.zeros(count, dtype=bool)
        touched[corners[outside].ravel()] = True
        return kept & ~touched[corners].any(axis=1)
    within = worn_back(covered, open_face)
    core = within
    for _ in range(WIDE - 1):
        core = worn_back(core, ~core & ~(np.zeros(len(core), dtype=bool) if beyond is None
                                         else np.asarray(beyond, dtype=bool)))
    # grown back as far as it was worn: a patch with a core comes back whole, rim and all, and a
    # patch without one (a strap) does not come back at all
    back = core
    for _ in range(WIDE):
        touched = np.zeros(count, dtype=bool)
        touched[corners[back].ravel()] = True
        back = covered & touched[corners].any(axis=1)
    return within | back


def face_coverers(body_positions, body_indices, outfit: Outfit, near_wall: bool = True) -> list:
    """:func:`coverage` of a body by *outfit*, owners as ``(part index, island)`` tuples."""
    corners, owners = outfit.corners()
    codes = owners[:, 0] * (1 << 20) + owners[:, 1] if len(owners) else np.zeros(0, dtype=np.int64)
    found = coverage(body_positions, body_indices, corners, codes, near_wall)
    return [frozenset((int(code) >> 20, int(code) & ((1 << 20) - 1)) for code in owned)
            for owned in found]


def _digest(body_positions, body_indices, outfit: Outfit) -> str:
    """What the measurement depends on: the body's triangles and the outfit's."""
    import hashlib
    digest = hashlib.sha256(b"pieces-2: " + RULE_NEAR_WALL.encode())
    for array in (body_positions, body_indices):
        digest.update(np.ascontiguousarray(array).tobytes())
    for index in sorted(outfit.parts):
        mesh = outfit.meshes[index]
        digest.update(np.ascontiguousarray(mesh.positions).tobytes())
        digest.update(np.ascontiguousarray(mesh.indices).tobytes())
    return digest.hexdigest()


#: The rule the masks under a part-worn outfit are made by (``controller._cover_under``: the pieces
#: that are on, the patches they close in, less the rim).  Part of the name the masks are kept
#: under: change the rule and this with it, or masks made by the old one are read back.
COVER_RULE = "cover-10"


def cached_thickness(positions, indices) -> float:
    """:func:`thickness`, kept on this machine by the surface it was measured on (seconds otherwise, every session)."""
    import hashlib
    from ..core.paths import derived_root
    digest = hashlib.sha256(b"thickness-1: ")
    for array in (positions, indices):
        digest.update(np.ascontiguousarray(array).tobytes())
    path = derived_root() / "assets_cache" / "endfield_pieces" / f"{digest.hexdigest()}.thickness.npy"
    if path.is_file():
        return float(np.load(path, allow_pickle=False))
    found = thickness(positions, indices)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.float64(found))
    return found


def cover_digest(body_positions, body_indices, outfit: Outfit) -> str:
    """The name a body and an outfit's cover measurements are kept under (:func:`cached_coverers`)."""
    return _digest(body_positions, body_indices, outfit)


def _masks_path(digest: str, off):
    import hashlib
    from ..core.paths import derived_root
    which = hashlib.sha256((COVER_RULE + repr(sorted((int(i), int(k)) for i, k in off))).encode()).hexdigest()[:24]
    return derived_root() / "assets_cache" / "endfield_pieces" / f"{digest}.cover" / f"{which}.npz"


def stored_masks(digest: str, off, counts: list) -> list | None:
    """The masks kept for the pieces *off* (one ``(F,)`` bool per body surface, *counts* faces each), or None.

    The set of pieces off is in the file as well as in its name; a file that
    does not say the same set, or whose masks are not these surfaces' sizes,
    is not read.
    """
    path = _masks_path(digest, off)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        same = np.array_equal(data["off"], np.array(sorted(off), dtype=np.int32).reshape(-1, 2))
        if not same or data["counts"].tolist() != [int(n) for n in counts]:
            return None
        bits = np.unpackbits(data["bits"])[:int(sum(counts))].astype(bool)
    return [part.copy() for part in np.split(bits, np.cumsum(counts)[:-1])]


def store_masks(digest: str, off, masks: list) -> None:
    """Keep *masks* (one ``(F,)`` bool per body surface) for the pieces *off* on this machine."""
    path = _masks_path(digest, off)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, off=np.array(sorted(off), dtype=np.int32).reshape(-1, 2),
                        counts=np.array([len(mask) for mask in masks], dtype=np.int64),
                        bits=np.packbits(np.concatenate(masks) if masks else np.zeros(0, dtype=bool)))


def cached_coverers(body_positions, body_indices, outfit: Outfit) -> list:
    """:func:`face_coverers`, kept on this machine by what it was measured from.

    The file is under the derived data's cache (``core.paths.derived_root``),
    named by a digest of the body's and the outfit's triangles, so a converted
    item that changes is measured again and nothing stale is ever read.
    """
    from ..core.paths import derived_root
    folder = derived_root() / "assets_cache" / "endfield_pieces"
    path = folder / f"{_digest(body_positions, body_indices, outfit)}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            start, owners = data["start"], data["owners"]
        return [frozenset(map(tuple, owners[a:b].tolist())) for a, b in itertools.pairwise(start)]
    found = face_coverers(body_positions, body_indices, outfit)
    start = np.cumsum([0] + [len(owned) for owned in found])
    owners = np.array([owner for owned in found for owner in sorted(owned)],
                      dtype=np.int32).reshape(-1, 2)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, start=start, owners=owners)
    return found
