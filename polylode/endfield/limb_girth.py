"""How thick a converted character's limbs are next to the lent body's: measured on her own skin.

The body fit (``body.fit_body``) stretches each lent limb segment along its
axis to the character's segment length, the way it always stretched a foot
along it; what it sizes the limb's cross-section by is this.  A segment's
length says nothing about its thickness — sized by it, a short forearm came
out a third thinner than the character's own forearm skin around it (Wulfa:
length 0.677 of the lent body's, her skin 0.91; §7.1281, §7.1283) and
the hand, sized by its own length, stood on it 1.33 times too big.

So the cross-section is the character's own: wherever her own skin (every
part drawn with the skin program, the garments painted into it included —
they lie on the skin) closes round a limb segment, its section is cut at
:data:`STATIONS` and its perimeter set against the lent body's at the same
fraction of the same segment.  One ratio per limb (:data:`LIMBS`), the median
of its rings on both sides: the segments of one limb then keep the lent
body's proportions to each other, and the hand its proportion to the
forearm.  A limb her skin does not go round keeps the lent body's
proportion to the trunk (``FitScales.girth``: the torso's ratio).
"""
from __future__ import annotations

import numpy as np

#: The program the character's own skin is drawn with (``part_roles``).
SKIN_PROGRAM = "HGRP/CharacterNPR_Skin"
#: ``limb: ((segment, source first joint, source last joint, human bone, human bone), …)``: the segments
#: whose rings size a limb, as ``body.FitScales.SPANS`` names them.
LIMBS = {
    "arm": (("upper arm", "cf_s_arm01", "cf_s_forearm01", "UpperArm", "LowerArm"),
            ("forearm", "cf_s_forearm01", "cf_s_hand", "LowerArm", "Hand")),
    "leg": (("thigh", "cf_s_thigh01", "cf_s_leg01", "UpperLeg", "LowerLeg"),
            ("shin", "cf_s_leg01", "cf_j_foot", "LowerLeg", "Foot")),
}
#: The human bone a limb's skin hangs under.
ROOT = {"arm": "UpperArm", "leg": "UpperLeg"}
#: Where along a segment (fraction from its first joint) a ring is cut: clear of both joints, whose
#: bends and helper bulges are no measure of the limb's girth.
STATIONS = (0.2, 0.35, 0.5, 0.65, 0.8)
#: A cut closes round the limb when its points fall in every one of this many sectors round the axis.
SECTORS = 12
#: A vertex is a limb's when at least this share of its weight is on the limb's bones.
MEMBER = 0.5
#: Cut points farther than this from the segment's axis are not the limb's.
REACH = 0.15


def ring(positions: np.ndarray, triangles: np.ndarray, member: np.ndarray, centre: np.ndarray,
         axis: np.ndarray) -> tuple[float | None, bool]:
    """``(perimeter, closed)`` of the section of the *member* triangles by the plane at *centre* across *axis*.

    The perimeter is the convex hull's of the cut (metres); ``closed`` is
    whether the cut goes all the way round the axis (:data:`SECTORS`).
    ``(None, False)`` when the plane cuts too little of it.
    """
    from scipy.spatial import ConvexHull
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    across = np.cross(axis, [0.0, 0.0, 1.0])
    if np.linalg.norm(across) < 1e-3:
        across = np.cross(axis, [1.0, 0.0, 0.0])
    across /= np.linalg.norm(across)
    other = np.cross(axis, across)
    triangles = triangles[member[triangles].all(axis=1)]
    height = (positions - centre) @ axis
    # a vertex on the plane counts as below it, so a plane through a ring of vertices still cuts
    corner = height[triangles]
    triangles = triangles[(corner.min(axis=1) <= 0) & (corner.max(axis=1) > 0)]
    points = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        first, second = triangles[:, a], triangles[:, b]
        ha, hb = height[first], height[second]
        cut = (ha <= 0) != (hb <= 0)
        t = ha[cut] / (ha[cut] - hb[cut])
        points.append(positions[first[cut]] + t[:, None] * (positions[second[cut]] - positions[first[cut]]))
    cut = np.concatenate(points) - centre if points else np.zeros((0, 3))
    flat = np.c_[cut @ across, cut @ other]
    flat = flat[np.linalg.norm(flat, axis=1) < REACH]
    if len(flat) < 6:
        return None, False
    sector = np.floor((np.arctan2(flat[:, 1], flat[:, 0]) + np.pi) / (2 * np.pi / SECTORS)).astype(int) % SECTORS
    hull = flat[ConvexHull(flat).vertices]
    return float(np.sum(np.linalg.norm(np.roll(hull, -1, axis=0) - hull, axis=1))), len(np.unique(sector)) == SECTORS


def measure(source_positions: np.ndarray, source_triangles: np.ndarray, source_member: dict,
            source_joints: dict, own: list, skeleton, humanoid: dict) -> dict:
    """The rings of the character's own skin against the lent body's, and each limb's ratio.

    *source_positions*/*source_triangles*: the lent body at its source rest;
    *source_member*: ``{(limb, side): bool per source vertex}``;
    *source_joints*: ``{name: position}`` of the source rig; *own*: the
    character's own skin, ``[(positions, triangles, bone index, bone weight)]``
    at her rest on *skeleton*.  Returns ``{"arm": ratio | None, "leg": …,
    "rings": [[limb, segment, side, station, ratio], …]}``.
    """
    out: dict = {"rings": []}
    # one surface: a ring her skin closes across two parts is still hers
    offsets = np.cumsum([0] + [len(row[0]) for row in own])
    positions = np.concatenate([row[0] for row in own]) if own else np.zeros((0, 3))
    triangles = (np.concatenate([row[1] + offset for row, offset in zip(own, offsets)]) if own
                 else np.zeros((0, 3), dtype=np.int64))
    width = max((row[2].shape[1] for row in own), default=1)
    index = (np.concatenate([np.pad(row[2], ((0, 0), (0, width - row[2].shape[1]))) for row in own]) if own
             else np.zeros((0, 1), dtype=np.int64)).astype(np.int64)
    weight = (np.concatenate([np.pad(row[3], ((0, 0), (0, width - row[3].shape[1]))) for row in own]) if own
              else np.zeros((0, 1)))
    for limb, segments in LIMBS.items():
        ratios = []
        for side, role in (("L", "Left"), ("R", "Right")):
            under = np.zeros(len(skeleton.bones), dtype=bool)
            under[skeleton.descendants(humanoid[role + ROOT[limb]])] = True
            inside = np.sum(weight * under[index], axis=1) >= MEMBER
            for segment, first, last, human_first, human_last in segments:
                s0, s1 = source_joints[f"{first}_{side}"], source_joints[f"{last}_{side}"]
                d0 = skeleton.world_position(humanoid[role + human_first])
                d1 = skeleton.world_position(humanoid[role + human_last])
                for station in STATIONS:
                    lent, _closed = ring(source_positions, source_triangles, source_member[(limb, side)],
                                         s0 + station * (s1 - s0), s1 - s0)
                    if not lent:
                        continue
                    perimeter, closed = ring(positions, triangles, inside, d0 + station * (d1 - d0), d1 - d0)
                    if closed:
                        ratios.append(perimeter / lent)
                        out["rings"].append([limb, segment, side, station, round(perimeter / lent, 4)])
        out[limb] = float(np.median(ratios)) if ratios else None
    return out
