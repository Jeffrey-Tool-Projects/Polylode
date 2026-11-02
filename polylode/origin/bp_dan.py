"""BetterPenetration's shaft placement, transcribed (§7.1120).

The installed ``KK_BetterPenetration.dll`` (4.5.5.2) solves the shaft every
frame in a postfix of ``Lookat_dan.LateUpdate``, after the game has aimed it:
from ``cm_J_dan101_00`` toward the row's entrance, its length squished by how
deep it is, and its tip kept inside her body by two ruled surfaces through
her nulls — the front ``k_f_kokan_00`` → ``a_n_waist_f`` → ``a_n_bust_f``,
the back ``k_f_ana_00`` → ``a_n_waist_b`` → ``a_n_back``.  A tip that comes
out through one is slid back along it, the shaft's length kept.  The scrotum
is not moved; ``cm_J_dan_f_top`` takes the shaft root's rotation.

Every step below names the plugin method it is.  The math runs in Unity's
space: this project's is its X mirror (``S·M·S``), which turns the side a
plane reports and the sign of every cross product, so inputs are carried
over, solved there, and carried back.  A transform's ``forward``,
``right`` and ``rotation`` are its rotation, which its scale does not enter
(:func:`unity_rotation`), not its world matrix's columns.

A body with the source game's base rig has two of the shaft points
(``cm_J_dan101_00`` and ``cm_J_dan109_00``): the plugin's two-point branch
(``m_bpDanPointsFound`` false), one straight segment from the root to the
constrained tip.  A man wearing a BetterPenetration penis pack has all
nine (§7.1186): the points run straight toward the entrance and the one
segment that crosses it bends toward the constrained tip
(``AdjustDanPointsToTargets``), each point turned along its own segment
(``AimDanPoints``, §7.1187).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core import math3d as m3
from ..mesh.skeleton import affine_matrix

#: ``BoneNames.DanBones`` (the found ones, in order); the base rig has the first and the last
DAN_POINTS = ("cm_J_dan101_00", "cm_J_dan102_00", "cm_J_dan103_00", "cm_J_dan104_00", "cm_J_dan105_00",
              "cm_J_dan106_00", "cm_J_dan107_00", "cm_J_dan108_00", "cm_J_dan109_00")
#: ``DanPoints.danEnd`` (``InitializeDan``): put where the last point is, turned as it; a pack's bone
DAN_END = "cm_J_dan119_00"
#: ``BoneNames.TamaTop``: without it the plugin solves nothing (``InitializeDan``)
DAN_TOP = "cm_J_dan_f_top"

KOKAN_NULLS = ("k_f_kokan_00", "cf_J_Vagina_root")
ANA_NULLS = ("k_f_ana_00", "cf_J_Ana_Root")
HEAD_NULL = "k_f_head_00"
#: her bones the plugin aims at while inside (``m_bpKokanTarget`` / ``m_bpAnaTarget``)
BP_KOKAN, BP_ANA = "cf_J_Vagina_root", "cf_J_Ana_Root"
#: the inner limits (``m_innerTarget`` / ``m_innerHeadTarget``); without them nothing is solved
INNER, INNER_HEAD = "cf_j_waist01", "cf_J_FaceBase"
#: her bones set back to identity local rotations every frame (``ReturnKokanBones`` / ``ReturnAnaBones``)
RETURNED = ("cf_J_Vagina_Inner", "cf_J_Ana_Root")

#: ``PopulateCollisionOptionsList``: ``(null, inward)`` front and back, and the offsets of the
#: installed ``animal42069.KKbetterpenetration.cfg`` (default 0.04 for the second front one; the
#: install has 0.08, §7.1120)
FRONT = (("k_f_kokan_00", False, -0.04), ("a_n_waist_f", False, 0.08), ("a_n_bust_f", False, 0.06))
BACK = (("k_f_ana_00", False, -0.04), ("a_n_waist_b", True, 0.02), ("a_n_back", True, 0.0))
#: ``Clipping Depth``; skipped in front only with belly bulge (the main game's default).
CLIPPING_DEPTH = 0.02
BELLY_BULGE = True
#: ``Penis: Squish Threshold`` / ``Squish Length Factor`` / ``Squish Girth Factor`` (defaults)
SQUISH_THRESHOLD, LENGTH_SQUISH, GIRTH_SQUISH = 0.2, 0.6, 0.2
#: ``Rotate Balls with Shaft`` (default)
ROTATE_TAMA = True
#: ``Oral Push/Pull``: Enable, Max Push, Max Pull, Push/Pull Rate, Return Rate (the install's, the defaults)
ORAL_PUSH, ORAL_MAX_PUSH, ORAL_MAX_PULL, ORAL_RATE, ORAL_RETURN = True, 0.002, 0.006, 18.0, 0.03
#: the bone the mouth push moves (``CollisionAgent.m_oralPullBone``)
MOUTH = "cf_J_MouthMove"

#: every bone of hers the solve reads
FRAME_NAMES = tuple(dict.fromkeys(KOKAN_NULLS + ANA_NULLS + (HEAD_NULL, INNER, INNER_HEAD)
                                  + tuple(n for n, _i, _o in FRONT + BACK)))

_S = np.diag([-1.0, 1.0, 1.0, 1.0])


# -- Unity's own helpers (spec §14) ---------------------------------------
def _normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-5 else np.zeros(3)


def _angle(a, b) -> float:
    """``Vector3.Angle`` in degrees."""
    a, b = _normalize(a), _normalize(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def _look_rotation(forward, up) -> np.ndarray:
    """``Quaternion.LookRotation`` as a 3×3 basis (columns right, up, forward), Unity's space."""
    f = _normalize(forward)
    r = _normalize(np.cross(up, f))
    if not r.any():
        r = _normalize(np.cross((0.0, 1.0, 0.0) if abs(f[1]) < 0.999 else (1.0, 0.0, 0.0), f))
    return np.column_stack([r, np.cross(f, r), f])


# -- MathHelpers (spec §7.4) ----------------------------------------------
def _zero(x: float) -> bool:
    return -0.0001 < x < 0.0001


def _vectors_equal(a, b, threshold: float) -> bool:
    return bool(np.all(np.abs(np.asarray(a) - np.asarray(b)) <= threshold))


def _solve_quadratic(a: float, b: float, c: float):
    if _zero(a):
        if _zero(b):
            return None
        return -c / b, -c / b
    d = b * b - 4.0 * a * c
    if d < 0.0:
        return None
    root = math.sqrt(d)
    return (-b + root) / (2.0 * a), (-b - root) / (2.0 * a)


def _ssa_side_c(a: float, b: float, angle_a: float) -> float:
    """``SolveSSATriangle``'s third side (0 for a degenerate triangle, as the plugin leaves it)."""
    if _zero(a) or _zero(b) or _zero(angle_a):
        return 0.0
    angle_b = math.asin(max(-1.0, min(1.0, math.sin(angle_a) * b / a)))
    angle_c = math.pi - angle_a - angle_b
    return a * math.sin(angle_c) / math.sin(angle_a)


def _cast_to_segment(p, start, vector) -> np.ndarray:
    """``CastToSegment``: divides by the vector's length, not its square (as the plugin does)."""
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    k = float(np.dot(np.asarray(p) - start, vector)) / length if length > 0.0 else 0.0
    return start + vector * k


def _cast_segment_to_segment(a, a_vec, b, b_vec) -> np.ndarray:
    a0 = a - b_vec * float(np.dot(a - b, b_vec))
    a1 = ((a + a_vec) - b_vec * float(np.dot((a + a_vec) - b, b_vec))) - a0
    length = float(np.linalg.norm(a1))
    k = float(np.dot(b - a0, a1)) / length if length > 0.0 else 0.0
    return _cast_to_segment(a + a_vec * k, b, b_vec)


# -- TwistedPlane (spec §7.3) ----------------------------------------------
@dataclass
class TwistedPlane:
    """A ruled surface between two lateral segments: ``S(u, v) = o1 + v1·v + (fwd + twist·v)·u``."""
    first_origin: np.ndarray
    first_vector: np.ndarray
    second_origin: np.ndarray
    second_vector: np.ndarray

    @property
    def forward(self) -> np.ndarray:
        return self.second_origin - self.first_origin

    @property
    def twist(self) -> np.ndarray:
        return self.second_vector - self.first_vector

    def _values(self, lv, ov, fv_lv, tw_lv, ov_lv, lv_fw, u):
        """``FindIntersectValues``: ``(v, t)`` or None."""
        v = None
        for k in range(3):
            den = fv_lv[k] + tw_lv[k] * u
            if not _zero(den):
                v = (ov_lv[k] + lv_fw[k] * u) / den
                break
        if v is None:
            return None
        for k in range(3):
            if not _zero(lv[k]):
                t = (-ov[k] + self.first_vector[k] * v + (self.forward[k] + self.twist[k] * v) * u) / lv[k]
                return v, t
        return None

    def intersect(self, start, end, extend_first: bool, extend_second: bool):
        """``IntersectLineOnTwistedPlane``: ``(hit, surface forward, distance to the second edge)`` or None."""
        lv, ov = end - start, start - self.first_origin
        fwd, tw, fv = self.forward, self.twist, self.first_vector
        a = float(np.dot(lv, np.cross(tw, fwd)))
        b = float(np.dot(lv, np.cross(fv, fwd) + np.cross(ov, tw)))
        c = float(np.dot(lv, np.cross(ov, fv)))
        roots = _solve_quadratic(a, b, c)
        if roots is None:
            return None
        fv_lv, tw_lv, ov_lv, lv_fw = np.cross(fv, lv), np.cross(tw, lv), np.cross(ov, lv), np.cross(lv, fwd)
        best = None
        for u in roots:
            if not (-4.0 < u < 4.0 and (u >= 0.0 or extend_first) and (u <= 1.0 or extend_second)):
                continue
            found = self._values(lv, ov, fv_lv, tw_lv, ov_lv, lv_fw, u)
            if found is None or not 0.0 <= found[1] <= 1.0:
                continue
            if best is None or found[1] < best:
                best = found[1]
        if best is None:
            return None
        hit = start + lv * best
        s1 = _cast_segment_to_segment(start, lv, self.first_origin, self.first_vector)
        s2 = _cast_segment_to_segment(start, lv, self.second_origin, self.second_vector)
        return hit, _normalize(s2 - s1), float(np.linalg.norm(s2 - hit))

    def constrain(self, start, end, length: float, extend_start: bool, extend_end: bool):
        """``ConstrainLineToTwistedPlane``: ``(end, extend_start, found)``."""
        direction = _normalize(end - start)
        normal = _normalize(np.cross(_normalize(self.first_vector + self.second_vector), self.forward))
        point = self.first_origin + self.forward / 2.0
        if not float(np.dot(normal, end)) - float(np.dot(normal, point)) > 0.0:
            return end, False, False                     # the tip is on the inside
        crossing = self.intersect(start, end, extend_start, extend_end)
        if crossing is None:
            return end, False, False
        hit, surface, to_second = crossing
        h = float(np.linalg.norm(hit - start))
        if h > length:
            return end, False, False
        c = _ssa_side_c(length, h, math.radians(_angle(direction, -surface)))
        if not extend_end and c > to_second:
            p = hit + surface * to_second
            return start + _normalize(p - start) * length, True, False
        return hit + surface * c, False, True


# -- the solve (spec §6–§8) ------------------------------------------------
@dataclass
class Shaft:
    """What the plugin writes on him: the points' world positions and rotations, Unity's space.

    *rotation* is the root's (the scrotum top takes it), *rotations* one per
    point.  *direction* and *distance* (root toward the target, and how far)
    are what ``AdjustPullBones`` is handed; None toward no target.
    """
    points: list
    rotation: np.ndarray
    girth: float
    direction: np.ndarray | None = None
    distance: float | None = None
    toward_head: bool = False
    rotations: list = field(default_factory=list)


def in_mouth(shaft: "Shaft", base: float) -> bool:
    """The plugin's test for the shaft in her mouth: aimed at her head null, that nearer than *base* (its length)."""
    return bool(shaft.toward_head and shaft.distance is not None and shaft.distance < base)


@dataclass
class OralPull:
    """The mouth push, carried frame to frame: ``CollisionAgent.currentOralPull`` and
    ``currentOralDanDirection``, and ``DanAgent.lastDanDistance`` (None until the first pull: the base length)."""

    current: float = 0.0
    direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last: float | None = None
    #: whether the last step had the shaft in her mouth: aimed at her head null and nearer than its length
    inside: bool = False

    def step(self, shaft: "Shaft", base: float, dt: float) -> bool:
        """``AdjustPullBones`` for the mouth (``PullOralBone`` / ``ReturnOralBones``); whether the bone is written.

        The plugin hands every target its ``AdjustPullBones``; only the head
        one asks for the oral pull, and only while the shaft is nearer than its
        length.  With no pull asked, ``ReturnOralBones`` eases it back (the
        hook passes ``false`` for the flag that would skip that).
        """
        oral = self.inside = in_mouth(shaft, base)
        last = base if self.last is None else self.last
        if oral:
            self.last = shaft.distance
            if ORAL_PUSH:
                pull = self.current + (last - shaft.distance) * ORAL_RATE * dt
                self.current = min(max(pull, -ORAL_MAX_PULL), ORAL_MAX_PUSH)
                self.direction = np.asarray(shaft.direction, dtype=np.float64)
                return True
        else:
            self.last = base
        if -1e-4 < self.current < 1e-4:                  # ``ApproximatelyZero``: left where it is
            return False
        back = ORAL_RETURN * dt
        self.current = self.current - back if self.current > back else (
            self.current + back if self.current < -back else 0.0)
        return True


def _surface(frames: dict, nulls, front: bool, belly_bulge: bool | None = None):
    """The points and lateral vectors of one surface (``ConstrainDanToCollisionPoints``), or None when a null is missing."""
    points, sides = [], []
    bulge = BELLY_BULGE if belly_bulge is None else belly_bulge
    for name, inward, offset in nulls:
        m = frames.get(name)
        if m is None:
            return None
        off = offset if (front and bulge) else offset + CLIPPING_DEPTH
        forward, right = _normalize(m[:3, 2]), _normalize(m[:3, 0])
        points.append(m[:3, 3] + off * forward if front == inward else m[:3, 3] - off * forward)
        sides.append(-right if front == inward else right)
    return points, sides


def _constrain_to_surface(start, end, length: float, surface, found: bool):
    """One surface's strips in turn, each from the original *end* (``ConstrainDanToCollisionPoints``)."""
    if found or surface is None:
        return end, found
    points, sides = surface
    result, extend_start = end, True
    for i in range(1, len(points)):
        if found:
            break
        plane = TwistedPlane(points[i - 1], sides[i - 1], points[i], sides[i])
        result, extend_start, found = plane.constrain(start, end, length, extend_start, i == len(points) - 1)
    return result, found


def constrain_to_body(frames: dict, start, direction, length: float,
                      belly_bulge: bool | None = None) -> np.ndarray:
    """``ConstrainDanToBody``: the tip kept inside her front and back surfaces."""
    end = start + direction * length
    end, found = _constrain_to_surface(start, end, length, _surface(frames, FRONT, True, belly_bulge), False)
    end, _found = _constrain_to_surface(start, end, length, _surface(frames, BACK, False), found)
    return end


def squished(base: float, dist: float) -> tuple[float, float]:
    """``GetSquishedDanLength`` / ``GetSquishedDanGirth``: ``(length, girth)``."""
    rest = base - base * SQUISH_THRESHOLD - dist
    if rest > 0.0:
        return base - rest * LENGTH_SQUISH, 1.0 + (rest / base) * GIRTH_SQUISH
    return base, 1.0


def reference(look_at: str, motion: str, top_stick: bool, has) -> tuple[str | None, bool]:
    """``SetupNewDanTarget``: the bone aimed at and whether the shaft is in (``m_danPenetration``).

    *has* says whether she has a bone.  ``Contains`` is case-sensitive.
    """
    if not motion:
        return None, False
    ref = look_at or None
    inside = (top_stick and ref is not None) or "IN" in motion
    deeper = "Insert" in motion or not ("Idle" in motion or "Pull" in motion or "OUT" in motion)
    for bp, outer in ((BP_KOKAN, "k_f_kokan_00"), (BP_ANA, "k_f_ana_00")):
        if inside and has(bp) and (ref is None or ref == outer) and deeper:
            return bp, inside
    return ref, inside


def adjust_points(start, bend, goal, length: float, dist: float, count: int, multi: bool) -> list:
    """``AdjustDanPointsToTargets``: *count* points from *start*, *length* long.

    With the plugin's own shaft bones (*multi*) they run toward *bend* (the
    entrance) and, past *dist* (how far it is), toward *goal* (the constrained
    tip); the segment that crosses it ends on the line out of *bend*, its
    length kept (``SolveSSATriangle``).  Straight toward *goal* without them,
    and straight toward *bend* when the two directions agree or the shaft
    does not reach it.
    """
    start = np.asarray(start, dtype=np.float64)
    first, after = _normalize(bend - start), _normalize(goal - bend)
    straight = False
    if not multi:
        first = after = _normalize(goal - start)
        straight = True
    elif _vectors_equal(first, after, 0.001) or not length > dist:
        after = first
        straight = True
    seg = length / float(count - 1)
    points = [start]
    for i in range(1, count):
        rest = dist - seg * (i - 1)
        if straight or rest < 0.001:
            points.append(points[-1] + after * seg)
        elif rest < seg - 0.001:
            c = _ssa_side_c(seg, rest, math.radians(_angle(first, -after)))
            points.append(np.asarray(bend, dtype=np.float64) + after * c)
        else:
            points.append(points[-1] + first * seg)
    return points


def aim_points(points: list, dan_top: np.ndarray) -> list:
    """``AimDanPoints``: each point along its segment, its up from the scrotum top's right; the last as the one before it."""
    right = _normalize(dan_top[:3, 0])
    out = []
    for p, q in zip(points, points[1:]):
        forward = _normalize(q - p)
        out.append(_look_rotation(forward, np.cross(forward, right)))
    return out + out[-1:]


def solve(dan: list, dan_top: np.ndarray, frames: dict, ref: str | None, inside: bool, base: float,
          belly_bulge: bool | None = None) -> Shaft | None:
    """``SetDanTarget``: where the shaft's points go, Unity's space.

    *dan* is the shaft points' world matrices (root first), *dan_top* the
    scrotum top's, *frames* her bones' (``name → 4×4``), *ref* and *inside*
    from :func:`reference`, *base* the shaft's length (``m_baseDanLength``).
    None when the plugin solves nothing (her inner limits missing).
    """
    if frames.get(INNER) is None or frames.get(INNER_HEAD) is None:
        return None
    start, end = dan[0][:3, 3], dan[-1][:3, 3]
    count, multi = len(dan), len(dan) > 2                # m_bpDanPointsFound: more than two found
    girth = 1.0
    direction = distance = None
    if ref is None or ref not in KOKAN_NULLS + ANA_NULLS + (HEAD_NULL,) or frames.get(ref) is None:
        # AdjustDanToTargetNull: straight on along where it points
        goal = start + _normalize(end - start) * base
        points = adjust_points(start, end, goal, base, float(np.linalg.norm(end - start)), count, multi)
    else:
        m = frames[ref]
        entry = m[:3, 3].copy()                          # every target offset is 0 in the config
        direction = _normalize(entry - start)
        dist = distance = float(np.linalg.norm(entry - start))
        length, g = squished(base, dist)
        tip = start + direction * base
        if ref in KOKAN_NULLS:
            girth = g
        if ref == HEAD_NULL:
            inner = frames[INNER_HEAD][:3, 3]
            length = min(length, dist + float(np.linalg.norm(entry - inner)))    # GetMaxDanLength
        if inside:
            if ref in KOKAN_NULLS + ANA_NULLS:
                tip = constrain_to_body(frames, start, direction, length, belly_bulge)
            else:
                tip = start + direction * length                                 # ConstrainDanToHead
        elif ref == "k_f_kokan_00" and base > dist:
            tip = start + direction * length                                     # ConstrainDanToPull
        points = adjust_points(start, entry, tip, length, dist, count, multi)
    rotations = aim_points(points, dan_top)
    return Shaft(points, rotations[0], girth, direction, distance, ref == HEAD_NULL, rotations=rotations)


@dataclass(frozen=True)
class Released:
    """What ``Lookat_dan`` hands the plugin on a row without a ``dan_*`` list, or before ``SetInfo`` found one.

    ``LoadList`` starts with ``Release``: ``strPlayMotion = string.Empty`` (not
    null), no null, ``bTopStick`` false; ``numFemale`` is not reset.  The
    plugin's ``Lookat_dan`` postfix returns only on a **null** motion, so it
    still solves every frame — toward no target, the shaft straightened along
    where it points at its length (``AdjustDanToTargetNull``, §7.1200).
    """
    num_female: int = 0
    motion: str = ""
    look_at_null: str = ""
    top_stick: bool = False


def to_unity(m) -> np.ndarray:
    """A world matrix of this project's space in Unity's (and back: the mirror is its own inverse)."""
    return _S @ np.asarray(m, dtype=np.float64) @ _S


# -- onto a pose -----------------------------------------------------------
def _world_rotation(m) -> np.ndarray:
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    return r / np.maximum(np.linalg.norm(r, axis=0), 1e-12)


def _local_rotation(bone, entry: dict | None) -> np.ndarray:
    """A bone's ``localRotation`` under its pose entry (3×3): what :meth:`Skeleton.local_of` turns it by."""
    entry = entry or {}
    if "matrix" in entry:
        return m3.quat_to_mat4(m3.decompose(affine_matrix(entry["matrix"]))[1])[:3, :3]
    rot = entry.get("rot")
    return m3.quat_to_mat4(bone.local_rot if rot is None else m3.quat_mul(bone.local_rot, rot))[:3, :3]


def unity_rotation(skel, pose: dict | None, xf, name: str) -> np.ndarray:
    """``Transform.rotation`` of bone *name* (3×3, this project's space): its local rotations multiplied down the chain.

    Unity keeps a transform's rotation apart from its scale, and the
    plugin's every read of a turn — ``forward`` and ``right`` of her nulls
    (``ConstrainDanToCollisionPoints``), ``right`` of the scrotum top
    (``AimDanPoints``), the parent a ``rotation`` is written under — is that
    rotation.  A world matrix's columns carry the scale along: a bone an H
    clip scales to nothing has none, though its rotation is all there
    (``k_f_kokan_00`` on the desk rows of mode 2).  *xf* is the model's
    transform; its turn is taken as its columns'.
    """
    pose = pose or {}
    r = _world_rotation(xf)
    for i in skel.chain(name):
        bone = skel.bones[i]
        r = r @ _local_rotation(bone, pose.get(bone.name))
    return r


def _with_rotation(m, rotation) -> np.ndarray:
    out = np.array(m, dtype=np.float64)
    out[:3, :3] = rotation
    return out


def _set_world(skel, pose: dict, xf, bone: str, rotation=None, position=None, scale=None) -> None:
    """Write *bone*'s world rotation / position (this project's space) into *pose* as its channels."""
    i = skel.id(bone)
    b = skel.bones[i]
    parent = b.parent
    world = skel.evaluate(pose, only=[parent]) if parent >= 0 else None
    parent_m = np.asarray(xf, dtype=np.float64) @ (world[parent] if parent >= 0 else np.eye(4))
    entry = dict(pose.get(bone) or {})
    if rotation is not None:
        # ``Transform.rotation = r`` sets ``localRotation = parent.rotation⁻¹ · r``
        turn = unity_rotation(skel, pose, xf, skel.bones[parent].name) if parent >= 0 else _world_rotation(xf)
        local = turn.T @ rotation
        q = m3.quat_from_mat4(np.block([[local, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
        entry["rot"] = m3.quat_normalize(m3.quat_mul(m3.quat_conj(b.local_rot), q))
    if position is not None:
        local_pos = (np.linalg.inv(parent_m) @ np.append(position, 1.0))[:3]
        rest = b.authored_local[:3, 3] if b.authored_local is not None else b.local_pos
        entry["pos"] = local_pos - np.asarray(rest, dtype=np.float64)
    if scale is not None:
        entry["scale"] = np.asarray(scale, dtype=np.float64)
    pose[bone] = entry


def base_length(skel, xf, points: list) -> float:
    """``m_baseDanLength``: the root-to-tip distance at rest, times the point count less one over that."""
    rest = [np.asarray(xf, dtype=np.float64) @ skel.bones[skel.id(n)].bind_world for n in points]
    return float(np.linalg.norm(rest[1][:3, 3] - rest[0][:3, 3])) * (len(points) - 1)


def apply(a_pose: dict, a_xf, a_skel, row, frames: dict, her=None, oral: OralPull | None = None,
          dt: float = 0.0, belly_bulge: bool | None = None) -> bool:
    """One frame of the plugin's solve on *a_pose* (his, after the game's aim); whether it ran.

    *row* is the ``dan_*`` row playing (``motion``, ``look_at_null``,
    ``top_stick``); *frames* her bones' world matrices in this project's
    space, positioned where his skeleton sees her (``h_runtime``); *her*,
    ``(pose, skeleton, transform)``, the skeleton they were read off, turns
    each of them as its transform is turned (:func:`unity_rotation`: the
    matrix of a bone scaled to nothing has no axes), takes the identity local
    rotations the plugin writes on her every frame and, with *oral* (the push
    carried between frames) over *dt* seconds, her mouth's push.
    *belly_bulge* selects the source plugin's front-clearance branch: None
    preserves its configured default, False retains Clipping Depth for a
    body without the belly-bulge solver.
    """
    if a_skel is None or row is None:
        return False
    found = [n for n in DAN_POINTS if a_skel.has(n)]
    if len(found) < 2 or not a_skel.has(DAN_TOP):
        return False                                    # InitializeDan: no shaft to solve
    xf = np.asarray(a_xf, dtype=np.float64)
    world = a_skel.evaluate(a_pose, only=[a_skel.id(n) for n in found + [DAN_TOP]])
    dan = [to_unity(xf @ world[a_skel.id(n)]) for n in found]
    top = to_unity(_with_rotation(xf @ world[a_skel.id(DAN_TOP)], unity_rotation(a_skel, a_pose, xf, DAN_TOP)))
    if her is not None and her[1] is not None:
        # her nulls turned as their transforms are, from the skeleton *frames* were read off
        h_pose, h_skel, h_xf = her
        frames = {name: (m if m is None or not h_skel.has(name)
                         else _with_rotation(m, unity_rotation(h_skel, h_pose, h_xf, name)))
                  for name, m in frames.items()}
    unity = {name: to_unity(m) for name, m in frames.items() if m is not None}
    ref, inside = reference(row.look_at_null, row.motion, bool(row.top_stick), lambda n: n in unity)
    shaft = solve(dan, top, unity, ref, inside, base_length(a_skel, xf, found), belly_bulge)
    if shaft is None:
        if oral is not None:
            oral.inside = False
        return False
    s = _S[:3, :3]
    rotations = [s @ r @ s for r in shaft.rotations]
    points = [s @ p for p in shaft.points]
    g = shaft.girth
    # SquishDanGirth: the root by g, every other point by (g − (g − 1)/n)/g, n the segments
    rest_g = (g - (g - 1.0) / float(len(found) - 1)) / g
    # parents first: the points are a chain (dan109 under dan101 on the base rig), the end under the last
    for i, name in enumerate(found):
        k = g if i == 0 else rest_g
        _set_world(a_skel, a_pose, xf, name, rotation=rotations[i],
                   position=points[i] if i else None, scale=(k, k, 1.0))
    if a_skel.has(DAN_END):
        _set_world(a_skel, a_pose, xf, DAN_END, rotation=rotations[-1], position=points[-1])
    if ROTATE_TAMA:
        _set_world(a_skel, a_pose, xf, DAN_TOP, rotation=rotations[0])
    if her is not None:
        pose, skel, her_xf = her
        for bone in RETURNED:
            if skel is not None and skel.has(bone):
                b = skel.bones[skel.id(bone)]
                pose[bone] = {**(pose.get(bone) or {}), "rot": m3.quat_conj(b.local_rot)}
        if oral is not None and skel is not None and skel.has(MOUTH):
            if oral.step(shaft, base_length(a_skel, xf, found), dt):
                _push_mouth(pose, skel, her_xf, oral)
        elif oral is not None:
            oral.inside = in_mouth(shaft, base_length(a_skel, xf, found))      # no mouth bone to push
    return True


def _push_mouth(pose: dict, skel, xf, oral: OralPull) -> None:
    """``m_oralPullBone.localPosition = currentOralPull * m_oralPullBone.InverseTransformDirection(direction)``.

    The direction is taken into the bone's own frame (its world turn), and
    the result replaces the bone's local position.  The mirror carries over:
    a direction and a frame both mirrored give the mirrored local vector.
    """
    i = skel.id(MOUTH)
    direction = _S[:3, :3] @ oral.direction
    local = oral.current * (unity_rotation(skel, pose, xf, MOUTH).T @ direction)
    b = skel.bones[i]
    rest = b.authored_local[:3, 3] if b.authored_local is not None else b.local_pos
    pose[MOUTH] = {**(pose.get(MOUTH) or {}), "pos": local - np.asarray(rest, dtype=np.float64)}
