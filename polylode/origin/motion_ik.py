"""The H scene's ``MotionIK``, transcribed (Animation_Progress D14).

``HSceneProc`` builds one ``MotionIK`` per character and loads each with the
data of the controller file its role plays (``motion_ik.json``).  On every
``SetPlay`` (and ``ChangeAnimator``'s ``Calc("Idle")``) each one re-binds,
from the state just played:

* ``FrameCorrect``: 24 additive local corrections on the character's own
  bones (hips … head, waist, fingers, bust), sized by a partner's body shape;
* ``IKCorrect``: the four limb effector targets and their bend goals
  (``cf_t_*``), each snapped every frame onto a bone of self or a partner
  (``part.sex`` / ``editNo`` index ``partners``, not a sex);
* the ``FullBodyBipedIK`` effector and bend weights, and whether it runs.

Every frame then runs, in the script execution order of the install
(``MonoScript.m_ExecutionOrder``): 5500 ``IKBeforeOfDankonProcess`` (the body
corrections), 6000 ``Lookat_dan`` (the caller's), 7000 ``IKBeforeProcess``
(every character's targets placed from every character's pre-IK pose), 9999
``FullBodyBipedIK`` (``anim.fbbik``), 10050 ``IKAfterProcess`` (the fingers).

The game's numbers are in Unity's frame; this project's is its X mirror
(``S·M·S``, as ``bp_dan.to_unity``): a position mirrors as ``(−x, y, z)``, a
quaternion as ``(x, −y, −z, w)``.  Rotations are taken the way ``Transform``
composes them (parent rotation times local rotation, scale left out); a local
translation under ``cf_n_height`` is in that node's unscaled units, which
the graft bakes into the bone positions (``Skeleton.rig_scale``).

The pose a solver initiates on (spec UNRESOLVED 1) is chosen in one place,
:meth:`Body.initiate`: the first enabled frame's animated pose, before that
frame's 5500 corrections.  Unity calls a component's deferred ``Start``
before it dispatches the behaviour's first ``LateUpdate``; this is the earliest
point it can run, and the one where ``FixTransforms`` restores exactly the
pose the Animator leaves on a channel it does not write.  A converted
character initiates on that frame at the source rig's lengths (its clip
offsets taken off), the shape a card's ``Initiate`` sees.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..anim.fbbik import FBBIKSettings, FullBodyBipedIK, settings_from_serialized
from ..core import math3d as m3
from .hlist import MOTION_IK_FRAMES, MOTION_IK_LIMB_FRAMES, MOTION_IK_PARTS

#: switch for ``tools/h_pair_compare.py --no-motion-ik`` (the game without this step); on in the game
ENABLED = True

#: ``FrameCorrect`` indexes whose process is ``IKBeforeOfDankonProcess`` (order 5500): the body
#: bones 0‥7 and the bust 18‥23; the fingers 8‥17 are ``IKAfterProcess`` (10050) — the prefab's
#: ``ProcOrderType`` First / Last on ``p_cf_body_bone``
BEFORE = tuple(range(8)) + tuple(range(18, 24))
AFTER = tuple(range(8, 18))
#: ``IKTargetPair.GetPairs``: (part, effector index, chain index) — ``leftHandEffector`` is
#: ``effectors[5]``, ``leftArmChain`` ``chain[1]`` (``IKSolverFullBodyBiped``)
PAIRS = tuple((part, 5 + i, 1 + i) for i, part in enumerate(MOTION_IK_PARTS))
#: ``IKTargetPair.IKTargetLength``: frames 2·i (effector target) and 2·i + 1 (bend goal) of limb i
LIMB_FRAMES = MOTION_IK_LIMB_FRAMES
#: how much of a limb's straight length a held target is kept within (``Body._keep_within_reach``):
#: a hair short of straight, so the chain is not left at the edge of pulling
REACH_SHORT = 0.999
_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


class MotionIKError(RuntimeError):
    """What would throw inside the game's ``SetPlay``: a partner index past the cast, a frame past the list."""


# -- Unity math --------------------------------------------------------------
def mirror_q(q) -> np.ndarray:
    """A rotation of this project's frame in Unity's, and back (``S·R·S``)."""
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], q[3]])


def mirror_v(v) -> np.ndarray:
    """A position or direction of this project's frame in Unity's, and back."""
    v = np.asarray(v, dtype=np.float64)
    return np.array([-v[0], v[1], v[2]])


def euler(ang) -> np.ndarray:
    """``Quaternion.Euler(x, y, z)`` in degrees: Z first, then X, then Y (``qY · qX · qZ``)."""
    x, y, z = (math.radians(float(a)) for a in ang)
    return m3.quat_from_euler(x, y, z, order="YXZ")


def inverse_lerp(a: float, b: float, v: float) -> float:
    """``Mathf.InverseLerp``: clamped to [0, 1]; 0 when ``a == b``."""
    if a == b:
        return 0.0
    return min(max((v - a) / (b - a), 0.0), 1.0)


def lerp_angle(a: float, b: float, t: float) -> float:
    """``Mathf.LerpAngle``: the short way round, not normalized afterwards."""
    d = (b - a) % 360.0                      # Mathf.Repeat
    if d > 180.0:
        d -= 360.0
    return a + d * min(max(t, 0.0), 1.0)


def shape_lerp_position(v: float, small, large) -> np.ndarray:
    """``MotionIK.GetShapeLerpPositionValue``: toward *small* below 0.5 (NaN too, ``bge``), *large* above."""
    small, large = np.asarray(small, dtype=np.float64), np.asarray(large, dtype=np.float64)
    if not v >= 0.5:
        return small + (np.zeros(3) - small) * inverse_lerp(0.0, 0.5, v)
    return large * inverse_lerp(0.5, 1.0, v)


def shape_lerp_angle(v: float, small, large) -> np.ndarray:
    """``MotionIK.GetShapeLerpAngleValue``: per axis ``LerpAngle`` (``blt.un`` sends NaN to the small side)."""
    if v >= 0.5:
        t = inverse_lerp(0.5, 1.0, v)
        return np.array([lerp_angle(0.0, float(c), t) for c in large])
    t = inverse_lerp(0.0, 0.5, v)
    return np.array([lerp_angle(float(c), 0.0, t) for c in small])


def shape_values(shape: Callable[[int], float | None], shapes: list) -> tuple:
    """``MotionIK.GetCorrectShapeValues``: ``(pos, ang)`` summed over *shapes* on a body whose sliders *shape* reads."""
    pos, ang = np.zeros(3), np.zeros(3)
    for sh in shapes:
        no = int(sh["shapeNo"])
        if no < 0:
            raise MotionIKError(f"shape {no} is not a body shape")
        value = shape(no)
        v = 0.0 if value is None else float(value)       # GetShapeBodyValue: 0 past ShapeBodyNum
        pos = pos + shape_lerp_position(v, sh["small"][0], sh["large"][0])
        ang = ang + shape_lerp_angle(v, sh["small"][1], sh["large"][1])
    return pos, ang


# -- the components on one character ----------------------------------------
@dataclass
class Correction:
    """``BaseData`` (pos, rot; Unity's frame) with its process's ``enabled``."""
    enabled: bool = False
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot: np.ndarray = field(default_factory=lambda: _IDENTITY.copy())
    #: ``BaseData.bone`` of an ``IKCorrect`` entry: ``(body, bone name)`` or None
    bone: tuple | None = None

    def reset(self, keep_bone: bool) -> None:
        """``InitFrameCalc``: disabled, no offset; an ``IKCorrect`` entry also loses its bone."""
        self.enabled = False
        self.pos, self.rot = np.zeros(3), _IDENTITY.copy()
        if not keep_bone:
            self.bone = None


def load_settings(data: dict) -> FBBIKSettings:
    """A fresh ``FBBIKSettings`` from the baked ``fbbik.json`` payload (the solver writes into it)."""
    names = data["names"]
    return settings_from_serialized(data["solver"], lambda pid: names.get(str(pid)))


def _ancestor_closure(skel, bones) -> list:
    """Every bone of *bones* with all its ancestors, parents first; ``KeyError`` for one the skeleton lacks."""
    need: set = set()
    for name in bones:
        if not skel.has(name):
            raise KeyError(f"the skeleton has no '{name}' for the IK")
        i = skel.id(name)
        while i >= 0 and i not in need:
            need.add(i)
            i = skel.bones[i].parent
    return [skel.bones[i].name for i in sorted(need)]


def _settings_bones(s: FBBIKSettings) -> set:
    """Every bone the solver moves or reads (not the targets)."""
    names = {n.bone for c in s.chains for n in c.nodes}
    names |= {b for e in s.effectors for b in (e.bone, *e.child_bones, e.plane_bone1, e.plane_bone2,
                                                e.plane_bone3) if b}
    names |= set(s.spine.spine_bones) | {b.bone for b in s.bone_mappings}
    names |= {b for m in s.limb_mappings for b in (m.bone1, m.bone2, m.bone3, m.parent_bone) if b}
    return names


def _rotation_of(m: np.ndarray) -> np.ndarray:
    """The rotation of a matrix's 3×3, its columns' scale left out."""
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    r = r / np.maximum(np.linalg.norm(r, axis=0), 1e-12)
    out = np.eye(4)
    out[:3, :3] = r
    return m3.quat_from_mat4(out)


# -- scalar quaternion work (``math3d``'s formulas on Python floats: this runs per bone, per frame) --
def _qmul(a, b) -> list:
    """``math3d.quat_mul``."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def _qrot(q, v) -> list:
    """``math3d.quat_rotate``."""
    ux, uy, uz, s = q
    vx, vy, vz = v
    d = 2.0 * (ux * vx + uy * vy + uz * vz)
    k = s * s - (ux * ux + uy * uy + uz * uz)
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return [d * ux + k * vx + 2.0 * s * cx, d * uy + k * vy + 2.0 * s * cy, d * uz + k * vz + 2.0 * s * cz]


def _qconj(q) -> list:
    return [-q[0], -q[1], -q[2], q[3]]


def _qmirror(q) -> list:
    """:func:`mirror_q` on a list."""
    return [q[0], -q[1], -q[2], q[3]]


def _vmirror(v) -> list:
    """:func:`mirror_v` on a list."""
    return [-v[0], v[1], v[2]]


def _dot4(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def _dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _f(v) -> list:
    """A quaternion or vector's components as Python floats."""
    return v.tolist() if isinstance(v, np.ndarray) else [float(c) for c in v]


def _q_matrix(q) -> list:
    """``math3d.quat_to_mat4``'s 3×3 (``q`` normalized first) as rows."""
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    x, y, z, w = (c / n for c in q) if n > m3.EPS else (0.0, 0.0, 0.0, 1.0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [[1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]]


def _matrix_q(m) -> list:
    """``math3d.quat_from_mat4`` on a 3×3 given as rows, normalized as there."""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s]
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        q = [0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s]
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        q = [(m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s]
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        q = [(m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s]
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return [c / n for c in q] if n > m3.EPS else [0.0, 0.0, 0.0, 1.0]


def _affine(m: np.ndarray) -> tuple:
    """A 4×4's linear part (rows) and translation as Python floats."""
    rows = m[:3].tolist()
    return [r[:3] for r in rows], [rows[0][3], rows[1][3], rows[2][3]]


def _compose(pm, pt, lm, lt) -> tuple:
    """``parent @ local`` for two affine transforms given as (rows, translation)."""
    m = [[pr[0] * lm[0][c] + pr[1] * lm[1][c] + pr[2] * lm[2][c] for c in range(3)] for pr in pm]
    t = [pr[0] * lt[0] + pr[1] * lt[1] + pr[2] * lt[2] + pt[r] for r, pr in enumerate(pm)]
    return m, t


class _Frame:
    """One look at a body: world transforms and ``Transform``-composed rotations (this project's frame).

    Built for the bones the body's MotionIK reads (an ancestor-closed set,
    parents first) on Python floats, and built again once the pose was written.
    ``lin`` / ``pos`` are each bone's world linear part (rows, scale in its
    columns) and position; ``rot`` its rotation.
    """

    def __init__(self, body: Body):
        skel, pose, xf = body.skel, body.pose, body.xf
        xm, xt = _affine(xf)
        xq = _rotation_of(xf).tolist()
        self.lin: dict = {}
        self.pos: dict = {}
        self.rot: dict = {}
        for i in body.need:
            b = skel.bones[i]
            lm, lt, lq = body.local(i, pose.get(b.name))
            if b.parent < 0:
                pm, pt, pq = xm, xt, xq
            else:
                pm, pt, pq = self.lin[b.parent], self.pos[b.parent], self.rot[b.parent]
            self.lin[i], self.pos[i] = _compose(pm, pt, lm, lt)
            self.rot[i] = _qmul(pq, lq)

    def scale(self, i: int) -> np.ndarray:
        """Bone *i*'s world scale: its linear part's column lengths."""
        m = self.lin[i]
        return np.array([math.sqrt(m[0][c] * m[0][c] + m[1][c] * m[1][c] + m[2][c] * m[2][c]) for c in range(3)])

    def unity(self, i: int) -> tuple:
        """Bone *i*'s ``(position, rotation)`` in Unity's frame, as lists."""
        return _vmirror(self.pos[i]), _qmirror(self.rot[i])


class Body:
    """One character's ``FrameCorrect`` / ``IKCorrect`` / ``FullBodyBipedIK`` and its frame's pose.

    It outlives a row: the component is initiated once, on the first frame it
    runs (``SolverManager.Start``), and keeps what FinalIK keeps.
    """

    def __init__(self, skel, shape: Callable[[int], float | None], settings: FBBIKSettings,
                 body_bone_rotation=_IDENTITY):
        self.skel, self.shape, self.settings = skel, shape, settings
        #: ``p_cf_body_bone``'s local rotation under the character (Unity's frame)
        self.body_bone_rotation = np.asarray(body_bone_rotation, dtype=np.float64)
        self.frames = [Correction(bone=(self, name)) for name in MOTION_IK_FRAMES]
        self.links = {name: Correction() for name in self._link_names()}
        self.ik_enabled = False
        self.solver: FullBodyBipedIK | None = None
        mapped = _settings_bones(self.settings)
        self.bones = _ancestor_closure(skel, mapped)
        #: the bones the solver writes, in the skeleton's order
        self.mapped = [n for n in self.bones if n in mapped]
        targets = {e.target for e in self.settings.effectors if e.target}
        targets |= {c.bend.bend_goal for c in self.settings.chains if c.bend.bend_goal}
        missing = sorted(n for n in targets | set(MOTION_IK_FRAMES) if not skel.has(n))
        if missing:
            raise KeyError(f"the skeleton has no {missing} for MotionIK")
        self.targets = sorted(targets)
        self.parents = {n: self._parent(n) for n in self.bones}
        #: the bones a frame is built for: the solver's and the targets', with their ancestors
        self.need = [skel.id(n) for n in _ancestor_closure(skel, set(self.bones) | targets)]
        self._rest: dict = {}
        self._bone: dict = {}
        #: the skeleton's ``rest_revision`` the two caches above were read at (:meth:`bind`)
        self._rest_revision = skel.rest_revision
        self.pose: dict = {}
        self.xf = np.eye(4)
        self.placed: dict = {}
        #: ``{joint: world position}`` (this project's frame) where a joint is to stand this frame instead of
        #: where the pose has it, for the goals that go with it (:meth:`goal_shift`); cleared by :meth:`bind`
        self.joint_places: dict = {}
        self._frame: _Frame | None = None
        self._stale = False

    def _link_names(self) -> list:
        """The ``cf_t_*`` the four limb pairs drive: effector targets and bend goals."""
        out = []
        for _part, e, c in PAIRS:
            out += [self.settings.effectors[e].target, self.settings.chains[c].bend.bend_goal]
        if any(n is None for n in out):
            raise ValueError("a limb effector has no target or bend goal")
        return out

    def _parent(self, name: str) -> str | None:
        p = self.skel.bones[self.skel.id(name)].parent
        return None if p < 0 else self.skel.bones[p].name

    def read_bone(self, name: str) -> None:
        """Have every frame cover *name*, a bone a partner's target sits on."""
        ids = {self.skel.id(n) for n in _ancestor_closure(self.skel, [name])}
        if not ids <= set(self.need):
            self.need = sorted(set(self.need) | ids)
            self._stale = True

    def bind(self, pose: dict, xf) -> None:
        """This frame's pose (changed in place) and the character's world transform."""
        self.pose, self.xf = pose, np.asarray(xf, dtype=np.float64)
        self.placed = {}
        self.joint_places = {}
        self._frame = None
        if self.skel.rest_revision != self._rest_revision:
            # a rest moved since the caches were read (the yure switch reshapes the bust's
            # ``cf_s_bust00_*``, ``reshape_live``): read rest locals again, or the hand targets
            # on ``k_f_mune00*`` sit where the old rest had them (2.5 mm on 1,12, §7.1225)
            self._rest.clear()
            self._bone.clear()
            self._rest_revision = self.skel.rest_revision

    def changed(self) -> None:
        """The pose was written outside this body (``Lookat_dan``, BetterPenetration): read it again."""
        self._stale = True

    def frame(self) -> _Frame:
        """This stage's :class:`_Frame`, built on first use and again after a write."""
        if self._frame is None or self._stale:
            self._frame = _Frame(self)
        self._stale = False
        return self._frame

    def local(self, i: int, entry: dict | None) -> tuple:
        """Bone *i*'s local ``(linear rows, translation, rotation)`` under its pose entry, as floats.

        A bone with no entry is at rest, computed once.  A plain TRS bone with
        a positive scale is composed here as ``math3d.compose`` does it; an
        authored or absolute matrix, or a scale that is not positive, goes
        through ``Skeleton.local_of``.
        """
        if entry is None:
            hit = self._rest.get(i)
            if hit is None:
                local = self.skel.local_of(i, {})
                hit = self._rest[i] = (*_affine(local), _rotation_of(local).tolist())
            return hit
        bone = self._bone.get(i)
        if bone is None:
            b = self.skel.bones[i]
            plain = b.authored_local is None and bool((np.asarray(b.local_scale) > 1e-9).all())
            bone = self._bone[i] = (plain, b.local_rot.tolist(), b.local_pos.tolist(),
                                    [float(c) for c in np.broadcast_to(b.local_scale, 3)])
        plain, lrot, lpos, (sx, sy, sz) = bone
        scale = entry.get("scale")
        if scale is not None:
            scale = np.asarray(scale, dtype=np.float64)
            ex, ey, ez = [float(scale)] * 3 if scale.ndim == 0 else scale.tolist()
            sx, sy, sz = ex * sx, ey * sy, ez * sz            # Skeleton.local_of: entry scale times the bone's
        if not plain or "matrix" in entry or not (sx > 1e-9 and sy > 1e-9 and sz > 1e-9):
            local = self.skel.local_of(i, entry)
            return (*_affine(local), _rotation_of(local).tolist())
        rot = entry.get("rot")
        off = entry.get("pos")
        r = _q_matrix(lrot if rot is None else _qmul(lrot, _f(rot)))
        t = lpos if off is None else [a + c for a, c in zip(lpos, _f(off))]
        return [[row[0] * sx, row[1] * sy, row[2] * sz] for row in r], t, _matrix_q(r)

    # -- reading the pose in Unity's frame ----------------------------------
    def body_rotation(self) -> np.ndarray:
        """``objBodyBone.transform.rotation`` in Unity's frame: the character's turn, then the prefab's."""
        return m3.quat_mul(mirror_q(_rotation_of(self.xf)), self.body_bone_rotation)

    def _bone_pose(self) -> dict:
        """The solver's bones' world ``(position, rotation)`` in Unity's frame, as float tuples (what ``load`` keeps)."""
        frame = self.frame()
        out = {}
        for name in self.bones:
            p, q = frame.unity(self.skel.id(name))
            out[name] = (tuple(p), tuple(q))
        return out

    def world(self, names) -> dict:
        """``{name: (position, rotation)}`` in Unity's frame, rotations composed as ``Transform`` does."""
        frame = self.frame()
        out = {}
        for name in names:
            p, q = frame.unity(self.skel.id(name))
            out[name] = (np.array(p), np.array(q))
        return out

    # -- 5500 / 10050 -------------------------------------------------------
    def correct(self, indexes) -> None:
        """``BaseProcess.LateUpdate``, type Target: ``localPosition += pos``, ``localRotation *= rot``."""
        k = float(getattr(self.skel, "rig_scale", 1.0) or 1.0)
        for idx in indexes:
            c = self.frames[idx]
            name = c.bone[1]
            if not c.enabled:
                continue
            entry = dict(self.pose.get(name) or {})
            entry["pos"] = np.asarray(entry.get("pos", np.zeros(3)), dtype=np.float64) + k * mirror_v(c.pos)
            entry["rot"] = m3.quat_mul(np.asarray(entry.get("rot", _IDENTITY), dtype=np.float64), mirror_q(c.rot))
            self.pose[name] = entry
            self._stale = True

    # -- 7000 ---------------------------------------------------------------
    def place_targets(self) -> None:
        """``BaseProcess.LateUpdate``, type Sync: each linked ``cf_t_*`` onto its bone (pre-IK), offset in the body's turn.

        The turn is ``objBodyBone``'s, and ``ChaControl`` finds that as
        ``FindLoop("cf_j_root")`` — the clip's root, turned by the clips that
        turn it (180° on 立ち正常位, 蓮華座), not the component's
        ``p_cf_body_bone``: with the latter the offsets came out mirrored on
        those rows, twice their length off (§7.1212).
        """
        turn = self.world(["cf_j_root"])["cf_j_root"][1]
        held = set()
        for name, link in self.links.items():
            if not link.enabled or link.bone is None:
                continue
            owner, bone = link.bone
            pos, rot = owner.world([bone])[bone]
            shift = owner.goal_shift(bone)
            if shift is not None:
                pos = pos + shift
                if owner is self and owner.held_goal(bone):
                    held.add(name)
            self.placed[name] = (pos + m3.quat_rotate(turn, link.pos), m3.quat_mul(rot, link.rot))
        self._keep_within_reach(held)

    def held_goal(self, goal: str) -> bool:
        """Whether *goal* goes with a joint given a place this frame (:attr:`joint_places`)."""
        joint = (getattr(self.skel, "goal_joints", None) or {}).get(goal)
        return joint is not None and joint in self.joint_places

    def _keep_within_reach(self, held: set) -> None:
        """A limb's target placed by a held joint (*held*) no farther from the limb's root than the limb reaches.

        A converted body's ankles are held where the clip put them while the
        body itself is moved after the corrections — rested as a card of its
        size, then come to her (``h_runtime.place_bare_feet`` /
        ``keep_his_feet``).  Out of reach, the solver does not leave the leg
        short: every limb chain pulls (``pull`` 1.0 in the prefab), and the
        whole body is dragged after the place, after the shaft was aimed at
        her — EndminM's hips 78–82 mm toward his feet in 騎乗位, the shaft 45–59°
        off her entrance; 19–38 mm down in 立ちフェラ, 15–36° off her mouth.
        So the target is brought along the root→place line to the limb's
        length (its two segments as they stand, a hair short of straight).
        A card holds no joint, and a target within reach is left alone (§7.1324).
        """
        if not held:
            return
        for _part, e_index, c_index in PAIRS:
            target = self.settings.effectors[e_index].target
            nodes = [n.bone for n in self.settings.chains[c_index].nodes]
            if target not in held or len(nodes) != 3:
                continue
            at = self.world(nodes)
            root, mid, end = (np.asarray(at[n][0], dtype=np.float64) for n in nodes)
            reach = (np.linalg.norm(mid - root) + np.linalg.norm(end - mid)) * REACH_SHORT
            pos, rot = self.placed[target]
            away = np.asarray(pos, dtype=np.float64) - root
            far = float(np.linalg.norm(away))
            if far > reach:
                self.placed[target] = (root + away * (reach / far), rot)

    def goal_shift(self, goal: str) -> np.ndarray | None:
        """How far *goal*'s joint stands from where the clip's own rig has it now (Unity's frame); None on a card.

        A converted character plays a clip at its own lengths
        (``Skeleton.clip_offsets``), but the clip's own goals (``cf_pv_*``)
        hang off the floor roots and stay where the clip put them: on the
        source game's joints at ``s`` times the reference rig, 36–86 mm from
        EndminF's hands and 32–52 mm from her ankles, a leg goal 1.06–1.14 of
        the leg's reach (design note §1.4).  So each goal goes with the joint
        it belongs to (``Skeleton.goal_joints``, ``converted_rig.remap_goals``):
        by that joint's place now less its place with every offset taken off
        the same pose — the clip's own rig, 5500's corrections and all.  A card
        has no goal table and its goals are read as the game reads them.  A
        joint given a place this frame (:attr:`joint_places`: a bare foot's
        ankle where the foot lies on the source game's body's,
        ``h_runtime.place_bare_feet``) is read there instead of where the pose has it.
        """
        joint = (getattr(self.skel, "goal_joints", None) or {}).get(goal)
        if joint is None:
            return None
        offsets = getattr(self.skel, "clip_offsets", None) or {}
        frame = self.frame()
        xm, xt = _affine(self.xf)
        chain = []
        i = self.skel.id(joint)
        while i >= 0:
            chain.append(i)
            i = self.skel.bones[i].parent
        pm, pt = xm, xt
        for i in reversed(chain):
            name = self.skel.bones[i].name
            lm, lt, _lq = self.local(i, self.pose.get(name))
            offset = offsets.get(name)
            if offset is not None:
                lt = [a - float(b) for a, b in zip(lt, offset)]
            pm, pt = _compose(pm, pt, lm, lt)
        place = self.joint_places.get(joint)
        at = frame.pos[self.skel.id(joint)] if place is None else [float(v) for v in place]
        moved = [a - b for a, b in zip(at, pt)]
        return np.array(_vmirror(moved))

    # -- 9999 ---------------------------------------------------------------
    def initiate(self) -> None:
        """``SolverManager.Start`` → ``IKSolver.Initiate`` on the pose bound now (see the module docstring).

        What ``Initiate`` keeps for good is each arm's and leg's bend
        direction (``IKConstraintBend.Initiate``): the upper bone's part square
        to the root→end line, in the upper bone's own frame.  In the game the
        body under every ``MotionIK`` is a card's, at the source rig's limb
        lengths (a card's upper arm and forearm 222 and 220 mm), and that is
        the shape the clips and the IK data were made on.  A converted
        character is that rig with its own lengths added as translations
        (``Skeleton.clip_offsets``); taken off (:meth:`clip_rig_pose`), the
        same frame is the card-shaped pose the game's ``Initiate`` reads, with
        the same turns.  Read at the character's own lengths it can come out
        the other way round: that part flips sign when the angle at the root
        between the upper bone and the root→end line passes 90°, which a
        folded limb only reaches when its lower bone is the longer one
        (EndminF's forearm 243 mm against her upper arm's 204).  Initiated so,
        the solver rolled her whole arm 179° each frame of 正常位 and held the
        hand on its target: the wrist twisted +147° against the clip's −32°
        (the card −35°), and only when a session began in such a row.  The
        solver then runs on the pose itself, its default local state read
        from it (``FixTransforms`` restores the character's lengths).
        """
        pose = self._bone_pose()
        source = self.clip_rig_pose()
        self.solver = FullBodyBipedIK(self.settings, self.parents, pose if source is None else source)
        if source is not None:
            self.solver.store_default_local_state(pose)

    def clip_rig_pose(self) -> dict | None:
        """The solver's bones as :meth:`_bone_pose` gives them, on the clip's own rig; None on a card.

        Every local translation less its ``Skeleton.clip_offsets`` entry — the
        rig at the source game's lengths, as :meth:`goal_shift` reads it — and
        the rotations as posed.  A skeleton without offsets is that rig already.
        """
        offsets = getattr(self.skel, "clip_offsets", None) or {}
        if not offsets:
            return None
        skel, pose = self.skel, self.pose
        xm, xt = _affine(self.xf)
        xq = _rotation_of(self.xf).tolist()
        lin: dict = {}
        pos: dict = {}
        rot: dict = {}
        for i in self.need:
            b = skel.bones[i]
            lm, lt, lq = self.local(i, pose.get(b.name))
            offset = offsets.get(b.name)
            if offset is not None:
                lt = [a - float(o) for a, o in zip(lt, offset)]
            pm, pt, pq = (xm, xt, xq) if b.parent < 0 else (lin[b.parent], pos[b.parent], rot[b.parent])
            lin[i], pos[i] = _compose(pm, pt, lm, lt)
            rot[i] = _qmul(pq, lq)
        return {name: (tuple(_vmirror(pos[skel.id(name)])), tuple(_qmirror(rot[skel.id(name)])))
                for name in self.bones}

    def fix_transforms(self) -> None:
        """``FixTransforms`` (Update) then the Animator: the mapped bones' channels the clip does not write take the default local state.

        A channel the clip writes is the clip's whatever FixTransforms left in
        it, so the pose is read only when some mapped channel is unwritten.
        """
        frame = None
        writes = {}
        for name, lp1, lq1 in self.solver.default_local_state():
            entry = self.pose.get(name) or {}
            open_rot, open_pos = "rot" not in entry, lp1 is not None and "pos" not in entry
            if not (open_rot or open_pos):
                continue
            frame = frame or self.frame()
            i, parent = self.skel.id(name), self.parents[name]
            lp0, lq0 = self._local_now(frame, i, parent)
            new = {}
            if open_rot and abs(_dot4(lq0, lq1.tolist())) < 1.0 - 1e-12:
                new["rot"] = self._rot_entry(i, lq1.tolist())
            if open_pos and _dist(lp0, lp1.tolist()) > 1e-9:
                pp, pq = self._parent_world(frame, parent)
                p = [a + b for a, b in zip(pp, _vmirror(_qrot(_qmirror(pq), lp1.tolist())))]
                new["pos"] = self._pos_entry(i, parent, p, pp, _qmirror(pq), frame)
            if new:
                writes[name] = {**entry, **new}
        self._apply(writes)

    def solve(self) -> None:
        """``IKSolver.Update`` with every target where 7000 put it (the rest where the pose has it)."""
        targets = self.world([n for n in self.targets if n not in self.placed])
        targets.update(self.placed)
        after = self.solver.update(self._bone_pose(), targets, self.body_rotation())
        frame = self.frame()
        writes = {}
        for name in self.mapped:
            i, parent = self.skel.id(name), self.parents[name]
            lp0, lq0 = self._local_now(frame, i, parent)
            p1, q1 = after[name][0].tolist(), after[name][1].tolist()
            if parent is None:
                pp, pq = self._parent_world(frame, None)
                pp, pq = _vmirror(pp), _qmirror(pq)
            else:
                pp, pq = after[parent][0].tolist(), after[parent][1].tolist()
            inv = _qconj(pq)
            lp1 = _qrot(inv, [a - b for a, b in zip(p1, pp)])
            lq1 = _qmul(inv, q1)
            entry = self.pose.get(name) or {}
            new = {}
            if abs(_dot4(lq0, lq1)) < 1.0 - 1e-12:
                new["rot"] = self._rot_entry(i, lq1)
            if _dist(lp0, lp1) > 1e-9:
                new["pos"] = self._pos_entry(i, parent, _vmirror(p1), _vmirror(pp), _qmirror(pq), frame)
            if new:
                writes[name] = {**entry, **new}
        self._apply(writes)

    # -- writing back -------------------------------------------------------
    def _parent_world(self, frame: _Frame, parent: str | None) -> tuple:
        """The parent's world ``(position, rotation)`` in this project's frame; the character's for a root."""
        if parent is None:
            return self.xf[:3, 3].tolist(), _rotation_of(self.xf).tolist()
        j = self.skel.id(parent)
        return frame.pos[j], frame.rot[j]

    def _local_now(self, frame: _Frame, i: int, parent: str | None) -> tuple:
        """Bone *i*'s local ``(position in the parent's turn, rotation)`` in Unity's frame, as ``fbbik._Transforms`` keeps them."""
        pp, pq = self._parent_world(frame, parent)
        inv = _qconj(_qmirror(pq))
        p = _vmirror(frame.pos[i])
        return _qrot(inv, [a - b for a, b in zip(p, _vmirror(pp))]), _qmul(inv, _qmirror(frame.rot[i]))

    def _rot_entry(self, i: int, local_unity) -> np.ndarray:
        """The ``rot`` entry that gives bone *i* the local rotation *local_unity* (Unity's frame)."""
        b = self.skel.bones[i]
        return m3.quat_normalize(m3.quat_mul(m3.quat_conj(b.local_rot), mirror_q(local_unity)))

    def _pos_entry(self, i: int, parent: str | None, p, pp, pq, frame: _Frame) -> np.ndarray:
        """The ``pos`` entry that puts bone *i* at *p* under a parent at *pp* turned *pq* (this project's frame).

        The parent keeps the scale it has now (its world columns' lengths).
        """
        b = self.skel.bones[i]
        rest = b.authored_local[:3, 3] if b.authored_local is not None else b.local_pos
        scale = np.linalg.norm(self.xf[:3, :3], axis=0) if parent is None else frame.scale(self.skel.id(parent))
        r = m3.quat_to_mat4(np.array(pq))[:3, :3] * scale
        return np.linalg.solve(r, np.array(p) - np.array(pp)) - np.asarray(rest, dtype=np.float64)

    def _apply(self, writes: dict) -> None:
        """Write the pose entries; the frame is built again when next read."""
        if writes:
            self.pose.update(writes)
            self._stale = True


# -- one MotionIK -------------------------------------------------------------
class MotionIK:
    """``MotionIK``: one character's loaded data and its partners (``partners[0]`` is itself)."""

    def __init__(self, body: Body, states: list | None):
        self.body, self.states = body, states
        self.partners: list = [self]

    def set_partners(self, cast: list) -> None:
        """``SetPartners``: self first, then the others in the order given."""
        self.partners = [self] + [m for m in cast if m is not self]

    def _partner(self, index: int) -> MotionIK:
        if not 0 <= index < len(self.partners):
            raise MotionIKError(f"partner {index} of a cast of {len(self.partners)}")
        return self.partners[index]

    def now_state(self, name: str) -> dict | None:
        """``GetNowState``: the first state of that exact name (ordinal), or None."""
        return next((s for s in self.states or () if s.get("name") == name), None)

    def _offsets(self, frame: dict | None) -> tuple:
        """``(pos, rot)`` of one frame: ``GetCorrectShapeValues`` on its ``editNo`` partner, the angle as ``Euler``."""
        if frame is None:
            return np.zeros(3), _IDENTITY.copy()
        pos, ang = shape_values(self._partner(int(frame["editNo"])).body.shape, frame.get("shapes") or [])
        return pos, euler(ang)

    def _find(self, index: int, name: str | None) -> tuple | None:
        """``LinkIK``'s ``getTarget``: the bone *name* on partner *index*, or None (an empty name too)."""
        if not name:
            return None
        owner = self._partner(index).body
        if not owner.skel.has(name):
            return None
        owner.read_bone(name)
        return owner, name

    def calc(self, name: str) -> None:
        """``MotionIK.Calc(stateName)``."""
        body = self.body
        for c in body.frames:
            c.reset(keep_bone=True)
        for c in body.links.values():
            c.reset(keep_bone=False)
        state = self.now_state(name)
        for fr in (state or {}).get("frames") or ():
            k = int(fr["frameNo"]) - LIMB_FRAMES
            if k < 0:
                continue
            if k >= len(body.frames):
                raise MotionIKError(f"frame {fr['frameNo']} past FrameCorrect's {len(body.frames)}")
            c = body.frames[k]
            c.enabled = True
            c.pos, c.rot = self._offsets(fr)
        body.ik_enabled = state is not None              # set_enabled: the component and IKCorrect
        for c in body.links.values():
            c.enabled = body.ik_enabled
        for index in range(len(PAIRS)):
            self.link_ik(index, state)

    def link_ik(self, index: int, state: dict | None) -> None:
        """``MotionIK.LinkIK``: one limb's target and bend goal, and their weights."""
        part_name, e_index, c_index = PAIRS[index]
        s = self.body.settings
        part = state["parts"][part_name] if state is not None else None
        frames = (state or {}).get("frames") or ()

        def frame(no: int) -> dict | None:
            return next((f for f in frames if int(f["frameNo"]) == no), None)

        eff = s.effectors[e_index]
        link = self.body.links[eff.target]
        link.bone = self._find(int(part["sex"]), part["target"]) if part is not None else None
        link.enabled = link.bone is not None
        link.pos, link.rot = self._offsets(frame(index * 2))
        if link.bone is None or part is None:
            eff.position_weight = eff.rotation_weight = 0.0
        else:
            eff.position_weight, eff.rotation_weight = float(part["weightPos"]), float(part["weightAng"])
        bend = s.chains[c_index].bend
        goal = self.body.links[bend.bend_goal]
        goal.bone = self._find(0, part["chain"]) if part is not None else None    # always self
        goal.enabled = goal.bone is not None
        goal.pos, goal.rot = self._offsets(frame(index * 2 + 1))
        bend.weight = 0.0 if goal.bone is None or part is None else float(part["weight"])


# -- the H scene -------------------------------------------------------------
def cast_order(mode: int, action_id: int, roles: dict) -> list:
    """``lstMotionIK``'s order for the roles present (``f``, ``m``, ``f1``, ``m1``).

    ``HSceneProc.Start``: female, male, the second woman, the second man.
    ``ChangeAnimator`` rebuilds it on list files 6 and 7 (the two-women 3P
    modes): ``[female0, male, female1]`` for an even action id,
    ``[female1, male, female0]`` for an odd one.
    """
    if mode in (6, 7):
        order = ("f", "m", "f1") if action_id % 2 == 0 else ("f1", "m", "f")
    else:
        order = ("f", "m", "f1", "m1")
    return [key for key in order if key in roles]


class Scene:
    """Every character's ``MotionIK`` in one H scene, and the per-frame order across them."""

    def __init__(self, fbbik: dict, files: dict):
        self.fbbik, self.files = fbbik, files
        self.bodies: dict = {}
        self.cast: list = []
        self.keys: list = []
        self.row = None
        self.played: str | None = None

    def body(self, skel, shape: Callable[[int], float | None]) -> Body:
        """The character's components, made once per skeleton."""
        key = id(skel)
        hit = self.bodies.get(key)
        if hit is None or hit.skel is not skel:
            hit = self.bodies[key] = Body(skel, shape, load_settings(self.fbbik),
                                          self.fbbik["body_bone_rotation"])
        hit.shape = shape
        return hit

    def change_animator(self, row, roles: dict) -> list:
        """``ChangeAnimator``: the cast in its order, each loaded with its role's file, then ``Calc("Idle")``.

        *row* is ``(mode, action id)``; *roles* maps ``f`` / ``m`` / ``f1`` /
        ``m1`` to ``(skeleton, shape reader, controller file)``.  Returns notes
        for the files the bake does not have.
        """
        mode, action_id = row
        notes = []
        cast = []
        keys = cast_order(mode, action_id, roles)
        for key in keys:
            skel, shape, file = roles[key]
            states = self.files.get(file)
            if states is None:
                notes.append(file)
            cast.append(MotionIK(self.body(skel, shape), states))
        for m in cast:
            m.set_partners(cast)
        self.cast, self.keys, self.row = cast, keys, row
        self.set_play("Idle")
        return notes

    def set_play(self, name: str) -> None:
        """``HActionBase.SetPlay``: every ``MotionIK`` re-binds to *name*."""
        self.played = name
        for m in self.cast:
            m.calc(name)

    def before(self, frames: dict) -> None:
        """Update and 5500: bind each role's ``(pose, transform)`` (*frames* by role key), initiate or fix, then the body corrections."""
        for key, m in zip(self.keys, self.cast):
            m.body.bind(*frames[key])
        for m in self.cast:
            b = m.body
            if not b.ik_enabled:
                continue
            if b.solver is None:
                b.initiate()
            else:
                b.fix_transforms()
        for m in self.cast:
            m.body.correct(BEFORE)

    def move(self, key: str, xf) -> None:
        """The role *key*'s body was moved after 5500 (``h_runtime.rest_where_a_card_would``).

        Its frame is read again.
        """
        if key not in self.keys:
            return
        body = self.cast[self.keys.index(key)].body
        body.xf = np.asarray(xf, dtype=np.float64)
        body.changed()

    def place_joints(self, key: str, places: dict) -> None:
        """Where the role *key*'s joints stand this frame (``{joint: world position}``, this project's frame).

        Its goals that go with a joint go with it there (``Body.goal_shift``);
        a role not in the scene is left alone.
        """
        if key not in self.keys:
            return
        self.cast[self.keys.index(key)].body.joint_places = dict(places)

    def after(self) -> None:
        """7000, 9999 and 10050: every target placed before any solve, then each solve, then the fingers."""
        for m in self.cast:
            m.body.changed()                    # Lookat_dan and BetterPenetration wrote since 5500
        for m in self.cast:
            if m.body.ik_enabled:
                m.body.place_targets()
        for m in self.cast:
            if m.body.ik_enabled:
                m.body.solve()
        for m in self.cast:
            m.body.correct(AFTER)


def load_scene() -> tuple:
    """``(Scene, None)`` from the two bakes, or ``(None, reason)`` when one is missing."""
    from . import hanim
    try:
        fbbik = json.loads(hanim.FBBIK_FILE.read_text(encoding="utf-8"))
        files = json.loads(hanim.MOTION_IK_FILE.read_text(encoding="utf-8")).get("files")
    except (OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(files, dict):
        return None, "motion_ik.json has no files"
    return Scene(fbbik, files), None
