"""Built-in pose and animation library, generated procedurally.

Poses are authored as per-bone euler triples in degrees (YXZ order, matching the
rig's bind orientation).  Clips are built from those poses or from analytic
sine curves so the project ships no motion-capture data.

Everything here is authored against the **standing rest** below, not against the
rig's bind pose.  See :func:`standing_rest`.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import math3d as m3
from ..core.i18n import N_
from .clip import AnimationClip, Pose

# --------------------------------------------------------------------------
# static poses: bone -> (rx, ry, rz) degrees
# --------------------------------------------------------------------------
POSES: dict[str, dict] = {
    "stand": {},
    "relaxed": {
        "cf_j_shoulder_L": (0, 0, -3), "cf_j_shoulder_R": (0, 0, 3), "cf_j_spine01": (2, 0, 0),
        "cf_j_head": (-3, 0, 0), "cf_j_arm00_L": (1, -6, -7), "cf_j_forearm01_L": (0, -13, 0),
        "cf_j_arm00_R": (1, 6, 7), "cf_j_forearm01_R": (0, 13, 0),
    },
    "hands_behind": {
        "cf_j_spine02": (-4, 0, 0), "cf_j_head": (-6, 0, 0), "cf_j_arm00_L": (70, 98, 72),
        "cf_j_forearm01_L": (0, -80, 0), "cf_j_arm00_R": (70, -98, -72),
        "cf_j_forearm01_R": (0, 80, 0),
    },
    "arms_crossed": {
        "cf_j_shoulder_L": (0, 0, -8), "cf_j_shoulder_R": (0, 0, 8),
        "cf_j_arm00_L": (67, -44, -3), "cf_j_forearm01_L": (0, -104, 0),
        "cf_j_arm00_R": (72, 37, 3), "cf_j_forearm01_R": (0, 102, 0),
    },
    "hands_on_hips": {
        "cf_j_spine01": (0, 0, 0), "cf_j_arm00_L": (47, 75, 76),
        "cf_j_forearm01_L": (0, -94, 0), "cf_j_arm00_R": (47, -75, -76),
        "cf_j_forearm01_R": (0, 94, 0),
    },
    "wave": {
        "cf_j_hand_L": (0, 0, -12), "cf_j_shoulder_L": (0, 0, -12), "cf_j_head": (-2, -6, 0),
        "cf_j_arm00_L": (1, -78, 90), "cf_j_forearm01_L": (0, -84, 0),
    },
    "peace": {
        "cf_j_hand_L": (0, 0, -8), "cf_j_middle01_L": (0, 0, -78),
        "cf_j_middle02_L": (0, 0, -70), "cf_j_ring01_L": (0, 0, -80),
        "cf_j_ring02_L": (0, 0, -72), "cf_j_little01_L": (0, 0, -82),
        "cf_j_little02_L": (0, 0, -74), "cf_j_head": (-4, -4, 4),
        "cf_j_arm00_L": (-12, -57, 47), "cf_j_forearm01_L": (0, -118, 0),
    },
    "point": {
        "cf_j_middle01_L": (0, 0, -84), "cf_j_ring01_L": (0, 0, -86),
        "cf_j_little01_L": (0, 0, -88), "cf_j_thumb01_L": (0, 0, -40),
        "cf_j_arm00_L": (15, -74, 7), "cf_j_forearm01_L": (0, -8, 0),
    },
    "shy": {
        "cf_j_head": (10, -8, 4), "cf_j_spine02": (5, 0, 0), "cf_j_thigh00_L": (0, 6, 0),
        "cf_j_thigh00_R": (0, -2, 0), "cf_j_arm00_L": (47, -5, -5),
        "cf_j_forearm01_L": (0, -68, 0), "cf_j_arm00_R": (47, 5, 5),
        "cf_j_forearm01_R": (0, 68, 0),
    },
    "think": {
        "cf_j_head": (6, 10, -4), "cf_j_arm00_L": (4, -41, -7),
        "cf_j_forearm01_L": (0, -132, 0), "cf_j_arm00_R": (61, 41, 26),
        "cf_j_forearm01_R": (0, 93, 0),
    },
    "sit": {
        "cf_j_hips": (0, 0, 0), "cf_j_thigh00_L": (-86, 3, 0), "cf_j_thigh00_R": (-86, -3, 0),
        "cf_j_leg01_L": (84, 0, 0), "cf_j_leg01_R": (84, 0, 0), "cf_j_leg03_L": (6, 0, 0),
        "cf_j_leg03_R": (6, 0, 0), "cf_j_spine01": (4, 0, 0), "cf_j_arm00_L": (45, -12, 7),
        "cf_j_forearm01_L": (0, -59, 0), "cf_j_arm00_R": (45, 12, -7),
        "cf_j_forearm01_R": (0, 59, 0),
    },
    "sit_floor": {
        "cf_j_thigh00_L": (-135, 12, 0), "cf_j_thigh00_R": (-135, -12, 0),
        "cf_j_leg01_L": (100, 0, 0), "cf_j_leg01_R": (100, 0, 0), "cf_j_spine01": (-6, 0, 0),
        "cf_j_spine02": (-4, 0, 0), "cf_j_arm00_L": (-1, 20, -7), "cf_j_forearm01_L": (0, 0, 0),
        "cf_j_arm00_R": (-1, -20, 7), "cf_j_forearm01_R": (0, 0, 0),
    },
    "kneel": {
        "cf_j_thigh00_L": (-75, 4, 0), "cf_j_thigh00_R": (-75, -4, 0),
        "cf_j_leg01_L": (165, 0, 0), "cf_j_leg01_R": (165, 0, 0), "cf_j_leg03_L": (60, 0, 0),
        "cf_j_leg03_R": (60, 0, 0), "cf_j_arm00_L": (68, -44, -19),
        "cf_j_forearm01_L": (0, -18, 0), "cf_j_arm00_R": (68, 44, 19),
        "cf_j_forearm01_R": (0, 18, 0),
    },
    "lie_back": {
        "cf_j_root": (-90, 0, 0), "cf_j_arm00_L": (32, 32, 27), "cf_j_forearm01_L": (0, -29, 0),
        "cf_j_arm00_R": (32, -32, -27), "cf_j_forearm01_R": (0, 30, 0),
    },
    "cheer": {
        "cf_j_head": (-10, 0, 0), "cf_j_spine02": (-6, 0, 0), "cf_j_arm00_L": (20, -12, 111),
        "cf_j_forearm01_L": (0, 0, 0), "cf_j_arm00_R": (20, 12, -111),
        "cf_j_forearm01_R": (0, 0, 0),
    },
    "bow": {
        "cf_j_spine01": (28, 0, 0), "cf_j_spine02": (16, 0, 0), "cf_j_spine03": (10, 0, 0),
        "cf_j_head": (12, 0, 0), "cf_j_arm00_L": (49, -50, -12),
        "cf_j_forearm01_L": (0, -35, 0), "cf_j_arm00_R": (48, 50, 12),
        "cf_j_forearm01_R": (0, 35, 0),
    },
    "run_pose": {
        "cf_j_thigh00_L": (-46, 0, 0), "cf_j_thigh00_R": (28, 0, 0), "cf_j_leg01_L": (54, 0, 0),
        "cf_j_leg01_R": (18, 0, 0), "cf_j_spine01": (10, 0, 0), "cf_j_arm00_L": (20, 67, 7),
        "cf_j_forearm01_L": (0, -71, 0), "cf_j_arm00_R": (-13, 17, 5),
        "cf_j_forearm01_R": (0, 107, 0),
    },
    "guard": {
        "cf_j_spine01": (8, 0, 0), "cf_j_head": (4, 0, 0), "cf_j_arm00_L": (-10, -43, -4),
        "cf_j_forearm01_L": (0, -114, 0), "cf_j_arm00_R": (-10, 42, 4),
        "cf_j_forearm01_R": (0, 114, 0),
    },
    "stretch": {
        "cf_j_spine02": (-12, 0, 0), "cf_j_head": (-16, 0, 0), "cf_j_arm00_L": (58, -34, 121),
        "cf_j_forearm01_L": (0, -20, 0), "cf_j_arm00_R": (58, 34, -121),
        "cf_j_forearm01_R": (0, 20, 0),
    },
    "lean": {
        "cf_j_spine01": (0, 0, 8), "cf_j_spine02": (0, 0, 6), "cf_j_head": (0, -6, -6),
        "cf_j_thigh00_L": (0, 0, 4), "cf_j_thigh00_R": (0, 0, 2), "cf_j_arm00_L": (33, 11, -2),
        "cf_j_forearm01_L": (0, -30, 0), "cf_j_arm00_R": (35, -73, -77),
        "cf_j_forearm01_R": (0, 106, 0),
    },
}

#: Built-in poses whose support is not two standing feet: played with the body's
#: own lowest point put on the floor (``anim.ground``, §7.470).  ``sit`` is on
#: a seat as high as her knees — the feet reach the floor — which is what the
#: pose's vertical shins describe.
GROUNDED_POSES = frozenset({"sit", "sit_floor", "kneel", "lie_back"})
#: Clip tags with the same meaning (``sit_idle``; the FSM's ``sit`` state).
GROUNDED_TAGS = frozenset({"sitting"})

POSE_LABELS = {
    "stand": N_("直立"), "relaxed": N_("放鬆"), "hands_behind": N_("背手"), "arms_crossed": N_("抱胸"),
    "hands_on_hips": N_("叉腰"), "wave": N_("揮手"), "peace": N_("比 YA"), "point": N_("指向"),
    "shy": N_("害羞"), "think": N_("思考"), "sit": N_("坐椅"), "sit_floor": N_("坐地"),
    "kneel": N_("跪坐"), "lie_back": N_("仰躺"), "cheer": N_("歡呼"), "bow": N_("鞠躬"),
    "run_pose": N_("奔跑"), "guard": N_("防禦"), "stretch": N_("伸展"), "lean": N_("倚靠"),
}


def euler_pose(spec: dict, order: str = "YXZ") -> Pose:
    """Convert a {bone: (rx, ry, rz) degrees} spec into a Pose."""
    out: Pose = {}
    for bone, e in spec.items():
        out[bone] = {"rot": m3.quat_from_euler(*np.deg2rad(np.asarray(e, dtype=float)),
                                               order=order)}
    return out


def hand_relax(amount: float = 1.0, sides=("L", "R")) -> Pose:
    """Curl the fingers so the default hand isn't a rake.

    **The same local angle on both sides.**  Origin's right finger bones are the
    left ones mirrored *and* turned 180° about X (bind frames of
    ``cf_j_index01_L`` / ``_R``: rows 2 and 3 negated), so a curl of +θ about
    one side's local Z mirrors to +θ about the other's.  A negated right side —
    written for the procedural rig §7.36 deleted — bent the right knuckles
    backwards: measured on the reference armature in ``relaxed``, index / middle
    knuckles 18.0° / 13.3° against the left's 28.1° / 29.2° and the fingertips
    pointing out (mirrored x +0.47 / +0.57 against +0.12), which is the splayed
    right hand on every card (§7.538).  The arms are **not** like that — their
    abduction does flip sign (``tools/pose_author.py --axes``) — so this is a
    fact about the finger frames, not a rule for every bone.
    """
    spec = {}
    for side in sides:
        for finger, base in (("index", -20), ("middle", -24), ("ring", -28),
                             ("little", -32), ("thumb", -14)):
            for seg, mul in ((1, 1.0), (2, 1.05), (3, 0.75)):
                spec[f"cf_j_{finger}0{seg}_{side}"] = (0, 0, base * mul * amount)
    return euler_pose(spec)


# --------------------------------------------------------------------------
# standing rest
# --------------------------------------------------------------------------
# Where the arms hang when no pose and no clip is playing, in degrees away from
# straight down, measured in the frontal plane.  Every pose and clip in this
# file adds to *this*, not to the rig's bind pose.
#
# It has to be applied at runtime because the bind pose is not ours to change:
# imported meshes are skinned against Illusion's rest skeleton, which is a
# T-pose, and baking the drop into the bind would tear every sleeve off its arm
# (``origin.apply.prepare_for_assets`` keeps ``apose=False`` for exactly that
# reason).  Before this existed the drop was simply missing -- §7.36 deleted the
# procedural body, whose bind *was* an A-pose, and every pose and clip has been
# playing on a T-posed character since.
#
# Both are read off ``reference/FullBody.png`` -- the game's own idle -- by
# fitting the bare arm's centreline **above and below the elbow separately**
# (§7.64).  Fitting it as one line, which is how §7.49 arrived at the previous
# pair, hides the shape of the error: the whole limb averages 13.3 deg on that
# frame, but its two halves are 12.4 and 14.5, and ours were 22 and 11.  One
# number cannot say "the upper arm is ten degrees too open while the forearm is
# three too tight".
#
# These are the *rest*, and the idle clip takes 5.1 deg off the upper arm and
# 18.8 off the forearm, so what has to land on the reference is the composed
# result: 12.9 / 11.9 deg upper and 14.2 / 13.2 forearm against the game's
# 12.3 / 12.5 and 14.8 / 14.1 -- inside the left-right spread of either frame.
#
# The rest is applied at runtime because the bind pose is not ours to change:
# imported meshes are skinned against Illusion's rest skeleton, which is a
# T-pose, and baking the drop into the bind would tear every sleeve off its arm
# (``origin.apply.prepare_for_assets`` keeps ``apose=False`` for exactly that
# reason).  Before this existed the drop was simply missing -- §7.36 deleted the
# procedural body, whose bind *was* an A-pose, and every pose and clip has been
# playing on a T-posed character since.
#
# Every authored pose composes on top of these two numbers, so moving them moves
# all twenty.  That is the point -- one pair rather than twenty edits -- but it
# also means a change here is only done once the whole set has been rendered and
# looked at (``POSES`` plus ``idle``).
ARM_REST_DEG = 18.0        # upper arm, from vertical
FOREARM_REST_DEG = 33.0    # forearm, from vertical

# (joint, the joint that marks the end of the segment, target angle).  Named
# rather than "first child": Illusion hangs twist helpers and skin joints off
# the arm too, and which of those comes first is an artefact of the bundle's
# child order, not something to aim a limb by.
_REST_TARGETS = (("cf_j_arm00_{side}", "cf_j_forearm01_{side}", ARM_REST_DEG),
                 ("cf_j_forearm01_{side}", "cf_j_hand_{side}", FOREARM_REST_DEG))


#: ``pose.arm_angle``, the maker's 「手臂張開」 slider, in degrees per unit (§7.377).
#: The same span the slider already has on the own-body scaffold
#: (``mesh/body.py``: 8‥48 degrees over 0‥1), anchored so the schema default
#: lands on the measured :data:`ARM_REST_DEG` — an untouched character stands
#: exactly as before.  The forearm keeps its measured bend relative to the
#: upper arm (``FOREARM_REST_DEG − ARM_REST_DEG``): the slider opens the arms,
#: it does not straighten the elbows.
ARM_SLIDER_DEG_PER_UNIT = 40.0


def rest_targets(params: dict | None = None):
    """:data:`_REST_TARGETS` with the upper-arm angle taken from *params*.

    ``None`` or a missing key is the schema default, i.e. the measured rest.
    """
    from ..chara import params as P
    default = next(float(d.default) for d in P.SCHEMA if d.key == "pose.arm_angle")
    v = default
    if params is not None:
        try:
            v = float(params.get("pose.arm_angle", default))
        except (TypeError, ValueError):
            v = default
    arm = ARM_REST_DEG + (v - default) * ARM_SLIDER_DEG_PER_UNIT
    fore = arm + (FOREARM_REST_DEG - ARM_REST_DEG)
    (a_joint, a_tip, _), (f_joint, f_tip, _) = _REST_TARGETS
    return ((a_joint, a_tip, arm), (f_joint, f_tip, fore))


def _frontal_angle(d: np.ndarray) -> float:
    """Signed angle of *d* away from straight down, in the XY plane (radians).

    Positive is toward +X, which is the character's left.
    """
    return math.atan2(float(d[0]), -float(d[1]))


def standing_rest(skeleton, targets=_REST_TARGETS) -> Pose:
    """The pose that takes *skeleton*'s bind arms down to the standing rest.

    Returned as a normal :class:`Pose`, so it composes with authored poses
    through :func:`compose_rest` and needs no special case anywhere else.

    The angles are absolute -- "this far from vertical" -- rather than "rotate
    by this much", so a rig that already hangs its arms lands on the same place
    instead of being dropped twice.  Each joint is solved against the direction
    it actually has once its parent has been moved, which is why this walks the
    chain instead of writing both rotations at once.
    """
    rest: Pose = {}
    if skeleton is None:
        return rest
    for side, sign in (("L", 1.0), ("R", -1.0)):
        for joint, tip, target_deg in targets:
            i = skeleton.index.get(joint.format(side=side), -1)
            j = skeleton.index.get(tip.format(side=side), -1)
            if i < 0 or j < 0:
                continue
            world = skeleton.evaluate(rest)
            d = world[j][:3, 3] - world[i][:3, 3]
            if float(np.linalg.norm(d[:2])) < 1e-6:
                continue
            turn = sign * math.radians(target_deg) - _frontal_angle(d)
            rest[joint.format(side=side)] = {
                "rot": _local_turn(skeleton, i, world, turn)}
    return rest


def _local_turn(skeleton, i: int, world: np.ndarray, angle: float) -> np.ndarray:
    """A local rotation for bone *i* worth turning it *angle* about world Z.

    ``Skeleton.evaluate`` multiplies pose rotations on the *right* of the bind
    local rotation, so a world-space turn has to be carried into the bone's own
    frame -- through the parent's world rotation and then the bone's own bind
    rotation.  Doing it in world space keeps the number meaningful ("the arm
    hangs 38 degrees out") instead of depending on how Illusion happened to
    orient that joint.
    """
    bone = skeleton.bones[i]
    p = bone.parent
    # decompose, not quat_from_mat4: shape_update leaves non-uniform scale on
    # the spine and shoulders, and reading a quaternion straight off a scaled
    # basis skews the axis
    parent_rot = (m3.decompose(world[p])[1] if p >= 0 else m3.quat_identity())
    q = m3.quat_from_axis_angle((0.0, 0.0, 1.0), angle)
    inner = m3.quat_mul(m3.quat_mul(m3.quat_conj(parent_rot), q), parent_rot)
    return m3.quat_normalize(
        m3.quat_mul(m3.quat_mul(m3.quat_conj(bone.local_rot), inner), bone.local_rot))


def compose_rest(rest: Pose, pose: Pose) -> Pose:
    """*pose* played on top of *rest* -- the rest first, then the pose.

    Bones the pose does not mention keep the rest; bones it does get the two
    rotations composed in that order, which is what "authored against the
    standing rest" means.  ``pos`` and ``scale`` entries pass through untouched:
    the rest only ever rotates.
    """
    if not rest:
        return pose
    # **Copies of the rest's entries, not the entries.**  Whatever runs after
    # this edits the result in place — ``IKRig.solve`` sets ``pose[bone]["rot"]``
    # — and a bone the pose does not mention used to hand back the rest's own
    # dict, so a hand goal on the standing pose rewrote ``CharacterController.
    # rest`` every frame and the arm stayed where the IK left it after the goal
    # was cleared (§7.468).
    out: Pose = {bone: dict(entry) for bone, entry in rest.items()}
    for bone, entry in pose.items():
        base = rest.get(bone)
        if base is None:
            out[bone] = entry
            continue
        merged = dict(entry)
        q = entry.get("rot")
        merged["rot"] = (base["rot"] if q is None
                         else m3.quat_normalize(m3.quat_mul(base["rot"], q)))
        out[bone] = merged
    return out


def override_rotations(pose: Pose, over: Pose) -> Pose:
    """*pose* with the rotation of every bone in *over* replaced, not composed.

    For a writer that sets ``localRotation`` outright after everything else —
    Studio's FK (``FKCtrl.LateUpdate``, §7.518).  ``pos`` / ``scale`` of the
    bone stay what *pose* had; the entries are copies, so the caller's dicts
    are not edited.
    """
    out: Pose = dict(pose)
    for bone, entry in over.items():
        q = entry.get("rot")
        if q is None:
            continue
        merged = dict(out.get(bone) or {})
        merged["rot"] = q
        out[bone] = merged
    return out


#: How far :func:`hand_relax` curls a pose's fingers (0.85 when not listed).
#: ``point`` / ``peace`` then override the fingers they shape; ``guard`` is a
#: pair of fists, which 0.85 of a relaxed curl read as open hands over the face.
HAND_CURL = {"point": 0.9, "peace": 0.9, "guard": 3.2}


def get_pose(name: str, relax_hands: bool = True) -> Pose:
    """A built-in pose with the fingers curled (unless *relax_hands* is False)."""
    p = euler_pose(POSES.get(name, {}))
    if not relax_hands:
        return p
    h = hand_relax(HAND_CURL.get(name, 0.85))
    h.update(p)
    return h


def pose_names() -> list[str]:
    return list(POSES)


# --------------------------------------------------------------------------
# procedural clips
# --------------------------------------------------------------------------
def _sine_curve(clip: AnimationClip, bone: str, samples: int, duration: float,
                fn) -> None:
    """fn(phase 0..1) -> (rx, ry, rz) in degrees."""
    t = np.linspace(0.0, duration, samples)
    eul = np.array([fn(x / duration) for x in t])
    clip.set_euler_curve(bone, t, eul)


def make_idle(name: str = "idle", duration: float = 4.0, amount: float = 1.0) -> AnimationClip:
    c = AnimationClip(name, duration, loop=True)
    n = 17
    tau = 2 * math.pi
    _sine_curve(c, "cf_j_spine01", n, duration,
                lambda p: (1.2 * amount * math.sin(tau * p), 0, 0.6 * amount * math.sin(tau * p + 1.0)))
    _sine_curve(c, "cf_j_spine03", n, duration,
                lambda p: (-0.9 * amount * math.sin(tau * p), 0.8 * amount * math.sin(tau * p * 0.5), 0))
    _sine_curve(c, "cf_j_head", n, duration,
                lambda p: (-1.6 * amount * math.sin(tau * p + 0.6),
                           2.2 * amount * math.sin(tau * p * 0.5 + 1.2), 0))
    for side, sgn in (("L", -1), ("R", 1)):
        _sine_curve(c, f"cf_j_arm00_{side}", n, duration,
                    lambda p, s=sgn: (1.4 * amount * math.sin(tau * p + 0.4), 0,
                                      s * (5.0 + 1.6 * amount * math.sin(tau * p))))
        _sine_curve(c, f"cf_j_forearm01_{side}", n, duration,
                    lambda p, s=sgn: (0, 0, s * (12.0 + 2.2 * amount * math.sin(tau * p + 0.9))))
    c.tags = {"idle", "loop"}
    return c


def make_breathe(name: str = "breathe", duration: float = 3.4) -> AnimationClip:
    """Additive layer: subtle chest rise."""
    c = AnimationClip(name, duration, loop=True)
    n = 13
    tau = 2 * math.pi
    _sine_curve(c, "cf_j_spine02", n, duration, lambda p: (-1.1 * math.sin(tau * p), 0, 0))
    _sine_curve(c, "cf_j_spine03", n, duration, lambda p: (-0.8 * math.sin(tau * p), 0, 0))
    for side, sgn in (("L", -1), ("R", 1)):
        _sine_curve(c, f"cf_j_shoulder_{side}", n, duration,
                    lambda p, s=sgn: (0, 0, s * 1.2 * math.sin(tau * p)))
    c.tags = {"additive", "loop"}
    return c


def make_walk(name: str = "walk", duration: float = 1.05, stride: float = 1.0) -> AnimationClip:
    c = AnimationClip(name, duration, loop=True)
    n = 25
    tau = 2 * math.pi
    for side, ph in (("L", 0.0), ("R", math.pi)):
        _sine_curve(c, f"cf_j_thigh00_{side}", n, duration,
                    lambda p, q=ph: (-26 * stride * math.sin(tau * p + q), 0, 0))
        _sine_curve(c, f"cf_j_leg01_{side}", n, duration,
                    lambda p, q=ph: (max(0.0, 34 * stride * math.sin(tau * p + q + 1.9)) + 6, 0, 0))
        _sine_curve(c, f"cf_j_leg03_{side}", n, duration,
                    lambda p, q=ph: (10 * stride * math.sin(tau * p + q + 2.7), 0, 0))
    # Arm swing is ``ry`` on ``cf_j_arm00`` and elbow bend ``ry`` on
    # ``cf_j_forearm01`` — after the standing rest, with the left arm's
    # forward / flexion negative and the right's positive (``tools/pose_author.py
    # --axes``).  This swung on ``rx`` and bent on ``rz``, which after the rest
    # are a twist and a sideways fold: the walk's arms turned in place.
    for side, ph, sgn in (("L", math.pi, -1), ("R", 0.0, 1)):
        _sine_curve(c, f"cf_j_arm00_{side}", n, duration,
                    lambda p, q=ph, s=sgn: (0, s * 22 * stride * math.sin(tau * p + q), s * 6))
        _sine_curve(c, f"cf_j_forearm01_{side}", n, duration,
                    lambda p, q=ph, s=sgn: (0, s * (18 + 10 * max(0.0, math.sin(tau * p + q))), 0))
    _sine_curve(c, "cf_j_hips", n, duration,
                lambda p: (0, 3.5 * math.sin(tau * p), 2.4 * math.sin(tau * p * 2 + 0.4)))
    _sine_curve(c, "cf_j_spine01", n, duration, lambda p: (3, -2.6 * math.sin(tau * p), 0))
    c.add_event(0.02, "footstep_L")
    c.add_event(duration * 0.5, "footstep_R")
    c.tags = {"locomotion", "loop"}
    return c


def make_run(name: str = "run", duration: float = 0.68) -> AnimationClip:
    c = make_walk(name, duration, stride=1.75)
    extra = AnimationClip(name, duration, loop=True)
    n = 21
    tau = 2 * math.pi
    _sine_curve(extra, "cf_j_spine01", n, duration, lambda p: (11, -3.5 * math.sin(tau * p), 0))
    _sine_curve(extra, "cf_j_head", n, duration, lambda p: (-7, 0, 0))
    c.curves.update(extra.curves)
    c.tags = {"locomotion", "loop"}
    return c


def make_wave(name: str = "wave", duration: float = 1.6) -> AnimationClip:
    c = AnimationClip(name, duration, loop=True)
    n = 21
    tau = 2 * math.pi
    # The ``wave`` pose, with the forearm swinging on its hinge: the upper arm
    # is out to the side and the forearm up, so the elbow's hinge points
    # forward and flexing it is the side-to-side of a wave.
    held = POSES["wave"]
    for bone, e in held.items():
        if bone not in ("cf_j_forearm01_L", "cf_j_hand_L", "cf_j_head"):
            _sine_curve(c, bone, n, duration, lambda p, e=e: e)
    fore = held["cf_j_forearm01_L"]
    _sine_curve(c, "cf_j_forearm01_L", n, duration,
                lambda p: (fore[0], fore[1] + 20 * math.sin(tau * p * 2), fore[2]))
    _sine_curve(c, "cf_j_hand_L", n, duration, lambda p: (0, 0, -12 + 16 * math.sin(tau * p * 2)))
    _sine_curve(c, "cf_j_head", n, duration, lambda p: (-3, -5, 2 * math.sin(tau * p)))
    c.tags = {"gesture", "loop"}
    return c


def make_nod(name: str = "nod", duration: float = 0.9, shake: bool = False) -> AnimationClip:
    c = AnimationClip(name, duration, loop=False)
    n = 15
    tau = 2 * math.pi
    if shake:
        _sine_curve(c, "cf_j_head", n, duration, lambda p: (0, 14 * math.sin(tau * p * 2), 0))
    else:
        _sine_curve(c, "cf_j_head", n, duration, lambda p: (13 * math.sin(tau * p * 2), 0, 0))
    c.tags = {"gesture"}
    return c


def make_jump(name: str = "jump", duration: float = 0.9) -> AnimationClip:
    c = AnimationClip(name, duration, loop=False)
    n = 19
    t = np.linspace(0, duration, n)
    p = t / duration
    crouch = np.where(p < 0.25, np.sin(p / 0.25 * math.pi) * 26, 0.0)
    air = np.clip((p - 0.3) / 0.4, 0, 1)
    for side in ("L", "R"):
        c.set_euler_curve(f"cf_j_thigh00_{side}",
                          t, np.stack([-crouch - air * 24, np.zeros(n), np.zeros(n)], axis=-1))
        c.set_euler_curve(f"cf_j_leg01_{side}",
                          t, np.stack([crouch * 2 + air * 30, np.zeros(n), np.zeros(n)], axis=-1))
    for side, sgn in (("L", -1), ("R", 1)):
        # forward and up on ``ry`` (left negative), not ``rx`` — see make_walk
        c.set_euler_curve(f"cf_j_arm00_{side}",
                          t, np.stack([np.zeros(n), sgn * (air * 120 - crouch),
                                       np.full(n, sgn * 8.0)], axis=-1))
    pos = np.stack([np.zeros(n), np.sin(np.clip((p - 0.22) / 0.62, 0, 1) * math.pi) * 0.28,
                    np.zeros(n)], axis=-1)
    c.set_curve("cf_j_root", t, poss=pos)
    c.add_event(duration * 0.25, "jump_launch")
    c.add_event(duration * 0.88, "jump_land")
    c.tags = {"action"}
    return c


def make_sit_idle(name: str = "sit_idle", duration: float = 4.5) -> AnimationClip:
    c = AnimationClip(name, duration, loop=True)
    n = 15
    tau = 2 * math.pi
    base = POSES["sit"]
    for bone, e in base.items():
        _sine_curve(c, bone, n, duration,
                    lambda p, b=np.asarray(e, dtype=float): tuple(
                        b + np.array([0.8 * math.sin(tau * p), 0.5 * math.sin(tau * p * 0.7), 0])))
    c.tags = {"idle", "loop", "sitting"}
    return c


def make_from_pose(name: str, pose_key: str, duration: float = 0.5) -> AnimationClip:
    """A one-frame clip holding a static pose (used as state-machine targets)."""
    c = AnimationClip(name, duration, loop=True)
    for bone, e in POSES.get(pose_key, {}).items():
        c.set_euler_curve(bone, [0.0, duration], [e, e])
    c.tags = {"static"}
    return c


def build_default_library() -> dict[str, AnimationClip]:
    lib = {
        "idle": make_idle(),
        "idle_soft": make_idle("idle_soft", 5.2, 0.6),
        "breathe": make_breathe(),
        "walk": make_walk(),
        "run": make_run(),
        "wave": make_wave(),
        "nod": make_nod(),
        "shake": make_nod("shake", 0.9, shake=True),
        "jump": make_jump(),
        "sit_idle": make_sit_idle(),
    }
    for key in ("relaxed", "arms_crossed", "hands_on_hips", "shy", "think",
                "cheer", "bow", "guard", "stretch", "lean", "kneel", "peace"):
        lib[f"pose_{key}"] = make_from_pose(f"pose_{key}", key)
    return lib


CLIP_LABELS = {
    "idle": N_("待機"), "idle_soft": N_("待機（輕）"), "breathe": N_("呼吸（疊加）"),
    "walk": N_("行走"), "run": N_("奔跑"), "wave": N_("揮手"), "nod": N_("點頭"), "shake": N_("搖頭"),
    "jump": N_("跳躍"), "sit_idle": N_("坐姿待機"),
}
