"""Unity humanoid muscle curves → bone transforms, by the Avatar's own definition.

A humanoid clip does not store the trunk and limbs as bone rotations.  It
stores *muscles*: one number per degree of freedom, −1‥1 across the range the
Avatar gives that joint, plus where the body's mass centre is and how it is
turned.  Turning those back into bones takes the Avatar's human definition
(:func:`.components.human_definition`): per bone a pre- and a post-rotation, a
sign per axis and a limit per axis.

What is fixed by the serialized data, and was checked against it (§7.568):

* a bone's three numbers are (roll about its own length, then the two swings),
  and its limits, signs and frames are the Avatar's ``m_AxesArray`` entry;
* the body's frame is rebuilt from the pose itself — the mass centre (each
  bone's mass at the middle of its length) and an orientation from the two
  hips and the two shoulders, **measured from the Avatar's default pose** —
  and the hips are then placed so that frame lands on the clip's root curve.
  The orientation rule reproduces every converted Avatar's own stored root
  rotation to 3e-7;
* a limb's roll is shared with the next bone down by the Avatar's twist
  settings, read as the share the bone *keeps*: under that reading the rig's
  own twist helper bones follow the result (the forearm's at one and two
  thirds of the hand's roll, correlation 1.00), under the other they do not;
* the limb goals a clip carries (where each hand and foot is in the body's
  frame, and how it is turned) are not used to pose anything; they are what
  the result is checked against (``tools/ef_muscle_check.py``): on all seven
  body types the solved limbs turn exactly as the goals do.

**What is not exact**: where along the trunk's bones their mass sits was not
identified.  The rule here leaves a few millimetres of every Avatar's stored
root position unexplained; that remainder is carried as a constant in the
body's frame, calibrated on the Avatar's stored root (:attr:`Human.calibrated`),
which the clips bear out to about a millimetre.

Everything in this module is in Unity's space, as serialized: left-handed, a
quaternion as ``xyzw``.  :func:`project_pose` is the one place it is carried
over into this project's.
"""
from __future__ import annotations

import numpy as np

from .components import HUMAN_BONES

#: A clip's muscle curves, by index: 7 for the motion node, 7 for the body's
#: root, 7 for each of the four goals, then the body's degrees of freedom.
MOTION, ROOT, GOALS = 0, 7, 14
BODY_DOF = 55
#: Which curve holds each of the 55 degrees of freedom, in the order
#: :data:`BONE_DOF` numbers them.  **This game's build lays them out with three
#: more curves after each leg** (71‥73 and 82‥84) than the engine's public
#: muscle list has; read off the curves themselves (§7.568): the two leg blocks
#: and the two arm blocks repeat with the same statistics at these offsets and
#: at no others.  What the six extra curves drive has not been read — they stay
#: within a few percent of zero — and nothing here uses them.
CURVE = np.array([*range(42, 51),        # spine, chest, upper chest
                  *range(51, 63),        # neck, head, eyes, jaw
                  *range(63, 71),        # left leg
                  *range(74, 82),        # right leg
                  *range(85, 94),        # left arm
                  *range(94, 103)],      # right arm
                 dtype=np.int64)
#: The four limb goals, in curve order, as the human bone each one follows.
GOAL_BONES = ("LeftFoot", "RightFoot", "LeftHand", "RightHand")

_BODY, _HEAD, _L_LEG, _R_LEG, _L_ARM, _R_ARM = 0, 9, 21, 29, 37, 46


def _leg(start: int) -> dict:
    return {"UpperLeg": (start + 2, start + 1, start), "LowerLeg": (start + 4, -1, start + 3),
            "Foot": (-1, start + 6, start + 5), "Toes": (-1, -1, start + 7)}


def _arm(start: int) -> dict:
    return {"Shoulder": (-1, start + 1, start), "UpperArm": (start + 4, start + 3, start + 2),
            "LowerArm": (start + 6, -1, start + 5), "Hand": (-1, start + 8, start + 7)}


def _bone_dof() -> np.ndarray:
    """Per human bone: the degree of freedom behind (roll, second axis, third axis)."""
    table = {"Hips": (-1, -1, -1),
             "Spine": (_BODY + 2, _BODY + 1, _BODY), "Chest": (_BODY + 5, _BODY + 4, _BODY + 3),
             "UpperChest": (_BODY + 8, _BODY + 7, _BODY + 6),
             "Neck": (_HEAD + 2, _HEAD + 1, _HEAD), "Head": (_HEAD + 5, _HEAD + 4, _HEAD + 3),
             "LeftEye": (-1, _HEAD + 7, _HEAD + 6), "RightEye": (-1, _HEAD + 9, _HEAD + 8),
             "Jaw": (-1, _HEAD + 11, _HEAD + 10)}
    for side, leg, arm in (("Left", _L_LEG, _L_ARM), ("Right", _R_LEG, _R_ARM)):
        table.update({side + name: dofs for name, dofs in _leg(leg).items()})
        table.update({side + name: dofs for name, dofs in _arm(arm).items()})
    return np.array([table[name] for name in HUMAN_BONES], dtype=np.int64)


BONE_DOF = _bone_dof()
#: (parent bone, child bone, the Avatar's setting): how a limb's roll is shared
#: between the bone that owns the muscle and the next one down.
TWISTS = (("UpperArm", "LowerArm", "m_ArmTwist"), ("LowerArm", "Hand", "m_ForeArmTwist"),
          ("UpperLeg", "LowerLeg", "m_UpperLegTwist"), ("LowerLeg", "Foot", "m_LegTwist"))


# -- quaternions, xyzw, over leading axes ----------------------------------------
def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def qconj(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[..., :3], q[..., 3:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _unit(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _from_frame(right: np.ndarray, up: np.ndarray, front: np.ndarray) -> np.ndarray:
    """The rotation whose columns are the three unit axes, over leading axes."""
    m = np.stack([right, up, front], axis=-1)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    out = np.empty(m.shape[:-2] + (4,))
    # the four branches of the usual conversion, picked per element by the
    # largest of (w, x, y, z) so the divisor never goes to zero
    w4 = 1.0 + trace
    x4 = 1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]
    y4 = 1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2]
    z4 = 1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]
    pick = np.argmax(np.stack([w4, x4, y4, z4], axis=-1), axis=-1)
    a, b, c = m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]
    d, e, f = m[..., 1, 0] + m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 2, 1] + m[..., 1, 2]
    rows = (np.stack([a, b, c, w4], axis=-1), np.stack([x4, d, e, a], axis=-1),
            np.stack([d, y4, f, b], axis=-1), np.stack([e, f, z4, c], axis=-1))
    for branch, row in enumerate(rows):
        mask = pick == branch
        out[mask] = row[mask]
    return _unit(out)


def _q(d: dict) -> list:
    return [d["X"], d["Y"], d["Z"], d["W"]]


def _v(d: dict) -> list:
    return [d["X"], d["Y"], d["Z"]]


class MuscleError(ValueError):
    """A human definition this module cannot pose (a joint type it has not read)."""


class Human:
    """One Avatar's human definition, ready to turn muscle values into a pose."""

    #: the only joint parametrisation the converted Avatars use: two swings,
    #: then a roll about the bone's own length
    SWING_THEN_ROLL = 1
    #: whether the mass centre carries the left forearm's offset, and whether
    #: it is calibrated on the Avatar's stored root (both switchable so the
    #: measurements that found them can be repeated)
    forearm_term = True
    calibrated = True

    def __init__(self, definition: dict, nodes: list):
        skeleton = definition["m_Skeleton"]
        self.nodes = list(nodes)
        self.parent = np.array([n["m_ParentId"] for n in skeleton["m_Node"]], dtype=np.int64)
        axes_of = np.array([n["m_AxesId"] for n in skeleton["m_Node"]], dtype=np.int64)
        axes = skeleton["m_AxesArray"]
        kinds = {a["m_Type"] for a in axes}
        if kinds - {self.SWING_THEN_ROLL}:
            raise MuscleError(f"joint types {sorted(kinds)}: only {self.SWING_THEN_ROLL} was read")
        count = len(self.parent)
        pose = definition["m_SkeletonPose"]["m_X"]
        self.rest_t = np.array([_v(x["t"]) for x in pose], dtype=np.float64)
        self.rest_q = np.array([_q(x["q"]) for x in pose], dtype=np.float64)
        self.pre = np.tile([0.0, 0.0, 0.0, 1.0], (count, 1))
        self.post = self.pre.copy()
        self.sign = np.ones((count, 3))
        self.low = np.zeros((count, 3))
        self.high = np.zeros((count, 3))
        self.length = np.zeros(count)
        for node, index in enumerate(axes_of):
            if index < 0:
                continue
            entry = axes[index]
            self.pre[node], self.post[node] = _q(entry["m_PreQ"]), _q(entry["m_PostQ"])
            self.sign[node] = _v(entry["m_Sgn"])
            self.low[node] = _v(entry["m_Limit"]["m_Min"])
            self.high[node] = _v(entry["m_Limit"]["m_Max"])
            self.length[node] = entry["m_Length"]
        self.has_axes = axes_of >= 0
        #: human bone → node of the human skeleton, −1 when the rig has none
        self.bone = np.array(definition["m_HumanBoneIndex"], dtype=np.int64)
        self.mass = np.array(definition["m_HumanBoneMass"], dtype=np.float64)
        self.scale = float(definition["m_Scale"])
        self.twist = {key: float(definition[key]) for _p, _c, key in TWISTS}
        self._order = self._depth_first()
        #: the nodes above the hips (the skeleton's own root), which no pose
        #: moves, and where the definition's pose has every node
        self._above_hips = self._ancestors(int(self.bone[0]))
        rest_t, rest_q = self.globals(self.rest_q[None].copy())
        self._rest_world = (rest_t[0], rest_q[0])
        #: the body's orientation in the Avatar's default pose: what a pose's
        #: orientation is measured from
        self.rest_turn = _unit(np.array(_q(definition["m_RootX"]["q"]), dtype=np.float64))
        self._offset = np.zeros(3)
        if self.calibrated:
            # what the rule above leaves of the Avatar's own stored root, in
            # its default pose, carried as a constant in the body's frame
            t, q = self.globals(self.rest_q[None].copy())
            centre, turn = self.body_frame(t, q)
            stored = np.array(_v(definition["m_RootX"]["t"]), dtype=np.float64)
            self._offset = qrot(qconj(turn[0]), stored - centre[0])

    def _depth_first(self) -> list:
        """Nodes with every parent before its children (the file's order is)."""
        if np.any(self.parent >= np.arange(len(self.parent))):
            raise MuscleError("human skeleton lists a child before its parent")
        return list(range(len(self.parent)))

    def _ancestors(self, node: int) -> list:
        out = []
        node = int(self.parent[node])
        while node >= 0:
            out.append(node)
            node = int(self.parent[node])
        return out

    def node(self, name: str) -> int:
        """The human-skeleton node of human bone *name*, −1 when absent."""
        return int(self.bone[HUMAN_BONES.index(name)])

    # -- muscles → local rotations ---------------------------------------------
    def _angles(self, dof: np.ndarray) -> np.ndarray:
        """Per frame and node: the three joint angles, radians, signs applied."""
        frames = dof.shape[0]
        value = np.zeros((frames, len(self.parent), 3))
        for bone, node in enumerate(self.bone):
            if node < 0:
                continue
            for axis, index in enumerate(BONE_DOF[bone]):
                if index >= 0:
                    value[:, node, axis] = dof[:, index]
        # a muscle runs 0‥1 over the positive limit and 0‥−1 over the negative
        return np.where(value < 0.0, -value * self.low, value * self.high) * self.sign

    def locals(self, dof: np.ndarray) -> np.ndarray:
        """``(frames, 55)`` muscle values → ``(frames, nodes, 4)`` local rotations.

        A node without an axes entry keeps the definition's own pose.  (The
        hips get their axes' rest like any bone; where the body is and how it
        is turned is settled afterwards, from the root curve.)  The limb rolls
        are shared out at the end (:meth:`_share_twist`).
        """
        angles = self._angles(np.asarray(dof, dtype=np.float64))
        half = np.tan(0.5 * np.clip(angles, -np.pi + 1e-6, np.pi - 1e-6))
        x, y, z = half[..., 0], half[..., 1], half[..., 2]
        # the swing about y and z, then the roll about x, as one rotation
        joint = _unit(np.stack([x, y + x * z, z - x * y, np.ones_like(x)], axis=-1))
        out = qmul(qmul(self.pre, joint), qconj(self.post))
        keep = ~self.has_axes
        out[:, keep] = self.rest_q[keep]
        self._share_twist(out, half[..., 0])
        return out

    def _share_twist(self, local: np.ndarray, half_roll: np.ndarray) -> None:
        """Move part of a limb bone's roll onto the next bone down, in place.

        A bone's rotation is its swing followed by a roll about its own
        length; the Avatar's twist setting is the share of that roll the bone
        keeps, and the rest is given to its child as a turn about the same
        axis, so the child ends up where it was.
        """
        for side in ("Left", "Right"):
            for parent_name, child_name, key in TWISTS:
                parent, child = self.node(side + parent_name), self.node(side + child_name)
                if parent < 0 or child < 0:
                    continue
                keep = self.twist[key]
                if keep == 1.0:
                    continue
                angle = 2.0 * np.arctan(half_roll[:, parent])
                moved = self._roll(parent, angle * (1.0 - keep))
                local[:, parent] = qmul(local[:, parent], qconj(moved))
                local[:, child] = qmul(moved, local[:, child])

    def _roll(self, node: int, angle: np.ndarray) -> np.ndarray:
        """A roll of *angle* about *node*'s own length, in its local frame."""
        roll = np.zeros(angle.shape + (4,))
        roll[..., 0], roll[..., 3] = np.sin(0.5 * angle), np.cos(0.5 * angle)
        return qmul(qmul(self.post[node], roll), qconj(self.post[node]))

    # -- the pose in the definition's space ---------------------------------------
    def globals(self, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Local rotations → every node's position and rotation, hips at rest."""
        frames = local.shape[0]
        t = np.zeros((frames, len(self.parent), 3))
        q = np.zeros((frames, len(self.parent), 4))
        for node in self._order:
            parent = self.parent[node]
            if parent < 0:
                t[:, node], q[:, node] = self.rest_t[node], local[:, node]
                continue
            t[:, node] = t[:, parent] + qrot(q[:, parent], self.rest_t[node])
            q[:, node] = qmul(q[:, parent], local[:, node])
        return t, q

    def body_frame(self, t: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The body's own frame in a pose: its mass centre and its orientation.

        The centre is each bone's mass at the middle of its length.  The
        orientation is built from the two hip joints and the two shoulder
        joints — up runs from between the hips to between the shoulders, right
        is the sum of hip-to-hip and shoulder-to-shoulder — and is then taken
        relative to the same construction on the Avatar's default pose, so
        that pose's own orientation is "not turned".
        """
        centre = np.zeros(t.shape[:1] + (3,))
        for bone, node in enumerate(self.bone):
            if node < 0 or not self.mass[bone]:
                continue
            centre += self.mass[bone] * (t[:, node] + 0.5 * self._along(q, node))
        if self.forearm_term:
            # the left forearm's mass sits half an upper-arm length back up the
            # left upper arm; the right one does not (measured, see the class)
            upper = self.node("LeftUpperArm")
            mass = self.mass[HUMAN_BONES.index("LeftLowerArm")]
            if upper >= 0:
                centre -= mass * 0.5 * self._along(q, upper)
        # (on every converted Avatar the hips bone sits exactly between the two
        # hip joints, so "between the hip joints" and "the hips" cannot be told
        # apart by the data; this is the first)
        hip_l, hip_r = t[:, self.node("LeftUpperLeg")], t[:, self.node("RightUpperLeg")]
        arm_l, arm_r = t[:, self.node("LeftUpperArm")], t[:, self.node("RightUpperArm")]
        up = _dir(0.5 * (arm_l + arm_r) - 0.5 * (hip_l + hip_r))
        right = _dir((hip_r - hip_l) + (arm_r - arm_l))
        front = _dir(np.cross(right, up))
        right = np.cross(up, front)
        turn = qmul(_from_frame(right, up, front), qconj(self.rest_turn))
        return centre + qrot(turn, self._offset), turn

    def _along(self, q: np.ndarray, node: int) -> np.ndarray:
        """*node*'s own length as a vector, per frame."""
        axis = qrot(qmul(q[:, node], self.post[node]), np.array([1.0, 0.0, 0.0]))
        return axis * self.length[node]

    def solve(self, dof: np.ndarray, root_t: np.ndarray,
              root_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Muscles and the body's root → every node's position and rotation.

        *root_t* is in the clip's normalised units (a body of height 1) and is
        scaled by this Avatar's size here.  The skeleton's own root stays where
        the definition has it: it is the hips that carry the body there.
        """
        t, q = self.globals(self.locals(dof))
        centre, turn = self.body_frame(t, q)
        delta = qmul(_unit(np.asarray(root_q, dtype=np.float64)), qconj(turn))[:, None]
        target = np.asarray(root_t, dtype=np.float64)[:, None] * self.scale
        t, q = target + qrot(delta, t - centre[:, None]), qmul(delta, q)
        above = self._above_hips
        t[:, above], q[:, above] = self._rest_world[0][above], self._rest_world[1][above]
        return t, q

    def goals(self, t: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Where the four goal bones sit in the body's frame: ``(frames, 4, 3)``.

        In the clip's normalised units — what a clip's goal curves hold for the
        pose it was made from, and so what a solved pose is checked against.
        A hand's goal is its wrist; a foot's is the far end of the foot bone
        (the sole under the ankle), which is what the curves turned out to hold:
        measured at the ankle every foot is off by exactly that bone's length.
        """
        centre, turn = self.body_frame(t, q)
        points = []
        for name in GOAL_BONES:
            node = self.node(name)
            point = t[:, node]
            if name.endswith("Foot"):
                point = point + self._along(q, node)
            points.append(point)
        points = np.stack(points, axis=1)
        return qrot(qconj(turn)[:, None], points - centre[:, None]) / self.scale


    def goal_turns(self, t: np.ndarray, q: np.ndarray) -> np.ndarray:
        """How the four goal bones are turned in the body's frame: ``(frames, 4, 4)``.

        A goal's rotation is its bone's own axes frame seen from the body's —
        on a character whose clips were made on its own Avatar the curves hold
        exactly this, to the last digit they store.
        """
        _centre, turn = self.body_frame(t, q)
        nodes = [self.node(name) for name in GOAL_BONES]
        return qmul(qconj(turn)[:, None], qmul(q[:, nodes], self.post[nodes]))

    def relative(self, t: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A world pose as each node's position and rotation in its parent's frame."""
        local_t, local_q = t.copy(), q.copy()
        for node, parent in enumerate(self.parent):
            if parent < 0:
                continue
            inverse = qconj(q[:, parent])
            local_t[:, node] = qrot(inverse, t[:, node] - t[:, parent])
            local_q[:, node] = qmul(inverse, q[:, node])
        return local_t, local_q


def _dir(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def curves(columns: dict) -> dict | None:
    """Where a clip keeps what :meth:`Human.solve` reads, or None when it does not.

    *columns* maps a muscle curve's index to its column in the clip's scalar
    samples.  The result has ``dof`` (55 columns), ``root`` and ``motion`` (7
    each: position, then rotation); ``motion`` is ``None`` for a clip without
    a motion node.  A clip that lacks any body curve has no body to solve.
    """
    need = [*range(ROOT, ROOT + 7), *CURVE.tolist()]
    if any(index not in columns for index in need):
        return None
    motion = [columns[i] for i in range(MOTION, MOTION + 7)] \
        if all(i in columns for i in range(MOTION, MOTION + 7)) else None
    return {"dof": [columns[int(i)] for i in CURVE],
            "root": [columns[i] for i in range(ROOT, ROOT + 7)], "motion": motion}


def in_place(root_t: np.ndarray, root_q: np.ndarray, motion_t: np.ndarray,
             motion_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The body's root seen from the clip's motion node.

    A clip that travels carries the travel on its motion node; seen from that
    node the body stays where the clip was authored to play.
    """
    inverse = qconj(_unit(motion_q))
    return qrot(inverse, root_t - motion_t), qmul(inverse, _unit(root_q))


#: Unity → this project (X mirrored): quaternion (x, −y, −z, w), vector (−x, y, z).
_Q_FLIP = np.array([1.0, -1.0, -1.0, 1.0])
_V_FLIP = np.array([-1.0, 1.0, 1.0])


def project_pose(local_t: np.ndarray, local_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A pose in Unity's space as the same pose in this project's."""
    return local_t * _V_FLIP, local_q * _Q_FLIP
