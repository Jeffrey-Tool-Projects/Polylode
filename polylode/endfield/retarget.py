"""Play this project's own poses and clips on an Endfield actor's skeleton.

Every pose, clip and state machine in ``polylode/anim`` is written for the
``cf_j_*`` humanoid.  An Endfield actor has a different skeleton (3ds Max Biped
names, its own rest pose, hundreds of extra bones), and the actor's Avatar says
which of its nodes is which human bone (``Actor.extra["humanoid"]``).  This
module carries a pose across that table.

**How.**  The pose is evaluated on the project's own scaffold skeleton
(``mesh.skeleton.build_humanoid`` — always available, no converted Koikatsu
data involved).  For each human bone both rigs have, the scaffold's world
rotation *away from its bind* is applied to the actor's bone in world space::

    target world = source turn · alignment · target rest

and the result is re-expressed as the local deltas ``Skeleton.evaluate`` takes.
Because the turn is a world-space one, it means the same thing on both rigs
whatever their local axes are.

**The scaffold binds in a T-pose** (:data:`SOURCE_ARM_DEG`).  The library's
poses are local euler angles composed on the standing rest, so what an angle
does depends on where the bone's frame was at bind; they are authored for a
rig that binds its arms straight out with world-aligned joints, and the
scaffold's default 42° would turn an elbow flexion into a twist.

**Alignment** exists for the arms only.  A turn carried onto a limb that rests
in a different direction lands somewhere else — the scaffold binds its arms
straight out, an actor may bind them at any angle — so the upper and lower arm
are first swung (shortest arc) onto the scaffold's bind direction, and the hand
is carried with the forearm.  The spine, the legs and the feet keep the actor's
own rest shape: the scaffold stands straight on flat feet, and aligning to that
would straighten a rig's spine curve and flatten a heeled shoe.

The hips' displacement is scaled by the ratio of hip heights.  Fingers are not
carried: the actors' Avatars do not map them.
"""
from __future__ import annotations

import numpy as np

from ..anim.clip import Pose
from ..core import math3d as m3
from ..mesh.skeleton import Proportions, Skeleton, build_humanoid

#: Unity human bone → the scaffold's bone.  ``_L`` is the character's left on
#: both sides of the table (X is mirrored out of Unity's space on import).
SOURCE_BONES = {
    "Hips": "cf_j_hips", "Spine": "cf_j_spine01", "Chest": "cf_j_spine02",
    "UpperChest": "cf_j_spine03", "Neck": "cf_j_neck", "Head": "cf_j_head",
    "LeftShoulder": "cf_j_shoulder_L", "RightShoulder": "cf_j_shoulder_R",
    "LeftUpperArm": "cf_j_arm00_L", "RightUpperArm": "cf_j_arm00_R",
    "LeftLowerArm": "cf_j_forearm01_L", "RightLowerArm": "cf_j_forearm01_R",
    "LeftHand": "cf_j_hand_L", "RightHand": "cf_j_hand_R",
    "LeftUpperLeg": "cf_j_thigh00_L", "RightUpperLeg": "cf_j_thigh00_R",
    "LeftLowerLeg": "cf_j_leg01_L", "RightLowerLeg": "cf_j_leg01_R",
    "LeftFoot": "cf_j_leg03_L", "RightFoot": "cf_j_leg03_R",
    "LeftToes": "cf_j_toes_L", "RightToes": "cf_j_toes_R",
}
#: Human bone → the bone its segment ends at, for the bones that are aligned.
_AIMED = {"LeftUpperArm": "LeftLowerArm", "LeftLowerArm": "LeftHand",
          "RightUpperArm": "RightLowerArm", "RightLowerArm": "RightHand"}
#: Human bone → the bone whose alignment it shares.
_CARRIED = {"LeftHand": "LeftLowerArm", "RightHand": "RightLowerArm"}
HIPS = "Hips"
#: The scaffold's bind arm angle from vertical: straight out.
SOURCE_ARM_DEG = 90.0


class RetargetError(ValueError):
    """The actor's humanoid table cannot be driven (no hips, or a missing node)."""


def source_skeleton() -> Skeleton:
    """The scaffold the project's poses are evaluated on before being carried over."""
    return build_humanoid(Proportions(arm_angle=SOURCE_ARM_DEG))


def _rotation(world: np.ndarray) -> np.ndarray:
    """The rotation of a world matrix as a quaternion, scale divided out."""
    return m3.decompose(world)[1]


def _direction(world: np.ndarray, a: int, b: int) -> np.ndarray | None:
    d = world[b][:3, 3] - world[a][:3, 3]
    n = float(np.linalg.norm(d))
    return None if n < 1e-9 else d / n


class Retarget:
    """A scaffold skeleton and the per-bone constants that carry its poses over.

    *target* is the actor's skeleton and *humanoid* its ``human bone → node``
    table.  :attr:`source` is the skeleton the project's poses and clips are
    evaluated on; :attr:`nodes` maps each carried scaffold bone to the actor's.

    *alike* (``bone → human bone``) gives further bones the alignment of
    another.  *source* and *names* (``human bone → source bone``) replace the scaffold
    with another skeleton that binds its arms straight out — the stand-in an
    H clip is sampled on (``endfield/hrole.py``).  *turns* (``human bone →
    source bone``) takes a bone's turn from another source bone than the one
    it stands at: a source whose joint below turns the part the human bone is.
    """

    def __init__(self, target: Skeleton, humanoid: dict,
                 source: Skeleton | None = None, names: dict | None = None, alike: dict | None = None,
                 turns: dict | None = None):
        if HIPS not in humanoid:
            raise RetargetError("the actor's Avatar maps no Hips bone")
        missing = [node for node in humanoid.values() if not target.has(node)]
        if missing:
            raise RetargetError(f"humanoid nodes not in the skeleton: {missing[:3]}")
        self.target = target
        names = SOURCE_BONES if names is None else names
        self.source = source_skeleton() if source is None else source
        self.bones = [b for b in names if b in humanoid]
        self.nodes = {names[b]: humanoid[b] for b in self.bones}
        src = {b: self.source.index[names[b]] for b in self.bones}
        tgt = {b: target.index[humanoid[b]] for b in self.bones}
        source_bind = self.source.evaluate({})
        target_rest = target.evaluate({})
        aim = {}
        for bone, tip in _AIMED.items():
            if bone not in tgt or tip not in tgt:
                continue
            to = _direction(source_bind, src[bone], src[tip])
            frm = _direction(target_rest, tgt[bone], tgt[tip])
            if to is not None and frm is not None:
                aim[bone] = m3.quat_from_to(frm, to)
        # *alike* adds bones that share another's alignment (fingers: their hand's — they bind
        # along the arm, and the two skeletons bind their arms at different angles)
        for bone, like in {**_CARRIED, **(alike or {})}.items():
            if like in aim:
                aim[bone] = aim[like]
        #: target bone index → ``alignment · rest world rotation``
        self._reference = {
            tgt[b]: m3.quat_mul(aim.get(b, m3.quat_identity()),
                                _rotation(target_rest[tgt[b]]))
            for b in self.bones}
        spin = {b: src[b] for b in self.bones}
        for bone, name in (turns or {}).items():
            if bone in spin and self.source.has(name):
                spin[bone] = self.source.index[name]
        self._source_of = {tgt[b]: spin[b] for b in self.bones}
        self._source_rows = np.array(sorted(set(spin.values())), dtype=np.int64)
        #: human bone → (source bone index, target bone index)
        self.pairs = {b: (src[b], tgt[b]) for b in self.bones}
        #: source bone index → the inverse of its bind world rotation
        self._unbind = {spin[b]: m3.quat_conj(_rotation(source_bind[spin[b]]))
                        for b in self.bones}
        self._rest_local = [np.asarray(b.local_rot, dtype=np.float64)
                            for b in target.bones]
        # only the mapped bones and their ancestors need a world rotation
        chain: set[int] = set()
        for index in self._reference:
            while index >= 0 and index not in chain:
                chain.add(index)
                index = target.bones[index].parent
        self._chain = sorted(chain)
        self._hips = (src[HIPS], tgt[HIPS])
        self._source_bind_hips = source_bind[src[HIPS]][:3, 3].copy()
        self._hip_ratio = float(target_rest[tgt[HIPS]][1, 3]
                                / self._source_bind_hips[1])
        # the hips' ``pos`` is added in its parent's frame, scale included
        parent = target.bones[tgt[HIPS]].parent
        above = target_rest[parent][:3, :3] if parent >= 0 else np.eye(3)
        self._into_hips_parent = np.linalg.inv(above)

    def apply(self, pose: Pose) -> Pose:
        """*pose* (scaffold bone names) as a pose for the actor's skeleton."""
        return self.carry(self.source.evaluate(pose))

    def carry(self, source: np.ndarray) -> Pose:
        """The source skeleton's world matrices *source* as a pose for the actor's."""
        world: dict[int, np.ndarray] = {}
        out: Pose = {}
        bones = self.target.bones
        # every mapped source bone's rotation at once (a per-bone decomposition was most of this)
        turned = dict(zip(self._source_rows.tolist(), m3.quats_from_matrices(source[self._source_rows])[0]))
        for index in self._chain:
            above = world.get(bones[index].parent, m3.quat_identity())
            reference = self._reference.get(index)
            if reference is None:
                world[index] = m3.quat_mul(above, self._rest_local[index])
                continue
            i = self._source_of[index]
            turn = m3.quat_mul(turned[i], self._unbind[i])
            world[index] = m3.quat_normalize(m3.quat_mul(turn, reference))
            local = m3.quat_mul(m3.quat_conj(above), world[index])
            out[bones[index].name] = {"rot": m3.quat_normalize(
                m3.quat_mul(m3.quat_conj(self._rest_local[index]), local))}
        source_hips, target_hips = self._hips
        moved = (source[source_hips][:3, 3] - self._source_bind_hips) * self._hip_ratio
        out[bones[target_hips].name]["pos"] = self._into_hips_parent @ moved
        return out
