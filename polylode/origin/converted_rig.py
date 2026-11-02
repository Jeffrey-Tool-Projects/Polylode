"""The source game's rig re-placed onto a converted character (PROGRESS #169, CLAUDE.md 四之二 §7.1209).

A converted character (Endfield, GF2, Wuthering Waves) has its own skeleton.
Everything that reads an H clip, aims the shaft, solves MotionIK or hangs a
collider asks for the source game's bone **names and hierarchy**
(``rig.ReferenceRig``), so a converted character gets that rig, fitted to its
body, with its own non-human bones kept under it.

This module is the computation and the stored form (:func:`to_payload` /
:func:`from_payload`), and the rebinding of a model drawn on the character's
own skeleton onto the rig (:class:`Rebound`); which store a table lives in
and when it is current is the source's (``endfield.origin_rig``).  Given

* the reference rig as a :class:`Skeleton` (``rig.graft_full_skeleton(…, 1.0)``),
* the character's own skeleton at its rest,
* its humanoid table (Unity human bone → its node) and the table from human
  bone to the reference joint the H path already uses (``endfield.hrole.H_BONES``),
* the finger pairs and the landmarks its certified body fit already placed
  (``hrole.finger_pairs``; the ``Origin_body_<name>`` nodes of ``body.helper_nodes``),

:func:`build` places every reference bone by one recorded rule:

=================  ===========================================================================
``anchor``         a human bone or finger: on the character's own joint
``landmark``       a joint the body fit placed: on that node
``joint-peer``     an unplaced correction driver beside a human joint at rest: follows its fitted spot
``coincident``     an IK bone standing on a reference joint at rest: on that joint's fitted spot
``root``           the bones above ``cf_n_height`` (``hanim.UNSCALED``): ``s ×`` reference
``pelvis-carried`` a bone hanging straight off the floor roots (``cf_j_hips``, ``cf_pv_elbo_*``…):
                   ``s ×`` reference plus the pelvis anchor's displacement — the rule
                   ``endfield.body.fit_body`` uses for every pelvis influence
``carried``        everything else: rigid in its parent's fitted frame, reference local × ``s``
``mouth``          with *mouth* given, the top of each run the head carries: moved as one so the
                   reference mouth stands at the middle of the character's lips (:func:`build`)
=================  ===========================================================================

``s`` is the character's hip height over the reference pelvis joint's — the
ratio the H path already sizes a converted body by (``Retarget._hip_ratio``).

Two frames per bone (design note §1.2):

* the **rest** (what the skeleton evaluates with no pose) has the reference
  rig's local rotations — a T-pose, like a card's — and the character's
  lengths along the reference's directions;
* the **bind** (what the inverse bind inverts) is the character's own rest:
  each limb bone turned by the shortest arc from the reference segment to the
  character's (inherited down the chain; the trunk below the neck is not
  turned, :data:`HUMAN_CHAINS`).  The mesh is not touched: posed by
  :meth:`ConvertedRig.bind_pose`, every skinning matrix is the identity.

The trunk stands on the pelvis as the reference's does: the rest puts the
first joint of the spine (:data:`TRUNK_ROOT`'s) over the pelvis at the
reference's front-back and side place, at the character's size, and keeps
only its height from the character.  Placed by the character's own joint it
carried that rig's stance into every pose: Texas's generated rig has its
spine 25 mm behind where the reference's stands over the pelvis, Lappland's
15 mm, and in every pose their upper bodies stood that far back, leaning
(§7.1440); the Endfield rigs have theirs 3.5-12 mm in front.  The bind keeps
the character's joint (the mesh does not move at the bind,
:meth:`ConvertedRig.bind_pose`), so a pose straightens the trunk over the pelvis.

Under an H clip the rotations are absolute, so only the translations need
the character: each bone carries ``offset = rest local − s · reference local``,
which ``hanim.absolute_pose_to_additive`` adds to the clip's translation
(``Skeleton.clip_offsets``) — the ABMX length/position form, KKABMX
``BoneModifier.Apply``.  IK goals (``cf_pv_*`` …) carry none: the clip's own
goals are moved with the joint they stand on instead (:func:`remap_goals`).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from ..core import math3d as m3
from ..mesh.skeleton import Skeleton, _inverse_bind
from .hanim import UNSCALED

#: The IK bone families ``rig.ReferenceRig.body_bone_names`` keeps: the clips' goals (``cf_pv_*``),
#: ``FullBodyBipedIK``'s effector targets (``cf_t_*``) and the nodes those hang under.
IK_FAMILIES = ("cf_pv_", "cf_t_", "cf_ik_", "cf_kk_")
#: The reference joints an IK bone can stand on.
JOINT_FAMILY = "cf_j_"
#: The clips' own goals, the ones :func:`remap_goals` moves (MotionIK's own ``target``/``chain`` rows).
GOAL_FAMILY = "cf_pv_"
#: An IK bone within this of a reference joint at rest stands on it.  Measured on the reference rig
#: (cache v4, ``origin_rig.json``): the 29 that stand on one are 0.000–0.001 mm from it, the nearest
#: of the rest (``cf_pv_elbo_L`` → ``cf_j_forearm01_L``) 499.6 mm, so any value between classifies alike.
COINCIDENT_M = 1e-3
#: ExpressionBone's driver family; joint-coincident siblings follow the same fitted joint.
DRIVER_FAMILY = "cf_d_"
#: Unity's human bone chains (``HumanTrait`` parents), root to tip, whose bones are turned onto the
#: character's segment to the next present bone of their chain.  Not the trunk's (Spine → Chest →
#: UpperChest → Neck): the lent body's trunk is fitted by moving it along her spine without turning
#: it (``body.TorsoField``), and turned at the bind its skin came back from every pose turned that
#: much more than the source's — Texas 9.1° at the upper chest, Lappland 18.6°, EndminF 12.1°
#: (§7.1443).  Unturned, her trunk keeps the bend it was made with and the lent one the source's.
#: The neck is turned onto her head: her head is her own, not fitted, and unturned it kept her
#: neck's angle to the reference's (Texas 16°) and looked at the floor.  The limbs are turned, as
#: their fit turns their skin (``body._limb_transform``).
HUMAN_CHAINS = (
    ("Neck", "Head"),
    ("LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand"),
    ("RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand"),
    ("LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes"),
    ("RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes"),
)
#: The human bone whose joint the character's size and the pelvis displacement are read at.
HIPS = "Hips"
#: The human bone at the root of the trunk: its rest stands over the pelvis as the reference's (module docstring).
TRUNK_ROOT = "Spine"
#: The human bone whose reference joint carries the face and the head's nulls (``build``'s *mouth*).
HEAD = "Head"
#: Below this a segment has no direction to align to.
_SHORT = 1e-9


class ConvertedRigError(ValueError):
    """The inputs cannot make a rig (a missing node, a table naming nothing, a name collision)."""


@dataclass
class RigBone:
    """One reference bone fitted to the character."""
    name: str
    parent: str | None
    #: local translation of the rest: the character's length along the reference's direction
    rest_pos: np.ndarray
    #: the reference rig's local rotation and scale (the rest is the reference's T-pose)
    rest_rot: np.ndarray
    rest_scale: np.ndarray
    #: the character's own rest world: the matrix the inverse bind inverts
    bind_world: np.ndarray
    #: ``rest_pos − s · reference local translation``; None on an IK bone
    offset: np.ndarray | None
    #: which placement rule (module docstring) put it there
    rule: str
    #: the source node it stands on (anchor, landmark), the joint (coincident) or the bone it rides
    source: str


@dataclass
class KeptNode:
    """A source node kept under its own name: hung under the origin bone that replaced its parent."""
    name: str
    parent: str | None
    #: local under *parent*'s bind, so at the bind pose it is where the source put it
    local: np.ndarray
    #: its source rest world (its inverse bind)
    bind_world: np.ndarray


@dataclass
class ConvertedRig:
    """The fitted rig, the maps from the source skeleton onto it, and the goal table."""
    scale: float
    bones: list[RigBone]
    #: source node → the origin bone that replaces it (anchors and landmarks)
    replace: dict[str, str]
    #: kept source node → its new parent, for the nodes whose source parent was replaced
    reparent: dict[str, str]
    #: every kept source node in the source's order (parents first)
    kept: list[KeptNode]
    #: the clip's own goal → the reference joint it is moved with (:func:`remap_goals`)
    goal_joints: dict[str, str]
    #: origin bones a landmark was given for that an anchor placed instead: their landmark nodes are kept
    shadowed: dict[str, str] = field(default_factory=dict)

    @property
    def offsets(self) -> dict[str, np.ndarray]:
        """``{bone: offset}`` for every bone that carries one (``Skeleton.clip_offsets``)."""
        return {b.name: b.offset for b in self.bones if b.offset is not None}

    def rules(self) -> Counter:
        """How many bones each rule placed."""
        return Counter(b.rule for b in self.bones)

    def bind_pose(self) -> dict:
        """The pose that turns the rest (reference T-pose) into the character's own rest.

        ``{bone: {"rot": conj(rest local) · bind local}}`` for every origin bone; the
        translations are the rest's already but the trunk root's, whose rest stands over
        the pelvis as the reference's and whose bind is the character's joint: that one also
        has ``"pos"``, the step from the one to the other.
        """
        bind = {b.name: b.bind_world for b in self.bones}
        out = {}
        for b in self.bones:
            local = bind[b.name] if b.parent is None else np.linalg.inv(bind[b.parent]) @ bind[b.name]
            turn = m3.decompose(local)[1]
            out[b.name] = {"rot": m3.quat_normalize(m3.quat_mul(m3.quat_conj(b.rest_rot), turn))}
            step = local[:3, 3] - b.rest_pos
            if np.abs(step).max() > 1e-12:
                out[b.name]["pos"] = step
        return out

    def skeleton(self, name: str = "converted_rig", *, kept: bool = True) -> Skeleton:
        """A :class:`Skeleton` with the rest, the character's binds and the clip offsets.

        *kept* adds the kept source nodes after the origin bones.  ``rig_scale``
        is ``s`` (what a clip's translation below ``cf_n_height`` is multiplied
        by), ``clip_offsets`` the per-bone offsets, ``always_pos`` stays empty:
        the shape pass's every-frame writes belong to a card's sliders.

        Two frames, kept apart (module docstring): ``evaluate({})`` is the rest
        (the reference's T-pose at the character's lengths), while every
        ``Bone.bind_world`` / ``inv_bind`` / ``_inv_bind`` is the character's
        own rest — the frame the mesh is bound in.  Whatever reads a bone's
        bind as the mesh's (the dual-quaternion ``rest_frames``, the head's
        light frame, the cloth's rest, the gaze's lid rings) then reads the
        mesh's; a card has the two equal and cannot tell them apart.
        """
        sk = Skeleton(name)
        binds = []
        for b in self.bones:
            sk.add(b.name, b.parent, b.rest_pos, b.rest_rot, b.rest_scale)
            binds.append(b.bind_world)
        for k in self.kept if kept else ():
            sk.add_matrix(k.name, k.parent, k.local)
            binds.append(k.bind_world)
        sk.finalize()
        # the binds are the character's rest, not the rest's (the pattern of ``endfield.body``
        # rebinding a lent skin): at the bind pose nothing moves
        for i, world in enumerate(binds):
            sk.bind_skin(i, _inverse_bind(world), world)
        sk.rig_scale = float(self.scale)
        sk.clip_offsets = {n: np.asarray(v, dtype=np.float64).copy() for n, v in self.offsets.items()}
        sk.goal_joints = {goal: joint for goal, joint in self.goal_joints.items()
                          if sk.has(goal) and sk.has(joint)}
        return sk


def scaled_reference(reference: Skeleton, scale: float, name: str = "reference_s") -> Skeleton:
    """*reference* (grafted at 1.0) at *scale*: every local translation times it.

    The same skeleton ``rig.graft_full_skeleton(…, scale)`` builds: that graft
    scales world translations and re-derives the locals through the parents'
    worlds, which (no shear on the reference: 0 of 680 world matrices) is the
    local translation times the scale.
    """
    out = Skeleton(name)
    for b in reference.bones:
        parent = reference.bones[b.parent].name if b.parent >= 0 else None
        out.add(b.name, parent, b.local_pos * float(scale), b.local_rot, b.local_scale)
    out.finalize()
    out.rig_scale = float(scale)
    return out


def remap_goals(goal_joints: dict, fitted: dict, reference: dict) -> dict:
    """The clip's own goals moved with the joint each belongs to: ``fitted[j] + (reference[goal] − reference[j])``.

    *reference* holds world positions of the goals and their joints on the
    ``s ×`` reference rig under the clip (where the clip puts its goals: on the
    reference's FK joints, 0.1 mm on EndminF's three clips, design note §1.4);
    *fitted* the same joints on the fitted rig.  On a card both are the same
    rig and nothing moves.
    """
    out = {}
    for goal, joint in goal_joints.items():
        for table, key in ((reference, goal), (reference, joint), (fitted, joint)):
            if key not in table:
                raise ConvertedRigError(f"no world position of '{key}' to move the goal '{goal}' by")
        out[goal] = np.asarray(fitted[joint], dtype=np.float64) + (
            np.asarray(reference[goal], dtype=np.float64) - np.asarray(reference[joint], dtype=np.float64))
    return out


def clip_rig_pose(skeleton, pose: dict) -> dict:
    """*pose* on *skeleton* with every clip offset taken off its bone's translation.

    Evaluated on the converted rig it is the clip on its own rig — ``s`` times
    the reference rig, every bone's translation the clip's — which is where a
    card of this size would stand (``Skeleton.clip_offsets``).  A card has no
    offsets and gets *pose* back.
    """
    offsets = getattr(skeleton, "clip_offsets", None) or {}
    if not offsets:
        return pose
    out = dict(pose)
    for bone, offset in offsets.items():
        entry = dict(out.get(bone) or {})
        entry["pos"] = np.asarray(entry.get("pos", np.zeros(3)), dtype=np.float64) - offset
        out[bone] = entry
    return out


def goal_joints_from_ik(settings, files: dict) -> dict:
    """``{goal: joint}`` for every clip goal MotionIK's limb rows name, read off the solver's own chains.

    A row's ``target`` is the limb effector's target (``IKTargetPair``: part
    *i* → ``effectors[5 + i]``), its ``chain`` the limb chain's bend goal
    (``chain[1 + i]``); the joint is that effector's bone, or the chain's
    middle node — the bend (``fbbik.json``: ``cf_j_arm00_L``,
    ``cf_j_forearm01_L``, ``cf_j_hand_L``).  Only the clip's own goals
    (:data:`GOAL_FAMILY`) are kept; a partner's null is a point on a body and
    needs no moving.  Two rows sending one goal to two joints is an error.
    """
    from .motion_ik import PAIRS
    limbs = {part: (e, c) for part, e, c in PAIRS}
    out: dict[str, str] = {}
    for states in files.values():
        for state in states:
            for part, row in (state.get("parts") or {}).items():
                if part not in limbs:
                    raise ConvertedRigError(f"MotionIK part '{part}' is not one of {sorted(limbs)}")
                e, c = limbs[part]
                for key, joint in (("target", settings.effectors[e].bone), ("chain", settings.chains[c].nodes[1].bone)):
                    goal = row.get(key)
                    if not goal or not goal.startswith(GOAL_FAMILY):
                        continue
                    if out.setdefault(goal, joint) != joint:
                        raise ConvertedRigError(f"'{goal}' goes with '{out[goal]}' and '{joint}'")
    return out


# -- building ----------------------------------------------------------------------------------
def _check_tables(reference: Skeleton, source: Skeleton, humanoid: dict, human_joints: dict,
                  fingers: dict, landmarks: dict) -> dict:
    """``{origin bone: source node}`` of the anchors; ``ConvertedRigError`` on anything that names nothing."""
    if HIPS not in humanoid or HIPS not in human_joints:
        raise ConvertedRigError("the humanoid table has no Hips")
    anchors: dict[str, str] = {}
    for human, node in humanoid.items():
        if human not in human_joints:
            continue
        anchors[human_joints[human]] = node
    for bone, node in fingers.items():
        if bone in anchors:
            raise ConvertedRigError(f"'{bone}' is both a human bone and a finger")
        anchors[bone] = node
    for table, what in ((anchors, "anchor"), (landmarks, "landmark")):
        for bone, node in table.items():
            if not reference.has(bone):
                raise ConvertedRigError(f"the reference rig has no {what} bone '{bone}'")
            if not source.has(node):
                raise ConvertedRigError(f"the character has no node '{node}' for the {what} '{bone}'")
    used = Counter(anchors.values()) + Counter(n for b, n in landmarks.items() if b not in anchors)
    twice = sorted(n for n, k in used.items() if k > 1)
    if twice:
        raise ConvertedRigError(f"source nodes placing two origin bones: {twice}")
    return anchors


def _coincident(reference: Skeleton) -> dict:
    """``{IK bone: the reference joint it stands on at rest}`` (:data:`COINCIDENT_M`)."""
    joints = [i for i, b in enumerate(reference.bones) if b.name.startswith(JOINT_FAMILY)]
    if not joints:
        return {}
    where = np.array([reference.bones[i].bind_world[:3, 3] for i in joints])
    out = {}
    for b in reference.bones:
        if not b.name.startswith(IK_FAMILIES):
            continue
        apart = np.linalg.norm(where - b.bind_world[:3, 3], axis=1)
        k = int(np.argmin(apart))
        if apart[k] < COINCIDENT_M:
            out[b.name] = reference.bones[joints[k]].name
    return out


def _driver_peers(reference: Skeleton, anchors: dict) -> dict:
    """Correction drivers colocated with an anchored sibling in the source rig.

    ExpressionBone copies the sibling joint's local position into its driver.
    Carrying the driver at the old limb length makes that update a spurious
    translation. Preserve their source displacement when placing the pair.
    Only unambiguous siblings qualify; arbitrary nearby bones never do.
    """
    out = {}
    for b in reference.bones:
        if not b.name.startswith(DRIVER_FAMILY):
            continue
        peers = [other for other in reference.bones
                 if other.name in anchors and other.parent == b.parent
                 and np.linalg.norm(other.bind_world[:3, 3] - b.bind_world[:3, 3]) < COINCIDENT_M]
        if len(peers) == 1:
            out[b.name] = peers[0].name
    return out


def _alignments(reference: Skeleton, rest: dict, anchors: dict, humanoid: dict, human_joints: dict) -> dict:
    """``{origin joint: shortest arc from its reference segment to the character's}`` along :data:`HUMAN_CHAINS`."""
    out = {}
    for chain in HUMAN_CHAINS:
        present = [h for h in chain if h in humanoid and h in human_joints and human_joints[h] in anchors]
        for a, b in pairwise(present):
            ja, jb = human_joints[a], human_joints[b]
            d_ref = reference.world_position(jb) - reference.world_position(ja)
            d_chr = rest[anchors[jb]] - rest[anchors[ja]]
            if np.linalg.norm(d_ref) < _SHORT or np.linalg.norm(d_chr) < _SHORT:
                raise ConvertedRigError(f"the segment {a} → {b} has no length to align")
            out[ja] = m3.quat_from_to(d_ref, d_chr)
    return out


def _kept_nodes(source: Skeleton, source_world: np.ndarray, replace: dict, bind: dict,
                origin: set) -> tuple[list, dict]:
    """The source nodes no origin bone replaces, hung under the replacement of their parent."""
    kept, reparent = [], {}
    for i, b in enumerate(source.bones):
        if b.name in replace:
            continue
        if b.name in origin:
            raise ConvertedRigError(f"the source node '{b.name}' has an origin bone's name; refusing to rename it")
        parent = source.bones[b.parent].name if b.parent >= 0 else None
        if parent in replace:
            parent = replace[parent]
            reparent[b.name] = parent
        world = source_world[i]
        local = world if parent is None else _inverse_bind(bind[parent]) @ world
        kept.append(KeptNode(b.name, parent, local, world.copy()))
        bind[b.name] = world
    return kept, reparent


def build(reference: Skeleton, source: Skeleton, humanoid: dict, human_joints: dict, *,
          fingers: dict | None = None, landmarks: dict | None = None,
          goal_joints: dict | None = None, mouth: tuple | None = None) -> ConvertedRig:
    """The reference rig fitted to *source* (module docstring); pure computation.

    *reference*: the reference rig at scale 1 (``rig.graft_full_skeleton``), parents first.
    *source*: the character's skeleton, finalized, at its rest.
    *humanoid*: Unity human bone → source node.  *human_joints*: human bone → reference joint.
    *fingers*: reference joint → source node (the fit's finger pairs).
    *landmarks*: reference bone → source node the fit placed it on (an anchor wins over one).
    *goal_joints*: clip goal → reference joint (:func:`goal_joints_from_ik`), kept on the result.
    *mouth*: ``(the reference rig's mouth bone, the character's lip nodes)`` — or, for a face with no lip
    joints, its measured lip points (an array, ``endfield.shape_face.lips_centre``).  A converted head is
    the character's own, not the reference's, and the joint each turns on is not at the same place
    in it: what the reference rig carries on its head (the face, the nulls the oral rows aim the
    shaft at) is then moved as one so that the reference mouth stands at the middle of the
    character's lips — the rule the stand-in reads those points by (``hrole.StandIn.point``),
    rule ``mouth``.  Without it the head's bones are carried like any other.
    """
    if mouth is not None:
        bone, lips = mouth
        measured = isinstance(lips, np.ndarray)
        if not reference.has(bone) or not len(lips) or (not measured and not all(source.has(n) for n in lips)):
            raise ConvertedRigError(f"the mouth '{bone}' or the lips {list(lips)} are not on the skeletons")
        first = build(reference, source, humanoid, human_joints, fingers=fingers, landmarks=landmarks,
                      goal_joints=goal_joints)
        if measured:
            # a face with no lip joints: its lips measured off its own surface (``endfield.shape_face``)
            middle = np.asarray(lips, dtype=np.float64).reshape(-1, 3).mean(axis=0)
        else:
            rest = source.evaluate({})
            middle = np.mean([rest[source.index[n]][:3, 3] for n in lips], axis=0)
        at = next(b for b in first.bones if b.name == bone).bind_world[:3, 3]
        return _build(reference, source, humanoid, human_joints, fingers, landmarks, goal_joints,
                      middle - at)
    return _build(reference, source, humanoid, human_joints, fingers, landmarks, goal_joints, None)


def _build(reference, source, humanoid, human_joints, fingers, landmarks, goal_joints, head_shift):
    """:func:`build`, the head's carried bones moved by *head_shift* (None: not moved)."""
    fingers, landmarks = dict(fingers or {}), dict(landmarks or {})
    anchors = _check_tables(reference, source, humanoid, human_joints, fingers, landmarks)
    shadowed = {b: landmarks.pop(b) for b in sorted(set(landmarks) & set(anchors))}
    source_world = source.evaluate({})
    rest = {b.name: source_world[i][:3, 3] for i, b in enumerate(source.bones)}
    pelvis = human_joints[HIPS]
    ref_pelvis = reference.world_position(pelvis)
    if ref_pelvis[1] <= _SHORT:
        raise ConvertedRigError(f"the reference '{pelvis}' is not above the floor")
    # the character's size: its hips over the reference pelvis joint (``Retarget._hip_ratio``)
    s = float(rest[anchors[pelvis]][1] / ref_pelvis[1])
    pelvis_shift = rest[anchors[pelvis]] - s * ref_pelvis
    align = _alignments(reference, rest, anchors, humanoid, human_joints)
    trunk_root = human_joints.get(TRUNK_ROOT) if TRUNK_ROOT in humanoid else None
    for bone, joint in (goal_joints or {}).items():
        if not reference.has(bone) or not reference.has(joint):
            raise ConvertedRigError(f"the goal table names '{bone}' → '{joint}', which the reference rig lacks")

    fit = _Placer(reference, s, pelvis, pelvis_shift, anchors, landmarks, _coincident(reference), rest,
                  driver_peers=_driver_peers(reference, anchors),
                  head=human_joints.get(HEAD), head_shift=head_shift)
    turn: list[np.ndarray] = []
    bones: list[RigBone] = []
    for i, b in enumerate(reference.bones):
        parent = reference.bones[b.parent].name if b.parent >= 0 else None
        turn.append(align[b.name] if b.name in align else (turn[b.parent] if b.parent >= 0 else m3.quat_identity()))
        rule, src = fit.place(b, parent, turn[i])
        world = fit.bind[b.name]
        local = world if parent is None else np.linalg.inv(fit.bind[parent]) @ world
        rest_pos = local[:3, 3].copy()
        if b.name == trunk_root and b.name in anchors:
            # over the pelvis as the reference's spine is, the height the character's (module docstring)
            rest_pos[[0, 2]] = s * b.local_pos[[0, 2]]
        offset = None if b.name.startswith(IK_FAMILIES) else rest_pos - s * b.local_pos
        bones.append(RigBone(b.name, parent, rest_pos, b.local_rot.copy(),
                             np.asarray(b.local_scale, dtype=np.float64).copy(), world.copy(), offset, rule, src))

    replace = {node: bone for bone, node in anchors.items()}
    replace.update({node: bone for bone, node in landmarks.items()})
    kept, reparent = _kept_nodes(source, source_world, replace, fit.bind, {b.name for b in bones})
    return ConvertedRig(s, bones, replace, reparent, kept, dict(goal_joints or {}), shadowed)


@dataclass
class _Placer:
    """The placement rules of :func:`build` (module docstring), one bone at a time, parents first."""
    reference: Skeleton
    scale: float
    pelvis: str
    pelvis_shift: np.ndarray
    anchors: dict
    landmarks: dict
    coincide: dict
    rest: dict
    driver_peers: dict = field(default_factory=dict)
    #: the head's reference joint, and how far what it carries is moved (``build``'s *mouth*)
    head: str | None = None
    head_shift: np.ndarray | None = None
    #: bone → its fitted bind world, filled as bones are placed
    bind: dict = field(default_factory=dict)
    #: the floor roots and the IK bones standing on them: a bone hanging off one is pelvis-carried
    floor: set = field(default_factory=set)
    #: the head's joint and every bone under it
    on_head: set = field(default_factory=set)
    #: bones placed from the character (anchor, landmark, coincident…): a carried bone under one
    #: is the top of a carried run
    placed: set = field(default_factory=set)

    def place(self, b, parent: str | None, turn: np.ndarray) -> tuple[str, str]:
        """Put *b*'s fitted bind world in :attr:`bind`; ``(rule, what it was placed from)``."""
        n, s = b.name, self.scale
        if n == self.head or parent in self.on_head:
            self.on_head.add(n)
        rule, src = self._place(b, parent, turn, n, s)
        if rule != "carried":
            self.placed.add(n)
        elif self.head_shift is not None and n in self.on_head and parent in self.placed:
            # the top of a run the head carries: moved, and everything under it with it
            self.bind[n] = self.bind[n].copy()
            self.bind[n][:3, 3] += self.head_shift
            return "mouth", parent
        return rule, src

    def _place(self, b, parent: str | None, turn: np.ndarray, n: str, s: float) -> tuple[str, str]:
        ref_world = b.bind_world
        if n in self.anchors:
            rule, src, pos = "anchor", self.anchors[n], self.rest[self.anchors[n]]
        elif n in self.landmarks:
            rule, src, pos = "landmark", self.landmarks[n], self.rest[self.landmarks[n]]
        elif n in self.driver_peers:
            rule, src = "joint-peer", self.driver_peers[n]
            if src not in self.bind:
                raise ConvertedRigError(f"driver '{n}' follows joint '{src}', which is listed after it")
            peer = self.reference.bones[self.reference.index[src]]
            displacement = s * (b.local_pos - peer.local_pos)
            frame = np.eye(4) if parent is None else self.bind[parent]
            pos = self.bind[src][:3, 3] + frame[:3, :3] @ displacement
        elif n in self.coincide:
            rule, src = "coincident", self.coincide[n]
            if src not in self.bind:
                raise ConvertedRigError(f"'{n}' stands on '{src}', which the reference lists after it")
            pos = self.bind[src][:3, 3]
            if src in self.floor:
                self.floor.add(n)
        elif n in UNSCALED:
            rule, src, pos = "root", "", s * ref_world[:3, 3]
            self.floor.add(n)
        elif parent is None or parent in self.floor:
            rule, src, pos = "pelvis-carried", self.pelvis, s * ref_world[:3, 3] + self.pelvis_shift
        else:
            # rigid in the parent's fitted frame: the reference's local, at the character's size
            local = np.linalg.inv(self.reference.bones[b.parent].bind_world) @ ref_world
            local[:3, 3] *= s
            self.bind[n] = self.bind[parent] @ local
            return "carried", parent
        _t, q_ref, sc_ref = m3.decompose(ref_world)
        self.bind[n] = m3.compose(np.asarray(pos, dtype=np.float64), m3.quat_mul(turn, q_ref), sc_ref)
        return rule, src


# -- the stored form ---------------------------------------------------------------------------
#: The version of :func:`to_payload`'s layout; a table of another one is not read.
PAYLOAD_VERSION = 1


def _floats(value) -> list:
    return [float(v) for v in np.asarray(value, dtype=np.float64).ravel()]


def to_payload(rig: ConvertedRig) -> dict:
    """*rig* as JSON: every origin bone and the maps; the kept nodes are the source skeleton's own.

    A bone's rest is stored as its translation, quaternion and scale (exactly
    what the skeleton is built from, so a round trip is bit for bit), its bind
    as the 4×4 world.  The kept nodes are not stored: they are the source
    skeleton's nodes hung under the replacements, which :func:`from_payload`
    makes again from that skeleton — the table cannot disagree with it.
    """
    return {
        "version": PAYLOAD_VERSION,
        "scale": float(rig.scale),
        "bones": [{"name": b.name, "parent": b.parent, "pos": _floats(b.rest_pos), "rot": _floats(b.rest_rot),
                   "scale": _floats(b.rest_scale), "bind_world": _floats(b.bind_world),
                   "offset": None if b.offset is None else _floats(b.offset), "rule": b.rule, "from": b.source}
                  for b in rig.bones],
        "replace": dict(sorted(rig.replace.items())),
        "reparent": dict(sorted(rig.reparent.items())),
        "shadowed": dict(sorted(rig.shadowed.items())),
        "goal_joints": dict(sorted(rig.goal_joints.items())),
    }


def from_payload(payload: dict, source: Skeleton) -> ConvertedRig:
    """The rig :func:`to_payload` stored, on *source* (the character's skeleton at its rest).

    ``ConvertedRigError`` when the payload is of another version, names a
    node *source* does not have, or its re-parent map is not the one *source*
    gives — the table was made for another skeleton.
    """
    if not isinstance(payload, dict) or payload.get("version") != PAYLOAD_VERSION:
        raise ConvertedRigError(f"the rig table is not version {PAYLOAD_VERSION}")
    bones = []
    for row in payload["bones"]:
        bones.append(RigBone(
            row["name"], row["parent"], np.asarray(row["pos"], dtype=np.float64),
            np.asarray(row["rot"], dtype=np.float64), np.asarray(row["scale"], dtype=np.float64),
            np.asarray(row["bind_world"], dtype=np.float64).reshape(4, 4),
            None if row["offset"] is None else np.asarray(row["offset"], dtype=np.float64),
            row["rule"], row["from"]))
    names = {b.name for b in bones}
    replace = dict(payload["replace"])
    for node, bone in replace.items():
        if not source.has(node):
            raise ConvertedRigError(f"the rig table replaces '{node}', which the character does not have")
        if bone not in names:
            raise ConvertedRigError(f"the rig table puts '{node}' on '{bone}', which it does not list")
    bind = {b.name: b.bind_world for b in bones}
    kept, reparent = _kept_nodes(source, source.evaluate({}), replace, dict(bind), names)
    if reparent != dict(payload["reparent"]):
        raise ConvertedRigError("the rig table's re-parent map is not the character skeleton's")
    return ConvertedRig(float(payload["scale"]), bones, replace, reparent, kept,
                        dict(payload.get("goal_joints") or {}), dict(payload.get("shadowed") or {}))


# -- the drawn model on the rig ------------------------------------------------------------------
#: Below this a parent's change of frame counts as a pure rotation (:meth:`Rebound.adapt`): the
#: linear parts compared are products of unit-scale rotations, the identity to ~1e-15 when they are.
_RIGID = 1e-9


class Rebound:
    """A model bound to a character's own skeleton (*native*), moved onto its fitted rig.

    Every native node is **tied** to one bone of the rig: the origin bone that
    replaces it, or itself, kept under its own name.  A tied pair moves rigidly
    together, ``W_native[i] = W_rig[j] · inverse(B_rig[j]) · B_native[i]``
    (``B``: the two binds, both the character's rest), so a vertex skinned on
    the rig lands where it lands on the native skeleton: the rig's inverse
    bind of *j* is ``inverse(B_rig[j]) · B_native[i] · inv_bind_native[i]`` —
    which keeps whatever inverse bind the native node carried
    (``endfield.body.unblend`` binds a lent joint at its source's rest, not at
    its own).  Nothing of the mesh is rewritten but its bone indices
    (:meth:`remap`).

    The rig's other bones (the IK bones, the nulls, the face's origin bones…)
    tie to nothing; with a pose given on the native skeleton they ride their
    parent as at the bind (:meth:`rig_world`).
    """

    def __init__(self, rig: ConvertedRig, native: Skeleton, name: str = "converted_rig"):
        self.rig = rig
        self.native = native
        sk = rig.skeleton(name, kept=True)
        self.skeleton = sk
        n = len(native.bones)
        index = np.empty(n, dtype=np.int64)
        for i, bone in enumerate(native.bones):
            target = rig.replace.get(bone.name, bone.name)
            if not sk.has(target):
                raise ConvertedRigError(f"'{bone.name}' has no bone on the rig ('{target}')")
            index[i] = sk.index[target]
        if len(set(index.tolist())) != n:
            raise ConvertedRigError("two native nodes are tied to one bone of the rig")
        #: native bone index → the rig bone it is tied to
        self.index = index
        native_bind = np.stack([b.bind_world for b in native.bones])
        rig_bind = np.stack([b.bind_world for b in sk.bones])
        #: ``W_rig[index[i]] = W_native[i] @ to_rig[i]``, and back
        self.to_rig = np.linalg.inv(native_bind) @ rig_bind[index]
        self.to_native = np.linalg.inv(rig_bind[index]) @ native_bind
        for i, j in enumerate(index):
            sk.bind_skin(int(j), self.to_native[i] @ native._inv_bind[i])
        tied = np.zeros(len(sk.bones), dtype=bool)
        tied[index] = True
        parents = sk._parents
        #: the rig bones tied to no native node, by depth, and ``inverse(parent bind) · own bind``
        self._loose = [rows[~tied[rows]] for rows in sk._levels]
        self._loose_local = np.zeros((len(sk.bones), 4, 4))
        for j in np.flatnonzero(~tied):
            p = parents[j]
            self._loose_local[j] = rig_bind[j] if p < 0 else np.linalg.inv(rig_bind[p]) @ rig_bind[j]
        #: ``{kept node: (inverse(new parent's bind) · old parent's bind, whether that is a pure
        #: rotation)}`` for the nodes re-parented (:meth:`adapt`)
        self._frames: dict = {}
        for node, parent in rig.reparent.items():
            old = native.bones[native.index[node]].parent
            if old < 0:
                continue
            change = np.linalg.inv(rig_bind[sk.index[parent]]) @ native_bind[old]
            linear = change[:3, :3]
            rigid = (abs(float(np.linalg.det(linear)) - 1.0) < _RIGID
                     and float(np.abs(linear.T @ linear - np.eye(3)).max()) < _RIGID)
            self._frames[node] = (change, rigid)
        self._replaced = [(i, b.name) for i, b in enumerate(native.bones) if b.name in rig.replace]

    # -- meshes ----------------------------------------------------------------------------------
    def remap(self, mesh) -> None:
        """Point *mesh*'s bone indices at the rig (a new array: copies made before keep their own)."""
        idx = getattr(mesh, "bone_idx", None)
        if idx is None or not len(idx):
            return
        mesh.bone_idx = self.index[np.asarray(idx, dtype=np.int64)].astype(np.int32)

    # -- worlds ----------------------------------------------------------------------------------
    def rig_world(self, native_world: np.ndarray) -> np.ndarray:
        """The rig's world matrices for the native skeleton's *native_world*."""
        out = np.empty((len(self.skeleton.bones), 4, 4))
        out[self.index] = native_world @ self.to_rig
        parents = self.skeleton._parents
        for rows in self._loose:
            if not len(rows):
                continue
            above = parents[rows]
            top = above < 0
            if top.any():
                out[rows[top]] = self._loose_local[rows[top]]
            if (~top).any():
                out[rows[~top]] = out[above[~top]] @ self._loose_local[rows[~top]]
        return out

    def native_world(self, rig_world: np.ndarray) -> np.ndarray:
        """The native skeleton's world matrices for the rig's *rig_world* (each node with its tied bone)."""
        return rig_world[self.index] @ self.to_native

    def pose_from_native(self, native_world: np.ndarray) -> dict:
        """A rig pose (``{"matrix": local}`` on every bone) that puts the rig where *native_world* has the model."""
        world = self.rig_world(native_world)
        parents = self.skeleton._parents
        local = world.copy()
        inner = parents >= 0
        local[inner] = np.linalg.inv(world[parents[inner]]) @ world[inner]
        return {bone.name: {"matrix": local[j]} for j, bone in enumerate(self.skeleton.bones)}

    def native_view(self, rig_world: np.ndarray) -> dict:
        """``{native node: {"rot": turn}}`` of every replaced node, as the native skeleton would be posed.

        What reads a pose by the native skeleton's names and rests (the source
        rig's twist links and corrective joints) reads this: each replaced
        node's turn from its own rest, where its tied bone has it.  A kept node
        is not in it: it rides its bone as at the bind, which is its rest.
        """
        if not self._replaced:
            return {}
        world = self.native_world(rig_world)
        bones = self.native.bones
        rows = np.array([i for i, _name in self._replaced], dtype=np.int64)
        parents = np.array([bones[i].parent for i in rows], dtype=np.int64)
        local = world[rows].copy()
        inner = parents >= 0
        local[inner] = np.linalg.inv(world[parents[inner]]) @ world[rows[inner]]
        quats, _scale = m3.quats_from_matrices(local)
        return {name: {"rot": m3.quat_normalize(m3.quat_mul(m3.quat_conj(bones[i].local_rot), quats[k]))}
                for k, (i, name) in enumerate(self._replaced)}

    # -- pose entries ------------------------------------------------------------------------------
    def adapt(self, entries: dict) -> dict:
        """Entries written for the native skeleton's kept nodes, for the rig.

        A kept node keeps its name and its world rest; one whose parent was
        replaced hangs in another parent frame, so what its entry says in the
        old frame is said in the new: a turn or a scale of its own (applied in
        its own frame) is the same, an offset is turned into the new frame, and
        where that change of frame is not a pure rotation the entry becomes its
        whole local matrix.  A replaced node is not a kept node: an entry for
        it is an error (it is posed by its bone, under the bone's name).
        """
        from ..mesh.skeleton import posed_local
        out = {}
        native = self.native
        for name, entry in entries.items():
            if name in self.rig.replace:
                raise ConvertedRigError(f"'{name}' is posed by '{self.rig.replace[name]}' on the rig")
            frame = self._frames.get(name)
            if frame is None or not entry:
                out[name] = entry
                continue
            change, rigid = frame
            if rigid and "matrix" not in entry:
                moved = dict(entry)
                if moved.get("pos") is not None:
                    moved["pos"] = change[:3, :3] @ np.asarray(moved["pos"], dtype=np.float64)
                out[name] = moved
                continue
            out[name] = {"matrix": change @ posed_local(native.bones[native.index[name]], entry)}
        return out
