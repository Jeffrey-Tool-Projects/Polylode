"""Rebuild our bind pose from Illusion's reference skeleton.

Imported meshes are skinned relative to Illusion's rest skeleton.  Our
procedural skeleton has the same bone *names* but a different rest shape: it
encodes its A-pose entirely as bone positions, with every bind rotation left as
identity.  That is fine for our own meshes, which are generated against it, but
it makes imported skinning impossible -- a sleeve vertex 20cm along their
straight-out arm keeps that offset unrotated and shoots past our hand as a flat
spike.

So when a card brings in real assets we adopt their rest pose wholesale, then
re-apply the A-pose as actual *rotations* on the arm chain.  Skinning then does
what it is supposed to: the garment's local offsets rotate with the limb.
"""
from __future__ import annotations

import math

import numpy as np

from ..anim import library as _lib
from ..core import math3d as m3
from .skinmap import EXPLICIT

# Rest-pose rotations layered back on after adopting their T-pose, in radians
# about Z (arms) -- what turns a T-pose into the A-pose we render in.
#
# Derived from the animation library rather than written out again: the same
# drop is applied at runtime by ``anim.library.standing_rest``, which is the
# path every screen actually takes (``apose`` is off everywhere -- baking the
# drop into the bind would unskin every imported sleeve).  Two copies of one
# number is how they end up disagreeing.
#
# ``FOREARM_DROP`` is a *relative* turn -- ``cf_j_forearm01`` is a child of
# ``cf_j_arm00`` and inherits its drop -- so it goes negative whenever the
# forearm is meant to hang wider than the upper arm, which is what the game's
# own idle does (§7.64).  The sum still lands the forearm at
# ``FOREARM_REST_DEG`` from vertical, which is the only thing either number is
# for.
ARM_DROP = math.radians(90.0 - _lib.ARM_REST_DEG)
FOREARM_DROP = math.radians(_lib.ARM_REST_DEG - _lib.FOREARM_REST_DEG)


def our_to_their(name: str, rig) -> str | None:
    """Which of their bones defines the rest pose of ours."""
    if rig.matrix(name) is not None:
        return name
    for their, ours in EXPLICIT.items():          # first exact reverse hit
        if ours == name and rig.matrix(their) is not None:
            return their
    return None


_our_to_their = our_to_their


def adopt(skeleton, rig, scale: float = 1.0, apose: bool = True) -> int:
    """Set our bind pose from ``rig``.  Returns how many bones were moved.

    ``scale`` maps their body height onto ours; bones they don't have keep
    their existing local transform, so procedural extras (skirt rings, prop
    attachment points) simply ride along with their new parents.
    """
    moved = 0
    desired: dict[int, np.ndarray] = {}
    for i, bone in enumerate(skeleton.bones):
        their = _our_to_their(bone.name, rig)
        if their is None:
            continue
        m = rig.matrix(their).copy()
        m[:3, 3] *= float(scale)
        desired[i] = m

    for i, bone in enumerate(skeleton.bones):
        if i not in desired:
            continue
        world = desired[i]
        p = bone.parent
        parent_world = desired.get(p) if p >= 0 else None
        if parent_world is None and p >= 0:
            parent_world = skeleton.bones[p].bind_world
        local = world if parent_world is None else m3.inverse(parent_world) @ world
        bone.local_pos = local[:3, 3].copy()
        bone.local_rot = m3.quat_from_mat4(local)
        moved += 1

    skeleton.finalize()
    if apose:
        rest_apose(skeleton)
    return moved


def rest_apose(skeleton) -> None:
    """Drop the arms from their T-pose into ours, as bind rotations."""
    for side, sign in (("L", -1.0), ("R", 1.0)):
        for name, angle in ((f"cf_j_arm00_{side}", ARM_DROP),
                            (f"cf_j_forearm01_{side}", FOREARM_DROP)):
            if not skeleton.has(name):
                continue
            b = skeleton.bones[skeleton.index[name]]
            q = m3.quat_from_axis_angle((0.0, 0.0, 1.0), sign * angle)
            b.local_rot = m3.quat_normalize(m3.quat_mul(q, b.local_rot))
    skeleton.finalize()
