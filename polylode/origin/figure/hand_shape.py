"""The card's hand shape (``ChaFileStatus.enableShapeHand``), transcribed.

``ChaControl.InitShapeHand`` (called from ``<LoadAsync>``) builds ``sibHand``
from ``chara/oo_hand.unity3d`` — ``cf_anmShapeHand`` as the
``AnimationKeyInfo``, ``cf_customhand`` as the category list — and then calls
``SetShapeHandValue(hand, 0, 0, 0f)`` for both hands, which **writes back** into
``fileStatus.shapeHandPtn`` / ``shapeHandBlendValue``.  So whatever pattern the
card stored, a loaded character's hands are pattern 0 at blend 0; the only
card field left with an effect is ``enableShapeHand`` (§7.388).

Every frame ``LateUpdateForce`` → ``UpdateAlwaysShapeHand`` sets
``updateMask`` bit 1 for ``enableShapeHand[0]`` and bit 2 for ``[1]``, and
``ShapeHandInfo.Update`` then ``SetLocalRotation``s ``dictDst`` 0‥14 (left) /
15‥29 (right) from ``dictSrc[i].vctRot`` — **after** the animation, so the
fingers of an enabled hand hold that shape whatever the clip does.

``ShapeInfoBase.ChangeValue(hand, ptnA, ptnB, blend)`` samples each entry by
**key index** (``GetInfo(name, ptnA)`` / ``GetInfo(name, ptnB)``) and
``LerpAngle``s the two; with 0, 0, 0 that is key 0's rotation.  The dst and src
enums (``ShapeHandInfo.DstName`` / ``SrcName``) are the same 30 finger bones in
the same order, so entry *i* drives bone *i* by name.
"""
from __future__ import annotations

import numpy as np

from ...core import math3d as m3

HAND_BUNDLE = "chara/oo_hand.unity3d"
HAND_CURVES = "cf_anmShapeHand"

#: ``ShapeHandInfo.DstName``, in order: 0‥14 the left hand, 15‥29 the right.
FINGERS = ("index01", "index02", "index03", "little01", "little02",
           "little03", "middle01", "middle02", "middle03", "ring01", "ring02",
           "ring03", "thumb01", "thumb02", "thumb03")
HAND_BONES = tuple(f"cf_j_{f}_L" for f in FINGERS) + tuple(
    f"cf_j_{f}_R" for f in FINGERS)

#: the pattern ``InitShapeHand`` leaves every loaded character on
LOADED_PATTERN = 0


def enabled_hands(card) -> tuple[bool, bool]:
    """``ChaFileStatus.enableShapeHand`` as (left, right); both off when absent.

    A card written before the field existed carries none, and
    ``ChaFileStatus.MemberInit`` initialises it to ``new bool[2]`` — off.
    """
    status = card.sections.get("Status") if hasattr(card, "sections") else None
    raw = (status or {}).get("enableShapeHand") if isinstance(status, dict) else None
    vals = list(raw) if isinstance(raw, (list, tuple, bytes, bytearray)) else []
    vals = (vals + [False, False])[:2]
    return bool(vals[0]), bool(vals[1])


def pattern_rotations(curves, pattern: int = LOADED_PATTERN) -> dict:
    """bone → our absolute local quaternion for *pattern* (key index).

    The curves hold Unity ``localEulerAngles`` in degrees; the conversion is
    the one every baked Origin clip goes through (``hanim._mirror_quat`` of
    ``quat_from_euler(…, order="YXZ")``).
    """
    from ..hanim import _mirror_quat
    out = {}
    for bone in HAND_BONES:
        keys = curves.entries.get(bone) if curves is not None else None
        if keys is None or len(keys) <= pattern:
            continue
        rot = keys[pattern][4:7]
        out[bone] = _mirror_quat(m3.quat_from_euler(
            *np.deg2rad([float(r) for r in rot]), order="YXZ"))
    return out


def hand_override(skeleton, card, curves) -> dict:
    """The additive pose entries that lock the card's enabled hands, or ``{}``.

    Returned relative to *skeleton*'s bind (``hanim.absolute_pose_to_additive``)
    so :class:`CharacterController` can drop them over the frame's pose.
    """
    from ..hanim import absolute_pose_to_additive
    left, right = enabled_hands(card)
    if not (left or right):
        return {}
    rots = pattern_rotations(curves)
    wanted = {b: {"rot": q} for b, q in rots.items()
              if (left and b.endswith("_L")) or (right and b.endswith("_R"))}
    return absolute_pose_to_additive(skeleton, wanted)


def apply(pose: dict, override: dict) -> dict:
    """``SetLocalRotation`` over *pose*: the rotation is replaced, the rest kept."""
    if not override:
        return pose
    out = dict(pose)
    for bone, entry in override.items():
        merged = dict(out.get(bone, {}))
        merged["rot"] = entry["rot"]
        out[bone] = merged
    return out
