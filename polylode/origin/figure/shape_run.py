"""Run Illusion's own shape program over a grafted armature.

This is ``ShapeInfoBase.ChangeValue`` followed by ``ShapeBodyInfoFemale.Update``
/ ``ShapeHeadInfoFemale.Update``, in that order, with the arithmetic taken from
:mod:`shape_update` (transcribed from the IL) rather than reconstructed.

Two stages, exactly as the game has them:

``ChangeValue(category, rate)``
    Samples every curve the slider lists and copies **only the flagged**
    components into that curve's slot.  The slot starts at Vector3.zero for
    all three vectors — ``GetSrcBoneInfo`` news up a bare ``BoneInfo`` — so a
    component no row ever flags stays 0 and is a real input to ``Update``, not
    a missing value to be filled in.

``Update()``
    Walks a fixed list of writes.  Each names one bone, one setter and an
    expression per component; nothing is searched for and nothing is verified,
    because the program says outright which bone a curve reaches.

What replaced what
------------------
Until now the loader matched curves to bones by name and then checked the
guess against the bone's bind pose per channel (§7.8, narrowed by §7.30's
relation table).  That answered "which bone" but never "how much": several
curves combine into one channel, some are multiplied by the height-compensation
curve, and the bone-type body adds a fixed offset on top.  Those are the terms
the search could not see, and they are why the figure came out 0.14% short of
the height its own slider asked for.

Coordinates
-----------
Project space mirrors X against Illusion's, and the graft scaled every bone
offset to the character's height, so a position from a curve travels the same
way: negate X, multiply by the graft scale.  Rotations are Euler degrees in
Unity's ``Quaternion.Euler`` order (Y·X·Z) with Y and Z negated for the mirror.
Scales are dimensionless and pass through untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...core import math3d as m3
from ...core import swallowed
from . import shape_correct

# ``ShapeBodyInfoFemale.UPDATE_MASK_ALL``.  ``ChaControl.UpdateShapeBody``
# assigns exactly this before every full update; the partial masks exist for
# the per-frame bust solve, which is not this code path.
UPDATE_MASK_ALL = 7

# ``cf_n_height`` is the uniform scale the height slider puts on the root, and
# the graft already carries it (``rig_scale``, §7.9).  Writing it here as well
# squares it.  This is the one write the loader drops, and it is dropped by
# *bone* rather than by slider: the same slider also drives ``cf_a_height_aid``,
# the compensation that keeps the head and hands from scaling with the body,
# and dropping the whole slider left that curve at zero — which multiplies the
# head's scale to nothing.
SKIP_BONES = ("cf_n_height",)


@dataclass
class Sources:
    """``dictSrc`` — one sampled TRS per curve, at rest until a row fills it.

    ``ShapeInfoBase.BoneInfo..ctor`` seeds position and rotation with
    ``Vector3.zero`` and scale with **``Vector3.one``**, and that is load-bearing:
    a row may flag only ``sy``, and ``Update`` still multiplies by ``sx`` and
    ``sz`` of the same curve.  Starting the whole slot at zero collapsed eight
    bones to a singular matrix — the nipple chain and the shoulder collision
    proxies, each of which is read on a component nothing writes.
    """

    values: dict[str, np.ndarray] = field(default_factory=dict)

    def get(self, curve: str) -> np.ndarray:
        row = self.values.get(curve)
        if row is None:
            row = np.zeros((3, 3))
            row[2] = 1.0
            self.values[curve] = row
        return row


def low_poly_value(v: float) -> float:
    """``ChaControl.<InitShapeBody>m__1`` — the non-hiPoly slider squeeze.

    A low-poly body cannot carry the extremes, so every slider is remapped into
    0.2…0.8 around its midpoint.  Height is exempt (the caller restores it) and
    so is nipStand, which is lerped separately.
    """
    if v >= 0.5:
        return 0.5 + (0.8 - 0.5) * _inverse_lerp(0.5, 1.0, v)
    return 0.2 + (0.5 - 0.2) * _inverse_lerp(0.0, 0.5, v)


def _inverse_lerp(a: float, b: float, v: float) -> float:
    if a == b:
        return 0.0
    return min(max((v - a) / (b - a), 0.0), 1.0)


def slider_values(values, *, hi_poly: bool = True, nip_stand_rate: float = 0.0,
                  nip_stand_index: int | None = None,
                  height: float | None = None) -> np.ndarray:
    """``InitShapeBody``'s preamble: the array the sliders are played at.

    ``shapeValueBody`` is not fed to ``ChangeValue`` verbatim.  On a low-poly
    body every value is squeezed toward the middle, height excepted, and the
    nipple-stand slider is lerped toward 1 by the character's arousal rate on
    both detail levels.

    *height* is ``UpdateShapeBodyValueFromCustomInfo``'s last statement, which
    only fires for men: after the whole array has been played it calls
    ``ChangeValue(0, exType != 0 ? 1f : 0.6f)`` again, so a male body's height
    slider is a **constant** and whatever the card stores in slot 0 never
    reaches the rig.  See :func:`shape_apply.height_override`.
    """
    out = np.asarray(values, dtype=np.float64).copy()
    if not hi_poly and out.size:
        keep_height = float(out[0])
        out = np.array([low_poly_value(float(v)) for v in out])
        out[0] = keep_height
    if nip_stand_index is not None and 0 <= nip_stand_index < out.size:
        rate = min(max(float(nip_stand_rate), 0.0), 1.0)
        out[nip_stand_index] += (1.0 - out[nip_stand_index]) * rate
    if height is not None and out.size:
        out[0] = float(height)
    return out


def sources(rows, values, curves) -> Sources:
    """Play every slider through its rows — ``ChangeValue`` for each category.

    Categories are visited in index order and their rows in table order, which
    is the order the game visits them in; where two rows write the same
    component of the same curve, the later one wins in both.
    """
    src = Sources()
    by_category: dict[int, list] = {}
    for row in rows:
        by_category.setdefault(int(row.category), []).append(row)
    for category in sorted(by_category):
        if category >= len(values):
            continue
        rate = float(values[category])
        for row in by_category[category]:
            if row.curve not in curves:
                continue
            pos, rot, scale = curves.sample(row.curve, rate)
            slot = src.get(row.curve)
            sample = (pos, rot, scale)
            for kind in range(3):
                for comp in range(3):
                    if row.flags[kind * 3 + comp]:
                        slot[kind][comp] = float(sample[kind][comp])
    return src


@dataclass
class Context:
    """Everything an expression can read besides the curves themselves."""

    src: Sources
    correct: dict = field(default_factory=dict)
    type_bone: bool = False
    correct_head_size: float = 1.0
    correct_neck_size: float = 1.0
    head_correct_value: float = 1.0
    skeleton: object = None

    def parent_scale(self, bone: str) -> np.ndarray:
        """``bone.parent.lossyScale`` — the head's compensation reads it.

        Taken from the parent's finalised world matrix rather than by
        multiplying local scales up the chain, so a rotated joint between here
        and the root gives what Unity would give.
        """
        skel = self.skeleton
        if skel is None or not skel.has(bone):
            return np.ones(3)
        parent = skel.bones[skel.id(bone)].parent
        if parent < 0:
            return np.ones(3)
        _, _, scale = m3.decompose(skel.bones[parent].bind_world)
        return np.asarray(scale, dtype=np.float64)


_COMPARE = {
    "eq": lambda a, b: a == b,
    "ge": lambda a, b: a >= b,
    "gt": lambda a, b: a > b,
    "le": lambda a, b: a <= b,
    "lt": lambda a, b: a < b,
}


def evaluate(node, ctx: Context) -> float:
    tag = node[0]
    if tag == "c":
        return float(node[1])
    if tag == "s":
        return float(ctx.src.get(node[1])[node[2]][node[3]])
    if tag == "k":
        row = ctx.correct.get(node[1])
        return 0.0 if row is None else float(row[node[2]][node[3]])
    if tag == "p":
        return float(ctx.parent_scale(node[1])[node[2]])
    if tag == "correctHeadSize":
        return ctx.correct_head_size
    if tag == "correctNeckSize":
        return ctx.correct_neck_size
    if tag == "headCorrectValue":
        return ctx.head_correct_value
    if tag == "tb":
        return evaluate(node[1] if ctx.type_bone else node[2], ctx)
    if tag == "q":
        cond = node[1]
        hit = _COMPARE[cond[0]](evaluate(cond[1], ctx), evaluate(cond[2], ctx))
        return evaluate(node[2] if hit else node[3], ctx)
    a = evaluate(node[1], ctx)
    b = evaluate(node[2], ctx)
    if tag == "+":
        return a + b
    if tag == "-":
        return a - b
    if tag == "*":
        return a * b
    if tag == "/":
        return a / b if b else 0.0
    raise ValueError(f"未知的塑形節點 {tag}")


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _set_position(bone, comps: dict[int, float], unit: float) -> None:
    for i, v in comps.items():
        bone.local_pos[i] = (-v if i == 0 else v) * unit


def _set_scale(bone, comps: dict[int, float], unit: float) -> None:
    for i, v in comps.items():
        bone.local_scale[i] = v


def _set_rotation(bone, comps: dict[int, float], unit: float = 1.0) -> None:
    have = np.degrees(m3.quat_to_euler(bone.local_rot, "YXZ"))
    # the stored rotation is already mirrored, so undo that before merging a
    # partial write back in
    have = np.array([have[0], -have[1], -have[2]])
    want = np.array([comps.get(i, have[i]) for i in range(3)])
    quat = m3.quat_from_euler(*np.radians(want), order="YXZ")
    bone.local_rot = m3.quat_normalize(
        np.array([quat[0], -quat[1], -quat[2], quat[3]]))


_SETTERS = {
    "Position": (_set_position, (0, 1, 2)),
    "PositionX": (_set_position, (0,)),
    "PositionY": (_set_position, (1,)),
    "PositionZ": (_set_position, (2,)),
    "Rotation": (_set_rotation, (0, 1, 2)),
    "RotationX": (_set_rotation, (0,)),
    "RotationY": (_set_rotation, (1,)),
    "RotationZ": (_set_rotation, (2,)),
    "Scale": (_set_scale, (0, 1, 2)),
    "ScaleX": (_set_scale, (0,)),
    "ScaleY": (_set_scale, (1,)),
    "ScaleZ": (_set_scale, (2,)),
}


@dataclass
class RunReport:
    label: str = ""
    applied: int = 0            # channel writes that landed on a bone
    bones: int = 0
    writes: int = 0             # program entries that ran
    missing: list = field(default_factory=list)   # bones the armature lacks
    curves: int = 0
    sampled: int = 0
    reason: str = ""

    def describe(self) -> str:
        if self.reason:
            return f"{self.label}: {self.reason}"
        out = (f"{self.label}: {self.writes} 筆寫入、"
               f"{self.applied} 個通道到 {self.bones} 根骨頭"
               f"（{self.sampled}/{self.curves} 條曲線有值）")
        if self.missing:
            out += f"，骨架缺 {len(self.missing)} 根"
        return out


def run(skeleton, program, ctx: Context, unit: float, *, label: str,
        mask: int = UPDATE_MASK_ALL, skip_bones=(), finalize: bool = True) -> RunReport:
    """Apply one transcribed ``Update`` to *skeleton*.

    *finalize* off leaves the bones' worlds as they were (only the locals are
    written): for a caller that runs the program many times over and composes
    the worlds itself (``origin.shape_measure``).  A parent's scale (``("p", …)``)
    is then read off the worlds as they were; only the head's program reads one.
    """
    report = RunReport(label=label)
    skip = set(skip_bones)
    touched: set[str] = set()
    missing: set[str] = set()
    for bone_name, setter, exprs, bits in program:
        if bits and not (mask & bits):
            continue
        if bone_name in skip:
            continue
        if not skeleton.has(bone_name):
            missing.add(bone_name)
            continue
        write, components = _SETTERS[setter]
        values = {components[i]: evaluate(exprs[i], ctx)
                  for i in range(len(components))}
        write(skeleton.bones[skeleton.id(bone_name)], values, unit)
        report.applied += len(values)
        report.writes += 1
        touched.add(bone_name)
    report.bones = len(touched)
    report.missing = sorted(missing)
    if report.applied and finalize:
        skeleton.finalize()
    return report


def body_context(card, skeleton, correct=None) -> Context:
    """The constants ``InitShapeInfo`` sets up for one card."""
    body = card.body() if hasattr(card, "body") else {}
    type_bone = bool((body or {}).get("typeBone") or 0)
    correct = shape_correct.offsets() if correct is None else correct
    male = _is_male(card)
    return Context(src=Sources(), correct=correct, type_bone=type_bone,
                   # ``ShapeBodyInfoFemale.InitShapeInfo``: ``sex == 0`` sets
                   # both to 0.91, otherwise they stay at the ctor's 1.0.  Zero
                   # is *male* in Illusion's numbering; this project's own
                   # ``sex_of`` is the other way round, so the two must not be
                   # compared without converting.
                   correct_head_size=0.91 if male else 1.0,
                   correct_neck_size=0.91 if male else 1.0,
                   head_correct_value=shape_correct.head_correct_value(correct),
                   skeleton=skeleton)


def _is_male(card) -> bool:
    from .body_head import sex_of
    try:
        return int(sex_of(card)) == 1        # project convention: 1 = male
    except Exception as exc:                             # noqa: BLE001
        swallowed.note("origin.figure.shape_run._is_male: return int(sex_of(card)) == 1", exc)
        return False


def program_curves(program) -> set[str]:
    """Every curve an ``Update`` program reads."""
    out: set[str] = set()

    def walk(node):
        if node[0] == "s":
            out.add(node[1])
            return
        for child in node[1:]:
            if isinstance(child, tuple):
                walk(child)

    for _bone, _setter, exprs, _bits in program:
        for expr in exprs:
            walk(expr)
    return out
