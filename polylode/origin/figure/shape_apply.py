"""``InitShapeBody`` / ``InitShapeFace`` — play the card's shape sliders.

A Origin card carries ``shapeValueBody[44]`` and ``shapeValueFace[52]``.  Until this
existed the pipeline read only ``shapeValueBody[0]`` (height) and scaled the
whole model, so every card came out on the default body with the default face:
correct clothes on the wrong person.

The game does it by moving bones, not vertices — the body mesh has no Unity
blendshapes (``m_Shapes`` is empty).  Each slider indexes ``cf_custombody`` /
``cf_customhead`` (:mod:`shape_cat`), which name the curves to play
(:mod:`shape_anim`) and the channels each row fills; what those curves then do
to the armature is :mod:`shape_update`, transcribed from ``Update``'s IL and
run by :mod:`shape_run`.  We do the same on the grafted armature and let the
existing retarget bake the rest shape, which is why clothes loaded afterwards
sit on the shaped body for free — same as ``ChaControl``, where shaping happens
before ``ChangeClothes``.

Why there is no bone search here any more
-----------------------------------------
``cf_custombody`` names ``cf_a_*`` curves and **no shipped bundle contains a
transform with those names**, so §7.8 matched them to bones by name and checked
each guess against the bone's own bind pose per channel.  That was sound as far
as it went — §7.30 narrowed it further with the relation read out of the IL —
but it could only ever answer *which* bone.  It could not see that
``cf_s_head``'s scale is the head curve times the height-compensation curve
times ``correctHeadSize``, that the bone-type body adds a fixed offset from
``shapecorrect``, or that a rotation channel is folded back into ±180°.  Those
terms are now read rather than inferred, and 32 face channels that the bind
check could not account for are simply written, because the program says so.
"""
from __future__ import annotations

import numpy as np

from ...core import swallowed
from . import shape_anim, shape_cat, shape_run, shape_update
from .shape_run import RunReport

# ``ChaFileDefine.BodyShapeIdx.Height``.  Its curve is a uniform scale on
# ``cf_n_height``, i.e. exactly the overall size the armature was already
# grafted at — the same slider reaches us twice, once through the card's
# ``body.height`` parameter and once here.  The *slider* is still played (it
# also drives ``cf_a_height_aid``, the compensation that keeps the head from
# growing with the body); only the duplicated write is dropped, by bone.
HEIGHT_SLIDER = shape_cat.HEIGHT_SLIDER

# ``ChaFileDefine.BodyShapeIdx.NipStand`` — lerped toward 1 by the character's
# arousal rate before the sliders are played, on both detail levels.
NIP_STAND_SLIDER = shape_cat.BODY_SHAPE_NAMES.index("NipStand")

# ``ChaFileDefine.cf_MouthShapeMaskID`` and ``cf_MouthShapeDefault``, out of the
# ``FieldRVA`` blobs behind ``RuntimeHelpers.InitializeArray`` (``--data``).
# The ids are face sliders; the table is [headKind][i] and headKind is 1 for
# head 200, 2 for head 201, 0 for everything else.
MOUTH_MASK_ID = (4, 41, 42, 43, 44, 45, 46)
MOUTH_MASK_DEFAULT = (
    (0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5),
    (0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5),
    (0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5),
)


#: KK_ExpandMaleMaker's GUID, which is also its ``KKEx`` section
EXPAND_MALE = "kokaiinum.KKExpandMaleMaker"


def male_height_kept(card) -> bool:
    """Does KK_ExpandMaleMaker let this man keep the card's height?

    Read from ``BepInEx/plugins/KK_ExpandMaleMaker.dll``: its transpilers
    (``ChaControl_SetShapeBodyValue_RemoveHeightLock`` / ``_Initialize_``) turn
    the ``sex`` test in front of the override into an unconditional branch, on
    every male.  Then ``MaleHeightCompatabilityController.OnReload``: with the
    setting *Compatability Mode* on, ``DelayCheck`` writes the vanilla value back
    (``exType != 0 ? 1 : 0.6`` into ``shapeValueBody[0]``) unless the card's
    section says ``MaleHeightEnabled`` **and** *Edit Male Height* is on; with it
    off, only *Edit Male Height* off does.  Both default on, and so are this
    install's (``kokaiinum.KKExpandMaleMaker.cfg``).
    """
    from .. import plugins as PLG
    if not PLG.installed(EXPAND_MALE):
        return False
    edit = PLG.flag(EXPAND_MALE, "Config.Edit Male Height", True)
    if not PLG.flag(EXPAND_MALE, "Config.Compatability Mode", True):
        return edit
    sec = ((getattr(card, "sections", None) or {}).get("KKEx") or {}).get(EXPAND_MALE)
    data = sec[1] if isinstance(sec, (list, tuple)) and len(sec) >= 2 else sec
    return edit and isinstance(data, dict) and data.get("MaleHeightEnabled") is True


def height_override(card) -> float | None:
    """The height a **male** body is shaped at, whatever the card says.

    ``ChaControl.UpdateShapeBodyValueFromCustomInfo`` plays the whole slider
    array and then, only when ``sex == 0`` (Illusion's numbering, i.e. male),
    plays slider 0 once more at ``exType != 0 ? 1f : 0.6f``.  So the male
    maker's missing height slider is not "the card happens to store 0.6" — the
    game overwrites it.

    It matters for the 8 of this install's 145 male cards that do *not* store
    0.6: a plugin-edited card carrying **-1.0** or **1.397** is out of the
    slider's own range and would otherwise be clamped to a body the game never
    shows.  Returns ``None`` for women, who keep their own value, and for a man
    :func:`male_height_kept` lets keep his.
    """
    if not shape_run._is_male(card) or male_height_kept(card):
        return None
    try:
        par = card.parameter if hasattr(card, "parameter") else {}
        ex = int(par.get("exType") or 0)
    except (TypeError, ValueError):
        ex = 0
    return 1.0 if ex else 0.6


def mouth_masked(card, values):
    """``UpdateShapeFace``'s first half — the mouth sliders under the mask.

    When ``status.disableMouthShapeMask`` is set, seven face sliders stop
    reading the card and take ``cf_MouthShapeDefault[headKind]`` instead: the
    lower face and the whole mouth group, neutralised so something else (a
    mouth accessory, an H scene) can drive them.  Every one of this install's
    1,386 cards leaves the flag off, so this is a mechanism kept honest rather
    than a visible change — but it is a card field, and ignoring a card field
    is how a character silently comes out wrong later.
    """
    status = getattr(card, "sections", {}).get("Status") or {}
    if not status.get("disableMouthShapeMask"):
        return values
    face = card.face() if hasattr(card, "face") else {}
    try:
        head_id = int((face or {}).get("headId") or 0)
    except (TypeError, ValueError):
        head_id = 0
    kind = 1 if head_id == 200 else 2 if head_id == 201 else 0
    out = np.asarray(values, dtype=np.float64).copy()
    for i, slider in enumerate(MOUTH_MASK_ID):
        if 0 <= slider < out.size:
            out[slider] = MOUTH_MASK_DEFAULT[kind][i]
    return out


def _report(label: str, reason: str) -> RunReport:
    return RunReport(label=label, reason=reason)


def _nip_stand_rate(card) -> float:
    status = getattr(card, "sections", {}).get("Status") or {}
    try:
        return float(status.get("nipStandRate") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def apply_shape(skeleton, values, rows, curves, program, ctx, unit: float, *,
                label: str, skip_bones=(), finalize: bool = True) -> RunReport:
    """Drive *skeleton* with one slider array.  Call before meshes are attached.

    *finalize* off writes the locals only (``shape_run.run``).
    """
    if curves is None:
        return _report(label, "找不到形狀曲線資產")
    if not rows:
        return _report(label, "找不到 customshape 對應表")
    if values is None or len(values) == 0:
        return _report(label, "卡片沒有滑桿陣列")

    ctx.src = shape_run.sources(rows, values, curves)
    report = shape_run.run(skeleton, program, ctx, unit, label=label,
                           skip_bones=skip_bones, finalize=finalize)
    wanted = shape_run.program_curves(program)
    report.curves = len(wanted)
    report.sampled = sum(1 for c in wanted if c in ctx.src.values)
    return report


#: ``ShapeBodyInfoFemale.UpdateAlways`` (254 bytes of IL), which ``ChaControl.LateUpdateForce`` runs
#: every frame **after** the animation: ``cf_d_kokan``'s local z from the ``cf_a_dan`` curve
#: (``dictDst[38]`` ← ``dictSrc[39].vctPos.z``), and the two ``fixCorrectBone``
#: (``cf_d_shoulder_L`` / ``_R``, the ctor) local x from ``correctValue[30]`` / ``[31]``
#: (``CorrectKeyName``: the same two bones) on a ``typeBone`` card, else the constants.  Not in
#: ``shape_update`` (that is ``Update``, generated): hand-transcribed, in its entry format.
#: Without it the clip's own ``cf_d_kokan`` z (0.020) stayed, and the entrance null the shaft
#: aims at sat 7 mm off the game's on every row; 21 mm from the shaft root, that turned the aim
#: by up to 20° (§7.1179).
BODY_ALWAYS = (
    ("cf_d_kokan", "PositionZ", (("s", "cf_a_dan", 0, 2),), 0),
    ("cf_d_shoulder_L", "PositionX", (("tb", ("k", "cf_d_shoulder_L", 0, 0), ("c", -0.0156337)),), 0),
    ("cf_d_shoulder_R", "PositionX", (("tb", ("k", "cf_d_shoulder_R", 0, 0), ("c", 0.0156015)),), 0),
)


def apply_always(skeleton, ctx, unit: float, *, finalize: bool = True) -> RunReport:
    """Run :data:`BODY_ALWAYS` on the rest and keep its values for every frame (``skeleton.always_pos``)."""
    report = shape_run.run(skeleton, BODY_ALWAYS, ctx, unit, label="體型（每格）", finalize=finalize)
    keep: dict = {}
    for bone, setter, exprs, _bits in BODY_ALWAYS:
        if not skeleton.has(bone):
            continue
        axis = {"PositionX": 0, "PositionY": 1, "PositionZ": 2}[setter]
        value = shape_run.evaluate(exprs[0], ctx)
        keep.setdefault(bone, {})[axis] = -value if axis == 0 else value
    skeleton.always_pos = keep
    return report


def apply_body_shape(skeleton, card, unit: float,
                     *, skip_bones=shape_run.SKIP_BONES, finalize: bool = True) -> RunReport:
    """The card's body sliders on *skeleton* (``ChaControl.UpdateShapeBody``), then ``UpdateAlways``.

    *finalize* off writes the bones' locals only (``shape_run.run``).
    """
    from .body_head import hi_poly
    body = card.body() if hasattr(card, "body") else {}
    values = (body or {}).get("shapeValueBody")
    if values is None or len(values) == 0:
        return _report("體型", "卡片沒有滑桿陣列")
    values = shape_run.slider_values(
        values, hi_poly=hi_poly(), nip_stand_rate=_nip_stand_rate(card),
        nip_stand_index=NIP_STAND_SLIDER, height=height_override(card))
    ctx = shape_run.body_context(card, skeleton)
    report = apply_shape(skeleton, values, shape_cat.body_rows(),
                         shape_anim.body_curves(), shape_update.BODY_UPDATE, ctx,
                         unit, label="體型", skip_bones=skip_bones, finalize=finalize)
    if not report.reason:
        apply_always(skeleton, ctx, unit, finalize=finalize)
    return report


def apply_face_shape(skeleton, card, unit: float, *, res=None) -> RunReport:
    """Run after the body: the head's own scale reads its parent's world scale.

    ``ChaControl.UpdateShapeFace`` hands the head ``1 / (1 + correctValue
    [cf_s_head].scl.y)`` and the body's ``typeBone`` flag, so the two shapers
    share one card-level state rather than each reading the card again.
    """
    face = card.face() if hasattr(card, "face") else {}
    values = (face or {}).get("shapeValueFace")
    if values is not None and len(values):
        values = mouth_masked(card, values)
    ctx = shape_run.body_context(card, skeleton)
    # The curve asset belongs to the head the card wears, not to head 00 --
    # ``bo_head``'s ShapeAnime column (§7.41).  A head we cannot resolve is
    # left to ``apply_head`` to report; shaping falls back to the stock table
    # rather than failing the whole character here.
    from .body_head import head_assets
    try:
        head = head_assets(card, res)
        curves = shape_anim.head_curves(*shape_anim.head_request(head))
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.shape_apply.apply_face_shape: head = head_assets(card, res)", exc)
        curves = shape_anim.head_curves()
    return apply_shape(skeleton, values, shape_cat.head_rows(),
                       curves, shape_update.HEAD_UPDATE, ctx,
                       unit, label="臉型")


# ---------------------------------------------------------------------------
# height, read straight off the curve
# ---------------------------------------------------------------------------
def height_scale(card) -> float | None:
    """The uniform scale the card's height slider puts on ``cf_n_height``.

    This is how tall the character is *in Illusion's own terms*, which is the
    only honest basis for sizing the graft: guessing it from our procedural
    proportions instead bakes our head-to-body ratio onto their body.

    Read through the program rather than through the category table, so it is
    the same expression the shaper would have written to the bone.

    The male override has to be applied here too, not just in
    :func:`apply_body_shape`: this is the scale the whole armature is grafted
    at, and if the two disagreed the shaping would be laid onto a rig of a
    different size than the one it was solved for.
    """
    body = card.body() if hasattr(card, "body") else {}
    values = (body or {}).get("shapeValueBody")
    if not values or HEIGHT_SLIDER >= len(values):
        return None
    forced = height_override(card)
    if forced is not None:
        values = list(values)
        values[HEIGHT_SLIDER] = forced
    curves = shape_anim.body_curves()
    rows = shape_cat.body_rows()
    if curves is None or not rows:
        return None
    ctx = shape_run.body_context(card, None)
    ctx.src = shape_run.sources(
        [r for r in rows if r.category == HEIGHT_SLIDER], values, curves)
    picked = [shape_run.evaluate(expr, ctx)
              for bone, setter, exprs, _bits in shape_update.BODY_UPDATE
              if bone == "cf_n_height" and setter == "Scale"
              for expr in exprs]
    if not picked:
        return None
    return sum(picked) / len(picked)


def card_height(card, reference=None) -> float | None:
    """Standing height the card asks for, in metres.

    The slider is not the whole answer: 178 of this install's cards also carry
    an ABMX scale on ``cf_n_height``, the same transform, from 0.25× to 1.38×
    (§7.98).  That factor belongs here and *not* in :func:`height_scale`, which
    is the size the armature is grafted at — ``bonemod`` writes the override
    onto the bone afterwards, so folding it into the graft as well would apply
    it twice.

    Only the height bone is read.  A card can move its standing height by other
    means (``cf_j_foot_*`` scale, leg length), and those are not modelled here;
    what this returns is "the height this character was authored at", which is
    what the callers — parameter import, and through it the preview framing —
    are asking for.
    """
    scale = height_scale(card)
    if scale is None:
        return None
    if reference is None:
        from ..rig import rig as _rig
        reference = _rig()
    full = reference.body_height()
    if full <= 0.1:
        return None
    from .bonemod import read_modifiers
    try:
        mods, _version, reason = read_modifiers(card)
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.shape_apply.card_height: mods, _version, reason = read_modifiers(card)", exc)
        mods, reason = {}, ""
    coords = mods.get("cf_n_height") if not reason else None
    mod = coords[0] if coords else None
    if mod is not None and mod.has_scale():
        scale *= float(mod.scale[1])
    return scale * full
