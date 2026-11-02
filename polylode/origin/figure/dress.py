"""Orchestrate a full Origin card dress onto a procedural character.

Pipeline (project ``assets_cache`` only)::

    parse outfit  →  resolve each id  →  load every mesh piece  →  attach

No drive-letter game path.  Every clothes slot / hair slot / accessory slot
is visited; multi-renderer prefabs keep **each** piece (no merge drop).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.i18n import tr
from ..install import configured, enabled
from .. import colorspace as CS
from .. import progress as PROG
from .. import sideloader as SL
from .. import textures as TEX
from ..apply import prepare_for_assets, current_outfit
from . import hair_acc as HA
from . import invisible_body as IB
from . import acc_states as AS
from . import pushup_live as PL
from . import outfit as O
from . import resolve as R
from . import load_mesh as M
from . import attach as A
from . import shape_apply as S
from . import bonemod as BM
from . import bust as BUST
from . import matedit as ME
from . import clothes_overlay as CO
from . import toon as TOON
from .body_head import (_list_says_none, apply_body, apply_body_mask,
                        apply_head, sex_of)
from .report import Attached, DressReport
from ...core import swallowed


def apply_card_shape(built, card, rig_scale: float, *, report=None,
                     res=None, coordinate: int = 0) -> list:
    """Run ``InitShapeBody`` / ``InitShapeFace`` on a freshly grafted armature.

    Must happen on the bare skeleton, before any mesh is retargeted: the game
    shapes the rig and *then* loads clothes, which is what makes a garment
    follow the wearer's waist instead of the default body's.

    Callers that skip the full dress and re-attach cached meshes have to run
    this too — the meshes were baked against the shaped bind, so leaving the
    skeleton at Illusion's rest pose would pose them about the wrong pivots.

    Illusion's sliders are not the last word on where a bone ends up: 58% of
    this install's cards also carry ABMX overrides, and those multiply onto the
    *shaped* rig (:mod:`bonemod`), so they run at the end of this function.
    """
    # The face curves belong to the head the card wears, and which zipmod owns
    # a modded head is a question only the card's own UAR answers -- resolving
    # it by (category, slot) alone can land on a different mod than the one
    # ``apply_head`` loads the mesh from (§7.41).
    if res is None:
        try:
            from .. import sideloader as SL
            res = SL.card_resolutions(card)
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.figure.dress.apply_card_shape: from .. import sideloader as SL", exc)
            res = None
    # ``UpdateShapeFace`` sizes the head's sway collider by the body type, and
    # the sway solve that reads it happens later, per garment, with no card in
    # reach -- so the flag is handed over here, where the card still is.
    from ..prefab import set_type_bone
    set_type_bone(bool(((card.body() if hasattr(card, "body") else {}) or {})
                       .get("typeBone") or 0))
    # ``BoneController.NeedsBaselineUpdate``: the plugin re-collects its
    # baseline whenever the shape pass has run again, and the pass below is
    # exactly that.  Dropping it here also keeps a second call on the same
    # graft from reading an already-modified bone back as its own baseline.
    BM.clear_baseline(built.skeleton)
    # Body first: the head's own scale compensation divides by its parent's
    # world scale, which the body's shaping is what sets.
    reps = [S.apply_body_shape(built.skeleton, card, rig_scale),
            S.apply_face_shape(built.skeleton, card, rig_scale, res=res)]
    # ``UpdateShapeFace`` also sizes the head's collider straight off the body
    # type — a write of its own, not a shape row (``prefab.body_colliders``)
    from ..prefab import HEAD_COLLIDER, head_collider_scale
    if built.skeleton.has(HEAD_COLLIDER):
        bone = built.skeleton.bones[built.skeleton.id(HEAD_COLLIDER)]
        bone.local_scale = np.full(3, head_collider_scale(
            bool(((card.body() if hasattr(card, "body") else {}) or {}).get("typeBone") or 0)))
        built.skeleton.finalize()
    # Then the chest hangs.  ``bustSoftness`` / ``bustWeight`` do not move a
    # bone of their own -- they retune the sway chains on ``cf_j_bustNN_*``, and
    # where those come to rest is the standing shape (§7.42).  It has to follow
    # the size sliders, whose curves are what put the chain's rest targets where
    # they are.
    bust = BUST.apply_bust_shape(built.skeleton, card, rig_scale)
    reps.append(bust)
    # kept for the per-frame solve (``bust_live``), which starts from this drape;
    # the buttocks join it there (``dictDynamicBoneBust`` 2 and 3)
    built.bust_chains = (list(bust.chain_objs)
                         + BUST.hip_chains(built.skeleton, rig_scale))
    # ABMX last, because its baseline is whatever the sliders just wrote:
    # ``<CollectBaselineCo>`` forces ``LateUpdateForce`` and only then reads
    # each bone's local TRS (§7.98).
    from . import pregnancy as PREG
    from . import lewdcrest as CREST
    effects = PREG.effects(card)
    for bone, mods in CREST.effects(card).items():
        effects.setdefault(bone, []).extend(mods)
    abm = BM.apply_bone_modifiers(built.skeleton, card, rig_scale,
                                  coordinate=coordinate, report=report,
                                  effects=effects)
    if abm.rows or abm.reason:
        reps.append(abm)
    if report is not None:
        report.replaced.extend(r.describe() for r in reps)
    return reps


@dataclass
class _Fit:
    """Everything the per-slot stages need, resolved once by stage 0."""
    built: object
    card: object
    report: DressReport
    mapper: object
    rig_scale: float = 1.0
    unit: float = 1.0
    body_scale: float = 1.0
    head_len: float = 0.22
    res: object = None
    # The renderers the card's MaterialEditor section switched off, resolved
    # once for the coordinate being worn (:mod:`matedit`).
    hidden: object = None
    # …and its float / colour / shader / queue rows (``matedit.MaterialEdits``).
    edits: object = None


def dress_character(built, card, *, outfit: int | None = None,
                    hair_color=None) -> DressReport:
    """Replace procedural meshes with full Origin card assets.

    Staged to follow ``ChaControl.Load``: rig → body → body shape → head →
    face shape → hair → clothes → accessories → release.  The order is not
    cosmetic — shaping has to land before any garment is retargeted, or the
    clothes are fitted to the default body and the wearer's waist shows
    through them.

    Requires ``assets_cache`` populated (``tools/sync_origin_local.py``).

    Each stage marks itself through :mod:`origin.progress` so the loading
    dialog can repaint — the dress runs on the GUI thread, so nothing else
    will.  The stage boundaries are only the labels: what keeps the window out
    of Windows' "not responding" state is the pumping inside the long stages
    (``bake.read_item``, the head's piece loop), because two of these stages
    run for far longer than the ~5 s it takes to be declared hung.

    The texture store is tidied when the outermost dress starts
    (:func:`origin.textures.building`): maps no live model references are let
    go then, and not in the middle of a build, whose maps are not on a
    material yet (§7.499).
    """
    with TEX.building():
        return _dress_character(built, card, outfit=outfit, hair_color=hair_color)


def _dress_character(built, card, *, outfit: int | None,
                     hair_color) -> DressReport:
    report = DressReport()
    if not enabled() or card is None:
        report.fail(tr("assets_cache 未就緒或無卡片"))
        return report
    if not configured().valid:
        report.fail(configured().reason or tr("無專案資產庫"))
        return report

    # The coordinate is picked before the body: an outfit can override the
    # character's makeup (ChaFileCoordinate.enableMakeup), and that decides
    # which lip / eyeshadow / cheek layers the skin textures get baked with.
    if outfit is None:
        outfit = current_outfit(card)
    outfit = int(outfit)
    # Pushup reshapes the chest a worn bra / top holds before anything reads
    # the body (:mod:`pushup_live`)
    try:
        pushup = pushup_for(card, outfit, report)
    except Exception as exc:                         # noqa: BLE001
        report.missing.append(f"Pushup：讀不到卡片（{exc}）")
        pushup = (None, (False, False))
    if pushup[0] is not None:
        card = PL.card_view(*pushup)
        if any(pushup[1]):
            report.replaced.append("胸型：Pushup（穿著的上衣／胸罩）")

    def _tick(msg: str) -> None:
        PROG.tick(msg, force=True)      # a stage label always lands on screen

    _tick(tr("骨架…"))
    fit = _stage_rig(built, card, report, outfit, pushup)   # 0
    _tick(tr("身體…"))
    _stage_body(fit, outfit)                                # 1 + 2 (shape)
    _tick(tr("頭部…"))
    _stage_head(fit, outfit)                                # 3 + 4 (face shape)
    _tick(tr("頭髮…"))
    _stage_hair(fit, hair_color)                            # 5
    _tick(tr("服裝…"))
    _stage_clothes(fit, outfit)                             # 6
    _tick(tr("皮膚遮罩…"))
    _stage_body_mask(fit, outfit)                           # 6b
    _tick(tr("配件…"))
    _stage_accessories(fit, outfit)                         # 7
    _stage_release()                                        # 8
    _check_body(built, report)
    _stage_character_renderers(fit)                         # 8b
    IB.apply(built, card, report)                           # 8c
    _tick(tr("臉部休息…"))
    _stage_rest_face(built, card, report)                   # 9
    _stage_hand_shape(built, card, report)                  # 10
    _tick(tr("穿著完成"))
    return report


def _stage_hand_shape(built, card, report: DressReport) -> None:
    """``enableShapeHand``: the fingers ``LateUpdateForce`` holds (§7.389).

    Resolved once, against the finished skeleton — the per-frame half is
    ``CharacterController.update`` laying it over the pose.
    """
    from . import hand_shape as HS
    left, right = HS.enabled_hands(card)
    built.hand_override = {}
    if not (left or right):
        return
    from .shape_anim import curves
    table = curves(HS.HAND_BUNDLE, HS.HAND_CURVES)
    if table is None:
        report.missing.append(
            f"手的形狀：{HS.HAND_BUNDLE} 的 {HS.HAND_CURVES} 讀不到（卡片開著手的形狀）")
        return
    built.hand_override = HS.hand_override(built.skeleton, card, table)
    report.replaced.append(
        "手的形狀：" + "、".join(n for n, on in (("左手", left), ("右手", right)) if on)
        + f" 鎖成 pattern {HS.LOADED_PATTERN}（{len(built.hand_override)} 根手指骨）")


def _stage_character_renderers(fit: _Fit) -> None:
    """MaterialEditor's ``ObjectType.Character`` rows, once everything is on.

    Those rows are handed ``chaCtrl.gameObject`` — the whole character — so
    they can only be resolved after the body, head, hair, clothes and
    accessories are all parts.  Locally they are eyebrows, eyelines, irises,
    teeth and one ``o_body_a``: authors switching off a feature their own
    accessory replaces.
    """
    done, unmatched = ME.character_hidden_parts(fit.built, fit.hidden)
    if done:
        fit.report.replaced.append(
            f"MaterialEditor：角色層關掉 {len(done)} 片"
            f"（{'、'.join(done[:4])}）")
    if unmatched:
        fit.report.missing.append(
            f"MaterialEditor: 角色層指名要關的 "
            f"{'、'.join(unmatched[:4])} 在這個角色上找不到")


def _stage_rest_face(built, card=None, report: "DressReport | None" = None) -> None:
    """Put the face where the game leaves it the moment the head finishes.

    This is ``<ChangeHeadAsync>``'s tail: after the head prefab is on the
    character it applies the card's own ``eyebrowPtn`` / ``eyesPtn`` /
    ``mouthPtn`` and their open rates, then the blink flag.  Fifty sections
    read that as maker-only state and skipped it, so every character loaded
    with the neutral face the mesh happens to be modelled in — locally
    **552 of 1,388 cards** were saved wearing something else (§7.137).

    ``FaceBlendShape`` runs every frame in Origin, so even a character doing
    nothing has the eye pair's Close target held at ``1 - correctOpenMax``
    (8% on a stock female head, before any gaze).  Nothing here drives
    expressions on a still render, so the rest state has to be applied at
    dress time — and through the same rig the expression controller will pick
    up, or the second rig would take the posed mesh as its base.

    This is the *undriven* rest, matching an iris map baked at a gaze of zero.
    A character whose eyes are actually being looked through sits lower than
    this (``chara.gaze``); the two stay consistent because both go through
    ``face_morph.gaze_open_max``.

    ``chara.expression.ExpressionController`` re-applies the same weights every
    frame from ``built.card_face``; it has to, because its ``update`` starts
    with ``rig.clear()`` and would otherwise wipe this on the first tick
    (§7.131).  Both ends call :func:`face_morph.rest_face_paths`, so there is
    one transcription of the rule and not two.
    """
    from ...mesh.blendshape import build_face_morphs, rest_blink
    from . import face_morph as FM

    face = None
    try:
        face = FM.card_face(card)
    except Exception as exc:                         # noqa: BLE001
        if report is not None:
            report.missing.append(f"face: 卡片表情讀取失敗（{exc}）")
    built.card_face = face
    rest = rest_blink(built)
    if face is None and rest <= 0.0:
        return
    try:
        rig = build_face_morphs(built)
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.dress._stage_rest_face: rig = build_face_morphs(built)", exc)
        return
    if face is None:
        return
    # Without ``cha_eyeset`` the eye number is the card's own, and that is only
    # right for rows 0‥20 — so say the mapping is missing rather than let a
    # 瞳にハート card quietly wear ぐるぐる目 (``CLAUDE.md`` 一之一).
    if face.eyes_ptn and not FM.eyeset_table() and report is not None:
        report.missing.append(
            f"face: cha_eyeset 讀不到（face_ptn.json 版本過舊？），"
            f"eyesPtn={face.eyes_ptn} 只能當成 FBS 編號用")
    written = FM.write_paths(rig, FM.rest_face_paths(face, eye_close=rest))
    if written:
        rig.apply()
    elif not face.is_default and report is not None:
        report.missing.append(
            f"face: 卡片表情 眉{face.eyebrow_ptn}/眼{face.eyes_fbs_ptn}"
            f"/口{face.mouth_ptn} 在這顆頭上找不到對應的 blend shape")
    # A gag-eye row swaps the iris for one of three quads and hides the real
    # eye behind it (``EyeObjNo`` / ``EyeBase`` / ``EyeHitomi``).  We draw the
    # blend-shape half and have no such object, so say so rather than let the
    # character quietly wear normal eyes (``CLAUDE.md`` 一之一).
    if face.gag_eye and report is not None:
        report.missing.append(
            f"face: eyesPtn={face.eyes_ptn} 是ギャグ目（{face.gag_eye}），"
            "我們沒有這個替換物件")


def _check_body(built, report: DressReport) -> None:
    """Say so, loudly, if ``o_body_a`` never arrived.

    ``external_mesh=True`` leaves the ``body`` part empty on the promise that
    the card fills it.  Nothing quietly substitutes a procedural body for it —
    a silently different character is worse than a visibly broken one, and the
    caller can only react to a failure it is told about.
    """
    part = built.model.get("body")
    if part is None or part.mesh is None or not part.mesh.vertex_count:
        report.fail(tr("body: 身體 (o_body_a) 未載入，角色沒有身體"))


def _stage_rig(built, card, report: DressReport, outfit: int = 0,
               pushup=None) -> _Fit:
    """Graft Illusion's armature at the card's own size and shape it.

    *outfit* is the coordinate being worn: an ABMX modifier can be stored per
    coordinate (1,302 of this install's 18,510 rows are), and
    ``BoneModifier.GetModifier`` picks by ``CurrentCoordinate``.
    """
    from ..rig import rig as _rig
    origin_head = _rig().head_length()
    try:
        head_len = float(built.prop.landmarks()["head_len"])
    except Exception as exc:  # noqa: BLE001
        swallowed.note("origin.figure.dress._stage_rig: head_len = float(built.prop.landmarks()['head_le", exc)
        head_len = 0.22
    unit = head_len / max(origin_head, 1e-3)
    # Size the graft from the card's own height slider when we can read it.
    # Fitting to our procedural head height instead stretches Illusion's body
    # to our head-to-body ratio, which is not the ratio the card was drawn at.
    card_scale = S.height_scale(card)
    # The face-local subtree belongs to the head this card wears, not to
    # oo_base's reference rig — read before the graft because the graft is what
    # consumes it (§7.102).  A head we cannot read is reported, and the graft
    # keeps oo_base's: an armature with no head subtree at all would take the
    # face meshes with it.
    head_nodes: dict = {}
    try:
        from .body_head import head_assets, head_bone_hierarchy
        head_nodes, why = head_bone_hierarchy(
            head_assets(card, SL.card_resolutions(card)))
    except Exception as exc:                         # noqa: BLE001
        head_nodes, why = {}, str(exc)
    if not head_nodes:
        report.missing.append(f"head rig: {why or '讀不到頭的骨架'}"
                              "，配件掛點退回 oo_base 的參考骨架")
    elif why != "bake":
        # Says so out loud, because it is a bundle open on the dress path and
        # the answer is identical either way — which is exactly the shape of
        # thing that stays unnoticed until somebody wonders where the seconds
        # went (CLAUDE.md 四: 執行期只讀我們的格式).
        report.replaced.append(f"head rig: {len(head_nodes)} 個節點讀自 {why}"
                               "（這個頭還沒 bake）")
    # full=True → graft complete Origin armature (no bone collapse), T-pose bind
    mapper, rig_scale, _ = prepare_for_assets(
        built, unit, enabled=True, report=report, apose=False, full=True,
        scale=card_scale, head_nodes=head_nodes)
    built.rig_scale = float(rig_scale)
    shape_rig(built, card, rig_scale, outfit=int(outfit), report=report,
              pushup=pushup)
    # MaterialEditor's switched-off renderers, read once for the coordinate
    # being worn.  A section that will not parse is reported here rather than
    # at each use, or one bad card would say the same thing twenty times.
    hidden = ME.read(card, coordinate=int(outfit))
    if hidden.reason:
        report.missing.append(f"MaterialEditor: {hidden.reason}")
    elif hidden.rows:
        report.replaced.append(
            f"MaterialEditor：卡片關掉 {hidden.rows} 個 renderer"
            f"（另有 {hidden.skipped_coordinate} 筆屬於別套 coordinate）"
            if hidden.skipped_coordinate else
            f"MaterialEditor：卡片關掉 {hidden.rows} 個 renderer")
    edits = ME.read_material_edits(card, coordinate=int(outfit))
    if edits.reason:
        report.missing.append(f"MaterialEditor: {edits.reason}")
    if edits.unapplied_textures:
        parts = "、".join(f"{k} ×{v}" for k, v in
                         sorted(edits.unapplied_textures.items(), key=lambda kv: -kv[1]))
        report.missing.append(f"MaterialEditor: 換圖沒有套（{parts}）")
    if edits.character_copies:
        report.missing.append(
            f"MaterialEditor: 角色本身的材質副本 {edits.character_copies} 筆沒有套"
            "（身體與頭照網格名辨認部件，副本會頂掉本尊）")
    if edits.foreign_shaders:
        # Another plugin's shader (``Goo``, ``IBL_Shader``, …): the plugin
        # swaps it in from a bundle this build does not have, so the material
        # keeps drawing with the shader it came with.
        names = sorted(set(edits.foreign_shaders))
        report.missing.append(
            f"MaterialEditor: {len(edits.foreign_shaders)} 筆換成外部 shader "
            f"（{'、'.join(names[:4])}），照原 shader 畫")
    return _Fit(built=built, card=card, report=report, mapper=mapper,
                rig_scale=rig_scale, unit=unit, head_len=head_len,
                res=SL.card_resolutions(card), hidden=hidden, edits=edits)


def _stage_body(fit: _Fit, outfit: int = 0) -> None:
    # The body prefab, its colour mask and its draw material all follow the
    # card's own sex byte; the head does not (there is no p_cm_head).
    apply_body(fit.built, fit.mapper, fit.report, fit.card, outfit, fit.res,
               sex=sex_of(fit.card))


def _stage_head(fit: _Fit, outfit: int = 0) -> None:
    from ..rig import rig as _rig
    from .attach import bone_origin
    apply_head(fit.built, fit.mapper, fit.report, fit.card, outfit, fit.res)
    origin_h = _rig().get("cf_j_head")
    my_h = bone_origin(fit.built.skeleton, "cf_j_head")
    fit.body_scale = (float(my_h[1]) / float(origin_h[1])) if (
        origin_h is not None and origin_h[1] > 0.1) else 1.0


def _stage_release() -> None:
    """Drop decompressed bundles — the meshes we keep are plain arrays now."""
    try:
        from ..meshimport import clear_cache
        from ..prefab import clear_loaders
        clear_loaders()
        clear_cache()
    except Exception as exc: # noqa: BLE001
        swallowed.note("origin.figure.dress._stage_release: clear_loaders", exc)


def _stage_hair(fit: _Fit, hair_color) -> None:
    built, report, mapper = fit.built, fit.report, fit.mapper
    unit, body_scale, head_len, res = (
        fit.unit, fit.body_scale, fit.head_len, fit.res)
    card = fit.card
    hparts = O.hair_parts(card)
    card_hair_rgb = (hair_color if hair_color is not None else _hair_rgb(hparts))
    hmat = A.hair_material(card_hair_rgb)
    # ChangeSettingHairGlossMask assigns one sheet to every hair renderer;
    # only the _ST that places it is per piece.
    gloss_id = TOON.card_gloss_id(card)
    gloss = TOON.gloss_map(gloss_id, res)
    if gloss is None and not _list_says_none("hairgloss", gloss_id):
        report.missing.append(f"hairgloss #{gloss_id} 解析不到（頭髮沒有高光圖）")
    # <ChangeHairAsync> parents every hair prefab to RefObjKey.HairParent with
    # worldPositionStays=false, so that transform's frame *is* the prefab's
    # space; see attach.HAIR_PARENT_BONE.  Skin to the same bone the frame came
    # from, or the hair would swim against its own placement once the head moves.
    hair_at = A.hair_frame(built.skeleton)
    hair_bone = A.HAIR_PARENT_BONE if hair_at is not None else "cf_j_head"
    for p in list(built.model.parts):
        if p.group == "hair" and str(p.name).startswith("hair_"):
            built.model.remove(p.name)
            report.replaced.append(p.name)

    for si, part in enumerate(hparts[:4]):
        PROG.tick(f"頭髮：slot {si}…", force=True)
        hid = int(part.get("id") or 0)
        if hid < 0:
            continue
        resolved = R.resolve_hair(si, hid, res=res)
        # id 0 goes through the resolver like any other: only the list row says
        # whether a slot is empty (CLAUDE.md 七, and see :func:`R.resolve_hair`).
        # ``hair_back`` has no "なし" row — its id 0 is a hairstyle.
        if is_empty_slot(resolved):
            continue                      # the card chose "none" for this slot
        item = _load_resolved(resolved, res, missing=report.missing)
        if item is None:
            report.missing.append(resolved.reason or f"hair slot {si} id {hid}")
            continue
        if resolved.reason:
            # the card's own hairstyle is not here; the game draws the slot's
            # default instead (``R.HAIR_DEFAULT_ID``) and so do we — out loud
            report.missing.append(resolved.reason)
        colors = _part_colors(part, fallback=card_hair_rgb)
        rejected: list = []
        # ``FindGameObject``: Hair is ``chaCtrl.GetHair(slot)``, and hair rows
        # carry no coordinate filter — the four slots are the whole key.
        hair_pieces = _material_editor(fit, item.pieces, ME.HAIR, si,
                                       f"hair slot {si}")
        _graft_hair_chains(built, mapper, item, hair_pieces, hair_bone, hair_at, part, report, si)
        n, tris = A.attach_pieces(
            built, hair_pieces, mapper=mapper, group="hair",
            name_prefix=f"origin_{resolved.category}",
            colors=colors, body_scale=body_scale, unit=unit,
            default_bone=hair_bone, attach_frame=hair_at,
            frame_absolute=hair_at is not None,
            hair_mat=hmat,
            category=resolved.category, head_len=head_len, rejected=rejected,
            gloss=gloss, res=res, hair_part=part)
        report.missing.extend(rejected)
        _report_hair_length(report, hair_pieces, part.get("length"), f"hair slot {si}")
        if not any("_hairComp" in (getattr(pc, "shader", None) or {})
                   for pc in hair_pieces):
            # the outline colour and the tie colours need the component's
            # lists, which archives baked before §7.446 do not carry
            report.missing.append(
                f"hair slot {si}（{item.label}）：封存沒有 ChaCustomHairComponent 的名單"
                "（bake 早於 §7.446，要重轉）—— 外框色與髮飾色沒套")
        _report_unsolved_sway(report, item)
        if n:
            report.attached.append(Attached(
                f"{item.label} ×{n}", resolved.category, item.source, tris))
        else:
            report.missing.append(f"{resolved.category}: {item.label} 無可用片")


def _stage_clothes(fit: _Fit, outfit: int | None) -> None:
    """All nine slots, every mesh piece."""
    built, report, mapper = fit.built, fit.report, fit.mapper
    unit, body_scale, head_len, res = (
        fit.unit, fit.body_scale, fit.head_len, fit.res)
    card = fit.card
    if outfit is None:
        outfit = current_outfit(card)
    of = O.parse_outfit(card, int(outfit))
    # Which ``objClothes`` are on is ``ChaControl.UpdateVisible``, one rule per
    # slot (:func:`outfit.slot_active`, §7.369): the slot's own
    # ``clothesState``, the two shoe slots by ``shoesType``, and the three
    # flags one garment sets to hide another (:func:`_not_flags`).  The bra is
    # loaded whenever that says so — a top's ``OverBraMask`` does not remove
    # it, it cuts it (``_cut_bra``, §7.367).
    states = O.clothes_state(card)
    # kept on the character: undress moves garments between these (§7.387)
    built.clothes_state = list(states) + [0] * max(0, 9 - len(states))
    shoes = O.shoes_type(card)
    flags = _not_flags(of, res, int(outfit), card)
    hidden = []
    # The pair ``shoesType`` does not name is never put on; everything else
    # ``UpdateVisible`` switches off is loaded and hidden, the way Origin loads
    # every slot and ``SetActive(false)``s it — so undress can put it back on.
    unworn = {n for k, n in O.SHOES_SLOT_FOR_TYPE.items() if k != shoes}
    # Overlay Mods' clothes overlays for this coordinate (``KCOX``)
    overlays = _overlays(fit, int(outfit))
    if overlays.reason:
        report.missing.append(f"服裝覆蓋層：{overlays.reason}")
    if overlays.unapplied:
        report.missing.append(
            f"服裝覆蓋層：{'、'.join(overlays.unapplied)} 還沒有套（遮罩／色遮罩覆寫）")
    for slot in of.clothes:
        active = O.slot_active(int(slot.index), states, shoes, flags)
        if not active:
            if slot.name in unworn:
                continue
            hidden.append(slot.name)
        PROG.tick(f"服裝：{slot.name}…", force=True)
        cat = O.LIST_CATEGORY.get(slot.name, slot.name)
        field = O.UAR_FIELDS.get(slot.name, "")
        resolved = R.resolve_clothes(
            slot.name, slot.item_id, field=field, outfit=int(outfit),
            sub_parts=of.sub_parts, res=res)
        if is_empty_slot(resolved):
            continue                      # the card chose "none" for this slot
        item = _load_resolved(resolved, res, missing=report.missing)
        if item is None:
            report.missing.append(
                resolved.reason or f"{slot.name}#{slot.item_id}")
            continue
        # ``ChangeCustomEmblem(clothesKind, no)`` reads
        # ``nowCoordinate.clothes.parts[clothesKind]``'s two emblem ids and
        # gives the texture to whatever that garment's ``ChaClothesComponent``
        # lists as an emblem renderer.  In practice only the top has any
        # (``ChangeClothesTopPartsAsync`` is the only caller, with kind 0), but
        # the id is per slot, so it is read per slot rather than assumed.
        item.pieces = M.apply_emblem(
            item.pieces, (slot.raw.get("emblemeId"),
                          slot.raw.get("emblemeId2")),
            res=res,
            props=tuple(f"outfit{of.coordinate}.ChaFileClothes.{p}"
                        for p in O.UAR_EMBLEM_FIELDS[slot.name])
            if slot.name in O.UAR_EMBLEM_FIELDS else None)
        # A skirt hangs on ``cf_j_sk_*``, which the body armature carries, so
        # the garment retargets onto *our* bones and never sees the prefab's
        # own settled chain.  Push the solve onto the skeleton instead — the
        # same thing the game's DynamicBone does when it writes back to the
        # character's transforms.  Must happen before the retarget below.
        if item.sway:
            moved = _apply_sway(built.skeleton, item.sway)
            if moved:
                report.replaced.append(
                    f"搖擺骨: {slot.name} {moved} 根骨頭沉降")
                # what the per-frame solve needs to take this settle back out
                # and run the chains against the pose (§7.540)
                built.sway_deltas.update(item.sway)
                # a spec without ``end`` was baked before the particles were
                # ``AppendParticles``'s (``bake.SWAY_REV`` 3): not solved
                if item.sway_spec and all("end" in c for c in item.sway_spec):
                    # the slot says which object the components sit on (``ct_clothesBot`` for
                    # the bottom), which the collider plugin keys on (``sway_live``)
                    built.sway_specs.extend({**c, "slot": slot.name} for c in item.sway_spec)
                else:
                    report.replaced.append(
                        f"搖擺骨: {slot.name} 的封存沒有逐格參數（舊轉檔），裙子不會跟著姿勢動；"
                        f"`bake_assets.py --sway` 重轉")
        rejected = []
        # tops must not settle (tears the torso mesh); skirts get a light drape
        # No settle — leave bind pose as Illusion authored it
        layer_boost = {"top": 8, "bottom": 8, "bra": 6, "panty": 6}.get(
            slot.name, 7)
        # ``UpdateVisible``'s state nodes (§7.379): every renderer is loaded —
        # Origin loads the whole prefab and ``SetActive``s the nodes — and the
        # ones this card's ``clothesState`` switches off are hidden below, so
        # undress can move a garment between states (§7.387).
        worn_pieces = list(item.pieces)
        # ``DrawOption``: the garment's objOpt01 / objOpt02 renderers are
        # switched off by the card's hideOpt bytes (§7.379).  An item baked
        # before the field existed carries no ``_clothOpt`` and keeps them.
        hide = O.hidden_options(of, slot)
        if any(hide):
            kept = [p for p in worn_pieces
                    if not (int((p.shader or {}).get("_clothOpt", 0) or 0) in (1, 2)
                            and hide[int(p.shader["_clothOpt"]) - 1])]
            if len(kept) != len(worn_pieces):
                report.replaced.append(
                    f"{slot.name}: 卡片關掉選項（hideOpt {list(hide)}）"
                    f"，{len(worn_pieces) - len(kept)} 片不畫")
            worn_pieces = kept
        # ``DrawSode``: the top's objSleeves01/02/03, of which the card's
        # ``sleevesType`` is drawn (§7.509).  It reads ``cusClothesCmp[0]``,
        # the top's own component — a combo top's parts are ``cusPartsCmp``.
        if slot.index == 0 and resolved.kind != "combo":
            kept = O.sleeve_pieces(worn_pieces, slot)
            if len(kept) != len(worn_pieces):
                report.replaced.append(
                    f"{slot.name}: 袖長 sleevesType {O.sleeves_type(slot)}，"
                    f"另外兩組袖子 {len(worn_pieces) - len(kept)} 片不畫")
            worn_pieces = kept
        # ``FindGameObject``: Clothing is ``chaCtrl.GetClothes(slot)``, and the
        # slot is the ``objClothes`` index — ``ClothesSlot.index`` here.
        worn_pieces = _material_editor(fit, worn_pieces, ME.CLOTHING,
                                       int(slot.index), f"{slot.name} 服裝")
        n, tris = A.attach_pieces(
            built, worn_pieces, mapper=mapper, group="clothes",
            name_prefix=f"origin_{slot.name}_{slot.index}",
            colors=slot.colors, body_scale=body_scale, unit=unit,
            default_bone="cf_j_hips",
            category=slot.name, head_len=head_len, rejected=rejected,
            layer=layer_boost, res=res,
            overlays=overlays, slot_index=int(slot.index),
            pattern_prop=(f"outfit{of.coordinate}.ChaFileClothes."
                          f"{O.UAR_PATTERN_FIELDS[slot.name]}"
                          if slot.name in O.UAR_PATTERN_FIELDS else None))
        report.missing.extend(rejected)
        _report_unsolved_sway(report, item)
        if n:
            report.attached.append(Attached(
                f"{item.label} ×{n}", cat, item.source, tris))
            if slot.name == "bra":
                _cut_bra(fit, slot, resolved, worn_pieces, of, int(outfit),
                         states)
            prefix = f"origin_{slot.name}_{slot.index}"
            for part in built.model.parts:
                if not part.name.startswith(prefix):
                    continue
                part.origin_slot_index = int(slot.index)
                part.visible = active and O.state_node_visible(
                    int(slot.index), getattr(part, "origin_state_node", ""),
                    states)
        else:
            report.missing.append(f"{slot.name}: {item.label} 全部片失敗")
    if hidden:
        report.replaced.append(
            f"衣服：UpdateVisible 關掉 {'、'.join(hidden)}（載入、不畫）"
            f"（clothesState {states[:9]}"
            f"{'、notBra' if flags.bra else ''}{'、notBot' if flags.bot else ''}"
            f"{'、notShorts' if flags.shorts else ''}）")


def _stage_body_mask(fit: _Fit, outfit: int) -> None:
    """``customMatBody._AlphaMask`` — the skin the worn top cuts away.

    The maps are **kept on the built character**, not just consumed here.
    Taking the top off has to put the skin back (``ChangeClothesTopAsync``
    assigns null to ``_AlphaMask`` when there is no top), and putting it on
    again has to cut the same holes — so undress needs both directions, which
    means it needs the maps.  Without them the only reachable direction was
    "restore", and a re-dress left the chest uncut (§7.133).
    """
    try:
        of = O.parse_outfit(fit.card, int(outfit))
        masks = _body_masks(of, fit.res, int(outfit),
                            missing=fit.report.missing)
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.dress._stage_body_mask: of = O.parse_outfit(fit.card, int(outfit))", exc)
        return
    # Overlay Mods' ``BodyMask``: with a top on, every ``SetTexture`` of the
    # body's ``_AlphaMask`` is answered with the card's image instead, and
    # ``ChangeClothesTopAsync`` makes that call whether or not the row names a
    # mask (:mod:`clothes_overlay`) ⇒ it replaces the list, not adds to it.
    card_mask = CO.mask_bytes(_overlays(fit, int(outfit)).by_id.get("BodyMask"))
    if card_mask is not None and _top_worn(of, fit.res, int(outfit)):
        masks = [card_mask]
        fit.report.replaced.append("身體遮罩：卡片的 Overlay Mods BodyMask")
    try:
        fit.built.body_masks = list(masks or ())
    except AttributeError:                           # noqa: BLE001
        pass
    if masks:
        states = O.clothes_state(fit.card)
        top = max(0, min(int(states[0]) if states else 0, 3))
        apply_body_mask(fit.built, masks, sex=sex_of(fit.card),
                        alpha_ab=ALPHA_BY_TOP_STATE[top])


def _hide_hair(resolved) -> bool:
    """``GetInfoInt(ChaListDefine.KeyType.HideHair)`` for a resolved accessory.

    One column, read from the item's own list row.
    ``UpdateAccessoryMoveFromInfo`` branches on it before it looks at the card
    at all, so a hat sits where the prefab puts it and the maker's sliders do
    nothing.  The same column also switches the wearer's hair off, under a
    **different test** and through a different call — see :func:`_hides_hair`.

    **It is not a boolean, and the test is ``== 1``, not "non-zero".**  The IL
    is ``GetInfoInt(24)`` → ``ldc.i4.1`` → ``bne.un``, and the column takes
    three values on this machine: ``1`` on 1 base row (the swim cap) and 6
    sideloader items, but ``2`` on 72 more (71 ``acc_head`` + 1 ``acc_face``).
    Reading it as a flag would silently throw away the card's offset on those
    72 — which is why the count is written down here rather than "78 hats".
    """
    return _hide_hair_col(resolved) == 1


def _hide_hair_col(resolved) -> int:
    """The raw ``HideHair`` column for a resolved accessory (0 if unreadable)."""
    ref = getattr(resolved, "ref", None)
    if ref is None:
        return 0
    raw = (ref.extra.get("HideHair") if hasattr(ref, "extra") else None)
    try:
        return int(str(raw or 0))
    except (TypeError, ValueError):
        return 0


def _hides_hair(resolved) -> bool:
    """Does this accessory switch the wearer's hair off?

    **Not the same test as** :func:`_hide_hair`, and that is the whole point of
    keeping them apart.  One column, two comparisons, transcribed from the IL::

        SetHideHairAccessory:          hideHairAcs[i] = GetInfoInt(24) != 0
        UpdateAccessoryMoveFromInfo:   if GetInfoInt(24) == 1: …

    ``brfalse`` against the raw value in the first, ``ldc.i4.1`` / ``bne.un`` in
    the second.  So the 72 items that carry a **2** hide the hair but still take
    the card's manual offset, and reading either test as "the other one" is
    wrong on a different set of items each way.
    """
    return _hide_hair_col(resolved) != 0


def _acc_copy_weights(resolved) -> int:
    """``<ChangeAccessoryAsync>``'s ``copyWeights``, from the item's own row.

    ::

        weight = 0; trfParent = null
        if ("null" == lib.GetInfo(ChaListDefine.KeyType.Parent))   # 54
            { weight = 2; trfParent = this.objTop.transform }

    Two things worth writing down, because the obvious readings are both wrong.
    There is **no ``weight`` column** — the value is computed from ``Parent``,
    the same cell :func:`_stage_accessories` uses as the default attach node,
    and ``ChaListDefine.KeyType`` has no ``Weight`` member at all.  And a row
    whose ``Parent`` is the literal ``"null"`` does not merely skip
    reparenting: it is the *skinned* case, rebound onto the head bones and left
    hanging off ``objTop``.  ``"0"`` is a different sentinel again, read by
    ``ChangeAccessoryParent`` (it returns early, leaving the item where
    ``LoadCharaFbxData`` put it) and it is not this test.

    Locally neither string appears: 0 of 357 base plus 12,168 zipmod accessory
    rows.  So this is always 0 here, which is why it is read rather than
    assumed — the assumption is the part that would rot.
    """
    ref = getattr(resolved, "ref", None)
    if ref is None:
        return 0
    # ``ModRef`` promotes the column to a field, ``AssetRef`` keeps it in
    # ``extra``; take whichever is populated.
    parent = getattr(ref, "parent", "") or (
        (ref.extra.get("Parent") or "") if hasattr(ref, "extra") else "")
    return 2 if str(parent).strip() == "null" else 0


def _material_editor(fit: _Fit, pieces, obj_type: int, slot: int, label: str):
    """This object's MaterialEditor rows: switched-off renderers, then materials.

    The renderer rows drop pieces (:func:`_hide_renderers`); the float /
    colour / shader / queue rows then rewrite the survivors' material dicts
    (``matedit.edit_pieces``) before ``attach`` turns them into materials.
    A material name the card edits that no piece carries is reported, for the
    same reason an unmatched renderer name is.
    """
    pieces = _hide_renderers(fit, pieces, obj_type, slot, label)
    edits = getattr(fit, "edits", None)
    if not edits:
        return pieces
    pieces, written, unmatched, counts = ME.edit_pieces(
        pieces, edits, obj_type, slot)
    if counts["copies"]:
        fit.report.replaced.append(
            f"MaterialEditor：{label} 複製 {counts['copies']} 份材質（多畫一趟）")
    if counts["uncopied"]:
        fit.report.missing.append(
            f"MaterialEditor: {label} 有 {counts['uncopied']} 份材質副本要畫的 submesh"
            "沒有被任何材質畫到、沒載入，沒有畫")
    if counts["bad_textures"]:
        fit.report.missing.append(
            f"MaterialEditor: {label} 有 {counts['bad_textures']} 張換圖在卡片的"
            " TextureDictionary 裡找不到或解不開")
    if written:
        fit.report.replaced.append(
            f"MaterialEditor：{label} 材質覆寫 {written} 個值")
    if unmatched:
        fit.report.missing.append(
            f"MaterialEditor: {label} 指名要改的材質 "
            f"{'、'.join(unmatched[:4])} 在這個 prefab 上找不到")
    return pieces


def _hide_renderers(fit: _Fit, pieces, obj_type: int, slot: int, label: str):
    """Drop the pieces this card's MaterialEditor section switched off.

    Filtering the *pieces* rather than the finished parts is what keeps the
    slot in scope: ``FindGameObject`` resolves ``(ObjectType, Slot)`` to one
    object and only that object's renderers are matched, so a name that two
    accessories share must not take both (``ObjectType.Character`` rows are the
    deliberate exception — the game hands those the whole character).

    A name the card asks to hide that no piece carries is **reported**, not
    ignored: in the game those rows are no-ops too (``SetRendererEnabled``
    finds nothing and returns false), but the two reasons a name can be absent
    — an orphaned row, or us resolving a different item than the card meant —
    look identical from here, and only one of them is fine (§一之一).

    Before the card's rows, the prefab's own: a renderer Unity never draws
    (an inactive node nothing switches on, or ``m_Enabled`` 0 with no
    ``Enabled 1`` row) is dropped too (``matedit.unity_drawn``, §7.509).
    """
    hidden = getattr(fit, "hidden", None)
    pieces, inactive, disabled = ME.unity_drawn(pieces, hidden, obj_type, slot)
    if inactive or disabled:
        fit.report.replaced.append(
            f"{label}：prefab 裡 Unity 不畫的 {inactive + disabled} 片不畫"
            f"（節點關閉 {inactive}、renderer 關閉 {disabled}）")
    if not hidden:
        return pieces
    keep, dropped, unmatched, nameless = ME.filter_pieces(
        pieces, hidden, obj_type, slot)
    if dropped:
        fit.report.replaced.append(
            f"MaterialEditor：{label} 關掉 {dropped} 片 renderer")
    if nameless:
        # Not an orphaned row — this item's archive predates the renderer name
        # and nothing here can match.  Says which, because "re-bake" and
        # "ignore" are the two answers and they are not interchangeable.
        fit.report.missing.append(
            f"MaterialEditor: {label} 的 bake 沒有 renderer 名（格式過舊），"
            f"{len(unmatched)} 筆關閉指令套不上，請重轉 bake")
    elif unmatched:
        fit.report.missing.append(
            f"MaterialEditor: {label} 指名要關的 "
            f"{'、'.join(unmatched[:4])} 在這個 prefab 上找不到")
    return keep


def _stage_accessories(fit: _Fit, outfit: int = 0) -> None:
    """The worn coordinate's non-empty slots, with the card's addMove.

    ``ChangeAccessory`` walks slots 0..19 of ``nowCoordinate`` — one coordinate,
    the one ``fileStatus.coordinateType`` selects — so the accessories belong to
    the outfit the rest of ``dress_character`` is already dressing.  Taking the
    union of all seven put a swimming coordinate's goggles on a character in
    uniform (§7.66).
    """
    built, report, mapper = fit.built, fit.report, fit.mapper
    unit, body_scale, head_len, res = (
        fit.unit, fit.body_scale, fit.head_len, fit.res)
    # ``ChangeAccessoryColor``'s tail writes ``fileHair.parts[0].startColor``
    # into all three colour slots of an accessory's ``rendHair`` renderers, so
    # the hair a swim cap or a wig brings with it matches the wearer.
    hair_rgb = _hair_start_rgb(fit.card)
    # ``CheckHideHair`` and ``UpdateVisible`` both read ``showAccessory``
    # after Accessory States rewrote the bound slots (:mod:`acc_states`)
    shown = AS.card_shown(fit.card)
    trig, reasons = AS.triggers(fit.card, int(outfit))
    report.missing.extend(reasons)
    built.acc_shown, built.acc_triggers = shown, trig
    states = list(getattr(built, "clothes_state", None) or [])
    hide_hair = False
    # HairAccessoryCustomizer (:mod:`hair_acc`): per-slot recolouring of the
    # hair accessories, against the wearer's ``hair.parts[0]`` and gloss sheet
    hacc, hacc_reason = HA.read(fit.card, int(outfit))
    if hacc_reason:
        report.missing.append(f"髮飾自訂：{hacc_reason}")
    hparts = O.hair_parts(fit.card)
    hair0 = hparts[0] if hparts and isinstance(hparts[0], dict) else {}
    hacc_gloss = (TOON.gloss_map(TOON.card_gloss_id(fit.card), res)
                  if any(i.get("HairGloss") for i in hacc.values()) else None)
    for acc in O.parse_outfit(fit.card, int(outfit)).accessories:
        resolved = R.resolve_accessory(
            acc.type_no, acc.item_id, res=res,
            acc_index=acc.index, coordinate=acc.coordinate)
        if is_empty_slot(resolved):
            continue                      # the card chose "none" for this slot
        item = _load_resolved(resolved, res, missing=report.missing)
        if item is None:
            report.missing.append(
                resolved.reason or f"acc {acc.type_no}/{acc.item_id}")
            continue
        # ``SetHideHairAccessory`` fills its array from the accessories the card
        # names and ``CheckHideHair`` ands it with ``showAccessory``:
        #
        #     for i in range(len(hideHairAcs)):
        #         if hideHairAcs[i] and fileStatus.showAccessory[i]:
        #             return True
        #
        # — so a hat the card has switched off must not take the hair with it.
        idx = int(getattr(acc, "index", -1) or 0)
        on = AS.slot_visible(idx, shown, trig, states)
        if _hides_hair(resolved) and on:
            hide_hair = True
        bone, frame = A.accessory_frame(
            built.skeleton, acc.parent_key, mapper.scale)
        # ``HideHair`` is ``ChaListDefine.KeyType`` 24 and only the item's own
        # list row carries it: a hat takes no manual offset at all, both move
        # nodes go to identity (``UpdateAccessoryMoveFromInfo``'s first branch).
        # A prefab with no such node drops that group, and so does the game:
        # ``FindLoop`` returns null, ``objAcsMove[slot, i]`` is null, and
        # ``UpdateAccessoryMoveFromInfo``'s loop starts with
        # ``if (null == objAcsMove[slot, i]) continue``.  Only 24 of the 357
        # base accessories even have an ``N_move2``, while 327 of 4,603 sampled
        # card slots (100 cards, all seven coordinates) write a non-identity
        # second group — most of those land on a prefab that has nowhere to put
        # it.  The card keeps the number, the game ignores it, and so do we.
        # Not a missing item.
        move = A.AccessoryMove(
            item.moves,
            A.card_move_locals(acc.raw.get("addMove"),
                               hide_hair=_hide_hair(resolved)))
        rejected = []
        before = len(built.model.parts)
        pieces = _material_editor(fit, item.pieces, ME.ACCESSORY, idx,
                                  f"acc {acc.type_no}/{acc.item_id}")
        n, tris = A.attach_pieces(
            built, pieces, mapper=mapper, group="accessory",
            name_prefix=f"origin_acc_{acc.type_no}_{acc.item_id}",
            colors=acc.colors, body_scale=body_scale, unit=unit,
            default_bone=bone,
            attach_frame=frame, frame_absolute=True,
            copy_weights=_acc_copy_weights(resolved),
            move=move, no_shake=bool(acc.raw.get("noShake")),
            category="accessory", head_len=head_len, rejected=rejected,
            hair_rgb=hair_rgb, res=res,
            hair_part=hair0, hair_acc=hacc.get(idx), gloss=hacc_gloss)
        report.missing.extend(rejected)
        # the slot, so undress can switch it with its garment (:mod:`acc_states`)
        for part in built.model.parts[before:]:
            part.origin_acc_slot = idx
            part.visible = on
        if not on:
            report.replaced.append(
                f"配件 slot {idx}（{item.label}）："
                + ("綁在衣服狀態上，這個狀態不顯示" if idx in trig
                   else "卡片的 showAccessory 關掉"))
        _report_hair_acc(report, hacc.get(idx), pieces, item.label)
        _report_unsolved_sway(report, item)
        if n:
            report.attached.append(Attached(
                f"{item.label} ×{n}", "accessory", item.source, tris))
        else:
            report.missing.append(f"accessory: {item.label} 無可用片")

    if hide_hair:
        # ``UpdateVisible``'s hair loop, transcribed:
        #
        #     hide = CheckHideHair()
        #     for i in range(len(objHair)):
        #         SetActiveControl(objHair[i], visibleAll and <body visible>
        #                          and not hide and not visibleSimple
        #                          and visibleHeadAlways and visibleBodyAlways)
        #
        # so ``CheckHideHair()`` true deactivates **every**
        # ``objHair[i]``.  A helmet or a swim cap is authored to replace the
        # hair, not to sit on top of it, so drawing both is what pushed strands
        # through the brim on every card wearing one.  The accessory's own
        # ``rendHair`` renderers are group "accessory" and unaffected — that is
        # how a wig keeps its hair while the wearer's is switched off.
        n = 0
        for part in list(built.model.parts):
            if str(getattr(part, "group", "")) == "hair":
                built.model.remove(str(part.name))
                n += 1
        if n:
            report.replaced.append(f"HideHair 配件：頭髮 {n} 片關閉")


def pushup_for(card, outfit: int, report=None):
    """``(pushup_live.Inputs | None, pair)`` for *card* dressed in *outfit*.

    ``CurrentlyWearing`` asks ``IsClothesStateKind(kind)`` — the slot holds a
    garment (``AddClothesStateKind`` registers every loaded one) — and
    ``clothesState[kind] == 0``; *pair* is that answer for the bra and the top
    under the card's stored states.  Undress moves it later
    (:func:`pushup_live.follow`).
    """
    of = O.parse_outfit(card, outfit)
    res = SL.card_resolutions(card)

    def worn(name: str) -> bool:
        slot = next((x for x in of.clothes if x.name == name), None)
        if slot is None:
            return False
        r = R.resolve_clothes(name, slot.item_id, field=O.UAR_FIELDS.get(name, ""),
                              outfit=outfit, sub_parts=of.sub_parts, res=res)
        return not is_empty_slot(r)

    inp = PL.inputs(card, outfit, report, worn)
    if inp is None:
        return None, (False, False)
    return inp, PL.counting(inp, O.clothes_state(card))


def shape_rig(built, card, rig_scale: float, *, outfit: int, report=None,
              pushup=None) -> None:
    """:func:`apply_card_shape` with *card*, keeping what Pushup and the maker need later.

    *pushup* is :func:`pushup_for`'s answer and *card* already its
    ``card_view``.  With Pushup data the bare graft is copied first, so undress
    can shape the other bra / top combinations on it (``built.pushup``); the
    maker's card shape sliders reshape from their own copy
    (``reshape_live``: ``built.bare_graft`` / ``built.shaped_rest``).
    """
    from . import reshape_live as RL
    inp, pair = pushup if pushup else (None, (False, False))
    live = PL.begin(built.skeleton, inp, pair, rig_scale) if inp is not None else None
    # the maker's card shape sliders reshape from the same bare graft (#12)
    RL.begin(built)
    apply_card_shape(built, card, rig_scale, report=report, coordinate=int(outfit))
    RL.settle(built)
    if live is not None:
        PL.settle(live, built.skeleton)
    built.pushup = live


def _report_hair_acc(report, info, pieces, label: str) -> None:
    """What of a slot's HairAccessoryCustomizer settings did not reach the screen."""
    if info is None:
        return
    roles = [(getattr(p, "shader", None) or {}).get("_hairComp") for p in pieces]
    if all(r is None for r in roles):
        report.missing.append(
            f"髮飾自訂：{label} 的封存沒有 ChaCustomHairComponent 的名單"
            "（bake 早於 §7.446，要重轉）—— 沒套")
        return
    if "hair" not in roles:
        return                  # not a hair accessory: the plugin returns early
    report.replaced.append(f"髮飾自訂：{label}")
    _report_hair_length(report, pieces, info.get("HairLength"), f"髮飾自訂：{label}")


def _report_hair_length(report, pieces, rate, label: str) -> None:
    """A non-zero ``lengthRate`` the item's archive cannot carry out.

    ``attach.HairLength`` moves ``trfLength`` from the ``_hairLen`` groups the
    bake records (§7.516); an archive from before that has none, and the hair
    keeps its authored length — which looks like a card asking for nothing.
    """
    if not rate:
        return
    if any((getattr(p, "shader", None) or {}).get("_hairLen") is not None for p in pieces):
        return
    if any((getattr(p, "shader", None) or {}).get("_hairComp") is not None for p in pieces):
        report.missing.append(
            f"{label} 的長度 {float(rate):.2f} 沒有畫：封存早於 §7.516（沒有 trfLength），要重轉")


def _overlays(fit: "_Fit", outfit: int) -> "CO.ClothesOverlays":
    """The card's clothes overlays for *outfit*, decoded once per dress."""
    cache = getattr(fit, "_overlay_cache", None)
    if cache is None:
        cache = {}
        try:
            fit._overlay_cache = cache
        except AttributeError:                       # noqa: BLE001
            pass
    if outfit not in cache:
        cache[outfit] = CO.read(fit.card, int(outfit))
    return cache[outfit]


def _top_worn(of, res, outfit: int) -> bool:
    """``objClothes[0] != null``: the card's top slot names a garment."""
    top = next((s for s in of.clothes if s.name == "top"), None)
    if top is None:
        return False
    resolved = R.resolve_clothes(
        "top", top.item_id, field=O.UAR_FIELDS.get("top", ""), outfit=outfit,
        sub_parts=of.sub_parts, res=res)
    return not is_empty_slot(resolved)


def _bra_masking_refs(of, res, outfit: int) -> list:
    """List rows for the worn top, including a p_dummy top's three sub-parts.

    The sub-parts are resolved **the same way** :func:`load_mesh.load_combo_top`
    resolves them — the card's own UAR entry first (``ClothesJacketSubA`` and
    friends, pinned to the sub-part's ``CategoryNo``), then the base table.
    They used to be looked up in the base table alone, which is a different
    garment's row whenever a sub-part is a zipmod: on ``Shinomiya Natsuha`` the
    jacket's ``subPartsId`` is ``[505, 0, 1]`` and 505 is a modded vest, so the
    lookup came back empty and the top contributed **no** ``OverBodyMask``.
    Nothing reported that — the mesh loads through the other path, so the vest
    is on screen and only the skin underneath it is missing its cut-out, which
    shows up as the wearer's nipples drawn on top of the garment (§7.74).
    """
    top = next((s for s in of.clothes if s.name == "top"), None)
    if top is None:
        return []
    resolved = R.resolve_clothes(
        "top", top.item_id, field=O.UAR_FIELDS.get("top", ""), outfit=outfit,
        sub_parts=of.sub_parts, res=res)
    if resolved.kind == "combo":
        from .. import lists as L
        from ..install import configured as _cfg
        # the loader's own resolution (``combo_sub_refs``), so the mask and the
        # mesh always come from the same row (§7.74)
        subs = M.combo_sub_refs(resolved.item_id, resolved.sub_parts, res,
                                L.index(_cfg()))
        return [s.mod or s.base for s in subs if (s.mod or s.base) is not None]
    return [resolved.ref] if resolved.ref is not None else []


def _body_masks(of, res, outfit: int, missing: list | None = None) -> list:
    """The ``OverBodyMask`` maps the worn garments contribute.

    Same rows the bra rule reads, different column — and a combo top supplies
    them from its three sub-parts rather than from the ``p_dummy`` row.

    Loaded through :func:`load_mesh._ref_texture`, **not** ``_named_texture``:
    a zipmod's row names a bundle inside its own archive, which the install-only
    lookup can never open.  ``Shinomiya Natsuha``'s vest asks for
    ``chara/ymd/mf_clothVariation.unity3d / mf_bodymask_hv`` and came back empty,
    so the skin under the garment kept its full alpha (§7.74).
    """
    from .load_mesh import _ref_texture
    out = []
    for r in _bra_masking_refs(of, res, outfit):
        spec, tex = _ref_texture(r, "OverBodyMaskAB", "OverBodyMask")
        if spec is None:
            continue                       # the row names no mask: normal
        if tex is None:
            if missing is not None:
                missing.append(
                    f"body mask {spec[0]}/{spec[1]} 載不到"
                    f"（{getattr(r, 'name', '?')}）")
            continue
        # ``_AlphaMask`` is a cut-out the forward pass hard-thresholds **after**
        # the fetch, so an sRGB-flagged mask is thresholded on its decoded
        # value (:func:`colorspace.sampled_bytes`, §7.386).
        out.append(CS.sampled_bytes(tex))
    return out


#: ``ChaControl.UpdateVisible_array4`` — ``byte[4, 2]`` built from
#: ``$field-555B10D7…`` in the constructor and read as
#: ``ChangeAlphaMask(array4[clothesState[0], 0], array4[clothesState[0], 1])``:
#: the ``_alpha_a`` / ``_alpha_b`` every masked material gets, **by the top's
#: state**.  Bytes ``01 01 00 01 00 01 00 00`` → dressed (1, 1), half (0, 1),
#: state 2 (0, 1), off (0, 0).  ``origin_il_dump --data`` prints the blob.
ALPHA_BY_TOP_STATE = ((1, 1), (0, 1), (0, 1), (0, 0))

#: ``ChangeClothesBraAsync``: the two renderers ``OverBraMask`` is handed to —
#: ``GetReferenceInfo(ObjBraDef / ObjBraNuge)`` → ``o_bra_a`` / ``o_bra_b``,
#: found by **GameObject** name, which is ``_rendName`` and not the mesh name
#: (the body suits' meshes are ``o_body_allbodytights_*`` under ``o_bra_a``).
BRA_MASKED_RENDERERS = ("o_bra_a", "o_bra_b")

#: ``SetTextureOffset`` / ``SetTextureScale`` on ``_AlphaMask`` for those two,
#: indexed by ``flag = row.Coordinate == 2``, overridden by the row's
#: ``MabUV`` (KeyType 105) when it has one: ``flag = MabUV == "1"``.
BRA_MASK_ST = (((0.0, 0.0), (1.0, 1.0)), ((0.0, -1.0), (1.0, 2.0)))


def _not_flags(of, res, outfit: int, card=None) -> "O.NotFlags":
    """``notBra`` / ``notBot`` / ``notShorts`` for this outfit (:class:`outfit.NotFlags`).

    Read off the same resolved rows the loader uses, so a modded top answers
    from its own list row.  A combo top decides ``notBra`` from its parts —
    sailor from part 0, jacket from part 0 or part 1 — and ``notBot`` comes
    from ``ChangeClothesTopAsync`` only, which a combo top does not run.
    """
    flags = O.NotFlags()
    if O.clothing_unlocked(card, outfit):
        return flags                     # KK_ClothingUnlocker (§7.447)

    def row(name):
        s = next((x for x in of.clothes if x.name == name), None)
        if s is None:
            return None
        r = R.resolve_clothes(name, s.item_id, field=O.UAR_FIELDS.get(name, ""),
                              outfit=outfit, sub_parts=of.sub_parts, res=res)
        return None if is_empty_slot(r) else r

    def extra(r):
        ref = getattr(r, "ref", None) if r is not None else r
        return (getattr(ref, "extra", None) or {}) if ref is not None else {}

    top = row("top")
    if top is not None and top.kind == "combo":
        kind, _cats = M.COMBO_TOP_CATS.get(int(top.item_id), ("", ()))
        parts = _bra_masking_refs(of, res, outfit)
        pick = parts[:1] if kind == "sailor" else parts[:2] if kind == "jacket" else []
        flags.bra = any((getattr(r, "extra", None) or {}).get("NotBra") == "1"
                        for r in pick)
    elif top is not None:
        flags.bra = extra(top).get("NotBra") == "1"
        flags.bot = extra(top).get("Coordinate") == "2"
    flags.shorts = extra(row("bra")).get("Coordinate") == "2"
    return flags


def _bra_mask(of, res, outfit: int, missing: list):
    """``texBraAlphaMask`` and its wrap mode, or ``(None, None)``.

    Each top part that loads assigns it, so with a combo top the **last** part
    whose row names one is what the bra is cut with — the loader's order.  A
    row that names a mask we cannot open, or whose wrap mode we cannot read,
    is reported and the bra left uncut: the address mode decides the texels
    outside the unit square, and 126 local bra renderers have some.
    """
    from .load_mesh import _ref_texture, ref_texture_wrap
    got = (None, None)
    for r in _bra_masking_refs(of, res, outfit):
        spec, tex = _ref_texture(r, "OverBraMaskAB", "OverBraMask")
        if spec is None:
            continue
        wrap = ref_texture_wrap(r, "OverBraMaskAB", "OverBraMask")
        if tex is None or wrap is None:
            what = "貼圖" if tex is None else "wrap mode"
            missing.append(f"bra mask {spec[0]}/{spec[1]} 讀不到{what}"
                           f"（{getattr(r, 'name', '?')}）")
            got = (None, None)
            continue
        # thresholded on the sampled value, like every ``_AlphaMask`` (§7.386)
        got = (CS.sampled_bytes(tex), wrap)
    return got


def _cut_bra(fit: "_Fit", slot, resolved, pieces, of, outfit: int,
             states: list) -> None:
    """``ChangeClothesBraAsync`` + ``ChangeAlphaMask`` for the attached bra.

    The top's ``OverBraMask`` goes on ``o_bra_a`` / ``o_bra_b``'s
    ``_AlphaMask`` with :data:`BRA_MASK_ST`, and ``_alpha_a/_b`` come from the
    top's state (:data:`ALPHA_BY_TOP_STATE`).  The shader reads it at
    ``lerp(uv0, uv1, _AlphaMaskuv) * scale + offset`` — ``_AlphaMaskuv`` is
    ``main_opaque2``'s one addition over ``main_opaque`` — so the cut is baked
    into each part's albedo through the mesh (:func:`mesh.uvraster.rasterise`),
    not by resizing the mask onto the albedo.
    """
    from ...mesh import uvraster as UVR
    report = fit.report
    mask, wrap = _bra_mask(of, fit.res, outfit, report.missing)
    # Overlay Mods' ``BraMask`` answers the bra renderers' ``_AlphaMask``
    # the same way ``BodyMask`` answers the body's (:mod:`clothes_overlay`);
    # a ``Texture2D`` from ``LoadImage`` keeps the default wrap, Repeat.
    card_mask = CO.mask_bytes(_overlays(fit, outfit).by_id.get("BraMask"))
    if card_mask is not None and _top_worn(of, fit.res, outfit):
        mask, wrap = card_mask, 0
        report.replaced.append("bra 遮罩：卡片的 Overlay Mods BraMask")
    if mask is None:
        return
    row = getattr(resolved, "ref", None)
    extra = (getattr(row, "extra", None) or {}) if row is not None else {}
    flag = extra.get("Coordinate") == "2"
    if "MabUV" in extra:
        flag = extra.get("MabUV") == "1"
    (ox, oy), (sx, sy) = BRA_MASK_ST[1 if flag else 0]
    top = states[0] if states else 0
    a, b = ALPHA_BY_TOP_STATE[max(0, min(int(top), 3))]
    k_of = {str((p.shader or {}).get("_rendName") or ""):
            float((p.shader or {}).get("_AlphaMaskuv", 0.0) or 0.0)
            for p in pieces}
    prefix = f"origin_{slot.name}_{slot.index}"
    cut = 0
    for part in fit.built.model.parts:
        if not part.name.startswith(prefix):
            continue
        rn = getattr(part, "origin_renderer", "")
        if rn not in BRA_MASKED_RENDERERS:
            continue
        key = part.material.texture
        alb = TEX.pixels_of(key)
        mesh = part.mesh
        if alb is None or mesh is None or not mesh.vertex_count:
            report.missing.append(f"bra: {rn} 沒有可挖的 albedo")
            continue
        k = k_of.get(rn, 0.0)
        uv = np.asarray(mesh.uvs, np.float64)
        if k:
            uv1 = getattr(mesh, "uv1", None)
            if uv1 is None or not len(uv1):
                report.missing.append(
                    f"bra: {rn} 的 _AlphaMaskuv={k} 但網格沒有 UV1")
                continue
            uv = (1.0 - k) * uv + k * np.asarray(uv1, np.float64)
        uv = uv * (sx, sy) + (ox, oy)
        field, covered = UVR.rasterise(mesh.uvs, mesh.indices, {"uv": uv},
                                       alb.shape[:2])
        out = TEX.apply_alpha_mask_uv(alb, mask, field["uv"], covered,
                                      a, b, repeat=(wrap == 0))
        new = TEX.store(out, srgb=bool(key[2]) if len(key) > 2 else True,
                        repeat=key[0] == "originasset_repeat")
        if new is None:
            continue
        part.material.texture = new
        part.material.alpha_test = TEX.ALPHA_CLIP
        cut += 1
    if cut:
        report.replaced.append(
            f"bra: 上衣的 OverBraMask 挖 {cut} 片（上衣狀態 {top} ⇒ "
            f"a/b {a}/{b}{'、ST (1,2,0,-1)' if flag else ''}）")


def is_empty_slot(resolved) -> bool:
    """A slot the card deliberately left empty, not one that failed.

    Illusion ships "なし" entries as real list rows pointing at ``p_dummy``,
    and mods do the same (``yamadamod.nohairB`` is a no-back-hair item at id
    500).  Reporting those as missing buried the genuine failures: 35 of 60
    cards "failed" to load hair they had explicitly chosen not to have.
    The combo top is the one ``p_dummy`` that does mean something, and it is
    resolved as ``kind == "combo"`` before this is consulted.
    """
    if resolved is None or resolved.kind == "combo":
        return False
    return M.names_nothing(getattr(resolved, "ref", None))


def _load_resolved(resolved: R.Resolved, res=None, missing: list | None = None):
    """Load one resolved item.

    ``missing`` collects the maps the item's list row names but that could not
    be loaded.  It is the report's channel for the failure that has no other
    symptom: a garment with no albedo still draws, so without this the only
    evidence is a flat-coloured shirt nobody can explain (§一之一).
    """
    if resolved.kind == "combo":
        return M.load_combo_top(resolved.item_id, resolved.sub_parts, res,
                                missing=missing)
    if resolved.kind == "base" and resolved.ref is not None:
        ref = resolved.ref
        return M.load_base(ref.bundle, ref.asset, label=ref.name,
                           category=resolved.category, ref=ref,
                           missing=missing)
    if resolved.kind == "mod" and resolved.ref is not None:
        return M.load_mod(resolved.ref, category=resolved.category,
                          ref=resolved.ref, missing=missing)
    return None


def _graft_hair_chains(built, mapper, item, pieces, hair_bone: str, hair_at, part, report, slot: int) -> None:
    """Put a hair prefab's swaying chains on the armature, settled as the bake has them (B4b).

    The game runs the hair's ``DynamicBone`` every frame on the prefab's own
    bones; the bake settled them once and froze the piece.  With the bones
    grafted (:func:`attach.graft_hair_bones`) the piece is skinned to them, the
    load-time settle goes onto them the way a skirt's does (``_apply_sway``),
    and the chains join the ones :class:`sway_live.SwayLive` solves against the
    pose.  A hair without solvable chains, or whose bone names the armature
    already has, keeps the frozen bake.
    """
    chains = [c for c in (getattr(item, "sway_spec", None) or ()) if "end" in c and not c.get("unsolved")]
    if not chains or hair_at is None:
        return
    added = A.graft_hair_bones(built.skeleton, mapper, pieces, chains, hair_bone, hair_at,
                               A.hair_length_rate(True, part, None))
    if not added:
        report.replaced.append(f"hair slot {slot}: 骨頭名稱已在骨架上，髮鏈維持載入時沉降的樣子")
        return
    _apply_sway(built.skeleton, item.sway)
    built.sway_deltas.update(item.sway)
    # the components sit on the prefab's root, which hangs on *hair_bone* with identity local:
    # that is the object whose travel and scale they read (``sway_live``)
    built.sway_specs.extend({**c, "object": hair_bone} for c in chains)
    report.replaced.append(f"hair slot {slot}: {len(added)} 根頭髮骨頭接上骨架，{len(chains)} 條髮鏈逐格解算")


def _report_unsolved_sway(report, item) -> None:
    """The ``DynamicBone`` trees the bake could not solve, which the game sways.

    ``prefab.sway_chain_specs`` marks them ``unsolved`` with the reason; such a
    tree hangs in the prefab's rest instead of settling, and is not solved per
    frame (§7.540).
    """
    for c in getattr(item, "sway_spec", None) or ():
        if c.get("unsolved"):
            root = next((b for b in c.get("bones") or () if b), "?")
            report.missing.append(f"搖擺骨: {item.label} 的 {root} 這條沒有解算 —— {c['unsolved']}")


def _apply_sway(skel, deltas: dict) -> int:
    """Post-multiply each named bone's local rotation by the solved delta.

    ``deltas`` is expressed in each bone's own parent frame, which is the frame
    the local rotation lives in, so this composes without needing the prefab's
    hierarchy to match ours bone-for-bone — only the names have to line up, and
    for a skirt they do (``cf_j_sk_NN_MM`` on both sides).

    Bones the armature does not have are skipped rather than mapped onto
    something near them: a chain the body has never heard of belongs to the
    piece's own prefab and is handled there (``Piece.bone_rest``).
    """
    from ...core import math3d as m3
    n = 0
    for name, rot in (deltas or {}).items():
        if not skel.has(name):
            continue
        b = skel.bones[skel.id(name)]
        m = np.eye(4)
        m[:3, :3] = np.asarray(rot, dtype=np.float64)
        b.local_rot = m3.quat_mul(b.local_rot, m3.quat_from_mat4(m))
        n += 1
    if n:
        skel.finalize()
    return n


def _hair_rgb(parts: list) -> tuple:
    for p in parts:
        c = p.get("baseColor")
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            return tuple(float(x) for x in c[:3])
    return (0.86, 0.84, 0.82)


def _hair_start_rgb(card) -> tuple:
    """``fileHair.parts[0].startColor`` — what an accessory's hair takes.

    Slot 0 specifically, not "the first slot that has a colour": the tail of
    ``ChangeAccessoryColor`` indexes ``parts[0]``, so a character with no back
    hair still lends a wig its own start tone.
    """
    parts = O.hair_parts(card)
    p = parts[0] if parts else {}
    for key in ("startColor", "baseColor"):
        c = p.get(key) if isinstance(p, dict) else None
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            return tuple(float(x) for x in c[:3])
    return _hair_rgb(parts)


def _part_colors(part: dict, fallback=None) -> list:
    out = []
    for key in ("baseColor", "startColor", "endColor", "outlineColor"):
        c = part.get(key)
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            out.append({"baseColor": list(c[:4]) if len(c) >= 4
                        else list(c[:3]) + [1.0]})
    for ac in part.get("acsColor") or []:
        if isinstance(ac, (list, tuple)) and len(ac) >= 3:
            if len(ac) > 3 and float(ac[3]) < 0.01:
                continue
            out.append({"baseColor": list(ac[:4]) if len(ac) >= 4
                        else list(ac[:3]) + [1.0]})
    if not out and fallback is not None:
        hc = list(fallback[:3]) if isinstance(fallback, (list, tuple)) else list(fallback)
        out.append({"baseColor": hc + [1.0]})
    return out
