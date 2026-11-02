"""Base body + complete head (face, dual iris, brows, lashes, tongue)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import colorspace as CS
from .. import progress as PROG
from .. import textures as TEX
from ..prefab import PrefabLoader
from ..install import configured
from . import create_tex as CT
from . import lewdcrest as LC
from . import skin_overlay as SO
from .attach import apply_shader, make_material
from ...core import swallowed

BODY_BUNDLE = "chara/oo_base.unity3d"
# ``<LoadAsync>c__Iterator0.MoveNext``: one bundle, the prefab picked by sex —
# ``sexFemale ? "p_cf_body_00" : "p_cm_body_00"`` (and the ``_low`` pair for
# the non-hiPoly path, which this loader does not use).  ``sexFemale`` is
# ``sex != 0``; the card's own byte is **0 = male, 1 = female**, confirmed
# against the install's UserData/chara/male and /female cards.
#
# Keys here are the *project's* sex convention, which is the other way round
# (``chara_dir(0)`` is female) — cardimport flips it on the way in.
BODY_PREFAB = {0: "p_cf_body_00", 1: "p_cm_body_00"}
# ``CreateBodyTexture``: the base albedo is **not** sex-switched (one string,
# ``cf_body_00_t``), only the colour mask is.
BODY_SKIN = "cf_body_00_t"
BODY_SKIN_MASK = {0: "cf_body_00_mc", 1: "cm_body_00_mc"}
# The draw material lives in mm_base, not on the prefab; there is no
# ``cm_m_body_create`` — the create material is shared.
BODY_MATERIAL = {0: "cf_m_body", 1: "cm_m_body"}
BODY_MESH = "o_body_a"
# Illusion ships genitals as separate skinned meshes on the same body prefab.
# We used to skip them and draw procedural stand-ins (policy §五, pre-2026-08).
# They are now loaded from the user bake the same way as ``o_body_a`` — our
# format only, never the raw ``.unity3d``.  ``o_gomu`` is the condom: a man's body carries it hidden
# (``BODY_GOMU``) and ``set_condom`` shows it while ``ChaFileStatus.visibleGomu`` is on.
GENITAL_MESHES = {
    0: ("o_mnpa", "o_mnpb"),                 # female outer / inner
    1: ("o_dankon", "o_dan_f"),              # male shaft / glans family
}
# Vanilla ``o_mnpa``/``o_mnpb`` are ~48/69 verts (censor stubs).  UncensorSelector
# bodies fold the real geometry into ``o_body_a`` (8.3k → 9–10k verts).  Male
# penis densification is a separate ``o_dankon`` mesh (SoS ~4k verts).
# Preference: densest common uncensor bodies (Nam mid_slit first).  BP packs
# that implant ``cf_J_Vagina_*`` were tried for Free H open and caused mesh
# warp — not preferred here until a safe path exists (§7.122–§7.125).
# Paths are relative to a zipmod root or ``assets_cache/mods/...``.
_FEMALE_BODY_UNCENSOR = (
    # (zipmod name substring, asset path inside zip / under mods)
    #
    # BetterPenetration first, and **not** because it is the densest — Nam is
    # (10,484 vs 9,727 vertices on ``o_body_a``).  It is first because it is
    # the only pack that ships the 18 skinned ``cf_J_Vagina_*`` bones and the
    # 13 ``DynamicBone`` chains on them, which is the whole of the crotch
    # opening (§7.128).  Both clear §五之一's floor, so this is a capability
    # preference, not a quality one.
    #
    # **It only works together with the implant.**  Our armature is grafted
    # from ``oo_base``'s reference rig, which has never heard of those bones,
    # and an unknown name is not skipped — ``skinmap._resolve``'s ``exact``
    # branch sends it to ``cf_j_hips``.  All 18 landing on the hips is what
    # drew two wedges from the groin to the feet and is what actually broke
    # §7.123 / §7.125 (§7.129).  ``body_implant_bones`` + the graft in
    # ``apply_body`` are the other half; do not enable one without the other.
    ("BP Profundis", "abdata/chara/oo_base_bpprofundis.unity3d"),
    ("Nam Outlined Pussy Pack", "abdata/chara/oo_base_Nam_OL_mid_slit.unity3d"),
    ("SAC] Uncensor Selector Pack", "abdata/chara/oo_base_sacmeaty.unity3d"),
    ("moderchan] Moderchan Uncensor", "abdata/chara/oo_base_moderchan.unity3d"),
    ("KK_LO_pussy", "abdata/chara/oo_base_KK_LO.unity3d"),
    ("KK_UncensorSelector Base", "abdata/chara/oo_base.unity3d"),
)
_MALE_PENIS_UNCENSOR = (
    ("SoS Uncensor", "abdata/chara/deathweasel/sos_dick.unity3d"),
    ("Cartoon Uncensor", "abdata/chara/cartoon_uncensor.unity3d"),
)
#: ``UncensorSelector..cctor``'s ``PenisParts``: the renderers ``ReloadCharacterPenis``
#: takes out of the penis prefab, each onto the body renderer of the same name
#: (``UpdateMeshRenderer``: mesh, bones and materials, unconditionally, §7.498).
PENIS_PARTS = ("o_dankon", "o_gomu", "cm_o_dan00")
#: ``UncensorSelector..cctor``'s ``BallsParts``: what ``ReloadCharacterBalls`` takes out of the balls
#: prefab onto the body's renderers of the same name (§7.1183).
BALLS_PARTS = ("o_dan_f", "cm_o_dan_f")
#: ``UncensorSelector..cctor``: ``ReloadCharacterBody`` copies a pack renderer of one of these names onto
#: the character's own — the meshes of ``BodyNames`` without their materials, ``BodyParts`` with them (§7.1191).
BODY_NAMES = ("o_body_a", "o_body_cf", "o_body_cm")
BODY_PARTS = ("o_dankon", "o_dan_f", "o_gomu", "o_mnpa", "o_mnpb", "o_shadowcaster", "cm_o_dan00", "cm_o_dan_f")
# Bake acceptance (CLAUDE.md §五之一).  Vanilla female o_body_a ≈ 8302,
# male o_dankon ≈ 257.  Uncensor packs land well above these floors.
UNCENSOR_BODY_A_MIN = 8500       # female o_body_a must clear this
UNCENSOR_DANKON_MIN = 1000       # male o_dankon must clear this

HEAD_BUNDLE = "chara/bo_head_00.unity3d"
# Not sex-switched: no ``p_cm_head`` exists anywhere in the install, and the
# base ``bo_head`` rows only ever name ``p_cf_head_*``.  Males wear this one.
# Only the *default* now (list id 0) — the head the card asks for comes from
# ``head_assets`` below.
HEAD_PREFAB = "p_cf_head_00"


@dataclass
class HeadAssets:
    """Everything one ``bo_head`` row names, resolved for a card.

    ``ChaControl.ChangeHead`` is ``ChangeHeadAsync(fileFace.headId)``, and the
    row it looks up carries five separate things.  Treating the head as a
    constant got four of them wrong at once for any card that changed it:
    the mesh, the curve asset the face sliders play (**ShapeAnime**), the face
    material, and the albedo/colour-mask pair the composite is built from.
    """
    ident: int
    bundle: str
    prefab: str
    shape_bundle: str
    shape_asset: str
    mat_bundle: str
    mat_asset: str
    tex: str
    mask: str
    source: str = "base"          # base | mod
    mod = None                    # ModRef when source == "mod"

    @property
    def bake_ident(self):
        """Bake key — must carry the head, or two heads share one entry.

        §7.33 lost a whole category this way once: a key missing a segment
        made one item serve another.  §7.475 did it again with the segment
        named: a modded head's ``ident`` is the **mod's own slot**, which
        authors number from 1, so nine local head mods were all
        ``head_mod_1`` and 302 cards wore whichever one was converted first.
        A mod head is keyed the way every other mod item is
        (``load_mesh.mod_bake_key``: GUID, category, slot).
        """
        if self.source == "base" and self.ident == 0:
            return "head"
        if self.source == "mod" and self.mod is not None:
            from .load_mesh import mod_bake_key
            key = mod_bake_key(self.mod)
            if key:
                return f"head_mod_{key}"
        return f"head_{self.source}_{self.ident}"


def _row_columns(row, bundle_default: str) -> dict:
    """``MainAB``/``MainData``/… out of a base AssetRef or a zipmod ModRef."""
    extra = dict(getattr(row, "extra", None) or {})
    def col(*names):
        for n in names:
            v = str(extra.get(n, "") or "").strip()
            if v and v not in ("0", "-", "abdata"):
                return v
        return ""
    return {
        "bundle": col("MainAB") or bundle_default,
        "prefab": col("MainData") or getattr(row, "asset", ""),
        "shape": col("ShapeAnime"),
        "mat_ab": col("MatAB"),
        "mat": col("MatData"),
        "tex_ab": col("MainTexAB"),
        "tex": col("MainTex"),
        "mask_ab": col("ColorMaskAB"),
        "mask": col("ColorMaskTex"),
    }


def _stock_head() -> HeadAssets:
    """Head 0's row, which is also what a modded row falls back to column by column."""
    return HeadAssets(0, HEAD_BUNDLE, HEAD_PREFAB, HEAD_BUNDLE,
                      "cf_anmShapeHead_00", HEAD_BUNDLE, "cf_m_face_00",
                      "cf_face_00_t", "cf_face_00_mc")


def head_assets(card=None, res: dict | None = None) -> HeadAssets:
    """Resolve the card's ``headId``; falls back to the stock head with no card.

    A head the card names but the machine does not have is **not** silently
    replaced -- the caller turns a raised error into ``report.missing``.  A
    head is the face; substituting one is substituting the character.
    """
    stock = _stock_head()
    face = {}
    if card is not None and hasattr(card, "face"):
        try:
            face = card.face() or {}
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.figure.body_head.head_assets: face = card.face() or {}", exc)
            face = {}
    if not isinstance(face, dict):
        face = {}
    try:
        ident = int(face.get("headId", 0) or 0)
    except (TypeError, ValueError):
        ident = 0
    if ident == 0 and card is None:
        return stock
    return head_assets_for(ident, res)


def head_assets_for(ident: int, res: dict | None = None) -> HeadAssets:
    """:func:`head_assets` for a ``headId`` with no card around it.

    *res* is the card's UAR map when there is one (a modded head is named by
    it); without it the id resolves against the base list.  The converter uses
    this to bake every head a machine can wear, not only head 0 (§7.472).
    """
    stock = _stock_head()
    from .resolve import resolve_head
    got = resolve_head(ident, res=res)
    if got.kind == "missing":
        raise FileNotFoundError(got.reason)
    row = got.ref
    c = _row_columns(row, getattr(row, "bundle", HEAD_BUNDLE))
    if not c["prefab"]:
        raise FileNotFoundError(f"headId {ident} 的清單列沒有 MainData")
    b = c["bundle"]
    out = HeadAssets(
        ident, b, c["prefab"],
        b, c["shape"] or stock.shape_asset,
        c["mat_ab"] or b, c["mat"] or stock.mat_asset,
        c["tex"] or stock.tex, c["mask"] or stock.mask,
        "mod" if got.kind == "mod" else "base")
    if got.kind == "mod":
        out.mod = row
        # A mod that names no ShapeAnime of its own is saying "head 00's
        # curves" -- it has to be, those are the sliders the card was built
        # with -- so the fallback above points back at the base bundle.
        if not c["shape"]:
            out.shape_bundle = stock.shape_bundle
    return out


def hi_poly() -> bool:
    """Illusion's ``ChaControl.hiPoly`` — the high-detail asset set.

    Not a distance LOD: ``set_hiPoly`` is called from ``ChaControl.Initialize``
    and ``ChaInfo.MemberInitializeAll`` and nowhere else, so it is a per
    character quality flag decided once.  Every ``_low`` asset in the game
    hangs off it — ``p_cf_body_00_low``, ``cf_body_00_t_low``,
    ``cf_face_00_mc_low`` — chosen as ``hiPoly ? "" : "_low"``.

    Default is on, which is what this loader has always done implicitly.
    """
    try:
        from ...core.paths import Config
        v = Config("startup").get("origin_hi_poly")
        return True if v is None else bool(v)
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.body_head.hi_poly: from ...core.paths import Config", exc)
        return True


def _lod(name: str) -> str:
    """``cf_body_00_t`` → ``cf_body_00_t_low`` when hiPoly is off."""
    return name if hi_poly() else name + "_low"


def sex_of(card, default: int = 0) -> int:
    """Project-convention sex (0 female, 1 male) for a card."""
    try:
        par = card.parameter if hasattr(card, "parameter") else {}
        raw = par.get("sex")
        if raw is None:
            return int(default)
        return 0 if int(raw) != 0 else 1      # card: 0 male, 1 female
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.body_head.sex_of: par = card.parameter if hasattr(card, 'parameter", exc)
        return int(default)
# Which head mesh plays which part is not guesswork and it is *not* what the
# names suggest.  ``ChaReference.CreateReferenceInfo`` maps RefObjKey → node
# name and ``ChaControl.<ChangeHeadAsync>`` hands those objects to the
# renderer fields:
#
#     64/65 ObjEyeL/R   cf_Ohitomi_L02 / _R02 → rendEye      the **iris**
#     66/67 ObjEyeWL/R  cf_Ohitomi_L   / _R   → rendEyeW     the **sclera**
#     60    ObjEyeline  cf_O_eyeline          → rendEyelineUp
#     61    ObjEyelineLow cf_O_eyeline_low    → rendEyelineDown
#     62    ObjEyebrow  cf_O_mayuge           → rendEyebrow
#     63    ObjNoseline cf_O_noseline         → rendNose
#
# The prefab agrees: ``cf_Ohitomi_L`` carries ``cf_m_sirome_00`` (白目) with
# ``cw_t_sirome`` already bound, while ``cf_Ohitomi_L02`` carries
# ``cf_m_hitomi_00`` whose ``_MainTex`` is assigned at run time.  Reading the
# "02" pair as a highlight overlay — what this loader used to do — put the
# iris on the sclera mesh, so the whole eye opening came out one flat colour
# with no white anywhere and the real iris mesh drew a stray grey patch.
IRIS_MESHES = ("cf_Ohitomi_L02", "cf_Ohitomi_R02")
WHITE_MESHES = ("cf_Ohitomi_L", "cf_Ohitomi_R")
HEAD_MESHES = (
    "cf_O_face", "cf_O_eyeline", "cf_O_eyeline_low", "cf_O_mayuge",
    "cf_O_noseline",
    "cf_Ohitomi_L", "cf_Ohitomi_R", "cf_Ohitomi_L02", "cf_Ohitomi_R02",
    "cf_O_tooth", "cf_O_canine", "o_tang",
    "cf_O_namida_S", "cf_O_namida_M", "cf_O_namida_L",
)

# ``ChaControl.UpdateVisible`` shows the three tear sheets off ``tearsLv``
# through ``UpdateVisible_array1[tearsLv, i]`` — a ``bool[4,3]`` literal in
# ``ChaControl..ctor`` (``000 100 010 001``) — and ``CreateReferenceInfo``
# names RefObjKey 121/122/123 ``cf_O_namida_S``/``_M``/``_L`` (§7.402): level
# 1/2/3 each shows one sheet, 0 shows none.  Drawn by ``toon_glasses_lod0``
# (§7.413).
TEARS_MESHES = ("cf_O_namida_S", "cf_O_namida_M", "cf_O_namida_L")
TEARS_VISIBLE = ((False, False, False), (True, False, False),
                 (False, True, False), (False, False, True))


#: ``ChaReference``'s ``S_TongueF`` / ``S_TongueB``: the head prefab's ``o_tang`` and the body prefab's
#: (``CreateReferenceInfo`` finds each by name in its own prefab), as the parts are named here
HEAD_TONGUE, BODY_TONGUE = "origin_head_o_tang", "origin_body_tongue_0"
#: the man's ``o_gomu`` (``ChaReference`` 91, ``UpdateVisible`` IL_024e–IL_0272: active while ``visibleGomu``),
#: drawn by its own ``toon_glasses_lod0`` material, under his genitals' own display
BODY_GOMU = "origin_body_gomu_0"


def set_condom(built, on: bool) -> bool:
    """``visibleGomu``: show or hide the man's condom (``Part.face_hidden``).  Returns whether he has one to show."""
    model = getattr(built, "model", None)
    part = model.get(BODY_GOMU) if model is not None else None
    if part is None:
        return False
    part.face_hidden = not on
    return True


def set_tongue_state(built, mouth_ptn: int) -> bool:
    """``ChangeMouthPtn``'s ``ChangeTongueState`` and ``UpdateVisible``: which tongue shows for a mouth pattern.

    Pattern 21 sets ``tongueState`` 1, which shows the body's tongue and
    hides the head's; every other pattern sets 0, the other way round
    (state 2, ``objTongueEx``, nothing in the game sets).  Returns whether
    the body's tongue shows.  Writes ``Part.face_hidden``, which the scene
    passes that show whole groups leave alone.
    """
    from .face_morph import tongue_out
    out = tongue_out(mouth_ptn)
    model = getattr(built, "model", None)
    head, body = (model.get(HEAD_TONGUE), model.get(BODY_TONGUE)) if model is not None else (None, None)
    if head is not None:
        head.face_hidden = out
    if body is not None:
        body.face_hidden = not out
    return out and body is not None


def set_tears(built, level: int) -> int:
    """``tearsLv`` → which tear sheet shows (``Part.face_hidden``).  Returns sheets found (0..3)."""
    row = TEARS_VISIBLE[max(0, min(3, int(level)))]
    n = 0
    for part in getattr(getattr(built, "model", None), "parts", None) or ():
        mesh = str(getattr(part, "origin_renderer", "") or "")
        if mesh in TEARS_MESHES:
            part.face_hidden = not row[TEARS_MESHES.index(mesh)]
            n += 1
    return n

#: the head prefab's teeth: the row of teeth and the fangs (``ObjDoubleTooth``, RefObjKey 69)
TEETH_MESHES = ("cf_O_tooth", "cf_O_canine")


def set_teeth_hidden(built, hidden: bool, mouth_ptn: int = 0) -> int:
    """Put a card head's teeth and its tongue away while her mouth holds his shaft, or show them again.

    Not the game's: its mouth does not open to the shaft's girth and draws the
    teeth and the head's tongue through it (§7.1325); the owner chose to put
    them away (2026-10-23, 「含住時藏起牙齒」).  Shown again, the tongue is the
    one *mouth_ptn* shows (:func:`set_tongue_state`).  Writes
    ``Part.face_hidden``; returns the teeth parts found.
    """
    model = getattr(built, "model", None)
    n = 0
    for part in getattr(model, "parts", None) or ():
        if str(getattr(part, "origin_renderer", "") or "") in TEETH_MESHES:
            part.face_hidden = bool(hidden)
            n += 1
    if hidden:
        head = model.get(HEAD_TONGUE) if model is not None else None
        if head is not None:
            head.face_hidden = True
    else:
        set_tongue_state(built, mouth_ptn)
    return n


# mesh → (list category, id field, colour field).  Every one of these is a
# ``ChangeTexture(renderer, GetListInfo(category, id), _MainTex)`` call, so the
# bundle and asset name come from the list row, never from the asset's name.
FACE_FEATURES = {
    "cf_O_eyeline": ("eyeline_up", "eyelineUpId", "eyelineColor"),
    "cf_O_eyeline_low": ("eyeline_down", "eyelineDownId", "eyelineColor"),
    "cf_O_mayuge": ("eyebrow", "eyebrowId", "eyebrowColor"),
    # cf_m_noseline_00 keeps _Color black and the map is a black line in the
    # alpha, so there is no card colour to apply here.
    "cf_O_noseline": ("nose", "noseId", None),
}
# The sclera is opaque in Origin (cf_m_sirome_00: _Mode 0, _ZWrite 1); everything
# else on the face is a cut-out sheet.
FACE_CUTOUT = {
    "cf_O_eyeline", "cf_O_eyeline_low", "cf_O_mayuge", "cf_O_noseline",
    "cf_Ohitomi_L02", "cf_Ohitomi_R02",
}
# These sheets are nearly coplanar, so their order has to be stated rather than
# left to depth — and Illusion states it, in the materials' ``m_CustomRenderQueue``
# (which overrides the shader's queue tag).  Read off bo_head_00:
#
#     2350  cf_m_face_00        main_skin  (AlphaTest-100)
#     2450  cf_m_tang/tooth     main_item  (AlphaTest)
#     2900  cf_m_mayuge_00      eyebrow
#     2901  cf_m_sirome_00      sclera
#     2902  cf_m_eyeline_kage / cf_m_hitomi_00   lid shadow, iris
#     2903  cf_m_eyeline_00_up / cf_m_eyeline_down
#     3000  cf_m_noseline_00    (shader tag Transparent)
#     4010  cf_m_namida_00      (Transparent+1010)
#
# Two of those were the wrong way round here before: the eyebrow drew *last*
# and the nose line drew before the lashes.  Second material slots take a step
# back so the lid shadow lands at 2902 alongside the iris, not with the lashes.
# ``ChaForegroundComponent.Initialize`` — the ten node names it looks up, and
# which of the two lists each goes in.  Written out as literals in the IL, so
# this is the whole set; a name test on "eye" would also catch ``cf_O_eyeline``'s
# lid-shadow slot, which is not in it.
#
# Each of the ten carries a ``SetRenderQueue_Custom`` whose ``m_queueDatas``
# holds exactly **one** entry, ``id 0`` — the renderer's *first* material.  So
# on ``cf_O_eyeline`` the lash (2803/2903) moves and ``cf_m_eyeline_kage``, the
# lid shadow in slot 1, stays at 2902 whatever the setting is.  Read off
# ``bo_head_00``'s own components:
#
#     cf_O_mayuge                       2800 / 2900
#     cf_Ohitomi_L, cf_Ohitomi_R        2801 / 2901
#     cf_Ohitomi_L02, _R02, gag_eye_*   2802 / 2902
#     cf_O_eyeline, cf_O_eyeline_low    2803 / 2903
#
# and the front hair sits at **2850**, between the two blocks — which is the
# whole mechanism (see :func:`toon.foreground`).
FOREGROUND_EYEBROW = frozenset(("cf_O_mayuge",))
FOREGROUND_EYES = frozenset((
    "cf_O_eyeline", "cf_O_eyeline_low",
    "cf_Ohitomi_L", "cf_Ohitomi_R", "cf_Ohitomi_L02", "cf_Ohitomi_R02",
    "cf_O_gag_eye_00", "cf_O_gag_eye_01", "cf_O_gag_eye_02",
))

#: IllusionFixes' ``KK_Fix_HairShadows`` (``origin.plugins``)
HAIR_SHADOWS_FIX = "KK_Fix_HairShadows"


def foreground_slot(name: str, slot: int) -> bool:
    """Does material *slot* of renderer *name* carry a foreground queue?

    ``Hooks.SetRenderQueue_CustomAwake`` in ``KK_Fix_HairShadows.dll`` rewrites
    the queue data of the ten components (front hair 2850 → 2475 in
    ``LoadCharaFbxDataAsync``, the foreground set 2800‥2803 → 2472‥2474, so
    the order between them is unchanged) and, on ``cf_O_eyeline`` only, adds a
    second entry ``id 1, m_queues 2473, m_queuesBackup 2902``.
    """
    if slot == 0:
        return True
    if slot == 1 and name == "cf_O_eyeline":
        from .. import plugins as PLG
        return PLG.installed(HAIR_SHADOWS_FIX)
    return False


FACE_LAYER = {
    "cf_O_face": 2,
    "cf_O_tooth": 4, "cf_O_canine": 4, "o_tang": 4,
    "cf_O_mayuge": 10,
    "cf_Ohitomi_L": 12, "cf_Ohitomi_R": 12,
    "cf_Ohitomi_L02": 14, "cf_Ohitomi_R02": 14,
    "cf_O_eyeline_low": 16, "cf_O_eyeline": 16,
    "cf_O_noseline": 18,
    "cf_O_namida_L": 20, "cf_O_namida_M": 20, "cf_O_namida_S": 20,
}


# Which Sideloader UAR property owns each texture-only list, so a modded map
# resolves through the card's own resolution table instead of being looked up
# blind.  Collected from the cards themselves rather than guessed: every name
# here appears in ``card_resolutions`` across the local library.
UAR_PROP = {
    "eye": "ChaFileFace.Pupil{n}", "eye_gradation": "ChaFileFace.PupilGradient{n}",
    "eye_white": "ChaFileFace.whiteId",
    "eye_hi_up": "ChaFileFace.hlUpId", "eye_hi_down": "ChaFileFace.hlDownId",
    "eyeline_up": "ChaFileFace.eyelineUpId",
    "eyeline_down": "ChaFileFace.eyelineDownId",
    "eyebrow": "ChaFileFace.eyebrowId", "nose": "ChaFileFace.noseId",
    "mole": "ChaFileFace.moleId", "lipline": "ChaFileFace.lipLineId",
    "face_detail": "ChaFileFace.detailId",
    "body_detail": "ChaFileBody.detailId", "nip": "ChaFileBody.nipId",
    "underhair": "ChaFileBody.underhairId",
    "sunburn": "ChaFileBody.sunburnId", "body_paint": "ChaFileBody.PaintID{n}",
    "face_paint": "ChaFileMakeup.PaintID{n}", "lip": "ChaFileMakeup.lipId",
    "cheek": "ChaFileMakeup.cheekId", "eyeshadow": "ChaFileMakeup.eyeshadowId",
    # Off ``StructReference.ChaFileHairGenerator`` (``433 / glossId``, prefix
    # ``ChaFileHair``); 122 of 1,388 local cards carry it.
    "hairgloss": "ChaFileHair.glossId",
}


_TEXTURE_PROPS: list | None = None


def texture_category_of(prop: str) -> str | None:
    """The texture-only list a card's UAR property names a row of, or ``None``.

    The inverse of the properties :func:`_list_texture` asks with —
    :data:`UAR_PROP` for the face and body layers, and the garment templates
    in :mod:`outfit` for the pattern tiles (``mt_pattern``) and badges
    (``mt_emblem``).  One table, read both ways, so the converter bakes exactly
    the rows the dressing code will ask for (§7.474).
    """
    global _TEXTURE_PROPS
    if _TEXTURE_PROPS is None:
        import re
        from . import outfit as O
        pairs = list(UAR_PROP.items())
        pairs += [("pattern", p) for p in O.UAR_PATTERN_FIELDS.values()]
        pairs += [("emblem", p) for two in O.UAR_EMBLEM_FIELDS.values() for p in two]
        _TEXTURE_PROPS = [
            (re.compile(re.escape(p).replace(re.escape("{n}"), r"\d+") + "$"), cat)
            for cat, p in pairs]
    for rx, cat in _TEXTURE_PROPS:
        if rx.match(str(prop)):
            return cat
    return None


def _list_texture(category: str, ident, res: dict | None = None,
                  slot: int = 1, prop: str | None = None, add: str = ""):
    """A map named by one of Illusion's texture-only lists (mt_body_detail …).

    Returns a :class:`colorspace.Map` — the bytes **and** the asset's
    ``m_ColorSpace``.  The two travel together because this is the one
    resolution path for the row, and a second lookup for the flag would be the
    split §7.74 records; the flag is also unrecoverable downstream, where the
    map is just a PNG (:mod:`colorspace`).

    ``0`` is **not** a sentinel for "none" — whether it means anything is the
    list's business, and the lists themselves already say so: every feature
    everyone has (``mt_eye``, ``mt_eyeline_up``, ``mt_nose``, ``mt_nip`` …)
    ships an id 0 called タイプ１, while the optional layers (cheek, lip, mole,
    body paint …) simply have no row 0.  Rejecting id 0 up front therefore
    dropped the *first* iris, eyeline, brow and nose in the game — the default
    on a card that never touched those sliders.

    ``res`` is the card's own UAR table, and it is asked **first**: sideloader
    rewrites a named id to the mod's LocalSlot before anything reads it, so a
    base row of the same number never gets a turn (see
    :func:`resolve.uar_named`).  These lists collide the same way the garment
    ones do — over 200 local cards, 34 texture slots carry a mod id the base
    list also has, 29 of them installed, which used to come back as an
    unrelated stock map: 「ムキムキ」 for a slender skin type, a base heart
    stamp for a BodyWriting tattoo, タイプ２ irises for a custom pair (§7.72).
    The bake cache cannot arbitrate that, because ``bake_textures`` only ever
    walks the *base* lists, so ``tex/<cat>/<id>.png`` is by construction the
    stock map.

    *add* is ``GetTexture``'s ``addStr`` (:func:`eyeline_head_suffix`): the
    head's lookup answers before the row's own map (:func:`_head_lookup`).
    """
    try:
        ident = int(ident)
    except (TypeError, ValueError):
        return None
    if ident < 0:
        return None
    from .. import bake, lists as L, sideloader as SL
    from . import resolve as R
    from .load_mesh import _named_texture, mod_texture

    # ``prop`` lets a caller name the UAR property itself.  The patterns and
    # emblems need that: their property is not a function of the *texture*
    # list they live in (``mt_pattern`` 430, ``mt_emblem`` 431) but of the
    # garment wearing them — ``ClothesBraPattern2``, ``ClothesTopEmblem2``.
    prop = prop or UAR_PROP.get(category)
    gs = R.uar_named(prop.format(n=slot), ident, res) if prop else None
    if gs:
        mref = SL.index().resolve(gs[0], category, gs[1])
        if mref is not None:
            decided, got = _head_lookup(mref.bundle, mref.asset, add)
            return got if decided else mod_texture(mref)
        # Named but not installed: ``ResolveStructure`` sets the id to
        # 99999999 unless the stock row of that number keeps it
        # (:func:`resolve.keeps_list_row` — always, for the five lists with
        # no ``MainAB`` column), and ``GetListInfo(99999999)`` is null.
        if not R.keeps_list_row(category, ident):
            return None
    # 「なし」, and therefore nothing — after the UAR because that is the card
    # naming a mod (whose own slot number may well *be* the none id: the local
    # ``[Wubsy]Cow Print Patterns`` registers ``pattern`` **0**), and before
    # everything that derives from the base list.
    #
    # Every step below would otherwise answer for a row that names no asset.
    # The blind ``find_any`` scan at the end really did: a garment whose card
    # chose 花紋 = なし came back wearing that cow print, because the scan only
    # matches (category, id) and 「なし」 rows have an id like any other.  That
    # is the rule this file already carries for garments (查不到就報錯，
    # 不要抓一個同號的頂替) reaching a texture list, and our own generators
    # were doing the same thing one step earlier (§7.195).
    if L.index().is_none(category, ident):
        return None
    ref = L.index().get(category, ident)
    if add and ref is not None:
        # a provider the setting chose replaces the game's answers, and the
        # head's lookup is the game's answer before the converted stock sheet
        hit = bake.provided_texture(category, ident)
        if hit is not None:
            return hit
        decided, hit = _head_lookup(ref.bundle, ref.asset, add)
        if decided:
            return hit
    hit = bake.read_texture(category, ident)
    if hit is not None:
        return hit
    if ref is not None:
        return _named_texture(ref.bundle, ref.asset)
    if gs:
        return None          # nothing owns the id: the mod really is missing
    mref = SL.index().find_any(category, ident)
    if mref is None:
        return None
    decided, got = _head_lookup(mref.bundle, mref.asset, add)
    return got if decided else mod_texture(mref)


def _row_texture(category: str, ident, ab_key: str, tex_key: str,
                 res: dict | None = None, slot: int = 1,
                 bake_category: str | None = None,
                 missing: list | None = None, add: str = ""):
    """A *secondary* map named by the same list row.

    ``mt_eyeline_up`` carries two: ``EyelineUpTex`` for the lash and
    ``EyelineShadowTex`` for the eyelid shadow that ``cf_m_eyeline_kage``
    draws in the renderer's second material pass.

    **The row is resolved the way** :func:`_list_texture` **resolves it** — the
    card's UAR first, then 「なし」, then ``bake_category`` (the generated or
    converted sheet, when the caller has one), then the base row, then
    ``CompatibilityResolve``'s first claimant (``find_any``).  It used to read the base list only, which is the §7.72 mistake
    again one column over: the lash came from the mod the card names and the lid
    shadow from whichever base row shared its number.  Over 1,388 local cards
    **419** name an installed modded lash row with no base row of that number,
    so their eye-socket shade was dropped without a word, and 2 more collide
    with a base row and wore another lash's shadow (§7.356).

    ``missing`` gets a line when the row names a map that does not load; a row
    that names none is the item's business and is not reported.

    *add* is the head suffix, as for :func:`_list_texture`; the variant answers
    after a provider (``bake_category``) and before the row.
    """
    from .. import bake, lists as L, sideloader as SL
    from . import resolve as R
    from .load_mesh import _ref_texture
    try:
        ident = int(ident)
    except (TypeError, ValueError):
        return None
    if ident < 0:
        return None

    def of(ref, where: str):
        spec = ref.texture_ref(ab_key, tex_key) if add and hasattr(ref, "texture_ref") else None
        decided, got = _head_lookup(spec[0], spec[1], add) if spec else (False, None)
        if decided:
            return got
        spec, got = _ref_texture(ref, ab_key, tex_key)
        if spec is not None and got is None and missing is not None:
            missing.append(f"{category}#{ident} 的 {tex_key}（{where}）載入不到")
        return got

    prop = UAR_PROP.get(category)
    gs = R.uar_named(prop.format(n=slot), ident, res) if prop else None
    if gs:
        mref = SL.index().resolve(gs[0], category, gs[1])
        if mref is not None:
            return of(mref, gs[0])
        if not R.keeps_list_row(category, ident):
            return None      # named, not installed: cleared (§7.493)
    if L.index().is_none(category, ident):
        return None
    if bake_category:
        hit = bake.read_texture(bake_category, ident)
        if hit is not None:
            return hit
    ref = L.index().get(category, ident)
    if ref is not None:
        return of(ref, "基礎清單")
    if gs:
        return None          # named, not installed, and no base row owns the id
    mref = SL.index().find_any(category, ident)
    return None if mref is None else of(mref, mref.guid)


#: The lists whose setter hands ``GetTexture`` an ``addStr``: only
#: ``ChangeSettingEyelineUp`` (lash and lid shadow) and
#: ``ChangeSettingEyelineDown`` do; the other ten ``ChangeTexture`` setters pass
#: ``String.Empty`` (§7.517).
HEAD_VARIANT_LISTS = ("eyeline_up", "eyeline_down")


def head_variant_ref(bundle: str, name: str, add: str) -> tuple[str, str] | None:
    """``(bundle, texture)`` ``ChaControl.GetTexture`` tries for an ``addStr``.

    ``GetRemoveStringRight(ab.Replace(".unity3d", ""), 2) + "50.unity3d"`` and
    ``texKey + addStr``: ``chara/mt_eyeline_up_00.unity3d`` /
    ``cw_t_eyeline_up_010`` → ``chara/mt_eyeline_up_50.unity3d`` /
    ``cw_t_eyeline_up_010_201``.  ``None`` without a suffix.
    """
    if not add or not bundle or not name:
        return None
    stem = str(bundle).replace(".unity3d", "")
    stem = stem[:-2] if len(stem) > 2 else ""
    return f"{stem}50.unity3d", f"{name}{add}"


#: IllusionFixes' ``KK_Fix_ModdedHeadEyeliner`` (``origin.plugins``)
MODDED_HEAD_EYELINER_FIX = "KK_Fix_ModdedHeadEyeliner"


def _head_lookup(bundle: str, name: str, add: str) -> tuple[bool, object]:
    """``(decided, map)``: what ``GetTexture`` makes of an ``addStr``.

    ``decided`` false means "the row's own map, as without a suffix".  The
    original loads :func:`head_variant_ref`, then the unsuffixed name **from
    the same ``…50`` bundle** — which for a row in ``mt_eyeline_up_00`` is not
    there, so vanilla hands the renderer ``null`` and the lash vanishes
    (``(True, None)``).  ``KK_Fix_ModdedHeadEyeliner`` is a postfix on
    ``GetTexture``: a null result with a non-empty ``addStr`` is retried with
    ``""``, the row's own map — installed here, so on this machine it is the
    variant when the bundle has it and the row's map otherwise.
    """
    ref = head_variant_ref(bundle, name, add)
    if ref is None:
        return False, None
    from .. import plugins
    from .load_mesh import _named_texture
    got = _named_texture(*ref) or _named_texture(ref[0], name)
    if got is not None:
        return True, got
    return (False, None) if plugins.installed(MODDED_HEAD_EYELINER_FIX) else (True, None)


def eyeline_head_suffix(face: dict, res: dict | None = None) -> str:
    """``ChangeSettingEyelineUp/Down``'s ``addStr``: ``"_{headId}"``, or ``""``.

    The setters build ``string.Format("_{0}", fileFace.headId)`` whenever
    ``headId`` is not 0.  That id is read **after** Sideloader's resolve, so a
    modded head (or one it cleared to 99999999) carries a LocalSlot that no
    bundle names a variant for — only a head the base list answers keeps the
    card's own number.  Locally that is タイプ０２／０３ (200 and 201), whose
    lashes ``mt_eyeline_up_50`` / ``mt_eyeline_down_50`` redraw for every row
    (5 of 1,388 cards).
    """
    try:
        hid = int((face or {}).get("headId", 0) or 0)
    except (TypeError, ValueError):
        return ""
    if hid == 0:
        return ""
    from .resolve import resolve_head
    return f"_{hid}" if resolve_head(hid, res=res).kind == "base" else ""


def _list_says_none(category: str, ident) -> bool:
    """``mt_nose#4`` is 「なし」, not a lookup failure.

    Illusion makes "no nose line", "no lower lash" etc. real list rows whose
    ``MainAB`` is ``0`` — the same trick as the ``p_dummy`` clothing rows
    §7.14 had to learn about.  Reporting those buries the failures that matter.
    """
    from .. import lists as L
    return L.index().is_none(category, ident)


# ``cf_m_face_create`` / ``cf_m_body_create`` carry values of their own that
# ChaControl never overwrites — most importantly ``_tex4uv`` (0.5, 0.64,
# -0.33, -4.0) and ``_tex5uv`` (0.5, 0.745, -8, -20), which place the blush and
# the lip line.  Reading the *shader's* property default instead (0, 0, 1, 1)
# collapses both layers to a single texel, so they have to come off the
# material asset.
_CREATE_PROPS: dict = {}

MM_BASE = "chara/mm_base.unity3d"
#: Every ``(material, bundle)`` the stock passes read with :func:`create_props`;
#: the converter bakes these, and each head's own material besides (§7.482).
STOCK_MATERIALS = (("cf_m_face_create", MM_BASE), ("cf_m_body_create", MM_BASE),
                   ("cf_m_eye_create", MM_BASE), (BODY_MATERIAL[0], MM_BASE),
                   (BODY_MATERIAL[1], MM_BASE), ("cf_m_face_00", HEAD_BUNDLE))


def props_key(material: str) -> str:
    """Name a material's properties are baked under — ``<material>.mat``, the
    container's own file name, so it cannot meet a TextAsset of the same name."""
    return f"{material}.mat"


def props_env(bundle: str, material: str = "", mod=None):
    """The bundle :func:`props_in` reads, or ``None``.

    A stock row's is the install's file.  A mod row's is the one Sideloader
    hands out for ``(bundle, material)`` — the archives that registered the
    path in load order, then the install (§7.481) — which is where a modded
    head keeps its own face material (§7.484).
    """
    from .. import sideloader as SL
    if mod is not None:
        return SL.load_mod_bundle(mod, bundle=bundle, asset=material)
    if SL.overrides_at(bundle):
        # a stock material a mod overrides — ``[KK] Base Liquid Extension``
        # ships its own ``cf_m_body`` in ``chara/mm_base.unity3d`` (§7.486)
        return SL.open_asset_bundle(bundle, material)
    import UnityPy
    path = configured().bundle(bundle)
    return UnityPy.load(str(path)) if path.exists() else None


def props_source(mod=None) -> str:
    """Bake source a material's properties are filed under: the install or the mod row's GUID."""
    from .. import bake
    return bake.INSTALL if mod is None else str(getattr(mod, "guid", "") or "")


def material_request(head) -> tuple:
    """``(material, bundle, mod)`` the face material of *head* (a ``HeadAssets``) is read with.

    The row's ``MatData`` / ``MatAB``.  A mod head asks through its row — nine
    of the ten local mod heads keep the material in their own bundle, which
    the install does not have, so asking the install read nothing and the
    stock ``cf_m_face_00`` stood in (§7.484).  One owner for the draw pass and
    the converter.
    """
    return head.mat_asset, head.mat_bundle, (head.mod if head.source == "mod" else None)


def props_in(env, material: str) -> dict:
    """Saved colours / vectors / texture ST of material *material* in *env*."""
    out: dict = {}
    for obj in env.objects:
        if obj.type.name != "Material":
            continue
        d = obj.read()
        if str(d.m_Name) != material:
            continue
        # Same reason ``prefab._material_floats`` reads it: the
        # keywords pick the compiled variant, and for these shaders
        # that decides whether the forward base discards.
        try:
            kw = str(getattr(d, "m_ShaderKeywords", "") or "")
        except Exception as exc:             # noqa: BLE001
            swallowed.note("origin.figure.body_head.props_in: kw = str(getattr(d, 'm_ShaderKeywords', '') or…", exc)
            kw = ""
        if kw:
            out["_keywords"] = kw
        sp = d.m_SavedProperties
        for k, v in sp.m_Colors:
            out[str(k)] = (v.r, v.g, v.b, v.a)
        for k, v in sp.m_Floats:
            out[str(k)] = float(v)
        for k, v in sp.m_TexEnvs:
            out[f"{k}_ST"] = (v.m_Scale.x, v.m_Scale.y,
                              v.m_Offset.x, v.m_Offset.y)
            # ``_paintmask`` is the one texture nothing ever assigns:
            # ChaControl never touches it, so the map bound on the
            # material asset (cf_body_01_mp / cf_face_00_mp) is what
            # the shader samples.  It gates where paint may land.
            tex = getattr(v, "m_Texture", None)
            if tex is not None and getattr(tex, "path_id", 0):
                try:
                    td = tex.read()
                    img = td.image
                    cs = getattr(td, "m_ColorSpace", None)
                    # Same rule as ``prefab._decode``: a map with no
                    # ``m_ColorSpace`` is not given one.  ``_paintmask``
                    # goes straight into a create pass, where a wrong
                    # flag gates where paint may land.
                    if img is not None and cs is not None:
                        out[str(k)] = CS.Map(
                            np.asarray(img.convert("RGBA"),
                                       dtype=np.uint8),
                            int(cs) == CS.SRGB)
                except Exception as exc:     # noqa: BLE001
                    swallowed.note("origin.figure.body_head.create_props: tex.read", exc)
        # Which shader the material declares decides its toon family,
        # and ``cf_m_body`` is only reachable this way: it lives in
        # mm_base while the body prefab is in oo_base, so the prefab
        # loader resolves nothing and ``piece.shader`` comes back empty.
        try:
            sh = getattr(d, "m_Shader", None)
            s = sh.read() if sh is not None else None
            pf = getattr(s, "m_ParsedForm", None) if s is not None else None
            nm = str(getattr(pf, "m_Name", "")
                     or getattr(s, "m_Name", "")) if s is not None else ""
            if nm:
                out["_shader"] = nm
        except Exception as exc:             # noqa: BLE001
            swallowed.note("origin.figure.body_head.create_props: sh.read", exc)
        break
    return out


def props_to_payload(source: str, bundle: str, material: str, props: dict) -> dict:
    """The bake's JSON of :func:`props_in` — the maps go in as named textures.

    Tuples become lists, a ``CS.Map`` becomes ``{"map": name}`` with the map
    itself written by ``bake.write_named_texture`` under ``<material>#<prop>``.
    """
    from .. import bake
    out: dict = {}
    for k, v in props.items():
        if isinstance(v, CS.Map):
            name = f"{material}#{k}"
            bake.write_named_texture(source, bundle, name, v)
            out[k] = {"map": name}
        elif isinstance(v, tuple):
            out[k] = [float(x) for x in v]
        else:
            out[k] = v
    return {"props": out}


def _props_from_payload(source: str, bundle: str, payload: dict | None) -> dict | None:
    """:func:`props_to_payload` back; ``None`` when a map it names is gone."""
    from .. import bake
    if payload is None:
        return {}
    out: dict = {}
    for k, v in payload["props"].items():
        if isinstance(v, dict):
            got = bake.read_named_texture(source, bundle, v["map"])
            if got is None:
                return None
            out[k] = got
        elif isinstance(v, list):
            out[k] = tuple(v)
        else:
            out[k] = v
    return out


def create_props(material: str, bundle: str = MM_BASE, mod=None, *,
                 from_bundle: bool = False) -> dict:
    """Saved colours / vectors / texture ST of one authored material.

    Most of them live in ``mm_base``; the draw material the face composites
    onto (``cf_m_face_00``) sits on the head prefab instead, hence *bundle*;
    *mod* is the list row when a zipmod names it (:func:`props_env`).  From
    the bake when the converter wrote it (§7.482), else out of the bundle;
    cached per process.  ``from_bundle`` is the converter's read: past the
    bake and the cache.
    """
    from .. import bake
    source = props_source(mod)
    key = (source, material, bundle)
    if not from_bundle:
        if key in _CREATE_PROPS:
            return _CREATE_PROPS[key]
        recorded, payload = (bake.named_table(source, bundle, props_key(material))
                             if source else (False, None))
        got = _props_from_payload(source, bundle, payload) if recorded else None
        if got is not None:
            _CREATE_PROPS[key] = got
            return got
    try:
        env = props_env(bundle, material, mod)
        out = props_in(env, material) if env is not None else {}
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.body_head.create_props", exc)
        out = {}
    if not from_bundle:
        _CREATE_PROPS[key] = out
    return out


# ``skinSubColor`` used to be pushed into ``Material.shadow_color`` here.  It
# is not the toon shadow tint: the card's sub tone is ``_Color2`` of the
# *create* pass (the mask's green region), and the material's own
# ``_ShadowColor`` — (1.00, 0.90, 0.84) on cf_m_body, (1.00, 0.90, 0.84) on
# cf_m_face_00 — is authored on the asset and never written at run time.
# ``apply_shader`` supplies it, the same as for clothing (§7.12).


def _card_blocks(card, outfit: int):
    body = card.body() if hasattr(card, "body") else {}
    face = card.face() if hasattr(card, "face") else {}
    return (body if isinstance(body, dict) else {},
            face if isinstance(face, dict) else {},
            _makeup(card, outfit))


def _tex(category, ident, res, slot=1, *, report=None, what: str = ""):
    """One texture-only list row, and **a line when the row names one we cannot
    load** (§7.363).

    A layer that resolves to nothing is simply not composited, which on screen
    is a face without its blush or its eyeshadow — and until this reported, the
    card came back ``missing 0``.  Three separate causes look identical here
    (the mod is not installed, its bundle would not open, the row is broken),
    so the line says the category and the id and leaves the diagnosis to the
    reader; a row that names **nothing** (「なし」, a negative id) is not
    reported, because that is the card's own choice.
    """
    got = _list_texture(category, ident, res, slot)
    if got is not None or report is None:
        return got
    try:
        ident_i = int(ident)
    except (TypeError, ValueError):
        return got
    if ident_i < 0 or _list_says_none(category, ident_i):
        return got
    line = f"{what or category}: {category}#{ident_i} 找不到貼圖"
    if line not in report.missing:
        report.missing.append(line)
    return got


def _pick(seq, i):
    return seq[i] if isinstance(seq, (list, tuple)) and i < len(seq) else None


# Overrides the card's ``cheekColor`` for every character built after it is
# set: ``{"colour": (r, g, b), "strength": float}``, or empty for "use the
# card".  ``CharacterController`` fills it from ``face.blush_color`` /
# ``face.blush`` just before it dresses the character — the same seam
# :data:`EYE_COLOUR_OVERRIDE` uses, and for the same reason: ``cardimport`` had
# been copying ``baseMakeup.cheekColor`` into those two parameters since
# forever and **nothing read them back**, so the maker's 腮紅濃度 and 腮紅色
# moved and the face did not.  Measured before the fix: 腮紅濃度 0 → 1 with
# 腮紅色 forced to pure green moved **0.00% of the render, 0 levels** (§7.157).
#
# What it does **not** override is *which* sheet is drawn.  That is the card's
# ``cheekId``, one of ``mt_cheek``'s rows, and a strength with no row behind it
# has nothing to tint — the same thing happens in Origin's own maker with 頬 = なし
# (the list has rows 1‥9 and no row 0, so ``cheekId = 0`` really is "no blush",
# which is 15 of 60 local cards).  Inventing a sheet there would be §一之四.
CHEEK_OVERRIDE: dict = {}


def _cheek_colour(makeup: dict):
    """``_Color4`` for the cheek decal — the card's, or the maker's override.

    Returns the card's value untouched when nothing is overridden, so the
    Illusion path is bit-identical when no parameter has been set.
    """
    card_col = makeup.get("cheekColor")
    ov = CHEEK_OVERRIDE
    if not ov:
        return card_col
    base = ([float(x) for x in card_col[:4]]
            if isinstance(card_col, (list, tuple)) and len(card_col) >= 4
            else [1.0, 1.0, 1.0, 1.0])
    rgb = ov.get("colour")
    if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
        base[:3] = [float(x) for x in rgb[:3]]
    if ov.get("strength") is not None:
        base[3] = float(np.clip(float(ov["strength"]), 0.0, 1.0))
    return base


_SKIN_CACHE: list = [None, None, None]


def skin_overlays(card, outfit: int = 0, report=None) -> dict:
    """``{TexType name: [RGBA]}`` — Overlay Mods' skin layers for *card* (:mod:`skin_overlay`).

    Decoded once per card and coordinate (face, body and both eyes ask).  A
    section that cannot be read, and entries that point nowhere, are reported
    the first time (``CLAUDE.md`` 一之一).
    """
    key = (id(card), int(outfit))
    if _SKIN_CACHE[0] == key and _SKIN_CACHE[1] is card:
        return _SKIN_CACHE[2]
    got, notes, reason = SO.read(card, int(outfit))
    if report is not None:
        if reason:
            report.missing.append(f"皮膚覆蓋層：{reason}")
        for n in notes:
            report.missing.append(f"皮膚覆蓋層：{n}")
        drawn = sorted(k for k in got if k in SO.APPLIED)
        if drawn:
            report.replaced.append("皮膚覆蓋層（Overlay Mods）：" + "、".join(drawn))
        for k in sorted(k for k in got if k not in SO.APPLIED):
            report.missing.append(f"皮膚覆蓋層：{k} 讀到了但沒有畫")
    out = {k: [v] for k, v in got.items()}
    _SKIN_CACHE[:] = [key, card, out]
    return out


def face_texture(card, main, mask, *, outfit: int = 0, res=None, mesh=None,
                 report=None):
    """``CreateFaceTexture`` then the draw pass, as one baked map.

    Layer order and placement are the ones ``create_head`` actually uses:
    paint1 (``_paint1``), paint2 (``_paint2``), cheek (``_tex4uv``), lip line
    (``_tex5uv``), mole (``_hokuro``, no rotation) — then ``cf_m_face_00``
    draws lip, blush and eyeshadow over the result.
    """
    body, face, makeup = _card_blocks(card, outfit)
    P = create_props("cf_m_face_create")
    uv4 = P.get("_tex4uv", (0.0, 0.0, 1.0, 1.0))
    uv5 = P.get("_tex5uv", (0.0, 0.0, 1.0, 1.0))
    paintmask = P.get("_paintmask")
    rep = report
    layers = [
        (_tex("face_paint", _pick(makeup.get("paintId"), 0), res, 1,
              report=rep, what="face"),
         _pick(makeup.get("paintColor"), 0),
         _layout_vec(_pick(makeup.get("paintLayout"), 0), PAINT_RANGES),
         P.get("_Texture3_ST"), "layout", True),
        (_tex("face_paint", _pick(makeup.get("paintId"), 1), res, 2,
              report=rep, what="face"),
         _pick(makeup.get("paintColor"), 1),
         _layout_vec(_pick(makeup.get("paintLayout"), 1), PAINT_RANGES),
         P.get("_Texture7_ST"), "layout", True),
        (_tex("cheek", makeup.get("cheekId"), res, report=rep, what="face"),
         _cheek_colour(makeup),
         uv4, P.get("_Texture4_ST"), "uv", False),
        (_tex("lipline", face.get("lipLineId"), res, report=rep, what="face"),
         face.get("lipLineColor"),
         uv5, P.get("_Texture5_ST"), "uv", False),
        (_tex("mole", face.get("moleId"), res, report=rep, what="face"),
         face.get("moleColor"),
         _layout_vec(face.get("moleLayout"), MOLE_RANGES),
         P.get("_Texture6_ST"), "mole", False),
    ]
    skin = skin_overlays(card, outfit, report)
    main = SO.composite_map(main, skin.get("FaceUnder", []))
    baked = CT.head(main, mask,
                    body.get("skinMainColor"), body.get("skinSubColor"),
                    layers, paintmask,
                    P.get("_MainTex_ST"), P.get("_ColorMask_ST"))
    if baked is None:
        return None, None
    baked = SO.composite_baked(baked, skin.get("FaceOver", []))
    draw = _face_draw_props(card, res)
    return _draw_pass(baked, [
        # ChangeSettingLip: category 403, into _overtex1 / _overcolor1.
        (_tex("lip", makeup.get("lipId"), res, report=rep, what="face"),
         makeup.get("lipColor")),
        # ChangeSettingEyeShadow puts the eyeshadow in _overtex**3**, and the
        # face's _overtex2 is a 1024² blush map authored on cf_m_face_00 that
        # nothing ever replaces.  Its strength is not the authored 0.251
        # either: ChangeHohoAkaRate runs on the load path and overwrites the
        # alpha with Mathf.Lerp(0, 0.2, status.hohoAkaRate).
        (draw.get("_overtex2"), _blush_colour(card, draw)),
        (_tex("eyeshadow", makeup.get("eyeshadowId"), res,
              report=rep, what="face"),
         makeup.get("eyeshadowColor")),
    ], _tex("face_detail", face.get("detailId"), res, report=rep, what="face"),
        # ChangeSettingFaceDetailPower reads fileFace.detailPower, not the
        # body's — they are separate sliders and separate card fields.
        face.get("detailPower"),
        draw, mesh=mesh, report=report, what="face")


# ``EtceteraSystem.Init`` ships ``hohoAka = false``, and ``ChangeHohoAkaRate``
# zeroes the blush outside Studio when that switch is off — so the game's
# out-of-the-box answer is "no blush" no matter what the card says.  It is a
# player-facing setting we have no equivalent of; keeping it as a constant
# means the rest of the chain is the game's arithmetic and one line turns it
# on for whoever wires up the option.  With the default, ``hohoAkaRate`` is 0
# on every card that never set it anyway, so the two agree.
ETC_HOHO_AKA = False


def _blush_colour(card, props: dict):
    """``_overcolor2`` on the face: authored RGB, ``hohoAkaRate`` alpha."""
    rgb = props.get("_overcolor2") or (1.0, 1.0, 1.0, 1.0)
    rate = 0.0
    if ETC_HOHO_AKA:
        status = card.sections.get("Status") if hasattr(card, "sections") else None
        if isinstance(status, dict):
            rate = _num(status.get("hohoAkaRate"), 0.0)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]),
            _lerp(0.0, 0.2, min(max(rate, 0.0), 1.0)))


def _face_draw_props(card, res):
    """``cf_m_face_00``'s toon settings — from *this* head's material.

    A modded head can carry its own material (``MatData``), and the row says
    which; falling back to the stock one is only right when it names none.
    """
    try:
        got = create_props(*material_request(head_assets(card, res)))
        if got:
            return got
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.body_head._face_draw_props: head_assets", exc)
    return create_props("cf_m_face_00", HEAD_BUNDLE)


def body_texture(card, main, mask, *, outfit: int = 0, res=None, sex: int = 0,
                 mesh=None, report=None, marks: bool = True, pack_layers=None):
    """``CreateBodyTexture`` then the draw pass (areola, under-hair).

    ``marks=False`` leaves out everything the card *painted on* the skin —
    the two body paints, the sunburn, the plugins' overlays under and over
    the skin, the crest — and keeps the skin itself: its colour, its detail,
    the areola and the under-hair.  That is the skin a body lent to a
    character of another game wears: the marks are the card character's.
    """
    body, face, makeup = _card_blocks(card, outfit)
    P = create_props("cf_m_body_create")
    lay_ids = body.get("paintLayoutId") or []
    paints = [] if not marks else [
        (_tex("body_paint", _pick(body.get("paintId"), 0), res, 1),
         _pick(body.get("paintColor"), 0),
         _layout_vec(_pick(body.get("paintLayout"), 0),
                     _body_paint_ranges(_pick(lay_ids, 0))),
         P.get("_Texture3_ST")),
        (_tex("body_paint", _pick(body.get("paintId"), 1), res, 2),
         _pick(body.get("paintColor"), 1),
         _layout_vec(_pick(body.get("paintLayout"), 1),
                     _body_paint_ranges(_pick(lay_ids, 1))),
         P.get("_Texture6_ST")),
    ]
    skin = skin_overlays(card, outfit, report) if marks else {}
    # a body pack's own layers come after the card's (``UpdateSkinOverlay``, §7.1197) and are part of the skin
    pack = pack_layers or {}
    main = SO.composite_map(main, list(skin.get("BodyUnder", [])) + list(pack.get("BodyUnder", [])))
    baked = CT.body(main, mask,
                    body.get("skinMainColor"), body.get("skinSubColor"),
                    body.get("nailColor"),
                    _tex("sunburn", body.get("sunburnId"), res) if marks else None,
                    body.get("sunburnColor"), paints, P.get("_paintmask"),
                    P.get("_MainTex_ST"), P.get("_ColorMask_ST"),
                    P.get("_Texture4_ST"))
    if baked is None:
        return None, None
    # the card's own ``BodyOver``, then the plugins' ``AdditionalTextures``
    # by ``ApplyOrder`` — LewdCrestX's crest is one (:mod:`lewdcrest`)
    crest = LC.overlay(card, report) if marks else None
    baked = SO.composite_baked(
        baked, list(skin.get("BodyOver", [])) + list(pack.get("BodyOver", []))
        + ([crest] if crest is not None else []))
    return _draw_pass(baked, [
        # ChangeSettingNip: category 423 → _overtex1 / _overcolor1.
        (_tex("nip", body.get("nipId"), res), body.get("nipColor")),
        # ChangeSettingUnderhair: category 424 → _overtex2 / _overcolor2.
        (_tex("underhair", body.get("underhairId"), res),
         body.get("underhairColor")),
        # Nothing in ChaControl ever assigns the body's _overtex3.
        (None, None),
    ], _tex("body_detail", body.get("detailId"), res),
        body.get("detailPower"), _body_draw_props(body, sex),
        mesh=mesh, report=report, what="body")


def _body_draw_props(body: dict, sex: int) -> dict:
    """``cf_m_body`` with the floats the card owns written over it.

    ``ChangeSettingAreolaSize`` puts ``custom.body.areolaSize`` into
    ``_nipsize`` and ``ChangeSettingNipGlossPower`` puts ``nipGlossPower``
    into ``_nip_specular``; both start from the material's authored 0.5, which
    is nobody's actual slider.  ``_nip`` and ``_tex1mask`` stay as authored —
    nothing in ChaControl ever assigns those two.

    ``_linetexon`` is the third, and it is a whole term rather than a tweak:
    ``ChaControl.VisibleAddBodyLine`` is
    ``matDraw.SetFloat(_linetexon, drawAddLine ? 1 : 0)``, and the flag splits
    this install almost evenly — 584 of 1,386 cards ask for the body line work
    and 802 ask for none, while the material's own default is 1.  The face is
    **not** included: ``VisibleAddBodyLine`` only touches
    ``customTexCtrlBody.matDraw`` (§7.99).
    """
    P = dict(create_props(BODY_MATERIAL[1 if sex else 0]))
    for key, field in (("_nipsize", "areolaSize"),
                       ("_nip_specular", "nipGlossPower")):
        got = body.get(field)
        if isinstance(got, (int, float)):
            P[key] = float(got)
    P["_linetexon"] = 1.0 if body.get("drawAddLine") else 0.0
    return P


def gloss_powers(card, *, head: bool):
    """``main_skin``'s two specular strengths, as the card sets them.

    Four ``ChangeSetting*GlossPower`` write this one pair of properties, and
    which of the four applies is decided by the *material* alone — body and
    face take the same two names from different card fields::

        body matDraw._SpecularPower     = Mathf.Lerp(skinGlossPower, 1,
                                                     status.skinTuyaRate)
        body matDraw._SpecularPowerNail = body.nailGlossPower
        face matDraw._SpecularPower     = face.cheekGlossPower
        face matDraw._SpecularPowerNail = face.lipGlossPower

    **``_SpecularPowerNail`` is not the fingernails.**  The fragment program
    picks between the two per texel with ``_DetailMask.a``::

        p = max(dm.a * _SpecularPower, (1 - dm.a) * _SpecularPowerNail)

    so the "Nail" property is just *the strength where the mask's alpha is 0*,
    and which body part that is belongs to the map, not to the name: on
    ``cf_body_00_dm`` the zero-alpha region is the fingernails (3,066 texels,
    1.17% of it), on ``cf_face_00_dm`` it is the lips (579, 0.06%).  Going by
    the property name would have hung the lip slider on the fingernails.

    Measuring the two halves separately also corrects §7.87's "nearly inert on
    the stock assets": the ``cf_body_00_dm.r`` p90 of 1/255 quoted there is the
    ``a >= 0.5`` half.  Almost all of the map's ink is in the other one — at
    slider 1 the added ``p * t^3`` reaches 0.642 on the face's lip texels (164
    levels of 255) and 0.924 on the body's nails (236).

    All four run out of ``ChangeCustomBodyWithoutCustomTexture`` /
    ``…FaceWithoutCustomTexture``, i.e. off ``<LoadAsync>`` / ``<ReloadAsync>``
    / ``<ChangeHeadAsync>``, so the card overwrites the authored value on every
    load rather than only when it differs.  That is the whole point here:
    ``cf_m_face_00`` ships 1.0 for both, while 936 of this install's 1,386
    cards ask for cheek 0 and 693 for lip 0 — keeping the authored pair ran
    every character at maximum lip gloss.

    **Not clamped.**  ``Material.SetFloat`` does not clamp and a shader's
    ``Range(0,1)`` only bounds the inspector slider (§7.84 一); these four
    arrive at −1.000‥2.984 across the install, so they are passed through.

    Returns ``(power, nail)`` with ``None`` for a field the card does not
    carry, which leaves the material's own value rather than inventing a
    plausible one (§一之一).
    """
    if card is None:
        return None, None
    if head:
        face = card.face() if hasattr(card, "face") else {}
        face = face if isinstance(face, dict) else {}
        return _opt(face.get("cheekGlossPower")), _opt(face.get("lipGlossPower"))
    body = card.body() if hasattr(card, "body") else {}
    body = body if isinstance(body, dict) else {}
    skin = _opt(body.get("skinGlossPower"))
    if skin is not None:
        # ``skinTuyaRate`` is the wet-skin rate an H scene raises; it is 0 on
        # all 1,386 local cards, so this lerp is the identity there — kept
        # because it is a run-time value, not a card field with one value.
        status = card.sections.get("Status") if hasattr(card, "sections") else None
        rate = _num(status.get("skinTuyaRate"), 0.0) \
            if isinstance(status, dict) else 0.0
        skin = _lerp(skin, 1.0, rate)
    return skin, _opt(body.get("nailGlossPower"))


def _opt(value):
    """The float, or ``None`` when the card does not carry the field.

    ``bool`` is excluded on purpose: it is an ``int`` subclass, so a mistyped
    field name that happens to land on one of the neighbouring flags would come
    back as a perfectly plausible 1.0 instead of a miss (§一之一).
    """
    return (float(value) if isinstance(value, (int, float))
            and not isinstance(value, bool) else None)


def _skin_detail(line, props: dict, power, shape) -> np.ndarray | None:
    """``main_skin``'s ``_LineMask`` / ``_DetailMask`` term, transcribed.

    ``ChangeSettingBodyDetail`` binds the card's ``detailId`` texture to both
    ``_NormalMapDetail`` and ``_LineMask``, and ``ChangeSettingBodyDetailPower``
    writes ``detailPower`` into ``_DetailNormalMapScale`` — which is the *k*
    below.  ``_DetailMask`` is not from the card at all: it is authored on the
    material (``cf_body_00_md`` / ``cf_face_00_md``).  Straight off the
    fragment program::

        r6.x = 1 - LineMask.z * k              // cb0[19].x = _DetailNormalMapScale
        r6.z = 1 - LineMask.x * k
        r8.x = 1 - DetailMask.y
        floor = 1 - 1.2 * _ShadowExtend        // cb0[15].y
        diffuse *= floor + (1 - floor) * min(r6.x, r8.x)
        ink      = min(0.5 + 0.5 * r6.z,
                       pow(0.2 + 0.8 * (1 - _linewidthG), NormalMask.z * _FaceShadowG))
        o.rgb    = lerp(albedo * _LineColorG, shaded, lerp(1, ink, _linetexon))

    Two of those globals are inert in the game and both fall out of the
    bytecode rather than being assumed: ``_FaceShadowG`` is only written by
    ``EtcInfo.SetFaceShadow``, which is an empty stub, so the exponent is 0 and
    the ``pow`` is 1 — the neighbouring ``max(NormalMask.z * _FaceShadowG, 1)``
    exists for exactly that reason.  ``ChaShader.ChangeLineColor`` builds
    ``Color(0.5, 0.5, 0.5, 1 - lineDepth)`` and ``lineDepth`` defaults to 1, so
    ``_LineColorG.a`` is 0 and the shader's blend collapses the line colour to
    black.  What is left is two plain multipliers, which is what this returns.

    So: **blue** is the shading channel and **red** the ink, not "whichever is
    brightest" — the shipped maps happen to put almost everything in blue
    (``cf_body_00_line`` averages R 0.0 / G 0.2 / B 94.8), which is why reading
    the max looked right while the floor and the ``_DetailMask`` term were
    simply missing.

    A missing map drops its own term rather than reading as white.  Unity would
    hand an unbound sampler ``Texture2D.whiteTexture``, and white here means
    *maximum* darkening — a texture the loader failed to find would dim the
    whole body instead of doing nothing.  Origin always has one bound, so this only
    ever fires on our own failure path, where no effect is the safe answer.
    """
    h, w = shape
    grid = CT._grid(shape)
    try:
        k = float(power)
    except (TypeError, ValueError):
        k = 0.0
    # Both maps are decoded by their own ``m_ColorSpace`` before any of this,
    # because every term below is a weight and the game reads them decoded
    # (§7.126).  ``line`` and ``_DetailMask`` are :class:`colorspace.Map`.
    lm = None
    if (line is not None and line.px.ndim == 3 and line.px.shape[2] >= 3
            and k > 0.0):
        lm = CT.fetch(line, grid, props.get("_LineMask_ST"), shape)
    term = np.ones((h, w), dtype=np.float32)
    if lm is not None:
        term = np.minimum(term, 1.0 - lm[..., 2] * k)
    md = props.get("_DetailMask")
    if md is not None and md.px.ndim == 3 and md.px.shape[2] >= 3:
        dm = CT.fetch(md, grid, props.get("_DetailMask_ST"), shape)
        term = np.minimum(term, 1.0 - dm[..., 1])
    floor = 1.0 - 1.2 * float(props.get("_ShadowExtend", 0.0))
    shade = floor + (1.0 - floor) * term
    ink = np.ones((h, w), dtype=np.float32)
    if lm is not None:
        # lerp(1, ink, _linetexon)
        raw = 0.5 + 0.5 * (1.0 - lm[..., 0] * k)
        ink = 1.0 + float(props.get("_linetexon", 0.0)) * (raw - 1.0)
    return np.clip(shade, 0.0, 1.0), np.clip(ink, 0.0, 1.0)


#: The per-vertex arrays ``main_skin``'s decal block reads: three coordinate
#: sets and two *single* COLOR channels — .g and .a are never read here, and
#: each channel it does read is 16 MB of raster on the 2048² body.  A mesh
#: without a COLOR stream (``cf_O_face``) simply has no ``vcolor``, and the
#: layers fall back to 1.0, which is what Unity feeds such a shader.
_DECAL_ATTRS = {"uv1": ("uv1", slice(0, 2)), "uv2": ("uv2", slice(0, 2)),
                "uv3": ("uv3", slice(0, 2)),
                "col_r": ("vcolor", slice(0, 1)),
                "col_b": ("vcolor", slice(2, 3))}

# Rasterising a mesh into its own UV0 grid costs a second or two and depends on
# nothing but the mesh, so every card that wears the same body pays it once.
# Keyed on the arrays themselves rather than on the object: ``dress`` hands out
# copies, and the shape sliders move vertices without touching a UV.
_DECAL_MAPS: dict = {}
_DECAL_MAPS_MAX = 4


def _decal_maps(mesh, shape, need: frozenset) -> dict:
    """UV1 / UV2 / UV3 / COLOR sampled at every texel of the UV0 layout.

    ``main_skin`` places its three decals on coordinate sets that are *not*
    the albedo's, so compositing them into the albedo map needs the value of
    those sets per texel — and only the mesh relates the two (§7.50 六).

    Returns ``{}`` when there is no mesh to walk; a missing individual set is
    simply absent from the dict, and the caller drops that layer rather than
    falling back to UV0.  Unity would feed a shader (0, 0) for a coordinate set
    the mesh lacks, which samples the decal's corner texel — transparent on
    every one of these maps — so dropping the layer *is* what the game does.

    *need* is the subset the live layers actually read.  A card with no
    eyeshadow does not pay for UV3, and one with no decals at all does not
    rasterise anything: 2.3 s for the 2048² body, 0.9 s for the 1024² face.
    """
    if mesh is None or mesh.vertex_count == 0 or mesh.tri_count == 0 or not need:
        return {}
    attrs = {}
    for name in need:
        attr, chan = _DECAL_ATTRS[name]
        arr = getattr(mesh, attr, None)
        if arr is not None and len(arr) == mesh.vertex_count:
            attrs[name] = np.asarray(arr, dtype=np.float32)[:, chan]
    if not attrs:
        return {}
    key = (mesh.name, tuple(shape), mesh.tri_count,
           hash(mesh.uvs.tobytes()), hash(mesh.indices.tobytes()),
           tuple(sorted((k, hash(v.tobytes())) for k, v in attrs.items())))
    got = _DECAL_MAPS.get(key)
    if got is None:
        from ...mesh import uvraster
        got, _ = uvraster.rasterise(mesh.uvs, mesh.indices, attrs, shape)
        if len(_DECAL_MAPS) >= _DECAL_MAPS_MAX:
            _DECAL_MAPS.clear()
        _DECAL_MAPS[key] = got
    return got


#: Which of :func:`_decal_maps`' outputs each slot reads.  Slot 1 and 2 gate on
#: a vertex-colour channel, slot 3 does not.
_DECAL_NEEDS = {1: ("uv1", "col_r"), 2: ("uv2", "col_b"), 3: ("uv3",)}


def _decal_live(tex, colour, props: dict, slot: int) -> bool:
    """Does this layer change anything?

    Both the shader's blends end in ``lerp(c, …, sample.a * _overcolorN.a)``,
    so a slot with no texture or a zero-alpha colour is the identity — and the
    zero-alpha case is the common one (this card's lip and blush are both 0).
    Skipping it is the same arithmetic, and it is what keeps a face that wears
    no makeup from rasterising the head.
    """
    if tex is None:
        return False
    c = colour if colour is not None else props.get(f"_overcolor{slot}")
    return float(CT.rgba_raw(c)[3]) > 0.0


def _decal_needs(slots, props: dict) -> frozenset:
    return frozenset(n for slot, (tex, colour) in enumerate(slots, start=1)
                     if _decal_live(tex, colour, props, slot)
                     for n in _DECAL_NEEDS[slot])


def _decal_layers(out, slots, props: dict, maps: dict) -> list:
    """``main_skin``'s three ``_overtex`` layers, transcribed.

    Straight off the forward base program (8th sub-program, ``mm_base``)::

        // _overtex1: areola (body) / lip (face) — UV1, gated by COLOR.r
        d      = uv1 - 0.5
        ramp   = saturate(|d| * 16.66667 - 1)   // inside r=0.06 untouched
        k      = 2 - 1.4 * _nipsize ;  b = 0.7 * _nipsize - 0.5
        q      = lerp(uv1, uv1 * k + b, ramp)
        p      = COLOR.r * (uv1 + _nip * (q - uv1))
        t      = _overtex1.Sample(p * _overtex1_ST.xy + _overtex1_ST.zw)
        nip    = lerp(t.rgb * _overcolor1.rgb,
                      t.r * (_overcolor1.rgb + 0.33 * t.g * _nip_specular),
                      saturate(_tex1mask))
        c      = lerp(MainTex(uv0).rgb, nip, t.a * _overcolor1.a)

        // _overtex2: under-hair (body) / blush (face) — UV2 × COLOR.b
        u      = _overtex2.Sample(uv2 * COLOR.b * _overtex2_ST.xy + _overtex2_ST.zw)
        c      = lerp(c, _overcolor2.rgb * u.rgb, u.a * _overcolor2.a)

        // _overtex3: eyeshadow (face) — UV3
        r      = _overtex3.Sample(uv3 * _overtex3_ST.xy + _overtex3_ST.zw)
        c      = lerp(c, r.rgb * _overcolor3.rgb, r.a * _overcolor3.a)

    ``COLOR.r`` is what keeps the areola on the breasts: 1,530 of the hi-poly
    body's 8,302 vertices have it above 0.5, another 100 sit in the falloff
    band and the rest are 0, where ``p`` collapses to the ST offset and the
    map's transparent corner.  ``COLOR.b`` gates the under-hair the same way
    (101 vertices on the lower belly).  Drawing all
    three on UV0 instead — which is what this did before — put the areola's
    disc (centred on (0.5, 0.5) of the *map*) across the right wrist's island
    (§7.50 六, §八 二 5b).

    An unbound slot contributes nothing.  The shader declares all three
    ``tex=black`` and never branches, so on the body — where ``_overcolor3`` is
    opaque white and nothing ever assigns ``_overtex3`` — a default with any
    alpha at all would lerp the whole skin to black.  It does not, so the
    default samples alpha 0, and skipping is the same arithmetic.

    Returns the names of the layers that had a texture but no coordinate set to
    place it on, so the caller can report them instead of drawing them wrong.
    """
    missing: list[str] = []
    for slot, (tex, colour) in enumerate(slots, start=1):
        if not _decal_live(tex, colour, props, slot):
            continue
        uv = maps.get(f"uv{slot}")     # slot N reads TEXCOORD N
        if uv is None:
            missing.append(f"_overtex{slot}")
            continue
        # ``_rgba`` / ``sample``, not the ``_raw`` pair: these layers are
        # ``main_skin``'s own ``_overtexN`` block and the game runs it on
        # decoded texels with engine-converted colours.  We fold it into the
        # albedo instead of into the frame, so *out* is decoded on the way in
        # and re-encoded on the way out (:func:`_draw_pass`) — the same
        # boundary, just evaluated early (§7.126).
        c = CT._rgba(colour if colour is not None
                     else props.get(f"_overcolor{slot}"))
        tex_st = props.get(f"_overtex{slot}_ST")
        if slot == 1:
            vr = maps.get("col_r", 1.0)
            nipsize = float(props.get("_nipsize", 0.5))
            d = uv - 0.5
            ramp = np.clip(np.sqrt((d * d).sum(axis=-1)) * 16.666670 - 1.0,
                           0.0, 1.0)[..., None]
            q = uv + ramp * ((uv * (2.0 - 1.4 * nipsize)
                              + (0.7 * nipsize - 0.5)) - uv)
            p = vr * (uv + float(props.get("_nip", 0.0)) * (q - uv))
            lay = CT.sample(tex, *CT.st((p[..., 0], p[..., 1]), tex_st))
            spec = float(props.get("_nip_specular", 0.0))
            bright = c[:3] + 0.33 * lay[..., 1:2] * spec
            plain = lay[..., :3] * c[:3]
            k = min(max(float(props.get("_tex1mask", 0.0)), 0.0), 1.0)
            rgb = plain + k * (lay[..., 0:1] * bright - plain)
        else:
            if slot == 2 and "col_b" in maps:
                uv = uv * maps["col_b"]
            lay = CT.sample(tex, *CT.st((uv[..., 0], uv[..., 1]), tex_st))
            rgb = lay[..., :3] * c[:3]
        a = (lay[..., 3] * c[3])[..., None]
        out[..., :3] = out[..., :3] * (1.0 - a) + rgb * a
    return missing


def _draw_decals(base, slots, props, maps):
    """Evaluate the original forward decal block in bounded tiles, encoding once per output texel."""
    h, w = base.shape[:2]
    albedo = np.empty((h, w, 4), dtype=np.uint8)
    missing = set()
    side = CT._GATHER_TILE_SIDE
    for y in range(0, h, side):
        for x in range(0, w, side):
            rows, columns = slice(y, y + side), slice(x, x + side)
            out = CS.decode(base[rows, columns], True)
            tile_maps = {name: values[rows, columns] for name, values in maps.items()}
            missing.update(_decal_layers(out, slots, props, tile_maps))
            albedo[rows, columns] = np.dstack([
                CS.encode(out[..., :3]), np.clip(out[..., 3] * 255, 0, 255).astype(np.uint8)])
    return albedo, sorted(missing)


def _draw_pass(base, slots, detail, power, props: dict, mesh=None,
               report=None, what: str = "skin"):
    """``cf_m_body`` / ``cf_m_face_00``: ``main_skin``'s three decal layers.

    These sit on the *draw* material, not the create one, which is why they
    are applied here rather than inside :mod:`create_tex`.  *slots* is
    ``[(tex, colour)] × 3`` in ``_overtex1`` / ``_overtex2`` / ``_overtex3``
    order; see :func:`_decal_layers` for what each is on each material.

    Returns ``(albedo, shade_mask)``.  The detail term is **not** folded into
    the albedo: ``main_skin`` multiplies it into the diffuse, after the light
    ramp, and the ink it derives from ``_LineMask.r`` multiplies the finished
    colour.  Baking either into the map froze the line work at whatever the
    lighting happened to be, so a back-lit character kept fully-lit creases.

    The mask's channels are the renderer's, not the source maps' --- see
    :func:`create_tex.shade_mask` for the same packing on clothing and hair.
    """
    # The create pass wrote sRGB bytes (its render target is sRGB), and the
    # decal block below is a *forward* pass step, which the game runs in linear.
    # So: decode, blend, encode — the composite that comes out is still sRGB,
    # which is what every consumer of it already expects (§7.126).
    h, w = base.shape[:2]
    maps = _decal_maps(mesh, (h, w), _decal_needs(slots, props))
    albedo, missing = _draw_decals(base, slots, props, maps)
    if missing and report is not None:
        report.missing.append(
            f"{what}: {'/'.join(missing)} 沒有對應的 UV 集，該層未套用")
    shade, ink = _skin_detail(detail, props, power, (h, w))
    if shade is None:
        return albedo, None
    ones = np.ones((h, w), np.float32)
    # B: main_skin's outline width scales with 1 - _DetailMask.b like every
    # other family's.  A: skin has no rim at all (the game multiplies it by
    # the liquid mask, which is 0 on a dry body), so nothing reads it.
    md = props.get("_DetailMask")
    if md is not None and md.px.ndim == 3 and md.px.shape[2] >= 3:
        dm = CT.fetch(md, CT._grid((h, w)),
                      props.get("_DetailMask_ST"), (h, w))
        width = 1.0 - dm[..., 2]
    else:
        width = ones
    mask = np.stack([shade, ink, width * ones, ones], axis=-1)
    return albedo, np.clip(mask * 255.0, 0, 255).astype(np.uint8)


def _makeup(card, outfit: int = 0) -> dict:
    """Which makeup block applies — the coordinate's, or the character's.

    Every ``ChangeSetting`` for lip / eyeshadow / cheek / face paint starts
    with ``nowCoordinate.enableMakeup ? nowCoordinate.makeup :
    fileFace.baseMakeup``, so an outfit can carry its own face.  Only
    ``baseMakeup`` was read before, which is right for most cards but silently
    wrong for any coordinate that overrides it.
    """
    if outfit is not None:
        try:
            from ..apply import outfit_makeup
            enable, mk = outfit_makeup(card, int(outfit))
            if enable and isinstance(mk, dict) and mk:
                return mk
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.figure.body_head._makeup: outfit_makeup", exc)
    custom = card.sections.get("Custom") if hasattr(card, "sections") else None
    face = (custom or {}).get("face") if isinstance(custom, dict) else None
    mk = (face or {}).get("baseMakeup") if isinstance(face, dict) else None
    return mk if isinstance(mk, dict) else {}


# ``Mathf.Lerp`` ranges CreateFaceTexture / CreateBodyTexture feed the layout
# vectors with, straight out of the IL.  ``w`` looks odd (-8 … 0.7) until you
# see what create_head does with it: the sampling scale is ``4 * (1 - w)``, so
# w = 0.7 is roughly 1:1 and the low end shrinks the decal to nothing.
PAINT_RANGES = ((0.25, -0.25), (0.3, -0.3), (1.0, -1.0), (-8.0, 0.7))
MOLE_RANGES = ((0.25, -0.25), (0.3, -0.3), (1.0, -1.0), (0.0, 0.7))


def _layout_vec(layout, ranges) -> tuple:
    if not isinstance(layout, (list, tuple)):
        layout = ()
    return tuple(_lerp(a, b, _num(_pick(layout, i), 0.5))
                 for i, (a, b) in enumerate(ranges))


def _body_paint_ranges(layout_id) -> tuple:
    """``CreateBodyTexture``'s Lerp bounds for a body-paint slot.

    The face uses literal constants, but the body reads six floats off the
    ``bodypaint_layout`` row the card selected::

        x: Lerp(CenterX - MoveX,         CenterX + MoveX,         layout.x)
        y: Lerp(CenterY - MoveY,         CenterY + MoveY,         layout.y)
        z: Lerp(-1, 1, layout.z)
        w: Lerp(CenterScale - AddScale,  CenterScale + AddScale,  layout.w)

    That is why the preset picks *where on the body* the decal can travel —
    a chest tattoo and a thigh tattoo do not share a range.
    """
    from .. import lists as L
    row = L.index().layout("bodypaint_layout", layout_id)
    if not row:
        return PAINT_RANGES

    def f(key, default=0.0):
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    cx, mx = f("CenterX"), f("MoveX")
    cy, my = f("CenterY"), f("MoveY")
    cs, as_ = f("CenterScale"), f("AddScale")
    return ((cx - mx, cx + mx), (cy - my, cy + my), (-1.0, 1.0),
            (cs - as_, cs + as_))


def _mods_roots() -> list:
    """Project ``assets_cache/mods`` then optional external install mods."""
    from ..install import cache_dir, project_mods_dir, source_install
    roots = []
    pm = project_mods_dir()
    if pm.is_dir():
        roots.append(pm)
    try:
        src = source_install()
        if src.valid and not src.local:
            for name in ("mods", "Mods"):
                p = src.root / name
                if p.is_dir() and p not in roots:
                    roots.append(p)
    except Exception as exc: # noqa: BLE001
        swallowed.note("origin.figure.body_head._mods_roots: source_install", exc)
    # also walk cache_dir/mods if different
    c = cache_dir() / "mods"
    if c.is_dir() and c not in roots:
        roots.append(c)
    return roots


def _zipmod_entry(zip_substr: str, inner: str):
    """``(zipmod path, entry name)`` of the first zipmod named like *zip_substr* that ships *inner*."""
    import zipfile
    inner = inner.replace("\\", "/")
    alt = inner.split("abdata/", 1)[-1]
    for root in _mods_roots():
        try:
            zips = list(root.rglob("*.zipmod"))
        except OSError:
            continue
        for zp in zips:
            if zip_substr.lower() not in zp.name.lower():
                continue
            try:
                with zipfile.ZipFile(zp) as z:
                    names = z.namelist()
            except (OSError, zipfile.BadZipFile):
                continue
            # tolerate either with or without leading abdata/
            key = inner if inner in names else next(
                (n for n in names if n.replace("\\", "/").endswith(alt)), None)
            if key is not None:
                return zp, key
    return None


def _extract_zipmod_entry(zp, key: str):
    """*key* out of zipmod *zp*, unpacked once into a stable cache; None if it cannot be read."""
    import zipfile
    from pathlib import Path
    dest = cache_dir_uncensor() / zp.stem / Path(key).name
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    try:
        with zipfile.ZipFile(zp) as z:
            data = z.read(key)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        swallowed.note("origin.figure.body_head._extract_zipmod_entry", exc)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def _find_zipmod_asset(zip_substr: str, inner: str):
    """Return a temp-extracted path to *inner* from the first matching zipmod."""
    hit = _zipmod_entry(zip_substr, inner)
    return None if hit is None else _extract_zipmod_entry(*hit)


def uncensor_manifest_asset(zp, kind: str, entry: str) -> str | None:
    """The ``<asset>`` a zipmod's ``<KK_UncensorSelector>`` section names for bundle *entry*, or None.

    ``PenisData``/``BallsData`` read four child elements of each ``<penis>`` /
    ``<balls>`` entry — ``guid``, ``displayName``, ``file`` (the bundle, from
    ``abdata``) and ``asset`` — and ``ReloadCharacterPenis`` hands exactly that
    ``(file, asset)`` pair to ``CommonLib.LoadAsset`` (§7.498).  *entry* is the
    zip entry name (``abdata/chara/…``); a manifest with no entry for it names
    nothing, and the game would not load that bundle as this part either.
    """
    import xml.etree.ElementTree as ET
    import zipfile
    try:
        with zipfile.ZipFile(zp) as z:
            name = next((n for n in z.namelist() if n.lower() == "manifest.xml"), None)
            xml = z.read(name) if name else b""
        root = ET.fromstring(xml) if xml else None
    except (OSError, zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        swallowed.note("origin.figure.body_head.uncensor_manifest_asset", exc)
        return None
    section = root.find("KK_UncensorSelector") if root is not None else None
    want = entry.replace("\\", "/").split("abdata/", 1)[-1].lower()
    for item in (section.findall(kind) if section is not None else []):
        file = (item.findtext("file") or "").replace("\\", "/").lower()
        asset = item.findtext("asset")
        if file == want and asset:
            return asset
    return None


def cache_dir_uncensor():
    from ..install import cache_dir
    d = cache_dir() / "_uncensor_extract"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_uncensor_body_bundle(sex: int = 0):
    """Path to an uncensored body ``.unity3d``, or None.

    Female: full ``oo_base_*`` with denser ``o_body_a`` (crotch in the body).
    Male: None here — penis is swapped via :func:`_overlay_male_penis`.
    """
    sex = 1 if int(sex or 0) else 0
    if sex != 0:
        return None
    for substr, inner in _FEMALE_BODY_UNCENSOR:
        hit = _find_zipmod_asset(substr, inner)
        if hit is not None and hit.is_file():
            return hit
    return None


def resolve_uncensor_penis():
    """``(bundle path, prefab name)`` of the first penis uncensor the user has, or None.

    The prefab is the one the zipmod's manifest names (:func:`uncensor_manifest_asset`);
    a pack whose manifest names none for its bundle is passed over, as the game
    would never load it as a penis.
    """
    for substr, inner in _MALE_PENIS_UNCENSOR:
        hit = _zipmod_entry(substr, inner)
        if hit is None:
            continue
        asset = uncensor_manifest_asset(hit[0], "penis", hit[1])
        path = _extract_zipmod_entry(*hit) if asset else None
        if path is not None:
            return path, asset
    return None


def uncensor_mods_available(sex: int = 0) -> bool:
    """True when the user has zipmods we can bake an uncensored body from."""
    sex = 1 if int(sex or 0) else 0
    if sex == 0:
        return resolve_uncensor_body_bundle(0) is not None
    return resolve_uncensor_penis() is not None


def pieces_look_uncensored(pieces: list, *, sex: int) -> bool:
    """Heuristic on loaded pieces (bake acceptance floors)."""
    sex = 1 if int(sex or 0) else 0
    body_v = 0
    dankon_v = 0
    for p in pieces or []:
        if p.mesh is None:
            continue
        n = str(p.mesh.name or "")
        if n == BODY_MESH:
            body_v = int(p.mesh.vertex_count)
        elif n == "o_dankon":
            dankon_v = int(p.mesh.vertex_count)
    if sex == 0:
        return body_v >= UNCENSOR_BODY_A_MIN
    return dankon_v >= UNCENSOR_DANKON_MIN


def baked_body_looks_uncensored(ident: str = "body") -> bool:
    """Read an existing bake archive and apply :func:`pieces_look_uncensored`."""
    from .. import bake
    hit = bake.read_item("base", ident)
    if hit is None or not hit.pieces:
        return False
    sex = 0 if ident in ("body", "body_low") else 1
    return pieces_look_uncensored(hit.pieces, sex=sex)


def _renderer_name(piece) -> str | None:
    """The GameObject name of the renderer a piece came from (``Renderer.name``)."""
    return (getattr(piece, "shader", None) or {}).get("_rendName")


def replace_renderers(body: list, src: list, parts) -> list:
    """*body* with each renderer named in *parts* that *src* also has swapped for *src*'s.

    ``ReloadCharacterPenis`` walks the loaded prefab's renderers; one whose name
    is in ``PenisParts`` finds the body renderer with the same name and
    ``UpdateMeshRenderer`` copies mesh, bones and materials onto it — whatever
    either one's vertex count.  A part the body has no renderer for is dropped,
    not added (§7.498).  A renderer's pieces are its material passes, so they
    are replaced together.
    """
    by_name: dict = {}
    for p in src:
        n = _renderer_name(p)
        if n in parts:
            by_name.setdefault(n, []).append(p)
    out, done = [], set()
    for p in body:
        n = _renderer_name(p)
        if n in by_name:
            if n not in done:
                out.extend(by_name[n])
                done.add(n)
            continue
        out.append(p)
    return out


def _by_renderer(pieces: list) -> dict:
    """``{renderer name: [its pieces]}`` in first-seen order (a renderer's pieces are its material passes)."""
    out: dict = {}
    for p in pieces:
        out.setdefault(_renderer_name(p), []).append(p)
    return out


def _mesh_only(src: list, dst: list) -> list:
    """*src*'s geometry under *dst*'s materials: ``UpdateMeshRenderer(…, copyMaterials: false)``."""
    out = []
    for i, s in enumerate(src):
        d = dst[min(i, len(dst) - 1)]
        out.append(s._replace(texture=d.texture, mask=d.mask, detail=d.detail, shader=d.shader, normal=d.normal,
                              tex_set=d.tex_set, alpha_mask=d.alpha_mask, ramp=d.ramp, line_mask=d.line_mask))
    return out


def compose_body(own: list, pack: list, additional=(), all_additional=frozenset()) -> list:
    """``ReloadCharacterBody``: the pack's renderers copied onto the character's *own* body, by name.

    Each renderer of hers: a ``BodyNames`` one takes the pack's mesh and
    keeps her materials; a ``BodyParts`` one takes the pack's with its
    materials; one some body lists as an additional part takes the pack's
    when this pack lists it (*additional*) and is destroyed otherwise; the
    rest stay hers.  A name the pack has no renderer for keeps hers
    (``UpdateMeshRenderer`` returns on a null source).  An additional part
    she has no renderer for is the pack's, added (instantiated there).
    """
    src = _by_renderer(pack)
    out, have = [], set()
    for name, passes in _by_renderer(own).items():
        have.add(name)
        got = src.get(name)
        if name in BODY_NAMES:
            out.extend(_mesh_only(got, passes) if got else passes)
        elif name in BODY_PARTS:
            out.extend(got or passes)
        elif name in all_additional:
            if name in additional:
                out.extend(got or passes)
        else:
            out.extend(passes)
    for name in additional:
        if name not in have and name in src:
            out.extend(src[name])
    return out


def color_match(pieces: list, entries) -> list:
    """``SetSkinColor``'s inputs on the parts the chosen packs colour-match, in the plugin's order (body, penis, balls).

    Each ``<colorMatch>`` row names a renderer, a create material and two maps
    in its pack's bundle; the plugin builds that renderer's ``_MainTex`` from
    them with the card's ``skinMainColor`` / ``skinSubColor``
    (:func:`color_match_texture`).  The piece carries the two maps and the
    create material's saved properties (``shader["_colorMatch"]``).  A row
    whose maps are not in the bundle is skipped, as the plugin returns on a
    null asset.  Bake-time only (it opens the packs' bundles).
    """
    for entry in entries:
        if entry is None or not entry.archive or not entry.color_match:
            continue
        loader = PrefabLoader(_pack_file(entry))
        try:
            for obj, create, main_name, mask_name in entry.color_match:
                main, mask = loader.texture_by_name(main_name), loader.texture_by_name(mask_name)
                if main is None or mask is None:
                    continue
                props = props_in(loader.env, create)
                pieces = [p._replace(texture=main.px, mask=mask.px,
                                     map_cs={**(p.map_cs or {}), "texture": main.space, "mask": mask.space},
                                     shader={**(p.shader or {}), "_colorMatch": props})
                          if _renderer_name(p) == obj else p for p in pieces]
        finally:
            loader.env = None
            loader.by_id = {}
    return pieces


def color_match_texture(piece, card, outfit: int = 0):
    """The ``_MainTex`` ``SetSkinColor`` builds for a colour-matched *piece*, or None when it is not one.

    ``CustomTextureCreate`` over the pack's create material (``create_body``,
    the body's own create shader): its main map, ``_Color`` = ``skinMainColor``,
    its colour mask, ``_Color2`` = ``skinSubColor``; everything else is the
    material's as saved (no tan map, so ``t`` is 1; ``_Color5`` its own).
    """
    props = (piece.shader or {}).get("_colorMatch")
    if props is None:
        return None
    body, _face, _makeup = _card_blocks(card, outfit)
    return CT.body(CS.of(piece, "texture"), CS.of(piece, "mask"),
                   body.get("skinMainColor"), body.get("skinSubColor"), props.get("_Color5"),
                   None, props.get("_Color4"), [], None,
                   props.get("_MainTex_ST"), props.get("_ColorMask_ST"), props.get("_Texture4_ST"))


#: ``Defaults.Normals``: the ``BustNormal`` asset ``SetChestNormals`` reads out of the body's bundle when its pack names none
BUST_NORMALS = "p_cf_body_00_Nml"


def bust_normals(pieces: list, body) -> list:
    """``SetChestNormals``: each renderer the chosen body's ``BustNormal`` asset covers, with its two normal sets.

    The plugin re-initialises ``ChaControl.dictBustNormal`` from
    ``BodyData.OOBase`` / ``BodyData.Normals`` (defaults ``chara/oo_base.unity3d``
    / :data:`BUST_NORMALS`); the asset is ``data`` rows of ``ObjectName``,
    ``NormalMin``, ``NormalMax``.  ``BustNormal.Blend`` writes a mesh only when
    its normal count equals the row's, so a piece gets ``shader["_bustNormal"]``
    (``{"min", "max"}``, X negated as the mesh import does) only then, and
    ``None`` otherwise — checked, nothing to blend.  A pack bundle without the
    asset answers nothing (``BustNormal.Init`` returns on a null load), which is
    every installed pack; the built-in bodies resolve the stock one (8,302
    normals, the vanilla ``o_body_a``).  Bake-time only.
    """
    if body is None:
        return pieces
    asset = body.normals or BUST_NORMALS
    if body.archive:
        loader = PrefabLoader(_pack_file(body))
    else:
        from .load_mesh import _open_resolved
        loader = _open_resolved(BODY_BUNDLE, asset)
    rows: dict = {}
    if loader is not None:
        try:
            for obj in loader.env.objects:
                if obj.type.name != "MonoBehaviour":
                    continue
                tree = obj.read_typetree()
                if tree.get("m_Name") != asset:
                    continue
                for row in tree.get("data") or []:
                    lo, hi = row.get("NormalMin") or [], row.get("NormalMax") or []
                    if len(lo) == len(hi):
                        rows[str(row.get("ObjectName") or "")] = (lo, hi)
                break
        finally:
            loader.env = None
            loader.by_id = {}

    def flip(vs):
        return [[round(-float(v["x"]), 6), round(float(v["y"]), 6), round(float(v["z"]), 6)] for v in vs]

    out = []
    for p in pieces:
        got = rows.get(_renderer_name(p))
        if got is None:
            out.append(p)
            continue
        n = len(getattr(p.mesh, "normals", None) if p.mesh is not None else ())
        value = {"min": flip(got[0]), "max": flip(got[1])} if n == len(got[0]) else None
        out.append(p._replace(shader={**(p.shader or {}), "_bustNormal": value}))
    return out


def _pack_layers(body) -> tuple:
    """``((skin type, map name), …)`` ``UpdateSkinOverlay`` adds for *body*: none unless its ``OOBase`` is not the default."""
    if body is None or not body.archive or not body.bundle or body.bundle == BODY_BUNDLE:
        return ()
    return tuple((kind, name) for kind, name in (("BodyOver", body.uncensor_overlay),
                                                 ("BodyUnder", body.uncensor_underlay)) if name)


def bake_pack_layers(body) -> int:
    """Convert *body*'s ``uncensorOverlay`` / ``Underlay`` maps out of its bundle (named, under the pack's GUID).

    A name the bundle lacks is recorded as absent: ``LoadAsset`` returns null,
    the plugin logs "Unable to apply uncensor overlay" and adds nothing.
    Returns how many were written.
    """
    from .. import bake
    layers = _pack_layers(body)
    if not layers:
        return 0
    loader = PrefabLoader(_pack_file(body))
    n = 0
    try:
        for _kind, name in layers:
            tex = loader.texture_by_name(name)
            if tex is None:
                bake.write_named_absent(body.guid, body.bundle, name)
            else:
                bake.write_named_texture(body.guid, body.bundle, name, tex)
                n += 1
    finally:
        loader.env = None
        loader.by_id = {}
    return n


def pack_skin_layers(body, report=None) -> dict:
    """``UpdateSkinOverlay``: ``{"BodyOver": [map], "BodyUnder": [map]}`` the chosen body pack adds.

    ``AdditionalTexture(tex, type, this)`` leaves ``ApplyOrder`` at 0, so each
    goes after the card's own layer of that type and before the crest (1010).
    A declared map that was not converted is reported, not drawn.
    """
    from .. import bake
    out: dict = {}
    for kind, name in _pack_layers(body):
        tex = bake.read_named_texture(body.guid, body.bundle, name)
        if tex is None:
            if report is not None and bake.named_entry(body.guid, body.bundle, name) is None:
                report.missing.append(f"身體包 {body.guid} 的 {name} 沒有轉出（重轉角色的身體）")
            continue
        out.setdefault(kind, []).append(tex)
    return out


def bust_normal_blend(piece, card):
    """The normals ``BustNormal.Blend`` gives *piece* for *card*, or None (nothing to blend, or a card without sliders).

    ``LateUpdateForce``: the rate is 1, or ``InverseLerp(0, 0.5, shapeValueBody[BustSize])``
    below 0.5; each normal is ``Vector3.Lerp(NormalMin, NormalMax, rate)``.
    """
    from .bust import BUST_SIZE_SLIDER
    bust = (piece.shader or {}).get("_bustNormal")
    if not bust or card is None:
        return None
    values = (_card_blocks(card, 0)[0].get("shapeValueBody") or [])
    if len(values) <= BUST_SIZE_SLIDER:
        return None
    v = float(values[BUST_SIZE_SLIDER])
    t = 1.0 if not v < 0.5 else min(max(v / 0.5, 0.0), 1.0)
    lo, hi = np.asarray(bust["min"], np.float32), np.asarray(bust["max"], np.float32)
    return (lo + (hi - lo) * t).astype(np.float32)


def _own_body_pieces(sex: int) -> list:
    """The character's own body prefab (``p_cf_body_00`` / ``p_cm_body_00``), as Sideloader answers the game's load by name."""
    from .load_mesh import _open_resolved
    prefab = _lod(BODY_PREFAB[sex])
    loader = _open_resolved(BODY_BUNDLE, prefab)
    if loader is None:
        raise FileNotFoundError(f"{BODY_BUNDLE} 沒有 {prefab}")
    try:
        return loader.build_skinned(prefab)
    finally:
        loader.env = None
        loader.by_id = {}


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text))


def body_ident(sex: int, body=None, penis=None, balls=None) -> str:
    """The bake identity of a body: ``body`` / ``body_m`` for the default pack, one per chosen pack otherwise.

    *body* / *penis* / *balls* are ``uncensor_choice.Entry`` (#93); the identity names the penis and
    balls too, which ``ReloadCharacterPenis`` / ``ReloadCharacterBalls`` put on the same archive —
    always for a man, for a woman when her card shows them (``uncensor_choice.worn``).
    """
    sex = 1 if int(sex or 0) else 0
    ident = "body" if sex == 0 else "body_m"
    if body is not None:
        ident += "@" + _slug(body.guid)
        if penis is not None:
            ident += "+" + _slug(penis.guid)
        if balls is not None:
            ident += "+" + _slug(balls.guid)
    if not hi_poly():
        ident += "_low"
    return ident


def _pack_file(entry):
    """``abdata/<entry.bundle>`` out of the zipmod *entry* came from, unpacked; FileNotFoundError when it is not there."""
    import zipfile
    from pathlib import Path
    zp = Path(entry.archive)
    want = ("abdata/" + entry.bundle.replace(chr(92), "/")).lower()
    try:
        with zipfile.ZipFile(zp) as z:
            key = next((n for n in z.namelist() if n.lower() == want), None)
    except (OSError, zipfile.BadZipFile) as exc:
        raise FileNotFoundError(f"{entry.guid}: {zp.name}: {exc}") from exc
    path = _extract_zipmod_entry(zp, key) if key else None
    if path is None:
        raise FileNotFoundError(f"{entry.guid}: {want} 不在 {zp.name}")
    return path


def body_sex(sex: int, body=None) -> int:
    """Our sex (0 woman, 1 man) of the body itself: the character's, unless the chosen pack is the other's.

    ``BodyData`` takes its stock asset, colour mask and material by the body's ``Sex``, not the character's.
    """
    if body is None or body.sex is None:
        return 1 if int(sex or 0) else 0
    return 0 if body.sex == 1 else 1


def body_source(sex: int, body=None):
    """``(bundle path, prefab name)`` a body loads from.

    *body* is the pack ``uncensor_choice`` chose (its ``<oo_base>`` file and asset; the stock
    ``oo_base`` for the built-in ones); None is the default pack (:func:`resolve_uncensor_body_bundle`).
    """
    sex = 1 if int(sex or 0) else 0
    inst = configured()
    if body is not None:
        # the body's own sex picks the stock asset (``BodyData``: ``AssetFemale`` / ``AssetMale``),
        # so a man the card dresses in a woman's body gets hers (``Genderbender Allowed``)
        stock = BODY_PREFAB[body_sex(sex, body)]
        if not body.archive:
            return inst.bundle(BODY_BUNDLE), _lod(stock)
        return _pack_file(body), _lod(body.asset or stock)
    path = resolve_uncensor_body_bundle(0) if sex == 0 else None
    if path is None or not path.exists():
        path = inst.bundle(BODY_BUNDLE)
    return path, _lod(BODY_PREFAB[sex])


def _overlay_balls(pieces: list, balls=None) -> list:
    """``ReloadCharacterBalls``: the chosen *balls*' ``o_dan_f`` / ``cm_o_dan_f`` onto the body's renderers of those names.

    The built-in ``Default.Balls`` is the stock body's own, so it changes nothing; so does no choice.
    """
    if balls is None or not balls.archive:
        return pieces
    loader = PrefabLoader(_pack_file(balls))
    try:
        src = loader.build_skinned(balls.asset)
    finally:
        loader.env = None
        loader.by_id = {}
    return replace_renderers(pieces, src, BALLS_PARTS)


def _overlay_male_penis(pieces: list, penis=None) -> list:
    """``ReloadCharacterPenis``: the chosen *penis* (``uncensor_choice.Entry``), else the first penis uncensor the user has.

    The built-in ``Default.Penis`` is the stock body's own renderers, so it changes nothing.
    """
    if penis is not None:
        if not penis.archive:
            return pieces
        got = (_pack_file(penis), penis.asset)
    else:
        got = resolve_uncensor_penis()
    if got is None:
        return pieces
    path, asset = got
    loader = PrefabLoader(path)
    try:
        src = loader.build_skinned(asset)
    finally:
        loader.env = None
        loader.by_id = {}
    return replace_renderers(pieces, src, PENIS_PARTS)


def body_sway_specs(sex: int = 0, body=None) -> list:
    """The body prefab's ``DynamicBone`` chains, in the form the bake carries.

    Bake-time only — it opens a ``.unity3d``, which the runtime is not allowed
    to do (四之一), which is exactly why the result has to travel in our own
    format (``bake.write_item(sway_spec=…)``).

    On a BetterPenetration female pack this returns the **13 ``cf_J_Vagina``
    chains** that are the entire mechanism behind the crotch opening: they
    carry ``m_Gravity = m_Force = 0``, so they do not move on their own and
    only respond to a collider being pushed into them (§7.128).  On a pack
    without them it returns whatever the body does have, and on vanilla
    ``oo_base`` that is nothing — an empty list, which every reader treats as
    "no per-frame chains", i.e. the behaviour they already had.
    """
    from ...core import math3d as m3
    path, prefab = body_source(sex, body)
    if not path.exists():
        return []
    loader = PrefabLoader(path)
    try:
        root = loader.find_prefab(prefab)
        if root is None:
            return []
        troot = loader._transform_of(root)
        if troot is None:
            return []
        worlds: dict = {}
        loader._collect_worlds(troot, m3.identity(), worlds)
        # each chain's component object (``m_ObjectMove`` is its travel: BetterPenetration's
        # ``cf_J_Vagina_root``), which the per-frame push carries its particles by (§7.418)
        return loader.sway_chain_specs(worlds, objects=True)
    except Exception as exc:                             # noqa: BLE001
        swallowed.note("origin.figure.body_head.body_sway_specs: root = loader.find_prefab(_lod(BODY_PREFAB[…", exc)
        return []


def body_implant_bones(sex: int = 0, body=None) -> list:
    """Bones the body prefab has that Illusion's **reference** armature has not.

    ``graft_full_skeleton`` builds our skeleton from ``oo_base``'s reference
    rig, so a bone that exists only in an uncensor pack is simply absent — and
    an absent bone is not skipped, it is **redirected**: ``skinmap._resolve``'s
    ``exact`` branch sends every unknown name to ``cf_j_hips``.  A
    BetterPenetration body skins its crotch to 18 ``cf_J_Vagina_*`` bones, so
    all 18 land on the hips and the crotch follows the pelvis — two wedges from
    the groin to the feet, drawn over the socks (§7.129).  That, and not the
    solver, is what broke §7.123 and §7.125.

    So the pack's own subtree has to travel with the body and be grafted on at
    dress time.  Returned as ``{name, parent, local}`` with *local* the 4×4 in
    the parent's frame — parent-relative is the one form that survives the
    graft's rescaling without a space conversion, which is the other half of
    what went wrong before.

    Bake-time only (it opens a ``.unity3d``, 四之一).  Empty for a pack that
    adds nothing, which is every stock body — the reader then does nothing and
    behaviour is unchanged.
    """
    path, prefab = body_source(sex, body)
    return prefab_implant_bones(path, prefab, f"body_implant_bones({sex})")


def pack_implant_bones(entry) -> list:
    """:func:`body_implant_bones` for a penis or balls pack (``uncensor_choice.Entry``): the bones its prefab
    has that the reference rig lacks, in the same rows.

    In the game a man wearing ``BP_SoS`` has the whole shaft ``cm_J_dan102``…``108``, ``cm_J_dan119`` and the
    scrotum's ``cm_J_dan_Pivot_f_*`` / ``cm_J_dan_Dynamic_f_*`` (the H scene's dump); the reference rig has
    ``dan100`` / ``101`` / ``109`` only, and the pack's mesh is skinned to the rest, which without them went
    to ``cf_j_hips`` the way §7.129's crotch did (§7.1186).  Empty for a built-in (``Default.*``) part.
    """
    if entry is None or not entry.archive:
        return []
    return prefab_implant_bones(_pack_file(entry), entry.asset, f"pack_implant_bones({entry.guid})")


def pack_chain_specs(ident: str, penis=None, balls=None) -> list:
    """The penis and balls packs' chains a body item carries (:func:`pack_sway_specs`, kept in its ``sway_spec``).

    From the bake; without an archive the body itself comes off the bundle, and so do these
    (the same rule as :func:`graft_implant_bones`).  An archive baked before the packs' chains
    were kept has none, and a body on it does not swing them.
    """
    from .. import bake
    hit = bake.read_item("base", ident)
    if hit is not None:
        return [c for c in (getattr(hit, "sway_spec", None) or ()) if c.get("pack")]
    return pack_sway_specs(penis) + pack_sway_specs(balls)


def pack_sway_specs(entry) -> list:
    """A penis or balls pack's own ``DynamicBone`` chains (``prefab.sway_chain_specs``), each marked ``pack``.

    BetterPenetration's scrotum is two of them, on the balls pack's ``cm_J_dan_Pivot_f_L`` /
    ``_R`` (``DanAgent.InitializeTama`` looks them up by that name): the component sits on the
    pivot itself, so each chain names its ``object``.  Bake-time only (it opens a ``.unity3d``,
    四之一); empty for a built-in (``Default.*``) part.
    """
    from ...core import math3d as m3
    if entry is None or not entry.archive:
        return []
    path = _pack_file(entry)
    if not path.exists():
        return []
    loader = PrefabLoader(path)
    try:
        root = loader.find_prefab(entry.asset)
        troot = loader._transform_of(root) if root is not None else None
        if troot is None:
            return []
        worlds: dict = {}
        loader._collect_worlds(troot, m3.identity(), worlds)
        return [{**c, "pack": str(entry.guid)} for c in loader.sway_chain_specs(worlds, objects=True)]
    except Exception as exc:                             # noqa: BLE001
        # not silent: no chains because the read threw looks like a pack without any
        swallowed.note(f"origin.figure.body_head.pack_sway_specs({entry.guid})", exc)
        return []


#: ``GetCollidersOfChaControl(male, "cm_J_dan")``'s filter: a ``DynamicBoneCollider`` whose object's name contains it
DAN_COLLIDER_NAME = "cm_J_dan"


def _dan_colliders(loader) -> dict:
    """``{GameObject path id: [collider row]}`` of the bundle's ``DynamicBoneCollider``s whose object names :data:`DAN_COLLIDER_NAME`."""
    from .. import dynbone as DB
    from ..prefab import _collider_script

    out: dict = {}
    for obj in loader.env.objects:
        if obj.type.name != "MonoBehaviour" or not _collider_script(loader, obj):
            continue
        tree = obj.read_typetree()
        col = DB.collider_from_tree(tree)
        if col is None:
            continue
        out.setdefault((tree.get("m_GameObject") or {}).get("m_PathID"), []).append(
            {"centre": [float(v) for v in col.centre], "radius": col.radius, "height": col.height,
             "direction": col.direction, "bound": col.bound})
    return out


def prefab_implant_bones(path, prefab: str, label: str) -> list:
    """The bones under *prefab* in the bundle at *path* that the reference rig lacks: ``{name, parent, local}`` rows.

    A bone whose name contains :data:`DAN_COLLIDER_NAME` and that carries ``DynamicBoneCollider``s
    also gets ``dan_colliders`` (its own rows, in the plugin's order); one already on the reference
    rig is listed for them alone (the graft adds nothing for it), §7.1187.
    """
    from ..prefab import _trs
    from ..rig import rig as _rig

    if not path.exists():
        return []
    try:
        reference = set(_rig().matrices)
    except Exception as exc:                             # noqa: BLE001
        swallowed.note("origin.figure.body_head.prefab_implant_bones: reference = set(_rig().matrices)", exc)
        return []
    loader = PrefabLoader(path)
    out: list = []
    try:
        root = loader.find_prefab(prefab)
        troot = loader._transform_of(root) if root is not None else None
        if troot is None:
            return []
        seen: set = set()
        colliders = _dan_colliders(loader)

        def walk(t, pname):
            tt = t.read_typetree()
            go_id = (tt.get("m_GameObject") or {}).get("m_PathID")
            go = loader.by_id.get(go_id)
            nm = go.read_typetree().get("m_Name", "") if go is not None else ""
            own = colliders.get(go_id) if DAN_COLLIDER_NAME in nm else None
            if nm and pname and (nm not in reference or own):
                # Only the ones the reference rig lacks; their parent may be
                # another implant (a Pivot under the pack's own root), which is
                # why the list keeps document order — the reader adds in order
                # and a parent is always already there.  Document order is
                # also ``GetComponentsInChildren``'s, the colliders' order.
                row = {"name": nm, "parent": pname, "local": _trs(tt).tolist()}
                if own:
                    row["dan_colliders"] = own
                out.append(row)
            for c in loader._children(tt, seen):
                walk(c, nm or pname)

        walk(troot, None)
    except Exception as exc:                             # noqa: BLE001
        # Not silent: an implant list that comes back empty because the walk
        # threw is indistinguishable from "this pack adds nothing", and the
        # difference is a body whose crotch is skinned to the hips (§7.129).
        print(f"  !! {label}: {exc}")
        return []
    return out


def load_body_prefab(sex: int = 0, from_bundle: bool = False, body=None, penis=None, balls=None):
    """(pieces, skin albedo, skin colour mask) — from the bake when present.

    *body* / *penis* are the packs ``uncensor_choice`` chose for the card (#93), baked under
    :func:`body_ident`; without them, the default pack.  Bundle loads prefer UncensorSelector
    body packs when the user has them in ``assets_cache/mods`` (or the sync-source install).
    Vanilla ``oo_base`` keeps the censor stubs; uncensor packs put the real crotch in ``o_body_a``.
    """
    from .. import bake
    sex = 1 if int(sex or 0) else 0
    ident = body_ident(sex, body, penis, balls)
    hit = None if from_bundle else bake.read_item("base", ident)
    if hit is not None:
        return hit.pieces, hit.maps.get("skin"), hit.maps.get("skin_mask")
    inst = configured()
    path, prefab = body_source(sex, body)
    if not path.exists():
        raise FileNotFoundError("oo_base 不在 assets_cache")
    loader = PrefabLoader(path)
    try:
        pieces = loader.build_skinned(prefab)
        if body is not None:
            # The game instantiates the character's own body and UncensorSelector copies the chosen
            # one's parts onto it (``ReloadCharacterBody``, §7.1191); without a choice the default
            # pack's prefab is the body.
            from .. import uncensor_choice as UC
            pieces = compose_body(_own_body_pieces(sex), pieces, body.additional,
                                  UC.all_additional_parts(from_bundle))
        # A chosen pack body's skin comes from its own bundle, by the names its manifest gives
        # (``CreateBodyTexture``'s ``chara/oo_base.unity3d`` / ``cf_body_00_t`` / ``c?_body_00_mc``
        # are rewritten to ``SetOOBase`` / ``SetBodyMainTex`` / ``SetColorMask``, §7.1184); the default
        # pack and the built-ins take the stock ``oo_base`` maps.
        mc = BODY_SKIN_MASK[body_sex(sex, body)]
        own = body is not None and bool(body.archive)
        tex_name = (body.main_tex or BODY_SKIN) if own else BODY_SKIN
        mask_name = (body.color_mask or mc) if own else mc
        stock = inst.bundle(BODY_BUNDLE)
        skin_loader = loader
        close_stock = False
        if not own and path.resolve() != stock.resolve() and stock.exists():
            skin_loader = PrefabLoader(stock)
            close_stock = True
        try:
            skin = skin_loader.texture_by_name(_lod(tex_name), tex_name + "_low")
            skin_mask = skin_loader.texture_by_name(_lod(mask_name), mask_name + "_low")
        finally:
            if close_stock:
                skin_loader.env = None
                skin_loader.by_id = {}
        if own and (skin is None or skin_mask is None):
            raise FileNotFoundError(f"{body.guid}: {tex_name if skin is None else mask_name} 不在 {path.name}")
    finally:
        loader.env = None
        loader.by_id = {}
    if sex == 1:
        pieces = _overlay_balls(_overlay_male_penis(pieces, penis), balls)
    else:
        # a woman whose card shows them: the same swap onto her body's own renderers of those names
        if penis is not None:
            pieces = _overlay_male_penis(pieces, penis)
        pieces = _overlay_balls(pieces, balls)
    return bust_normals(color_match(pieces, (body, penis, balls)), body), skin, skin_mask


def _own_skin(pieces, skin, skin_mask):
    """Swap the body's albedo and colour mask for generated ones, if enabled.

    Called from :func:`apply_body` — the **dress** path — and deliberately not
    from :func:`load_body_prefab`, which ``tools/bake_assets.py`` also walks
    (``origin.convert``).  Putting it there would write our own sheets into
    ``assets_baked`` the next time anybody re-baked with the switch on, and a
    generated map that has quietly become the library's copy of Illusion's is
    the kind of contamination nothing reports.

    The albedo is white and the mask is binary, so this is exact on the first
    and loses only the nail region on the second — see :mod:`mesh.parttex`.
    """
    from ...mesh import parttex as PT
    if not PT.enabled():
        return skin, skin_mask
    body = next((p for p in pieces
                 if str(getattr(getattr(p, "mesh", None), "name", "") or "")
                 == "o_body_a"), None)
    return PT.skin_sheets(body)


def load_head_prefab(head: "HeadAssets | None" = None,
                     from_bundle: bool = False):
    """(pieces, face albedo, face colour mask) — from the bake when present."""
    from .. import bake
    head = head or head_assets()
    hit = None if from_bundle else bake.read_item("base", head.bake_ident)
    if hit is not None:
        return hit.pieces, hit.maps.get("face_t"), hit.maps.get("face_mc")
    if head.source == "mod":
        from .load_mesh import mod_pieces
        return mod_pieces(head)
    from .load_mesh import _open_bundle, _stock_texture
    # The prefab, and the albedo / colour mask ``InitBaseCustomTextureFace``
    # asks for by name, come from whichever source Sideloader hands them out
    # of — ``[KK]Face Fixes`` re-registers ``chara/bo_head_00.unity3d`` with its
    # own ``p_cf_head_00`` (§7.489).  Only a path a mod can answer at leaves the
    # install; every other head is the one open it always was.
    loader = _open_bundle(head.bundle, head.prefab)
    if loader is None:
        raise FileNotFoundError(f"{head.bundle} 不在 assets_cache")
    # ``<ChangeHeadAsync>`` passes ``copyDynamicBone = false``, so the head's
    # chains keep the prefab's own (null) collider list instead of the body's
    # set — see :meth:`prefab.PrefabLoader._colliders_for`.
    loader.copy_dynamic_bone = False
    try:
        pieces = loader.build_skinned(head.prefab)
    finally:
        loader.env = None
        loader.by_id = {}
    face_t = _stock_texture(head.bundle, _lod(head.tex))
    face_mc = _stock_texture(head.bundle, _lod(head.mask))
    return pieces, face_t, face_mc


def eyes_open_max(card, look: dict | None = None) -> float:
    """``EyesCtrl.OpenMax`` — how far the lids can open, and it is not 1.

    The clamp is read rather than assumed::

        ChaControl.ChangeEyesOpenMax:
            fileStatus.eyesOpenMax = Clamp(v, 0, 1)
            eyesCtrl.OpenMax       = Clamp(v, 0, 0.92)      <-- 0.92, not 1

    ``FaceBlendShape.LateUpdate`` then caps it again per frame at
    ``1 - EyeLookUpCorrect`` and adds the gaze correction on top — that part is
    :func:`face_morph.gaze_open_max`, and with the gaze at rest it comes back
    to ``min(1 - upCorrect, OpenMax)``.  What matters here is that ``OpenMax``
    itself is the number the correction is *scaled* by, so this returns it
    whole rather than pre-capped; capping first cost the male head 2% of its
    gaze travel (its ``EyeLookUpCorrect`` is 0.1, so the cap bites).

    ``FBSBase.CalculateBlendShape`` then plays the pattern's ``Close`` /
    ``Open`` targets at ``100 - rate*100`` / ``rate*100``.  For eye pattern 0
    the pair is (``*_def_cl``, ``*_def_op``) and ``*_def_op`` ships **zero**
    vertices — it *is* the base mesh — so all the resting state amounts to is
    ``def_cl`` held at ``1 - correctOpenMax``: **8%** on the stock female head
    with a card asking for fully open eyes, 10% on the male one.

    Not applying it is why our lids sat higher than the game's.  ``def_cl`` is
    the same shape the blink uses, on all nine meshes ``EyesCtrl`` drives
    (face, nose line, both sclerae, both lash sheets, three tear meshes), which
    is exactly what :func:`face_morph.register` keys by suffix.
    """
    status = {}
    if hasattr(card, "sections"):
        status = card.sections.get("Status") or {}
    want = _num(status.get("eyesOpenMax"), 1.0)
    # KK_EyeControl's prefix caps the argument before the game clamps it
    from .face_morph import eye_control
    cap = eye_control(card)[0]
    if want > cap:
        want = cap
    return float(np.clip(want, 0.0, 0.92))


_HEAD_NODE_CACHE: dict[tuple, tuple] = {}


def head_bone_hierarchy(head: "HeadAssets | None" = None,
                        from_bundle: bool = False) -> tuple[dict, str]:
    """``(name → (parent, local 4×4), where it came from)`` for the card's head.

    The head bone rig on a character **is** the head list item's own prefab.
    ``ChangeHeadAsync`` instantiates it and parents it under ``cf_s_head``;
    ``oo_base``'s ``p_cf_head_bone`` is a reference rig that is never on a
    character at all.  Fifty sections never noticed because the two agree on
    every ``cf_J_*`` joint of head 0 — but not on the ``a_n_*`` attach frames,
    which is where every head accessory hangs (§7.102):

    ======================  ========
    node                    apart
    ======================  ========
    ``a_n_hair_pin``/``_R``  92.6 mm
    ``a_n_hair_pony``        45.4 mm
    ``a_n_headflont``        37.7 mm
    ``a_n_head``             36.7 mm
    ``a_n_headtop``          28.2 mm
    ======================  ========

    and the 200/201 heads move 27 ``cf_J_*`` joints as well, so this is not a
    table of attach points — it is the head's hierarchy, whatever is in it.

    Returns an **empty** dict with a reason when the head cannot be read.  The
    caller keeps the reference rig and reports; it does not get a table that
    looks usable (CLAUDE.md 一之一).
    """
    from .. import bake
    head = head or head_assets()
    key = (head.bundle, head.prefab, bool(from_bundle))
    if key in _HEAD_NODE_CACHE:
        return _HEAD_NODE_CACHE[key]
    got: dict = {}
    why = ""
    if not from_bundle:
        got = bake.read_nodes("base", head.bake_ident) or {}
        why = "bake" if got else ""
    if not got:
        # Same fallback shape as ``load_head_prefab``: an item this machine has
        # not baked is read from the bundle rather than skipped, because the
        # alternative is placing the accessories off a rig this head does not
        # use.  It costs a bundle open, which is why it says so.
        loader = None
        try:
            if head.source == "mod" and head.mod is not None:
                from .load_mesh import _open_zipmod
                loader = _open_zipmod(head.mod)
                if loader is None:
                    raise FileNotFoundError(f"讀不到 zipmod 的頭：{head.prefab}")
            else:
                from .load_mesh import _open_bundle
                # the prefab ``load_head_prefab`` builds — the same source (§7.489)
                loader = _open_bundle(head.bundle, head.prefab)
                if loader is None:
                    raise FileNotFoundError(f"{head.bundle} 不在 assets_cache")
            loader.copy_dynamic_bone = False
            got = loader.node_hierarchy(head.prefab)
            why = "bundle" if got else f"{head.bundle} 裡沒有 {head.prefab}"
        except Exception as exc:                     # noqa: BLE001
            got, why = {}, str(exc)
        finally:
            if loader is not None:
                loader.env = None
                loader.by_id = {}
    _HEAD_NODE_CACHE[key] = (got, why)
    return got, why


_EYE_LOOK_CACHE: dict[tuple, dict] = {}


def eye_look_settings(head: "HeadAssets | None", side: int) -> dict | None:
    """``EyeLookMaterialControll`` for one eye, read from the head prefab.

    Only reached when the piece did not bring it along — an ``assets_baked``
    tree written before the component travelled inside ``piece.shader``.  The
    prefab read is the same one ``load_head_prefab`` does, so it costs about a
    second, once per head per process.
    """
    head = head or head_assets()
    key = (head.bundle, head.prefab, int(side))
    if key in _EYE_LOOK_CACHE:
        return _EYE_LOOK_CACHE[key]
    found: dict[int, dict] = {}
    try:
        pieces, _, _ = load_head_prefab(head, from_bundle=True)
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.body_head.eye_look_settings: pieces, _, _ = load_head_prefab(head, fro…", exc)
        pieces = []
    for piece in pieces:
        look = (piece.shader or {}).get("_eyeLook")
        if look:
            found[int(look.get("eyeLR", 0) or 0)] = look
    for lr in (0, 1):
        _EYE_LOOK_CACHE[(head.bundle, head.prefab, lr)] = found.get(lr)
    return _EYE_LOOK_CACHE.get(key)


def graft_implant_bones(skeleton, mapper, sex: int = 0, *,
                        report=None, body=None, ident: str | None = None, penis=None, balls=None) -> int:
    """Add the body pack's own bones to the armature, before anything skins.

    ``graft_full_skeleton`` builds from Illusion's **reference** rig, so a bone
    that exists only in an uncensor pack is missing — and missing is not
    harmless: ``skinmap._resolve``'s ``exact`` branch redirects every unknown
    name to ``cf_j_hips``, so the geometry bound to it follows the pelvis.  On
    a BetterPenetration body that is the whole crotch, which is what drew two
    wedges from the groin to the feet in §7.123 / §7.125 and again the moment
    the pack was enabled on its own (§7.129).

    The entries carry a **parent-relative** 4×4 in document order, so a bone's
    parent is always already present when it is reached and no space conversion
    is needed — only the translation is scaled, exactly as
    ``graft_full_skeleton`` does (the rest of the graft carries the rig's size
    in the translations, not in the 3×3).

    Returns how many were added.  Zero for every stock body, which is the state
    the loader was already in.
    """
    from ...core import math3d as m3
    from .. import bake

    ident = ident or body_ident(sex, body)
    hit = bake.read_item("base", ident)
    if hit is not None:
        rows = list(getattr(hit, "implant", None) or [])
    else:
        # **No archive at all is not "this pack adds nothing".**  Without a
        # bake the body itself comes off the bundle (``load_body_prefab``), so
        # its own subtree has to come from the same place — reading only the
        # bake left the bundle path with 0 implants and every crotch vertex on
        # the hips, which is what turned ``bake_verify``'s *control* red on
        # every card: 2,229 vertices, 953.6 mm, while the two ``o_body_a``
        # pieces were bit-identical (§7.138 六, fixed §7.353).  An archive that
        # exists but carries no ``implant`` is a stock body and stays at 0.
        rows = body_implant_bones(sex, body)
        # a man's penis and balls packs bring their own bones too (§7.1186)
        names = {r["name"] for r in rows}
        for row in pack_implant_bones(penis) + pack_implant_bones(balls):
            if row["name"] not in names:
                names.add(row["name"])
                rows.append(row)
    scale = float(getattr(mapper, "scale", 1.0) or 1.0)
    skeleton.dan_colliders = _scaled_dan_colliders(rows, scale)
    if not rows:
        return 0
    added = skipped = 0
    for row in rows:
        name = str(row.get("name") or "")
        parent = str(row.get("parent") or "")
        loc = row.get("local")
        if not name or not parent or loc is None or skeleton.has(name):
            continue
        if not skeleton.has(parent):
            # Its parent is neither on the rig nor earlier in this list — a
            # detached subtree we would be guessing the placement of, so it is
            # reported rather than hung off something plausible (§一之一).
            skipped += 1
            continue
        m = np.asarray(loc, dtype=np.float64).reshape(4, 4).copy()
        m[:3, 3] *= scale
        t, q, sc = m3.decompose(m)
        skeleton.add(name, parent, t, q, sc)
        added += 1
    if added:
        skeleton.finalize()
        # ``_resolve`` memoises, and it has already been asked for some of
        # these names by an earlier piece — a stale entry would keep pointing
        # at the hips for the rest of the load.
        cache = getattr(mapper, "_cache", None)
        if isinstance(cache, dict):
            cache.clear()
    if skipped and report is not None:
        report.missing.append(
            f"body: {skipped} 根 pack 自帶的骨頭找不到父節點，未植入")
    return added


def _scaled_dan_colliders(rows: list, scale: float) -> list:
    """``[(bone, dynbone.Collider)]`` of the rows' ``dan_colliders``, at the rig's size.

    The graft puts the rig's size in the translations, not in any 3×3, so the
    bone's lossy scale no longer carries it the way ``cf_n_height`` makes the
    game's do; the collider's own lengths take it instead.
    """
    from .. import dynbone as DB

    out = []
    for row in rows:
        for c in row.get("dan_colliders") or ():
            out.append((str(row.get("name") or ""), DB.Collider(
                centre=np.asarray(c["centre"], dtype=np.float64) * scale, radius=float(c["radius"]) * scale,
                height=float(c["height"]) * scale, direction=int(c["direction"]), bound=int(c["bound"]))))
    return out


def _material_editor_character(pieces, card, report, label: str) -> list:
    """MaterialEditor's ``ObjectType.Character`` material rows on *pieces*.

    ``FindGameObject(Character)`` is the whole character, so these rows match
    a material by name wherever it is — the body and the head are each handed
    the full set and take what they carry.  An unmatched name is therefore not
    reported here (it may well be the other prefab's); the renderer rows'
    ``character_hidden_parts`` makes the same call.
    """
    from . import matedit as ME
    if card is None:
        return list(pieces)
    edits = ME.read_material_edits(card)
    pieces, written, _, _counts = ME.edit_pieces(pieces, edits, ME.CHARACTER, 0)
    if written:
        report.replaced.append(f"MaterialEditor：{label} 材質覆寫 {written} 個值")
    return pieces


def chosen_packs(card, sex: int) -> tuple:
    """``(body, penis, balls)`` UncensorSelector puts on *card* as our *sex* (0 woman, 1 man); all None for the default pack.

    The default pack when there is no card or the install runs no UncensorSelector (``uncensor_choice.for_card``,
    #93).  A woman's penis and balls are None unless her card shows them (``uncensor_choice.worn``).
    """
    if card is None:
        return None, None, None
    from .. import uncensor_choice as UC
    sex = 1 if int(sex or 0) else 0
    got = UC.for_card(card, 0 if sex == 1 else 1)
    if got is None:
        return None, None, None
    return got["body"], got["penis"], got["balls"]


def apply_body(built, mapper, report, card=None, outfit: int = 0,
               res: dict | None = None, sex: int = 0) -> bool:
    part = built.model.get("body")
    if part is None:
        return False
    sex = 1 if int(sex or 0) else 0
    body, penis, balls = chosen_packs(card, sex)
    ident = body_ident(sex, body, penis, balls)
    try:
        pieces, skin, skin_mask = load_body_prefab(sex, body=body, penis=penis, balls=balls)
    except Exception as exc:  # noqa: BLE001
        report.missing.append(f"body: {exc}")
        return False
    #: which body archive this character wears — the BetterPenetration push reads its chains from it
    built.body_ident = ident
    skin, skin_mask = _own_skin(pieces, skin, skin_mask)
    # ``o_body_a``'s material lives in mm_base, so its piece carries only the
    # bookkeeping keys and the authored properties are merged in below
    # (``create_props``).  MaterialEditor edits that material by name, so the
    # merge has to happen first — and the name has to be there — or its 343
    # local body rows match nothing (§7.385).
    bmat = BODY_MATERIAL[body_sex(sex, body)]
    pieces = [p._replace(shader={**create_props(bmat), **(p.shader or {}),
                                 "_matName": bmat})
              if str(getattr(p.mesh, "name", "") or "") == BODY_MESH else p
              for p in pieces]
    pieces = _material_editor_character(pieces, card, report, "body")

    # **Before anything is retargeted.**  The pack's own bones have to be on
    # the armature by the time ``mapper.remap`` sees a vertex weighted to them,
    # or the name resolves to ``cf_j_hips`` and is then memoised (§7.129).
    n_imp = graft_implant_bones(built.skeleton, mapper, sex, report=report, body=body, ident=ident,
                                penis=penis, balls=balls)
    if n_imp:
        from .report import Attached as _Att
        report.attached.append(_Att(f"implant×{n_imp}", "body", "pack", 0))
    # the packs' own DynamicBones (BetterPenetration's scrotum), solved every frame like the body's
    # other chains (``sway_live``, ``body``): their components are part of the body, so neither the
    # clothes' nor the hair's switch stops them
    for chain in pack_chain_specs(ident, penis, balls):
        if chain.get("unsolved"):
            root = next((b for b in chain.get("bones") or () if b), "?")
            report.missing.append(f"搖擺骨: body pack {chain.get('pack', '?')} 的 {root} 這條沒有解算 —— {chain['unsolved']}")
        else:
            built.sway_specs.append({**chain, "body": True})

    # Drop any procedural genital stand-ins — Origin meshes replace them.
    for old in list(built.model.by_group("genitals")):
        if str(old.name).startswith("gen_"):
            built.model.remove(old.name)

    want_gen = set(GENITAL_MESHES.get(sex) or ())
    if sex == 1 and card is not None:
        # what his card shows (``uncensor_choice.shown``): the balls hang under the shaft's object
        from .. import uncensor_choice as UC
        shaft_on, balls_on = UC.shown_for_card(card, 0)
        want_gen -= ({"o_dankon"} if not shaft_on else set()) | ({"o_dan_f"} if not balls_on else set())
    if sex == 0:
        # what a woman's card shows (``visibleSonAlways`` / the ``o_dan_f`` object): her body's own
        # renderers of those names, which a stock female body has only the shaft of
        have = {str(p.mesh.name or "") for p in pieces if p.mesh is not None}
        want_gen |= {n for n, on in (("o_dankon", penis is not None), ("o_dan_f", balls is not None))
                     if on and n in have}
    body_ok = False
    body_tex_key = None
    body_shade_key = None
    body_nrm_key = None
    body_detail_key = None
    body_shader = None
    power = nail = None

    # Drop leftover BP open experiment handles (implant/sim — caused warp).
    if built is not None:
        for attr in ("_kokan_dyn_sim", "_kokan_body_db"):
            if hasattr(built, attr):
                delattr(built, attr)

    tongue = None
    gomu = None
    for piece in pieces:
        PROG.tick()
        mesh = piece.mesh
        if mesh is None:
            continue
        mname = str(mesh.name or "")
        if mname == "o_tang":
            tongue = piece                   # ``S_TongueB``: attached after, hidden (``set_tongue_state``)
            continue
        if mname == "o_gomu" and sex == 1:
            gomu = piece                     # attached after with its own material, hidden (``set_condom``)
            continue
        is_body = mname == BODY_MESH
        is_gen = mname in want_gen
        if not is_body and not is_gen:
            continue
        if piece.skin is None or not piece.bones:
            continue
        blended = bust_normal_blend(piece, card) if is_body else None
        if blended is not None:
            # ``BustNormal.Blend`` replaces the mesh's normals (bind space), so before the retarget carries them
            mesh.normals = blended
        if not mapper.retarget(mesh, piece.bones, piece.skin, piece.binds):
            continue
        idx, wt = mapper.remap(piece.bones, piece.skin)
        mesh.bone_idx, mesh.bone_wt = idx, wt
        # ``retarget`` already carried the authored normals through the blend;
        # recomputing them here replaced them with the triangles' own, which is
        # what put a hard ramp step down the sides of the neck (§7.61).
        mesh.normalize_weights()

        if is_body:
            part.mesh = mesh
            # ``o_body_a`` — MaterialEditor's ``ObjectType.Character`` rows name
            # it directly, and one local card switches the whole body off
            # (§7.138).
            part.origin_renderer = BODY_MESH
            # A MaterialEditor ``_MainTex`` replaces the create composite the
            # game bound, so it is not an input to it (§7.385).
            me_main = (CS.of(piece, "texture")
                       if (piece.shader or {}).get("_meMainTex") else None)
            main = skin if me_main is not None else (CS.of(piece, "texture") or skin)
            mask = CS.of(piece, "mask") or skin_mask
            tex = CS.pixels(main)
            # §7.12 for the body too.  ``o_body_a``'s material is ``cf_m_body``,
            # which lives in **mm_base**, not in oo_base — an external reference
            # the prefab loader cannot resolve, so ``piece.shader`` comes back
            # empty and the body kept our hard-coded toon defaults.  Its real
            # settings are a warm near-white shadow (1.00, 0.90, 0.84) and a
            # half-alpha grey line, the same story as the clothes in §7.12.
            props = create_props(BODY_MATERIAL[sex])
            # **Merge, do not choose.**  ``piece.shader`` for the body is a
            # **one-key** dict — ``{'_rendName': …}``, the bookkeeping field
            # §7.138 added so MaterialEditor rows can be matched — and it is
            # truthy, so ``or`` handed it 73 authored properties' job.  Among
            # them ``_shader``, which is the only thing ``TOON.family`` reads:
            # the body came out ``FAMILY_NONE`` and the shader's whole Origin
            # forward pass is gated on ``uToonFamily != 0``, so **the body was
            # drawn with no ramp, no shade mask, no detail mask and no ink**,
            # lit by the scene light instead of the character light — while
            # ``cf_O_face`` next to it, whose ``piece.shader`` is the real
            # thing, was correct (§7.231).
            #
            # The prefab's own properties still win per key; what they cannot
            # do any more is displace the set by being present at all.
            body_shader = {**props, **(piece.shader or {})}
            apply_shader(part.material, body_shader)
            power, nail = gloss_powers(card, head=False)
            if power is not None:
                part.material.origin_specular = power
            if nail is not None:
                part.material.spec_nail = nail
            from . import toon as TOON
            part.material.ramp = TOON.ramp_key()
            from .attach import shader_normal
            # Each map keeps its own ``m_ColorSpace`` all the way to the GPU
            # (§7.126): the body's four are all linear-flagged, but that is a
            # measurement, not a rule — a modded skin brings its own flags.
            nrm_map = props.get("_NormalMap")
            nrm = shader_normal(CS.pixels(nrm_map))
            if nrm is not None and nrm_map is not None:
                body_nrm_key = TEX.store(nrm, srgb=nrm_map.srgb)
                if body_nrm_key is not None:
                    part.material.normal_map = body_nrm_key
            det_map = props.get("_DetailMask")
            dmap = CT.detail_map(CS.pixels(det_map), props)
            if dmap is not None and det_map is not None:
                body_detail_key = TEX.store(dmap, srgb=det_map.srgb)
                if body_detail_key is not None:
                    part.material.detail_mask = body_detail_key
            if card is not None:
                baked, shade = body_texture(card, main, mask, outfit=outfit,
                                            res=res, sex=sex, mesh=mesh,
                                            report=report, pack_layers=pack_skin_layers(body, report))
                if baked is not None:
                    tex = baked
                    part.material.base_color = (1.0, 1.0, 1.0, 1.0)
                if me_main is not None:
                    tex = me_main.px
                    part.material.base_color = (1.0, 1.0, 1.0, 1.0)
                if shade is not None:
                    # packed coefficients -> linear; the albedo beside it
                    # is a create-pass composite -> sRGB (§7.126)
                    body_shade_key = TEX.store(shade, srgb=False)
                    part.material.shade_mask = body_shade_key
            body_tex_key = TEX.store(tex, srgb=True)
            part.material.texture = body_tex_key
            part.body_albedo = tex          # kept so the clothes can cut into it
            # …and the key it went in under, so undressing can hand the same
            # one back instead of storing a second copy (see restore_body_skin).
            part.body_albedo_key = body_tex_key
            from .report import Attached
            report.attached.append(Attached("o_body_a", "body", "base",
                                            mesh.tri_count))
            body_ok = True
            continue

        # ---- Origin genital mesh --------------------------------------------
        from .report import Attached
        from . import toon as TOON
        gname = f"origin_{mname}"
        # remove a stale part of the same name (re-dress)
        if built.model.get(gname) is not None:
            built.model.remove(gname)
        mat = make_material(f"origin_{mname}_mat")
        # Same merge as above, and for the same reason: a one-key ``_rendName``
        # dict must not stand in for the authored material (§7.231).
        apply_shader(mat, body_shader
                     or {**create_props(BODY_MATERIAL[sex]),
                         **(piece.shader or {})})
        if power is not None:
            mat.origin_specular = power
        if nail is not None:
            mat.spec_nail = nail
        mat.ramp = TOON.ramp_key()
        # A part its pack colour-matches draws its own map, built from the card's skin (§7.1194); the body's
        # maps are in the body's UV layout, so none of them is bound to it.
        own_tex = color_match_texture(piece, card, outfit) if card is not None else None
        if own_tex is not None:
            mat.texture = TEX.store(own_tex, srgb=True)
            mat.base_color = (1.0, 1.0, 1.0, 1.0)
        elif body_tex_key is not None:
            mat.texture = body_tex_key
            mat.base_color = (1.0, 1.0, 1.0, 1.0)
        if own_tex is None and body_shade_key is not None:
            mat.shade_mask = body_shade_key
        if own_tex is None and body_nrm_key is not None:
            mat.normal_map = body_nrm_key
        if own_tex is None and body_detail_key is not None:
            mat.detail_mask = body_detail_key
        # No cutout on genital shells — body create materials carry
        # ``_alpha_a/_b`` thresholds meant for OverBodyMask holes; on the tiny
        # o_mnpa/o_mnpb UVs that clips the whole pad to invisible.
        mat.alpha_test = 0.0
        gpart = built.model.add_mesh(gname, mesh, mat, group="genitals",
                                     layer=3)
        # It is holding the body's albedo key, so ``restore_body_skin`` has to
        # move it when that key is replaced (§7.104); a colour-matched one keeps its own.
        gpart.body_skin = own_tex is None
        gpart.skinned = True
        gpart.dynamic = False
        # Under clothes by default — Free H / undress flips visibility when
        # the crotch is exposed (same rule as the old procedural parts).
        gpart.visible = False
        report.attached.append(Attached(mname, "genitals", "base",
                                        mesh.tri_count))

    if tongue is not None:
        from .attach import attach_pieces
        if built.model.get(BODY_TONGUE) is not None:
            built.model.remove(BODY_TONGUE)
        # its own group: ``apply_head`` runs after and clears the "face" group to rebuild the head
        attach_pieces(built, [tongue], mapper=mapper, group="tongue", name_prefix="origin_body_tongue", layer=2)
        shown = built.model.get(BODY_TONGUE)
        if shown is None:
            report.missing.append("body: 身體的舌頭 o_tang 沒有接上")
        else:
            shown.face_hidden = True         # ``tongueState`` 0 until a mouth pattern 21
    if gomu is not None:
        from .attach import attach_pieces
        if built.model.get(BODY_GOMU) is not None:
            built.model.remove(BODY_GOMU)
        attach_pieces(built, [gomu], mapper=mapper, group="genitals", name_prefix="origin_body_gomu", layer=3)
        worn = built.model.get(BODY_GOMU)
        if worn is None:
            report.missing.append("body: 保險套 o_gomu 沒有接上")
        else:
            worn.face_hidden = True          # ``visibleGomu`` off until the condom goes on
    if not body_ok:
        report.missing.append("body: 找不到 o_body_a")
        return False
    got = {a.name for a in (report.attached or [])}
    for need in want_gen:
        if need not in got:
            report.missing.append(f"genitals: bake 缺 {need}")
    return True


def apply_body_mask(built, masks: list, sex: int = 0,
                    alpha_ab: tuple | None = None) -> bool:
    """Cut the skin away where a garment covers it (``_AlphaMask``).

    ``ChangeClothesTopAsync`` loads the top's ``OverBodyMask`` and assigns it
    to ``customMatBody._AlphaMask``; taking the top off sets it back to null.
    Without it the skin pokes through a tight garment wherever the two
    surfaces nearly coincide.

    The channel rule is the shared one — see
    :func:`textures.apply_alpha_mask`, transcribed from ``main_skin``.
    ``_alpha_a`` / ``_alpha_b`` are **not** the asset's: ``UpdateVisible``
    calls ``ChangeAlphaMask`` with ``UpdateVisible_array4[clothesState[0]]``
    every time, so they follow the top's state — *alpha_ab*, from
    ``dress.ALPHA_BY_TOP_STATE`` (§7.369).  ``None`` keeps ``cf_m_body``'s own,
    which is (1, 1) and equals the dressed row; that is what undress passes
    when the top goes back on.
    """
    part = built.model.get("body")
    base = getattr(part, "body_albedo", None) if part is not None else None
    if part is None or base is None or not masks:
        return False
    if alpha_ab is None:
        P = create_props(BODY_MATERIAL[1 if sex else 0])
        alpha_ab = (float(P.get("_alpha_a", 1.0)), float(P.get("_alpha_b", 1.0)))
    alpha_ab = (float(alpha_ab[0]), float(alpha_ab[1]))
    # Cache the cut sheet the same way ``restore_body_skin`` caches the whole
    # one.  Undress can call this on every toggle, and the result does not
    # depend on the toggle — the mask belongs to the garment.  Compositing and
    # storing each time meant a fresh 2048² RGBA (16 MB) per click; measured
    # ten toggles walking the store key 82 → 96 (§7.104).  Every one of those
    # sheets hangs off the live body, so the store keeps them all (§7.499).
    # Keyed by the two floats as well: the sheet depends on them, and a cache
    # that ignored them would hand a half-off top the dressed cut.
    key = getattr(part, "body_masked_key", None)
    if (key is None or not TEX.alive(key)
            or getattr(part, "body_masked_ab", None) != alpha_ab):
        out = base
        for m in masks:
            out = TEX.apply_alpha_mask(out, m, alpha_ab[0], alpha_ab[1])
        if out is None or int(out[..., 3].min()) >= 128:
            return False
        key = TEX.store(out, srgb=True)
        part.body_masked_key = key
        part.body_masked_ab = alpha_ab
    part.material.texture = key
    part.material.alpha_test = TEX.ALPHA_CLIP   # the shader's own literal
    return True


def restore_body_skin(built) -> bool:
    """Undo :func:`apply_body_mask` — put the full skin albedo back.

    Free H / undress only hide clothes meshes; they used to leave the top's
    ``OverBodyMask`` holes in place, so the chest and crotch looked
    "missing" even though the body mesh and genitals were loaded.  Origin sets
    ``_AlphaMask`` to null when the top comes off; this is that step.

    The genital shells loaded from the body bake share the body's albedo:
    :func:`apply_body` hands them the very same ``TEX.store`` key.  A fresh key
    is assigned here, so theirs has to move with it — otherwise they keep the
    pre-restore sheet, the one with the garment's holes cut in (§7.104).

    **Which parts those are is ``Part.body_skin``, set by :func:`apply_body`
    where the sharing is created.**  Neither of the two things it is easy to
    reach for works here: a name test cannot tell a body shell from an uncensor
    mesh (which brings its own albedo and must keep it), and comparing against
    the body's current key finds nothing, because ``apply_body_mask`` has
    already re-stored the body with the ``OverBodyMask`` holes in between —
    measured on ``帰国子女``: body ``('originasset', 84)`` while both shells still
    hold the attach-time ``('originasset', 5)``.
    """
    part = built.model.get("body") if built is not None else None
    if part is None or part.material is None:
        return False
    base = getattr(part, "body_albedo", None)
    if base is None:
        return False
    # Undress runs this every time the crotch is exposed, so storing
    # unconditionally means another copy of a multi-MB skin sheet per toggle,
    # each kept while the body lives (§7.104, §7.499).  Re-store only once the
    # key is gone.
    key = getattr(part, "body_albedo_key", None)
    if key is None or not TEX.alive(key):
        key = TEX.store(base, srgb=True)
        part.body_albedo_key = key
    part.material.texture = key
    part.material.alpha_test = 0.0
    sync_genital_skin(built)
    return True


def sync_genital_skin(built) -> int:
    """Point the body-bake genital shells at the body's **current** albedo.

    They were handed the body's ``TEX.store`` key at attach time (see
    :func:`apply_body`), so every re-store of the body has to move them too or
    they keep drawing the sheet from before it — holes and all (§7.104).  (The
    old key cannot go white: while they point at it, the store keeps it, §7.499.)

    ``Part.body_skin`` is the marker, not the name: a name test cannot tell a
    body shell from an uncensor mesh, and the latter brings its own albedo and
    must keep it.
    """
    part = built.model.get("body") if built is not None else None
    if part is None or part.material is None:
        return 0
    key = part.material.texture
    mat = part.material
    n = 0
    for gpart in built.model.by_group("genitals"):
        if gpart.material is None or not gpart.body_skin:
            continue
        gpart.material.texture = key
        gpart.material.base_color = (1.0, 1.0, 1.0, 1.0)
        gpart.material.alpha_test = 0.0
        # shade/normal/detail from body so genitals match the skin response
        if mat.shade_mask is not None:
            gpart.material.shade_mask = mat.shade_mask
        if mat.normal_map is not None:
            gpart.material.normal_map = mat.normal_map
        if mat.detail_mask is not None:
            gpart.material.detail_mask = mat.detail_mask
        n += 1
    return n


def refit_body_mask(built, *, sex: int | None = None) -> str:
    """Cut or restore the skin to match whether a top is currently on.

    ``ChangeClothesTopAsync`` is what owns ``customMatBody._AlphaMask``: it
    assigns the top's ``OverBodyMask`` when a top loads and **null** when there
    is none.  Undressing is therefore not "hide the mesh" — hiding the garment
    without releasing the mask leaves the holes it cut, which reads as the
    chest and crotch being *missing* rather than bare (measured: taking the top
    off left ``body.material`` on the masked key with ``alpha_test`` 0.5, and
    nothing put it back unless the crotch happened to be exposed too — §7.133).

    Origin has no per-renderer undress, so the rule for one is ours: the mask
    belongs to the top as a garment, so it stays while **any** top piece is
    still drawn and lifts once the last one is hidden.

    Returns ``"masked"`` / ``"bare"`` / ``""`` (nothing to do).
    """
    if built is None or built.model is None:
        return ""
    part = built.model.get("body")
    if part is None or getattr(part, "body_albedo", None) is None:
        return ""
    from ...game.intimate.undress import logical_slot_of

    top_on = any(
        p.visible and p.group == "clothes"
        and logical_slot_of(p.name, p.group) == "top"
        for p in built.model.parts)
    masks = list(getattr(built, "body_masks", None) or ())
    if top_on and masks:
        sex = int(getattr(built, "sex", 0) if sex is None else sex)
        # ``UpdateVisible_array4[clothesState[0]]``: the top's state picks the
        # mask's two enables, so a half-off top cuts differently (§7.387).
        from .dress import ALPHA_BY_TOP_STATE
        state = list(getattr(built, "clothes_state", None) or [0])
        ab = ALPHA_BY_TOP_STATE[max(0, min(int(state[0]), 3))]
        if not apply_body_mask(built, masks, sex=sex, alpha_ab=ab):
            return ""
        # The shells share the body's albedo either way — see
        # :func:`sync_genital_skin`.  ``restore_body_skin`` does this itself.
        sync_genital_skin(built)
        return "masked"
    return "bare" if restore_body_skin(built) else ""


def _rgb(value, fallback):
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return tuple(float(x) for x in value[:3])
    return tuple(fallback)


def _multiply(tex, rgb):
    """``toon_eyew_lod0``: ``o.rgb = _MainTex.rgb * _Color``, ``o.a = _MainTex.a``.

    Transcribed rather than assumed — the fragment program is four
    instructions long and this is all of the colour work in it.  Lashes, brows
    and the sclera all use this shader.
    """
    if tex is None or rgb is None:
        return tex
    out = tex.astype(np.float32)
    out[..., :3] *= np.asarray(rgb, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _nose_line(tex, colour):
    """``toon_nose_lod0`` — *not* a multiply::

        m     = _MainTex.Sample(uv * _MainTex_ST)
        o.rgb = _Color.rgb * (1 - m.r) + m.g
        o.a   = m.a * _Color.a

    Red is a mask on the colour and **green** is the line itself; the map's
    blue never reaches the screen.  ``cf_m_noseline_00`` ships ``_Color``
    (0, 0, 0, 1), so on the stock material the result is the green channel on
    its own — which is why passing the map through unchanged happened to look
    right, and would stop looking right the moment a mod set a colour.
    """
    if tex is None or tex.ndim != 3 or tex.shape[2] < 3:
        return tex
    m = tex.astype(np.float32) / 255.0
    c = np.asarray(colour if colour is not None else (0.0, 0.0, 0.0, 1.0),
                   dtype=np.float32)
    out = np.empty(tex.shape[:2] + (4,), dtype=np.float32)
    out[..., :3] = c[:3] * (1.0 - m[..., 0])[..., None] + m[..., 1][..., None]
    a = m[..., 3] if tex.shape[2] >= 4 else np.ones(tex.shape[:2], np.float32)
    out[..., 3] = a * (c[3] if len(c) >= 4 else 1.0)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _num(value, fallback: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clampf(v: float, lo: float, hi: float) -> float:
    """``Mathf.Clamp`` on plain floats — see the note on :func:`_inv_lerp`."""
    return lo if v < lo else (hi if v > hi else v)


def _sample(src, shape, scale=(1.0, 1.0), offset=(0.0, 0.0)):
    """*src* resampled under Unity's ``uv' = uv * scale + offset``.

    Texture arrays run top-down (row 0 is ``v = 1``), so the row index is
    ``(1 - v) * h``.  Sampling is nearest — these maps are hard-edged stencils
    and a bilinear tap bleeds the iris rim out over the white.
    """
    if src is None or src.ndim != 3:
        return None
    h, w = shape
    sh, sw = src.shape[:2]
    u = (np.arange(w, dtype=np.float32) + 0.5) / w * scale[0] + offset[0]
    v = (1.0 - (np.arange(h, dtype=np.float32) + 0.5) / h) * scale[1] + offset[1]
    xi = np.clip((u * sw).astype(np.int32), 0, sw - 1)
    yi = np.clip(((1.0 - v) * sh).astype(np.int32), 0, sh - 1)
    out = src[yi][:, xi]
    # anything the transform pulled in from outside [0,1] is off the map, and
    # clamping alone would smear the border pixel across the whole margin
    if src.shape[2] >= 4:
        out = out.copy()
        oob = ((u < 0) | (u > 1))[None, :] | ((v < 0) | (v > 1))[:, None]
        out[oob, 3] = 0
    return out


def _pupil(face: dict, side: int) -> dict:
    """``ChaFileFace.pupil[side]`` — Origin stores one entry per eye."""
    pupils = face.get("pupil")
    if isinstance(pupils, list) and pupils:
        p = pupils[min(side, len(pupils) - 1)]
        if isinstance(p, dict):
            return p
    return {}


def iris_drawn(card, res: dict | None = None) -> bool:
    """Does this card's own iris sheet draw anything?

    ``False`` when the row resolves to nothing **or** to a fully transparent
    sheet — which is a real answer a card can give: ``com.DeathWeasel.NoIris``
    is two 512² images with alpha 0 and the default male card names it for both
    pupils, so every eye measurement taken off that card measured nothing
    (§7.170).  The card is not defective and neither is the loader; what is
    wrong is using it as a *fixture* (§7.365).
    """
    face = None
    try:
        face = (card.sections.get("Custom") or {}).get("face") or {}
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.body_head.iris_drawn: face = (card.sections.get('Custom') or {}).get('", exc)
        return False
    if res is None:
        try:
            from .. import sideloader as SL
            res = SL.card_resolutions(card)
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.figure.body_head.iris_drawn: from .. import sideloader as SL", exc)
            res = None
    got = _list_texture("eye", _pupil(face, 0).get("id"), res, 1)
    px = CS.pixels(got)
    if px is None or getattr(px, "ndim", 0) != 3:
        return False
    if px.shape[2] < 4:
        return True
    return int(px[..., 3].max()) > 0


def measurement_card(sex: int = 0):
    """A card whose **eyes** can be measured, for the tools that need one.

    :func:`core.paths.default_card` answers "what does this build open with",
    which is a product decision and stays; this answers "what can this
    measurement be taken on".  They are the same path unless the default's iris
    draws nothing (:func:`iris_drawn`), and then the newest card of that sex
    that does is used and the swap is **printed** — a fixture that changed
    quietly is how a measurement gets cited against the wrong card
    (``CLAUDE.md`` 七).  ``None`` only when there are no cards at all.
    """
    from ...chara.cardimport import read_origin_card
    from ...core.paths import chara_dir, default_card, list_cards

    first = default_card(sex)
    if first is not None:
        try:
            if iris_drawn(read_origin_card(first)):
                return first
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.figure.body_head.measurement_card: read_origin_card", exc)
    cards = sorted(list_cards(chara_dir(sex)),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in cards[:40]:
        if first is not None and path == first:
            continue
        try:
            if iris_drawn(read_origin_card(path)):
                print(f"[measurement_card] {first.name if first else '—'} "
                      f"的虹膜什麼都不畫（NoIris 之類）⇒ 改量 {path.name}")
                return path
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.figure.body_head.measurement_card: read_origin_card", exc)
            continue
    print(f"[measurement_card] 找不到虹膜畫得出東西的 sex={sex} 卡片，"
          f"照舊用 {first.name if first else '—'}")
    return first


def eye_white_texture(face: dict, res: dict | None = None) -> np.ndarray | None:
    """``ChangeSettingWhiteOfEye`` → ``cf_m_eyewhite_create``.

    Three inputs and nothing else: ``_MainTex`` = ``mt_eye_white[whiteId]``,
    ``_Color`` = ``whiteBaseColor``, ``_Color2`` = ``whiteSubColor``.  The map
    (``cw_t_sirome_kage_NNN`` — 影 = shadow) is a greyscale stencil: white
    inside the lit part of the sclera, dark where the lid shades it.  Entry 0
    is an 8×8 pure white sheet, which is how "no shadow" is spelled.
    """
    white = _list_texture("eye_white", face.get("whiteId"), res)
    return CT.eyewhite(white, face.get("whiteBaseColor"),
                       face.get("whiteSubColor"), shape=(64, 64))


# Plain arithmetic rather than ``np.clip``: these run per frame per eye once a
# gaze is being driven, and a numpy call on a single float costs more than the
# whole of ``eye_look_st`` around it (48 of them a frame, a quarter of the
# gaze's cost, before this).
def _inv_lerp(a: float, b: float, v: float) -> float:
    """``Mathf.InverseLerp`` — clamped to 0..1, 0 when the range is empty."""
    if a == b:
        return 0.0
    t = (v - a) / (b - a)
    return 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)


def _mlerp(a: float, b: float, t: float) -> float:
    """``Mathf.Lerp`` — the interpolant is clamped, unlike :func:`_lerp`."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return a + (b - a) * t


def eye_look_st(face: dict, side: int, look: dict | None,
                gaze: tuple[float, float] = (0.0, 0.0),
                shake: tuple[float, float] | None = None) -> tuple:
    """``EyeLookMaterialControll.Update`` — the ST it writes on the iris.

    *shake* is ``YureAddScale`` for a slot with ``isYure`` set (the highlights
    under ``ChangeEyesShaking(true)``, :class:`HighlightShake`): that slot
    follows the gaze with ``power * (0.8, 0.5)`` and its texture scale is
    ``scale * YureAddScale`` in both the offset's centring and the tiling.
    The ``Lerp(1, 5, scale)`` inside the offset keeps the unshaken scale.

    Returns ``(tiling_x, tiling_y, offset_x, offset_y, hl_up_dy, hl_down_dy)``.

    ``ChaControl`` fills the component in from the card, one call per slider::

        ChangeSettingEyePosX        offset.x = ±Lerp( 0.2, -0.6, pupilX)
        ChangeSettingEyePosY        offset.y =  Lerp(-0.5,  0.5, pupilY)
        ChangeSettingEyeScaleWidth  scale.x  =  Lerp( 1.8, -0.2, pupilWidth)
        ChangeSettingEyeScaleHeight scale.y  =  Lerp( 1.8, -0.2, pupilHeight)
        ChangeSettingEyeHLUpPosY    hlUpOffsetY   = Lerp(0.1, -0.1, hlUpY)
        ChangeSettingEyeHLDownPosY  hlDownOffsetY = Lerp(0.1, -0.1, hlDownY)

    (``SetEyeTexOffsetX`` negates its argument for the right eye, which is why
    the two irises sit mirrored rather than both pulled the same way.)  Then,
    every frame, for each of the three slots ``_MainTex`` / ``_overtex1`` /
    ``_overtex2``::

        dir   = (GetAngleHRate(eyeLR) + offset.x, GetAngleVRate() + offset.y)
        if |dir| > 1: dir = dir.normalized
        xWait = Lerp(InsideWait, OutsideWait, InverseLerp(-1, 1, dir.x))
        yWait = Lerp(DownWait,   UpWait,      InverseLerp(-1, 1, dir.y))
        o     = (Clamp(xWait * power * Lerp(1, 5, scale.x), InsideLimit, OutsideLimit),
                 Clamp(yWait * power * Lerp(1, 5, scale.y), UpLimit,     DownLimit))
        o.y  += hlUpOffsetY (slot 1) / hlDownOffsetY (slot 2)
        SetTextureOffset(slot, o + scale * -0.5)
        SetTextureScale (slot, 1 + scale)

    ``GetAngleHRate`` / ``GetAngleVRate`` are the *look* direction and are 0
    when the eyes are level and forward (``AngleVRateCalc`` returns
    ``-InverseLerp(0, upBendingAngle, 0)``), so a still frame passes
    ``gaze = (0, 0)`` — but **the gaze term does not vanish with it**, because
    ``offset`` is inside it.  §7.19 and §7.45 both dropped the whole term and
    were left with ``offset = -scale/2``, which is only right for a card whose
    ``pupilX`` is 0.25 and ``pupilY`` 0.5.  The stock component ships
    ``offset = (∓0.2, -0.2)``, i.e. what ``pupilX = pupilY = 0.5`` produces —
    the neutral card is already off-centre.

    The waits and limits come from the head prefab because they are not
    constants: ``Initialize`` would set 156/100/156/100 but nothing calls it,
    and 46 of the 1,068 components in the installed head mods carry values
    other than the stock ±100 / ±100 / -80..80.
    """
    look = look or {}
    scale = (_mlerp(1.8, -0.2, _num(face.get("pupilWidth"))),
             _mlerp(1.8, -0.2, _num(face.get("pupilHeight"))))
    off_x = _mlerp(0.2, -0.6, _num(face.get("pupilX")))
    offset = (-off_x if int(look.get("eyeLR", side) or 0) == 1 else off_x,
              _mlerp(-0.5, 0.5, _num(face.get("pupilY"))))

    dx, dy = gaze[0] + offset[0], gaze[1] + offset[1]
    mag = math.hypot(dx, dy)
    if mag > 1.0:
        dx, dy = dx / mag, dy / mag
    x_wait = _mlerp(_num(look.get("InsideWait"), -100.0),
                    _num(look.get("OutsideWait"), 100.0), _inv_lerp(-1, 1, dx))
    y_wait = _mlerp(_num(look.get("DownWait"), 100.0),
                    _num(look.get("UpWait"), -100.0), _inv_lerp(-1, 1, dy))
    power = _num(look.get("power"), 0.001)
    px, py = (power * 0.8, power * 0.5) if shake is not None else (power, power)
    ox = _clampf(x_wait * px * _lerp(1.0, 5.0, scale[0]),
                 _num(look.get("InsideLimit"), -100.0),
                 _num(look.get("OutsideLimit"), 100.0))
    oy = _clampf(y_wait * py * _lerp(1.0, 5.0, scale[1]),
                 _num(look.get("UpLimit"), -80.0),
                 _num(look.get("DownLimit"), 80.0))
    sx, sy = ((scale[0] * shake[0], scale[1] * shake[1]) if shake is not None
              else scale)
    return (1.0 + sx, 1.0 + sy,
            ox - 0.5 * sx, oy - 0.5 * sy,
            _mlerp(0.1, -0.1, _num(face.get("hlUpY"))),
            _mlerp(0.1, -0.1, _num(face.get("hlDownY"))))


def eye_look_delta(face: dict, side: int, look: dict | None,
                   gaze: tuple[float, float], rest=None) -> tuple[float, float]:
    """The iris material's ``uv_offset`` for a gaze — the runtime half of the ST.

    :func:`eye_texture` bakes the whole create pass at the ST a gaze of ``(0, 0)``
    produces, so the baked map is ``T(p) = C(p * tiling + o0)`` over the *drawn*
    surface coordinate.  Wanting ``C(p * tiling + o)`` from that map means
    sampling it at ``p + (o - o0) / tiling`` — the divide is the part a plain
    "offset difference" gets wrong, and it matters: ``tiling`` is ``1 + scale``
    and a card's pupil sliders push that to 1.13 either way.

    ``Update`` adds the same gaze to ``_MainTex``, ``_overtex1`` and
    ``_overtex2``; the two highlights are their own map on UV1 and take it
    through :func:`eye_highlight_xf`, which with no shake is this same shift.

    *rest* is that eye's baked ST if the caller kept it (``built.eye_st``); it
    does not change for the life of the character, and recomputing it every
    frame doubled the work.
    """
    tx, ty, ox0, oy0 = (rest if rest is not None
                        else eye_look_st(face, side, look)[:4])
    _, _, ox, oy, _, _ = eye_look_st(face, side, look, gaze)
    return ((ox - ox0) / tx if abs(tx) > 1e-6 else 0.0,
            (oy - oy0) / ty if abs(ty) > 1e-6 else 0.0)


# Overrides the card's ``pupil`` colours for every character built after it is
# set: ``{"base": (r, g, b), "sub": (r, g, b), "blend": float}``, or empty for
# "use the card".
# ``CharacterController`` fills it from ``face.iris_color`` /
# ``face.iris_grad_color`` just before it dresses the character, which is what
# stopped those two being orphans: ``cardimport`` had copied the card's colours
# into them since forever and nothing read them back, so the sliders moved and
# the eye did not (§7.148 二十七).
EYE_COLOUR_OVERRIDE: dict = {}


def eye_highlight_xf(face: dict, side: int, look: dict | None,
                     gaze: tuple[float, float], rest=None,
                     shake: tuple[float, float] | None = None) -> tuple:
    """``(kx, ky, bx, by)``: where the UV1 highlight map is read this frame.

    :func:`eye_highlight` bakes ``A(p * T0 + O0)`` at the gaze-0 ST; ``Update``
    wants ``A(p * T + O)``, which is the baked map at ``p * T/T0 + (O - O0)/T0``.
    With no shake ``T = T0`` and this is :func:`eye_look_delta`'s shift; the
    shake (*shake* = ``YureAddScale``) changes the tiling as well.  The two
    slots' ``hlUp``/``hlDownOffsetY`` sit in both ``O`` and ``O0`` and cancel.
    """
    tx0, ty0, ox0, oy0 = (rest if rest is not None
                          else eye_look_st(face, side, look)[:4])
    tx, ty, ox, oy, _, _ = eye_look_st(face, side, look, gaze, shake)
    if abs(tx0) < 1e-6 or abs(ty0) < 1e-6:
        return (1.0, 1.0, 0.0, 0.0)
    return (tx / tx0, ty / ty0, (ox - ox0) / tx0, (oy - oy0) / ty0)


class HighlightShake:
    """One iris's ``YureTimer`` / ``YureAddScale`` — ``Update``'s shake state.

    Per frame, before the slot loop::

        flag = YureTime < YureTimer;  if flag: YureTimer = 0
        YureTimer += dt
        s = scale * YureAddScale                  # this frame's, not the new one

    and inside it, for each slot with ``isYure`` (the two highlights)::

        if flag: YureAddScale = (Random(1, 2), Random(1, 1.5)); YureTimer += dt

    ``YureAddScale`` is not serialised, so it starts at ``(0, 0)`` — the first
    shaken frames draw the highlights at tiling 1 until the timer fires — and
    ``YureAddVec`` has no writer at all.  The timer runs whether or not the
    eyes shake; only the re-roll waits for ``isYure``.  Each eye is its own
    component, so each gets its own instance.
    """

    SLOTS = 2                                     # _overtex1, _overtex2

    def __init__(self, period: float, rng=None):
        import random
        self.period = float(period)
        self.timer = 0.0
        self.add = (0.0, 0.0)
        self.on = False
        self.current = self.add
        self._rng = rng or random.Random()

    def tick(self, dt: float) -> tuple[float, float]:
        """Advance one frame; returns (and keeps as ``current``) this frame's scale."""
        flag = self.period < self.timer
        if flag:
            self.timer = 0.0
        self.timer += float(dt)
        self.current = self.add
        if self.on and flag:
            for _ in range(self.SLOTS):
                self.add = (self._rng.uniform(1.0, 2.0),
                            self._rng.uniform(1.0, 1.5))
                self.timer += float(dt)
        return self.current


def eye_texture(face: dict, side: int = 0, res: dict | None = None, *,
                look: dict | None = None, report=None,
                skin: dict | None = None) -> np.ndarray | None:
    """``ChangeSettingEye`` → ``cf_m_eye_create``: the iris map, on UV0.

    The two highlights ``toon_eye_lod0`` lays over it are :func:`eye_highlight`
    — their own map on UV1, sampled at draw time (§7.403).

    The composite itself is :func:`create_tex.eye`, transcribed from the
    shader's bytecode.  What ``ChaControl`` feeds it::

        _MainTex   = mt_eye[pupil[side].id]
        _ColorMask = mt_eye_gradation[pupil[side].gradMaskId]
        _Color     = pupil.baseColor      _Color2 = pupil.subColor
        _Blend     = pupil.gradBlend
        _grad      = (0, lerp(-.5, .5, gradOffsetY), 0, lerp(-1, 1, gradScale))

    The ST that map is then sampled with is :func:`eye_look_st`.
    """
    pupil = _pupil(face, side)
    iris = _list_texture("eye", pupil.get("id"), res, side + 1)
    if iris is None or iris.px.ndim != 3 or iris.px.shape[2] < 4:
        return None
    # A map that resolved fine and is **entirely transparent** is a legitimate
    # answer — ``com.DeathWeasel.NoIris`` ships a 512² sheet with alpha 0 for
    # exactly this — but it is a surprising one, and unreported it reads as a
    # defect: the default male card wears that mod, so its eyes are sclera and
    # catchlights only, ``report.missing`` is 0, and §7.143 七 spent a section
    # calling it "the male iris is white" with the whole male face blocked
    # behind it (§7.170).  One line here is the difference between a question
    # and an answer.  Not ``missing`` — nothing is missing.
    if report is not None and int(np.asarray(iris.px)[..., 3].max()) == 0:
        line = (f"虹膜：卡片指名的 mt_eye#{pupil.get('id')} 整張透明"
                "（NoIris 之類的 mod），只會畫出鞏膜與高光")
        if line not in report.replaced:
            report.replaced.append(line)
    # Overlay Mods: ``ctCreateEyeL``／``R`` (side 0 / 1, ``ChangeSettingEye``'s
    # array order) get their underlays on ``texMain`` — the iris map — before
    # the create pass and their overlays on the result (:mod:`skin_overlay`).
    lr = "L" if side == 0 else "R"
    iris = SO.composite_map(iris, (skin or {}).get(f"EyeUnder{lr}", []))
    h, w = iris.px.shape[:2]

    tx, ty, ox, oy, up_dy, down_dy = eye_look_st(face, side, look)

    grad = _list_texture("eye_gradation", pupil.get("gradMaskId"), res, side + 1)
    grad_vec = (0.0,
                _lerp(-0.5, 0.5, _num(pupil.get("gradOffsetY"))),
                0.0,
                _lerp(-1.0, 1.0, _num(pupil.get("gradScale"))))
    P = create_props("cf_m_eye_create")
    # The create pass runs at the *drawn* surface coordinate: its own
    # ``_MainTex_ST`` is the identity and the window belongs to the material
    # the result is drawn with, so it moves the gradient mask too.
    grid = CT._grid((h, w))
    # ``_Blend`` is **not** clamped to its declared ``Range(0, 1)``:
    # ``ChangeSettingEye`` hands the card's ``gradBlend`` straight to
    # ``SetFloat`` and the range only ever bound the inspector slider, so the
    # final ``mad _Blend, vivid - tint*m, tint*m`` extrapolates past the vivid
    # layer.  Only the vivid term itself is saturated (``movc_sat``); the write
    # is clamped by the ARGB32 render target, which is what ``_u8`` does here.
    # Cards do reach for it: 96 of the 1,241 local female cards and 3 of the 145
    # male ones sit outside [0, 1], over -0.357 .. 2.0.  ``Kayama Nemuri`` ships
    # 2.0, where the clip was worth 0.306 -> 0.180 of iris saturation; every
    # in-range card composites bit-for-bit as before.
    # **The card's two eye colours had no reader.**  ``cardimport`` has always
    # copied them into ``face.iris_color`` / ``face.iris_grad_color``, and a
    # grep for those keys finds the declaration in ``params`` and the write in
    # ``cardimport`` and **nothing else** — three orphan parameters, so the
    # sliders moved and the eye did not (``CLAUDE.md`` 七: an orphan is not
    # "kept for later", it is "the place that should call it does not").
    # Wiring them here rather than at the draw call because this is where the
    # colour enters: ``create_eye`` bakes it into the map, and a tint applied
    # afterwards would multiply the highlights too.
    _base, _sub = pupil.get("baseColor"), pupil.get("subColor")
    _ov = EYE_COLOUR_OVERRIDE
    if _ov:
        _base = _ov.get("base", _base) or _base
        _sub = _ov.get("sub", _sub) or _sub
    _blend = _num(_ov.get("blend") if _ov and "blend" in _ov
                  else pupil.get("gradBlend"), 0.0)
    out = CT.eye(iris, grad, _base, _sub, _blend,
                 grad_vec, P.get("_MainTex_ST"), P.get("_ColorMask_ST"),
                 uv=(grid[0] * tx + ox, grid[1] * ty + oy))
    if out is None:
        return None
    # ``createTex`` is unwindowed and our map bakes the window in ⇒ the overlay
    # is read where each of our texels sits in it
    return SO.composite_baked(out, (skin or {}).get(f"EyeOver{lr}", []),
                              uv=(grid[0] * tx + ox, grid[1] * ty + oy))


def eye_texture_replaced(img, face: dict, side: int = 0,
                         look: dict | None = None) -> np.ndarray | None:
    """A MaterialEditor ``_MainTex`` on the iris, baked the way :func:`eye_texture` is.

    ME's ``MaterialAPI.SetTexture`` puts the image on the **draw** material
    (``cf_m_hitomi_00``) in place of the ``create_eye`` render texture, and
    gives it that texture's ``wrapMode`` (``SetTexture``: the new map takes
    ``GetTexture(...).wrapMode``, Repeat only when the slot was empty).  The
    ST ``EyeLookMaterialControll`` writes is on the same material, so the image
    is sampled exactly where the composite was: through the gaze-0 window
    here, moved by the difference at draw time.  Same sampler, same window —
    the one question the two share (the render texture's own wrap) is the
    composite's too.  The highlights are their own map and stay (§7.403).
    """
    if img is None or img.px.ndim != 3:
        return None
    tx, ty, ox, oy, _up, _down = eye_look_st(face, side, look)
    h, w = img.px.shape[:2]
    grid = CT._grid((h, w))
    px = CT.sample(img, grid[0] * tx + ox, grid[1] * ty + oy)
    if px.shape[-1] < 4:
        return CT._u8(px[..., :3], 1.0)
    return CT._u8(px[..., :3], px[..., 3])


def eye_highlight(face: dict, side: int = 0, res: dict | None = None, *,
                  look: dict | None = None, size: tuple = (512, 512)
                  ) -> np.ndarray | None:
    """``toon_eye_lod0``'s two highlights as one RGBA map on **UV1**, or ``None``.

    The highlights are **not** part of the create pass: they are two more
    texture slots on ``cf_m_hitomi_00`` (``_overtex1`` = ``hlUpId`` with
    ``_overcolor1`` = ``hlUpColor``, ``_overtex2`` = ``hlDownId``), and the
    draw shader combines them like this — the tail of ``toon_eye_lod0``'s
    forward pass, register by register::

        r3 = _overtex2.Sample(uv2 * _overtex2_ST).a * _overcolor2.wxyz
        r4 = _overtex1.Sample(uv1 * _overtex1_ST).a * _overcolor1.wxyz
        r3 = max(r3, r4)                       // per channel
        w  = r3.x * _isHighLight
        o.rgb = lerp(base.rgb, r3.yzw, w)
        o.a   = max(base.a, w)

    Two things there that the previous version had wrong: the two highlights
    are combined by a per-channel **max**, not applied one after the other, and
    the colour lerped toward is ``alpha * _overcolorN.rgb``, not the colour
    itself — a half-transparent white sparkle pulls the iris to mid-grey, not
    to white.  The strength is likewise ``alpha * _overcolorN.a``.

    Baked here at the gaze-0 ST on UV1's own texel grid — the iris mesh's UV2
    is UV1 bit for bit, so one grid serves both slots — with ``rgb`` the
    ``alpha * _overcolorN.rgb`` the shader lerps toward (sRGB-encoded, so the
    sampler hands back linear) and ``a`` the weight before ``_isHighLight``.
    Both slots share ``Update``'s tiling and gaze offset and differ only by the
    constant ``hlUp``/``hlDownOffsetY``, which cancels out of any change of ST,
    so a single ``uv1 * k + b`` at draw time reproduces them both — the gaze
    and the shake (:func:`eye_highlight_xf`).  Before §7.403 they were baked
    into the iris map through a least-squares UV0 → UV1 fit (residual up to
    0.018 UV), moved with the iris, and never reached ``lit``.
    """
    tx, ty, ox, oy, up_dy, down_dy = eye_look_st(face, side, look)
    grid = CT._grid(tuple(size))
    best = None                                   # max(alpha * colour.wxyz)
    for cat, id_key, col_key, dy in (
            ("eye_hi_down", "hlDownId", "hlDownColor", down_dy),   # _overtex2
            ("eye_hi_up", "hlUpId", "hlUpColor", up_dy)):          # _overtex1
        hl = _list_texture(cat, face.get(id_key), res)
        if hl is None or hl.px.ndim != 3 or hl.px.shape[2] < 4:
            continue
        # A forward-pass layer: decoded map, engine-converted colour (§7.126).
        # Only ``.a`` (the ``.wxyz`` swizzle's first component) is coverage and
        # therefore unconverted either way.
        hm = CT.sample(hl, *CT.st(grid, (tx, ty, ox, oy + dy)), wrap=False)
        c = CT._rgba(face.get(col_key))
        lay = hm[..., 3:4] * np.asarray(
            (c[3], c[0], c[1], c[2]), np.float32)      # .wxyz, as the shader
        best = lay if best is None else np.maximum(best, lay)
    if best is None:
        return None
    out = np.empty(best.shape[:2] + (4,), np.uint8)
    out[..., :3] = CS.encode(np.clip(best[..., 1:4], 0.0, 1.0))
    out[..., 3] = np.clip(best[..., 0] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def apply_head(built, mapper, report, card, outfit: int = 0,
               res: dict | None = None) -> bool:
    try:
        head = head_assets(card, res)
        pieces, face_t, face_mc = load_head_prefab(head)
    except Exception as exc:  # noqa: BLE001
        report.missing.append(f"head: {exc}")
        return False
    pieces = _material_editor_character(pieces, card, report, "head")
    # Here rather than inside ``load_head_prefab`` for the reason ``_own_skin``
    # is: the loader is also the converter's path (§7.144 五).
    from ...mesh import parttex as PT
    if PT.enabled():
        face_t, face_mc = PT.face_sheets()
    if head.ident:
        report.replaced.append(
            f"頭：{head.source} id {head.ident} → {head.prefab}"
            f"（曲線 {head.shape_asset}）")

    wanted = [p for p in pieces if p.mesh and p.mesh.name in HEAD_MESHES
              and p.skin is not None and p.bones]
    # ``ChaControl.VisibleDoubleTooth`` is one line:
    # ``SetActiveIfDifferent(GetReferenceInfo(RefObjKey.ObjDoubleTooth),
    # fileFace.doubleTooth)``, and key 69 is ``cf_O_canine``.  The prefab always
    # ships the fangs, so loading every head mesh gives 994 of this install's
    # 1,386 cards a pair they asked not to have (§7.99).
    if not (card.face() if hasattr(card, "face") else {}).get("doubleTooth"):
        wanted = [p for p in wanted if p.mesh.name != "cf_O_canine"]
    if not wanted:
        report.missing.append("head: prefab 無可用 mesh")
        return False

    # strip procedural face
    for part in list(built.model.parts):
        if str(getattr(part, "group", "")) == "face" or part.name == "head":
            built.model.remove(part.name)
            report.replaced.append(part.name)

    body_part = built.model.get("body")
    skin_rgba = (body_part.material.base_color if body_part is not None
                 else (1.0, 0.9, 0.85, 1.0))
    face = card.face() if hasattr(card, "face") else {}
    if not isinstance(face, dict):
        face = {}
    from . import toon as TOON
    fg_eyes, fg_brow = TOON.foreground(face)

    total = 0
    n = 0
    reported: set[str] = set()          # one line per unresolved map, not four
    # A renderer's extra material slots draw the **same** geometry, so they get
    # the same ``Mesh`` object and not a copy of it.  A copy is what was here,
    # and the two halves then drifted apart the moment anything deformed the
    # first: the expression blend shapes are registered on material 0 only, so
    # the resting eye-close (0.08 of ``def_cl``, on every character) moved the
    # lash sheet 0.68 mm and left its lid-shadow twin behind — which put the
    # twin **0.46 mm under the skin** (it starts 0.17 mm outside) and the depth
    # test then dropped it.  Cost: ``EyelineShadowTex``, the eye-socket shade
    # every one of the 62 stock lash rows carries, never reached the screen at
    # all — Illusion's map no more than ours (§7.168).  One geometry, one
    # owner: registering the shapes twice on a shared mesh would apply them
    # twice instead (§7.131 三之三).
    # Matched by adjacency rather than by name: the loader emits a renderer's
    # slots consecutively, and a bundle can hold two renderers with the same
    # mesh name (``bo_head_50`` carries ``cf_O_eyeline`` twice).
    prev_name, prev_mesh = None, None
    for piece in wanted:
        PROG.tick(f"頭部：{piece.mesh.name}…")
        reuse = prev_mesh if (int(getattr(piece, "mat_slot", 0) or 0)
                              and piece.mesh.name == prev_name) else None
        if reuse is not None:
            mesh = reuse
        else:
            mesh = piece.mesh.copy()
            if not mapper.retarget(mesh, piece.bones, piece.skin, piece.binds):
                continue
            idx, wt = mapper.remap(piece.bones, piece.skin)
            mesh.bone_idx, mesh.bone_wt = idx, wt
            # Same as the body: the head's authored normals are the drawing.  On
            # ``cf_O_face`` they differ from the recomputed ones by a mean of 16
            # degrees and flip 93 outer-cheek vertices across ``_RampG``'s binary
            # step at N·L = 0.5 (§7.61).
            mesh.normalize_weights()
            prev_name, prev_mesh = piece.mesh.name, mesh

        name = mesh.name
        skinlike = name == "cf_O_face"
        if skinlike and body_part is not None:
            mat = body_part.material.copy()
            mat.base_color = skin_rgba
            # A copy brings the body's *texture bindings* along, and the two
            # materials do not share a single map: ``cf_m_face_00`` binds its
            # own 1024² ``_DetailMask`` (the body's is 512²) and **no**
            # ``_NormalMap`` at all — ``ChangeSettingFaceDetail`` assigns the
            # card's detail map to ``_NormalMapDetail`` and ``_LineMask``,
            # never to ``_NormalMap``.  Inherited, the face sampled
            # ``cf_body_00_base_n`` and ``cf_body_00_dm`` at face uv0, and both
            # halves of the specular split then landed on the wrong pixels: the
            # cheek slider covered 29,423 of them across the whole head instead
            # of the nose and cheeks, and the lip slider reached **3**, off at
            # the edge of the frame (§7.100 四).  Cleared here, then rebound
            # below from the head's own material; an unbound slot is the
            # shader's own default, which is what the renderer's "not bound"
            # path already reproduces.
            mat.normal_map = None
            mat.detail_mask = None
        else:
            mat = make_material(f"origin_head_{name}", cutout=name in FACE_CUTOUT)
        # §7.12 for the head too: every one of these materials states its own
        # toon settings, and the sclera's is a near-white blue
        # (cf_m_sirome_00 ``_shadowcolor`` 0.90/0.94/1.00).  Left at our grey
        # default the underside of the eyeball turned into a dark crescent.
        apply_shader(mat, piece.shader)
        # The head's own pieces need the same global ramp / line masks the
        # clothes get; tongue and teeth are main_item, the face is main_skin.
        from .attach import attach_maps
        attach_maps(mat, piece)
        # A head piece the card has no colour for still is not the loader's
        # grey default: Origin's colour chain starts at the material's own
        # ``_Color``, and for the pieces that bind ``cf_dummy_mc`` (a flat red
        # mask) the chain resolves to exactly that.  The tongue is the visible
        # case — (1.0, 0.699, 0.699), not the invented pink it used to carry.
        # The per-name branches below set their own base where they composite
        # a texture, so this only fills in the ones that do not.
        own = (piece.shader or {}).get("_Color")
        if isinstance(own, (list, tuple)) and len(own) >= 3:
            mat.base_color = (*(float(c) for c in own[:3]), 1.0)

        tex = piece.texture
        # MaterialEditor's ``_MainTex`` for this material, when the card set
        # one (``matedit.edit_pieces`` put it in ``piece.texture``).  Each
        # branch below says what ``_MainTex`` is for its material (§7.385).
        me_main = (CS.of(piece, "texture")
                   if (piece.shader or {}).get("_meMainTex") else None)
        me_cut = False
        # Which space these bytes are in is per branch below, so it is
        # carried alongside rather than guessed at the store (§7.126):
        # a create-pass composite is sRGB (Origin's RT), an asset map keeps
        # its own ``m_ColorSpace``.
        tex_cs = CS.of(piece, "texture")
        tex_srgb = bool(tex_cs.srgb) if tex_cs is not None else True
        if skinlike:
            # ``create_head``'s two map inputs, as Maps — the head prefab's own
            # pair is the fallback and carries its own ``m_ColorSpace`` too.
            baked, shade = face_texture(
                card,
                face_t if face_t is not None else CS.of(piece, "texture"),
                face_mc if face_mc is not None else CS.of(piece, "mask"),
                outfit=outfit, res=res, mesh=mesh, report=report)
            if baked is not None:
                tex = baked
                tex_srgb = True                 # create_head composite
                mat.base_color = (1.0, 1.0, 1.0, 1.0)
            if me_main is not None:
                # replaces the composite outright (§7.385); sRGB like every
                # MaterialEditor image
                tex, tex_srgb = me_main.px, True
                mat.base_color = (1.0, 1.0, 1.0, 1.0)
            if (piece.shader or {}).get("_meAlphaMask") and tex is not None:
                # ``cf_m_face_00`` binds no ``_AlphaMask`` (the property's
                # default is white, nothing is cut); a card that gives it one
                # cuts the face with ``main_skin``'s own rule (§7.386).
                fsh = piece.shader or {}
                tex = TEX.apply_alpha_mask(
                    tex, CS.sampled_bytes(CS.of(piece, "alpha_mask")),
                    float(fsh.get("_alpha_a", 1.0)), float(fsh.get("_alpha_b", 1.0)))
                me_cut = True
            mat.shade_mask = (TEX.store(shade, srgb=False)
                              if shade is not None else None)
            # The face's own ``_DetailMask``, resampled through its ``_ST``
            # onto uv0 — the same call the body makes, and needed for the same
            # reason: ``main_skin`` reads the map a second time at a
            # view-shifted uv, which no packed mask can hold
            # (:func:`create_tex.detail_map`).  ``_face_draw_props`` so that a
            # modded head brings its own.
            fprops = _face_draw_props(card, res)
            fdet = fprops.get("_DetailMask")
            dmap = CT.detail_map(CS.pixels(fdet), fprops)
            if dmap is not None and fdet is not None:
                key = TEX.store(dmap, srgb=fdet.srgb)
                if key is not None:
                    mat.detail_mask = key
            # ``ChangeSettingCheekGlossPower`` / ``…LipGlossPower``.
            power, nail = gloss_powers(card, head=True)
            if power is not None:
                mat.origin_specular = power
            if nail is not None:
                mat.spec_nail = nail
        if name in WHITE_MESHES:
            tex = (me_main.px if me_main is not None
                   else eye_white_texture(face, res))
            tex_srgb = True                     # create_eyewhite composite
            mat.base_color = (1.0, 1.0, 1.0, 1.0)
        elif name in IRIS_MESHES:
            # ``_eyeLook`` rides on the piece's material dict (see
            # ``PrefabLoader._eye_look``); an archive baked before that has to
            # go back to the prefab for it, because the waits it carries are
            # not the same on every head.
            side = 1 if "_R" in name else 0
            look = (piece.shader or {}).get("_eyeLook") \
                or eye_look_settings(head, side)
            if look is None and "eyeLook" not in reported:
                reported.add("eyeLook")
                report.missing.append(
                    f"eye: {head.prefab} 沒有 EyeLookMaterialControll，"
                    "虹膜 ST 無法還原")
            built.eyes_open_max = eyes_open_max(card)
            # What the gaze needs at run time, kept where the driver can reach
            # it: the ST is the card's pupil sliders through *this eye's*
            # component, and the lid correction is the head-wide half of
            # ``FaceBlendShape`` that rides in the same dict (see
            # ``prefab.build_skinned``).
            built.eye_face = face
            built.eye_look[side] = look
            # The ST the map below is baked at, kept so the runtime gaze can
            # difference against it without recomputing it every frame.
            built.eye_st[side] = eye_look_st(face, side, look)[:4]
            period = (look or {}).get("YureTime")
            if period is not None:
                built.eye_shake[side] = HighlightShake(float(period))
            elif look is not None and "YureTime" not in reported:
                reported.add("YureTime")
                report.missing.append(
                    f"eye: {head.prefab} 的 bake 沒有 YureTime，"
                    "H 的眼睛顫動（yure）不會動（重轉 base/head）")
            tex = (eye_texture_replaced(me_main, face, side, look)
                   if me_main is not None else
                   eye_texture(face, side, res, look=look, report=report,
                               skin=skin_overlays(card, outfit, report)))
            # The catchlights are their own map on UV1, drawn by the shader
            # rather than baked into the iris (§7.403): they add to ``lit`` and
            # they shake on their own when an H face row asks (``yure``).
            hl = eye_highlight(face, side, res, look=look,
                               size=(tex.shape[:2] if tex is not None
                                     else (512, 512)))
            if hl is not None:
                mat.highlight = TEX.store(hl, srgb=True)
            mat.highlight_on = _num((piece.shader or {}).get("_isHighLight"), 1.0)
            tex_srgb = True                     # create_eye composite
            if (tex is None and me_main is None and name not in reported
                    and not _list_says_none("eye", _pupil(face, 0).get("id"))):
                reported.add(name)
                report.missing.append(
                    f"eye: mt_eye#{_pupil(face, 0).get('id')} 找不到虹膜貼圖")
            mat.base_color = (1.0, 1.0, 1.0, 1.0)
        elif name == "cf_O_eyeline" and getattr(piece, "mat_slot", 0) == 1:
            # ``cf_m_eyeline_kage``: the same lash geometry drawn again as the
            # shadow the upper lid casts on the skin.  ``ChangeSettingEyelineUp``
            # feeds it ``EyelineShadowTex`` off the same row, and
            # ``UpdateEyelineShadowColor`` tints it with **skinMainColor** —
            # it is a shadow on skin, not another lash.
            # A draw material, not a create pass, and it is a plain
            # map × colour — the multiply is the same number in either space
            # only if both sides are, so the map keeps its own flag and the
            # colour is the card's byte (``_multiply`` works on bytes).
            # **The generator answers before the base row** (``bake_category``).
            # This map is named by the eyeline row rather than being a category
            # of its own, so it never went through ``bake.read_texture`` and
            # the generated-sheet path had no way to answer it — a build with
            # no converted textures drew the lashes and no eye socket shade at
            # all.  ``facetex.eyeline_shadow`` is measured off all 62 stock rows
            # (§7.165).  A mod the card names answers before either (§7.356).
            # a MaterialEditor ``_MainTex`` is the map; the skin tint below is
            # still the shader's ``_Color`` (§7.385)
            shadow_map = me_main or _row_texture(
                "eyeline_up", face.get("eyelineUpId"), "MainAB",
                "EyelineShadowTex", res, bake_category="eyeline_shadow",
                missing=report.missing, add=eyeline_head_suffix(face, res))
            tex = CS.pixels(shadow_map)
            tex_srgb = bool(shadow_map.srgb) if shadow_map is not None else True
            body_rgb = (card.body() or {}).get("skinMainColor") \
                if hasattr(card, "body") else None
            tex = _multiply(tex, _rgb(body_rgb, (1.0, 1.0, 1.0)))
            mat.base_color = (1.0, 1.0, 1.0, 1.0)
        else:
            spec = FACE_FEATURES.get(name)
            if spec is not None:
                # the list row's map is the ``_MainTex`` a MaterialEditor
                # image replaces; the ``_Color`` multiply stays (§7.385)
                add = (eyeline_head_suffix(face, res)
                       if spec[0] in HEAD_VARIANT_LISTS else "")
                found = me_main or _list_texture(spec[0], face.get(spec[1]),
                                                 res, add=add)
                if (found is None and name not in reported
                        and not _list_says_none(spec[0], face.get(spec[1]))):
                    reported.add(name)
                    report.missing.append(
                        f"face: {spec[0]}#{face.get(spec[1])} 找不到貼圖")
                if found is not None:
                    # These maps carry their shading in RGB and their shape in
                    # alpha, and the material multiplies by a plain ``_Color``
                    # (ChangeSettingEyelineColor / …EyebrowColor).  The old
                    # luminance ``recolor`` re-normalised against the map's own
                    # peak, which turned a mid-grey lash line into the card
                    # colour at full brightness.  The nose runs a different
                    # shader and needs its own transcription.
                    # Also draw materials — same space as the eyelid shadow.
                    if name == "cf_O_noseline":
                        tex = _nose_line(found.px,
                                         (piece.shader or {}).get("_Color"))
                    else:
                        tex = _multiply(found.px,
                                        _rgb(face.get(spec[2]), (1.0,) * 3)
                                        if spec[2] else None)
                    tex_srgb = bool(found.srgb)
                    mat.base_color = (1.0, 1.0, 1.0, 1.0)
        if (piece.shader or {}).get("_meAlphaMask") and not skinlike:
            report.missing.append(
                f"MaterialEditor: {name} 的 AlphaMask 沒有套（只接了臉的）")
        if name in FACE_CUTOUT and tex is None:
            continue
        # The tongue used to be forced to a flat invented pink with no
        # outline, because the albedo search kept handing it ``cf_tang_md``.
        # It is flat in the game too, but not that colour and not by accident:
        # ``cf_m_tang`` binds **no** ``_MainTex`` (so the shader samples white)
        # and its colour chain resolves to ``_Color`` = (1.0, 0.699, 0.699)
        # through ``cf_dummy_mc``.  With ``_main_texture`` no longer returning
        # a detail map (see ``prefab``), the ordinary path produces exactly
        # that, and ``main_item``'s outline pass applies like anything else.
        # Cut-out or blend is the *shader's* business, not the material's:
        # cf_m_hitomi_00 carries ``_alphaTest 0.5`` and toon_eye_lod0 has no
        # discard instruction at all, so the iris blends.  ``_Mode`` / ``_ZWrite``
        # then say whether it also wants the blended pass (§7.23).
        sh = piece.shader or {}
        if sh.get("_shader"):
            mat.alpha_test = TEX.shader_clip(sh)
            mat.blend = TEX.shader_blends(sh)
        if me_cut:
            # ``main_skin``'s forward base clips at the literal 0.5 once the
            # mask is bound (§7.386), whatever the unbound material said
            mat.alpha_test = TEX.ALPHA_CLIP
        if (name in FACE_CUTOUT or name in WHITE_MESHES) and not sh.get("_shader"):
            # Normally ``apply_shader`` decides this from the pass list — the
            # eye / sclera / nose-line shaders have no Cull Front pass, so no
            # hull.  This only covers a piece whose shader could not be read
            # at all, where an inverted hull would draw a dark ring across the
            # bottom of the sclera.
            mat.outline_width = 0.0
        if tex is not None:
            key = TEX.store(tex, srgb=tex_srgb)
            if key is not None:
                mat.texture = key
                if name == "cf_O_face":
                    mat.base_color = skin_rgba

        slot = getattr(piece, "mat_slot", 0)
        suffix = "" if slot == 0 else f"_m{slot}"
        layer = FACE_LAYER.get(name, 2 + n) - (1 if slot else 0)
        part = built.model.add_mesh(f"origin_head_{name}{suffix}", mesh, mat,
                                    group="face", layer=layer)
        # ``name`` here *is* the renderer's GameObject name, which is what
        # MaterialEditor's ``ObjectType.Character`` rows address (§7.138).
        part.origin_renderer = str(name)
        # Only material slot 0 — that is the single ``QueueData.id`` each of the
        # ten components carries, so the lid shadow in slot 1 does not follow
        # the lashes forward.  Unless the install runs IllusionFixes'
        # HairShadows: its ``SetRenderQueue_Custom.Awake`` postfix adds an
        # ``id 1`` entry (2473 / 2902) to ``cf_O_eyeline`` alone — not
        # ``_low`` — and the lid shadow goes forward with the lashes
        # (:func:`foreground_slot`).
        if foreground_slot(name, slot):
            part.foreground = (fg_eyes if name in FOREGROUND_EYES else
                               fg_brow if name in FOREGROUND_EYEBROW else False)
        # Illusion's expressions are blend shapes on these very meshes, not
        # bone poses; carry them through so ExpressionController can drive
        # them (see figure.face_morph).  The extra material slots share that
        # geometry rather than copying it, so only the slot that owns it
        # registers the shapes — a second registration on the same mesh would
        # apply every weight twice (§7.131 三之三).
        if piece.shapes and reuse is None:
            part.origin_shapes = piece.shapes
        if name in TEARS_MESHES:
            state = (piece.shader or {}).get("_glassRamp")
            if state not in ("bound", "none") and "glassRamp" not in reported:
                reported.add("glassRamp")
                report.missing.append(
                    f"head: {name} 的 _GlassRamp "
                    + ("引用解不到" if state == "unresolved" else
                       "沒有進封存（bake 早於 §7.413，重轉 base/head）"))
        total += mesh.tri_count
        n += 1

    if n:
        from .report import Attached
        report.attached.append(
            Attached(f"p_cf_head_00 ({n} meshes)", "head", "base", total))
        # the card's own ``tearsLv`` (``ChaFileStatus``, 0 on every local card)
        # until an H face row sets another (``FaceListCtrl.SetFace``)
        status = (getattr(card, "sections", None) or {}).get("Status") or {}
        try:
            built.card_tears = int(status.get("tearsLv") or 0)
        except (TypeError, ValueError):
            built.card_tears = 0
        set_tears(built, built.card_tears)
        return True
    report.missing.append("head: 全部 mesh 失敗")
    return False
