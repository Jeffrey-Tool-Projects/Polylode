"""Illusion's *global* toon settings, and the two maps they pull in.

Every Origin character shader (``main_skin`` / ``main_opaque`` / ``main_hair`` /
``main_item``) reads four globals that no material carries.  They are not
guesses: ``Character.UpdateGlobalShader`` writes all four from
``Config.EtcData``, whose constructor supplies the defaults::

    EtceteraSystem..ctor      rampId = 1   shadowDepth = 0.26
                              lineDepth = 1.0   lineWidth = 0.307

    ChaShader.ChangeAmbientShaodwColor(a) -> _ambientshadowG =
                                             Color(0.019, 0.027, 0.039, a)
    ChaShader.ChangeLineColor(d)          -> _LineColorG =
                                             Color(0.5, 0.5, 0.5, 1 - d)
    ChaShader.ChangeLineWidth(w)          -> _linewidthG = w
    ChaShader.ChangeRampTexture(t)        -> _RampG = GetListInfo(432, rampId)

``_LineColorG.a`` therefore starts at **0**, and every outline pass ends with a
hard-light against that alpha — with alpha 0 the multiply branch wins and the
whole thing collapses to black.  That is why the game's outline is black even
though the colour channels say mid-grey.

``_ambientshadowG`` is the *base* the shadow tint is built on.  With alpha 0.26
the overlay in the shader resolves to roughly (0.274, 0.280, 0.289) — a dark
blue-grey — which each shader then hard-lights with either its own
``_ShadowColor`` (hair) or the albedo's own hue (skin and clothing).

Studio replaces the ambient value with ``SceneInfo.ambientShadow`` (#808080 by
default) and its own alpha; that path is a scene setting, not a character one,
so it is not modelled here.
"""
from __future__ import annotations

import numpy as np

from ...core import swallowed


# ``EtceteraSystem.Init``'s own values, straight off the IL.  These are where
# the game *starts*; they are not where it necessarily *is* (see :func:`etc`).
#
# ``ForegroundEyebrow`` / ``ForegroundEyes`` are the same kind of thing as
# ``rampId`` — a graphics setting, written to the same ``<Etc>`` block — and
# ``EtceteraSystem.Init`` starts both at **false** (``ldc.i4.0``, right after
# ``lineWidth``).  They are read here rather than assumed for the reason
# CLAUDE.md 七 gives: "this is the game's default" is not "this is the value on
# this machine".  On this install both are ``False``.
#
# ``rampGUID`` is not the game's: Sideloader's ``Hooks.XMLWritePostfix`` adds it
# to the same block when the chosen ramp is a mod's, and ``XMLReadPostfix`` reads
# it back (:func:`ramp_setting`).  Empty is "no mod named", which is every
# install that never picked a modded ramp — this one included.
ETC_DEFAULTS = {"rampId": 1, "shadowDepth": 0.26,
                "lineDepth": 1.0, "lineWidth": 0.307,
                "ForegroundEyebrow": False, "ForegroundEyes": False,
                "rampGUID": "",
                # グラフィック's camera effects, ``true`` in ``EtceteraSystem.Init``
                # (:func:`camera_effects`)
                "DepthOfField": True, "Fog": True, "Bloom": True,
                "Vignette": True, "AmplifyOcclus": True, "SunShafts": True}

#: The camera effects a player can switch off (``CameraEffectorConfig``).
CAMERA_EFFECTS = ("DepthOfField", "Fog", "Bloom", "Vignette", "AmplifyOcclus", "SunShafts")

# Sideloader numbers the rows it registers from here up, so an id at or above
# it is a LocalSlot handed out by *that* session and means nothing after a
# restart (``UniversalAutoResolver.BaseSlotID``).
LOCAL_SLOT_BASE = 100_000_000

# The four globals, as this machine's install has them.  Kept as module-level
# names for the readers that want one value; :func:`etc` is the source.
AMBIENT_SHADOW_RGB = (0.019, 0.027, 0.039)


def etc() -> dict:
    """``Config.EtcData`` — read from the install, not assumed.

    ``EtceteraSystem.Init`` supplies ``rampId 1 / shadowDepth 0.26 /
    lineDepth 1.0 / lineWidth 0.307``, and for three of the four that is also
    what the game ends up running.  ``rampId`` is different: it is a **player
    setting** (グラフィック → 影), the list behind it holds seven ramps whose
    names are the menu entries, and this install's ``system.xml`` says **2**
    (``gt_ramp_02``, 深いなめらか影) rather than 1 (``gt_ramp_01``,
    深いくっきり影).

    That is not a detail.  Ramp 1 is a **binary step at N·L = 0.5**, so it puts
    a hard two-tone edge round every limb; ramp 2 is a smooth curve that only
    starts to fall below N·L ≈ 0.75.  Fitted against ``reference/``'s own frame
    — a scan across the thigh at y=540, which is a cylinder and therefore a
    clean N·L sweep — ramp 2 reproduces the pixels to a mean of **1.85 steps**
    and every other ramp in the list is 10 to 17 steps out.  The photometry and
    the config file agree without either being told about the other (§7.61).

    Falling back to ``ETC_DEFAULTS`` when there is no install or no file is the
    same policy as everywhere else here: read the user's own install, and where
    there is nothing to read, use the value the game itself starts from.
    """
    hit = _ETC.get("v")
    if hit is None:
        hit = _ETC["v"] = _read_etc()
    return hit


_ETC: dict = {}


def camera_effects() -> dict:
    """``{effect: on}`` for the camera effects, as the player's config has them.

    ``CameraEffectorConfig.Update`` runs every frame on a camera that carries
    it: for each effect whose ``_useX`` is set, ``effector.useX = EtcData.X``,
    which switches that image effect's ``enabled`` (``_dof`` / ``_fog`` /
    ``_bloom`` / ``_vignette`` / ``_amplifyOcclus`` / ``_sunShafts``).  Turning
    one off under グラフィック turns it off on every such camera (§7.554).
    """
    e = etc()
    return {k: bool(e.get(k, True)) for k in CAMERA_EFFECTS}


def _read_etc() -> dict:
    """``system.xml``'s ``<Etc>`` block, or ``EtceteraSystem.Init``'s values.

    The cached copy first (runtime reads the project cache only, §four), then a
    live install if one is still configured, so a machine that has the game but
    has not synced yet is not stuck on the wrong ramp.
    """
    import xml.etree.ElementTree as ET
    from ..install import etc_config_path, source_install

    out = dict(ETC_DEFAULTS)
    paths = [etc_config_path()]
    src = source_install()
    if getattr(src, "valid", False):
        paths.append(src.root / "UserData" / "config" / "system.xml")
    for path in paths:
        if not path.is_file():
            continue
        try:
            node = ET.parse(path).getroot().find("Etc")
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.figure.toon._read_etc: ET.parse", exc)
            continue
        if node is None:
            continue
        for key, default in ETC_DEFAULTS.items():
            el = node.find(key)
            if el is None or not (el.text or "").strip():
                continue
            text = el.text.strip()
            # ``bool`` before ``int``: it *is* an int in Python, so the numeric
            # branch would take "False" to ``int("False")`` -> ValueError and
            # silently leave the default in place.  That failure looks exactly
            # like "the file does not have this setting".
            try:
                if isinstance(default, str):
                    out[key] = text
                elif isinstance(default, bool):
                    out[key] = text.lower() == "true"
                elif isinstance(default, int):
                    out[key] = int(text)
                else:
                    out[key] = float(text)
            except ValueError:
                pass
        out["_source"] = str(path)
        break
    return out


def ambient_shadow() -> tuple:
    """``ChaShader.ChangeAmbientShaodwColor(EtceteraSystem.shadowDepth)``."""
    return (*AMBIENT_SHADOW_RGB, float(etc()["shadowDepth"]))


def line_color_g() -> tuple:
    """``ChaShader.ChangeLineColor(lineDepth)`` → ``Color(.5,.5,.5,1-depth)``."""
    return (0.5, 0.5, 0.5, 1.0 - float(etc()["lineDepth"]))


def line_width_g() -> float:
    """``ChaShader.ChangeLineWidth(EtceteraSystem.lineWidth)``."""
    return float(etc()["lineWidth"])


def ramp_id() -> int:
    """``EtceteraSystem.rampId`` → ``ChaListDefine.CategoryNo`` 432 (mt_ramp).

    The id as ``system.xml`` stores it.  Which row that names is
    :func:`ramp_setting`'s question, because Sideloader rewrites it on load.
    """
    return int(etc()["rampId"])


def ramp_setting():
    """``(base_id, mod or None)`` — the ramp after Sideloader's rewrite.

    Transcribed from ``Sideloader.Hooks.XMLReadPostfix``, which runs right after
    the game reads ``<Etc>``::

        if (etc.rampId >= 100000000) etc.rampId = 1;
        else if (!IsNullOrWhiteSpace(rampGUID)) {
            var info = TryGetResolutionInfo(etc.rampId, "Ramp", rampGUID);
            etc.rampId = info == null ? 1 : info.LocalSlot;
        }

    So a modded ramp is named by the **GUID**, not by the id alone — the id is
    the mod's own slot and collides with base rows (5 of the 36 ramps local
    zipmods register number themselves 1 or 2, and 2 is this install's base
    ramp).  And a named ramp
    whose mod is gone does **not** fall to the base row of the same number: the
    game resets it to 1, 深いくっきり影.  The second element is the
    ``sideloader.ModRef`` that answers; ``None`` means the base list answers
    ``base_id``.
    """
    cfg = etc()
    rid = ramp_id()
    if rid >= LOCAL_SLOT_BASE:
        return 1, None
    guid = str(cfg.get("rampGUID") or "").strip()
    if not guid:
        return rid, None
    from .. import sideloader as SL
    mref = SL.index().resolve(guid, "ramp", rid)
    return (1, None) if mref is None else (rid, mref)


def foreground(face: dict | None) -> tuple[bool, bool]:
    """``ChaControl.SetForegroundEyesAndEyebrow`` — ``(eyes, eyebrow)``.

    Transcribed instruction for instruction; the card's field is a **tri-state
    override**, not a bool::

        bool eyes = true;
        if (EtcData != null) eyes = EtcData.ForegroundEyes;   // the player's setting
        if      (fileFace.foregroundEyes == 1) eyes = false;  // card forces off
        else if (fileFace.foregroundEyes == 2) eyes = true;   // card forces on
        // eyebrow: the same block again
        foregroundCtrl.SetForeground(eyes, eyebrow);

    So **0 is not "off"** — it is "whatever the player set", and on this install
    that is ``False``.  Reading 0 as off and 1 as on would put 429 of the local
    1,386 cards in the wrong state and, worse, agree with the truth on the other
    957 — the shape of a bug that looks like it works.

    Counts on this install (1,386 cards, 0 unreadable): eyebrow 294 / 429 /
    **663**, eyes 1,050 / 213 / **123** for values 0 / 1 / 2.  With the setting
    off, only value 2 changes anything, so this is 47.8% of cards for the
    eyebrow and 8.9% for the eyes (§7.103).
    """
    cfg = etc()
    eyes = bool(cfg.get("ForegroundEyes", False))
    brow = bool(cfg.get("ForegroundEyebrow", False))
    f = face or {}
    for key, name in ((("foregroundEyes"), "eyes"),
                      (("foregroundEyebrow"), "brow")):
        try:
            v = int(f.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if v not in (1, 2):
            continue
        if name == "eyes":
            eyes = (v == 2)
        else:
            brow = (v == 2)
    return eyes, brow

# Which of Origin's character shaders a material declares.  The forward passes
# differ in two places only — where the shadow tint comes from and how the
# per-texel shading term combines with the light ramp — so the renderer keys
# both off this rather than off a material name.
FAMILY_NONE = 0        # this project's own procedural materials
FAMILY_SKIN = 1        # main_skin: tint from the albedo's hue, shade multiplies
FAMILY_CLOTH = 2       # main_opaque / main_alpha / main_item: same as skin
FAMILY_HAIR = 3        # main_hair: tint from _ShadowColor, shade is a max()
FAMILY_EMBLEM = 4      # main_emblem: same shape, its own constants (see below)
FAMILY_EYE = 5         # toon_eye_lod0: same tint, **no ramp** (§7.345)
FAMILY_EYEW = 6        # toon_eyew_lod0: same tint, **with** the ramp (§7.359)
FAMILY_UNLIT = 7       # toon_nose_lod0: no lighting at all (§7.366)
FAMILY_GLASS = 8       # toon_glasses_lod0: lenses, tears, rendAlpha (§7.413)

_FAMILIES = (
    ("main_skin", FAMILY_SKIN),
    ("main_hair", FAMILY_HAIR),
    ("main_opaque", FAMILY_CLOTH),
    # ``main_opaque2`` (chara/co_bra_10, the body suits) is ``main_opaque`` with
    # one change: ``_AlphaMask`` is read at ``lerp(uv0, uv1, _AlphaMaskuv)``.
    # Every other instruction of every fragment program is the same (§7.367
    # diffed them), so the shading family is the same.  The prefix rule never
    # claimed it — there is no underscore after the stem.  Its sibling
    # ``main_opaque_low2`` has not been diffed and nothing baked uses it.
    ("main_opaque2", FAMILY_CLOTH),
    ("main_alpha", FAMILY_CLOTH),
    ("main_item", FAMILY_CLOTH),
    # The chest badge.  Same shape as main_opaque's forward base — rebuild the
    # albedo's hue, hard-light it against an ambient colour, ``lerp(shadow,
    # albedo, _RampG)``, then ``* max(_LightColor0 * 0.6 + 0.4, ambient)`` —
    # with four differences, all of them read off its own bytecode:
    #
    #   * no ``_ShadowColor`` multiply.  The material declares ``_shadowcolor``
    #     and the program never samples it.
    #   * the hue rebuild keeps **half** the saturation, not 0.66, and the hard
    #     light's upper layer is scaled by 0.8, not 0.9 (the branch tests
    #     ``0.625 < t`` and the multiply is ``1.6 * base * t``).
    #   * the ambient is a *shader-local* ``_ambientshadow``, **not** the global
    #     ``_ambientshadowG``: nothing in ``Assembly-CSharp`` ever writes that
    #     name and no material carries it, so it is 0 and the badge's shadow
    #     side is darker than a garment's.
    #   * there is no detail/line mask, no rim, no specular and no back-face
    #     term; the program declares two samplers and reads no ``is_front_face``.
    ("main_emblem", FAMILY_EMBLEM),
    # The iris.  It **is** shaded -- §7.158 read the render as "the iris is
    # toon shaded" and was right about that -- and until §7.345 it fell through
    # to FAMILY_NONE, i.e. to this project's own toon, because nothing had
    # disassembled it.  Its fragment program (``bo_head_00``) says:
    #
    #   tint = hardLight(hardLight(_ambientshadowG.rgb, 0.5*_asG.a + 0.5),
    #                    _shadowcolor)                     <- FAMILY_HAIR's tint
    #   lit  = saturate(max(N·L, 0)                        <- **no _RampG**
    #                   + _isHighLight * hi + _overtex2.a * _exppower)
    #   out  = saturate(lerp(tint*albedo, albedo, lit)
    #                   * max(_LightColor0*0.6 + 0.4, _ambientshadowG.rgb))
    #
    # The one difference from every other family is the middle line: the
    # program **never samples ``_RampG``** (``origin_shader_dump --shader
    # toon_eye_lod0`` finds no reference to it in any of the six variants), so
    # the terminator that makes a cheek read as toon-shaded does not exist on
    # an iris -- it is a plain Lambert against the character light, and on a
    # sphere facing the camera that is ``lit ~ 1`` over the whole visible disc.
    #
    ("toon_eye_lod0", FAMILY_EYE),
    # The sclera (§7.359).  A different program from the iris beside it, and
    # the difference is exactly the line the iris lacks -- it **does** sample
    # ``_RampG``, with the raw N·L (no ``max``; the clamp is the wrap mode):
    #
    #   c    = _MainTex(uv);  clip(c.a - 0.5)
    #   tint = hardLight(hardLight(_asG.rgb, 0.5*_asG.a + 0.5), _shadowcolor)
    #   out  = lerp(tint * c*_Color, c*_Color, _RampG(N·L).r)
    #          * max(_LightColor0*0.6 + 0.4, _ambientshadowG.rgb)
    #
    # No mask, no rim, no specular, no view ramp, and -- like the iris -- no
    # ``SV_IsFrontFace`` input, so no back-face 0.66 either.  The stock
    # ``cf_m_sirome_00`` carries ``_Color`` white and ``_shadowcolor``
    # (0.903, 0.942, 1.0).  Before this it fell to FAMILY_NONE, this
    # project's own half-lambert toon.
    ("toon_eyew_lod0", FAMILY_EYEW),
    # The noseline (§7.366).  Its whole forward fragment, and there is only
    # one variant of it::
    #
    #   m     = _MainTex(uv * _MainTex_ST)
    #   o.rgb = _Color.rgb * (1 - m.r) + m.g
    #   o.a   = m.a * _Color.a
    #
    # **No lighting**: no N·L, no ramp, no tint, no ``_LightColor0``, no
    # ``_ambientshadowG``, and the input signature is POSITION + TEXCOORD0 (so
    # no ``SV_IsFrontFace`` either).  The declared property table is three
    # entries (``_Color``, ``_MainTex``, ``_Cutoff``) and the pass is
    # ``Blend SrcAlpha/OneMinusSrcAlpha, ZWrite off, Cull Back`` with **no**
    # clip instruction — an alpha-blended decal.  ``body_head._nose_line``
    # already composes the rgb above into the map, so the family's only job is
    # to stop the renderer shading it: before this it fell to FAMILY_NONE and
    # took our half-lambert, our rim (the head material's default 0.20/0.22,
    # the same leak §7.359 found on the iris) and the ``max(_LightColor0*0.6 +
    # 0.4, …)`` light term, whose floor alone is 0.4.
    ("toon_nose_lod0", FAMILY_UNLIT),
    # Lenses, gems, the accessory ``rendAlpha`` renderers and the head's tears
    # (§7.366 五 transcribed it, §7.413 draws it).  Not the ``main_*`` shape at
    # all: a glass ramp sampled by ``|R·H| * (N.x+N.y+N.z)``, a rim and a hard
    # specular sorted onto ``_Color``/``_Color4``, the albedo tinted by an
    # ``|N·L|`` lerp instead of ``_RampG``, and the whole lit by
    # ``max(N·L,0) * _LightColor0 + ambient`` — the one Origin character shader
    # that reads the scene's ambient (Unity's SH: in the game it is the active
    # scene's, which in life and H is the map loaded as the base scene; ours is
    # the scene's own ``LightSettings.ambient``).  Premultiplied output.
    ("toon_glasses_lod0", FAMILY_GLASS),
)

# Stems whose ``_`` suffixes are **not** the same shader.  ``main_emblem_clothes``
# (on ``co_bra_00``) declares ``_Color``, ``_AlphaMask``, ``_alphamaskuv1`` and
# ``_alpha_a/b`` — none of which ``main_emblem`` has — so letting the prefix
# rule claim it would assert a transcription nobody has read.  It stays
# FAMILY_NONE, which is where it was before the emblem family existed.
EXACT_ONLY = frozenset(("main_emblem", "toon_eye_lod0", "toon_eyew_lod0",
                        # ``toon_glasses_lod0_2`` is a mod's shader nobody read
                        "toon_glasses_lod0"))
# ``toon_eye_lod0`` is exact for the same reason: ``toon_eyew_lod0``
# starts with neither prefix, but ``toon_eye_lod0_low`` would be
# claimed by the ``_low`` rule and no such shader has been read.

# Which shaders draw an inverted hull.  The material's ``_outline`` /
# ``_linewidth`` floats look like the switch but no program in any of these
# shaders reads them — they are dead Shader Forge properties, like
# ``_ColorContrast``.  What decides it is whether the shader itself ships an
# OUTLINE pass, which is a fact about the asset, so this list is the whole
# answer rather than a sample: ``origin_shader_dump.py --passes`` over **every**
# bundle in ``assets_cache`` (710 of them, 53 distinct shaders) gives an
# OUTLINE pass on exactly these nine and on nothing else — not on
# ``main_emblem``, ``main_texture``, ``main_color``, any ``*_low``, any
# ``toon_*`` or any ``create_*``.
#
# ``main_emblem`` is the one that mattered: it is a ``main_*`` shader, so the
# old "every main_* has one" rule put a black hull round the chest badge.
#
# Shipping the pass is not the whole story for ``main_item*``: its width has a
# per-material factor as well, which can be 0.  See :func:`line_width_s`.
OUTLINE_SHADERS = frozenset((
    "main_skin", "main_opaque", "main_opaque2", "main_alpha",
    "main_item", "main_item_emission", "main_item_studio",
    "main_hair", "main_hair_front",
))


# The one place in the whole character pipeline that uses the stencil buffer,
# and the whole of it: "eyebrows and eyes drawn through the fringe".
#
#     toon_eyew_lod0        FORWARD           Ref 2  Comp Always    Pass Replace
#     toon_eye_lod0         FORWARD           Ref 2  Comp Always    Pass Replace
#     toon_textureanimation FORWARD           Ref 2  Comp Always    Pass Replace
#     main_hair_front       FORWARD, OUTLINE  Ref 2  Comp NotEqual  Pass Keep
#
# Read off the serialised ``m_State`` of **every** shader in ``assets_cache``
# (712 bundles, 51 distinct shaders): those four are the only ones whose stencil
# state is not ``Comp Always / Keep``, so this table is the complete answer
# rather than a sample — the same standard :data:`OUTLINE_SHADERS` is held to.
#
# The three writers are exactly the meshes ``ChaForegroundComponent.Initialize``
# collects: ``cf_m_mayuge_00`` / ``cf_m_sirome_00`` / the two eyeline materials
# (``toon_eyew_lod0``), the iris (``toon_eye_lod0``) and the three gag eyes
# (``toon_textureanimation``, ``cf_m_gageye_0*``).  So the cut-out is not a
# property of "the eyebrow": it is a property of the sheets, and the *queue*
# decides whether it happens — see :func:`foreground`.
STENCIL_REF = 2
STENCIL_WRITERS = frozenset(("toon_eyew_lod0", "toon_eye_lod0",
                             "toon_textureanimation"))
STENCIL_MASKED = frozenset(("main_hair_front",))

# The stamp has the shape of the **ink**, not of the quad, because a discarded
# fragment writes no stencil.  Two of the three writers discard and one does
# not, and that is a fact about the program rather than about the material:
# disassembling every fragment sub-program of all 51 shaders in
# ``assets_cache`` gives ``toon_eyew_lod0`` 4 of 4 discarding,
# ``toon_textureanimation`` 4 of 4, ``toon_eye_lod0`` **0 of 2**.  The
# threshold is the flat 0.5 every Illusion clip uses
# (``add r, a, -0.5 / lt / discard_nz``), with no keyword in front of it —
# these shaders declare no ``_ALPHA_*`` keyword at all.
#
# It is used **only for the stencil-writing draw**.  The colour pass takes its
# threshold from ``textures.shader_clip``, which gates on the material's
# keywords and therefore answers 0 for these sheets; whether that is right for
# them is a separate question with its own callers (§7.90), and it is written
# up in PROGRESS.md rather than changed here on the way past.
STENCIL_CLIP = {"toon_eyew_lod0": 0.5, "toon_textureanimation": 0.5}


def stencil_write(shader: dict | None) -> int:
    """Stencil value this shader stamps where it draws (0 = it writes none)."""
    leaf = str((shader or {}).get("_shader") or "").rsplit("/", 1)[-1]
    return STENCIL_REF if leaf in STENCIL_WRITERS else 0


def stencil_reject(shader: dict | None) -> int:
    """Stencil value this shader refuses to draw over (0 = it tests none)."""
    leaf = str((shader or {}).get("_shader") or "").rsplit("/", 1)[-1]
    return STENCIL_REF if leaf in STENCIL_MASKED else 0


def stencil_clip(shader: dict | None) -> float:
    """Alpha below which this writer's program discards — and stamps nothing."""
    leaf = str((shader or {}).get("_shader") or "").rsplit("/", 1)[-1]
    return float(STENCIL_CLIP.get(leaf, 0.0))


def family(shader: dict | None) -> int:
    """Which Origin shader family a piece's material belongs to."""
    name = str((shader or {}).get("_shader") or "")
    if not name:
        return FAMILY_NONE
    leaf = name.rsplit("/", 1)[-1]
    # ``_low`` variants have no outline pass but shade identically
    for stem, fam in _FAMILIES:
        if leaf == stem or leaf == stem + "_low":
            return fam
        if stem not in EXACT_ONLY and leaf.startswith(stem + "_"):
            return fam
    return FAMILY_NONE


def is_emblem(shader: dict | None) -> bool:
    """``main_emblem`` — the chest badge, shaded but not tinted.

    Its forward base samples exactly two textures (``_MainTex`` and the global
    ``_RampG``) and reads no colour, no mask, no rim and no specular, so the
    badge's albedo is its own map and every per-material tint on it is a
    property the shader never looks at.
    """
    return str((shader or {}).get("_shader") or "").rsplit(
        "/", 1)[-1] == "main_emblem"


# ``round(pow(a, 0.3)) < 0.5 -> discard`` in main_emblem's forward base, i.e.
# a < 0.5 ** (1 / 0.3).  Every other character shader clips at a flat 0.5.
EMBLEM_CLIP = 0.5 ** (1.0 / 0.3)


def has_outline(shader: dict | None) -> bool:
    """Does this material's shader ship an OUTLINE pass?"""
    name = str((shader or {}).get("_shader") or "")
    return name.rsplit("/", 1)[-1] in OUTLINE_SHADERS


# The one per-material term in any of the outline widths.  Everything else in
# the extrusion is global (``_linewidthG``), per-vertex (the camera distance) or
# per-texel (``1 - _DetailMask.b``), which is why the width was treated as the
# same on every material for fifty sections -- and on skin, clothing and hair it
# is.  ``main_item``'s OUTLINE vertex program has one extra instruction::
#
#     mul r0.x, r0.x, cb0[15] /*+4:_LineWidthS*/.y
#
# All three ``main_item*`` shaders that ship an OUTLINE pass do it: ``main_item``
# (chara/ao_head_00), ``main_item_studio`` (studio/00, which applies it after the
# object-scale division instead of before -- same product) and
# ``main_item_emission`` (chara/ao_arm_08).  The fourth,
# ``main_item_studio_alpha``, has no OUTLINE pass at all, which is why it is not
# in ``OUTLINE_SHADERS``.  No ``main_skin`` / ``main_opaque`` / ``main_alpha`` /
# ``main_hair`` program reads it, so it stays 1 there even on the materials that
# carry the float (2 of the 74 ``main_opaque`` materials on 12 cards do).
ITEM_STEM = "main_item"

# The shader's own declared default, from its property table
# (``origin_shader_dump --props``): ``_LineWidthS type=2 [1, 0, 0, 0]``.  Not a
# neutral value picked because it is neutral.
LINE_WIDTH_S_DEFAULT = 1.0


def line_width_s(shader: dict | None) -> float:
    """``_LineWidthS`` -- ``main_item``'s per-material scale on the hull.

    0 is a real value and means *no hull*, which is how an accessory author
    turns the outline off for a lens or a decal without leaving the shader; 0.05
    and 0.1 also occur.  Of 79 ``main_item`` materials across 12 cards, 4 are
    not 1.
    """
    sh = shader or {}
    leaf = str(sh.get("_shader") or "").rsplit("/", 1)[-1]
    if not leaf.startswith(ITEM_STEM):
        return 1.0
    try:
        return max(0.0, float(sh.get("_LineWidthS", LINE_WIDTH_S_DEFAULT)))
    except (TypeError, ValueError):
        return LINE_WIDTH_S_DEFAULT


# ---------------------------------------------------------------------------
# the two global maps
# ---------------------------------------------------------------------------
def _list_map(category: str, ident):
    """One row of a texture-only list as a ``colorspace.Map``.

    From the bake if it is there.  The ramp keeps the flag and hands it to the
    uploader: the renderer decodes at the fetch now, and the ramp is the slot
    where that matters most — 41 of the 43 base ``ramp`` rows are flagged sRGB,
    and the ramp's output *is* the shading term (§7.126).

    Base rows only, and on purpose: this is for an id nothing has named, which
    for the ramp is a setting with no ``rampGUID``.  A *card* field goes through
    :func:`body_head._list_texture` instead, because a card can name a mod.
    """
    from .. import lists as L
    from .load_mesh import _named_texture
    try:
        ident = int(ident)
    except (TypeError, ValueError):
        return None
    try:
        from .. import bake
        hit = bake.read_texture(category, ident)
        if hit is not None:
            return hit
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.toon._list_map: bake.read_texture", exc)
    ref = L.index().get(category, ident)
    if ref is None:
        return None
    return _named_texture(ref.bundle, ref.asset)


def _setting_ramp():
    """The ramp :func:`ramp_setting` names, from the mod when a GUID names it."""
    rid, mref = ramp_setting()
    if mref is None:
        return _list_map("ramp", rid)
    from .load_mesh import mod_texture
    return mod_texture(mref)


def ramp_map(ident: int | None = None) -> np.ndarray | None:
    """``_RampG`` — the diffuse toon ramp, 4x256, sampled by ``max(N·L, 0)``.

    Which of the seven the game hands over is a **player setting**, so it is
    read rather than fixed (:func:`etc`).  They are not variations on a theme:
    id 1 (深いくっきり影) is a binary step at 0.5, id 6 (影なし) is white, and
    the id this install runs — 2, 深いなめらか影 — is a smooth curve that does
    not leave 1.0 until N·L drops under about 0.75.  Sampling the game's own
    map instead of quantising N·L ourselves is the difference between "a cel
    band" and *the game's* band.

    With no ``ident`` this is the player's setting, which may name a mod
    (:func:`ramp_setting`); an explicit ``ident`` is a base row.
    """
    if ident is None:
        return _setting_ramp()
    return _list_map("ramp", int(ident))


def gloss_map(gloss_id, res: dict | None = None):
    """``_HairGloss`` — the strand highlight sheet from ``hair.glossId``.

    ``ChaControl.LoadHairGlossMask`` reads it with ``GetTexture(433, glossId,
    MainTexAB, MainTex)`` and ``ChangeSettingHairGlossMask`` assigns it to every
    hair renderer's material.  It is a character-wide map; only the ``_ST`` that
    positions it is per hair piece.

    ``glossId`` is a card field like any other, so Sideloader rewrites it
    before the game reads it (``StructReference.ChaFileHairGenerator`` lists
    ``433 / glossId``) and ``res`` — the card's UAR table — has to answer
    first.  Reading the base list alone lost the sheet outright on the 122 of
    1,388 local cards whose UAR names an installed gloss mod: none of those ids
    is a base row, so the lookup came back ``None`` and nothing said so.
    """
    from .body_head import _list_texture
    return _list_texture("hairgloss", gloss_id, res)


_RAMP: dict = {"arr": None, "key": None, "tried": False}


def ramp_key(ident: int | None = None):
    """The texture-cache key for ``_RampG``, kept alive across characters.

    It is one 4x256 map shared by every material in the game, so it is fetched
    once and re-registered only if :mod:`origin.textures` has released it —
    which it does only while no live model draws it (§7.499).
    """
    from .. import textures as TEX
    if not _RAMP["tried"]:
        _RAMP["tried"] = True
        _RAMP["arr"] = ramp_map(ident)
    if _RAMP["arr"] is None:
        return None
    if _RAMP["key"] is None or not TEX.alive(_RAMP["key"]):
        _RAMP["key"] = TEX.store(_RAMP["arr"].px, srgb=_RAMP["arr"].srgb)
    return _RAMP["key"]


def card_gloss_id(card) -> int:
    """``ChaFileHair.glossId``; 0 when the card predates the field."""
    try:
        hair = (card.sections.get("Custom") or {}).get("hair") or {}
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.toon.card_gloss_id: hair = (card.sections.get('Custom') or {}).get('", exc)
        return 0
    try:
        return int(hair.get("glossId", 0) or 0)
    except (TypeError, ValueError):
        return 0
