"""Attach loaded mesh pieces onto a procedural character skeleton."""
from __future__ import annotations

import numpy as np

from ...core import math3d as m3
from ...mesh import parttex as PT
from ...mesh.mesh import Material
from .. import colorspace as CS
from .. import progress as PROG
from .. import textures as TEX
from . import clothes_overlay as CO
from ..apply import is_explicit_overlay
from ..skinmap import BoneMapper
from . import create_tex as CT
from . import toon as TOON


def bone_origin(skel, name: str) -> np.ndarray:
    if not skel.has(name):
        return np.zeros(3)
    return skel.bones[skel.id(name)].bind_world[:3, 3].copy()


# ``ChaReference.RefObjKey.HairParent`` is key **1**, and
# ``CreateReferenceInfo`` files that index under this name exactly once
# (``dictRefObj[1] = GetObjectFromName(root, "cf_J_FaceUp_ty")``; nothing else
# in the method writes index 1).  ``<ChangeHairAsync>`` reads it back with
# ``GetReferenceInfo(1)`` and hands the transform to ``LoadCharaFbxData``,
# which ends at
#
#     newObj.transform.SetParent(trfParent, worldPositionStays)
#
# with ``worldPositionStays = false`` at that call site and no
# ``set_local*`` after it.  A hair prefab's root TRS is identity, so a hair
# prefab's own space **is** this bone's frame -- position, rotation and the
# scale it inherits from ``cf_s_head``, all of it.
#
# The armature agrees independently: the five ``a_n_hair_*`` attach points
# (``RefObjKey`` 2-6, right behind ``HairParent``) are all children of this
# very bone.
#
# Hanging hair off ``cf_j_head`` instead -- which is what "the head bone"
# suggests and what this did before -- drops it by the gap between the two,
# 4.23 cm on the test card, and that is what buried the eyebrows (§7.58).
HAIR_PARENT_BONE = "cf_J_FaceUp_ty"


def hair_frame(skel) -> np.ndarray | None:
    """The frame a hair prefab lives in, or ``None`` if the rig lacks it."""
    if not skel.has(HAIR_PARENT_BONE):
        return None
    return skel.bones[skel.id(HAIR_PARENT_BONE)].bind_world.copy()


def pattern_sheets(colors, *, res=None, missing: list | None = None,
                   prop: str | None = None) -> dict:
    """colour slot → the ``mt_pattern`` tile that slot asks for.

    ``ColorInfo.pattern`` is an id into Illusion's own pattern list, exactly
    like every other id on a card; the row gives the bundle and texture name.

    **An id the base table has not got is a Sideloader slot**, the same as
    every other list on a card, so this goes through the shared texture-only
    lookup (:func:`body_head._list_texture`) instead of consulting the base
    table alone.  Reading only the base table meant a modded pattern silently
    became "no pattern" — and "no pattern" is a legal, common state, so nothing
    anywhere looked wrong.  An id that nothing can answer for is reported
    (§一之一).

    ``prop`` is that garment's UAR property template — ``ClothesBraPattern{n}``
    and friends, straight off ``StructReference.ChaFileClothesGenerator``.
    §7.69 recorded that cards carry no pattern entry in their UAR table, but
    that was one card: 67 of 300 sampled cards have them, and without the
    template this fell back to ``find_any`` — ``CompatibilityResolve``, which
    answers with whichever claimant's GUID sorts first the moment two installed
    mods number a tile alike, not the mod the card names (§7.72 十, §7.478).
    """
    from .body_head import _list_texture

    specs = TEX.card_patterns(colors)
    if not any(specs):
        return {}
    out: dict[int, object] = {}
    for slot, spec in enumerate(specs):
        if not spec:
            continue
        pid = int(spec["id"])
        tex = _list_texture("pattern", pid, res, slot=slot, prop=prop)
        if tex is not None:
            out[slot] = tex
        elif missing is not None:
            missing.append(f"pattern #{pid} 解析不到（顏色槽 {slot}）")
    return out


#: The one shader whose ``_Cutoff`` feeds a screen-space dither (``main_item_ditherd``, §7.1199)
DITHER_SHADER = "main_item_ditherd"


def apply_shader(mat: Material, shader: dict | None, base_rgb=None) -> Material:
    """Take the toon settings off the material instead of assuming them.

    Every Origin material carries its own ``_ShadowColor`` and its shader decides
    the rest.  A sailor blouse asks for a shadow of (0.79, 0.87, 0.98) — a pale
    blue that barely darkens — where this loader used to multiply shadows down
    to 0.55 grey, which is most of why a correctly-coloured character still
    read heavier and dirtier than the game's own render.

    **The outline is not switched by ``_outline`` / ``_linewidth``.**  Those two
    floats look like the control (hair ships 0, clothing 1) but no program in
    ``main_skin`` / ``main_opaque`` / ``main_hair`` / ``main_item`` ever reads
    either — they are dead Shader Forge properties, like ``_ColorContrast``.
    What actually decides it is whether the shader ships an OUTLINE pass, which
    is a fact about the asset (``TOON.OUTLINE_SHADERS``, read with
    ``origin_shader_dump --passes``).  The width comes from the global
    ``_linewidthG`` (see :mod:`toon`) — **except** for ``main_item*``, the one
    family whose vertex program also multiplies by a per-material float,
    ``_LineWidthS`` (``TOON.line_width_s``, §7.78).  The colour resolves to
    black for every family except hair, whose outline pass hard-lights the
    strand colour against the material's own ``_LineColor`` instead of the
    global one.

    ``_rimpower`` is not a strength either: ``main_opaque`` puts it in the
    **exponent** (``1 + 9 * _rimpower``) and takes the strength from ``_rimV``
    and the colour from ``_SpecularColor``.  ``main_skin``'s rim is multiplied
    by the liquid's coverage, which is 0 on a dry body — so dry skin has no rim
    in the game; the liquid layer (``origin.figure.siru``) draws it where wet.

    **``_specular`` is not a strength** — that was ours (PROGRESS §7.50 七).
    ``main_opaque``'s analytic lobe is
    ``saturate(_SpecularPower * (N·H)^(256*_SpecularPower) * _SpecularColor.a)``
    and every garment on the test card ships ``_SpecularPower = 0``, i.e. no
    lobe at all, while ours ran ``0.16 * _specular`` through a ``_SpeclarHeight``
    threshold — and since the character light comes from the camera, ``N·H`` is
    near 1 over most of what faces us, so that invented lobe washed out every
    saturated garment (the tie by 47 steps, the skirt by 59).  ``_SpeclarHeight``
    is the parallax offset on the *texture* specular, not a cut-off.
    """
    if not shader:
        return mat
    fam = TOON.family(shader)
    mat.toon_family = fam
    # The head materials spell the same two properties in lower case
    # (``cf_m_face_00`` carries ``_shadowcolor`` and ``_linecolor``, the body
    # and clothing shaders the capitalised pair), so both names are read.
    shadow = shader.get("_ShadowColor")
    if not isinstance(shadow, (list, tuple)):
        shadow = shader.get("_shadowcolor")
    if isinstance(shadow, (list, tuple)) and len(shadow) >= 3:
        mat.shadow_color = (*(float(c) for c in shadow[:3]), 1.0)
    if TOON.has_outline(shader):
        # 1.0 = Origin's own width; the renderer applies the distance term and
        # ``_linewidthG`` itself, so this stays a plain scale.  ``main_item`` is
        # the one family whose vertex program also multiplies by a per-material
        # float, and 0 there means the author wanted no hull on this piece
        # (see :func:`toon.line_width_s`).
        mat.outline_width = TOON.line_width_s(shader)
        if fam == TOON.FAMILY_HAIR:
            line = shader.get("_LineColor")
            if not isinstance(line, (list, tuple)):
                line = shader.get("_linecolor")
            rgb = ([float(c) for c in line[:3]]
                   if isinstance(line, (list, tuple)) and len(line) >= 3
                   else [0.5, 0.5, 0.5])
            mat.outline_color = (*rgb, 1.0)
        else:
            # hardLight(colour, _LineColorG) with _LineColorG.a = 0 -> black
            mat.outline_color = (0.0, 0.0, 0.0, 1.0)
    elif shader.get("_shader"):
        # A named shader with no Cull Front pass draws no hull: every
        # ``toon_*`` (eyes, sclera, nose line, glasses) and every ``*_low``
        # variant.  An inverted hull round the eyeball paints a dark ring
        # across the bottom of the sclera.
        mat.outline_width = 0.0
    if TOON.is_emblem(shader):
        # main_emblem's forward base is albedo -> hue rebuild -> ramp; there is
        # no rim and no specular lobe in it at all, so the material's
        # ``_rimpower`` / ``_SpecularColor`` are properties nothing samples.
        mat.rim_strength = 0.0
        mat.specular_strength = 0.0
        return mat
    if fam == TOON.FAMILY_GLASS:
        # The program's own rim and specular are part of its formula; ours
        # must not fire on top (§7.413).  Material colours as authored — an
        # accessory's card colours are laid over them by the caller, per
        # ``ChangeAccessoryColor``'s role rules.
        mat.rim_strength = 0.0
        mat.specular_strength = 0.0
        for attr, key in (("glass_color", "_Color"), ("glass_color4", "_Color4")):
            c = shader.get(key)
            if isinstance(c, (list, tuple)) and len(c) >= 4:
                setattr(mat, attr, tuple(float(x) for x in c[:4]))
        mat.glass_params = (float(shader.get("_ColorInverse", 0.0) or 0.0),
                            float(shader.get("_ColorSort", 0.0) or 0.0),
                            float(shader.get("_RimPower", 1.0) or 1.0))
        mat.base_color = (1.0, 1.0, 1.0, 1.0)
    if fam in (TOON.FAMILY_EYE, TOON.FAMILY_EYEW, TOON.FAMILY_UNLIT):
        # The two eye programs have neither a rim nor a lobe -- no ``pow``
        # anywhere in either (§7.359) -- and the noseline has no lighting at
        # all (§7.366).  The head material's own default rim
        # (``make_material``) survived §7.345 moving the iris into the Origin
        # branch, and the renderer adds ours wherever the strength is > 0.
        # No early return: the blend, cut-out and stencil below all apply.
        mat.rim_strength = 0.0
        mat.specular_strength = 0.0
    if "_rimpower" in shader:
        mat.rim_strength = (0.0 if fam == TOON.FAMILY_SKIN
                            else float(shader["_rimpower"]))
        if fam == TOON.FAMILY_SKIN:
            # the skin's rim is the liquid's (``origin.figure.siru``): only the liquid layer reads it
            mat.liquid_rim_power = float(shader["_rimpower"] or 0.0)
    if "_specular" in shader:
        # No character shader reads this.  It survives only for the materials
        # that fall through to *this project's* banded model — ``toon_glasses``
        # and the other ``toon_*`` sheets — where a per-material hint is better
        # than none.  Every Origin family below overwrites it with 0.
        mat.specular_strength = float(mat.specular_strength) * float(
            shader["_specular"])
    if fam in (TOON.FAMILY_SKIN, TOON.FAMILY_CLOTH):
        # The four floats main_skin's and main_opaque's specular block reads.
        # ``specular_strength`` goes to zero here on purpose: it drives *this
        # project's* Blinn lobe, and the two must not both fire.
        mat.origin_specular = float(shader.get("_SpecularPower", 0.0) or 0.0)
        mat.spec_nail = float(shader.get("_SpecularPowerNail", 0.0) or 0.0)
        mat.no_tex_spec = float(shader.get("_notusetexspecular", 0.0) or 0.0)
        mat.rim_v = float(shader.get("_rimV", 0.0) or 0.0)
        spec = shader.get("_SpecularColor")
        if isinstance(spec, (list, tuple)) and len(spec) >= 4:
            # the alpha is not decoration: it scales the whole lobe, and it is
            # 0 on ``cf_m_body`` and on half the garments
            mat.specular_color = tuple(float(c) for c in spec[:4])
        mat.specular_strength = 0.0
    if fam == TOON.FAMILY_HAIR:
        # hair's highlight is _HairGloss, not a Blinn lobe
        mat.specular_strength = 0.0
        mat.shade_floor = 1.0 - float(shader.get("_ShadowExtend", 0.0) or 0.0)
    elif fam == TOON.FAMILY_CLOTH:
        # the floor under main_opaque's shadow-tint scale — a different
        # property from the hair one despite the similar name
        mat.shade_floor = 1.0 - float(
            shader.get("_ShadowExtendAnother", 0.0) or 0.0)
    if "_SpeclarHeight" in shader:
        # One property, three readers, and none of them a threshold:
        # main_opaque uses it as the parallax scale on the texture specular
        # (``specular_threshold``), main_hair as the shift that walks the
        # strand band with the view angle (``gloss_height``), and this
        # project's own banded model as a cut-off (the only reader that treats
        # it as one, and the only one it is not from).
        mat.specular_threshold = float(shader["_SpeclarHeight"])
        mat.gloss_height = float(shader["_SpeclarHeight"])
    if "_BumpScale" in shader:
        mat.normal_scale = float(shader["_BumpScale"])
    # ``_Cutoff`` is only a threshold for a shader whose program reads it;
    # every Shader Forge shader declares it ``[HideInInspector]`` and compares
    # against a literal instead, which :data:`TEX.CLIP_BY_SHADER` holds —
    # letting the float win put main_item back at 0.5 (§7.372).
    leaf = str(shader.get("_shader") or "").split("/")[-1]
    if leaf == DITHER_SHADER and "_Cutoff" in shader:
        # its program reads ``_Cutoff`` in a screen-space dither instead of a plain clip (§7.1199); the
        # shader declares the property twice with two defaults, so only a material's saved value is used
        mat.dither_cut = float(shader["_Cutoff"])
        mat.alpha_test = 0.0
    elif ("_Cutoff" in shader and mat.alpha_test > 0.0
            and leaf not in TEX.CLIP_BY_SHADER):
        mat.alpha_test = float(shader["_Cutoff"])
    # Two independent signals, and both are needed.  ``_Mode 3`` with ZWrite
    # off is Illusion's transparent *material* setup, which the eye and nose
    # sheets use.  But a material can be ``_Mode 0`` and still be drawn by a
    # shader that has no discard and sits in the Transparent queue —
    # ``toon_glasses_lod0`` is exactly that, and a lens material ships
    # ``_Mode 0``, so testing only the mode left glasses fully opaque over the
    # eyes.  The shader is the authority on whether a pass blends at all.
    # ``shader_discards``, **not** ``shader_clip``: the question here is whether
    # the shader has a discarding pass at all, which is a fact about the asset.
    # ``shader_clip`` answers the narrower "did this material switch it on"
    # (§7.88's keywords), and reading that here turned every accessory whose
    # material carries no ``_ALPHA_*`` keyword into a blended surface — which is
    # how BR-Chan's helmet lost the green band painted on its shell (§7.90).
    if TEX.shader_blends(shader) or not TEX.shader_discards(shader):
        mat.blend = True
    mat.depth_write = leaf in TEX.BLEND_ZWRITE
    queue = shader.get("_queue", -1)
    if queue is None or int(queue) < 0:
        queue = shader.get("_shaderQueue", -1)
    mat.render_queue = int(queue) if queue is not None else -1
    if leaf in TEX.CLIP_AND_BLEND:
        # a discard that only drops the invisible, in front of a blended pass
        # whose alpha is ``_MainTex.a × _alpha`` (§7.373)
        mat.blend = True
        r, g, b, a = mat.base_color
        mat.base_color = (r, g, b, a * float(shader.get("_alpha", 1.0)))
    # The stencil pair is a property of the shader asset, like the OUTLINE pass
    # above, so it is read from the same place and not from a material float.
    mat.stencil_write = TOON.stencil_write(shader)
    mat.stencil_reject = TOON.stencil_reject(shader)
    mat.stencil_clip = TOON.stencil_clip(shader)
    return mat


def tangent_normal(arr):
    """A usable tangent-space normal map, or ``None``.

    Illusion's ``_n`` maps are DXT5nm — Y replicated across R/G/B, X in alpha —
    so reading them as ``rgb * 2 - 1`` gives a flat grey and the lighting turns
    to noise.  That is why they used to be skipped entirely.  They are not
    unusable, just packed: :func:`textures.unpack_normal` rebuilds the vector
    and the result comes out unit length (|n| mean 0.994, the rest is 8-bit
    quantisation).  A mod that ships a conventional map is passed through.
    """
    if arr is None or getattr(arr, "ndim", 0) != 3 or arr.shape[2] < 3:
        return None
    if TEX.is_packed_normal(arr):
        return TEX.unpack_normal(arr)
    a = arr[..., :3].astype(np.float32)
    b_mean = float(a[..., 2].mean())
    rg_mean = float(a[..., :2].mean())
    return arr if (b_mean > 170.0 and b_mean > rg_mean * 1.15) else None


def shader_normal(arr):
    """A ``_NormalMap`` as the source game's character shaders read it, every map alike; ``None`` without one.

    ``main_item``, ``main_item_studio`` and ``main_skin`` (D3D11 disassembly, §7.1163) sample the
    map and take **x from alpha, y from green**, ``z = sqrt(1 - min(x² + y², 1))`` — Unity's
    ``UnpackNormal`` on a DXT5nm platform, with no test of what the map looks like.  A map without
    an alpha channel samples alpha 1.  :func:`tangent_normal` judged each map by its look and
    dropped the ones it could not place, silently (a hat's 16× tiled map among them).
    """
    if arr is None or getattr(arr, "ndim", 0) != 3 or arr.shape[2] < 3:
        return None
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, dtype=arr.dtype)], axis=2)
    return TEX.unpack_normal(arr)


def attach_maps(mat: Material, piece, gloss=None) -> None:
    """Hand the renderer ``_NormalMap`` / ``_AnotherRamp`` / the shading masks.

    ``_AnotherRamp`` used to be gated behind a "does this look like a toon
    ramp?" test, because feeding it in as the diffuse band blew the surface
    out.  It is not a diffuse ramp: ``main_hair`` samples it by **N·V** and
    combines it with the global ``_RampG`` by ``min``, and ``main_opaque``
    samples it by **|N·H|** and blends it in per texel.  Read that way it never
    needed a guess — the slot is exactly what it says, we were pointing it at
    the wrong angle.

    Unbound slots are left alone rather than filled with white: the shader's
    property table gives ``_DetailMask`` / ``_LineMask`` / ``_HairGloss`` a
    **black** default and ``_AnotherRamp`` a white one, which is what the
    renderer's "not bound" path already reproduces.
    """
    shader = getattr(piece, "shader", None) or {}
    if TOON.is_emblem(shader):
        # main_emblem declares three properties — ``_MainTex``, ``_shadowcolor``
        # and ``_Cutoff`` — and samples two textures: its own map and the
        # global ramp.  Every other slot here would be a map the badge's
        # program has no sampler for.
        mat.ramp = TOON.ramp_key()
        return
    # Every map below goes to the GPU with the asset's own ``m_ColorSpace``, so
    # the hardware decodes it exactly where Koikatsu's does — at the fetch, and
    # therefore *before* the bilinear filter (§7.126).  The flag is read off the
    # piece with ``colorspace.of``; it is never re-derived from the field name,
    # because the local library splits both ways on three of these five slots.
    nrm_map = CS.of(piece, "normal")
    nrm = shader_normal(getattr(piece, "normal", None))
    if nrm is not None and nrm_map is not None:
        # The wrap mode is the asset's (``m_TextureSettings``, carried as
        # ``_mapWrap``): a tiled ``_NormalMap_ST`` repeats only on a Repeat map.
        wrap = (shader.get("_mapWrap") or {}).get("normal")
        key = TEX.store(nrm, srgb=nrm_map.srgb, repeat=wrap == 0)
        if key is not None:
            mat.normal_map = key
            mat.normal_transform = tuple(
                shader.get("_NormalMap_ST") or (1.0, 1.0, 0.0, 0.0))

    ramp_map = CS.of(piece, "ramp")
    if ramp_map is not None:
        key = TEX.store(ramp_map.px, srgb=ramp_map.srgb)
        if key is not None:
            mat.view_ramp = key
            mat.view_ramp_transform = tuple(
                shader.get("_AnotherRamp_ST") or (1.0, 1.0, 0.0, 0.0))

    props = dict(shader)
    mask = CT.shade_mask(CS.of(piece, "detail"), CS.of(piece, "line_mask"),
                         props, mat.toon_family)
    if mask is not None:
        # a packed coefficient, not a colour: linear
        key = TEX.store(mask, srgb=False)
        if key is not None:
            mat.shade_mask = key

    if mat.toon_family == TOON.FAMILY_CLOTH:
        # main_opaque is the family that reads raw ``_DetailMask`` channels the
        # packed mask cannot hold, including one at a view-dependent uv (see
        # :func:`create_tex.detail_map`).  Only this family, because carrying a
        # second full-size map for every piece is what the packing exists to
        # avoid.
        dm_map = CS.of(piece, "detail")
        dm = CT.detail_map(getattr(piece, "detail", None), props)
        if dm is not None and dm_map is not None:
            # the map as itself: it keeps ``_md``'s own flag so the shader's
            # four raw channels read what the game's read
            key = TEX.store(dm, srgb=dm_map.srgb)
            if key is not None:
                mat.detail_mask = key

    if mat.toon_family != TOON.FAMILY_NONE:
        mat.ramp = TOON.ramp_key()

    if mat.toon_family == TOON.FAMILY_GLASS:
        gr = CS.of(piece, "glass_ramp")
        if gr is not None:
            wrap = (shader.get("_mapWrap") or {}).get("glass_ramp")
            key = TEX.store(gr.px, srgb=gr.srgb, repeat=wrap == 0)
            if key is not None:
                mat.glass_ramp = key
                mat.glass_ramp_xf = tuple(
                    shader.get("_GlassRamp_ST") or (1.0, 1.0, 0.0, 0.0))

    if gloss is not None and mat.toon_family == TOON.FAMILY_HAIR:
        # Repeat, not clamp: the gloss sheet is m_WrapMode 0 and the UV1 the
        # hair carries runs well outside 0..1 (-2.53 .. 4.40 on the sampled
        # style), so clamping would smear one column across the whole head.
        key = TEX.store(gloss.px, srgb=gloss.srgb, repeat=True)
        if key is not None:
            mat.gloss = key
            mat.gloss_transform = tuple(
                shader.get("_HairGloss_ST") or (1.0, 1.0, 0.0, 0.0))


def clothes_slots(colors, sheets: dict | None = None,
                  shader: dict | None = None, hair_rgb=None) -> list:
    """``colorInfo`` as ``create_topN``'s three slots.

    Each is (baseColor, patternColor, pattern map, tiling) — the four things
    ``ChangeCustomClothes`` writes per slot into ``_Color{i}`` / ``_Color{i}_2``
    / ``_PatternMask{i}`` / ``_PatternScale{i}{u,v}``.

    **Slot 0 is not always ``colorInfo[0]``.**  A card's top carries *four*
    colours, and ``ChangeCustomClothes`` ends with

        if (!main && kind == 2) {
            SetColor(_Color,    colorInfo[3].baseColor);
            SetColor(_Color1_2, colorInfo[3].patternColor);
            ...
            goto slot1;                      // skip the colorInfo[0] block
        }
        SetColor(_Color, colorInfo[0].baseColor);
      slot1: … colorInfo[1] → _Color2, colorInfo[2] → _Color3

    so the **third sub-part of a combo top** (``cpo_sailor_c`` /
    ``cpo_jacket_c`` — the neckerchief) takes the fourth colour where every
    other piece takes the first.  Its colour mask is pure red, i.e. entirely
    slot 0, so reading only ``colorInfo[0..2]`` painted the ribbon with the
    blouse's white and lost it against the shirt.  ``load_combo_top`` tags the
    pieces with ``_comboSub`` because after the three sub-parts are merged
    nothing else distinguishes them.

    **The pattern moves with the colour, and the same IL says so.**  The method
    builds its property table as ``[_PatternMask1, _PatternMask2,
    _PatternMask3, _PatternMask1]`` and runs the pattern loop over
    ``(!main && kind == 2) ? 4 : 3`` slots, so on the neckerchief the fourth
    swatch's ``pattern`` and ``tiling`` land in **slot 1's** mask — the same
    slot its ``baseColor`` and ``patternColor`` land in.  Taking the colour
    from ``colorInfo[3]`` while leaving the map on ``colorInfo[0]`` mixed two
    swatches into one slot: a ribbon whose own swatch asks for stripes came out
    plain, and one whose blouse asks for a print wore the blouse's print.

    Accessories are the exception, and ``ChaAccessoryComponent`` says how (see
    :meth:`prefab.PrefabLoader.accessory_roles`).  ``useColor0N = 0`` means the
    card never sets that slot, so the material's own ``_ColorN`` is the item's
    intended colour: a straw hat's band is dark brown and a swim cap's trim
    near-black, and overwriting them with the card's unused white swatch is
    what left hats looking bleached.  A ``rendHair`` renderer takes the
    *wearer's* hair start colour in all three slots instead.
    """
    entries = list(colors or [])
    sheets = sheets or {}
    sh = shader or {}
    role = str(sh.get("_accRole") or "")
    use = list(sh.get("_useColor") or ())
    # ``!main && kind == 2`` — the combo top's neckerchief reads colour 4 here
    slot0 = 3 if (int(sh.get("_comboSub", -1)) == 2
                  and len(entries) > 3) else 0
    out = []
    for i in range(3):
        src = slot0 if i == 0 else i
        c = entries[src] if src < len(entries) else None
        # An **accessory** slot is a bare ``Color``: ``ChaFileAccessory.PartsInfo``
        # declares ``color`` as ``Color[4]`` and ``ChangeAccessoryColor`` writes
        # ``color[0..2]`` straight into ``_Color`` / ``_Color2`` / ``_Color3``.
        # Only a garment wraps its colour in a ``ColorInfo`` with the pattern
        # fields beside it.  Reading dicts only meant every accessory's three
        # slots came back empty and the composite fell through to
        # ``make_material``'s (0.82, 0.82, 0.85) — the grey every hat, choker
        # and hairpin on every card was drawn in (§7.66).
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            c = {"baseColor": list(c)}
        c = c if isinstance(c, dict) else {}
        base = c.get("baseColor")
        if role == "hair" and hair_rgb is not None:
            base = list(hair_rgb[:3]) + [1.0]
        elif role and i < len(use) and not use[i]:
            authored = sh.get("_Color" if i == 0 else f"_Color{i + 1}")
            if isinstance(authored, (list, tuple)) and len(authored) >= 3:
                base = list(authored)
        out.append((base, c.get("patternColor"),
                    sheets.get(src), c.get("tiling") or (0.0, 0.0), None))
    return out


def hair_outline_rgb(part: dict | None):
    """``parts[slot].outlineColor`` as an RGB triple, or ``None``."""
    c = (part or {}).get("outlineColor")
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return tuple(float(x) for x in c[:3])
    return None


def _rgb3(c):
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return tuple(float(x) for x in c[:3])
    return None


def customised_hair_accessory_slots(info: dict, hair_part: dict | None,
                                    shader: dict | None, colors):
    """HairAccessoryCustomizer's colours for one accessory piece (:mod:`hair_acc`).

    ``rendHair`` with ``ColorMatch`` takes the wearer's ``parts[0]`` base /
    start / end in ``_Color`` / ``_Color2`` / ``_Color3``; ``rendAccessory``
    takes ``AccessoryColor`` in all three.  Both write every slot the material
    has, whatever the accessory component's ``useColor0N`` say, so those are
    set to 1.  A ``rendHair`` piece without ``ColorMatch`` keeps the vanilla
    accessory colours (*colors*).  Returns ``(colours, shader copy)``.
    """
    sh = dict(shader or {})
    role = sh.get("_hairComp")
    if role == "hair" and info.get("ColorMatch"):
        p = hair_part or {}
        cols = [p.get(k) for k in ("baseColor", "startColor", "endColor")]
    elif role == "accessory" and info.get("AccessoryColor") is not None:
        cols = [info["AccessoryColor"]] * 3
    else:
        return colors, sh
    if not all(isinstance(c, (list, tuple)) and len(c) >= 3 for c in cols):
        return colors, sh
    sh["_useColor"] = [1, 1, 1]
    return [list(c[:3]) + [1.0] for c in cols], sh


def hair_accessory_slots(part: dict | None, shader: dict | None):
    """``ChangeSettingHairAcsColor`` for one non-``rendHair`` piece of a hair.

    Transcribed::

        num = acsDefColor.Length                     // GetHairAcsColorNum
        for j in range(num):
            if acsColor[j].a < 1: acsColor[j] = Color(r, g, b, 1)
            rendAccessory[i].material.SetColor([_Color, _Color2, _Color3][j],
                                               acsColor[j])

    ⇒ the first *num* slots take the card's ``acsColor`` (alpha forced to 1)
    and the rest keep the material's own ``_ColorN``, which is exactly what an
    accessory's ``useColor0N = 0`` does in :func:`clothes_slots` — so the piece
    goes through that path as ``_accRole = "normal"`` with ``_useColor`` set to
    the first *num*.  Returns ``(colours, shader)`` (a copy of *shader*).
    """
    sh = dict(shader or {})
    n = (max(0, min(3, int(sh.get("_hairAcsN") or 0)))
         if sh.get("_hairComp") == "accessory" else 0)
    acs = [c for c in ((part or {}).get("acsColor") or [])]
    cols = []
    for j in range(3):
        c = acs[j] if j < len(acs) else None
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            cols.append([float(x) for x in c[:3]] + [1.0])
        else:
            cols.append(None)
    use = [1 if (j < n and cols[j] is not None) else 0 for j in range(3)]
    sh["_accRole"] = "normal"
    sh["_useColor"] = use
    return [c if c is not None else [1.0, 1.0, 1.0, 1.0] for c in cols], sh


def emblem_material(name: str, texture_arr) -> Material:
    """``main_emblem``: the badge's own map, clipped, with no tint and no hull.

    Everything ``create_topN`` does — the three colours, the pattern layers,
    the colour mask — belongs to the garment's composite, and the emblem is not
    part of it: ``ChangeCustomEmblem`` binds one texture and stops.
    """
    mat = Material(
        name=name,
        base_color=(1.0, 1.0, 1.0, 1.0),
        shadow_color=(0.55, 0.55, 0.58, 1.0),
        specular_strength=0.0,
        rim_strength=0.0,
        outline_width=0.0,
        shade_steps=2.0,
        shade_feather=0.06,
        two_sided=True,
        alpha_test=TOON.EMBLEM_CLIP,
    )
    # The badge goes to the GPU verbatim: ``main_emblem`` runs no create pass,
    # so the bytes are the asset's and so is the flag.  All 51 base emblem
    # textures read linear (§7.86), but that is measured, not assumed here.
    key = (TEX.store(texture_arr.px, srgb=texture_arr.srgb)
           if texture_arr is not None else None)
    if key is not None:
        mat.texture = key
    return mat


def _overlay_key(piece) -> tuple:
    """``(combo sub-part or -1, composite index)`` of one garment renderer."""
    sh = piece.shader or {}
    sub = int(sh.get("_comboSub", -1))
    role = str(sh.get("_clothRole") or "")
    return sub, {"normal02": 1, "normal03": 2}.get(role, 0)


def _overlay_times(pieces, overlays, slot_index: int) -> dict:
    """``{(sub, composite): (Overlay, times)}`` for one garment's pieces.

    A composite is shared by every renderer it feeds, so the count is per
    composite: ``ApplyOverlays`` blits into ``ct[0]`` once for the first
    ``rendNormal01`` renderer and into ``ct[1]`` once for the first
    ``rendNormal02`` one (:func:`clothes_overlay.applicable_sets`).
    """
    roles: dict = {}
    for p in pieces:
        sh = p.shader or {}
        roles.setdefault(int(sh.get("_comboSub", -1)), set()).add(
            str(sh.get("_clothRole") or ""))
    out: dict = {}
    for sub, rs in roles.items():
        ov = overlays.for_garment(slot_index, None if sub < 0 else sub)
        if ov is None:
            continue
        for ct in CO.applicable_sets(rs):
            prev = out.get((sub, ct), (ov, 0))
            out[(sub, ct)] = (ov, prev[1] + 1)
    return out


def make_material(name: str, colors=None, *, texture_arr=None,
                  mask=None, cutout: bool = False,
                  allow_auto_cutout: bool = True,
                  sheets: dict | None = None,
                  alpha_mask=None, shader: dict | None = None,
                  hair_rgb=None, unbound_rgb=None,
                  overlay=None, overlay_times: int = 0) -> Material:
    """One piece's material.

    ``unbound_rgb`` is what the shader samples when **no** map is bound, for
    the families where that is a stated fact rather than a guess.  A garment
    composites through ``create_topN``, whose property table declares
    ``_MainTex`` = white and ``_ColorMask`` = black, and
    ``InitBaseCustomTextureClothes`` only assigns either when the list row
    names it (the ``!= "0"`` guards) — ``cf_m_clothesN_create`` itself binds
    neither.  With both unbound the colour chain collapses to 1 and the
    composite is white, so that is what a mapless garment is, **not** the
    card's first swatch: painting it ``colorInfo[0]`` invented a colour that
    no channel of the game ever produces, and made a failed texture lookup
    look like a design decision (§一之一).
    """
    if TOON.is_emblem(shader):
        return emblem_material(name, texture_arr)
    rgb_list = TEX.card_colors(colors)
    base = rgb_list[0] if rgb_list else (0.82, 0.82, 0.85)
    if unbound_rgb is not None and texture_arr is None and mask is None:
        base = tuple(float(c) for c in unbound_rgb[:3])
    # ``create_topN``, transcribed — see :mod:`create_tex`.  The previous
    # version averaged the three colours by mask coverage and painted the
    # pattern on as ink; the game lerps R→G→B in sequence and treats the
    # pattern as each slot's *second colour*.
    if texture_arr is not None or mask is not None:
        baked = CT.clothes(texture_arr, mask,
                           clothes_slots(colors, sheets, shader, hair_rgb))
    else:
        baked = None
    if baked is not None and overlay is not None and overlay_times:
        # ``ApplyOverlays`` runs on the composite before the shader reads it,
        # so the ``_AlphaMask`` cut below sees the overlaid alpha.
        baked = CO.composite(baked, overlay, overlay_times)
    sh = shader or {}
    before = baked
    baked = TEX.apply_alpha_mask(baked, alpha_mask,
                                 float(sh.get("_alpha_a", 1.0)),
                                 float(sh.get("_alpha_b", 1.0)))
    # An ``_AlphaMask`` that actually removed something has to be honoured
    # even on a "solid" garment: the shader clips at 0.5 whenever the slot is
    # bound, and this is how an outer layer punches a hole in the shirt under
    # it.  Without this the mask was computed and then never tested — the
    # jacket's inner shirt kept the 42.9% of itself the blazer covers.
    masked = (alpha_mask is not None and baked is not None
              and (before is None or int(baked[..., 3].min()) < 128))
    # **The material says whether it cuts out.**  ``m_ShaderKeywords`` selects
    # the discarding forward base, and the test it runs is
    # ``min(alpha-mask term, _MainTex.a) < 0.5`` — the albedo's *own* alpha is
    # in it.  The pixel heuristic that used to stand in for this was switched
    # off for tops and bottoms ("a few semi-transparent edge texels punch holes
    # in jackets"), which is exactly the case it got wrong: an albedo whose
    # transparent texels carry compressed-block garbage then showed that
    # garbage.  ``小悪魔トップス``'s ``cf_top_manken_d_00_t`` is 7.63% alpha < 10,
    # and the 99th percentile of the saturation there is **255** — the black
    # strips and rainbow speckles in the lacing (§7.88).  The head path already
    # took the clip from the shader; this one did not (the §7.66 五 pattern:
    # one call site honouring a rule and the others not).
    clip = TEX.shader_clip(shader)
    use_cut = cutout or masked or clip > 0.0 \
        or (allow_auto_cutout and TEX.needs_cutout(baked))
    if baked is not None:
        mat = Material(
            name=name,
            base_color=(1.0, 1.0, 1.0, 1.0),
            shadow_color=(0.55, 0.55, 0.58, 1.0),
            specular_strength=0.16,
            specular_power=48.0,
            rim_strength=0.22,
            outline_color=(0.15, 0.15, 0.16, 1.0),
            outline_width=0.0 if use_cut else 0.85,
            shade_steps=2.0,
            shade_feather=0.06,
            two_sided=True,
            alpha_test=(clip or TEX.ALPHA_CLIP) if use_cut else 0.0,
        )
        # a create-pass composite: the game's own render target is sRGB, so
        # these bytes are gamma-encoded exactly like Origin's (§7.85 二)
        key = TEX.store(baked, srgb=True)
        if key is not None:
            mat.texture = key
        return mat
    return Material(
        name=name,
        base_color=(*base, 1.0),
        shadow_color=(*(np.clip(np.array(base) * 0.68 + 0.08, 0, 1)), 1.0),
        specular_strength=0.14,
        specular_power=44.0,
        rim_strength=0.20,
        outline_color=(*(np.array(base) * 0.30), 1.0),
        outline_width=0.85,
        shade_steps=2.0,
        shade_feather=0.06,
        two_sided=True,
    )


def hair_material(rgb=(0.86, 0.84, 0.82)) -> Material:
    base = tuple(float(c) for c in rgb[:3])
    return Material(
        name="origin_hair",
        base_color=(*base, 1.0),
        shadow_color=(*(np.clip(np.array(base) * 0.68 + 0.06, 0, 1)), 1.0),
        specular_color=(1.0, 0.98, 0.95, 1.0),
        specular_strength=0.32,
        specular_power=70.0,
        rim_strength=0.28,
        outline_color=(*(np.array(base) * 0.30), 1.0),
        outline_width=0.95,
        shade_steps=2.0,
        shade_feather=0.06,
        two_sided=True,
        alpha_test=TEX.ALPHA_CLIP,
    )


# There used to be a ``_settle_mesh`` here: it rotated a whole attached piece
# toward gravity about its attach point, by ``amount × the angle between the
# piece's centre and straight down``.  Nothing in the game does that.  An
# accessory is parented to its ``a_n_*`` node and stays there
# (``ChangeAccessoryParent``); the only motion it has is its own ``DynamicBone``
# chains, which ``ChangeShakeAccessory`` merely enables or disables from the
# card's ``noShake``.  Those chains are solved to steady state by
# ``prefab.sway_rest`` and used by ``BoneMapper.rest_pose``, which switches its
# own per-bone approximation off wherever it has a solved answer — and then this
# rotated the solved shape anyway.
#
# On a piece whose centre sits *above* the pivot — every hat, hairpin and fringe
# on ``a_n_headside`` / ``a_n_headtop`` — the arm points up, the angle to "down"
# is ~180°, and 0.55 of it is a 99° flop about an axis that is numerically
# arbitrary when the two vectors are nearly antiparallel.  That is what laid
# three front-hair strands flat across a face (§7.66).  The guard meant to keep
# it off heads never fired: it compared ``accessory_frame``'s first return value
# against ``cf_j_head``, and that value is the ``a_n_*`` node name whenever the
# node exists — which, since the full graft, is always.
#
# A ``plausible`` gate that dropped pieces whose centre fell outside a per-slot
# height band lived next to it and had no caller left (§7.350); a band in
# metres is a guess about every card's proportions, and a dropped piece is
# the silent kind of failure 一之一 forbids — a piece in the wrong place is a
# ``report.missing`` about its bone, not a height test.


def attach_pieces(built, pieces, *, mapper: BoneMapper, group: str,
                  name_prefix: str, colors=None, body_scale: float = 1.0,
                  unit: float = 1.0,
                  default_bone: str = "cf_j_hips",
                  hair_mat: Material | None = None,
                  attach_frame: np.ndarray | None = None,
                  move: "AccessoryMove | None" = None,
                  no_shake: bool = False,
                  category: str = "",
                  head_len: float = 0.22,
                  rejected: list | None = None,
                  layer: int | None = None,
                  frame_absolute: bool = False,
                  copy_weights: int | None = None,
                  gloss=None, hair_rgb=None, res=None,
                  pattern_prop: str | None = None,
                  overlays=None, slot_index: int = -1,
                  hair_part: dict | None = None,
                  hair_acc: dict | None = None) -> tuple[int, int]:
    """Attach every piece as its own Part. Returns (n_parts, total_tris).

    Skinning policy
    ---------------
    * Weights map onto our body rig (``mapper.knows >= 0.5``) → retarget + remap.
    * Weights only on prefab sway bones → rest_pose inside the prefab, then hang
      on the attach frame (accessories / trailing hair).
    * No weights → scale + attach frame / default bone.

    ``copy_weights`` is ``LoadCharaFbxDataAsync``'s argument of that name, and
    passing it replaces the bone-name match above with the game's own rule.
    Only ``AssignedWeightsAndSetBounds`` rebinds a prefab's bones onto the
    character, and it runs only for a non-zero value::

        if      (copyWeights == 1) aaWeightsBody.AssignedWeightsAndSetBounds(
                                       newObj, "cf_j_root",       bounds, trfRoot)
        else if (copyWeights == 2) aaWeightsHead.AssignedWeightsAndSetBounds(
                                       newObj, "cf_J_N_FaceRoot", bounds, trfRoot)

    The nine ``<ChangeClothes*Async>`` loaders pass **1**; hair and the head
    pass **0**.  ``<ChangeAccessoryAsync>`` computes it, and *not* from a
    ``weight`` column — there is no such column, and no ``Weight`` in
    ``ChaListDefine.KeyType`` either::

        weight = 0; trfParent = null
        if ("null" == lib.GetInfo(ChaListDefine.KeyType.Parent))   # 54
            { weight = 2; trfParent = this.objTop.transform }

    So an accessory is rebound onto the head skeleton exactly when its list
    row's ``Parent`` cell reads the literal string ``"null"`` instead of an
    ``a_n_*`` node — and locally **no row does**: 0 of 357 base plus 12,168
    zipmod accessory rows.  Every accessory here therefore keeps its own bones
    and rides its attach node rigidly, which is what ``rest_pose`` does.

    The bone-name match already agreed, on all three groups.  Over 30 cards,
    ``mapper.knows`` is **1.000 for all 224 skinned garment pieces** and
    **0.000 for all 122 skinned hair pieces and all 133 skinned accessory
    ones** — nothing lands in between, so ``know_gate`` has never actually
    decided anything: a garment's bones *are* the character's bones by name,
    and hair and accessories bring their own.  That is the same split
    ``copyWeights`` encodes.  Passing it changes no pixel today; it stops a mod
    whose bones happen to be named ``cf_j_*`` from being skinned to the wearer
    when the game would not have done that.

    ``frame_absolute`` says ``attach_frame`` is the parent transform's whole
    ``bind_world`` — the game's ``SetParent(parent, worldPositionStays: false)`` —
    so it already carries the bone's own scale and ``unit`` must not be applied
    again.  It does **not** carry the graft's height rescale: the graft only
    scales bone *offsets* (``rig.graft_full_skeleton``: ``loc[:3, 3] *= scale``)
    and leaves every 3×3 at Illusion's native size, which is exactly what
    ``BoneMapper.to_our_scale`` exists to finish for a retargeted mesh.  A
    prefab hung off that frame needs the same factor or it stays at Illusion's
    size on a rig that is not — 13.8% too large on the test card, which put the
    hair crown 2.8 cm above where it belongs (§7.61).

    ``move`` is the card's ``addMove`` for an accessory, already resolved
    against the item's own ``N_move`` chain (:class:`AccessoryMove`).  It is a
    transform in the *prefab's* space, so it lands on the bone tracks or on the
    mesh before anything places the piece on the body.  ``no_shake`` is the
    card's ``PartsInfo.noShake``: ``ChangeShakeAccessory`` does nothing with it
    but ``dynamicBone.enabled = !noShake``, and our equivalent of those chains
    running is ``bone_rest``, so switching it off means falling back to the
    authored track.
    """
    skel = built.skeleton
    if frame_absolute and attach_frame is not None:
        attach_frame = mapper.to_our_scale(attach_frame)
    n_ok = tris = 0
    head_default = default_bone if skel.has(default_bone) else "cf_j_hips"
    cat = category or group
    # Clothes need a slightly lower threshold: some skirt bones are only in the
    # reference armature, not as named joints on our skeleton, but still map
    # via nearest-bone.  Accessories stay strict so stray props don't stick
    # to the hips.
    # Full Origin graft → knows≈1.0; never raise the gate to drop pieces.
    know_gate = 0.15
    # One garment, one set of pattern tiles: ``colorInfo`` belongs to the slot,
    # not to the renderer, so this is per call and not per piece — computing it
    # inside the loop also repeated its "cannot resolve" line once per piece.
    sheets = pattern_sheets(colors, res=res, missing=rejected,
                            prop=pattern_prop) if hair_mat is None else {}
    # Our own maps, when this build draws its own (``PROGRESS.md`` F0-6).  Asked
    # once per call and not once per piece: it reads a setting and the manifest,
    # and a garment comes through here with up to a dozen renderers.
    own_maps = PT.enabled()
    # Overlay Mods' clothes overlay (:mod:`clothes_overlay`): which composite of
    # which garment it is blitted into, and how many times.  Decided per
    # ``ChaClothesComponent`` — the slot's own, or one per combo sub-part.
    ov_times = _overlay_times(pieces, overlays, slot_index) if overlays else {}

    for i, piece in enumerate(pieces):
        # Hair goes through here 9 times per character with a bundle load
        # between calls, and the whole hair stage carried no pump at all:
        # 4.66 s of one Free H open, the longest single stretch left (§7.108).
        # Clothes and accessories share the path.
        PROG.tick()
        mesh = piece.mesh
        if mesh is None or mesh.vertex_count == 0:
            continue
        mname = (getattr(mesh, "name", None) or "").lower()
        if is_explicit_overlay(mname):
            continue
        # **Before** anything reads the maps.  The albedo composite, the packed
        # shading mask and the raw ``_DetailMask`` all take them off the piece,
        # so this is the one place that keeps the three in step.
        if own_maps:
            piece = PT.reskin(piece, cat)
        mesh = mesh.copy()
        bone = head_default

        known = mapper.knows(piece.bones) if piece.bones else 0.0
        # ``UpdateAccessoryMoveFromInfo`` writes into the middle of the prefab
        # hierarchy, so the override belongs on the bone tracks (a skinned
        # piece's geometry follows its bones) or on the vertices (a rigid one
        # already has the chain baked in) — not on the finished item.
        bworld, brest = piece.bone_world, piece.bone_rest
        # ``ChangeSettingHairLength``: a hair slot's ``lengthRate`` is the
        # card's ``hair.parts[slot].length`` (``MemberInit`` 0); an accessory
        # keeps its prefab's unless HairAccessoryCustomizer sets ``HairLength``.
        # ``trfLength`` sits below ``N_move`` in any prefab that has both, so
        # this goes on first (§7.516).
        length = HairLength.of(piece, hair_length_rate(hair_mat is not None, hair_part, hair_acc))
        len_move = None
        if length is not None:
            bworld, brest = length.bone_tracks(piece, bworld, brest)
            len_move = length.mesh_matrix(piece)
        if no_shake:
            # ChangeShakeAccessory: dynamicBone.enabled = !noShake.  Our
            # ``bone_rest`` *is* those chains having run, so the card asking
            # for no shake means the authored track stands.
            brest = None
        mesh_move = None
        if move is not None:
            bworld, brest = move.bone_tracks(piece, bworld, brest)
            mesh_move = move.mesh_matrix(piece)
        if copy_weights == 0:
            rebind = False
        elif copy_weights:
            rebind = piece.skin is not None and bool(piece.bones)
        else:
            rebind = (piece.skin is not None and piece.bones
                      and known >= know_gate)
        if rebind:
            # AssignedAnotherWeights: exact-name remap + rest from full bind
            if mesh_move is not None or (move is not None and np.any(
                    piece.move_bones if piece.move_bones is not None else 0)):
                # This path skins to *our* armature and never looks at the
                # prefab's own transforms, so there is nowhere to write the
                # move onto — say so rather than drop the card's offset
                # without a word.  ``copy_weights`` keeps this unreachable for
                # accessories on every item installed here; see its note in the
                # signature.
                if rejected is not None:
                    rejected.append(
                        f"{cat}: {mname or name_prefix} — 走了 retarget 路徑，"
                        f"addMove 沒有套用")
            idx, wt = mapper.remap(piece.bones, piece.skin)
            if not mapper.retarget(mesh, piece.bones, piece.skin, piece.binds,
                                   native=True):
                mesh.scale(body_scale)
            mesh.bone_idx, mesh.bone_wt = idx, wt
            # No settle on skinned Origin meshes — game leaves them in bind pose
        else:
            if piece.skin is not None:
                mapper.rest_pose(mesh, piece.skin, piece.binds,
                                 bworld, settle=0.0, bone_rest=brest,
                                 report=rejected,
                                 label=f"{cat}: {mname or name_prefix}")
            elif mesh_move is not None or len_move is not None:
                # A rigid renderer arrives with the whole prefab chain already
                # multiplied into its vertices, so the override goes on here —
                # and *before* any of the placement below, because it is a
                # transform in the prefab's own space.
                if len_move is not None:
                    mesh.transform(len_move)
                if mesh_move is not None:
                    mesh.transform(mesh_move)
            if frame_absolute and attach_frame is not None:
                # The frame is the parent transform's own bind_world, so it
                # already carries the scale the game's SetParent hands down.
                # Multiplying by ``unit`` on top of it scales twice.
                pass
            # unit scale first (Illusion cm-ish → our metres via head ratio)
            elif abs(unit - 1.0) > 1e-4:
                mesh.scale(unit)
            elif group != "accessory":
                mesh.scale(body_scale)
            if attach_frame is not None:
                mesh.transform(attach_frame)
            elif group != "clothes":
                if skel.has(bone):
                    mesh.translate(bone_origin(skel, bone))
            mesh.set_bone(skel.id(bone) if skel.has(bone) else skel.id("cf_j_hips"))

        # Soft size gate only (catastrophic scale mismatch).  Do not drop
        # pieces for Y-range — that was "compressing" the Origin read path.
        lo, hi = mesh.bounds()
        size = float(np.max(hi - lo))
        if size > head_len * 20.0:
            if rejected is not None:
                rejected.append(f"{cat}: {mname or name_prefix} — 尺寸 {size:.2f}m")
            continue

        # The normals are the ones the piece was authored with, carried through
        # skinning by ``skinmap.skin_normals`` and through the attach frame by
        # ``Mesh.transform``.  Recomputing them here — which is what this did —
        # threw all of that away one line before the piece was handed over, and
        # on a head the authored normals are the drawing, not a derivative of
        # the triangles (§7.61).  Only a piece that arrived without any gets
        # them computed.
        if mesh.vertex_count and not np.any(mesh.normals):
            mesh.compute_normals()
        mesh.normalize_weights()

        # ``ChaCustomHairComponent``'s lists (§7.446): the hair colours and
        # the outline colour go to ``rendHair`` and nowhere else;
        # ``rendAccessory`` (a tie, a ribbon) is a prop with its own map and
        # takes the part's ``acsColor`` instead, and a renderer on neither list
        # is written by nothing — ``p_cf_hair_f_01``'s tie is one — so it keeps
        # its material's own colours (:func:`hair_accessory_slots` with
        # nothing to write).  An archive baked before the lists were read has
        # no key and keeps the old path; ``dress`` reports it.
        hcomp = (piece.shader or {}).get("_hairComp")
        hair_acs = hair_mat is not None and hcomp in ("accessory", "")
        piece_colors = colors
        if hair_acs:
            piece_colors, sh_new = hair_accessory_slots(hair_part, piece.shader)
            piece = piece._replace(shader=sh_new)
        # HairAccessoryCustomizer on an accessory slot (:mod:`hair_acc`); for
        # an accessory *hair_part* is the wearer's ``hair.parts[0]``
        acc_hair = (hair_acc is not None and hair_mat is None
                    and hcomp in ("hair", "accessory"))
        if acc_hair:
            piece_colors, sh_new = customised_hair_accessory_slots(
                hair_acc, hair_part, piece.shader, piece_colors)
            piece = piece._replace(shader=sh_new)
        if hair_mat is not None and not hair_acs:
            rgb = TEX.card_colors(colors)
            hair_rgb = rgb[0] if rgb else (0.86, 0.84, 0.82)
            mat = hair_material(hair_rgb)
            # ``create_hair``, transcribed: the card's three hair colours are
            # applied by sequential lerp through _ColorMask's R, G, B and the
            # result multiplies _MainTex.  ``ChangeSettingHairColor`` fills
            # them from the part's baseColor / startColor / endColor.
            #
            # The old version averaged the three by mask coverage and then
            # *added* a highlight scaled by an invented ``GLOSS_GAIN``.  Origin has
            # no such term here — ``_DetailMask`` / ``_HairGloss`` feed the
            # render shader's specular, not the albedo (see §7.10).
            cols = list(colors or []) or [{"baseColor": list(hair_rgb) + [1.0]}]

            def _slot(i):
                c = cols[i] if i < len(cols) else cols[0]
                return c.get("baseColor") if isinstance(c, dict) else c

            # **No _MainTex.**  Neither the prefab material nor the ``bo_hair_*``
            # list row names one (the row has no MainTex column at all), so the
            # shader multiplies by white and the colour chain is the whole
            # albedo.  ``piece.texture`` here is only what ``_main_texture``
            # guessed from the material's name — for hair that is the green
            # strand sheet, and multiplying by it dyes a blonde card black.
            if (piece.shader or {}).get("_meMainTex"):
                # MaterialEditor replaced ``_MainTex``, which on hair is this
                # very composite (``create_hair``'s render target): the card's
                # image is what the shader samples (§7.382).
                baked = CS.pixels(CS.of(piece, "texture"))
            else:
                baked = CT.hair(None, CS.of(piece, "mask"),
                                _slot(0), _slot(1), _slot(2))
            # The strand cut-out is ``_AlphaMask.r`` and **nothing else**
            # (:func:`TEX.apply_hair_alpha_mask`).  What was here fell back to
            # "whichever of _DetailMask / _ColorMask carries some alpha", the
            # `有 alpha 就用 alpha` heuristic §一之二 rules out — and it is not
            # a harmless one: ``main_hair`` samples neither of those for alpha,
            # ``create_hair`` throws the strand sheet's own alpha away
            # (``mov o0.w, l(1.0)``), and the two maps it reached for happen to
            # carry alpha on real assets.  ``ロングヘア`` lost 93.6% of itself
            # and came out a tuft on the crown; ``ほそロング``'s map is alpha 0
            # everywhere, so the test card's two side locks were drawn and then
            # discarded in full.  Neither reported anything.
            alpha_mask = CS.sampled_bytes(CS.of(piece, "alpha_mask"))
            baked = TEX.apply_hair_alpha_mask(baked, alpha_mask)
            if baked is not None:
                key = TEX.store(baked, srgb=True)      # create_hair composite
                if key is not None:
                    mat.texture = key
                    mat.base_color = (1.0, 1.0, 1.0, 1.0)
                    # With no ``_AlphaMask`` bound the shader's own default is
                    # white, so ``a = _MainTex.a``: 1 for ``create_hair``'s
                    # composite, where the 0.5 clip can never fire, and the
                    # image's own alpha for a MaterialEditor ``_MainTex``.
                    mat.alpha_test = (TEX.ALPHA_CLIP
                                      if alpha_mask is not None
                                      or (piece.shader or {}).get("_meMainTex")
                                      else 0.0)
        else:
            # The three composites a garment builds differ only in *which map*
            # they run over: ``ChangeCustomClothes`` writes ``colorInfo[0..2]``
            # into ``_Color`` / ``_Color2`` / ``_Color3`` on all three alike, so
            # a renderer's texture set says nothing about the colours it takes.
            # (The one exception is the combo top's third sub-part, and that is
            # keyed off ``_comboSub`` inside :func:`clothes_slots`.)  This used
            # to hand set 1 ``colorInfo[3:]``, which is the fourth swatch — the
            # one only the neckerchief reads.
            slot_colors = piece_colors
            # lace-ish slots may auto-cutout; solid clothing must not
            solid = cat in (
                "top", "bottom", "shoes", "shoes_inner", "shoes_outer",
                "gloves", "clothes",
            )
            # A garment is composited by ``create_topN`` and every one of them
            # names an albedo; arriving with neither albedo nor colour mask
            # means the row's map did not resolve, and the only other sign of
            # that is a flat-coloured shirt.  Say it, and let the composite
            # fall to the shader's own declared default rather than to a
            # colour of our choosing (:func:`make_material`).
            garment = group == "clothes"
            if (garment and piece.texture is None and piece.mask is None
                    and rejected is not None):
                pname_ = (getattr(mesh, "name", None) or f"片 {i}")
                rejected.append(
                    f"{cat}: {pname_} 沒有 albedo 也沒有色遮罩"
                    " —— 清單列的貼圖沒解析到，畫成 shader 的預設白")
            # MaterialEditor's ``_MainTex`` on a garment replaces the
            # ``create_topN`` composite the game already bound, so it is the
            # final albedo: no colour mask means a colour chain of 1 and the
            # composite passes it through (§7.382).  An accessory's
            # ``main_item`` colours ``_MainTex`` in the shader, so there the
            # image is an input like any other.
            me_final = garment and bool((piece.shader or {}).get("_meMainTex"))
            ov, times = ov_times.get(_overlay_key(piece), (None, 0))
            mat = make_material(
                f"{name_prefix}_{i}", slot_colors,
                texture_arr=CS.of(piece, "texture"),
                mask=None if me_final else CS.of(piece, "mask"),
                sheets=None if me_final else sheets,
                allow_auto_cutout=not solid,
                # the value the shader's fetch returns, not the bytes (§7.386)
                alpha_mask=CS.sampled_bytes(CS.of(piece, "alpha_mask")),
                shader=getattr(piece, "shader", None),
                hair_rgb=hair_rgb,
                unbound_rgb=(1.0, 1.0, 1.0) if garment else None,
                overlay=None if me_final else ov, overlay_times=times)

        # blend the line against the garment's own colour, not the material's
        # base — a textured material's base is white, which would wash it out
        surf = TEX.card_colors(piece_colors)
        psh = getattr(piece, "shader", None) or {}
        apply_shader(mat, psh, base_rgb=surf[0] if surf else mat.base_color)
        # an accessory's ``_HairGloss`` is the character's sheet only where
        # the plugin put it there (``HairGloss``); nothing else hands it one
        piece_gloss = gloss if hair_mat is not None else (
            gloss if acc_hair and hcomp == "hair" and hair_acc.get("HairGloss")
            else None)
        attach_maps(mat, piece, gloss=piece_gloss)
        line = None
        if hair_mat is not None and hcomp == "hair":
            line = hair_outline_rgb(hair_part)
        elif acc_hair and hcomp == "hair":
            line = (hair_outline_rgb(hair_part) if hair_acc.get("ColorMatch")
                    else _rgb3(hair_acc.get("OutlineColor")))
        if line is not None:
            # ``ChangeSettingHairOutlineColor`` (and the plugin's twin):
            # ``rendHair[i].material._LineColor``; only ``main_hair*``'s hull
            # reads the material's own ``_LineColor`` (:func:`apply_shader`)
            if mat.outline_width > 0.0 and mat.toon_family == TOON.FAMILY_HAIR:
                mat.outline_color = (*line, 1.0)
        # ``noOutline`` on the accessory component: some props (lenses, wings,
        # decals) are authored without a hull even though their shader has the
        # pass.  ``ChaAccessoryComponent`` states it, so it is read, not judged.
        if psh.get("_accRole") and not int(psh.get("_accOutline", 1) or 0):
            mat.outline_width = 0.0
        role = str(psh.get("_accRole") or "")
        if mat.toon_family == TOON.FAMILY_GLASS:
            _glass_card_colours(mat, role, psh, colors, hair_rgb)
            state = psh.get("_glassRamp")
            if state != "bound" and state != "none" and rejected is not None:
                pname_ = (getattr(mesh, "name", None) or f"片 {i}")
                rejected.append(
                    f"{category or group}: {pname_} 的 _GlassRamp "
                    + ("引用解不到" if state == "unresolved" else
                       "沒有進封存（bake 早於 §7.413，重轉："
                       "bake_assets --shader toon_glasses_lod0）"))
        # ``rendAlpha`` is switched off entirely when the fourth colour's
        # alpha is 0 — that is how a card turns off an optional strap or veil
        # (``SetActiveIfDifferent(color[3].a != 0)``).  An accessory colour is
        # a bare ``Color[4]``; reading only the ``ColorInfo`` dict shape meant
        # this switch never fired on an accessory at all (§7.413).
        if role == "alpha":
            c4 = _card_rgba(colors, 3)
            if c4 is not None and c4[3] <= 0.0:
                continue

        pname = f"{name_prefix}_{i}"
        n = 0
        while built.model.get(pname) is not None:
            n += 1
            pname = f"{name_prefix}_{i}_{n}"
        if layer is not None:
            use_layer = int(layer)
        elif group == "hair":
            use_layer = 1
        elif group == "face":
            use_layer = 2
        elif group == "accessory":
            use_layer = 3
        else:
            use_layer = 7
        part = built.model.add_mesh(pname, mesh, mat, group=group, layer=use_layer)
        part.dynamic = False
        # The name Illusion's renderer had.  MaterialEditor addresses renderers
        # by it and nothing else on a part carries it (§7.138); our own
        # ``pname`` is ``origin_acc_{type}_{id}_{i}``, which says nothing about
        # which of the prefab's renderers this is.
        part.origin_renderer = str(psh.get("_rendName") or "")
        # The ``UpdateVisible`` state node it hangs under (§7.379); undress
        # switches these per ``clothesState`` (§7.387).
        part.origin_state_node = str(psh.get("_stateNode") or "")
        part.cast_shadows = int(psh.get("_castShadows", 1))
        n_ok += 1
        tris += mesh.tri_count
    return n_ok, tris


# Accessory parent keys → our bones
PARENT_BONE = {
    "a_n_hair_pony": "cf_j_head", "a_n_hair_twin_L": "cf_j_head",
    "a_n_hair_twin_R": "cf_j_head", "a_n_hair_pin": "cf_j_head",
    "a_n_hair_pin_R": "cf_j_head", "a_n_headtop": "cf_j_head",
    "a_n_headside": "cf_j_head", "a_n_headflont": "cf_j_head",
    "a_n_head": "cf_j_head", "a_n_face": "cf_j_head",
    "a_n_earrings_L": "cf_j_head", "a_n_earrings_R": "cf_j_head",
    "a_n_nose": "cf_j_head", "a_n_mouth": "cf_j_head",
    "a_n_megane": "cf_j_head", "a_n_glasses": "cf_j_head",
    "a_n_neck": "cf_j_neck", "a_n_bust_f": "cf_j_spine03",
    "a_n_bust": "cf_j_spine03", "a_n_bust_L": "cf_j_spine03",
    "a_n_bust_R": "cf_j_spine03", "a_n_back": "cf_j_spine02",
    "a_n_back_L": "cf_j_spine02", "a_n_back_R": "cf_j_spine02",
    "a_n_waist": "cf_j_hips", "a_n_waist_f": "cf_j_hips",
    "a_n_waist_b": "cf_j_hips", "a_n_waist_L": "cf_j_hips",
    "a_n_waist_R": "cf_j_hips",
    "a_n_arm_L": "cf_j_arm00_L", "a_n_arm_R": "cf_j_arm00_R",
    "a_n_wrist_L": "cf_j_hand_L", "a_n_wrist_R": "cf_j_hand_R",
    "a_n_hand_L": "cf_j_hand_L", "a_n_hand_R": "cf_j_hand_R",
    "a_n_ind_L": "cf_j_index01_L", "a_n_ind_R": "cf_j_index01_R",
    "a_n_mid_L": "cf_j_middle01_L", "a_n_mid_R": "cf_j_middle01_R",
    "a_n_ring_L": "cf_j_ring01_L", "a_n_ring_R": "cf_j_ring01_R",
    "a_n_leg_L": "cf_j_thigh00_L", "a_n_leg_R": "cf_j_thigh00_R",
    "a_n_knee_L": "cf_j_leg01_L", "a_n_knee_R": "cf_j_leg01_R",
    "a_n_ankle_L": "cf_j_leg03_L", "a_n_ankle_R": "cf_j_leg03_R",
    "a_n_heel_L": "cf_j_foot_L", "a_n_heel_R": "cf_j_foot_R",
    "a_n_shoulder_L": "cf_j_shoulder_L", "a_n_shoulder_R": "cf_j_shoulder_R",
    "a_n_dan": "cf_j_hips", "a_n_kokan": "cf_j_hips",
}


def _card_rgba(colors, i: int) -> tuple | None:
    """Card colour *i* as ``(r, g, b, a)`` — an accessory's bare ``Color`` or a
    garment's ``ColorInfo.baseColor``; ``None`` when the card has none."""
    c = (colors or [])[i] if i < len(colors or []) else None
    if isinstance(c, dict):
        c = c.get("baseColor")
    if isinstance(c, (list, tuple)) and len(c) >= 4:
        return tuple(float(x) for x in c[:4])
    return None


def _glass_card_colours(mat: Material, role: str, shader: dict, colors,
                        hair_rgb) -> None:
    """``ChangeAccessoryColor`` onto a ``toon_glasses_lod0`` material (§7.413).

    ``rendNormal``: ``_Color`` = colour 1 when ``useColor01`` (``_Color2``/``3``
    are not read by this shader); ``rendAlpha``: **only** ``_Color4`` = colour
    4, the rest stays the material's; ``rendHair``: the wearer's hair colour.
    A piece that is not an accessory keeps its material as authored.
    """
    if role == "alpha":
        c4 = _card_rgba(colors, 3)
        if c4 is not None:
            mat.glass_color4 = c4
    elif role == "normal":
        use = list(shader.get("_useColor") or (1, 1, 1))
        c0 = _card_rgba(colors, 0)
        if use and use[0] and c0 is not None:
            mat.glass_color = c0
    elif role == "hair" and hair_rgb is not None:
        mat.glass_color = (*(float(x) for x in hair_rgb[:3]), 1.0)


def accessory_frame(skel, parent_key: str,
                    scale: float = 1.0) -> tuple[str, np.ndarray]:
    """Return (bone, attach_frame) for one accessory slot.

    ``ChangeAccessoryParent`` ends at

        obj.transform.SetParent(GetReferenceInfo(key).transform, false)

    — ``worldPositionStays: false``, the same call hair goes through — so the
    attach frame *is* that ``a_n_*`` transform and the prefab's own space is its
    frame.  ``attach_pieces`` is therefore given it as ``frame_absolute``, which
    is what finishes the graft's height rescale on it (``BoneMapper``'s
    ``to_our_scale``; see §7.61).  Scaling the mesh by the head-length ratio
    ``unit`` instead — which is what this did — left every accessory **6.4%**
    too big, because ``unit`` is 0.9352 on the test card where the graft is
    0.8787 (§7.61 九).

    The full graft carries all 54 attach nodes (``rig.body_bone_names`` keeps
    ``a_n_*``), so the fallback only runs when the armature has none — a graft
    that failed.  It used to be the only path, and it could not work:
    ``ReferenceRig.attach_offset`` decided "offset or world position?" from
    ``|y| < 0.6``, which read ``a_n_back`` (world y 1.38) as *no offset at all*
    and dropped a backpack inside the ribcage, while reading ``a_n_heel``
    (world y 0.034) as an offset and moved a shoe charm by its own height.
    ``scale`` pre-multiplies that offset so it survives ``to_our_scale``, which
    only touches the 3×3.
    """
    from ..rig import rig
    bone = PARENT_BONE.get(parent_key, "cf_j_head")
    if parent_key and skel.has(parent_key):
        return parent_key, skel.bones[skel.id(parent_key)].bind_world.copy()
    frame = np.eye(4)
    if skel.has(bone):
        frame = skel.bones[skel.id(bone)].bind_world.copy()
    off = rig().attach_offset(parent_key, relative_to=bone)
    if off is not None:
        frame = frame @ m3.translate(np.asarray(off, dtype=np.float64) * scale)
    return bone, frame


def _vec(v, default) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v.get("x", default[0])),
                         float(v.get("y", default[1])),
                         float(v.get("z", default[2]))])
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return np.array([float(v[0]), float(v[1]), float(v[2])])
    return np.asarray(default, dtype=float)


def card_move_locals(mv, *, hide_hair: bool = False) -> list:
    """``PartsInfo.addMove`` → the two local TRS matrices the game writes.

    ``ChaControl.UpdateAccessoryMoveFromInfo``, transcribed::

        if GetCategoryInfo(type)[id].GetInfoInt(24 /* HideHair */) == 1:
            for i in 0, 1:                 # a hat takes no manual offset
                objAcsMove[slot][i].transform.localPosition    = (0, 0, 0)
                objAcsMove[slot][i].transform.localEulerAngles = (0, 0, 0)
                objAcsMove[slot][i].transform.localScale       = (1, 1, 1)
        else:
            for i in 0, 1:
                t = objAcsMove[slot][i].transform
                t.localPosition    = addMove[i, 0] * 0.01     # centimetres
                t.localEulerAngles = addMove[i, 1]            # degrees
                t.localScale       = addMove[i, 2]

    The card stores ``addMove`` as a 2×3 array of ``Vector3`` — msgpack writes
    it as ``[2, 3, [pos0, rot0, scale0, pos1, rot1, scale1]]`` — so **there are
    two groups**, one per move node, and reading only the first is why the
    second node kept whatever the prefab authored.

    Rotation goes through the same X mirror as every other Illusion transform:
    ``mirror(Quaternion.Euler(x, y, z))`` is exactly
    ``quat_from_euler(x, -y, -z, "YXZ")`` — checked against Unity's documented
    Z→X→Y composition over 4,000 random angles, max error 0.
    """
    if hide_hair:
        return [m3.identity(), m3.identity()]
    slots = mv[2] if isinstance(mv, list) and len(mv) >= 3 else None
    if not isinstance(slots, list):
        return [m3.identity(), m3.identity()]
    out = []
    for i in range(2):
        base = i * 3
        if base + 2 >= len(slots):
            # ``Vector3[2,3]`` always serialises all six; a short one is a card
            # we cannot read, and identity is what the node would hold anyway
            out.append(m3.identity())
            continue
        pos = _vec(slots[base], (0, 0, 0)) * 0.01
        rot = _vec(slots[base + 1], (0, 0, 0))
        scl = _vec(slots[base + 2], (1, 1, 1))
        q = m3.quat_from_euler(*np.deg2rad([rot[0], -rot[1], -rot[2]]),
                               order="YXZ")
        out.append(m3.compose(np.array([-pos[0], pos[1], pos[2]]), q,
                              np.maximum(scl, 1e-3)))
    return out


def hair_length_rate(hair_slot: bool, hair_part: dict | None, hair_acc: dict | None):
    """The ``lengthRate`` a piece's ``ChaCustomHairComponent`` runs with (§7.516).

    A hair slot's is ``ChangeSettingHairLength``'s ``hair.parts[slot].length``
    (``MemberInit`` 0); an accessory's is HairAccessoryCustomizer's
    ``HairLength`` where the plugin has the slot, else ``None`` — the prefab's
    own serialised rate, which nothing else writes.
    """
    if hair_slot:
        return float((hair_part or {}).get("length", 0.0) or 0.0)
    return None if hair_acc is None else hair_acc.get("HairLength")


def graft_hair_bones(skel, mapper, pieces, chains: list, frame_bone: str, frame: np.ndarray,
                     length_rate) -> list:
    """Hang a hair prefab's own bones on the armature, so its ``DynamicBone`` chains move every frame (B4b).

    ``<ChangeHairAsync>`` parents the whole prefab, its bones with it, to the
    hair frame, and the prefab's ``DynamicBone`` runs every frame on them.
    Here they are added under *frame_bone* where the attach puts the piece
    (*frame* through :meth:`BoneMapper.to_our_scale`, the authored tracks with
    the card's hair length on them), so a piece skinned to them lands where the
    baked one did.  A chain bone hangs from its chain parent (the order the
    solver walks it); every other bone of the prefab sits still on the head and
    hangs from the frame.  Returns the names added — none when a name is
    already on the armature (two slots sharing a bone name), and the piece is
    then left to the bake.
    """
    from ...core import math3d as m3
    to_model = mapper.to_our_scale(np.asarray(frame, dtype=np.float64))
    world: dict = {}
    for piece in pieces:
        if piece.skin is None or not piece.bones or piece.bone_world is None:
            continue
        tracks = piece.bone_world
        length = HairLength.of(piece, length_rate)
        if length is not None:
            tracks, _rest = length.bone_tracks(piece, tracks, piece.bone_rest)
        for name, track in zip(piece.bones, tracks):
            # the armature's frames are unit-sized with scaled offsets (``[R | s·t]``); the skin's
            # rebuild scales the 3×3 back (``to_our_scale``), so the vertices land where the bake had them
            placed = to_model @ np.asarray(track, dtype=np.float64)
            placed[:3, :3] /= float(getattr(mapper, "scale", 1.0) or 1.0)
            world.setdefault(str(name), placed)
    if not world or any(skel.has(name) for name in world) or not skel.has(frame_bone):
        return []
    parent: dict = {}
    chain_order: list = []
    for chain in chains:
        bones = [str(b) for b in chain.get("bones") or ()]
        for name, p, end in zip(bones, chain.get("parent") or (), chain.get("end") or ()):
            if end or name not in world:
                continue
            parent[name] = bones[int(p)] if int(p) >= 0 and bones[int(p)] in world else frame_bone
            chain_order.append(name)
    order = [name for name in world if name not in parent] + chain_order
    added: list = []
    for name in order:
        above = parent.get(name, frame_bone)
        above_world = world[above] if above in world else skel.bones[skel.id(above)].bind_world
        t, q, s = m3.decompose(np.linalg.inv(above_world) @ world[name])
        skel.add(name, above, t, q, s)
        added.append(name)
    skel.finalize()
    cache = getattr(mapper, "_cache", None)
    if isinstance(cache, dict):
        cache.clear()                    # ``_resolve`` memoised these names as unknown
    return added


class HairLength:
    """``ChaCustomHairComponent``'s ``trfLength`` writes, as prefab-space corrections (§7.516).

    ``Update`` sets ``trfLength[i].localPosition.y = baseLength[i] +
    Mathf.Lerp(0, addLength, lengthRate)`` every frame.  Replacing a local y
    translates that node and its subtree by the parent's world rotation and
    scale applied to ``(0, Δy, 0)`` (``prefab.length_chain`` baked that
    direction and the rest y), so each group is one translation, applied to
    the bone tracks of the bones under it and to a rigid piece's vertices —
    the same shape as :class:`AccessoryMove`.  ``Mathf.Lerp`` clamps the rate.
    A group whose node is a ``DynamicBone`` particle (``dyn``) is left alone:
    the chain puts the serialised position back before it solves.
    """

    __slots__ = ("_t",)

    def __init__(self, info: dict, rate: float):
        rate = min(max(float(rate), 0.0), 1.0)
        add = float(info.get("add", 0.0) or 0.0)
        self._t: list = []
        for g, base in zip(info.get("groups") or (), info.get("base") or ()):
            if g is None or g.get("dyn"):
                self._t.append(None)
                continue
            dy = float(base) + add * rate - float(g["y0"])
            t = np.eye(4)
            t[:3, 3] = np.asarray(g["dir"], dtype=np.float64) * dy
            self._t.append(t)

    @classmethod
    def of(cls, piece, rate) -> "HairLength | None":
        """The piece's correction for *rate*, or ``None`` (no component, or a
        disabled one — its ``Update`` never runs).  A ``None`` *rate* is the
        prefab's own ``lengthRate``."""
        info = (getattr(piece, "shader", None) or {}).get("_hairLen")
        if not info or not info.get("on", True):
            return None
        return cls(info, info.get("rate", 0.0) if rate is None else rate)

    def _for_mask(self, mask: int):
        out = None
        for bit, t in enumerate(self._t):
            if t is None or not (mask >> bit) & 1:
                continue
            out = t if out is None else out @ t
        return out

    def bone_tracks(self, piece, bone_world, bone_rest):
        """``(bone_world, bone_rest)`` with each bone's groups applied."""
        bits = (getattr(piece, "shader", None) or {}).get("_lenBones")
        if not bits:
            return bone_world, bone_rest
        out = []
        for track in (bone_world, bone_rest):
            if track is None:
                out.append(None)
                continue
            moved = list(track)
            for k in range(min(len(moved), len(bits))):
                t = self._for_mask(int(bits[k])) if moved[k] is not None else None
                if t is not None:
                    moved[k] = t @ np.asarray(moved[k], dtype=np.float64)
            out.append(moved)
        return out[0], out[1]

    def mesh_matrix(self, piece):
        """The correction for a rigid renderer, whose vertices carry its chain."""
        return self._for_mask(int((getattr(piece, "shader", None) or {}).get("_lenMask", 0) or 0))


class AccessoryMove:
    """The card's ``addMove`` resolved against one item's ``N_move`` chain.

    The game's world matrix for a piece of an accessory is

        world = A · move0 · B · move1 · C

    with A / B / C the prefab-local stretches between the two move nodes
    (:meth:`prefab.PrefabLoader.move_chain`).  Given the node's parent world
    ``P`` and its own world ``W`` as the prefab authored them, replacing that
    node's local by ``M`` is the prefab-space correction

        T = P · M · W⁻¹

    and ``T0 · T1 · world`` reproduces the line above exactly — ``W⁻¹``
    cancels the stretch above the node and ``P · M`` puts the new one back.
    A node under ``N_move2`` only (a pair prefab keeps its right half there,
    outside ``N_move``) takes ``T1`` alone, which is why the mask is per
    renderer and per bone rather than one flag for the item.

    What this replaces multiplied ``addMove`` onto the *outside* of the
    finished item.  That is the same thing only while everything above
    ``N_move`` is identity, and it is not on 131 of the 357 base accessories:
    ``p_acs_valletta_heart1`` hangs under a 1.5× ``nc`` node, so a card asking
    for −2 cm gets −3 cm in the game and got −2 cm here (§7.66 二).
    """

    __slots__ = ("_t", "_any")

    def __init__(self, moves, locals_):
        self._t: list = []
        nodes = list(moves or [])
        for i in range(2):
            node = nodes[i] if i < len(nodes) else None
            local = locals_[i] if i < len(locals_) else None
            if node is None or local is None:
                self._t.append(None)
                continue
            parent, world = node
            parent = np.asarray(parent, dtype=np.float64).reshape(4, 4)
            world = np.asarray(world, dtype=np.float64).reshape(4, 4)
            try:
                inv = np.linalg.inv(world)
            except np.linalg.LinAlgError:
                self._t.append(None)      # a degenerate node moves nothing
                continue
            t = parent @ np.asarray(local, dtype=np.float64) @ inv
            self._t.append(None if np.allclose(t, np.eye(4), atol=1e-9) else t)
        self._any = any(t is not None for t in self._t)

    def __bool__(self) -> bool:
        return self._any

    def _for_mask(self, mask: int):
        """``T0 · T1`` for the nodes this transform hangs under."""
        out = None
        for bit in (0, 1):
            if not (int(mask) >> bit) & 1:
                continue
            t = self._t[bit]
            if t is None:
                continue
            out = t if out is None else out @ t
        return out

    def mesh_matrix(self, piece):
        """The correction for a rigid renderer's own transform."""
        if not self._any:
            return None
        return self._for_mask(int(getattr(piece, "move_mask", 0) or 0))

    def bone_tracks(self, piece, bone_world, bone_rest):
        """``(bone_world, bone_rest)`` with each bone's own correction applied."""
        mask = getattr(piece, "move_bones", None)
        if not self._any or mask is None or not len(mask):
            return bone_world, bone_rest
        out = []
        for track in (bone_world, bone_rest):
            if track is None:
                out.append(None)
                continue
            moved = list(track)
            for k in range(min(len(moved), len(mask))):
                if moved[k] is None:
                    continue
                t = self._for_mask(int(mask[k]))
                if t is not None:
                    moved[k] = t @ np.asarray(moved[k], dtype=np.float64)
            out.append(moved)
        return out[0], out[1]
