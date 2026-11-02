"""A converted Endfield material as this renderer's ``Material``.

The lighting itself is ``gl/shaders_endfield.py``; this is the table that feeds
it — which material property goes to which uniform and which map to which
sampler — read off the same programs (``docs/updates/2026-10-02/7.566.md``).

A material carries the keywords of every pass of its shader.  The ones the
transcription does not cover are **named in the note** that comes back with the
material rather than dropped: the surface is drawn with what was transcribed,
and the note says what is missing from it.
"""
from __future__ import annotations

import numpy as np
from OpenGL import GL

from ..gl import shaders_endfield as SE
from ..mesh.mesh import Material
from .part_roles import DECLARED_ROLE

#: Keywords the transcription handles, per family (render-state and other-pass
#: keywords included: they are accounted for, not ignored).
HANDLED = {
    SE.FAMILY_MAIN: {"_NORMALMAP", "_METALLICSPECGLOSSMAP", "_ALPHATEST_ON",
                     "_DIFF_RAMP_ON", "_EMISSION", "_SPEC_RAMP_ON", "_ALPHABLEND_ON",
                     "_OUTLINE_MASK", "_SHADOW_LUT_TEX", "_ANISOTROPY_SPECULAR_ON", "_CLEARCOAT"},
    SE.FAMILY_SKIN: {"_DIFF_RAMP_ON", "_SHADOW_LUT_TEX", "_NORMALMAP", "_SDFLIGHTMAP",
                     "_OUTLINE_MASK", "_EMOTION_MAP", "_HIGHLIGHT_MAP"},
    SE.FAMILY_HAIR: {"_DIFF_RAMP_ON", "_METALLICSPECGLOSSMAP", "_NORMALMAP",
                     "_SPECULAR_LINE", "_SPEC_RAMP_ON", "_SPECULAR_NORMALMAP",
                     "_OUTLINE_MASK", "_ALPHABLEND_ON", "_DRAW_UNDER_BROW",
                     "_ALPHATEST_ON", "_SHADOW_LUT_TEX"},
    SE.FAMILY_EYE: {"_DIFF_RAMP_ON", "_EYE_HIGHLIGHT", "_MATCAP_ON", "_SHADOW_LUT_TEX"},
    SE.FAMILY_OVERLAY: {"DISABLE_DRAW_UNDER_HAIR"},
}
#: Skin, hair and eye were only read with a diffuse ramp; without one the
#: material is not drawn.
NEEDS_RAMP = frozenset({SE.FAMILY_SKIN, SE.FAMILY_HAIR, SE.FAMILY_EYE})

# (Material attribute, shader property, repeat?, keyword that binds it or "")
# ``repeat`` is the sampler the program reads it with, not the texture's own.
_MAPS = {
    SE.FAMILY_MAIN: (
        ("texture", "_BaseMap", True, ""),
        ("normal_map", "_BumpMap", True, "_NORMALMAP"),
        ("shade_mask", "_MetallicGlossMap", True, "_METALLICSPECGLOSSMAP"),
        ("ramp", "_DiffRampMap", False, "_DIFF_RAMP_ON"),
        ("view_ramp", "_SpecRampMap", False, "_SPEC_RAMP_ON"),
        ("glass_ramp", "_ShadowLutTex", False, "_SHADOW_LUT_TEX"),
        ("pattern", "_EmissionMap", True, "_EMISSION"),
        ("detail_mask", "_OutlineMask", True, "_OUTLINE_MASK"),
        ("gloss", "_ClearCoatMask", True, "_CLEARCOAT")),
    SE.FAMILY_SKIN: (
        ("texture", "_BaseMap", True, ""),
        ("normal_map", "_BumpMap", True, "_NORMALMAP"),
        ("ramp", "_DiffRampMap", False, "_DIFF_RAMP_ON"),
        ("glass_ramp", "_ShadowLutTex", False, "_SHADOW_LUT_TEX"),
        ("shade_mask", "_SDFMask", False, "_SDFLIGHTMAP"),
        ("view_ramp", "_SDFLightmap", False, "_SDFLIGHTMAP"),
        ("gloss", "_HighlightMap", True, "_HIGHLIGHT_MAP"),
        ("pattern", "_EmotionMap", True, "_EMOTION_MAP"),
        ("detail_mask", "_OutlineMask", True, "_OUTLINE_MASK")),
    SE.FAMILY_HAIR: (
        ("texture", "_BaseMap", True, ""),
        ("shade_mask", "_MetallicGlossMap", True, "_METALLICSPECGLOSSMAP"),
        ("ramp", "_DiffRampMap", False, "_DIFF_RAMP_ON"),
        ("view_ramp", "_SpecRampMap", False, "_SPEC_RAMP_ON"),
        ("glass_ramp", "_ShadowLutTex", False, "_SHADOW_LUT_TEX"),
        ("gloss", "_LineMap", True, "_SPECULAR_LINE"),
        ("pattern", "_HairBrowMask", True, "_DRAW_UNDER_BROW"),
        ("detail_mask", "_OutlineMask", True, "_OUTLINE_MASK")),
    SE.FAMILY_EYE: (
        ("texture", "_BaseMap", True, ""),
        ("ramp", "_DiffRampMap", False, "_DIFF_RAMP_ON"),
        ("glass_ramp", "_ShadowLutTex", False, "_SHADOW_LUT_TEX"),
        ("gloss", "_MatcapTex", False, "_MATCAP_ON")),
    SE.FAMILY_OVERLAY: (("texture", "_BaseMap", True, ""),),
    SE.FAMILY_PRINCIPLED: (("texture", "_BaseMap", True, ""),),
}

#: Maps a shading term looks up — the diffuse ramp by N·L, the shadow LUT by the
#: base colour (``gl/shaders_endfield.py``) — rather than by the mesh's UV, so
#: they mean the same on a surface that has other UVs.
LOOKUP_MAPS = ("_DiffRampMap", "_ShadowLutTex")
#: … and the keywords whose branches read them
LOOKUP_KEYWORDS = ("_DIFF_RAMP_ON", "_SHADOW_LUT_TEX")

#: map property → the keyword whose branch reads it, where the property's
#: default texture turns that branch into a no-op
_NO_OP_WITHOUT = {"_EmissionMap": "_EMISSION", "_HairBrowMask": "_DRAW_UNDER_BROW",
                  "_EmotionMap": "_EMOTION_MAP", "_HighlightMap": "_HIGHLIGHT_MAP"}

#: ``UnityEngine.Rendering.BlendMode`` → GL.
_BLEND = {0: GL.GL_ZERO, 1: GL.GL_ONE, 2: GL.GL_DST_COLOR, 3: GL.GL_SRC_COLOR,
          4: GL.GL_ONE_MINUS_DST_COLOR, 5: GL.GL_SRC_ALPHA,
          6: GL.GL_ONE_MINUS_SRC_COLOR, 7: GL.GL_DST_ALPHA,
          8: GL.GL_ONE_MINUS_DST_ALPHA, 9: GL.GL_SRC_ALPHA_SATURATE,
          10: GL.GL_ONE_MINUS_SRC_ALPHA}
_ONE, _ZERO, _CULL_OFF, _TRANSPARENT = 1, 0, 0.0, 1.0
#: ``UnityEngine.Rendering.CompareFunction.Less``.
_LESS = 2.0
#: The overlay's forward pass: ``Blend Zero SrcColor`` — a multiply.
_OVERLAY_BLEND = (GL.GL_ZERO, GL.GL_SRC_COLOR)


def linear(color) -> tuple:
    """An authored (sRGB) colour as the linear one a shader receives; alpha as is."""
    def one(c: float) -> float:
        c = float(c)
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (*(one(c) for c in color[:3]), float(color[3]) if len(color) > 3 else 1.0)


#: Key of a shader record whose part carries a colour field per vertex
#: (``Mesh.vcolor``: linear rgb, and in alpha the base map's share).  Ours, not
#: a keyword of the game's: a reconstructed surface's colour that is not one
#: value — it runs on from the native skin's edge, or fades into its map.
VERTEX_FIELD = "_vertexField"


#: Key of a shader record whose part carries a tint per vertex (``Mesh.colors``)
#: that multiplies its base colour: a skin lent from elsewhere, matched to the
#: character's own along the seams.  Ours too.
VERTEX_TINT = "_vertexTint"


#: A cell of an emotion sheet is a blush when its overlay's colour is this much redder than it is
#: green or blue (0–1).  The sheets' other overlays are grey.
BLUSH_RED = 0.1


def blush_cell(pixels) -> int | None:
    """The ``_EmotionIndex`` of the reddest overlay on an emotion sheet, or None when none is a blush.

    The sheet is 2×2 overlays laid over the face's base colour by their
    alpha; the skin program reads cell ``index`` at column ``index % 2`` of
    row ``index // 2`` from the bottom (*pixels* runs top-down).  Which cell
    is the blush is not in the material, so it is read off the sheet: the
    overlay's colour is its pixels weighed by their alpha.
    """
    sheet = np.asarray(pixels, dtype=np.float64) / 255.0
    if sheet.ndim != 3 or sheet.shape[2] < 4:
        return None
    high, wide = sheet.shape[0] // 2, sheet.shape[1] // 2
    best, reddest = None, BLUSH_RED
    for index in range(4):
        cell = sheet[high * (1 - index // 2):high * (2 - index // 2), wide * (index % 2):wide * (index % 2 + 1)]
        cover = float(cell[..., 3].sum())
        if cover <= 0.0:
            continue
        colour = (cell[..., :3] * cell[..., 3:4]).sum(axis=(0, 1)) / cover
        red = float(colour[0] - max(colour[1], colour[2]))
        if red > reddest:
            best, reddest = index, red
    return best


FACE_PROPERTIES = {"_EmotionIndex": 2, "_EmotionBlend": 3}


def supports_face_parameters(material, values: dict) -> bool:
    """Whether the material actually draws all requested scalar face properties."""
    entries = getattr(material, "extra_uniforms", ())
    flags = next((args[0] for _, name, args in entries if name == "uEfFlags"), 0)
    return (not set(values).difference(FACE_PROPERTIES) and bool(flags & SE.FLAGS["_EMOTION_MAP"])
            and getattr(material, "pattern", None) is not None
            and any(name == "uEfSkin" for _, name, _ in entries))


def _emotion_source(material):
    """The authored defaults and measured blush cell, before either owner writes uniforms."""
    made = getattr(material, "emotion_made", None)
    if made is None:
        entries = getattr(material, "extra_uniforms", ())
        held = next((args for _kind, name, args in entries if name == "uEfSkin"), None)
        flags = next((args[0] for _kind, name, args in entries if name == "uEfFlags"), 0)
        cell = None
        if held is not None and flags & SE.FLAGS["_EMOTION_MAP"]:
            from ..origin.textures import pixels_of
            sheet = pixels_of(getattr(material, "pattern", None))
            cell = None if sheet is None else blush_cell(sheet)
        material.emotion_made = made = (held, cell)
    return made


def _write_emotion(material) -> None:
    held, cell = _emotion_source(material)
    now = list(held)
    for prop, value in getattr(material, "emotion_controls", {}).items():
        now[FACE_PROPERTIES[prop]] = value
    amount = getattr(material, "emotion_blush", 0.0)
    if amount > 0 and cell is not None:
        now[2:4] = [float(cell), float(amount)]
    entries = tuple((kind, name, tuple(now) if name == "uEfSkin" else args)
                    for kind, name, args in material.extra_uniforms)
    if entries != material.extra_uniforms:
        material.extra_uniforms = entries


def face_parameters(material, values: dict) -> bool:
    """Replace the manual scalar controls; H blush overlays them until it releases."""
    if not supports_face_parameters(material, values):
        return False
    converted = {prop: float(value) for prop, value in values.items()}
    if not all(np.isfinite(value) for value in converted.values()):
        raise ValueError("nonfinite face material control")
    _emotion_source(material)
    if getattr(material, "emotion_controls", {}) != converted:
        material.emotion_controls = converted
    _write_emotion(material)
    return True


def blush(material, amount: float) -> bool:
    """Overlay the measured blush; release restores manual controls or authored defaults."""
    held, cell = _emotion_source(material)
    if held is None or cell is None:
        return False
    if getattr(material, "emotion_blush", 0.0) != float(amount):
        material.emotion_blush = float(amount)
    _write_emotion(material)
    return True


def _flags(keywords) -> int:
    return sum(bit for name, bit in SE.FLAGS.items() if name in keywords)


def effective_keywords(shader: dict) -> set:
    """The material's keywords, less the ones an unbound map makes a no-op.

    A sampler nothing is bound to reads the property's declared default.  For
    the maps in :data:`_NO_OP_WITHOUT` that default (black, a flat normal, or
    white under a threshold) makes the keyword's whole branch do nothing, so
    dropping the keyword is exact — and it keeps this renderer's own stand-in
    for an unbound sampler, which is white, out of a term whose default is not.
    The pre-pass's brow cut belongs to the opaque fringe only.
    """
    keywords = set(shader["keywords"])
    for prop, keyword in _NO_OP_WITHOUT.items():
        if prop not in shader["maps"]:
            keywords.discard(keyword)
    if shader["_shader"].endswith("_Hair"):
        split = "_SPECULAR_NORMALMAP" in keywords
        if (split and "_SplitNormalMap" not in shader["maps"]) or (
                not split and "_BumpMap" not in shader["maps"]):
            keywords -= {"_SPECULAR_NORMALMAP", "_NORMALMAP"}
    elif "_BumpMap" not in shader["maps"]:
        keywords.discard("_NORMALMAP")
    if float(shader["floats"].get("_SurfaceType", 0.0)) == _TRANSPARENT:
        keywords.discard("_DRAW_UNDER_BROW")
    return keywords


def _uniforms(family: int, shader: dict) -> tuple:
    """The family's own uniforms: ``((kind, name, args), …)``."""
    f = lambda name, default=0.0: float(shader["floats"].get(name, default))
    colors = shader["colors"]
    plain = lambda name, default: tuple(colors.get(name, default))
    authored = lambda name, default: linear(colors.get(name, default))
    white, black = (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)
    out = {
        "uEfShadow": (f("_ShadowColorBrightness", 0.5), f("_ShadowColorSaturation", 1.0),
                      f("_BumpScale", 1.0), f("_AlphaPremultiply")),
        "uEfPbr": (f("_Metallic"), f("_Smoothness", 0.5), f("_Specular"),
                   f("_SpecRampIridescentMode")),
    }
    if family == SE.FAMILY_MAIN:
        emission = authored("_EmissionColor", black)
        out["uEfColorC"] = (*emission[:3], f("_EmissionBrightness", 1.0))
        out["uEfAniso"] = (f("_AnisotropyDirectionMain"), f("_AnisotropyDirectionAdditional"),
                           f("_AnisotropyOffsetAdditional"), f("_AnisotropyIntensityMultiplier", 1.0))
        out["uEfAnisoColor"] = (*authored("_AnisotropyColorAdditional", black)[:3],
                                f("_AnisotropyUseGeometryTangent"))
        out["uEfCoat"] = (f("_ClearCoatSmoothness", 1.0), f("_ClearCoatMetallic"), f("_ClearCoatNormalMode"), 0.0)
        out["uEfCoatColor"] = authored("_ClearCoatColor", white)
    elif family == SE.FAMILY_SKIN:
        out["uEfSkin"] = (f("_SkinRimOffScale"), f("_FaceRimOffScale"),
                          f("_EmotionIndex"), f("_EmotionBlend"))
        # the occlusion of skin drawn without a base map (the map's alpha is where the program
        # reads it): the native skin's own level for a flat reconstructed surface, 1 for anything
        # else.  (A colour field carries its own per vertex.)
        level = float((shader.get("reconstruction_skin_color") or {}).get("occlusion", 1.0))
        out["uEfMisc"] = (0.0, 0.0, 0.0, 1.0 if "_BaseMap" in shader["maps"] else level)
        out["uEfRimColor"] = authored("_SDFRimColor", white)
        out["uEfVec"] = plain("_HighlightMapVector", (0.0, 0.0, 0.0, 0.0))
    elif family == SE.FAMILY_EYE:
        out["uEfVec"] = (f("_ParallaxScale", 0.1), f("_MatcapNormalScale", 1.0), 0.0, 0.0)
        # [HDR] colours reach the shader as authored
        out["uEfColorA"] = plain("_EyeHighLightColor", (2.0, 2.0, 2.0, 1.0))
        out["uEfColorB"] = plain("_EyeScatteringColor", white)
        out["uEfColorC"] = plain("_MatcapColor", white)
    elif family == SE.FAMILY_HAIR:
        out["uEfColorA"] = authored("_AnisotropyColor2", black)
        out["uEfHairA"] = (f("_AnisotropyValue"), f("_AnisotropyValue2"),
                           f("_AnisotropyIntensity"), f("_AnisotropyEdgeFade"))
        out["uEfHairB"] = (f("_AnisotropyDirX"), f("_AnisotropyRange2"),
                           f("_SpecBumpScale", 1.0), f("_UseLineMap"))
        out["uEfHairC"] = (f("_LineAmount"), f("_LineValue"), f("_LineRange"),
                           f("_LineIntensity"))
        out["uEfHairD"] = plain("_LineMap_ST", (1.0, 1.0, 0.0, 0.0))
        out["uEfMisc"] = (f("_LineSaturation", 1.0), 0.0,
                          f("_HairBrowMaskThreshold", 0.5), 0.0)
    elif family == SE.FAMILY_OVERLAY:
        out["uEfMisc"] = (0.0, f("_UseGrayAsAlpha"), 0.0, 0.0)
    flags = _flags(effective_keywords(shader))
    if shader.get(VERTEX_FIELD):
        flags |= SE.VERTEX_FIELD_FLAG
    if shader.get(VERTEX_TINT):
        flags |= SE.VERTEX_TINT_FLAG
    entries = [("i", "uEfFlags", (flags,))]
    entries += [("v4", name, tuple(float(v) for v in value))
                for name, value in sorted(out.items())]
    return tuple(entries)


def _stencil(family: int, shader: dict, blended: bool):
    """The depth pre-pass's stencil state: ``(ref, compare, read mask, write mask)``.

    The pre-pass is where the game stamps what a pixel is — skin 36, an eye 36
    or 52 (``_PreZStencilRefOption``; bit 16 = "draw over hair"), hair
    ``_HairStencilRef`` tested ``GEqual`` on bit 16 — and its forward pass then
    draws only where its own depth won.  One forward pass in queue order with
    the same test and write gives the same picture.  A blended surface has no
    pre-pass and the overlay only reads.
    """
    floats = shader["floats"]
    if family == SE.FAMILY_OVERLAY:
        return (int(floats.get("_ShadowOverIris", 4.0)), "equal", 20, 0)
    if blended:
        return None
    if family == SE.FAMILY_SKIN:
        return (36, "always", 255, 255)
    if family == SE.FAMILY_HAIR:
        return (int(floats.get("_HairStencilRef", 36.0)), "gequal", 16, 239)
    return (int(floats.get("_PreZStencilRefOption", 36.0)), "always", 255, 255)


def base_map(shader: dict) -> str | None:
    """The property whose texture the piece's shader draws as its base colour.

    ``None`` for a shader with no counterpart here and for a material that
    binds nothing to that property.
    """
    family = SE.FAMILIES.get(shader["_shader"])
    for attr, prop, _repeat, _keyword in _MAPS.get(family, ()):
        if attr == "texture" and prop in shader["maps"]:
            return prop
    return None


#: The Principled render methods drawn with dithered alpha (Blender's ``HASHED``, EEVEE's "Dithered").
DITHERED = frozenset({"HASHED"})


def principled(shader: dict, texture) -> Material:
    """A converted source's Principled BSDF record (``converted.runtime_items.shader_of``) as a material.

    Its base map is the colour and, in alpha, the coverage; its constants are
    Metallic, Roughness and Specular IOR Level; its render method and culling
    are the source's.  It casts no outline: the source draws none.
    """
    floats = shader["floats"]
    out = Material(shader["_material"])
    out.toon_family = SE.FAMILY_PRINCIPLED
    entry = shader["maps"].get("_BaseMap")
    if entry is not None:
        out.texture = texture(entry, True)
    if "_BaseColor" in shader.get("colors", {}):
        out.base_color = tuple(float(v) for v in shader["colors"]["_BaseColor"])
    flags = SE.VERTEX_TINT_FLAG if shader.get(VERTEX_TINT) else 0
    keywords = set(shader.get("keywords", ()))
    # glTF's packing as the source wires it: a tangent-space normal map, and one image whose green is
    # the roughness and blue the metalness (``blender_source.describe_material``)
    for attr, prop, keyword in (("normal_map", "_BumpMap", "_NORMALMAP"),
                                ("shade_mask", "_MetallicGlossMap", "_METALLICSPECGLOSSMAP")):
        if (prop in shader["maps"]) != (keyword in keywords):
            raise ValueError(f"{prop} and its keyword {keyword} are not declared together")
        if keyword in keywords:
            setattr(out, attr, texture(shader["maps"][prop], True))
            flags |= SE.FLAGS[keyword]
    out.two_sided = bool(shader.get("two_sided"))
    out.blend = False
    out.depth_write = True
    out.render_queue = int(shader["_queue"])
    out.cast_outline = False
    # what the conversion declared the renderer to be (``part_roles.DECLARED_ROLE``), for the lists that
    # group parts (``controller.piece_group``): the program alone does not say it
    out.declared_role = shader.get(DECLARED_ROLE)
    # No coverage source (``alpha_channel`` None): Blender reads the Alpha constant 1, so nothing dithers.
    dithered = 1.0 if shader.get("blend_method") in DITHERED and shader.get("alpha_channel", "a") is not None else 0.0
    emotion = shader["maps"].get("_EmotionMap")
    emotion_uniforms = ()
    if emotion is not None:
        if "_EMOTION_MAP" not in shader.get("keywords", ()) or shader.get("emotion_uv") != 1:
            raise ValueError("transferred emotion atlas requires its declared secondary UV")
        out.pattern = texture(emotion, True)
        flags |= SE.FLAGS["_EMOTION_MAP"]
        emotion_uniforms = (("v4", "uEfSkin", (0.0, 0.0, float(floats["_EmotionIndex"]),
                                               float(floats["_EmotionBlend"]))),)
    # the normal map's Strength rides in the bump-scale slot every family reads it from
    normal_uniforms = (("v4", "uEfShadow", (0.0, 0.0, float(floats["_BumpScale"]), 0.0)),)         if flags & SE.FLAGS["_NORMALMAP"] else ()
    out.extra_uniforms = (("i", "uEfFlags", (flags,)),
                          ("v4", "uEfPbr", (float(floats.get("Metallic", 0.0)), float(floats.get("Roughness", 0.5)),
                                            float(floats.get("Specular IOR Level", 0.5)), dithered)),
                          *normal_uniforms, *emotion_uniforms)
    return out


#: A dithered material is drawn blended by its coverage when more than this share of its faces sit on
#: texels of partial alpha.  Blender converges dithered alpha over its samples to the coverage itself;
#: drawn once a frame it is a speckle (Lappland's blush: a pink grain over the cheeks).  Of her 29 dithered
#: parts, 28 have no face on partial alpha (0.00) and the expression overlay 0.57.
TRANSLUCENT_SHARE = 0.25
#: The render queue a coverage-blended material is drawn in: after every opaque one, sorted.
TRANSLUCENT_QUEUE = 3000


def partial_share(pixels: np.ndarray | None, uvs: np.ndarray, triangles: np.ndarray) -> float:
    """The share of *triangles* whose UV centre falls on a texel of partial alpha (0.02–0.98) in *pixels*."""
    if pixels is None or pixels.ndim != 3 or pixels.shape[2] < 4 or not len(triangles) or not len(uvs):
        return 0.0
    centre = np.asarray(uvs, dtype=np.float64)[triangles].mean(axis=1) % 1.0
    high, wide = pixels.shape[:2]
    alpha = pixels[np.clip(((1.0 - centre[:, 1]) * (high - 1)).astype(int), 0, high - 1),
                   np.clip((centre[:, 0] * (wide - 1)).astype(int), 0, wide - 1), 3] / 255.0
    return float(np.mean((alpha >= 0.02) & (alpha < 0.98)))


def coverage_blended(material, uvs: np.ndarray, triangles: np.ndarray) -> bool:
    """Draw a dithered Principled *material* blended by its coverage when its faces sit on partial alpha.

    True when it was switched (:data:`TRANSLUCENT_SHARE`): blended, no depth
    writes, drawn in :data:`TRANSLUCENT_QUEUE`, and the program told to keep
    the map's alpha instead of dithering it.
    """
    from ..origin.textures import pixels_of
    entries = list(getattr(material, "extra_uniforms", ()))
    pbr = next((index for index, (_kind, name, _args) in enumerate(entries) if name == "uEfPbr"), None)
    if pbr is None or entries[pbr][2][3] != 1.0:
        return False
    if partial_share(pixels_of(material.texture), uvs, triangles) <= TRANSLUCENT_SHARE:
        return False
    kind, name, args = entries[pbr]
    entries[pbr] = (kind, name, (args[0], args[1], args[2], 2.0))
    material.extra_uniforms = tuple(entries)
    material.blend = True
    material.depth_write = False
    material.render_queue = max(int(material.render_queue), TRANSLUCENT_QUEUE)
    return True


def material(shader: dict, texture) -> tuple[Material | None, str]:
    """``(material, note)`` for one converted piece's ``shader`` record.

    *texture* turns a ``maps`` entry into a texture-cache key, given the
    sampler's wrap: ``texture(entry, repeat)``.  ``material`` is ``None`` when
    the shader has no counterpart here; *note* then says which shader, and
    otherwise names the keywords drawn without (``""`` when none).
    """
    family = SE.FAMILIES.get(shader["_shader"])
    if family is None:
        return None, shader["_shader"]
    if family == SE.FAMILY_PRINCIPLED:
        return principled(shader, texture), ""
    declared = set(shader["keywords"])
    if family in NEEDS_RAMP and "_DIFF_RAMP_ON" not in declared:
        return None, f"{shader['_shader']} without _DIFF_RAMP_ON (not read)"
    keywords = effective_keywords(shader)
    floats, colors = shader["floats"], shader["colors"]
    out = Material(shader["_material"])
    out.toon_family = family
    for attr, prop, repeat, keyword in _MAPS[family]:
        entry = shader["maps"].get(prop)
        if entry is None or (keyword and keyword not in keywords):
            continue
        setattr(out, attr, texture(entry, repeat))
    if family == SE.FAMILY_HAIR and keywords & {"_SPECULAR_NORMALMAP", "_NORMALMAP"}:
        # one sampler for either: the split map when the shader reads two
        # normals out of it, the plain one otherwise
        source = "_SplitNormalMap" if "_SPECULAR_NORMALMAP" in keywords else "_BumpMap"
        out.normal_map = texture(shader["maps"][source], True)
    if "_BaseColor" in colors:
        out.base_color = tuple(colors["_BaseColor"])
    out.two_sided = floats.get("_Cull") == _CULL_OFF
    blend = (int(floats.get("_SrcBlend", _ONE)), int(floats.get("_DstBlend", _ZERO)))
    out.blend = blend != (_ONE, _ZERO) or floats.get("_SurfaceType") == _TRANSPARENT
    if family == SE.FAMILY_OVERLAY:
        out.blend, out.blend_func = True, _OVERLAY_BLEND
    elif out.blend and blend != (5, 10):
        out.blend_func = (_BLEND[blend[0]], _BLEND[blend[1]])
    out.depth_write = family != SE.FAMILY_OVERLAY and floats.get("_ZWrite", 1.0) == 1.0
    # ``ZTest Less`` on a blended surface: the blended copy of a mesh draws
    # only where its opaque copy did not (the fringe over the eyes)
    out.depth_less = out.blend and floats.get("_ZTest") == _LESS
    if "_ALPHATEST_ON" in keywords:
        out.alpha_test = float(floats.get("_AlphaClipThreshold", 0.0))
    out.render_queue = int(shader["_queue"])
    out.cast_outline = family in (SE.FAMILY_MAIN, SE.FAMILY_SKIN, SE.FAMILY_HAIR) \
        and floats.get("_EnableOutline", 0.0) == 1.0 and not out.blend
    out.outline_width = float(floats.get("_OutlineWidth", 0.0))
    # ``CharacterOutline``'s vertex program pushes the hull along TEXCOORD2's averaged normal when this is set
    out.outline_average = float(floats.get("_OutlineAverageNormal", 0.0)) > 0.5
    out.outline_tone = (float(floats.get("_OutlineColorBrightness", 1.0)),
                        float(floats.get("_OutlineColorSaturation", 1.0)),
                        float(floats.get("_OutlineOffsetZ", 0.0)))
    out.extra_uniforms = _uniforms(family, shader)
    out.stencil_state = _stencil(family, shader, out.blend)
    missing = sorted(declared - HANDLED[family])
    return out, ("without " + " ".join(missing)) if missing else ""
