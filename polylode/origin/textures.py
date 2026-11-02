"""Hold imported albedo maps and hand them to the GL texture cache.

The renderer resolves ``Material.texture`` through a factory registry keyed by
the first element of the key tuple, generating pixels lazily on first draw.
Imported textures are already decoded, so the factory here is just a lookup --
but it has to keep the arrays alive somewhere, and that store is what needs
bounding.  They are held as uint8 (a 1024² albedo is 4MB, not 16MB) and the
oldest are dropped once the store is full; a dropped texture falls back to
white rather than failing a frame.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager

import numpy as np

from ..core import swallowed

# The CPU-side keep-alive for composited maps.  Bounded by **bytes**, not by
# count: these are RGBA uint8 arrays and they are not the same size at all —
# the body albedo is 2048² (16 MB) while an eye highlight is 8² (256 B).  A cap
# of 256 entries meant anywhere between 64 KB and 4 GB depending on what a card
# happened to use, which is not a limit.  The share of the importer's own
# budget is deliberately small; the rest of that budget is for the bundles
# being decompressed, which is where the peaks are.
STORE_SHARE = 0.25
MIN_STORE_MB = 64.0
MAX_TEXTURES = 4096          # a backstop on entry count, not the real limit

_store: "OrderedDict[int, np.ndarray]" = OrderedDict()
_store_bytes: dict = {}
_total_bytes = [0]
_next_id = [1]


def store_budget_bytes() -> int:
    """How many bytes of composited maps to keep, for *this* machine."""
    try:
        from .memguard import budget_mb
        mb = max(MIN_STORE_MB, budget_mb() * STORE_SHARE)
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.textures.store_budget_bytes: from .memguard import budget_mb", exc)
        mb = MIN_STORE_MB
    return int(mb * 1024 * 1024)


_registered = [False]


def card_colors(colors) -> list[tuple]:
    """Normalise a card's ``colorInfo`` into plain RGB triples.

    Entries come through as dicts carrying ``baseColor`` (and pattern data we
    don't use yet), though some blocks store bare ``[r, g, b, a]`` lists.
    Treating the dict form as unusable is what left every garment white.
    """
    out = []
    for c in colors or []:
        if isinstance(c, dict):
            v = c.get("baseColor") or c.get("color")
        else:
            v = c
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            out.append(tuple(float(x) for x in v[:3]))
    return out


# ``hair_texture`` / ``_detail_shading`` / ``DETAIL_FLOOR`` / ``GLOSS_GAIN``
# used to live here and had already lost their last caller.  They read the hair
# ``_md`` map as an anisotropic strand highlight to *add*, normalised against
# its own 90th percentile and floored at 0.55 — every part of that invented.
#
# ``main_hair`` does something else entirely with the map, and none of it is a
# highlight::
#
#     d      = _DetailMask.Sample(uv * _DetailMask_ST)
#     shade  = max(shadowAtten, 1 - d.y)          // green: diffuse shadow
#     shade  = max(shade, 1 - _ShadowExtend)      // floor, straight off the material
#     rgb   *= lerp(_ShadowColor, 1, shade)
#     rim   *= (1 - d.y) * (1 - d.z) * _rimV      // blue gates the rim only
#
# The strand highlight comes from ``_HairGloss``, a separate map sampled twice
# a few texels apart and centred by ``_SpeclarHeight`` — not from ``_md``.
# Both of those are lighting, which this renderer does itself, so nothing here
# bakes them; :func:`figure.create_tex.hair` handles the part that *is* texture
# work (``_ColorMask`` picking between the card's three colours).


def card_patterns(colors) -> list[dict]:
    """``colorInfo`` entries that ask for a pattern, in colour-slot order."""
    out = []
    for c in colors or []:
        if not isinstance(c, dict):
            out.append({})
            continue
        pid = c.get("pattern")
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            pid = 0
        out.append({} if pid <= 0 else {
            "id": pid,
            "tiling": c.get("tiling") or [0.0, 0.0],
            "color": c.get("patternColor") or [1.0, 1.0, 1.0, 1.0],
        })
    return out


# ``_line_shading`` used to live here: ``1 - max(R, G, B)`` scaled by
# ``detailPower``, on the guess that the brightest channel was the line.  The
# real rule is in :func:`figure.body_head._skin_detail`, transcribed from
# ``main_skin`` — blue drives the shading, red the ink, and both are bounded by
# a floor the material supplies.  Nothing calls a second, differently-wrong
# version, so it is gone rather than deprecated.


# ``skin_texture`` used to live here: an approximation of CreateBodyTexture /
# CreateFaceTexture built from the card's colours and a weighted blend of the
# ``_mc`` channels.  It is gone because the real thing is now transcribed from
# the shaders themselves in :mod:`figure.create_tex` — keeping a second,
# differently-wrong version around only invites someone to call it.


# Every Illusion pass that clips ends in the same three instructions —
# ``add r, x, l(-0.5) / lt r, r, l(0) / discard_nz r`` — and what differs is
# ``x``.  In main_skin, main_opaque(_low, 2), main_alpha, main_emblem, the
# main_hair trio and toon_eyew_lod0 it is the alpha itself (or a ``min`` of
# alphas), so the albedo's alpha is cut at 0.5.  **main_item is not in that
# list** (§7.372): see :data:`CLIP_BY_SHADER`.  The material's ``_Cutoff``
# float is a ``[HideInInspector]`` Shader Forge leftover — none of these
# programs reads it.
#
# The assorted 0.15 / 0.20 / 0.30 / 0.35 / 0.40 thresholds this replaced were
# invented to hide artefacts; they cost real texels at every cut-out edge.
ALPHA_CLIP = 0.5

#: **Where the raw albedo alpha is cut, per shader** (§7.372), for every
#: discarding shader whose fragment program compares against a literal.
#: The main_item family doubles the alpha first — ``add_sat r.w, r.w, r.w``
#: straight after sampling ``_MainTex``, in the forward, OUTLINE and
#: shadow-caster programs alike — so ``saturate(2a) < 0.5`` is ``a < 0.25``.
#: Read with ``tools/origin_shader_dump.py --shader <name> --grep discard``.
#: A shader missing here clips at :data:`ALPHA_CLIP` unless its material's
#: ``_Cutoff`` says otherwise (``attach.apply_shader``).
CLIP_BY_SHADER: dict[str, float] = {
    "main_item": 0.25, "main_item_low": 0.25,
    "main_item_emission": 0.25, "main_item_studio": 0.25,
    "main_skin": ALPHA_CLIP,
    "main_opaque": ALPHA_CLIP, "main_opaque_low": ALPHA_CLIP,
    "main_opaque2": ALPHA_CLIP, "main_alpha": ALPHA_CLIP,
    "main_emblem": ALPHA_CLIP, "main_hair": ALPHA_CLIP,
    "main_hair_front": ALPHA_CLIP, "main_hair_low": ALPHA_CLIP,
    "toon_eyew_lod0": ALPHA_CLIP,
    # ``discard if 0.501·(a·_alpha) − 0.001 < 0``: only the fully clear texels
    # go, the rest is blended (:data:`CLIP_AND_BLEND`, §7.373)
    "main_item_studio_alpha": 0.001 / 0.501,
}

#: Shaders that discard **and** blend (§7.373).  ``main_item_studio_alpha``'s
#: forward pass is ``Blend SrcAlpha OneMinusSrcAlpha`` in the Transparent
#: queue with ``o.a = _MainTex.a × _alpha`` (``origin_shader_dump --state``), and
#: its discard only drops what is already invisible — so "it has a discard"
#: does not make it a cut-out surface, which is what :func:`shader_discards`
#: alone would say.
CLIP_AND_BLEND = frozenset({"main_item_studio_alpha"})

#: Blended shaders whose forward pass is ``ZWrite On`` (§7.376) — read with
#: ``origin_shader_dump --state``, not from the material's ``_ZWrite`` float,
#: which is an inspector value the pass may not read.  Every other blended
#: pass this build draws is ``ZWrite Off``: ``toon_eye_lod0``,
#: ``toon_eyew_lod0``, ``toon_nose_lod0`` (``chara/bo_head_00``).
BLEND_ZWRITE = frozenset({"main_item_studio_alpha", "toon_glasses_lod0"})


#: **What :func:`shader_clip` was asked and what it answered** (§7.342).
#:
#: The gate this function used to carry — "did the material name
#: ``_ALPHA_A_ON`` / ``_ALPHA_B_ON``" — turned out to be about two strings no
#: shader compiles against, so removing it changes the answer for every
#: material that does **not** name them.  How many that is, is a fact about
#: this install's cards, and it belongs next to the run that dresses them
#: rather than in a throwaway script (七): when this is a list ``origin_audit``
#: fills it and prints the split.
#:
#: Off by default; nothing on the drawing path turns it on.
CLIP_TAP: list | None = None

# Shaders whose fragment program has **no** discard at all: these blend
# (``_Mode 3``, SrcAlpha/OneMinusSrcAlpha, ZWrite off) instead of clipping, and
# their materials carry an ``_alphaTest`` float the shader never reads —
# ``cf_m_hitomi_00`` ships 0.5 and ``toon_eye_lod0`` discards nothing.
BLEND_SHADERS = frozenset({
    "toon_eye_lod0",        # iris
    "toon_nose_lod0",       # nose line
    "toon_glasses_lod0",    # tears, glasses
    "main_skin_low",
    "mnpb", "shadowcast", "Silhouette",
})


def unpack_normal(arr: np.ndarray | None) -> np.ndarray | None:
    """Illusion's ``_NormalMap`` → a conventional tangent-space normal.

    Unity's ``DXT5nm``: the DXT5 colour block carries **Y replicated into
    R = G = B** and the alpha block carries **X**, so the usual ``rgb*2-1``
    reads a grey image as a degenerate direction.  ``UnpackNormalDXT5nm`` is::

        x = a * 2 - 1
        y = g * 2 - 1
        z = sqrt(1 - x*x - y*y)

    Measured rather than assumed, on ``co_top_00``'s maps: R, G and B share a
    mean *and a standard deviation* to three decimals (0.498 / 0.078) while A
    differs (0.496 / 0.055), and ``x² + y²`` stays inside the unit disc for
    99.9–100% of texels when x comes from alpha versus 98.5–99.4% from red.
    The formats themselves are plain DXT5/DXT1/BC7, which UnityPy already
    decodes — the packing was the only obstacle.

    Returned in 0..1 so the shader's own ``rgb * 2 - 1`` recovers the vector.
    """
    if arr is None or arr.ndim != 3 or arr.shape[2] < 4:
        return None
    a = arr.astype(np.float32) / 255.0
    x = a[..., 3] * 2.0 - 1.0
    y = a[..., 1] * 2.0 - 1.0
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    out = np.empty(arr.shape[:2] + (4,), dtype=np.uint8)
    out[..., 0] = np.clip((x * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    out[..., 1] = np.clip((y * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    out[..., 2] = np.clip((z * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def is_packed_normal(arr) -> bool:
    """Is this a DXT5nm-packed map rather than a conventional one?

    The tell is that the three colour channels are the *same* image — a plain
    normal map has a blue channel that differs from red and green.
    """
    if arr is None or arr.ndim != 3 or arr.shape[2] < 4:
        return False
    a = arr[..., :3].astype(np.float32)
    same = (float(np.abs(a[..., 0] - a[..., 1]).mean()) < 1.5
            and float(np.abs(a[..., 1] - a[..., 2]).mean()) < 1.5)
    return same and float(arr[..., 3].std()) > 2.0


def shader_discards(shader: dict | None) -> bool:
    """Whether the *shader* ships a discarding pass at all.

    A fact about the asset, and a **different question** from
    :func:`shader_clip`, which asks whether *this material* switched that pass
    on.  They were one function until §7.90, and merging them was wrong in a way
    that only showed up on a helmet: the blend decision in
    :func:`figure.attach.apply_shader` reads "this shader has no discard, so it
    must be a blended pass", and once ``shader_clip`` started answering 0 for a
    material whose keywords are off, every such accessory turned into a
    transparent one.
    """
    name = str((shader or {}).get("_shader") or "").split("/")[-1]
    return bool(name) and name not in BLEND_SHADERS


def shader_clip(shader: dict | None) -> float:
    """``alpha_test`` for a material — the threshold, if this shader discards.

    The threshold is the shader's own (:data:`CLIP_BY_SHADER`): 0.5 for most,
    0.25 for the main_item family, which doubles the alpha before the test.

    ``main_opaque``'s cut-out, transcribed::

        a = min(max(1 - _alpha_a, _AlphaMask.r),
                max(1 - _alpha_b, _AlphaMask.g))
        a = min(a, _MainTex.a)                   ; the albedo's own alpha
        discard if a < 0.5

    **This used to gate on the material carrying ``_ALPHA_A_ON`` /
    ``_ALPHA_B_ON``, and those keywords do not exist** (§7.109 八, measured
    again in §7.342 with ``origin_shader_dump --keywords``, which reads each pass's
    own ``m_NameIndices`` and each fragment variant's ``m_KeywordIndices``):

    ===================  ========  ==============  ========================
    shader               variants  all discard?    names with ALPHA in them
    ===================  ========  ==============  ========================
    ``main_skin``        6         **yes 6/6**     the four uniforms only
    ``main_opaque``      6         **yes 6/6**     the four uniforms only
    ``main_opaque_low``  4         **yes 4/4**     the four uniforms only
    ``main_alpha``       6         **yes 6/6**     the four uniforms only
    ``main_emblem``      17        **yes 17/17**   **none at all**
    ``main_item``        6         **yes 6/6**     **none at all**
    ``main_skin_low``    4         **no 4/4**      **none at all**
    ===================  ========  ==============  ========================

    Every keyword any variant selects on is ``DIRECTIONAL`` / ``POINT`` /
    ``SPOT`` / ``SHADOWS_*``; ``_AlphaMask``, ``_AlphaMask_ST``, ``_alpha_a``
    and ``_alpha_b`` appear in those tables because the table also holds
    **uniform** names, and no sub-program's ``m_KeywordIndices`` points at one.
    So there is no such thing as "this material switched the cut-out on":
    a shader either ships a discarding program or it does not, and the two
    floats do their own job **inside** it — ``max(1 - _alpha_a, mask.r)`` is
    identically 1 when the float is 0, i.e. that half of the mask is off.
    That half is :func:`apply_alpha_mask`, which already takes both floats.

    Which leaves this the float form of :func:`shader_discards`.  Both stay,
    because the two call sites ask different questions and 七's rule is about
    two *implementations*, not about a one-line adapter: ``attach.apply_shader``
    asks "does this shader blend" (a fact about the asset) and this one answers
    "what threshold does this material draw with".
    """
    name = str((shader or {}).get("_shader") or "").split("/")[-1]
    out = CLIP_BY_SHADER.get(name, ALPHA_CLIP) if shader_discards(shader) else 0.0
    if CLIP_TAP is not None:
        sh = shader or {}
        kw = str(sh.get("_keywords") or "")
        CLIP_TAP.append((str(sh.get("_shader") or "").split("/")[-1],
                         ("_ALPHA_A_ON" in kw or "_ALPHA_B_ON" in kw),
                         float(out)))
    return out


def shader_blends(shader: dict | None) -> bool:
    """``_Mode 3`` with ZWrite off — Illusion's transparent material setup."""
    sh = shader or {}
    try:
        return float(sh.get("_Mode", 0.0)) >= 2.0 and \
            float(sh.get("_ZWrite", 1.0)) < 0.5
    except (TypeError, ValueError):
        return False


def apply_alpha_mask(arr: np.ndarray | None,
                     alpha_mask: np.ndarray | None,
                     alpha_a: float = 1.0,
                     alpha_b: float = 1.0) -> np.ndarray | None:
    """Multiply Illusion's ``_AlphaMask`` into a baked map's alpha.

    Origin keeps the cut-out shape out of the albedo so one texture can serve
    several garments, and it also uses this slot to let an outer layer punch a
    hole in what is underneath (``OverBodyMask`` / ``OverBraMask`` /
    ``OverInnerMask``).

    ``main_opaque`` and ``main_skin`` read it identically — transcribed from
    the bytecode::

        m = _AlphaMask.Sample(uv * _AlphaMask_ST)
        a = min( max(m.r, 1 - _alpha_a), max(m.g, 1 - _alpha_b) )
        clip( min(a, MainTex.a) - 0.5 )

    So the mask lives in **R and G**, each with its own enable knob, and the
    two combine by ``min``.  Reading the alpha channel instead — what this
    used to do, on the heuristic "use alpha when it has transparency" — cuts
    nothing at all on these ``_mab`` maps: ``cf_sailor01_body_mab``'s alpha
    runs 230…255 while the shape sits in RGB.
    """
    if arr is None or alpha_mask is None or arr.ndim != 3:
        return arr
    m = alpha_mask
    if m.ndim != 3 or m.shape[2] < 3:
        return arr
    h, w = arr.shape[:2]
    if m.shape[:2] != (h, w):
        yi = (np.arange(h) * m.shape[0] // h).clip(0, m.shape[0] - 1)
        xi = (np.arange(w) * m.shape[1] // w).clip(0, m.shape[1] - 1)
        m = m[yi][:, xi]
    r = np.maximum(m[..., 0].astype(np.float32) / 255.0, 1.0 - float(alpha_a))
    g = np.maximum(m[..., 1].astype(np.float32) / 255.0, 1.0 - float(alpha_b))
    keep = np.minimum(r, g)
    out = arr.copy()
    if out.shape[2] < 4:
        out = np.dstack([out, np.full((h, w), 255, np.uint8)])
    # ``min`` with what the albedo already says, exactly as the shader does;
    # the 0.5 clip is the material's own alpha test, which the caller sets.
    out[..., 3] = np.minimum(out[..., 3],
                             (keep * 255.0).astype(np.uint8))
    return out


def _sample_wrapped(img: np.ndarray, uv: np.ndarray, repeat: bool) -> np.ndarray:
    """Bilinear sample of *img* at Unity UVs ``(h, w, 2)`` → ``(h, w, C)`` float.

    Row 0 is v = 1 (the convention every map here is stored in, see
    :mod:`mesh.uvraster`), texel centres at half-integers, and the address mode
    applied per axis before the four taps are read — Repeat wraps the integer
    part away, Clamp holds the edge texel.
    """
    ih, iw = img.shape[:2]
    x = uv[..., 0] * iw - 0.5
    y = (1.0 - uv[..., 1]) * ih - 0.5
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)

    def addr(i, n):
        return np.mod(i, n) if repeat else np.clip(i, 0, n - 1)

    xa, xb = addr(x0, iw), addr(x0 + 1, iw)
    ya, yb = addr(y0, ih), addr(y0 + 1, ih)
    f = img.astype(np.float32)
    top = f[ya, xa] * (1 - fx) + f[ya, xb] * fx
    bot = f[yb, xa] * (1 - fx) + f[yb, xb] * fx
    return top * (1 - fy) + bot * fy


def apply_alpha_mask_uv(arr: np.ndarray | None,
                        alpha_mask: np.ndarray | None,
                        mask_uv: np.ndarray,
                        covered: np.ndarray,
                        alpha_a: float, alpha_b: float,
                        repeat: bool) -> np.ndarray | None:
    """:func:`apply_alpha_mask` when the mask is not laid out on the albedo's UV.

    ``main_opaque`` reads ``_AlphaMask`` at ``uv * _AlphaMask_ST`` and
    ``main_opaque2`` at ``lerp(uv0, uv1, _AlphaMaskuv) * _AlphaMask_ST`` (the
    only difference between the two programs, §7.367), and
    ``ChangeClothesBraAsync`` gives ``OverBraMask`` an ST of ``(1, 2, 0, -1)``
    on some bras — so the texel the fragment reads is **not** the texel of the
    albedo it is drawn over.  *mask_uv* is that UV per albedo texel (rasterised
    through the mesh, :func:`mesh.uvraster.rasterise`), *covered* which texels
    it means anything for; the rest keep their alpha.  The channel rule is the
    shader's, unchanged: ``min(max(R, 1 - a), max(G, 1 - b))`` then ``min``
    with the albedo's own alpha.
    """
    if arr is None or alpha_mask is None or arr.ndim != 3:
        return arr
    if alpha_mask.ndim != 3 or alpha_mask.shape[2] < 3:
        return arr
    h, w = arr.shape[:2]
    if mask_uv.shape[:2] != (h, w) or covered.shape != (h, w):
        raise ValueError("mask_uv / covered must be on the albedo's texel grid")
    m = _sample_wrapped(alpha_mask[..., :2], mask_uv, repeat) / 255.0
    r = np.maximum(m[..., 0], 1.0 - float(alpha_a))
    g = np.maximum(m[..., 1], 1.0 - float(alpha_b))
    keep = np.minimum(r, g)
    out = arr.copy()
    if out.shape[2] < 4:
        out = np.dstack([out, np.full((h, w), 255, np.uint8)])
    cut = np.minimum(out[..., 3].astype(np.float32),
                     np.clip(keep * 255.0 + 0.5, 0, 255))
    out[..., 3] = np.where(covered, cut, out[..., 3]).astype(np.uint8)
    return out


def pixels_of(key) -> np.ndarray | None:
    """The array a :func:`store` key holds, or ``None`` once it is evicted."""
    tid = _key_id(key)
    return None if tid is None else _store.get(tid)


def apply_hair_alpha_mask(arr: np.ndarray | None,
                          alpha_mask: np.ndarray | None) -> np.ndarray | None:
    """``main_hair``'s cut-out, which is **not** the garment one.

    Its forward base, transcribed (``bo_hair_b_00`` / ``main_hair``, ps 8)::

        r0 = _AlphaMask.Sample(uv * _AlphaMask_ST)
        r1 = _MainTex.Sample(uv * _MainTex_ST)
        r0.y = r0.x * r1.w - 0.5
        discard_nz (r0.y < 0)

    Two things follow, and both were got wrong before:

    * it reads ``_AlphaMask.**r**`` alone.  The ``min(max(r, 1 - _alpha_a),
      max(g, 1 - _alpha_b))`` form is ``main_opaque`` / ``main_skin``, and
      applying it here clips on a channel this shader never samples.
    * ``_MainTex`` is whatever ``create_hair`` produced, and that shader's last
      instruction is ``mov o0.w, l(1.000000)`` — the strand sheet's own alpha
      is discarded.  With ``_AlphaMask`` unbound (its property default is
      **white**) a hair card is therefore **solid**, and the gaps between
      strands are geometry (§7.57).

    Feeding the colour mask's alpha in here instead, on "it has transparency so
    it must be a cut-out", deleted 93.6% of ``ロングヘア`` and **all** of
    ``ほそロング`` — whose map is alpha 0 everywhere.
    """
    if arr is None or alpha_mask is None or arr.ndim != 3:
        return arr
    m = alpha_mask
    if m.ndim != 3 or m.shape[2] < 1:
        return arr
    h, w = arr.shape[:2]
    if m.shape[:2] != (h, w):
        yi = (np.arange(h) * m.shape[0] // h).clip(0, m.shape[0] - 1)
        xi = (np.arange(w) * m.shape[1] // w).clip(0, m.shape[1] - 1)
        m = m[yi][:, xi]
    out = arr.copy()
    if out.shape[2] < 4:
        out = np.dstack([out, np.full((h, w), 255, np.uint8)])
    # ``r0.x * r1.w``: the product, as written.  ``create_hair``'s alpha is 1,
    # where the product and a ``min`` agree; a ``_MainTex`` MaterialEditor put
    # there carries its own alpha, and there they do not (§7.382).
    prod = out[..., 3].astype(np.uint32) * m[..., 0].astype(np.uint32)
    out[..., 3] = ((prod + 127) // 255).astype(np.uint8)
    return out


def needs_cutout(arr: np.ndarray | None, threshold: float = 0.08,
                 alpha_level: int = 200) -> bool:
    """Does this albedo carry real transparency?

    Illusion puts lace holes, fishnet mesh and lash shapes in the alpha channel
    rather than in geometry.  Drawn opaque those become solid blocks.

    Thresholds are deliberately loose: many solid garments ship with a few
    near-opaque edge pixels (alpha 240–254).  Treating those as cutout punched
    holes through jackets and boots (the "invisible top / black boot blotch"
    failure mode).
    """
    if arr is None or arr.ndim != 3 or arr.shape[2] < 4:
        return False
    return float((arr[..., 3] < alpha_level).mean()) > threshold


def store(arr: np.ndarray | None, *, srgb: bool,
          repeat: bool = False) -> tuple | None:
    """Keep an RGBA uint8 array and return the material key for it.

    *srgb* is the bytes' own colour space and has **no default**, for the same
    reason :func:`colorspace.decode` has none: on this install the ``*_mc``
    colour masks split 60/40 and the ``*_md`` detail maps 38/62, so a default
    would be a coin toss on a third of the library dressed up as a convention
    (CLAUDE.md 一之一).  It rides in the returned key because the key is the
    only thing that reaches :func:`gl.textures.resolve_texture`.

    Two kinds of caller and they answer differently:

    * a map that came out of a bundle passes the asset's ``m_ColorSpace``
      (``colorspace.Map.srgb`` — never re-derived from the field name);
    * a map **this project composited** passes the space it wrote.  The six
      create passes end at an sRGB render target (§7.85 二), so their output is
      ``srgb=True``; a mask we packed ourselves out of decoded inputs is data
      and is ``srgb=False``.

    *repeat* picks the wrap mode, which is a property of the source texture
    (``m_TextureSettings.m_WrapMode``) and not a rendering preference: the hair
    gloss sheets ship Repeat and are sampled well outside 0..1, while every
    ramp and colour mask ships Clamp and would tile a seam across itself.
    """
    if arr is None or arr.size == 0:
        return None
    _ensure_factory()
    tid = _next_id[0]
    _next_id[0] += 1
    data = np.ascontiguousarray(arr, dtype=np.uint8)
    _store[tid] = data
    _store_bytes[tid] = int(data.nbytes)
    _total_bytes[0] += _store_bytes[tid]
    _evict_to_budget()
    return ("originasset_repeat" if repeat else "originasset", tid, 1 if srgb else 0)


def _key_id(key) -> int | None:
    if not isinstance(key, (tuple, list)) or len(key) < 2:
        return None
    try:
        return int(key[1])
    except (TypeError, ValueError):
        return None


def alive(key) -> bool:
    """Is this key still backed by an array, or has the budget dropped it?"""
    tid = _key_id(key)
    return tid is not None and tid in _store


#: Maps dropped from the CPU store that something still needed.  Since §7.499
#: the store only lets go of a map no live :class:`~polylode.mesh.mesh.Model`
#: references, so this stays 0; it is kept because runs whose result depends on
#: the store (``bake_verify`` hashing every part's albedo) check that it is 0.
_evicted = [0]
#: Maps released because nothing referenced them any more (a character that
#: was replaced or went out of scope) — the store working, not a failure.
_released = [0]
#: Keys asked for after their map was gone: each one drew as a white sheet.
_unbacked = [0]
#: Ids at or above this were stored since the last :func:`collect` — the build
#: in progress, whose maps are not on a material yet and must not be dropped.
_gen_start = [1]


def evicted_count() -> int:
    """How many maps were dropped while something still needed them (0 since §7.499)."""
    return int(_evicted[0])


def released_count() -> int:
    """How many unreferenced maps the store has let go of in this process."""
    return int(_released[0])


def unbacked_count() -> int:
    """How many draws asked for a map the store no longer had (each drew white)."""
    return int(_unbacked[0])


def _keys_in(value, out: set, depth: int = 0) -> None:
    if isinstance(value, tuple) and len(value) >= 2 and value[0] in ("originasset", "originasset_repeat"):
        tid = _key_id(value)
        if tid is not None:
            out.add(tid)
        return
    if depth >= 2:
        return
    if isinstance(value, (list, tuple, set)):
        for v in value:
            _keys_in(v, out, depth + 1)
    elif isinstance(value, dict):
        for v in value.values():
            _keys_in(v, out, depth + 1)


def live_ids() -> set:
    """Store ids some live model's part or material still points at."""
    from ..mesh.mesh import LIVE_MODELS
    out: set = set()
    for model in list(LIVE_MODELS):
        for part in list(getattr(model, "parts", ()) or ()):
            for obj in (part, getattr(part, "material", None)):
                if obj is not None and hasattr(obj, "__dict__"):
                    for v in vars(obj).values():
                        _keys_in(v, out)
    return out


def _release(tid: int) -> None:
    _store.pop(tid, None)
    _total_bytes[0] -= _store_bytes.pop(tid, 0)
    _released[0] += 1


def collect() -> int:
    """Let go of every map no live model references; returns how many.

    Called where no build is half done (the start of ``dress_character``), so
    every map stored before this point is either on a live material or
    garbage.  It also closes the current generation: what is stored after it
    is protected until the next call.
    """
    import gc
    gc.collect()
    live = live_ids()
    n = 0
    for tid in list(_store):
        if tid not in live:
            _release(tid)
            n += 1
    _gen_start[0] = _next_id[0]
    return n


_build_depth = [0]


@contextmanager
def building():
    """Bracket one character build; the outermost one starts with :func:`collect`.

    The dress runs on the GUI thread and pumps events, so a second dress can
    start inside the first; tidying there would release maps the outer build
    stored and has not attached yet.  Only the outermost build tidies.
    """
    if _build_depth[0] == 0:
        collect()
    _build_depth[0] += 1
    try:
        yield
    finally:
        _build_depth[0] -= 1


def _evict_to_budget() -> None:
    """Over budget, release **unreferenced** maps from earlier builds, LRU first.

    Until §7.499 this dropped the least recently used map whatever it was, and
    the budget follows the machine's free memory: with 250 MB free it bottoms
    out at 96 MB, less than one dressed character, so Free H painted live
    characters white (the 4×4 sheet :func:`_make` returned for a dropped key).
    A map a live model draws, or one the build in progress stored and has not
    attached yet, is never a candidate; being over budget with nothing
    releasable just means the characters on screen need that much.
    """
    budget = store_budget_bytes()
    if _total_bytes[0] <= budget and len(_store) <= MAX_TEXTURES:
        return
    live = live_ids()
    for tid in list(_store):
        if _total_bytes[0] <= budget and len(_store) <= MAX_TEXTURES:
            break
        if tid in live or tid >= _gen_start[0]:
            continue
        _release(tid)


def _make(tid: int, srgb: int = 0) -> np.ndarray:
    # ``srgb`` is consumed by the uploader (it picks the internal format), not
    # here — the bytes are handed over untouched either way.  It is still in the
    # signature because ``resolve_texture`` splats the whole key.
    arr = _store.get(int(tid))
    if arr is None:
        # Something drew a key whose map is gone: that part comes out white.
        # Counted and said once, never silent (一之一).
        _unbacked[0] += 1
        if _unbacked[0] == 1:
            import sys
            print(f"[textures] map {tid} was asked for after it was released; drawn white",
                  file=sys.stderr)
        return np.ones((4, 4, 4), dtype=np.float32)
    _store.move_to_end(int(tid))
    # The stored bytes go to the uploader as they are.  Converting to float
    # and back cost ~1.5 s of GL-thread time on the first frame of a dressed
    # character (§7.526) and changes no value: v/255·255 round-trips all 256
    # bytes exactly in float32.
    return arr


def _ensure_factory() -> None:
    if _registered[0]:
        return
    from ..gl.textures import TEXTURE_FACTORY
    TEXTURE_FACTORY["originasset"] = _make
    TEXTURE_FACTORY["originasset_repeat"] = _make
    _registered[0] = True


def clear() -> None:
    _store.clear()
    _store_bytes.clear()
    _total_bytes[0] = 0


def stats() -> dict:
    return {"count": len(_store),
            "mb": _total_bytes[0] / (1024 * 1024),
            "budget_mb": store_budget_bytes() / (1024 * 1024)}
