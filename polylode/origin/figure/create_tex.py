"""Illusion's ``*_create`` compositing shaders, transcribed from their bytecode.

``ChaControl`` does not hand a finished texture to a material.  It renders one
with a dedicated shader — ``cf_m_eye_create`` / ``cf_m_eyewhite_create`` /
``cf_m_face_create`` (``create_head``) / ``cf_m_body_create`` (``create_body``),
all in ``chara/mm_base.unity3d`` — and the parameters it feeds them are the
card's colours and ids.

Those shaders ship as DXBC, which UnityPy prints as "disassembly not
supported".  Windows disassembles it perfectly well: ``d3dcompiler_47.dll``
exports ``D3DDisassemble``, and the ``$Globals`` block in the same blob names
every constant register.  The functions here are line-by-line transcriptions
of that output, so the compositing rules below are **read**, not inferred:

* the eye's iris map is the *blend layer* of a Multiply↔VividLight pair whose
  base is the gradient colour — it is not a multiplier and not a mix factor;
* the face's colour mask uses red and green as two independent tints that are
  multiplied together, and **blue as a floor**, not as a third colour;
* the body's base skin tone comes out of a lerp driven by the *sunburn* map,
  and its mask's blue channel multiplies in the nail colour at the very end;
* every decal layer is ``lerp(under, tex.rgb * colour, tex.a * colour.a)`` —
  the map's own RGB participates, it is not just a stencil.

Everything works in top-down arrays (row 0 = v 1), the project convention.

**These passes run in linear.**  Koikatsu is a Linear project, so the GPU hands
each of these shaders a decoded texel — sRGB-flagged maps through the hardware's
conversion, linear-flagged ones raw — and Unity converts every ``Color``
property on the way in.  The composite is then written back through an sRGB
render target, which is why the bytes this module returns are still sRGB and
everything downstream of it is unaffected.  :mod:`colorspace` has the evidence
for all three halves of that sentence.  :func:`shade_mask` at the bottom of
this file is now part of it too: the forward passes turned out to have their own
encode boundary — the camera's own sRGB render target — so its inputs are
decoded like everything else and what it packs is a linear coefficient
(§7.126).  :func:`detail_map` is the one exception, and for the opposite
reason: it hands the map along rather than reading a value out of it.
"""
from __future__ import annotations

import numpy as np

from .. import colorspace as CS
from .. import progress as PROG

TAU = 6.283185307179586


# ---------------------------------------------------------------- sampling
def _grid(shape):
    h, w = shape
    v = 1.0 - (np.arange(h, dtype=np.float32) + 0.5) / h
    u = (np.arange(w, dtype=np.float32) + 0.5) / w
    return np.broadcast_to(u[None, :], (h, w)), np.broadcast_to(v[:, None], (h, w))


#: Bound the sampler's scratch arrays without changing its output resolution or arithmetic (§7.1352).
_GATHER_TILE_SIDE = 256


def _gather(t, u, v, wrap: bool = False):
    """The same bilinear fetch in bounded 2D tiles; one full result, no full-image tap temporaries."""
    u, v = np.broadcast_arrays(u, v)
    side = _GATHER_TILE_SIDE
    if u.ndim != 2 or (u.shape[0] <= side and u.shape[1] <= side):
        return _gather_tile(t, u, v, wrap)
    out = None
    for y in range(0, u.shape[0], side):
        for x in range(0, u.shape[1], side):
            rows, columns = slice(y, y + side), slice(x, x + side)
            part = _gather_tile(t, u[rows, columns], v[rows, columns], wrap)
            if out is None:
                out = np.empty((*u.shape, part.shape[-1]), dtype=part.dtype)
            out[rows, columns] = part
    return out


def _gather_tile(t, u, v, wrap: bool = False):
    """Bilinear over an already-decoded float image.

    Bilinear because that is what the maps ask for (``m_FilterMode = 1``), and
    it matters here: several of these transforms magnify, and point sampling
    turns the iris's shading ramp into visible steps.

    Clamp, not repeat: every one of these maps ships with
    ``m_TextureSettings.m_WrapMode = 1`` (mt_eye_hi_up, mt_eye_gradation,
    mt_cheek, mt_lipline, mt_face_paint, mt_mole — 100% clamp; mt_eye 146 of
    178).  It matters because several of the transforms deliberately leave the
    unit square — ``_tex5uv`` scales v by 21 — and repeating there tiles a
    second copy of the iris across the eye.
    """
    # The skin composite is the longest unbroken stretch of a card load: 16 of
    # these add up to 4.6 s of the body stage's 6.8 s, with nothing in between
    # for the loading dialog to repaint on.  Ticking per call puts a pump every
    # ~0.3 s; ``progress.tick`` rate-limits itself, so this costs a clock read.
    PROG.tick()
    h, w = t.shape[:2]
    x = u * w - 0.5
    y = (1.0 - v) * h - 0.5
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0)[..., None].astype(np.float32)
    fy = (y - y0)[..., None].astype(np.float32)

    def idx(a, n):
        a = a.astype(np.int64)
        return np.mod(a, n) if wrap else np.clip(a, 0, n - 1)

    x0i, x1i = idx(x0, w), idx(x0 + 1, w)
    y0i, y1i = idx(y0, h), idx(y0 + 1, h)
    top = t[y0i, x0i] * (1.0 - fx) + t[y0i, x1i] * fx
    bot = t[y1i, x0i] * (1.0 - fx) + t[y1i, x1i] * fx
    out = top * (1.0 - fy) + bot * fy
    if out.shape[-1] == 3:
        out = np.dstack([out, np.ones(out.shape[:2], np.float32)])
    return out


def sample(tex, u, v, wrap: bool = False):
    """``tex.Sample(s, float2(u, v))`` for a create pass — linear RGBA.

    *tex* is a :class:`colorspace.Map`.  The decode happens **before** the
    filter, which is the order the hardware uses: an sRGB texture fetch converts
    each texel and blends the four in linear.  Filtering the bytes and
    converting the result is a different number wherever the four texels differ,
    which is every edge in the map.
    """
    if tex is None:
        return None
    return _gather(CS.decode(tex.px, tex.srgb), u, v, wrap)


def sample_raw(px, u, v, wrap: bool = False):
    """The same fetch with **no** colour conversion.

    One caller left, and it is not a shortcut: :func:`detail_map` re-samples
    ``_DetailMask`` through its own ``_ST`` and hands the **map itself** to the
    GPU, which then decodes it by the asset's own ``m_ColorSpace``.  Decoding
    here as well would apply the curve twice.  Everything that consumes a
    *value* rather than passing the map along uses :func:`sample` (§7.126).
    """
    if px is None:
        return None
    return _gather(px.astype(np.float32) / 255.0, u, v, wrap)


def st(uv, tex_st):
    """``uv * _Tex_ST.xy + _Tex_ST.zw``."""
    u, v = uv
    if tex_st is None:
        return u, v
    sx, sy, ox, oy = tex_st
    return u * sx + ox, v * sy + oy


def _identity(tex_st) -> bool:
    if tex_st is None:
        return True
    sx, sy, ox, oy = tex_st
    return (abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6
            and abs(ox) < 1e-6 and abs(oy) < 1e-6)


def fetch(tex, uv, tex_st, shape):
    """``sample`` with the common case short-circuited.

    The base albedo and colour mask are read at ``_Tex_ST`` = (1,1,0,0) over a
    grid that is the texture's own — a 2048² body map costs four gathers and
    twelve multiplies per texel to arrive back at itself.  Detecting that is
    worth about three seconds a card.
    """
    if tex is None:
        return None
    if _identity(tex_st) and tex.px.shape[:2] == tuple(shape):
        return CS.decode(tex.px, tex.srgb)
    return sample(tex, *st(uv, tex_st))


def fetch_raw(px, uv, tex_st, shape):
    """:func:`fetch` without the colour conversion — see :func:`sample_raw`.

    Note the resample here is on **bytes**, so it filters before the decode
    where the hardware would filter after.  It is identity for most
    ``_DetailMask_ST`` — 79 of the 5,777 baked materials carry another one
    (§7.383, mostly hair), and those are off by a fraction of a level at
    edges.
    """
    if px is None:
        return None
    if _identity(tex_st) and px.shape[:2] == tuple(shape):
        out = px.astype(np.float32) / 255.0
        if out.shape[-1] == 3:
            out = np.dstack([out, np.ones(out.shape[:2], np.float32)])
        return out
    return sample_raw(px, *st(uv, tex_st))


def layout(uv, vec, tex_st=None, rotate: bool = True):
    """The ``(u/v/r/s)`` decal placement ``_paint1`` / ``_paint2`` / ``_hokuro``.

    From the bytecode::

        k   = 2 * (1 - vec.w)
        p   = (uv + vec.xy) * 2k + (0.5 - k)     # scale about the centre
        p  -= 0.5 ; p = rot(p, vec.z * 2pi) ; p += 0.5
        p   = p * _Tex_ST.xy + _Tex_ST.zw

    ``_hokuro`` takes the same path **without** the rotate — the mole's z is
    read into the vector but never used, so it is skipped here too.
    """
    u, v = uv
    ox, oy, rot, w = (float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3]))
    k = 2.0 * (1.0 - w)
    scale = k + k
    bias = 0.5 - k
    pu = (u + ox) * scale + bias
    pv = (v + oy) * scale + bias
    if rotate:
        pu, pv = pu - 0.5, pv - 0.5
        s, c = np.sin(rot * TAU), np.cos(rot * TAU)
        pu, pv = pu * c + pv * s, -pu * s + pv * c
        pu, pv = pu + 0.5, pv + 0.5
    return st((pu, pv), tex_st)


def uv_vector(uv, vec, tex_st=None):
    """``_tex4uv`` / ``_tex5uv`` placement (cheek, lip line).

    From the bytecode::

        f = 1 - vec.zw
        p = (uv + vec.xy) * f - f + 0.5

    These two are never written by ``ChaControl``; the values that matter are
    the ones saved on ``cf_m_face_create`` itself — (0.5, 0.64, -0.33, -4.0)
    and (0.5, 0.745, -8, -20).  Reading them off the shader's *property
    default* instead gives f = 0 and the whole layer collapses to one texel.
    """
    u, v = uv
    ox, oy, sx, sy = (float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3]))
    fu, fv = 1.0 - sx, 1.0 - sy
    return st(((u + ox) * fu - fu + 0.5, (v + oy) * fv - fv + 0.5), tex_st)


def _rgba(colour, fallback=(1.0, 1.0, 1.0, 1.0)):
    """A card / material colour as the **GPU** sees it: RGB linear, alpha raw.

    Every colour these six shaders read is declared ``Color`` in the shader's
    property table and none carries ``[Gamma]``, so in a linear project Unity
    converts it on the way to the constant buffer.  ``ChaControl`` does no
    conversion of its own — ``Mathf.GammaToLinearSpace`` appears once in the
    whole of ``Assembly-CSharp`` and it is in ``BloomSettings``.
    """
    if isinstance(colour, (list, tuple, np.ndarray)) and len(colour) >= 3:
        a = float(colour[3]) if len(colour) >= 4 else 1.0
        return CS.colour_to_linear((float(colour[0]), float(colour[1]),
                                    float(colour[2]), a))
    return CS.colour_to_linear(fallback)


def _is_rgb(colour) -> bool:
    """Whether *colour* is a value :func:`_rgba` will use rather than replace."""
    return (isinstance(colour, (list, tuple, np.ndarray))
            and len(colour) >= 3)


def rgba_raw(colour, fallback=(1.0, 1.0, 1.0, 1.0)):
    """A colour left in the space it was authored in — forward-pass callers.

    The counterpart to :func:`_rgba`.  ``main_skin``'s ``_overcolorN`` decals and
    the iris draw material's highlights are composited by this project into an
    sRGB albedo, so converting the colour while its use stays in sRGB would be
    half a change.  Both halves are named so neither can be picked by accident.
    """
    if isinstance(colour, (list, tuple, np.ndarray)) and len(colour) >= 3:
        a = float(colour[3]) if len(colour) >= 4 else 1.0
        return np.asarray([float(colour[0]), float(colour[1]),
                           float(colour[2]), a], np.float32)
    return np.asarray(fallback, np.float32)


def _over(base, layer_rgb, alpha):
    """``lerp(base, layer, alpha)`` — every decal in these shaders."""
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    return base + (layer_rgb - base) * a


def _u8(rgb, alpha):
    """The pass's linear result as the bytes ``createTex`` holds.

    ``CustomTextureCreate`` blits into ``new RenderTexture(w, h, 0, ARGB32)``
    with ``GL.sRGBWrite`` on, so the store is a linear→sRGB encode into unorm:
    the clamp is the render target's (which is what lets ``create_eye``
    extrapolate past 1 and simply be cut off), and the curve is the sRGB one.
    Alpha is written straight through — the conversion has always been RGB-only.
    """
    out = np.empty(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = CS.encode(rgb)
    out[..., 3] = (np.clip(alpha, 0.0, 1.0) * 255.0 if np.ndim(alpha)
                   else round(float(np.clip(alpha, 0.0, 1.0)) * 255))
    return out


# ------------------------------------------------------------------- eyes
def eyewhite(main, color, color2, shape=None):
    """``create_eyewhite``::

        o.rgb = lerp(_Color2, _Color, MainTex.r)
        o.a   = 1
    """
    c1, c2 = _rgba(color, (0.96,) * 3 + (1.0,)), _rgba(color2)
    if main is None:
        h, w = shape or (64, 64)
        return _u8(np.broadcast_to(c1[:3], (h, w, 3)), np.ones((h, w), np.float32))
    h, w = main.px.shape[:2]
    uv = _grid((h, w))
    m = fetch(main, uv, None, (h, w))
    t = m[..., 0:1]
    return _u8(c2[:3] + (c1[:3] - c2[:3]) * t, np.ones((h, w), np.float32))


def _vivid_light(base, blend):
    """Photoshop Vivid Light, exactly as the bytecode branches it."""
    burn = 1.0 - (1.0 - base) / np.maximum(2.0 * blend, 1e-6)
    dodge = base / np.maximum(2.0 * (1.0 - blend), 1e-6)
    return np.clip(np.where(blend > 0.5, dodge, burn), 0.0, 1.0)


def eye(main, colormask, color, color2, blend, grad,
        main_st=None, mask_st=None, uv=None):
    """``create_eye``::

        g   = _ColorMask.Sample(((uv + _grad.xy) * 2*(1-_grad.w)
                                 + (_grad.w - 0.5)) * _ColorMask_ST)
        tint= lerp(_Color2, _Color, g.r)
        m   = _MainTex.Sample(uv * _MainTex_ST)
        o.rgb = _Blend * (VividLight(tint, m.r) - tint * m.r) + tint * m.r
        o.a   = m.a * _Color.a

    That last line is written as a ``mad``, **not** a saturating lerp, and
    *blend* is whatever the card holds -- ``_Blend`` is declared ``Range(0, 1)``
    but ``ChangeSettingEye`` calls ``SetFloat`` with the raw ``gradBlend`` and a
    Range only ever bound the inspector slider.  Above 1 the result extrapolates
    past the vivid layer; the only clamps in the pass are the ``movc_sat`` on
    VividLight itself and the ARGB32 render target on the way out (:func:`_u8`).

    Note the gradient's transform is **not** the identity at ``_grad = 0``:
    it is ``uv * 2 - 0.5``, a 2× zoom about the centre.  The card's
    ``gradScale`` widens the ramp by shrinking that factor toward 0.

    *uv* is the surface coordinate the pass runs over.  It exists because the
    ST ``EyeLookMaterialControll`` writes lives on the **draw** material, not
    on this one: ``ReSetupMaterial`` assigns ``_material = _renderer.material``
    and ``Update`` writes ``SetTextureScale(texID, 1 + scale)`` there, while
    ``cf_m_eye_create`` ships ``_MainTex_ST`` = identity.  So the window has to
    move the gradient mask as well as the iris map: baking it into ``main_st``
    alone leaves the colour ramp behind while the iris slides under it.
    """
    if main is None:
        return None
    h, w = main.px.shape[:2]
    if uv is None:                        # the identity short-circuit in
        uv = _grid((h, w))                # ``fetch`` only holds for that grid
        m = fetch(main, uv, main_st, (h, w))
    else:
        m = sample(main, *st(uv, main_st))
    c1, c2 = _rgba(color), _rgba(color2)

    if colormask is not None:
        gx, gy, _, gw = (float(grad[0]), float(grad[1]),
                         float(grad[2]), float(grad[3]))
        k = 2.0 * (1.0 - gw)
        bias = gw - 0.5
        gu = (uv[0] + gx) * k + bias
        gv = (uv[1] + gy) * k + bias
        g = sample(colormask, *st((gu, gv), mask_st))[..., 0:1]
    else:
        g = np.ones((h, w, 1), np.float32)

    tint = c2[:3] + (c1[:3] - c2[:3]) * g
    lum = m[..., 0:1]
    mul = tint * lum
    rgb = mul + (_vivid_light(tint, lum) - mul) * float(blend)
    return _u8(rgb, m[..., 3] * c1[3])


# --------------------------------------------------------------- garments
def clothes(main, colormask, slots, main_st=None, mask_st=None):
    """``create_topN`` (= ``cf_m_clothesN_create``), the clothing composite.

    ``slots`` is three ``(base, pattern_colour, pattern_map, tiling, mask_st)``
    entries — ``ChangeCustomClothes`` fills them from ``colorInfo[i]``'s
    ``baseColor`` / ``patternColor`` / ``mt_pattern[pattern]`` / ``tiling``::

        for i in 1..3:
            s     = 20 - 19 * PatternScale{i}{u,v}
            p     = PatternMask{i}.Sample(uv * s * PatternMask{i}_ST).r
            t     = max(p, 1 - Color{i}_2.a)
            col_i = lerp(Color{i}_2, Color{i}, t)
        c = lerp(1,  col_1, mask.r)
        c = lerp(c,  col_2, mask.g)
        c = lerp(c,  col_3, mask.b)
        o.rgb = c * saturate(MainTex.rgb)     o.a = MainTex.a

    Three things this settles that the old hand-rolled version had wrong:

    * a **pattern is a second colour**, chosen per texel by the pattern map's
      red channel — not an ink stencil painted on top.  ``patternColor.a`` is
      its strength, and at 0 the ``max`` pins t to 1 so the base colour wins;
      that is how "no pattern" is spelled, and why an empty ``patternColor``
      must not be treated as "paint it black".
    * ``tiling`` maps as ``20 - 19*t``, per axis.  §7.11 flagged that mapping
      as the one constant with no data behind it; here it is, and the invented
      ``PATTERN_MAX_REPEAT`` was never close.
    * the three colours are applied by **sequential lerp** in R, G, B order,
      not a coverage-weighted average — where two channels overlap, blue wins
      outright rather than averaging with red.
    """
    if main is None and colormask is None:
        return None
    src = main if main is not None else colormask
    h, w = src.px.shape[:2]
    uv = _grid((h, w))

    cols = []
    for base, pat_colour, pat_map, tiling, pat_st in slots:
        c1 = _rgba(base, (0.82, 0.82, 0.85, 1.0))
        c2 = _rgba(pat_colour, (0.0, 0.0, 0.0, 0.0))
        if pat_map is None:
            cols.append(c1[:3].reshape(1, 1, 3))
            continue
        try:
            tu, tv = float(tiling[0]), float(tiling[1])
        except (TypeError, ValueError, IndexError):
            tu = tv = 0.0
        su, sv = 20.0 - 19.0 * tu, 20.0 - 19.0 * tv
        p = sample(pat_map, *st((uv[0] * su, uv[1] * sv), pat_st),
                   wrap=True)[..., 0:1]
        t = np.maximum(p, 1.0 - c2[3])
        cols.append(c2[:3] + (c1[:3] - c2[:3]) * t)

    if colormask is not None:
        mk = fetch(colormask, uv, mask_st, (h, w))
        c = 1.0 + mk[..., 0:1] * (cols[0] - 1.0)
        c = c + mk[..., 1:2] * (cols[1] - c)
        c = c + mk[..., 2:3] * (cols[2] - c)
    else:
        c = cols[0]

    if main is not None:
        m = fetch(main, uv, main_st, (h, w))
        rgb = c * np.clip(m[..., :3], 0.0, 1.0)
        alpha = m[..., 3]
    else:
        rgb = np.broadcast_to(c, (h, w, 3))
        alpha = 1.0
    return _u8(np.ascontiguousarray(np.broadcast_to(rgb, (h, w, 3))), alpha)


def hair(main, colormask, color, color2, color3, main_st=None, mask_st=None):
    """``create_hair`` (= ``cf_m_hair_create``)::

        c = lerp(1, _Color,  mask.r)
        c = lerp(c, _Color2, mask.g)
        c = lerp(c, _Color3, mask.b)
        o.rgb = c * MainTex.rgb        o.a = 1

    Same sequential-lerp shape as the garment shader, minus the patterns.
    """
    if main is None and colormask is None:
        return None
    src = main if main is not None else colormask
    h, w = src.px.shape[:2]
    uv = _grid((h, w))
    # Slots 2 and 3 fall back to slot 1.  They take the **converted** c1 rather
    # than passing the raw colour in as ``_rgba``'s fallback: that would run it
    # through the gamma conversion a second time and darken any hair part that
    # simply left a slot empty.
    c1 = _rgba(color)
    c2 = _rgba(color2) if _is_rgb(color2) else c1
    c3 = _rgba(color3) if _is_rgb(color3) else c1
    if colormask is not None:
        mk = fetch(colormask, uv, mask_st, (h, w))
        c = 1.0 + mk[..., 0:1] * (c1[:3] - 1.0)
        c = c + mk[..., 1:2] * (c2[:3] - c)
        c = c + mk[..., 2:3] * (c3[:3] - c)
    else:
        c = c1[:3].reshape(1, 1, 3)
    if main is not None:
        c = c * fetch(main, uv, main_st, (h, w))[..., :3]
    return _u8(np.ascontiguousarray(np.broadcast_to(c, (h, w, 3))), 1.0)


# ------------------------------------------------------------------- skin
def head(main, colormask, color, color2, layers, paintmask=None,
         main_st=None, mask_st=None):
    """``create_head`` (= ``cf_m_face_create``).

    ``layers`` is the decal stack in the order the shader applies it::

        (texture, colour, placement, tex_st, kind, use_paintmask)

    where *kind* selects the placement maths — ``"uv"`` for ``_tex4uv`` /
    ``_tex5uv``, ``"layout"`` for ``_paint1`` / ``_paint2``, ``"mole"`` for
    ``_hokuro`` (same as layout but with no rotation).

    The base is what the previous loader got wrong: red and green are two
    *independent* tints multiplied together, and blue is a **floor**, not a
    third colour.  Blue marks the eye whites and teeth, so ``max`` keeps them
    from being dyed with a skin tone.
    """
    src = main if main is not None else colormask
    if src is None:
        return None
    h, w = src.px.shape[:2]
    uv = _grid((h, w))
    c1, c2 = _rgba(color), _rgba(color2)

    if colormask is not None:
        mk = fetch(colormask, uv, mask_st, (h, w))
        r, g, b = mk[..., 0:1], mk[..., 1:2], mk[..., 2:3]
        col = (1.0 + r * (c1[:3] - 1.0)) * (1.0 + g * (c2[:3] - 1.0))
        col = np.maximum(b, col)
    else:
        col = c1[:3].reshape(1, 1, 3)
    if main is not None:
        col = col * fetch(main, uv, main_st, (h, w))[..., :3]
    col = np.ascontiguousarray(np.broadcast_to(col, (h, w, 3)))

    pm = (fetch(paintmask, uv, None, (h, w))[..., 0] if paintmask is not None
          else np.ones((h, w), np.float32))
    for tex, colour, place, tex_st, kind, masked in layers:
        if tex is None:
            continue
        if kind == "uv":
            p = uv_vector(uv, place, tex_st)
        else:
            p = layout(uv, place, tex_st, rotate=(kind != "mole"))
        lay = sample(tex, *p)
        c = _rgba(colour)
        a = lay[..., 3] * c[3]
        if masked:
            a = a * pm
        col = _over(col, lay[..., :3] * c[:3], a)
    return _u8(col, 1.0)


def body(main, colormask, color, color2, color5, sunburn, sunburn_color,
         paints, paintmask=None, main_st=None, mask_st=None, sun_st=None):
    """``create_body`` (= ``cf_m_body_create``).

    Different from the face in three ways that matter::

        t    = max(Sunburn.r, 1 - _Color4.a)
        skin = lerp(_Color4, _Color, t)                 # tan line drives it
        main.rgb *= lerp(1, _Color2, mask.g * _Color2.a)
        …paint layers…
        o.rgb = lerp(col, col * _Color5, mask.b)        # nails, at the end

    So the body's mask has no "third colour" either: green is the shading
    tone, blue multiplies in the nail colour, and red is unused.
    """
    src = main if main is not None else colormask
    if src is None:
        return None
    h, w = src.px.shape[:2]
    uv = _grid((h, w))
    c1, c2 = _rgba(color), _rgba(color2)
    c4 = _rgba(sunburn_color)
    c5 = _rgba(color5)

    # ``t`` is 1 everywhere without a tan map, which makes ``skin`` a constant
    # — worth keeping as a 1×1 so a 2048² body does not materialise three
    # 50 MB temporaries to arrive at one colour.
    if sunburn is not None:
        t = np.maximum(fetch(sunburn, uv, sun_st, (h, w))[..., 0:1], 1.0 - c4[3])
        skin = c4[:3] + (c1[:3] - c4[:3]) * t
    else:
        skin = c1[:3].reshape(1, 1, 3)

    albedo = (fetch(main, uv, main_st, (h, w))[..., :3] if main is not None
              else np.ones((1, 1, 3), np.float32))
    mk = (fetch(colormask, uv, mask_st, (h, w)) if colormask is not None
          else np.zeros((1, 1, 4), np.float32))
    if colormask is not None:
        albedo = albedo * (1.0 + mk[..., 1:2] * c2[3] * (c2[:3] - 1.0))
    col = np.ascontiguousarray(np.broadcast_to(skin * albedo, (h, w, 3)))

    pm = (fetch(paintmask, uv, None, (h, w))[..., 0] if paintmask is not None
          else np.ones((h, w), np.float32))
    for tex, colour, place, tex_st in paints:
        if tex is None:
            continue
        lay = sample(tex, *layout(uv, place, tex_st))
        c = _rgba(colour)
        col = _over(col, lay[..., :3] * c[:3], lay[..., 3] * c[3] * pm)

    if colormask is not None:
        col = col + (col * c5[:3] - col) * mk[..., 2:3]
    return _u8(col, 1.0)


# ---------------------------------------------------------------------------
# the shading term the forward passes read out of _LineMask / _DetailMask
# ---------------------------------------------------------------------------
def shade_mask(detail, line, props: dict, family: int,
               shape=None) -> np.ndarray | None:
    """Pack what ``main_*`` does with its two mask maps into one RGBA texture.

    None of this belongs in the albedo — Origin multiplies these into the *diffuse*
    term, after the light ramp, which is why a character lit from behind still
    shows its line work at full strength while the surface itself goes dark.
    Baking them into the map (what this loader used to do) locks the line work
    to whatever the lighting was when the card was dressed.

    The channels are what the shader needs, not what the source maps hold::

        R  the multiplier on the diffuse term
        G  the ink that darkens line work in the final colour
        B  the outline width scale, ``1 - _DetailMask.b``
        A  the ``_AnotherRamp`` blend weight, which the same register also uses
           as the scale on the shadow tint (see the note at the bottom)

    The rim's own mask is **not** in here — it is ``1 - _DetailMask.g``, a raw
    channel, and it travels with :func:`detail_map` along with the other three
    that main_opaque reads unpacked.

    ``main_skin`` is not built here — it needs the card's ``detailPower`` and
    its own ``_LineMask`` channel assignment, and :func:`body_head._skin_detail`
    already transcribes that pass.

    Both maps default to **black** in the shader's property table, so a slot
    nothing binds contributes 0 and every ``1 - x`` term above reads 1.  That
    is why an unbound mask has to mean "no effect" rather than "white".

    *detail* and *line* are :class:`colorspace.Map`, not bytes, and the two
    are **decoded by their own flags** before any of the arithmetic below:
    ``_md`` splits 38/62 linear/sRGB on this install, and every term here is a
    weight, so reading byte 128 as 0.502 where the game reads 0.216 moves the
    line work rather than blurring it (§7.126).  What comes out is a packed
    **coefficient** and therefore linear — it is uploaded raw, not as sRGB.
    """
    if family not in (2, 3):
        return None
    src = detail if detail is not None else line
    if src is None:
        return None
    h, w = (src.px.shape[:2] if shape is None else shape)
    uv = _grid((h, w))
    d = fetch(detail, uv, props.get("_DetailMask_ST"), (h, w))
    lm = fetch(line, uv, props.get("_LineMask_ST"), (h, w))
    if d is None:
        d = np.zeros((1, 1, 4), np.float32)
    if lm is None:
        lm = np.zeros((1, 1, 4), np.float32)
    ones = np.ones((h, w), np.float32)

    if family == 3:
        # main_hair: R is ``1 - _DetailMask.g`` on its own — the shader uses it
        # twice, once as ``lit * R`` and once inside
        # ``max(lit, R, 1 - _ShadowExtend)``, so the floor cannot be folded in
        # here.  It travels as ``Material.shade_floor`` instead.
        r = (1.0 - d[..., 1]) * ones
        g = ones
        b = (1.0 - d[..., 2]) * ones
        a = (1.0 - d[..., 1]) * (1.0 - d[..., 2]) * ones
    else:
        # main_opaque / main_alpha / main_item:
        #   shade = floor + (1 - floor) * min(1 - LineMask.b, 1 - DetailMask.g)
        #   ink   = pow(0.01 + 0.89 * (1 - _linewidthG),
        #               lerp(LineMask.g, DetailMask.b, _DetailBLineG))
        #   ramp weight = lerp(LineMask.r, DetailMask.r, _DetailRLineR)
        from . import toon as TOON
        floor = 1.0 - 1.2 * float(props.get("_ShadowExtend", 0.0))
        term = np.minimum(1.0 - lm[..., 2], 1.0 - d[..., 1]) * ones
        r = floor + (1.0 - floor) * term
        rl = float(props.get("_DetailRLineR", 0.0))
        bl = float(props.get("_DetailBLineG", 0.0))
        ink_exp = lm[..., 1] + bl * (d[..., 2] - lm[..., 1])
        base = 0.01 + 0.89 * (1.0 - TOON.line_width_g())
        g = np.power(base, np.maximum(ink_exp, 0.0)) * ones
        b = (1.0 - d[..., 2]) * ones
        # A: ``lerp(LineMask.r, DetailMask.r, _DetailRLineR)``.  This looked
        # like two competing quantities — a ramp blend weight and a scale on
        # the shadow tint — but the bytecode uses the *same register* for both:
        #
        #     r3.x = lerp(LineMask.r, DetailMask.r, _DetailRLineR)
        #     r1.w = r3.x * (anotherRamp - rampG) + rampG
        #     r3.x = max(r3.x, 1 - _ShadowExtendAnother)
        #     tint = saturate(r3.x * tint)
        #
        # so one channel serves both and the floor is a per-material constant
        # (``Material.shade_floor``), not a fifth per-texel value.
        a = (lm[..., 0] + rl * (d[..., 0] - lm[..., 0])) * ones

    out = np.stack([r, g, b, a], axis=-1)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def detail_map(detail, props: dict, shape=None) -> np.ndarray | None:
    """``_DetailMask`` alone, resampled through its own ``_ST`` onto uv0.

    :func:`shade_mask` folds the two mask maps into the four combined terms the
    diffuse needs, and that is all four channels — but ``main_opaque``'s forward
    base reads three more raw channels (``.r`` masks the analytic lobe, ``.g``
    masks the rim, ``.a`` is the specular map the "nail" term reads) and it
    reads ``.r`` a **second** time at a uv the view direction shifts::

        r2.z  = (_SpeclarHeight - 1) * 0.8
        uv'   = uv + r2.z * float2(dot(tangent, V), dot(bitangent, V))
        r4    = _DetailMask(uv' * _DetailMask_ST.xy + _DetailMask_ST.zw)

    A parallax lookup cannot be baked into a packed mask at any number of
    channels, so the map travels as itself.  Because ``_ST`` is applied *here*
    and the grid is uv0, the shader's offset lands on exactly the texel the
    game's does.

    ``None`` for an unbound slot: the shader's property table gives
    ``_DetailMask`` a **black** default — RGBA(0,0,0,0), not opaque black — and
    the renderer's "not bound" path reproduces that.
    """
    if detail is None:
        return None
    h, w = (detail.shape[:2] if shape is None else shape)
    d = fetch_raw(detail, _grid((h, w)), props.get("_DetailMask_ST"), (h, w))
    if d is None:
        return None
    return np.clip(d * 255.0, 0, 255).astype(np.uint8)
