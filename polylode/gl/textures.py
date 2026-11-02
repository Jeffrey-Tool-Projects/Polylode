"""Procedural textures: anime eyes, lashes, brows, mouths, blush, fabric patterns.

Everything is generated with numpy so the project ships no binary art assets.
Generators return float RGBA in [0,1]; ``TextureCache`` uploads and reuses them.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
from OpenGL import GL

from ..core import swallowed

RES = 256


def _uv(res: int = RES):
    u = (np.arange(res) + 0.5) / res
    v = (np.arange(res) + 0.5) / res
    return np.meshgrid(u, v, indexing="xy")   # U varies along x, V along y


def _aa(d: np.ndarray, w: float = 0.012) -> np.ndarray:
    """Antialiased inside-mask from a signed distance-ish field (<0 = inside)."""
    return np.clip(0.5 - d / max(w, 1e-6), 0.0, 1.0)


def _circle(U, V, cx, cy, r):
    return np.sqrt((U - cx) ** 2 + (V - cy) ** 2) - r


def _over(dst: np.ndarray, rgb, alpha: np.ndarray) -> np.ndarray:
    """Alpha-composite a flat colour over an RGBA buffer."""
    a = alpha[..., None]
    col = np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3)
    out_a = alpha + dst[..., 3] * (1.0 - alpha)
    safe = np.maximum(out_a, 1e-6)[..., None]
    dst[..., :3] = (col * a + dst[..., :3] * dst[..., 3:4] * (1.0 - a)) / safe
    dst[..., 3] = out_a
    return dst


# --------------------------------------------------------------------------
# eyes
# --------------------------------------------------------------------------
def eye_shape(U, V, style: int = 0):
    """Signed field for the visible eye opening (anime almond)."""
    presets = [
        # (half_w, top, bottom, slant, sharp_x, sharp_y)
        (0.475, 0.380, 0.320, 0.055, 2.70, 2.15),   # 0 standard
        (0.485, 0.315, 0.265, 0.045, 3.00, 2.35),   # 1 narrow / cool
        (0.450, 0.430, 0.385, 0.060, 2.35, 1.95),   # 2 round / cute
        (0.480, 0.360, 0.270, 0.115, 2.80, 2.25),   # 3 tsurime (upturned)
        (0.480, 0.355, 0.340, -0.085, 2.80, 2.25),  # 4 tareme (downturned)
        (0.495, 0.265, 0.225, 0.030, 3.40, 2.60),   # 5 sleepy slit
    ]
    hw, top, bot, slant, sx, sy = presets[style % len(presets)]
    du = (U - 0.5) / hw
    vc = 0.47 + slant * (U - 0.5) * 2.0
    dv = V - vc
    rad = np.where(dv >= 0, top, bot)
    corner = 1.0 - 0.22 * np.clip((U - 0.55) / 0.45, 0, 1) ** 2
    return np.abs(du) ** sx + np.abs(dv / (rad * corner)) ** sy - 1.0


def gen_eye(iris=(0.35, 0.62, 0.86), pupil=(0.10, 0.09, 0.14),
            sclera=(1.0, 0.99, 0.99), highlight=(1.0, 1.0, 1.0),
            style: int = 0, iris_style: int = 0, iris_size: float = 0.5,
            pupil_size: float = 0.5, hl_size: float = 0.5,
            grad_color=None, grad_blend: float = 0.0, grad_dir: float = 0.0,
            res: int = RES) -> np.ndarray:
    """``grad_color``/``grad_blend`` add a vertical two-tone through the iris,
    which is how most anime eyes get their depth (and what Illusion's cards
    store as pupil subColor + gradBlend)."""
    U, V = _uv(res)
    out = np.zeros((res, res, 4), dtype=np.float32)

    shape = eye_shape(U, V, style)
    inside = _aa(shape, 0.020)

    # sclera with a soft shadow cast by the upper lid
    lid = np.clip((V - 0.50) / 0.34, 0.0, 1.0) ** 1.4
    sc = np.asarray(sclera, dtype=np.float32).reshape(1, 1, 3) * (1.0 - lid * 0.22)[..., None]
    out[..., :3] = sc
    out[..., 3] = inside

    # iris — narrower than tall in texture space, because the decal patch is
    # stretched horizontally on the face; on screen this reads as a tall iris
    ir = 0.20 + iris_size * 0.20
    icx, icy = 0.495, 0.455
    ax = ir * 0.74
    d = np.sqrt(((U - icx) / ax) ** 2 + ((V - icy) / ir) ** 2) * ir
    im = _aa(d - ir, 0.010) * inside

    base = np.asarray(iris, dtype=np.float32)
    grad = np.clip((V - (icy - ir)) / (2 * ir), 0.0, 1.0)
    icol = (base.reshape(1, 1, 3) * (0.52 + 0.68 * grad)[..., None])
    if grad_color is not None and grad_blend > 0.001:
        g2 = np.asarray(grad_color, dtype=np.float32).reshape(1, 1, 3)
        t = grad if grad_dir >= 0 else (1.0 - grad)
        t = np.clip(t, 0, 1) ** 1.25
        icol = icol * (1.0 - (t * grad_blend)[..., None]) \
            + g2 * (0.55 + 0.85 * grad)[..., None] * (t * grad_blend)[..., None]
    if iris_style in (1, 3):    # radial spokes
        ang = np.arctan2(V - icy, U - icx)
        spokes = 0.90 + 0.14 * np.cos(ang * 14.0) * np.clip(d / ir, 0, 1)
        icol *= spokes[..., None]
    if iris_style in (2, 3):    # bright inner ring
        ring = np.exp(-((d - ir * 0.45) / (ir * 0.16)) ** 2)
        icol += base.reshape(1, 1, 3) * ring[..., None] * 0.45
    limbal = np.exp(-((d - ir * 0.94) / (ir * 0.13)) ** 2)
    icol *= (1.0 - 0.62 * limbal)[..., None]
    out[..., :3] = out[..., :3] * (1 - im[..., None]) + np.clip(icol, 0, 1) * im[..., None]

    # pupil
    pr = ir * (0.26 + pupil_size * 0.34)
    pm = _aa(d - pr, 0.008) * inside
    pc = np.asarray(pupil, dtype=np.float32).reshape(1, 1, 3)
    out[..., :3] = out[..., :3] * (1 - pm[..., None]) + pc * pm[..., None]

    # highlights
    hs = 0.6 + hl_size * 0.9
    h1 = _aa(_circle(U, V, icx - 0.105, icy + 0.150, 0.082 * hs), 0.010) * inside
    h2 = _aa(_circle(U, V, icx + 0.115, icy - 0.135, 0.042 * hs), 0.010) * inside
    h3 = _aa(_circle(U, V, icx + 0.02, icy + 0.045, 0.022 * hs), 0.008) * inside * 0.5
    hm = np.clip(h1 + h2 + h3, 0, 1)
    hc = np.asarray(highlight, dtype=np.float32).reshape(1, 1, 3)
    out[..., :3] = out[..., :3] * (1 - hm[..., None]) + hc * hm[..., None]

    # dark rim where the lid meets the eyeball
    rim = np.clip(1.0 - np.abs(shape) / 0.10, 0, 1) ** 2 * inside
    out[..., :3] *= (1.0 - 0.35 * rim)[..., None]
    return np.clip(out, 0, 1)


def gen_lash(style: int = 0, thickness: float = 0.5, res: int = RES) -> np.ndarray:
    """White RGB + alpha mask; the material tints it."""
    U, V = _uv(res)
    out = np.zeros((res, res, 4), dtype=np.float32)
    out[..., :3] = 1.0
    t = 0.055 + thickness * 0.085
    presets = [
        (0.46, 0.30, 1.8, 1.00),   # 0 standard flick
        (0.48, 0.16, 1.4, 0.80),   # 1 subtle
        (0.44, 0.44, 2.1, 1.25),   # 2 dramatic
        (0.50, 0.05, 1.0, 0.70),   # 3 flat / cool
        (0.42, 0.34, 1.6, 1.10),   # 4 downturned outer
    ]
    base, rise, power, mul = presets[style % len(presets)]
    vc = base + rise * np.clip(U, 0, 1) ** power
    if style == 4:
        vc = base + rise * np.clip(U, 0, 1) ** power - 0.16 * np.clip(U - 0.6, 0, 1)
    half = t * mul * (0.55 + 0.85 * np.clip(U + 0.15, 0, 1))
    half *= np.clip((U + 0.02) / 0.10, 0, 1)              # taper at the inner corner
    half *= np.clip((1.03 - U) / 0.12, 0, 1) ** 0.6       # taper at the tip
    a = _aa(np.abs(V - vc) - half, 0.014)
    # lower lash line: thin and short
    lower = _aa(np.abs(V - (base - 0.30 - 0.05 * U)) - t * 0.30 *
                np.clip((U - 0.25) / 0.5, 0, 1), 0.012) * 0.75
    out[..., 3] = np.clip(a + lower, 0, 1)
    return out


def gen_brow(style: int = 0, thickness: float = 0.5, res: int = RES) -> np.ndarray:
    U, V = _uv(res)
    out = np.zeros((res, res, 4), dtype=np.float32)
    out[..., :3] = 1.0
    presets = [
        (0.36, 0.42, -0.30, 1.00),   # 0 gentle arch
        (0.40, 0.22, -0.14, 0.85),   # 1 straight
        (0.30, 0.60, -0.46, 1.15),   # 2 high arch
        (0.44, 0.16, 0.00, 1.30),    # 3 thick / strong
        (0.34, 0.34, -0.24, 0.62),   # 4 thin
    ]
    b, rise, curve, mul = presets[style % len(presets)]
    vc = b + rise * U + curve * U * U
    half = (0.045 + thickness * 0.075) * mul
    half = half * (1.0 - 0.55 * np.clip((U - 0.45) / 0.55, 0, 1) ** 1.4)
    half = half * np.clip((U + 0.05) / 0.14, 0, 1) ** 0.5
    half = half * np.clip((1.05 - U) / 0.18, 0, 1) ** 0.7
    out[..., 3] = _aa(np.abs(V - vc) - half, 0.016)
    return out


MOUTH_STYLES = ["neutral", "smile", "grin", "open", "small_open", "frown",
                "pout", "wide_smile", "surprised", "wavy"]


def gen_mouth(style: str = "neutral", lip_color=(0.82, 0.36, 0.40),
              inner=(0.42, 0.16, 0.20), tongue=(0.90, 0.48, 0.52),
              res: int = RES) -> np.ndarray:
    U, V = _uv(res)
    out = np.zeros((res, res, 4), dtype=np.float32)
    out[..., :3] = 1.0
    du = (U - 0.5)

    def line(curve: float, thick: float, width: float = 0.30):
        vc = 0.5 + curve * (1.0 - (du / width) ** 2) * (np.abs(du) < width)
        a = _aa(np.abs(V - vc) - thick, 0.012)
        return a * _aa(np.abs(du) - width, 0.03)

    def opening(w: float, h: float, curve: float = 0.0):
        d = (du / w) ** 2 + ((V - (0.5 + curve)) / h) ** 2 - 1.0
        return _aa(d, 0.035)

    if style == "neutral":
        out[..., 3] = line(0.0, 0.014, 0.24)
        out[..., :3] = np.asarray(lip_color).reshape(1, 1, 3)
    elif style == "smile":
        out[..., 3] = line(0.075, 0.016, 0.28)
        out[..., :3] = np.asarray(lip_color).reshape(1, 1, 3)
    elif style == "frown":
        out[..., 3] = line(-0.070, 0.016, 0.26)
        out[..., :3] = np.asarray(lip_color).reshape(1, 1, 3)
    elif style == "pout":
        m = opening(0.13, 0.10)
        out[..., 3] = m
        out[..., :3] = np.asarray(lip_color).reshape(1, 1, 3)
    elif style == "wavy":
        vc = 0.5 + 0.045 * np.sin(du * 26.0)
        out[..., 3] = _aa(np.abs(V - vc) - 0.015, 0.012) * _aa(np.abs(du) - 0.26, 0.03)
        out[..., :3] = np.asarray(lip_color).reshape(1, 1, 3)
    else:
        cfg = {"grin": (0.30, 0.16, 0.03), "open": (0.22, 0.24, 0.0),
               "small_open": (0.13, 0.15, 0.0), "wide_smile": (0.34, 0.13, 0.05),
               "surprised": (0.16, 0.26, 0.0)}[style]
        w, h, cv = cfg
        m = opening(w, h, cv)
        out = _over(out, inner, m)
        if style in ("grin", "wide_smile"):
            teeth = _aa(((du / (w * 0.92)) ** 2 +
                         ((V - (0.5 + cv + h * 0.55)) / (h * 0.42)) ** 2) - 1.0, 0.03) * m
            out = _over(out, (0.99, 0.97, 0.96), teeth)
        if style in ("open", "surprised"):
            tg = _aa(((du / (w * 0.62)) ** 2 +
                      ((V - (0.5 + cv - h * 0.52)) / (h * 0.45)) ** 2) - 1.0, 0.03) * m
            out = _over(out, tongue, tg)
        lip = _aa(np.abs(np.sqrt((du / w) ** 2 + ((V - (0.5 + cv)) / h) ** 2) - 1.0) - 0.075, 0.03)
        out = _over(out, lip_color, lip * 0.85)
    return np.clip(out, 0, 1)


def gen_blush(style: int = 0, res: int = RES) -> np.ndarray:
    U, V = _uv(res)
    out = np.zeros((res, res, 4), dtype=np.float32)
    out[..., :3] = 1.0
    d = np.sqrt(((U - 0.5) / 0.46) ** 2 + ((V - 0.5) / 0.34) ** 2)
    soft = np.clip(1.0 - d, 0, 1) ** 1.6
    if style == 0:
        out[..., 3] = soft
    elif style == 1:      # classic diagonal stripes
        stripes = (np.sin((U * 8.0 + V * 4.0) * math.pi) > 0.35).astype(np.float32)
        out[..., 3] = soft * (0.30 + 0.70 * stripes)
    else:                 # freckle dots
        dots = np.zeros_like(U)
        rs = np.random.RandomState(7)
        for _ in range(26):
            cx, cy = rs.uniform(0.15, 0.85, 2)
            dots += np.exp(-(((U - cx) ** 2 + (V - cy) ** 2) / 0.0009))
        out[..., 3] = soft * np.clip(dots, 0, 1)
    return out


# --------------------------------------------------------------------------
# fabric patterns (alpha masks tinted by the material)
# --------------------------------------------------------------------------


def gen_pattern(name: str, res: int = RES) -> np.ndarray:
    U, V = _uv(res)
    out = np.zeros((res, res, 4), dtype=np.float32)
    out[..., :3] = 1.0
    a = np.zeros_like(U)
    if name == "stripe_h":
        a = (np.sin(V * math.pi * 12) > 0).astype(np.float32)
    elif name == "stripe_v":
        a = (np.sin(U * math.pi * 12) > 0).astype(np.float32)
    elif name == "check":
        a = (((U * 8).astype(int) + (V * 8).astype(int)) % 2).astype(np.float32)
    elif name == "gingham":
        a = np.clip((np.sin(U * math.pi * 10) > 0).astype(np.float32) * 0.5 +
                    (np.sin(V * math.pi * 10) > 0).astype(np.float32) * 0.5, 0, 1)
    elif name == "plaid":
        a = np.clip(((np.abs(np.sin(U * math.pi * 6)) > 0.86).astype(np.float32) +
                     (np.abs(np.sin(V * math.pi * 6)) > 0.86).astype(np.float32) * 0.9 +
                     (np.abs(np.sin(U * math.pi * 12)) > 0.94).astype(np.float32) * 0.5), 0, 1)
    elif name == "dots":
        fu, fv = (U * 8) % 1.0 - 0.5, (V * 8) % 1.0 - 0.5
        a = _aa(np.sqrt(fu ** 2 + fv ** 2) - 0.26, 0.03)
    elif name == "stars":
        fu, fv = (U * 6) % 1.0 - 0.5, (V * 6) % 1.0 - 0.5
        ang = np.arctan2(fv, fu)
        r = np.sqrt(fu ** 2 + fv ** 2)
        a = _aa(r - (0.14 + 0.10 * np.cos(ang * 5.0)), 0.02)
    elif name == "hearts":
        fu, fv = ((U * 7) % 1.0 - 0.5) * 2.2, ((V * 7) % 1.0 - 0.5) * 2.2
        h = (fu ** 2 + fv ** 2 - 0.42) ** 3 - fu ** 2 * fv ** 3
        a = _aa(h, 0.06)
    elif name == "argyle":
        fu, fv = (U * 5) % 1.0 - 0.5, (V * 5) % 1.0 - 0.5
        a = _aa(np.abs(fu) + np.abs(fv) - 0.34, 0.03)
    elif name == "camo":
        rs = np.random.RandomState(3)
        for _ in range(9):
            cx, cy, r = rs.uniform(0, 1), rs.uniform(0, 1), rs.uniform(0.10, 0.24)
            warp = 0.05 * np.sin(U * 21 + cy * 9) + 0.05 * np.cos(V * 17 + cx * 7)
            a += _aa(np.sqrt((U - cx) ** 2 + (V - cy) ** 2) + warp - r, 0.03)
        a = np.clip(a, 0, 1)
    elif name == "leopard":
        rs = np.random.RandomState(11)
        for _ in range(48):
            cx, cy = rs.uniform(0, 1, 2)
            rx, ry = rs.uniform(0.020, 0.045), rs.uniform(0.016, 0.038)
            a += _aa(np.sqrt(((U - cx) / rx) ** 2 + ((V - cy) / ry) ** 2) - 1.0, 0.25)
        a = np.clip(a, 0, 1)
    elif name == "gradient":
        a = np.clip(V, 0, 1) ** 1.4
    elif name == "lace":
        fu, fv = (U * 12) % 1.0 - 0.5, (V * 12) % 1.0 - 0.5
        r = np.sqrt(fu ** 2 + fv ** 2)
        a = _aa(np.abs(r - 0.30) - 0.07, 0.03)
    out[..., 3] = a
    return out


def gen_white(res: int = 4) -> np.ndarray:
    return np.ones((res, res, 4), dtype=np.float32)


# --------------------------------------------------------------------------
# key-driven factory: a Material's ``texture`` field is a hashable key tuple
# of the form (generator_name, *args) which is generated lazily on first use.
# --------------------------------------------------------------------------
def gen_prop(name: str, seed: int = 0) -> np.ndarray:
    """A scenery material map (``mesh.proptex``), repeated across a prop in metres."""
    from ..mesh import proptex
    return proptex.generate(name, int(seed))


TEXTURE_FACTORY = {
    "prop": gen_prop,
    "white": gen_white,
    "eye": gen_eye,
    "lash": gen_lash,
    "brow": gen_brow,
    "mouth": gen_mouth,
    "blush": gen_blush,
    "pattern": gen_pattern,
}


def resolve_texture(cache: "TextureCache", key) -> int:
    """Look up (or lazily generate) the GL texture for a material key.

    Material fields must hold *keys* (``(\"originasset\", id)``, ``(\"white\",)``,
    …), never raw GL names.  A bare ``int`` used to be treated as a GL name;
    after LRU eviction that name is deleted and ``glBindTexture`` raises
    ``GL_INVALID_OPERATION`` (1282) — the Free H freeze after a few seconds.
    Bare ints / numpy integers fall through to white instead.
    """
    if key is None or key is False:
        return cache.get_or_make(("white",), gen_white, mipmap=False)
    # numpy integer / plain int are *not* safe GL names (eviction deletes them).
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        return cache.get_or_make(("white",), gen_white, mipmap=False)
    if isinstance(key, str):
        key = (key,)
    try:
        key = tuple(key)
    except TypeError:
        return cache.get_or_make(("white",), gen_white, mipmap=False)
    if key and key[0] == 'gf2map':
        from ..gf2.normal_maps import resolve_texture as resolve_gf2_texture
        return resolve_gf2_texture(cache, key)
    if key and key[0] == 'gf2source':
        from ..gf2.texture_sources import resolve_texture as resolve_gf2_source
        return resolve_gf2_source(cache, key)
    if key and key[0] == 'convertedskin':
        from ..converted.donor_skin import resolve_texture as resolve_donor_skin
        return resolve_donor_skin(cache, key)
    tid = cache.get(key)
    if tid is not None:
        return tid
    fn = TEXTURE_FACTORY.get(key[0])
    if fn is None:
        return cache.get_or_make(("white",), gen_white, mipmap=False)
    try:
        arr = fn(*key[1:])
    except Exception as exc:                      # noqa: BLE001 - never kill a frame
        import warnings
        warnings.warn(f"texture {key[0]} failed to generate: {exc!r}", stacklevel=2)
        arr = gen_white()
    wrap = (GL.GL_REPEAT if key[0] in ("pattern", "originasset_repeat", "prop")
            else GL.GL_CLAMP_TO_EDGE)
    # An imported map's colour space rides in the key rather than beside it:
    # the key is the only thing that survives from the resolver to this call, so
    # a flag stored anywhere else is one that a caller can forget and the upload
    # would then silently pick a default — the failure mode :mod:`colorspace`
    # exists to make impossible.  Procedurally generated maps (``white``,
    # ``pattern``, …) have no such flag and stay raw: they are masks and
    # neutral fills, and 255 decodes to 1.0 either way.
    return cache.upload(key, arr, mipmap=True, wrap=wrap,
                        srgb=bool(key[-1]) if key[0].startswith("originasset")
                        and len(key) >= 3 else False)


# --------------------------------------------------------------------------
# how much video memory this machine has
# --------------------------------------------------------------------------
# Neither vendor exposes this in core GL, so it is two extensions or nothing:
#   NVX_gpu_memory_info  DEDICATED_VIDMEM 0x9047, CURRENT_AVAILABLE 0x9049 (KB)
#   ATI_meminfo          TEXTURE_FREE_MEMORY_ATI 0x87FC, 4 ints, [0] free (KB)
# Intel exposes neither.  Unknown is reported as (0, 0) and the caller falls
# back to a fixed budget rather than guessing from the screen size or similar.
_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX = 0x9047
_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX = 0x9049
_TEXTURE_FREE_MEMORY_ATI = 0x87FC

# Widest write any of the queries above performs.  ATI_meminfo's is four.
_MEMINFO_INTS = 4

_vram: list = [None]


def _get_ints(pname: int, count: int = _MEMINFO_INTS) -> np.ndarray:
    """``glGetIntegerv`` with a destination *we* size.

    PyOpenGL picks the output buffer from a table of known enums; for one it
    does not know -- and every enum here comes from an extension -- it hands
    the driver a **single** GLint.  ``GL_TEXTURE_FREE_MEMORY_ATI`` writes
    four, so the call overruns that allocation by 12 bytes.  Nothing fails at
    the call site: the value returned is correct, and the damage surfaces
    later as ``0xC0000374`` (STATUS_HEAP_CORRUPTION) at whatever unrelated
    free happens to walk into it -- which is why this read as a
    teardown-ordering problem for three sections (see PROGRESS §7.39).

    Measured on the Radeon 760M this was found on: 20 calls the PyOpenGL way
    crashed 5 runs out of 6, the same 20 calls with a four-int destination
    crashed 0 out of 6.
    """
    buf = np.zeros(count, dtype=np.int32)
    GL.glGetIntegerv(pname, buf)
    return buf


def extension_names() -> str:
    """The current context's extensions, space separated; ``""`` when they cannot be read.

    Enumerated, which is the only way a core profile answers — every context
    this project makes is one, and asking it for the old single string is a
    ``GL_INVALID_ENUM`` on each start.  The single string is asked only of a
    context that cannot enumerate (a compatibility one).
    """
    try:
        count = int(GL.glGetIntegerv(GL.GL_NUM_EXTENSIONS))
        return " ".join(GL.glGetStringi(GL.GL_EXTENSIONS, i).decode("ascii", "replace") for i in range(count))
    except Exception as enumerating:                 # noqa: BLE001
        try:
            whole = GL.glGetString(GL.GL_EXTENSIONS)
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("gl.textures.extension_names: neither glGetStringi nor glGetString",
                           RuntimeError(f"{enumerating!r}; {exc!r}"))
            return ""
        return whole.decode("ascii", "replace") if isinstance(whole, bytes) else ""


def vram_mb() -> tuple[float, float]:
    """``(total, free)`` video memory in MB; ``(0, 0)`` when unknown.

    Requires a current context.  Queried once — the numbers move, but not by
    enough to be worth a driver round-trip per frame, and the budget derived
    from them is a ceiling rather than a running total.
    """
    if _vram[0] is not None:
        return _vram[0]
    out = (0.0, 0.0)
    names = extension_names()
    try:
        if "GL_NVX_gpu_memory_info" in names:
            # one int each by the extension's own wording, but sized the same
            # way as the ATI query so no driver can write past the buffer
            total = int(_get_ints(_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX)[0])
            free = int(_get_ints(
                _GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX)[0])
            out = (total / 1024.0, free / 1024.0)
        elif "GL_ATI_meminfo" in names:
            free = float(_get_ints(_TEXTURE_FREE_MEMORY_ATI)[0])
            out = (free / 1024.0, free / 1024.0)
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("gl.textures.vram_mb: if 'GL_NVX_gpu_memory_info' in names:", exc)
        out = (0.0, 0.0)
    _vram[0] = out
    return out


# Share of video memory the texture cache may hold.  The rest of the frame
# needs room too: the MSAA target, its resolve, the depth texture, the bloom
# chain and every mesh's vertex buffers all live there.
VRAM_SHARE = 0.35
MIN_TEXTURE_BUDGET_MB = 128.0
MAX_TEXTURE_BUDGET_MB = 2048.0
# Used when neither extension is present (Intel, or a driver that hides it).
UNKNOWN_VRAM_BUDGET_MB = 256.0


def texture_budget_bytes() -> int:
    total, free = vram_mb()
    if total <= 0.0:
        return int(UNKNOWN_VRAM_BUDGET_MB * 1024 * 1024)
    share = max(total, free) * VRAM_SHARE
    mb = min(MAX_TEXTURE_BUDGET_MB, max(MIN_TEXTURE_BUDGET_MB, share))
    return int(mb * 1024 * 1024)


# --------------------------------------------------------------------------
# GPU cache
# --------------------------------------------------------------------------
class TextureCache:
    """Uploaded textures, bounded by what the card actually has.

    Nothing evicted anything before: a session that dressed a few characters
    kept every map any of them had ever used resident until the renderer was
    released.  Eviction is safe because ``resolve_texture`` regenerates a key
    on miss — worst case a map is re-uploaded, or comes back white if the
    CPU-side store dropped it too, which is the same degradation that store
    already had.

    Keys registered with :meth:`pin` (the 1×1 white fallback, grading LUT, …)
    are never deleted.  The renderer keeps those GL names in fields such as
    ``self.white`` and binds them every frame *without* going through
    ``resolve_texture``; if LRU deleted them, the next ``glBindTexture`` is
    ``GL_INVALID_OPERATION`` and Free H freezes after a few seconds of thrash.
    """

    def __init__(self, budget_bytes: int | None = None):
        self._tex: "OrderedDict[object, int]" = OrderedDict()
        self._bytes: dict = {}
        self._total = 0
        self._budget = budget_bytes
        self._pinned: set = set()
        # **Eviction has to be countable.**  When the budget is short the LRU
        # deletes a GL name and ``glGenTextures`` hands that same integer back
        # for the next upload, so anything holding the raw id — rather than the
        # key — starts sampling another texture.  The failure is not the
        # documented 1x1 white (§7.62): it is *some other map*, so the render
        # still looks like a render and the difference reads as "our textures
        # are wrong".  Nothing reported it, and nothing could, until this
        # counter existed (§7.148).
        self.evicted = 0

    # -- budget --------------------------------------------------------
    @property
    def budget(self) -> int:
        if self._budget is None:
            self._budget = texture_budget_bytes()
        return self._budget

    def pin(self, key) -> None:
        """Never evict this key (call after the first upload)."""
        self._pinned.add(key if not isinstance(key, list) else tuple(key))

    def touch(self, key) -> None:
        """Mark a key as most recently used."""
        if key in self._tex:
            self._tex.move_to_end(key)

    def _evict_to_fit(self, incoming: int) -> None:
        budget = self.budget
        # Walk LRU → MRU; skip pinned.  Stop if only pinned entries remain.
        while self._tex and self._total + incoming > budget:
            victim = None
            for key in self._tex:
                if key not in self._pinned:
                    victim = key
                    break
            if victim is None:
                break
            tid = self._tex.pop(victim)
            self._total -= self._bytes.pop(victim, 0)
            GL.glDeleteTextures(1, [tid])
            self.evicted += 1

    def upload_native(self, key, size: int, fill) -> int:
        """Own a native upload with exact byte accounting; failed uploads leave no name."""
        resident = self.get(key)
        if resident is not None:
            return resident
        if type(size) is not int or size <= 0:
            raise ValueError('invalid native texture size')
        self._evict_to_fit(size)
        tid = int(GL.glGenTextures(1))
        try:
            GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
            fill()
        except BaseException:
            GL.glDeleteTextures(1, [tid])
            raise
        finally:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._tex[key] = tid
        self._bytes[key] = size
        self._total += size
        return tid

    def upload(self, key, array: np.ndarray, mipmap: bool = True,
               wrap=GL.GL_REPEAT, srgb: bool = False) -> int:
        """Upload *array*: RGBA ``uint8`` as-is, or float 0..1.

        *srgb* is the source asset's own ``Texture2D.m_ColorSpace``, not a
        guess from what the map is used for: it decides the **internal format**,
        and therefore whether the GPU decodes on fetch.  Koikatsu is a Linear
        project, so a map it flagged sRGB has to arrive as ``GL_SRGB8_ALPHA8``
        or every shader downstream reads byte 128 as 0.502 where the game reads
        0.216 (§7.85 三, §7.126).  It also puts the decode **before** the
        bilinear filter, which is the hardware order and not reproducible by
        converting the array here.
        """
        if array.dtype == np.uint8:
            data = array
        else:
            data = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        data = np.ascontiguousarray(data[::-1])          # GL origin is bottom-left
        h, w = data.shape[:2]
        # 4 bytes a texel, plus a third again when mipmapped
        size = int(w * h * 4 * (4.0 / 3.0 if mipmap else 1.0))
        tid = self._tex.get(key)
        if tid is None:
            self._evict_to_fit(size)
            tid = int(GL.glGenTextures(1))
        self._total += size - self._bytes.get(key, 0)
        self._bytes[key] = size
        self._tex[key] = tid
        self._tex.move_to_end(key)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0,
                        GL.GL_SRGB8_ALPHA8 if srgb else GL.GL_RGBA8, w, h, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
        wrap_s, wrap_t = wrap if isinstance(wrap, tuple) else (wrap, wrap)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_s)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_t)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        if mipmap:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                               GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        else:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return tid

    def get(self, key) -> int | None:
        tid = self._tex.get(key)
        if tid is not None:
            self._tex.move_to_end(key)      # this is the LRU's only input
            return int(tid)
        return None

    def get_or_make(self, key, factory, **kw) -> int:
        tid = self.get(key)
        if tid is None:
            tid = self.upload(key, factory(), **kw)
        return tid

    def invalidate(self, prefix: str) -> None:
        dead = [k for k in self._tex
                if isinstance(k, tuple) and k and k[0] == prefix
                and k not in self._pinned]
        for k in dead:
            GL.glDeleteTextures(1, [self._tex.pop(k)])
            self._total -= self._bytes.pop(k, 0)

    def stats(self) -> dict:
        total, free = vram_mb()
        return {"textures": len(self._tex),
                "evicted": self.evicted,
                "mb": self._total / (1024 * 1024),
                "budget_mb": self.budget / (1024 * 1024),
                "vram_total_mb": total, "vram_free_mb": free,
                "pinned": len(self._pinned)}

    def release(self, include_pinned: bool = False) -> None:
        """Drop every unpinned map.

        **Pinned names survive**, because the class's own contract says they
        do: the renderer keeps ``self.white`` and friends as raw GL integers
        and binds them every frame without going back through the cache.  The
        first version cleared ``_pinned`` as well, and the very next
        ``glBindTexture(GL_TEXTURE_2D, self.white)`` was
        ``GL_INVALID_OPERATION`` — a method that deletes exactly what the
        docstring twelve lines above promises never to delete (§7.148).

        Pass *include_pinned* only when the GL context is going away with it
        (``Renderer.release``), where leaving them behind would leak.
        """
        for key in [k for k in self._tex
                    if include_pinned or k not in self._pinned]:
            GL.glDeleteTextures(1, [self._tex.pop(key)])
            self._total -= self._bytes.pop(key, 0)
        if include_pinned:
            self._pinned.clear()
        self.evicted = 0
