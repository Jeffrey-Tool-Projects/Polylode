"""sRGB ↔ linear, and which of the two a Koikatsu map's bytes are in.

Koikatsu is a **Linear** project: ``Koikatu_Data/globalgamemanagers`` holds
``PlayerSettings.m_ActiveColorSpace = 1``.  That single number decides where
every one of Illusion's compositing shaders does its arithmetic, and this
project transcribed all six of them without it — the maths ran on sRGB bytes
because that is what a PNG hands you.

Three things follow, and all three are read rather than assumed:

* **A texture's colour space is per texture**, not per use.  Unity stores it on
  the asset as ``Texture2D.m_ColorSpace`` (0 = linear, 1 = sRGB) and Illusion's
  import settings are genuinely inconsistent.  Over the baked library: all 634
  ``*_n`` normal maps are linear (which is what fixes the enum's direction),
  648 of 678 ``*_t`` albedos are sRGB — but the ``*_mc`` colour masks split
  **241 linear / 440 sRGB**, and the ``mt_eye`` list splits 19 / 76.  Guessing
  "albedo is sRGB, masks are data" would be wrong on a third of the masks, and a
  mask is a *lerp weight*: byte 128 reads 0.502 one way and **0.216** the other,
  which moves the colour rather than its precision.  :func:`decode` takes the
  flag; it does not take a name.

* **Colour properties are converted by the engine.**  Every colour input of the
  six ``*_create`` shaders is declared ``Color`` in the shader's property table
  (``create_eye._Color`` / ``_Color2``, ``create_head._Color``.. ``_Color7``,
  ``create_body``'s eight, ``create_hair``'s three, ``create_eyewhite``'s two),
  and none of them carries ``[Gamma]``.  In a linear project Unity converts a
  ``Color`` property from gamma to linear on its way to the GPU, which is why
  ``ChaControl`` can call ``SetColor`` with the card's raw bytes and no code in
  ``Assembly-CSharp`` ever mentions a conversion — ``Mathf.GammaToLinearSpace``
  appears exactly once in the whole assembly, in ``BloomSettings``, and
  ``Color.linear`` once, in ``FogComponent``.  Neither is on the character path.

* **The composite comes back out as sRGB bytes.**  ``CustomTextureCreate``
  builds its target as ``new RenderTexture(w, h, 0, ARGB32)`` — the four-argument
  overload, so ``RenderTextureReadWrite.Default``, which in a linear project is
  sRGB — and ``RebuildTextureAndSetMaterial`` wraps the ``Graphics.Blit`` in
  ``GL.sRGBWrite = true`` (restoring the previous value after).  So the bytes
  the game keeps are sRGB-encoded, exactly like the PNG this project bakes.

That last point is what makes this change land as one piece instead of moving
the error somewhere else.  The game has a **real encode boundary** at the end of
each create pass, so transcribing the pass correctly means: decode each input by
its own flag, do the arithmetic in linear, encode the result back to sRGB.  What
comes out is the same byte array Origin stores, and everything downstream of it —
the forward passes, the ramps, the outline, all calibrated in sRGB — sees
exactly what it saw before.

**The forward passes have their own boundary, and it is the camera.**  §7.85
recorded that they had none and left them in sRGB.  That was wrong, and the
proof is in the game's own image effect: ``Hidden/Amplify Color/Base`` opens by
sampling ``_MainTex`` and immediately running ``max(1.055·x^(1/2.4) − 0.055, 0)``
— ``LinearToGammaSpace``.  A pass only converts what it just sampled if what it
sampled was linear, so the camera target it reads is sRGB, exactly like the
create passes' render texture.  The forward half therefore lands the same way:
decode each map by its own flag, convert every ``Color`` property, shade in
linear, and let the framebuffer encode.  That is §7.126; the conversions live in
:mod:`gl.shaders` and :mod:`gl.framebuffer` because they are the GPU's, not
NumPy's — an 8-bit texture fetch has to decode **before** the bilinear filter,
which no array pass can reproduce.

Alpha is never converted, in either direction.  It is coverage, not colour, and
the GPU's sRGB conversion has always been RGB-only.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

# ``Texture2D.m_ColorSpace``.  Confirmed by the one group that cannot be the
# other way round: all 634 ``*_n`` normal maps in the baked library read 0, and
# a normal map is never sRGB.
LINEAR = 0
SRGB = 1


class Map(NamedTuple):
    """Texture bytes together with the colour space they are in.

    A pair rather than an attribute on the array: the flag has to survive being
    written to a PNG and read back, and it has to be *impossible* to drop
    silently.  A loader that forgets to carry it fails at the call — an array
    that quietly lost an attribute would fall back on whatever default the
    reader picked, which is the failure this whole change is about.
    """
    px: np.ndarray
    srgb: bool
    # ``m_TextureSettings.m_WrapMode`` (0 Repeat, 1 Clamp) when the map came
    # off a Unity asset; ``None`` for images that carry no such setting
    wrap: int | None = None

    @property
    def space(self) -> int:
        return SRGB if self.srgb else LINEAR


def pair(px, srgb) -> Map | None:
    """``Map`` or ``None`` — the shape every resolver returns."""
    return None if px is None else Map(px, bool(srgb))


def pixels(m) -> np.ndarray | None:
    """The bytes of a :class:`Map`, for the paths that only move them around."""
    return None if m is None else m.px


def of(piece, field: str) -> Map | None:
    """One of a piece's maps as a :class:`Map`, or ``None``.

    ``None`` covers both "this piece has no such map" — normal, the shader
    samples its declared default — and "it has one but no ``m_ColorSpace``",
    which is a resolution failure.  They are the same answer on purpose: the
    composite then falls to the shader's own default, which is a white or
    unpainted surface the caller already reports, instead of to a colour picked
    here.  Nothing baked at format 13 can hit the second case; a piece built
    some other way and missing the flag has genuinely not been resolved.
    """
    px = getattr(piece, field, None)
    if px is None:
        return None
    space = (getattr(piece, "map_cs", None) or {}).get(field)
    return None if space is None else Map(px, int(space) == SRGB)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """The GPU's sRGB texture fetch, exactly.

    The same piecewise curve the sRGB spec defines and Unity's
    ``Mathf.GammaToLinearSpace`` implements.  A plain ``x ** 2.2`` is off by up
    to 0.008 near black — small in absolute terms and *not* small where it
    matters, because ``create_eye``'s Vivid Light divides by ``2 * (1 - blend)``
    and the iris map's dark end is exactly where that denominator collapses.
    """
    return np.where(x <= 0.04045, x / 12.92,
                    np.power((x + 0.055) / 1.055, 2.4))


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """The GPU's sRGB write, exactly — the inverse of :func:`_srgb_to_linear`."""
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(np.maximum(x, 0.0), 1.0 / 2.4) - 0.055)


# The 8-bit fetch is a lookup, not a curve evaluation: a 2048² body map is four
# million texels and ``np.power`` over that is seconds, per map, per card.
_LUT = _srgb_to_linear(np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)


def decode(px: np.ndarray, srgb: bool) -> np.ndarray:
    """Texture bytes → the float RGBA the shader samples, alpha untouched.

    *srgb* is the asset's own ``m_ColorSpace``, and it is a required argument
    rather than a default because the local library says the two cases are
    47/53 on colour masks — a default would be a coin toss dressed up as a
    convention (CLAUDE.md 一之一).
    """
    a = np.asarray(px)
    if a.dtype != np.uint8:
        raise TypeError(f"texture bytes must be uint8, got {a.dtype}")
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"expected HxWx3 or HxWx4, got {a.shape}")
    out = np.empty(a.shape[:2] + (4,), np.float32)
    rgb = _LUT[a[..., :3]] if srgb else (a[..., :3].astype(np.float32) / 255.0)
    out[..., :3] = rgb
    out[..., 3] = (a[..., 3].astype(np.float32) / 255.0 if a.shape[2] == 4
                   else 1.0)
    return out


def sampled_bytes(m: "Map | None") -> np.ndarray | None:
    """The values a shader's fetch returns, back as bytes — for a threshold.

    A cut-out (``_AlphaMask``) compares the **sampled** value against 0.5, and
    an sRGB-flagged map is decoded by that fetch: byte 128 samples as 0.216,
    and 0.5 is byte 188.  The CPU-side cut-outs work on bytes, so an sRGB mask
    is handed over as its decoded RGB (alpha untouched) and a linear one as it
    is (§7.386).
    """
    if m is None:
        return None
    if not m.srgb:
        return m.px
    out = np.array(m.px, dtype=np.uint8, copy=True)
    out[..., :3] = np.clip(decode(m.px, True)[..., :3] * 255.0 + 0.5,
                           0, 255).astype(np.uint8)
    return out


def encode(rgb: np.ndarray) -> np.ndarray:
    """Linear float → the sRGB bytes the ARGB32 render target stores.

    The clamp is the render target's, not ours: ``ARGB32`` is unorm, so a pass
    that extrapolates past 1 (``create_eye`` writes a ``mad``, not a saturating
    lerp, and 99 local cards hold ``gradBlend`` outside [0, 1]) is cut off by
    the write rather than by the arithmetic.
    """
    return np.clip(_linear_to_srgb(np.clip(np.asarray(rgb, np.float32), 0.0, 1.0))
                   * 255.0 + 0.5, 0, 255).astype(np.uint8)


def colour_to_linear(c) -> np.ndarray:
    """A shader ``Color`` property as the GPU sees it in a linear project.

    RGB through Unity's ``GammaToLinearSpace``, alpha verbatim.  The ``> 1``
    branch is Unity's own: above 1 it drops the piecewise curve for a plain
    2.2 power.

    **The one link in this chain that is not read off an asset.**  That the
    engine converts a ``Color`` property comes from the shader's own property
    table (type ``Color``, no ``[Gamma]``), but whether it also converts the one
    declared ``[HDR]`` — ``create_body._Color``, the skin tone — lives in
    Unity's native code and cannot be disassembled.  It is treated the same as
    the others because the game's own output says so: the body's four maps are
    all linear-flagged and ``cf_body_00_t`` is near-white, so converting leaves
    the skin where it was (``reference/``'s neck patch reads +2.4, unchanged)
    while not converting would raise 0.9 to 0.958 — about 15 levels, which that
    patch would have caught.  See PROGRESS §7.85 五.
    """
    v = np.asarray(c, np.float64).reshape(-1)
    out = np.array([0.0, 0.0, 0.0, 1.0])
    out[:min(4, v.size)] = v[:4]
    rgb = out[:3]
    out[:3] = np.where(rgb <= 0.04045, rgb / 12.92,
                       np.where(rgb < 1.0,
                                np.power((np.abs(rgb) + 0.055) / 1.055, 2.4),
                                np.power(np.maximum(rgb, 0.0), 2.2)))
    return out.astype(np.float32)


def linear_to_colour(c) -> np.ndarray:
    """The ``Color`` property :func:`colour_to_linear` reads as linear *c*.

    RGB within 0‥1 (the piecewise curve's own range; above it the forward
    conversion switches to a plain power and this is not its inverse), alpha
    verbatim.
    """
    v = np.asarray(c, np.float64).reshape(-1)
    if v.size not in (3, 4) or not np.isfinite(v).all() or np.any(v[:3] < 0) or np.any(v[:3] > 1):
        raise ValueError("a linear colour needs three finite components within 0..1")
    out = np.array([0.0, 0.0, 0.0, 1.0])
    out[:v.size] = v
    out[:3] = _linear_to_srgb(out[:3])
    return out.astype(np.float32)
