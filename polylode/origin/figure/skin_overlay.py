"""Overlay Mods' skin overlays (``KoiSkinOverlayX``, card data ``KSOX``).

Read from the local ``BepInEx/plugins/KK_OverlayMods.dll`` (6.3):

* **Where it is stored.**  ``KKEx["KSOX"]`` is ``(version, data)``.  From
  version 2 on, ``data["Lookup"]`` is MessagePack
  ``{coordinate: {TexType: texture id}}`` and every image is stored once as
  ``data["_TextureID_<id>"]`` (``TextureStorage``, deduplicated).  Version ≤ 1
  (``ReadLegacyData``) keys the PNG by the ``TexType`` *name* and the same set
  goes to every coordinate; ``EyeOver``／``EyeUnder`` become the ``L`` and ``R``
  pair there.
* **Which coordinate.**  ``GetCurrentOverlayTextures`` asks for the worn
  coordinate; when the lookup has none for it and the data is not per
  coordinate (``IsPerCoord``: every stored set sequence-equal to the first),
  a copy of the first coordinate's set, otherwise nothing.
* **Where.**  A transpiler on ``CustomTextureCreate.RebuildTextureAndSetMaterial``
  replaces its ``Graphics.Blit(texMain, createTex, matCreate, 0)``: the
  *under*lays of the face / body / each eye are composited onto a copy of
  ``texMain`` before the create pass samples it, the *over*lays onto
  ``createTex`` after.  Both with the plugin's ``Unlit/composite``
  (:func:`clothes_overlay.composite`), in list order.
"""
from __future__ import annotations

import io

import numpy as np

from .. import colorspace as CS
from . import create_tex as CT

GUID = "KSOX"
LOOKUP_KEY = "Lookup"
ID_PREFIX = "_TextureID_"

#: ``KoiSkinOverlayX.TexType`` (``origin_il_dump --enum``)
TEX_TYPES = {0: "Unknown", 1: "BodyOver", 2: "FaceOver", 3: "BodyUnder",
             4: "FaceUnder", 5: "EyeUnder", 6: "EyeOver", 7: "EyeUnderL",
             8: "EyeOverL", 9: "EyeUnderR", 10: "EyeOverR",
             20: "EyebrowUnder", 30: "EyelineUnder"}
_BY_NAME = {v: k for k, v in TEX_TYPES.items()}
#: the layers :mod:`body_head` composites; the rest (``EyebrowUnder``,
#: ``EyelineUnder``, and a bare ``EyeOver``／``EyeUnder`` outside the legacy
#: format) are read and reported, not drawn
APPLIED = frozenset({"FaceOver", "FaceUnder", "BodyOver", "BodyUnder",
                     "EyeOverL", "EyeOverR", "EyeUnderL", "EyeUnderR"})


def _png(data):
    from PIL import Image
    im = Image.open(io.BytesIO(bytes(data))).convert("RGBA")
    return np.asarray(im, np.uint8).copy()


def read(card, coordinate: int = 0) -> tuple[dict, list, str]:
    """``({TexType name: RGBA bytes}, notes, reason)`` for the worn coordinate.

    *reason* is non-empty when the section exists but cannot be read — that is
    reported, never turned into "no overlays" (``CLAUDE.md`` 一之一).
    """
    from .bonemod import _msgpack_lz4
    try:
        ext = (card.sections.get("KKEx") or {}) if card is not None else {}
    except AttributeError:
        return {}, [], ""
    sec = ext.get(GUID)
    if sec is None:
        return {}, [], ""
    if not isinstance(sec, (list, tuple)) or len(sec) < 2 or not isinstance(sec[1], dict):
        return {}, [], "KSOX 區段不是 (version, data) 這個形狀"
    version, data = sec[0], sec[1]
    notes: list = []
    out: dict = {}
    try:
        version = int(version)
    except (TypeError, ValueError):
        return {}, [], f"KSOX 的版本讀不懂（{version!r}）"
    if version <= 1:
        for name in TEX_TYPES.values():
            b = data.get(name)
            if name == "Unknown" or not isinstance(b, (bytes, bytearray)) or not b:
                continue
            try:
                px = _png(b)
            except Exception as exc:                          # noqa: BLE001
                notes.append(f"{name}（PNG 解不開：{exc}）")
                continue
            if name in ("EyeOver", "EyeUnder"):
                out[name + "L"] = out[name + "R"] = px
            else:
                out[name] = px
        return out, notes, ""
    blob = data.get(LOOKUP_KEY)
    if not blob:
        return {}, [], ""
    try:
        lookup = _msgpack_lz4(blob)
    except Exception as exc:                                  # noqa: BLE001
        return {}, [], f"KSOX 的 Lookup 解不開（{exc}）"
    if not isinstance(lookup, dict) or not lookup:
        return {}, [], ""
    per = lookup.get(str(int(coordinate)), lookup.get(int(coordinate)))
    if per is None:
        firsts = [list((v or {}).items()) for v in lookup.values()]
        # ``IsPerCoord`` false (every stored set equal *in order* — the IL
        # compares the dictionaries as sequences) ⇒ the first set stands in
        if all(v == firsts[0] for v in firsts):
            per = next(iter(lookup.values()))
    for tt, tid in (per or {}).items():
        try:
            name = TEX_TYPES.get(int(tt), None) if str(tt).lstrip("-").isdigit() else str(tt)
        except (TypeError, ValueError):
            name = None
        if name is None or name not in _BY_NAME:
            notes.append(f"TexType {tt!r} 不認得")
            continue
        b = data.get(f"{ID_PREFIX}{tid}")
        if not isinstance(b, (bytes, bytearray)) or not b:
            notes.append(f"{name} 指到不存在的貼圖 {tid}")
            continue
        try:
            out[name] = _png(b)
        except Exception as exc:                              # noqa: BLE001
            notes.append(f"{name}（PNG 解不開：{exc}）")
    return out, notes, ""


def _blend(lin_rgb, alpha, layers, uv):
    for layer in layers:
        # a stored overlay is PNG bytes (sRGB, ``Util.TextureFromBytes``); a
        # plugin's ``AdditionalTexture`` arrives as a Map with its own flag
        m = layer if isinstance(layer, CS.Map) else CS.Map(layer, True)
        over = CT.sample(m, *uv)
        a = over[..., 3:4]
        lin_rgb = lin_rgb + (over[..., :3] - lin_rgb) * a
        alpha = np.maximum(alpha, over[..., 3])
    return lin_rgb, alpha


def composite_baked(baked: np.ndarray, layers, uv=None) -> np.ndarray:
    """*layers* composited onto a create-pass result (sRGB bytes) — the overlay step.

    *uv* is where each of our texels sits in the render target the plugin
    blits into; the default is the identity.  The iris composite bakes the
    gaze window in (:func:`body_head.eye_texture`), so there it is that window
    — the overlay was drawn on the unwindowed ``createTex``.
    """
    if baked is None or not layers:
        return baked
    h, w = baked.shape[:2]
    lin = CS.decode(baked, True)
    rgb, alpha = _blend(lin[..., :3], lin[..., 3], layers,
                        uv if uv is not None else CT._grid((h, w)))
    return CT._u8(rgb, alpha)


def composite_map(m: "CS.Map", layers) -> "CS.Map":
    """*layers* composited onto a copy of *m* — the underlay step.

    ``Graphics.Blit(texMain, trt)`` copies ``texMain`` into a render target of
    the destination's (sRGB) format: an sRGB map is decoded and re-encoded, a
    linear one is written as its raw value and then *read back* as that value
    ⇒ the copy holds ``decode(texMain)`` in either case.  The composite then
    runs in linear and the create pass samples the result as an sRGB map.
    """
    if m is None or not layers:
        return m
    h, w = m.px.shape[:2]
    lin = CS.decode(m.px, m.srgb)
    rgb, alpha = _blend(lin[..., :3], lin[..., 3], layers, CT._grid((h, w)))
    return CS.Map(CT._u8(rgb, alpha), True, m.wrap)
