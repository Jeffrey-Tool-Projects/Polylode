"""Overlay Mods' clothes overlays (``KoiClothesOverlayX``, card data ``KCOX``).

Read from the local ``BepInEx/plugins/KK_OverlayMods.dll`` (6.3):

* **Where it is stored.**  ``KKEx["KCOX"]`` → ``data["Overlays"]`` is
  MessagePack ``{coordinate: {clothes id: [png bytes, Override]}}``
  (``ClothesTexData``: ``[Key(0)] TextureBytes``, ``[Key(1)] Override``).
  The clothes id is the garment's GameObject name —
  ``ct_clothesTop`` … ``ct_shoes_outer`` for ``objClothes[0..8]``
  (``<ChangeClothesTopAsync>`` names them) and ``ct_top_parts_A/B/C`` for the
  combo top's sub-parts.  ``BodyMask``／``InnerMask``／``BraMask`` replace the
  body/inner/bra cut masks (:func:`mask_bytes`) and ``Colormask_<id>`` a colour
  mask; ``InnerMask`` and the colour masks are read and reported, not applied
  yet (``PROGRESS.md``).
* **The masks.**  A prefix on ``Material.SetTexture(int, Texture)``
  (``BodyMaskHook``): when the id is ``_AlphaMask`` and the material is
  ``customMatBody`` / a ``rendBra`` renderer's / a ``rendInner`` one's, and
  the character has a top (``objClothes[0] != null``), the value is replaced
  by the card's image.  ``ChangeClothesTopAsync`` calls that ``SetTexture``
  on the body **unconditionally** once the top is loaded (with
  ``texBodyAlphaMask``, null or not), and on each bra renderer whenever the
  bra is loaded ⇒ with a top on, the image *is* the mask.  It loads through
  the same ``ClothesTexData.Texture`` getter as the garments (sRGB, default
  wrap = Repeat); the row's ``_AlphaMask`` offset/scale still apply.
* **When.**  A postfix on ``ChaControl.ChangeCustomClothes`` —
  ``ApplyOverlays(clothesCtrl)``: after the garment's ``create_topN``
  composite is in its render target.
* **Where.**  ``GetApplicableRenderers`` walks ``rendNormal01, rendNormal02,
  rendAlpha01, rendAlpha02`` two at a time and stops at the first pair whose
  first array has a renderer: the first renderer of ``rendNormal01`` and of
  ``rendNormal02`` (or of the two alpha arrays).  For each it takes
  ``material.mainTexture as RenderTexture`` — **the shared composite**, so the
  overlay lands on every renderer that composite feeds — and ``return``s
  outright when one is not a render target.
* **What.**  ``Override`` first clears the target to ``Color.clear``; then
  ``Graphics.Blit(main, tmp, OverlayMat)`` + ``Blit(tmp, main)`` with the
  plugin's ``Unlit/composite`` (embedded ``composite.unity3d``, disassembled)::

      o.rgb = lerp(main.rgb, over.rgb, over.a)      o.a = max(main.a, over.a)

  on an sRGB render target (linear project), with the overlay a
  ``new Texture2D(2, 2, ARGB32, mipChain: false)`` from ``LoadImage`` — sRGB,
  no mips ⇒ one bilinear tap per target texel, in linear.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np

from .. import colorspace as CS
from . import create_tex as CT

GUID = "KCOX"
MAIN_IDS = ("ct_clothesTop", "ct_clothesBot", "ct_bra", "ct_shorts", "ct_gloves",
            "ct_panst", "ct_socks", "ct_shoes_inner", "ct_shoes_outer")
SUB_IDS = ("ct_top_parts_A", "ct_top_parts_B", "ct_top_parts_C")
MASK_IDS = ("BodyMask", "InnerMask", "BraMask")
#: the masks applied here; ``InnerMask`` lives on the combo top's inner shirt,
#: which :func:`load_mesh.load_combo_top` cuts before any card data reaches it
APPLIED_MASKS = ("BodyMask", "BraMask")
COLORMASK_PREFIX = "Colormask_"


@dataclass
class Overlay:
    """One ``ClothesTexData``: the decoded image (RGBA bytes) and ``Override``."""
    rgba: np.ndarray | None
    override: bool


@dataclass
class ClothesOverlays:
    """The worn coordinate's overlays, keyed by clothes id."""
    by_id: dict = field(default_factory=dict)
    unapplied: list = field(default_factory=list)   # ids read but not drawn yet
    reason: str = ""                                # non-empty: section unreadable

    def for_garment(self, slot_index: int, combo_sub: int | None = None):
        """The overlay for ``objClothes[slot_index]`` or its combo sub-part."""
        if combo_sub is not None and 0 <= combo_sub < len(SUB_IDS):
            return self.by_id.get(SUB_IDS[combo_sub])
        if 0 <= slot_index < len(MAIN_IDS):
            return self.by_id.get(MAIN_IDS[slot_index])
        return None


def read(card, coordinate: int = 0) -> ClothesOverlays:
    """The card's ``KCOX`` overlays for *coordinate*.

    An unreadable section comes back with a ``reason`` — not as "no overlays",
    which looks the same on screen (``CLAUDE.md`` 一之一).
    """
    from .bonemod import _msgpack_lz4
    try:
        ext = (card.sections.get("KKEx") or {}) if card is not None else {}
    except AttributeError:
        return ClothesOverlays()
    data = ext.get(GUID)
    if data is None:
        return ClothesOverlays()
    if not isinstance(data, (list, tuple)) or len(data) < 2 or not isinstance(data[1], dict):
        return ClothesOverlays(reason="KCOX 區段不是 (version, data) 這個形狀")
    blob = data[1].get("Overlays")
    if not blob:
        return ClothesOverlays()
    try:
        allc = _msgpack_lz4(blob)
    except Exception as exc:                                  # noqa: BLE001
        return ClothesOverlays(reason=f"KCOX 的 Overlays 解不開（{exc}）")
    if not isinstance(allc, dict):
        return ClothesOverlays(reason="KCOX 的 Overlays 不是字典")
    # ``CoordinateType`` keys come back as strings from this card format
    per = allc.get(str(int(coordinate)), allc.get(int(coordinate))) or {}
    out = ClothesOverlays()
    for cid, rec in per.items():
        if not isinstance(rec, (list, tuple)) or not rec:
            continue
        png = rec[0]
        override = bool(rec[1]) if len(rec) > 1 else False
        if not png and not override:
            continue                     # ``IsEmpty``: nothing to do
        cid = str(cid)
        if (cid in MASK_IDS and cid not in APPLIED_MASKS) or cid.startswith(COLORMASK_PREFIX):
            out.unapplied.append(cid)
            continue
        rgba = None
        if png:
            try:
                from PIL import Image
                im = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
                rgba = np.asarray(im, np.uint8).copy()
            except Exception as exc:                          # noqa: BLE001
                out.unapplied.append(f"{cid}（PNG 解不開：{exc}）")
                continue
        out.by_id[cid] = Overlay(rgba, override)
    return out


def mask_bytes(overlay: "Overlay | None"):
    """The card's replacement ``_AlphaMask`` as the bytes a cut-out thresholds.

    ``None`` without an image — an ``Override`` flag on a mask id carries no
    meaning of its own (``GetBodyMask`` reads ``.Texture`` only).  sRGB, so the
    decoded value is what the shader compares against 0.5 (§7.386).
    """
    if overlay is None or overlay.rgba is None:
        return None
    return CS.sampled_bytes(CS.Map(overlay.rgba, True))


def applicable_sets(roles) -> list[int]:
    """Which composites (``ct[i]``) ``ApplyOverlays`` writes into, in order.

    *roles* are the ``ChaClothesComponent`` roles present on one garment.  The
    walk stops at the first renderer pair; a pair's second renderer is taken
    only when its array has one.  ``rendAlpha02`` is not handed a composite by
    ``ChangeCustomClothes`` (:data:`load_mesh._ROLE_SET` keeps it on the
    prefab's own map), so it is not a render target and the plugin ``return``s
    there — nothing after it is written.
    """
    roles = set(roles)
    if "normal01" in roles:
        return [0, 1] if "normal02" in roles else [0]
    if "alpha01" in roles:
        return [0]
    return []


def composite(baked: np.ndarray, overlay: Overlay, times: int = 1) -> np.ndarray:
    """*baked* (a ``create_topN`` result, sRGB bytes) with *overlay* blitted *times*."""
    if times <= 0 or overlay is None:
        return baked
    h, w = baked.shape[:2]
    out = np.array(baked, np.uint8, copy=True)
    over = None
    if overlay.rgba is not None:
        u, v = CT._grid((h, w))
        over = CT.sample(CS.Map(overlay.rgba, True), u, v)
    for _ in range(times):
        if overlay.override:
            out[...] = 0                               # ``GL.Clear(Color.clear)``
        if over is None:
            continue
        main = CS.decode(out, True)
        a = over[..., 3:4]
        rgb = main[..., :3] + (over[..., :3] - main[..., :3]) * a
        alpha = np.maximum(main[..., 3], over[..., 3])
        out = CT._u8(rgb, alpha)
    return out
