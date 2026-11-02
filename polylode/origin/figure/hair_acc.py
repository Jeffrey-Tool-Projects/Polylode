"""HairAccessoryCustomizer (``com.deathweasel.bepinex.hairaccessorycustomizer``).

A hair accessory is an accessory whose prefab root carries a
``ChaCustomHairComponent`` — a ponytail, a wig, a braid sold as an accessory.
The vanilla game colours it like any accessory (the card's accessory colours
into ``rendNormal``); this plugin re-colours it like hair.

Read from the local ``BepInEx/plugins/KK_Plugins/KK_HairAccessoryCustomizer.dll``:

* **Where it is stored.**  ``KKEx[GUID]["HairAccessories"]`` is MessagePack
  ``{coordinate: {slot: HairAccessoryInfo}}``; the info's keys are
  ``HairGloss``, ``ColorMatch``, ``OutlineColor``, ``AccessoryColor`` (both
  ``Color`` as four floats) and ``HairLength``.
* **When.**  ``HairAccessoryController.LoadData`` (a coroutine off
  ``OnReload``) deserialises it and calls ``UpdateAccessories(true)`` ⇒
  ``UpdateAccessory(slot, true)`` for every slot of the worn coordinate.
* **What** (``UpdateAccessory``, outside the maker; it returns early unless the
  slot's object has a ``ChaCustomHairComponent`` with ``rendHair`` and a
  ``ChaAccessoryComponent`` with ``rendNormal``)::

      if ColorMatch:
          for r in hairComp.rendHair:           # sharedMaterial, where present
              _Color  = hair.parts[0].baseColor
              _Color2 = hair.parts[0].startColor
              _Color3 = hair.parts[0].endColor
      for r in hairComp.rendHair:
          _HairGloss = HairGloss ? chaControl.texHairGloss : null
          _LineColor = ColorMatch ? hair.parts[0].outlineColor : OutlineColor
      for r in hairComp.rendAccessory:
          _Color = _Color2 = _Color3 = AccessoryColor
      hairComp.lengthRate = HairLength

  ``texHairGloss`` is the character's gloss sheet — the one
  ``ChangeSettingHairGlossMask`` hands the real hair.
"""
from __future__ import annotations

GUID = "com.deathweasel.bepinex.hairaccessorycustomizer"
DATA_KEY = "HairAccessories"


def _color(v):
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        try:
            return [float(x) for x in v[:4]] + [1.0] * (4 - len(v[:4]))
        except (TypeError, ValueError):
            return None
    return None


def _info(raw: dict) -> dict:
    return {
        "ColorMatch": bool(raw.get("ColorMatch", True)),
        "HairGloss": bool(raw.get("HairGloss", True)),
        "OutlineColor": _color(raw.get("OutlineColor")),
        "AccessoryColor": _color(raw.get("AccessoryColor")),
        "HairLength": float(raw.get("HairLength") or 0.0),
    }


def read(card, coordinate: int = 0) -> tuple[dict, str]:
    """``({slot: info}, reason)`` for *coordinate*.

    *reason* is non-empty when the section is there but cannot be read — a
    reported failure, not "no customisation" (``CLAUDE.md`` 一之一).
    """
    from ...chara.cardimport import _unpack

    try:
        ext = (card.sections.get("KKEx") or {}) if card is not None else {}
    except AttributeError:
        return {}, ""
    sec = ext.get(GUID)
    if sec is None:
        return {}, ""
    data = sec[1] if isinstance(sec, (list, tuple)) and len(sec) >= 2 else None
    if not isinstance(data, dict):
        return {}, "HairAccessoryCustomizer 區段不是 (version, data) 這個形狀"
    blob = data.get(DATA_KEY)
    if not blob:
        return {}, ""
    try:
        allc = _unpack(bytes(blob))
    except Exception as exc:                                  # noqa: BLE001
        return {}, f"HairAccessoryCustomizer 解不開（{exc}）"
    if not isinstance(allc, dict):
        return {}, "HairAccessoryCustomizer 不是 {座標: {槽: 設定}}"
    per = allc.get(int(coordinate), allc.get(str(int(coordinate))))
    out: dict = {}
    for slot, raw in (per or {}).items():
        try:
            out[int(slot)] = _info(raw if isinstance(raw, dict) else {})
        except (TypeError, ValueError):
            continue
    return out, ""
