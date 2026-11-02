"""MoreAccessories (``moreAccessories``) — the accessory slots past the base 20.

Origin's own card has room for **20** accessories per coordinate
(``ChaFileAccessory.parts``).  Joan6694's plugin keeps the rest in the card's
extended data and splices them in on load, so a card wearing 30 accessories
looks like a card wearing 20 to anything that reads only ``Coordinate``.

Transcribed from ``BepInEx/plugins/MoreAccessories.dll``
(``MoreAccessories.OnActualCharaLoad``) rather than inferred (``CLAUDE.md``
一之二)::

    var data = ExtendedSave.GetExtendedDataById(file, "moreAccessories");
    if (data == null) return;
    if (data.version == 1)
        foreach (var c in file.coordinate)          // trim anything that leaked
            if (c.accessory.parts.Length > 20)
                c.accessory.parts = c.accessory.parts.Take(20).ToArray();
    …
    var additional = new CharAdditionalData(data);
    foreach (var kv in additional.rawAccessoriesInfos)
        if (kv.Key < file.coordinate.Length)
            file.coordinate[kv.Key].accessory.parts =
                file.coordinate[kv.Key].accessory.parts.Concat(kv.Value).ToArray();

**Concat** is the whole mechanism for a version-1 section: the extra slots are
ordinary ``ChaFileAccessory.PartsInfo`` entries appended after the base twenty,
so slot ``20 + i`` is the *i*-th entry of this coordinate's ``accessorySet``.

The plugin the install ships (2.0.21.1) saves **version 2**: the parts array
itself is longer than twenty and the game's own serialiser writes it, so the
extra slots are already in the card's ``Coordinate`` block.  The XML is still
written (``BackwardCompatibility``) but the loader never reads it back —
``OnActualCharaLoad`` branches on the version (§7.487)::

    if (data.version == 1) trim every parts array to 20
    if (data.version == 2) { Studio: ShowAccessories; return; }   // XML unread
    concat the XML sets                                           // 1 and 0

— so :func:`accessory_slots` is the one place that decides where slot ``20+i``
comes from.  That is
also the number Sideloader's resolution table uses —
``outfit0.accessory25.ChaFileAccessory.PartsInfo.id`` occurs on local cards —
so :func:`resolve.resolve_accessory` needs nothing new.

The payload is XML in ``additionalAccessories``, one ``accessorySet`` per
coordinate, and every field of ``PartsInfo`` is an attribute of its own
(``addMove00x`` … ``addMove12z`` are the two move groups' position, rotation and
scale).  Measured over the local library: **1,129** cards carry the section,
**240** have at least one real entry (``type`` 120 is 「なし」, and most sets are
padding), **162** have one in the coordinate they are actually wearing, and those
add up to **2,237** accessories — one card wears 164 (§7.364).
"""
from __future__ import annotations

#: ``KKEx`` section key, from the plugin's ``GUID``.
EXT_KEY = "moreAccessories"

#: What ``ChaFileAccessory.parts`` holds without the plugin.
BASE_SLOTS = 20

#: ``ao_none`` — the plugin writes a full set and marks the empty ones with it.
NONE_TYPE = 120


def _f(attrib: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(attrib.get(key, default))
    except (TypeError, ValueError):
        return default


def _vec3(attrib: dict, prefix: str, default=(0.0, 0.0, 0.0)) -> list:
    return [_f(attrib, prefix + ax, d) for ax, d in zip("xyz", default)]


def _part(attrib: dict) -> dict:
    """One XML ``accessory`` element as a ``PartsInfo``-shaped dict.

    Same keys and same nesting as the card's own accessory entries, because the
    game splices these into the same array — ``addMove`` keeps msgpack's
    dimensions-first form (``[2, 3, [pos, rot, scale, pos, rot, scale]]``) so
    every reader downstream (``attach.card_move_locals``, the colour chain)
    works unchanged.
    """
    return {
        "type": int(_f(attrib, "type", NONE_TYPE)),
        "id": int(_f(attrib, "id", -1)),
        "parentKey": str(attrib.get("parentKey") or ""),
        "addMove": [2, 3, [
            _vec3(attrib, "addMove00"),
            _vec3(attrib, "addMove01"),
            _vec3(attrib, "addMove02", (1.0, 1.0, 1.0)),
            _vec3(attrib, "addMove10"),
            _vec3(attrib, "addMove11"),
            _vec3(attrib, "addMove12", (1.0, 1.0, 1.0)),
        ]],
        "color": [[_f(attrib, f"color{i}{c}", 1.0) for c in "rgba"]
                  for i in range(4)],
        "hideCategory": int(_f(attrib, "hideCategory", 0)),
        "noShake": str(attrib.get("noShake", "false")).lower() == "true",
    }


def section_version(card) -> int | None:
    """``PluginData.version`` of the card's ``moreAccessories`` section; ``None`` without one."""
    blob = ((getattr(card, "sections", None) or {}).get("KKEx") or {}).get(EXT_KEY)
    if isinstance(blob, list):
        return next((int(x) for x in blob if isinstance(x, int) and not isinstance(x, bool)), None)
    if isinstance(blob, dict):
        try:
            return int(blob.get("version"))
        except (TypeError, ValueError):
            return None
    return None


def accessory_slots(card, coordinate: int, native: list) -> list:
    """The accessory array ``OnActualCharaLoad`` leaves for *coordinate*.

    *native* is the coordinate's own ``parts``.  No section: the plugin
    returns before touching it.  Version 2: the array as saved — its slots past
    twenty are the extra accessories and the XML is not read.  Version 1: the
    first twenty, then the XML set.  Any other version: the array as saved,
    then the XML set.  Index ``i`` of the answer is slot ``i`` — the number the
    card's UAR table and MaterialEditor rows use.
    """
    native = list(native or [])
    ver = section_version(card)
    if ver is None or ver == 2:
        return native
    slots = native[:BASE_SLOTS] if ver == 1 else native
    extra = extra_parts(card, coordinate)
    if extra:
        # Pad to twenty first: the plugin appends to the game's array, which
        # ``ChaFileAccessory.MemberInit`` makes twenty long whatever a card filled.
        slots = slots + [None] * (BASE_SLOTS - len(slots)) + extra
    return slots


def extra_parts(card, coordinate: int) -> list[dict]:
    """The accessory entries this card adds to *coordinate* past slot 20.

    In the plugin's own order, including its 「なし」 padding (``type`` 120), so
    the caller's index is the slot number the card's UAR table and
    MaterialEditor rows use: ``BASE_SLOTS + i``.  Empty for a card without the
    section, and a payload that will not parse is **said** rather than dropped
    silently (一之一) — the alternative is a character quietly wearing twenty of
    its thirty accessories.
    """
    import xml.etree.ElementTree as ET

    sec = (getattr(card, "sections", None) or {}).get("KKEx") or {}
    blob = sec.get(EXT_KEY)
    data = None
    if isinstance(blob, list):
        for item in blob:
            if isinstance(item, dict) and "additionalAccessories" in item:
                data = item.get("additionalAccessories")
                break
    elif isinstance(blob, dict):
        data = blob.get("additionalAccessories")
    if not isinstance(data, str) or "<accessory" not in data:
        return []
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        print(f"[moreacc] additionalAccessories 解析不了：{exc}")
        return []
    want = str(int(coordinate))
    out: list[dict] = []
    for node in root.findall("accessorySet"):
        if str(node.get("type")) != want:
            continue
        for acc in node.findall("accessory"):
            out.append(_part(acc.attrib))
        break
    return out
