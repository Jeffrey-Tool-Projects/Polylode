"""Pushup (``com.deathweasel.bepinex.pushup``) — the bust a worn bra or top shapes.

Read from the local ``BepInEx/plugins/KK_Plugins/KK_Pushup.dll``:

* **The card holds the unpushed body.**  ``OnCardBeingSaved`` writes
  ``BaseData`` back into ``fileBody`` (``MapBodyInfoToChaFile``) before the
  card is written, and also stores it as ``Pushup_BodyData``.  So what every
  reader of ``shapeValueBody`` sees is the bare chest, and the push is
  computed on load.  In the main game ``OnReload`` takes ``BaseData`` from
  ``Pushup_BodyData`` (``GetCurrentGameMode() == 3``), elsewhere from
  ``fileBody``; the two are the same save.
* **Per coordinate, per garment.**  ``Pushup_BraData`` / ``Pushup_TopData``
  are MessagePack ``{coordinate: ClothData}``; a ``ClothData`` is a
  ``BodyData`` (the twelve bust values) plus ``Firmness``, ``Lift``,
  ``PushTogether``, ``Squeeze``, ``CenterNipples``, ``EnablePushup``,
  ``FlattenNipples``, ``UseAdvanced``.
* **When it applies** (``CurrentlyWearing``): the bra counts when
  ``IsClothesStateKind(2)`` (the slot holds a garment),
  ``clothesState[2] == 0`` and its data's ``EnablePushup``; the top the same
  with slot 0.  Neither ⇒ ``BaseData``.
* **What** (``CalculatePush`` / ``CalculatePushFromClothes``): one garment
  alone uses its own data (``UseAdvanced`` ⇒ its stored body values are copied
  whole); both ⇒ the top if it is advanced, else the bra if it is, else a
  fresh ``ClothData`` whose five numbers are the ``Math.Max`` of the two and
  ``FlattenNipples`` the OR.  Then, from the base::

      Softness           = min(base, 1 - Firmness)
      VerticalPosition   = max(base, Lift)
      HorizontalAngle    = min(base, 1 - PushTogether)
      HorizontalPosition = min(base, 1 - PushTogether)
      if 1 - Squeeze < base.Depth:
          Depth = 1 - Squeeze
          Size  = base.Size + (base.Depth - (1 - Squeeze)) / 10
      else: Depth, Size = base
      NippleDepth, AreolaDepth = 0, 0 if FlattenNipples else base
      NippleWidth, Weight, Roundness = base
      v = 0.5 - base.VerticalAngle
      VerticalAngle      = 0.5 - (v - v * CenterNipples)

  (each comparison is the IL's ``bge.un`` / ``ble.un``, so a tie keeps the
  base).  ``MapBodyInfoToChaFile`` writes ``bustSoftness`` / ``bustWeight`` and
  ``shapeValueBody[4..13]``.
"""
from __future__ import annotations

GUID = "com.deathweasel.bepinex.pushup"

#: ``MapBodyInfoToChaFile``: ``shapeValueBody`` index of each ``BodyData`` field
SHAPE_INDEX = {"Size": 4, "VerticalPosition": 5, "HorizontalAngle": 6,
               "HorizontalPosition": 7, "VerticalAngle": 8, "Depth": 9,
               "Roundness": 10, "AreolaDepth": 11, "NippleWidth": 12,
               "NippleDepth": 13}
BODY_FIELDS = tuple(SHAPE_INDEX) + ("Softness", "Weight")
#: the ``ClothData`` numbers ``CalculatePush`` takes the max of
PUSH_FIELDS = ("Firmness", "Lift", "PushTogether", "Squeeze", "CenterNipples")

#: ``ChaListDefine`` clothes kinds ``IsClothesStateKind`` is asked about
TOP_KIND, BRA_KIND = 0, 2


def base_from_body(body: dict) -> dict | None:
    """``new BodyData(fileBody)``: the twelve values off a card's body block."""
    sv = (body or {}).get("shapeValueBody")
    if not sv or len(sv) <= max(SHAPE_INDEX.values()):
        return None
    out = {k: float(sv[i]) for k, i in SHAPE_INDEX.items()}
    out["Softness"] = float(body.get("bustSoftness") or 0.0)
    out["Weight"] = float(body.get("bustWeight") or 0.0)
    return out


def read(card) -> tuple[dict, dict, dict | None, str]:
    """``(bra, top, base, reason)``: ``{coordinate: ClothData}`` twice and
    ``Pushup_BodyData`` (or ``None``).  *reason* is non-empty when the
    section is there but cannot be read (一之一)."""
    from ...chara.cardimport import _unpack

    try:
        sec = ((card.sections.get("KKEx") or {}) if card is not None else {}).get(GUID)
    except AttributeError:
        return {}, {}, None, ""
    if sec is None:
        return {}, {}, None, ""
    data = sec[1] if isinstance(sec, (list, tuple)) and len(sec) >= 2 else None
    if not data:
        return {}, {}, None, ""        # ``SetExtendedData(null)``: nothing enabled
    if not isinstance(data, dict):
        return {}, {}, None, "Pushup 區段不是 (version, data) 這個形狀"
    out = []
    try:
        for key in ("Pushup_BraData", "Pushup_TopData"):
            blob = data.get(key)
            table = _unpack(bytes(blob)) if blob else {}
            out.append({int(k): v for k, v in (table or {}).items()
                        if isinstance(v, dict)})
        blob = data.get("Pushup_BodyData")
        base = _unpack(bytes(blob)) if blob else None
    except Exception as exc:                                  # noqa: BLE001
        return {}, {}, None, f"Pushup 解不開（{exc}）"
    return out[0], out[1], base if isinstance(base, dict) else None, ""


def _num(d: dict, k: str) -> float:
    return float(d.get(k) or 0.0)


def push_from_clothes(base: dict, cloth: dict, advanced: bool) -> dict:
    """``CalculatePushFromClothes(cloth, advanced)`` from *base*."""
    if advanced:
        # ``BodyData.CopyTo`` (the argument is a BodyData): the garment's own
        # twelve stored values, whole
        return {k: _num(cloth, k) for k in BODY_FIELDS}
    cur = dict(base)
    firm = 1.0 - _num(cloth, "Firmness")
    if not firm >= base["Softness"]:
        cur["Softness"] = firm
    lift = _num(cloth, "Lift")
    if not lift <= base["VerticalPosition"]:
        cur["VerticalPosition"] = lift
    together = 1.0 - _num(cloth, "PushTogether")
    for k in ("HorizontalAngle", "HorizontalPosition"):
        if not together >= base[k]:
            cur[k] = together
    squeeze = 1.0 - _num(cloth, "Squeeze")
    if not squeeze >= base["Depth"]:
        cur["Depth"] = squeeze
        cur["Size"] = base["Size"] + (base["Depth"] - squeeze) / 10.0
    if cloth.get("FlattenNipples"):
        cur["NippleDepth"] = 0.0
        cur["AreolaDepth"] = 0.0
    v = 0.5 - base["VerticalAngle"]
    cur["VerticalAngle"] = 0.5 - (v - v * _num(cloth, "CenterNipples"))
    return cur


def pushed(base: dict, bra: dict | None, top: dict | None) -> dict | None:
    """``CalculatePush(CurrentlyWearing)``: *bra* / *top* are the worn,
    enabled garments' data (``None`` when that one does not count).
    ``None`` when neither does — the body stays as it is."""
    if bra is None and top is None:
        return None
    if top is None:
        return push_from_clothes(base, bra, bool(bra.get("UseAdvanced")))
    if bra is None:
        return push_from_clothes(base, top, bool(top.get("UseAdvanced")))
    # both on: an advanced garment wins outright, the top first
    if top.get("UseAdvanced"):
        return push_from_clothes(base, top, True)
    if bra.get("UseAdvanced"):
        return push_from_clothes(base, bra, True)
    combo = {k: max(_num(bra, k), _num(top, k)) for k in PUSH_FIELDS}
    combo["FlattenNipples"] = bool(bra.get("FlattenNipples")) or bool(top.get("FlattenNipples"))
    return push_from_clothes(base, combo, False)


def body_with(body: dict, values: dict) -> dict:
    """``MapBodyInfoToChaFile`` onto a copy of *body*."""
    out = dict(body)
    sv = list(out.get("shapeValueBody") or [])
    for k, i in SHAPE_INDEX.items():
        if i < len(sv):
            sv[i] = float(values[k])
    out["shapeValueBody"] = sv
    out["bustSoftness"] = float(values["Softness"])
    out["bustWeight"] = float(values["Weight"])
    return out


def worn_data(table: dict, coordinate: int, slot_worn: bool, state: int):
    """The garment's ``ClothData`` when it counts toward the push, else ``None``."""
    d = table.get(int(coordinate))
    if not slot_worn or int(state) != 0 or not d or not d.get("EnablePushup"):
        return None
    return d
