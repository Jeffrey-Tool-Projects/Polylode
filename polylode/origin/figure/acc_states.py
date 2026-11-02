"""Which accessory slots are drawn: ``showAccessory`` and madevil's Accessory States.

**The card's own switch.**  ``ChaControl.UpdateVisible``, transcribed::

    for i in range(len(objAccessory)):
        SetActiveControl(objAccessory[i],
                         visibleAll and fileStatus.showAccessory[i]
                         and not visibleSimple and not <head/body hidden>)

``showAccessory`` is loaded with the card and nothing outside H resets it: the
only callers of ``SetAccessoryStateAll`` are ``HScene``/``HSceneProc.Start``,
``HSceneProc.ResetScene`` (``SetAccessoryStateAll(true)`` on the heroine) and
the H sprite's button.  So life and the maker draw the card's switches, and an
H scene starts with every slot on.  Slots past the base 20 are MoreAccessories',
which keeps its own list in ``<visibility>`` of ``additionalAccessories``
(``OnActualCharaSave``); no local card writes one (0 / 1,268), and a missing
entry is "shown".

**madevil's Accessory States** (``KKEx["madevil.kk.ass"]``, read by the local
``BepInEx/plugins/KK_Accessory_States.dll``) binds a slot to one garment's
``clothesState``.  ``CharaEvent.ChangedOutfit(kind, state)`` walks the slots
bound to ``kind`` and calls ``ChaControl.SetAccessoryState(slot, show)`` —
i.e. it rewrites ``showAccessory`` for those slots.  It runs for every kind on
load (``OnReload`` → ``ChangeOutfitCoroutine``), when H starts
(``GameEvent.OnStartH`` → ``Refresh``, after the game's all-on) and whenever
``SetClothesState`` changes a garment.  Two storage forms:

* **v ≤ 5** — ``CharaTriggerInfo = {coordinate: {Parts: {slot: {Kind, Group,
  State[4]}}}}``.  ``Migration.Migrate`` turns a ``Kind < 9`` entry into four
  ``TriggerProperty(RefKind=Kind, RefState=i, Visible=State[i])`` — so the slot
  shows iff ``State[clothesState[Kind]]``.
* **v 6** — ``TriggerPropertyList``: those rows as stored.

``Kind >= 9`` is a *virtual group*, a toggle the player flips in H.  The
migration builds ``TriggerGroup(coordinate, kind, label, state = old.State ? 0
: 1, startup = 0, secondary)`` and two rows, ``RefState 0 → State[0]`` and
``RefState 1 → State[3]``.  A group stored **on** is in state 0 whichever of
``state`` / ``startup`` the load honours, so its slots show ``State[0]`` until
the player flips it (no H toggle here yet); a group stored off is where the two
readings part, and is reported.  So is a slot bound to more than one garment
(the ``Priority`` rule decides it).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import outfit as O

GUID = "madevil.kk.ass"
#: ``clothesState`` indices; the card's garment slots in the same order
KINDS = len(O.CLOTHES_SLOT_NAMES)


#: :attr:`Trigger.kind` of a virtual group's slot — it follows no garment
GROUP = -1


@dataclass(frozen=True)
class Trigger:
    """A slot bound to garment *kind*: shown in the states where ``shown[s]``.

    ``kind == GROUP`` is a virtual group's slot, fixed at ``shown[0]``.
    """
    kind: int
    shown: tuple[bool, bool, bool, bool]

    def visible(self, state: int) -> bool:
        s = 0 if self.kind == GROUP else int(state)
        return bool(self.shown[s]) if 0 <= s < len(self.shown) else False


def card_shown(card) -> list[bool]:
    """``ChaFileStatus.showAccessory`` (the base 20 slots)."""
    st = (getattr(card, "sections", None) or {}).get("Status") or {}
    show = st.get("showAccessory")
    return [bool(x) for x in show] if isinstance(show, (list, tuple)) else []


def slot_shown(shown: list[bool], slot: int) -> bool:
    """The card's switch for *slot*; MoreAccessories' slots read as shown."""
    return bool(shown[slot]) if 0 <= slot < len(shown) else True


def _section(card):
    kx = (getattr(card, "sections", None) or {}).get("KKEx") or {}
    sec = kx.get(GUID)
    if isinstance(sec, (list, tuple)) and len(sec) >= 2:
        return sec[0], sec[1]
    return None, None


def _unpack(blob):
    if isinstance(blob, (bytes, bytearray)):
        from ...chara.cardimport import _unpack as unpack
        return unpack(bytes(blob))
    return blob


def _from_v5(data: dict, coordinate: int, report: list) -> dict[int, Trigger]:
    info = _unpack(data.get("CharaTriggerInfo"))
    if not isinstance(info, dict):
        return {}
    # the coordinate key is a string in this form ("0"), an int in v6
    coord = info.get(coordinate, info.get(str(coordinate))) or {}
    parts = coord.get("Parts") if isinstance(coord, dict) else None
    groups = _unpack(data.get("CharaVirtualGroupInfo"))
    groups = (groups.get(coordinate, groups.get(str(coordinate))) or {})         if isinstance(groups, dict) else {}
    out: dict[int, Trigger] = {}
    for slot, t in (parts or {}).items():
        if not isinstance(t, dict):
            continue
        kind, state = t.get("Kind"), list(t.get("State") or [])
        if not isinstance(kind, int) or len(state) != 4:
            report.append(f"Accessory States：slot {slot} 的綁定形狀不對（{t!r:.60}）")
            continue
        if kind >= KINDS:
            g = groups.get(t.get("Group")) if isinstance(groups, dict) else None
            if not (isinstance(g, dict) and g.get("State") is True):
                report.append(f"Accessory States：slot {slot} 綁在虛擬群組"
                              f"「{t.get('Group')}」，群組存成關著（載入時的狀態"
                              f"是 state 還是 startup 沒抄），沒有套")
                continue
            out[int(slot)] = Trigger(GROUP, tuple(bool(x) for x in state))
            continue
        out[int(slot)] = Trigger(kind, tuple(bool(x) for x in state))
    return out


def _from_v6(data: dict, coordinate: int, report: list) -> dict[int, Trigger]:
    rows = _unpack(data.get("TriggerPropertyList")) or []
    by_slot: dict[int, list] = {}
    for r in rows:
        if isinstance(r, dict) and r.get("Coordinate") == coordinate:
            by_slot.setdefault(int(r.get("Slot", -1)), []).append(r)
    out: dict[int, Trigger] = {}
    for slot, rs in by_slot.items():
        kinds = {r.get("RefKind") for r in rs}
        if len(kinds) != 1:
            report.append(f"Accessory States：slot {slot} 綁了 {len(kinds)} 件衣服"
                          f"（依 Priority 決定），沒有套")
            continue
        kind = kinds.pop()
        if not isinstance(kind, int) or kind >= KINDS:
            report.append(f"Accessory States：slot {slot} 綁在虛擬群組 {kind}，沒有套")
            continue
        shown = [False] * 4
        for r in rs:
            s = r.get("RefState")
            if isinstance(s, int) and 0 <= s < 4:
                shown[s] = bool(r.get("Visible"))
        out[slot] = Trigger(kind, tuple(shown))
    return out


def triggers(card, coordinate: int) -> tuple[dict[int, Trigger], list[str]]:
    """``({slot: Trigger}, reasons)`` for *coordinate*; empty without the section."""
    _ver, data = _section(card)
    if not isinstance(data, dict):
        return {}, []
    report: list[str] = []
    if "TriggerPropertyList" in data:
        return _from_v6(data, int(coordinate), report), report
    return _from_v5(data, int(coordinate), report), report


def slot_visible(slot: int, shown: list[bool], trig: dict[int, Trigger],
                 states) -> bool:
    """Is accessory *slot* drawn for garment *states* (``clothesState``)?

    A bound slot follows its garment (``ChangedOutfit`` overwrote the card's
    switch); every other slot keeps *shown*.
    """
    t = trig.get(int(slot))
    if t is None:
        return slot_shown(shown, int(slot))
    st = list(states or [])
    return t.visible(st[t.kind] if 0 <= t.kind < len(st) else 0)


def apply(built, states=None, shown=None) -> int:
    """Set every tagged accessory part's ``visible``; returns how many are off.

    *states* defaults to ``built.clothes_state``; *shown* to the card's
    switches kept on *built* — pass ``[]`` (every slot on) for an H scene,
    which starts with ``SetAccessoryStateAll(true)``.
    """
    if built is None or getattr(built, "model", None) is None:
        return 0
    trig = getattr(built, "acc_triggers", None) or {}
    if shown is None:
        shown = getattr(built, "acc_shown", None) or []
    if states is None:
        states = getattr(built, "clothes_state", None) or []
    off = 0
    for part in built.model.parts:
        slot = getattr(part, "origin_acc_slot", None)
        if slot is None:
            continue
        part.visible = slot_visible(slot, shown, trig, states)
        off += not part.visible
    return off
