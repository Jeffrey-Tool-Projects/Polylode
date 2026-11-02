"""Parse Coordinate blocks from a Origin card into clothes + accessories."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import moreacc as MA
from ...core import swallowed


# ChaFileClothes.parts[i] layout (Koikatsu)
CLOTHES_SLOT_NAMES = (
    "top", "bottom", "bra", "panty", "gloves",
    "legwear", "socks", "shoes_inner", "shoes_outer",
)

# list-table category for each slot
LIST_CATEGORY = {
    "top": "top", "bottom": "bottom", "bra": "bra", "panty": "panty",
    "gloves": "gloves", "legwear": "legwear", "socks": "socks",
    "shoes_inner": "shoes", "shoes_outer": "shoes",
}

# UAR property names, transcribed row by row from
# ``StructReference.ChaFileClothesGenerator`` in ``Sideloader.dll`` — the table
# that *writes* them — rather than derived from the field names on the card.
# Those two disagree in one place and it is not guessable: ``ChaFileClothes``
# calls slot 5 ``parts[5]`` / ``panst`` and Illusion's category is ``co_panst``,
# but the UAR property is **``ClothesPants``**.  ``ClothesPanst`` here matched
# nothing, so a modded pair of tights never went through the card's own
# resolution table at all — 215 of 300 sampled cards carry a ``ClothesPants``
# entry and none carries a ``ClothesPanst`` one (§7.72 九).
UAR_FIELDS = {
    "top": "ClothesTop", "bottom": "ClothesBot", "bra": "ClothesBra",
    "panty": "ClothesShorts", "gloves": "ClothesGloves",
    "legwear": "ClothesPants", "socks": "ClothesSocks",
    "shoes_inner": "ClothesShoesInner", "shoes_outer": "ClothesShoesOuter",
}

# Same table, the ``mt_pattern`` (CategoryNo 430) and ``mt_emblem`` (431) rows.
# ``{n}`` is the colour slot the tile belongs to (0..3); an emblem has two.
UAR_PATTERN_FIELDS = {k: f"{v}Pattern{{n}}" for k, v in UAR_FIELDS.items()}
UAR_EMBLEM_FIELDS = {k: (f"{v}Emblem", f"{v}Emblem2") for k, v in UAR_FIELDS.items()}


@dataclass
class ClothSlot:
    name: str                 # top / bottom / …
    index: int
    item_id: int
    colors: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


@dataclass
class AccSlot:
    index: int
    type_no: int              # Illusion accessory category (121+)
    item_id: int
    parent_key: str = ""
    colors: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    coordinate: int = 0


@dataclass
class Outfit:
    coordinate: int = 0
    sub_parts: list[int] = field(default_factory=list)
    clothes: list[ClothSlot] = field(default_factory=list)
    accessories: list[AccSlot] = field(default_factory=list)
    # ``ChaFileClothes.hideBraOpt`` / ``hideShortsOpt`` — the bra and panty
    # slots' option switches, which ``DrawOption`` reads instead of the slot's
    # own ``hideOpt`` (§7.379).
    hide_bra_opt: tuple = (False, False)
    hide_shorts_opt: tuple = (False, False)


def _unpack_coordinate_entry(blob: bytes) -> list[dict]:
    """A Coordinate[i] is repeated [int32 len][msgpack] blocks (clothes, acc, …)."""
    from ...chara.cardimport import _Reader, _unpack
    data = bytes(blob)
    r = _Reader(data, 0)
    out: list[dict] = []
    while r.i + 4 <= len(data):
        try:
            n = r.i32()
        except Exception as exc:  # noqa: BLE001
            swallowed.note("origin.figure.outfit._unpack_coordinate_entry: n = r.i32()", exc)
            break
        if n <= 0 or r.i + n > len(data):
            break
        try:
            obj = _unpack(r.bytes(n))
        except Exception as exc:  # noqa: BLE001
            swallowed.note("origin.figure.outfit._unpack_coordinate_entry: obj = _unpack(r.bytes(n))", exc)
            break
        if isinstance(obj, dict):
            out.append(obj)
    return out


def clothes_state(card) -> list[int]:
    """``ChaFileStatus.clothesState`` — one byte per clothing slot, 0..3.

    0 dressed, 1 and 2 half off, 3 off.  What each value does to each slot is
    :func:`slot_active` (the objects) and :func:`state_node_visible` (the
    state nodes inside one).  On this install 75 of 1,388 cards store something
    other than 0, and almost all of it is 3 (§7.369).
    """
    status = card.sections.get("Status") if hasattr(card, "sections") else None
    raw = (status or {}).get("clothesState") if isinstance(status, dict) else None
    if isinstance(raw, (bytes, bytearray)):
        return [int(b) for b in raw]
    if isinstance(raw, list):
        out = []
        for v in raw:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                out.append(0)
        return out
    return []


#: ``ChaFileClothes.parts`` index of each shoe slot, and the ``shoesType`` value
#: that makes it the worn one.  ``ChaControl.UpdateVisible`` gates them
#: separately: ``objClothes[7]`` needs ``shoesType == 0`` and ``objClothes[8]``
#: needs ``shoesType == 1``, so exactly one pair is ever on the character.
SHOES_SLOT_FOR_TYPE = {0: "shoes_inner", 1: "shoes_outer"}


@dataclass
class NotFlags:
    """``ChaControl.notBot`` / ``notBra`` / ``notShorts`` — one garment hiding another.

    Set while the outfit loads, from the list rows (``origin_il_dump``):

    * ``notBra``: ``ChangeClothesTopAsync`` → top row ``NotBra`` (KeyType 46)
      ``== 1``; for a combo top ``ChangeClothesTopPartsAsync`` decides instead —
      sailor (type 1) from part 0's ``NotBra``, jacket (type 2) from part 0
      **or** part 1's.
    * ``notBot``: top row ``Coordinate`` (KeyType 8) ``== 2`` — a one-piece.
    * ``notShorts``: bra row ``Coordinate == "2"`` — a body suit.
    """
    bot: bool = False
    bra: bool = False
    shorts: bool = False


#: ``KK_ClothingUnlocker``'s card data (``KKEx``), MessagePack
#: ``{coordinate: bool}`` under ``ClothingUnlocked``
CLOTHING_UNLOCKER = "com.deathweasel.bepinex.clothingunlocker"


def clothing_unlocked(card, coordinate: int) -> bool:
    """Has ``KK_ClothingUnlocker`` unlocked *coordinate*'s clothes?

    Its ``Hooks.GetInfoPostfix`` / ``GetInfoIntPostfix`` (on ``ListInfoBase``)
    answer ``"0"`` for ``NotBra`` (46), ``Coordinate`` (8) and ``HideShorts``
    (106) while a character whose controller says so is changing clothes
    (``CheckOverride``) — and those are the reads :class:`NotFlags` is made
    of (``GetInfoInt(46)`` in ``ChangeClothesTop*Async``, ``GetInfo(8)`` for
    the top and the bra), so an unlocked coordinate has all three false.
    (``HideShorts`` itself is read straight off ``dictInfo`` into
    ``hideShortsBot``, which nothing in ``Assembly-CSharp`` reads.)  An
    unreadable section counts as locked and is not guessed at (§7.447).
    """
    from ...chara.cardimport import _unpack

    try:
        sec = ((card.sections.get("KKEx") or {}) if card is not None else {}
               ).get(CLOTHING_UNLOCKER)
        blob = sec[1].get("ClothingUnlocked") if sec is not None else None
        table = _unpack(bytes(blob)) if blob else {}
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    # the project's reader keys a map by the key's string form
    table = {str(k): v for k, v in (table or {}).items()}
    return bool(table.get(str(int(coordinate)), False))


def slot_active(index: int, states: list[int], shoes: int,
                flags: "NotFlags | None" = None) -> bool:
    """Does ``ChaControl.UpdateVisible`` activate ``objClothes[index]``?

    Transcribed for a character outside H and not in simple mode (the two
    locals the method opens with are then ``visibleBody = true`` and
    ``visibleSimple = false``), with ``visibleAll`` / ``visibleBodyAlways``
    true — which is every character this build puts on screen:

    ====  ==========  ===============================================
    slot  ``index``   active when
    ====  ==========  ===============================================
    top   0           state != 3
    bot   1           not ``notBot`` and state != 3
    bra   2           not ``notBra`` and state != 3
    shorts 3          not ``notShorts`` and state != 3
    gloves 4          state == 0
    panst 5           state not in (2, 3)
    socks 6           state == 0
    shoes 7           ``shoesType`` == 0 and state == 0
    shoes 8           ``shoesType`` == 1 and state == 0
    ====  ==========  ===============================================

    States are per slot (``clothesState[index]``).  Used to be approximated
    by "draw the last state variant", which put a removed garment back on
    68 cards whose top is stored as 3.
    """
    f = flags or NotFlags()
    st = states[index] if index < len(states) else 0
    if index == 0:
        return st != 3
    if index == 1:
        return not f.bot and st != 3
    if index == 2:
        return not f.bra and st != 3
    if index == 3:
        return not f.shorts and st != 3
    if index in (4, 6):
        return st == 0
    if index == 5:
        return st not in (2, 3)
    if index == 7:
        return int(shoes) == 0 and st == 0
    if index == 8:
        return int(shoes) == 1 and st == 0
    return True


def _st(states: list[int], i: int) -> int:
    return int(states[i]) if i < len(states) else 0


def state_node_visible(index: int, node: str, states: list[int]) -> bool:
    """Is the garment's state node *node* active in ``objClothes[index]``?

    ``ChaControl.UpdateVisible`` switches the nodes ``CreateReferenceInfo``
    found inside each clothing prefab (``prefab.CLOTHES_STATE_NODES``; a
    renderer carries its nearest one as ``_stateNode``, §7.379).  ``sN`` is
    ``clothesState[N]``:

    ======  ===========  ==============================================
    slot    node         active when
    ======  ===========  ==============================================
    top 0   n_top_a      s0 == 0  (also each combo part's ``n_top_a``)
            n_top_b      s0 == 1  (also each combo part's ``n_top_b``)
            n_bot_a      s1 == 0 and s0 != 3   (the hem follows the bottom)
            n_bot_b      s1 != 0 and s0 != 3
    bot 1   n_top_a      s0 == 0 and s1 != 2   (the waist follows the top)
            n_top_b      s0 != 0 and s1 != 2
            n_bot_a      s1 == 0
            n_bot_b      s1 == 1
    bra 2   n_top_a      s2 == 0
            n_top_b      s2 == 1
            n_bot_a      s3 == 0 and s2 != 3
            n_bot_b      s3 != 0 and s2 != 3
    shorts  n_bot_a/b/c  s3 == 0 / 1 / 2
    panst   n_panst_a/b  s5 == 0 / 1
    ======  ===========  ==============================================

    A node the method never touches for that slot (the panty's ``n_top_*``,
    anything in gloves / socks / shoes) stays as the prefab has it — active.
    """
    s0, s1, s2, s3, s5 = (_st(states, i) for i in (0, 1, 2, 3, 5))
    rules = {
        0: {"n_top_a": s0 == 0, "n_top_b": s0 == 1,
            "n_bot_a": s1 == 0 and s0 != 3, "n_bot_b": s1 != 0 and s0 != 3},
        1: {"n_top_a": s0 == 0 and s1 != 2, "n_top_b": s0 != 0 and s1 != 2,
            "n_bot_a": s1 == 0, "n_bot_b": s1 == 1},
        2: {"n_top_a": s2 == 0, "n_top_b": s2 == 1,
            "n_bot_a": s3 == 0 and s2 != 3, "n_bot_b": s3 != 0 and s2 != 3},
        3: {"n_bot_a": s3 == 0, "n_bot_b": s3 == 1, "n_bot_c": s3 == 2},
        5: {"n_panst_a": s5 == 0, "n_panst_b": s5 == 1},
    }
    return rules.get(int(index), {}).get(str(node), True)


def shoes_type(card) -> int:
    """``ChaFileStatus.shoesType`` — 0 indoor (slot 7), 1 outdoor (slot 8).

    A card always stores *both* shoe slots, so this is not an optional detail:
    on this install 865 of 1,386 cards (62.4%) have two different ids in them,
    and drawing both puts two pairs of shoes on one pair of feet.  1,234 cards
    ask for outdoor, 152 for indoor.
    """
    status = card.sections.get("Status") if hasattr(card, "sections") else None
    if not isinstance(status, dict):
        return 1
    try:
        return int(status.get("shoesType", 1))
    except (TypeError, ValueError):
        return 1


def _two_flags(raw) -> tuple:
    """A serialised ``bool[2]`` (``hideOpt`` and friends) as two bools."""
    vals = list(raw) if isinstance(raw, (list, tuple)) else []
    vals = (vals + [False, False])[:2]
    return tuple(bool(v) for v in vals)


def sleeves_type(slot: ClothSlot) -> int:
    """The slot's ``sleevesType`` (``MemberInit`` 0), which ``ChaControl.DrawSode``
    reads off the top: objSleeves group ``sleevesType + 1`` is the one drawn."""
    try:
        return int((slot.raw or {}).get("sleevesType", 0) or 0)
    except (TypeError, ValueError):
        return 0


def sleeve_pieces(pieces, slot: ClothSlot) -> list:
    """The pieces of the top *slot* that ``ChaControl.DrawSode`` leaves on.

    ``DrawSode`` runs ``SetActiveControl(objSleevesN[i], sleevesType == N - 1)``
    over the three groups of ``cusClothesCmp[0]``, so one sleeve length is
    drawn.  A piece outside every group (``_clothSleeve`` 0 or absent — the
    field is §7.509's) is not the component's to switch.
    """
    want = sleeves_type(slot) + 1
    return [p for p in pieces or ()
            if int(((getattr(p, "shader", None) or {}).get("_clothSleeve", 0)) or 0)
            in (0, want)]


def hidden_options(outfit: Outfit, slot: ClothSlot) -> tuple:
    """``(hide objOpt01, hide objOpt02)`` for *slot*, as ``ChaControl.DrawOption``.

    The IL reads ``parts[kind].hideOpt`` first and then, for kind 2 (bra) and
    kind 3 (panty), overwrites both with ``hideBraOpt`` / ``hideShortsOpt``
    (§7.379).  ``True`` is "that group is switched off".
    """
    if slot.index == 2:
        return tuple(outfit.hide_bra_opt)
    if slot.index == 3:
        return tuple(outfit.hide_shorts_opt)
    return _two_flags((slot.raw or {}).get("hideOpt"))


def parse_outfit(card, coordinate: int = 0) -> Outfit:
    """Build a structured Outfit from the card's Coordinate section."""
    out = Outfit(coordinate=int(coordinate))
    coord = card.sections.get("Coordinate")
    blocks: list[dict] = []
    if isinstance(coord, list):
        if 0 <= coordinate < len(coord):
            entry = coord[coordinate]
            if isinstance(entry, (bytes, bytearray)):
                blocks = _unpack_coordinate_entry(bytes(entry))
            elif isinstance(entry, dict):
                blocks = [entry]
    elif isinstance(coord, dict):
        blocks = [coord]

    clothes_block = None
    acc_block = None
    for b in blocks:
        parts = b.get("parts")
        if not isinstance(parts, list) or not parts:
            continue
        first = parts[0] if isinstance(parts[0], dict) else {}
        if "parentKey" in first or "addMove" in first or "type" in first:
            acc_block = b
        elif "id" in first:
            clothes_block = b

    if clothes_block is not None:
        out.hide_bra_opt = _two_flags(clothes_block.get("hideBraOpt"))
        out.hide_shorts_opt = _two_flags(clothes_block.get("hideShortsOpt"))
        subs = clothes_block.get("subPartsId") or []
        out.sub_parts = [int(s) for s in subs if isinstance(s, (int, float))]
        for i, name in enumerate(CLOTHES_SLOT_NAMES):
            parts = clothes_block.get("parts") or []
            if i >= len(parts) or not isinstance(parts[i], dict):
                continue
            raw = parts[i]
            cid = int(raw.get("id") or 0)
            # id 0 is a row, not "nothing".  Every clothing category's id 0 does
            # happen to be the "なし" row (``p_dummy``), which
            # ``dress.is_empty_slot`` already drops — but that is a fact about
            # the shipped lists, not a rule, and writing it as one here is what
            # cost the back hair and the id-0 accessories below.  Let the row
            # decide; only a negative id is not a row.
            if cid < 0:
                continue
            colors = raw.get("colorInfo") or raw.get("color") or []
            out.clothes.append(ClothSlot(
                name=name, index=i, item_id=cid,
                colors=list(colors) if isinstance(colors, list) else [],
                raw=raw,
            ))

    # The card's own array, as MoreAccessories leaves it on load: slot ``i``
    # is its *i*-th entry whether the plugin kept the extras in the XML
    # (version 1) or in the array itself (version 2), so the same loop
    # resolves all of them (``moreacc.accessory_slots``, §7.364 / §7.487).
    slots = MA.accessory_slots(
        card, coordinate,
        list(acc_block.get("parts") or []) if acc_block is not None else [])
    if slots:
        for i, raw in enumerate(slots):
            if not isinstance(raw, dict):
                continue
            tid = int(raw.get("type") or 0)
            iid = int(raw.get("id") or 0)
            # ``<ChangeAccessoryAsync>`` decides this in two tests and neither
            # is "id <= 0"::
            #
            #     if !InRange(121, type, 130): load = false   // ao_none is 120
            #     if id == -1:                 load = false
            #
            # so **id 0 is a real accessory** — the first row of every ao_
            # category (``p_acs_megane01`` glasses, ``p_acs_head_ribbon``,
            # ``p_acs_back01``…).  Dropping it took an item off 13 of 200 female
            # and 3 of 40 male cards in coordinate 0 alone, most of them the
            # glasses on ``a_n_megane``.
            if iid < 0 or not (121 <= tid <= 130):
                continue
            colors = raw.get("colorInfo") or raw.get("color") or raw.get("acsColor") or []
            out.accessories.append(AccSlot(
                index=i, type_no=tid, item_id=iid,
                parent_key=str(raw.get("parentKey") or ""),
                colors=list(colors) if isinstance(colors, list) else [],
                raw=raw, coordinate=coordinate,
            ))
    return out


def parse_all_accessories(card) -> list[AccSlot]:
    """Accessories from every coordinate tab (deduped by type+id+parent).

    For **baking**, where the point is to have every asset a card could ask for
    on disk whichever coordinate it is later worn in.  Dressing a character uses
    one coordinate (``parse_outfit(card, outfit).accessories``) the way
    ``ChangeAccessory`` walks ``nowCoordinate`` — see ``dress._stage_accessories``.
    """
    seen: set[tuple] = set()
    out: list[AccSlot] = []
    coord = card.sections.get("Coordinate")
    n = len(coord) if isinstance(coord, list) else 1
    for oi in range(n):
        outfit = parse_outfit(card, oi)
        for acc in outfit.accessories:
            key = (acc.type_no, acc.item_id, acc.parent_key)
            if key in seen:
                continue
            seen.add(key)
            out.append(acc)
    return out


def hair_parts(card) -> list[dict]:
    """ChaFileHair.parts — up to 4 slots (back/front/side/option)."""
    hair = card.hair() if hasattr(card, "hair") else {}
    if not isinstance(hair, dict):
        return []
    parts = hair.get("parts") or []
    return [p for p in parts if isinstance(p, dict)]
