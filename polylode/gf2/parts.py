"""Which of an outfit's swappable parts GF2 dresses a character in: the game's own parts tables.

Some outfits carry two or three versions of a slot in one prefab — Nikketa's SSR0102 has three tops, three skirts
and three pairs of legs (``P1_cloth1``, ``P2_cloth1``, ``P3_cloth1``…), every renderer active — and the game picks
one per slot at run time.  The pick is data, in the tables under ``LocalCache/Data/Table`` (protobuf: a 4-byte
length, an index of ``(row id, (offset, length))`` as field 3, then the rows):

* ``ModelConfigData`` names each model (field 1 its id, field 2 its name: ``105502`` → ``NikketaSSR0102``);
* ``PartsGroupData`` lists an outfit's slots (field 1 the slot id, field 2 its order, field 5 the outfit);
  the outfit is the model's id plus 1 000 000 (``1105502``), which holds for all 36 outfits the table lists;
* ``PartsListData`` lists a slot's choices (field 7 the slot, field 2 its index — the P number — and field 1 the
  choice's id); field 4 is 1 on the choice worn without unlocking, and every other choice carries field 5 (what
  unlocks it) instead — in all 111 slots the free one is index 1;
* ``PartsResourceData`` names a choice's nodes (row id = the choice's id; field 2 of each entry a node name,
  ``P1_hip2``, which the prefab's renderers carry as ``…_P1_hip2_trans_lod0``).

The code that applies them is encrypted (HybridCLR); these tables, and the string literals next to its
``ReplaceableModelRes`` (``PartsGroupList``, ``partsResource``, ``_P1_``…), are what is read here.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

#: the tables' folder in an installation
TABLES = Path("GF2_Exilium_Data") / "LocalCache" / "Data" / "Table"
#: an outfit's id in the parts tables is its model's id plus this (ModelConfigData ↔ PartsGroupData)
OUTFIT_OFFSET = 1_000_000


def _varint(data: bytes, at: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[at]
        at += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, at


def _fields(data: bytes) -> list[tuple[int, object]]:
    """A protobuf message's fields in order, a length-delimited one as its bytes."""
    out, at = [], 0
    while at < len(data):
        key, at = _varint(data, at)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, at = _varint(data, at)
        elif wire == 2:
            length, at = _varint(data, at)
            value, at = data[at:at + length], at + length
        elif wire == 1:
            value, at = data[at:at + 8], at + 8
        elif wire == 5:
            value, at = data[at:at + 4], at + 4
        else:
            raise ValueError(f"protobuf wire type {wire}")
        out.append((field, value))
    return out


def _packed(value) -> list[int]:
    if isinstance(value, int):
        return [value]
    out, at = [], 0
    while at < len(value):
        number, at = _varint(value, at)
        out.append(number)
    return out


def rows(path: Path) -> dict[int, list[dict[int, object]]]:
    """A table's rows: row id → its entries (field 1 of the row), each as {field: value}."""
    data = Path(path).read_bytes()
    (length,) = struct.unpack_from("<I", data, 0)
    body = data[4 + length:]
    out = {}
    for field, value in _fields(data[4:4 + length]):
        if field != 3:
            continue
        index = dict(_fields(value))
        span = dict(_fields(index[2]))
        start = span.get(1, 0)
        row = _fields(body[start:start + span[2]])
        out[index[1]] = [dict(_fields(entry)) for number, entry in row if number == 1]
    return out


def choices(install: Path, ident: str) -> dict | None:
    """*ident*'s swappable slots as the game dresses it: ``{'worn': [node names], 'spare': [node names]}`` —
    each slot's free choice, and the others — or None when the tables list no slots for it."""
    folder = Path(install) / TABLES
    names = [folder / f"{table}.bytes" for table in ("ModelConfigData", "PartsGroupData", "PartsListData",
                                                      "PartsResourceData")]
    if not all(path.is_file() for path in names):
        return None
    models = {}
    for entries in rows(names[0]).values():
        for entry in entries:
            if isinstance(entry.get(2), bytes):
                models[entry[2].decode("utf-8")] = entry[1]
    # the model whose prefab this is: c_<name>_…, the longest name that fits (NikketaSSR0102, not NikketaSSR01)
    fitting = [name for name in models if ident.startswith(f"c_{name}_")]
    if not fitting:
        return None
    outfit = models[max(fitting, key=len)] + OUTFIT_OFFSET
    slots = {entry[1] for entries in rows(names[1]).values() for entry in entries if entry.get(5) == outfit}
    if not slots:
        return None
    resources = rows(names[3])
    # an option's resources are the ids in its field 3 (one, or a packed list: Helen SSR0102's top lists 15), each a
    # PartsResourceData entry (field 1 the id, field 2 the node); the table's row of the option's id holds the same
    # entries for most options, not for those
    by_id = {part[1]: part for entries in resources.values() for part in entries if isinstance(part.get(1), int)}
    worn, spare = [], []
    for entries in rows(names[2]).values():
        for entry in entries:
            if entry.get(7) not in slots:
                continue
            listed = [by_id[i] for i in _packed(entry[3]) if i in by_id] if 3 in entry else []
            listed += resources.get(entry[1], [])
            nodes = [part[2].decode("utf-8") for part in listed if isinstance(part.get(2), bytes)]
            (worn if entry.get(4) == 1 else spare).extend(nodes)
    return {"outfit": outfit, "worn": sorted(set(worn)), "spare": sorted(set(spare) - set(worn))}


def of(name: str, nodes: list[str]) -> bool:
    """Whether the node or part *name* belongs to one of the choices *nodes* (``…_P2_cloth1_lod0`` to ``P2_cloth1``).
    The level is not part of a choice's name: Helen SSR0102's ``P1_cloth1_Bend`` is the node ``…_P1_cloth1_lod0_Bend``."""
    bare = re.sub(r"_lodm?\d+", "", name, flags=re.IGNORECASE)
    return any(f"_{node}_" in f"_{name}_" or f"_{node}_" in f"_{bare}_" for node in nodes)


def hidden(parts: list[dict], picked: dict) -> dict[str, bool]:
    """Each part hidden when it is a node of a choice not worn (``…_P2_cloth1_lod0`` for ``P2_cloth1``) and of
    no worn one."""
    return {part["name"]: of(part["name"], picked["spare"]) and not of(part["name"], picked["worn"])
            for part in parts}
