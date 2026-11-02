"""Which converted items belong to which character.

The game keys a character's resources by a character id and writes that id
into the resource names: ``<id>_postmodel``, ``<id>_uimodel``, ``<id>_deco_<n>``.
The id itself is data in exactly one place the converter reads — the
``_charId`` field of ``CharUIModelMono`` on a character's UI-model prefab — so
the set of ids comes from there (:meth:`Pipeline.char_ids`) and an item is
filed under the id its name starts with.  Nothing is inferred from a name that
no component declared.

Not covered, because nothing in the prefabs says it: which weapon a character
carries and which node it hangs from, and which ``att_widget_*`` attachment is
whose (``PROGRESS.md``).
"""
from __future__ import annotations

import json
from typing import Iterable

from ..core.paths import write_json_atomic
from . import store

FILE = "links.json"
#: The categories whose items carry a character id in their name.
CATEGORIES = ("chara", "uimodel", "deco")


def build(ids: Iterable[str]) -> dict:
    """``{char id: {category: [idents]}}`` over what the store holds now.

    An ident belongs to the **longest** declared id it starts with (followed by
    ``_``), so an id that is a prefix of another does not claim its items.
    """
    declared = sorted(set(ids), key=len, reverse=True)
    out: dict = {}
    for category in CATEGORIES:
        for ident in store.idents(category):
            name = ident.rsplit("/", 1)[-1]
            owner = next((i for i in declared if name.startswith(f"{i}_")), None)
            if owner is not None:
                out.setdefault(owner, {}).setdefault(category, []).append(ident)
    return dict(sorted(out.items()))


def unlinked(links: dict) -> dict:
    """``{category: [idents]}`` of the stored items no declared id claimed."""
    claimed = {(category, ident) for groups in links.values()
               for category, idents in groups.items() for ident in idents}
    out: dict = {}
    for category in CATEGORIES:
        loose = [i for i in store.idents(category) if (category, i) not in claimed]
        if loose:
            out[category] = loose
    return out


def write(links: dict) -> None:
    """Store *links* beside the items."""
    write_json_atomic(store.root() / FILE, links, indent=1)


def read() -> dict:
    """The stored links (``{}`` when none were built)."""
    path = store.root() / FILE
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
