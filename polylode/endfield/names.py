"""The name a converted Endfield character's ``Charas`` folder is given, read from the game's own tables.

A character's folder (``Charas/Endfield/<character>/<model>/``, :mod:`.charas`)
is named the way the game names that character in traditional Chinese: the
``CharacterTable`` row of its character id gives a text id (``name.id``) and
``I18nTextTable_TC`` gives the text.  Both are tables dumped from this
machine's install (``docs/tools.md``: the name tables), so a folder name is
data this machine holds, never a translation made up for it.

The character id is the one the converter read off the model's own UI prefab
(``links.json``, :mod:`.links`); when no link names the model, the longest
``CharacterTable`` id the model's name starts with.  When the tables are not
there, the row has no text, or the text cannot be a folder name, the folder is
the character id itself and the identity says it is **not verified** —
nothing is guessed in its place.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ..core.paths import derived_root

#: Points the name lookup at another folder holding the two tables (tests, another extraction).
TABLES_ENV = "ISLANDLIFE_ENDFIELD_NAME_TABLES"
CHARACTER_TABLE = "CharacterTable.json"
LANGUAGE = "TC"
TEXT_TABLE = f"I18nTextTable_{LANGUAGE}.json"
#: Characters a Windows folder name cannot hold.
_UNSAFE = set('<>:"/\\|?*')


@dataclass(frozen=True)
class Identity:
    """Who a converted model is: its character id, its folder name, and whether the name was read."""
    ident: str
    char_id: str | None
    folder: str
    verified: bool
    evidence: dict = field(default_factory=dict)

    def describe(self) -> dict:
        """The identity as the folder's descriptor keeps it."""
        return {"char_id": self.char_id, "name": self.folder, "name_verified": self.verified,
                "name_evidence": self.evidence}


def tables_dir() -> Path:
    """The folder holding ``CharacterTable.json`` and the traditional Chinese text table."""
    env = os.environ.get(TABLES_ENV)
    return Path(env) if env else derived_root() / "Endfield_Converted" / "_extraction" / "character_names" / "Table"


def owner(ident: str, declared: Iterable[str]) -> str | None:
    """The longest declared character id *ident* is, or starts with followed by ``_``."""
    found = [cid for cid in declared if ident == cid or ident.startswith(f"{cid}_")]
    return max(found, key=len) if found else None


@lru_cache(maxsize=4)
def _load(path: str, stamp: tuple) -> tuple[dict, str]:
    """A JSON table and its SHA-256, kept while the file's size and mtime stay the same."""
    data = Path(path).read_bytes()
    return json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest()


def _table(path: Path) -> tuple[dict, str]:
    info = path.stat()
    return _load(str(path), (info.st_size, info.st_mtime_ns))


def _folder_safe(text: str) -> bool:
    return (bool(text) and text == text.strip() and not text.endswith(".")
            and not _UNSAFE.intersection(text) and all(ord(c) >= 32 for c in text))


def _read_name(char_id: str, tables: Path) -> tuple[str | None, dict]:
    """``(name, evidence)`` of *char_id*, or ``(None, {"reason": …})`` when it cannot be read."""
    rows_path, text_path = tables / CHARACTER_TABLE, tables / TEXT_TABLE
    try:
        rows, rows_sha = _table(rows_path)
        texts, texts_sha = _table(text_path)
    except (OSError, ValueError) as exc:
        return None, {"reason": f"name tables unreadable: {exc}"}
    row = rows.get(char_id)
    if not isinstance(row, dict):
        return None, {"reason": f"{CHARACTER_TABLE} has no row {char_id}"}
    text_id = str((row.get("name") or {}).get("id", ""))
    text = texts.get(text_id) if text_id else None
    evidence = {"character_table": str(rows_path), "character_table_sha256": rows_sha,
                "text_table": str(text_path), "text_table_sha256": texts_sha,
                "field": f"{char_id}.name.id", "text_id": text_id, "language": LANGUAGE,
                "eng_name": row.get("engName")}
    if not isinstance(text, str) or not _folder_safe(text):
        return None, {**evidence, "reason": f"text {text_id!r} is missing or cannot name a folder"}
    return text, evidence


def _declared_ids(tables: Path) -> list[str]:
    try:
        return list(_table(tables / CHARACTER_TABLE)[0])
    except (OSError, ValueError):
        return []


def identify(ident: str, linked: Iterable[str] = (), tables: Path | None = None) -> Identity:
    """Name *ident*'s character folder from the local tables; unverified falls back to the character id.

    *linked*: the character ids the converter recorded (``links.json``); they win over a
    name-prefix match against the table.  With no character id at all the folder is *ident*.
    """
    tables = tables_dir() if tables is None else tables
    char_id = owner(ident, linked) or owner(ident, _declared_ids(tables))
    if char_id is None:
        return Identity(ident, None, ident, False, {"reason": "no character id claims this model"})
    name, evidence = _read_name(char_id, tables)
    if name is None:
        return Identity(ident, char_id, char_id, False, evidence)
    return Identity(ident, char_id, name, True, evidence)
