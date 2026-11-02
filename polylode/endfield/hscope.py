"""Local age declarations for H roles, separate from anatomical body construction."""
from __future__ import annotations

import re

from . import store
from .body_scope import ADULT_HUMAN_TYPES, FEMALE_HUMAN_TYPES, declared_scope
from .model import folder_of

FEMALE_FAMILIES = FEMALE_HUMAN_TYPES


def declared_age(extra: dict) -> int | None:
    """Read a model's age from the owner's local ``chara_age/age.md`` table.

    Entries use ``model_name: integer_age``. Missing declarations return None;
    malformed or duplicate entries fail instead of changing a role's scope.
    """
    path = store.root() / "table" / store.SOURCE / "chara_age" / "age.md"
    if not path.is_file():
        return None
    ages: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*([a-z0-9_]+)\s*:\s*([0-9]+)\s*", line)
        if match is None or match[1] in ages:
            raise ValueError(f"{path}: invalid or duplicate age declaration")
        ages[match[1]] = int(match[2])
    folder = folder_of(str(extra.get("model") or ""))
    return ages.get(folder.partition("/")[2])


def family(extra: dict, character: str) -> str:
    """Authorize an H role by rig support and explicit age when supplied.

    The adult reference families retain their existing scope. A best_girl
    female rig needs a declared age of at least 18; this never authorizes
    changes the original model declaration. Body construction separately
    validates its source card and the same adult age declaration.
    """
    declared = declared_scope(extra, character)
    if declared is not None:
        return declared
    kind = folder_of(str(extra.get("model") or "")).partition("/")[0]
    age = declared_age(extra)
    if age is not None and age < 18:
        raise ValueError(f"{character}: H role requires an adult age")
    if kind in ADULT_HUMAN_TYPES or (kind == "best_girl" and age is not None):
        return kind
    raise ValueError(f"{character}: H role has no supported adult declaration")
