"""Staged objects, addressed the way Unity addresses them.

An exported file is named after its object, and two serialized files can each
hold an object of the same name — a prefab that carries its own copy of a mesh
names it like the model's.  The exporter's manifest says which serialized file
and path id every file came from, so everything here is keyed by
``(serialized file, path id)``: the two halves of a resolved ``PPtr``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .cabmap import Ref

__all__ = ["Ref", "Staged"]


class Staged:
    """Every object the exporter wrote, by reference and by export kind.

    The kind is the file suffix: ``.json`` (the decoded object), ``.txt`` (a
    type-tree dump) or ``.png`` (a converted texture).
    """

    def __init__(self) -> None:
        self._files: dict[tuple[Ref, str], Path] = {}
        self._types: dict[Ref, str] = {}
        self._names: dict[Ref, str] = {}

    def add(self, rows: Iterable[dict]) -> None:
        """Take manifest rows (:meth:`Exporter.run`).

        Raises ``ValueError`` when one file is claimed by two objects: the
        exporter then overwrote one with the other, and which survived is not
        something to find out from the pixels.
        """
        owners = {path: key for key, path in self._files.items()}
        for row in rows:
            ref = Ref(str(row["serializedFile"]), int(row["pathId"]))
            path = Path(row["output"])
            key = (ref, path.suffix.lower())
            if owners.get(path, key) != key:
                raise ValueError(f"{path} is the export of both {owners[path][0]} "
                                 f"and {ref}")
            owners[path] = key
            self._files[key] = path
            self._types[ref] = str(row["type"])
            self._names[ref] = str(row.get("name") or "")

    def has(self, ref: Ref, kind: str = ".json") -> bool:
        """Was *ref* exported as *kind*?"""
        return (ref, kind) in self._files

    def path(self, ref: Ref, kind: str = ".json") -> Path:
        """The staged file for *ref*; ``LookupError`` when it was not exported."""
        try:
            return self._files[(ref, kind)]
        except KeyError:
            raise LookupError(f"not staged: {ref.cab} path id {ref.path_id} "
                              f"({kind})") from None

    def json(self, ref: Ref) -> dict:
        """The decoded object *ref* points at."""
        return json.loads(self.path(ref).read_text(encoding="utf-8"))

    def text(self, ref: Ref) -> str:
        """The type-tree dump of *ref*."""
        return self.path(ref, ".txt").read_text(encoding="utf-8")

    def name(self, ref: Ref) -> str:
        """The object's own name as the exporter read it (``""`` if unknown)."""
        return self._names.get(ref, "")

    def of_type(self, type_name: str, cab: str | None = None) -> Iterator[Ref]:
        """Every staged object of *type_name*, optionally from one file only."""
        for ref, kind in sorted(self._types.items()):
            if kind == type_name and (cab is None or ref.cab == cab):
                yield ref
