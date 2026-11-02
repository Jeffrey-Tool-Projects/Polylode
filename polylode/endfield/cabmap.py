"""AnimeStudio's CAB map: where a serialized file lives and what it depends on.

A Unity ``PPtr`` is ``(m_FileID, m_PathID)``.  File id 0 is the file the pointer
sits in; file id *k* is the *k*-th entry of that file's externals.  The CAB map
records exactly that list per serialized file ("CAB"), in order, so
``external(cab, k)`` is the file a pointer means — measured on a character
prefab: 125 of 125 external mesh and material references name
``deps[file_id - 1]`` (§7.562).

The file is a .NET ``BinaryWriter`` stream (the writer is AnimeStudio's
``AssetsHelper.DumpCABMap``): ``string BaseFolder``, ``int32 count``, then per
entry a CAB name, a path relative to the base folder, an ``int64`` byte offset
into that chunk file, an ``int32`` dependency count and that many CAB names.
Strings carry .NET's 7-bit encoded length prefix.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple


class Ref(NamedTuple):
    """A resolved ``PPtr``: the serialized file and the object's path id in it."""
    cab: str
    path_id: int


@dataclass(frozen=True)
class CabEntry:
    """One serialized file: the bundle slot it sits in and its externals."""
    name: str
    root: Path                 # the map's base folder (one VFS root)
    source: Path               # absolute path of the chunk file
    offset: int                # byte offset of the bundle inside it
    deps: tuple[str, ...]      # externals, in file-id order (file id 1 first)


class _Reader:
    """Cursor over a ``BinaryWriter`` stream."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def string(self) -> str:
        length = shift = 0
        while True:
            byte = self.data[self.pos]
            self.pos += 1
            length |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        text = self.data[self.pos:self.pos + length].decode("utf-8")
        self.pos += length
        return text

    def unpack(self, fmt: str) -> int:
        (value,) = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += struct.calcsize(fmt)
        return value


def read_cabmap(path: Path) -> dict[str, CabEntry]:
    """Every entry of one CAB map file, by CAB name.

    Raises ``ValueError`` when the stream does not end exactly where the last
    entry does — a file read with the wrong framing must not look like a map.
    """
    reader = _Reader(Path(path).read_bytes())
    root = Path(reader.string())
    count = reader.unpack("<i")
    entries: dict[str, CabEntry] = {}
    for _ in range(count):
        name = reader.string()
        rel = reader.string()
        offset = reader.unpack("<q")
        deps = tuple(reader.string() for _ in range(reader.unpack("<i")))
        entries[name] = CabEntry(name, root, root / rel.replace("\\", "/"),
                                 offset, deps)
    if reader.pos != len(reader.data):
        raise ValueError(f"{path}: {len(reader.data) - reader.pos} bytes left "
                         f"after {count} entries — not a CAB map")
    return entries


class CabIndex:
    """CAB name → entry across several maps; **later maps win**.

    The game overlays its downloaded layer on the installed one, and a CAB the
    newer layer re-ships replaces the installed copy.  Pass the maps in that
    order (installed first).
    """

    def __init__(self, entries: dict[str, CabEntry]):
        self._entries = entries
        self._slots = {(str(e.source).lower(), e.offset): e.name
                       for e in entries.values()}

    @classmethod
    def load(cls, paths: Iterable[Path]) -> CabIndex:
        """Read *paths* in order; an entry in a later map replaces an earlier one."""
        merged: dict[str, CabEntry] = {}
        for path in paths:
            merged.update(read_cabmap(path))
        return cls(merged)

    def __len__(self) -> int:
        """How many serialized files the maps list."""
        return len(self._entries)

    def outside(self, root: Path) -> int:
        """How many entries' chunk files are not below *root*: maps made from another install say so here."""
        prefix = str(Path(root)).lower().rstrip("\\/")
        return sum(not str(e.source).lower().startswith((prefix + "\\", prefix + "/"))
                   for e in self._entries.values())

    def pointer(self, cab: str, pptr: dict | None) -> Ref | None:
        """An exported ``{m_FileID, m_PathID}`` inside *cab*, resolved.

        ``None`` for a null pointer (path id 0) or a missing one.
        """
        if not pptr or not int(pptr.get("m_PathID", 0)):
            return None
        return Ref(self.external(cab, int(pptr["m_FileID"])), int(pptr["m_PathID"]))

    def entry(self, cab: str) -> CabEntry:
        """The entry for *cab*; ``LookupError`` when no map lists it."""
        try:
            return self._entries[cab]
        except KeyError:
            raise LookupError(f"CAB map has no entry for {cab}") from None

    def at(self, source: Path | str, offset: int) -> str | None:
        """The CAB in bundle slot ``(source, offset)``, or ``None``."""
        return self._slots.get((str(Path(source)).lower(), int(offset)))

    def external(self, cab: str, file_id: int) -> str:
        """The serialized file a ``PPtr`` with *file_id* inside *cab* points into."""
        if file_id == 0:
            return cab
        deps = self.entry(cab).deps
        if not 0 < file_id <= len(deps):
            raise LookupError(f"{cab} has {len(deps)} externals, "
                              f"a pointer asks for file id {file_id}")
        return deps[file_id - 1]
