"""AnimeStudio's asset map, reduced to the questions the converter asks.

The map is one JSON document per VFS root (877 MB for the installed layer)
whose ``AssetEntries`` are ``{Name, Container, Source, PathID, Type, Offset}``.
It lists named assets, not every object, so it answers:

* **which asset path a bundle slot holds** — every distinct
  ``(Container, Source, Offset)``: which prefabs exist and where, and which
  model file a serialized file (an Avatar's, say) belongs to;
* **what a shader is called** — a material points at its shader by ``PPtr``,
  and the name lives in the shader's own (large) file;
* **which animation clips exist** under a container folder.

Everything else — which mesh a renderer draws, which texture a material binds —
is read from the objects themselves (:mod:`.prefab`, :mod:`.material`).

Reading the documents takes about half a minute, so the answers are kept in a
small SQLite file beside the staging data and rebuilt when a map's size or
modification time, or this module's schema, changes.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple

Log = Callable[[str], None] | None

_CHUNK = 1 << 22
#: Bumped when the tables change, so an index built by an older build is redone.
_SCHEMA_VERSION = 2
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS slot (
    container TEXT, source TEXT, offset INTEGER,
    PRIMARY KEY (container, source, offset));
CREATE INDEX IF NOT EXISTS slot_by_place ON slot (source, offset);
CREATE TABLE IF NOT EXISTS shader (
    source TEXT, offset INTEGER, path_id INTEGER, name TEXT,
    PRIMARY KEY (source, offset, path_id));
CREATE TABLE IF NOT EXISTS clip (
    container TEXT, name TEXT, source TEXT, offset INTEGER, path_id INTEGER,
    PRIMARY KEY (source, offset, path_id));
CREATE INDEX IF NOT EXISTS clip_by_container ON clip (container);
"""
_TABLES = ("slot", "shader", "clip")


class ClipRow(NamedTuple):
    """One ``AnimationClip`` the map lists."""
    container: str
    name: str
    source: Path
    offset: int
    path_id: int


def iter_entries(path: Path) -> Iterator[dict]:
    """Each object of the document's ``AssetEntries`` array, without loading it.

    Decodes one object at a time off a sliding buffer, so it does not depend on
    how the writer laid the document out.
    """
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as fh:
        buf = fh.read(_CHUNK)
        start = buf.find('"AssetEntries"')
        if start < 0:
            raise ValueError(f"{path}: no AssetEntries array")
        pos = buf.index("[", start) + 1
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                entry, pos = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = fh.read(_CHUNK)
                if not more:
                    raise ValueError(f"{path}: truncated AssetEntries") from None
                buf = buf[pos:] + more
                pos = 0
                continue
            yield entry


def _signature(paths: Iterable[Path]) -> str:
    return json.dumps([_SCHEMA_VERSION] + [
        [str(p), p.stat().st_size, int(p.stat().st_mtime)] for p in paths])


class AssetMap:
    """The slot, shader and clip rows of one or more asset maps."""

    def __init__(self, db: sqlite3.Connection):
        self._db = db

    @classmethod
    def open(cls, maps: list[Path], index: Path, log: Log = None) -> AssetMap:
        """Open *index*, (re)building it from *maps* when they changed."""
        index.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(index)
        db.executescript(_SCHEMA)
        want = _signature(maps)
        row = db.execute("SELECT value FROM meta WHERE key='maps'").fetchone()
        if row is None or row[0] != want:
            cls._build(db, maps, log)
            db.execute("INSERT OR REPLACE INTO meta VALUES ('maps', ?)", (want,))
            db.commit()
        return cls(db)

    @staticmethod
    def _build(db: sqlite3.Connection, maps: list[Path], log: Log) -> None:
        db.execute("DROP TABLE IF EXISTS prefab")       # schema 1's table
        for table in _TABLES:
            db.execute(f"DELETE FROM {table}")
        for path in maps:
            if log:
                log(f"indexing {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
            for entry in iter_entries(path):
                container = str(entry.get("Container") or "")
                slot = (str(Path(entry["Source"])), int(entry["Offset"]))
                kind = entry.get("Type")
                if container:
                    db.execute("INSERT OR IGNORE INTO slot VALUES (?, ?, ?)",
                               (container, *slot))
                if kind == "Shader":
                    db.execute("INSERT OR REPLACE INTO shader VALUES (?, ?, ?, ?)",
                               (*slot, int(entry["PathID"]), str(entry["Name"])))
                elif kind == "AnimationClip" and container:
                    db.execute("INSERT OR REPLACE INTO clip VALUES (?, ?, ?, ?, ?)",
                               (container, str(entry["Name"]), *slot,
                                int(entry["PathID"])))

    def prefabs(self, like: str) -> dict[str, list[tuple[Path, int]]]:
        """Container → bundle slots, for containers matching SQL ``LIKE`` *like*."""
        out: dict[str, list[tuple[Path, int]]] = {}
        rows = self._db.execute(
            "SELECT container, source, offset FROM slot "
            "WHERE container LIKE ? ORDER BY container, source, offset", (like,))
        for container, source, offset in rows:
            out.setdefault(container, []).append((Path(source), int(offset)))
        return out

    def containers_at(self, source: Path, offset: int) -> list[str]:
        """Every asset path the map files under bundle slot ``(source, offset)``."""
        rows = self._db.execute(
            "SELECT container FROM slot WHERE source=? AND offset=? "
            "ORDER BY container", (str(Path(source)), int(offset)))
        return [row[0] for row in rows]

    def shader_name(self, source: Path, offset: int, path_id: int) -> str | None:
        """The shader at ``(slot, path id)``, or ``None`` when the map lists none."""
        row = self._db.execute(
            "SELECT name FROM shader WHERE source=? AND offset=? AND path_id=?",
            (str(Path(source)), int(offset), int(path_id))).fetchone()
        return None if row is None else row[0]

    def shaders(self, like: str) -> list[tuple[str, Path, int, int]]:
        """``(name, source, offset, path id)`` of the shaders named like *like*."""
        rows = self._db.execute(
            "SELECT name, source, offset, path_id FROM shader WHERE name LIKE ? "
            "ORDER BY name", (like,))
        return [(n, Path(s), int(o), int(p)) for n, s, o, p in rows]

    def clips(self, like: str) -> list[ClipRow]:
        """The clips whose container matches SQL ``LIKE`` *like*, by container."""
        rows = self._db.execute(
            "SELECT container, name, source, offset, path_id FROM clip "
            "WHERE container LIKE ? ORDER BY container, name", (like,))
        return [ClipRow(c, n, Path(s), int(o), int(p)) for c, n, s, o, p in rows]
