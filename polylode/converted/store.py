"""Read self-contained converted-character packages (GF2, Wuthering Waves, Arknights) without source fallbacks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..core.paths import derived_root

SOURCES = {"gf2": "GF2", "wuthering_waves": "WutheringWaves", "arknights": "Arknights"}
VERSION = 1


class WarehouseError(ValueError):
    """A conversion is missing, stale, or not ready for the requested use."""


def root() -> Path:
    """The user's consolidated conversion folder, never a distributable asset."""
    return derived_root() / "Charas"


def digest(path: Path) -> str:
    """Hash a file in bounded memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def child(folder: Path, relative: str) -> Path:
    """Resolve a required package member, refusing absolute paths and traversal."""
    relative = str(relative)
    target = (folder / relative).resolve()
    if Path(relative).is_absolute() or not target.is_relative_to(folder.resolve()) or not target.is_file():
        raise WarehouseError(f"missing or external warehouse member: {relative}")
    return target


def read(folder: Path, *, verify: bool = True) -> dict:
    """Read a package and optionally verify every member before geometry is loaded."""
    folder = Path(folder)
    data = json.loads((folder / "warehouse.json").read_text(encoding="utf-8"))
    if data.get("version") != VERSION or data.get("source") not in SOURCES or not data.get("ident"):
        raise WarehouseError(f"unsupported character warehouse: {folder}")
    if not data.get("files"):
        raise WarehouseError("warehouse has no certified files")
    for relative, expected in data["files"].items():
        path = child(folder, relative)
        if verify and digest(path) != expected:
            raise WarehouseError(f"warehouse member changed; rebuild: {relative}")
    return data


def locate(source: str, ident: str, *, base: Path | None = None) -> Path:
    """Dynamically find one source-qualified item; an index or character-name list is not required."""
    from .discovery import scan
    if source not in SOURCES:
        raise WarehouseError(f"unsupported conversion source: {source}")
    found = scan(base, source=source)
    rows = [row for row in found.items if row["ident"] == ident]
    if len(rows) != 1:
        raise WarehouseError(f"warehouse item missing or ambiguous: {source}/{ident}")
    descriptor = rows[0]["path"]
    data = read(descriptor.parent)
    if data["source"] != source or data["ident"] != ident:
        raise WarehouseError("warehouse index identifies another character")
    return descriptor.parent
