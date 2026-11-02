"""Discover conversions from actual folders, without a character-name list or a required index."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from . import store


@dataclass
class Scan:
    """Discovered descriptors and explicit invalid-package diagnostics."""
    items: list[dict] = field(default_factory=list)
    problems: list[dict] = field(default_factory=list)


def scan(base: Path | None = None, *, source: str | None = None) -> Scan:
    """Scan descriptor filenames only; original textures and model arrays are never thumbnail-scanned.

    Games are known source adapters. Character and outfit names come entirely
    from folder contents. Index files are build reports, not the discovery list.
    """
    base = Path(base) if base is not None else store.root()
    result = Scan()
    if source is not None and source not in store.SOURCES:
        raise store.WarehouseError(f"unsupported conversion source: {source}")
    roots = [(key, base / game) for key, game in store.SOURCES.items() if source is None or source == key]
    for expected, folder in roots:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*/*/warehouse.json")):
            try:
                if not path.resolve().is_relative_to(folder.resolve()):
                    raise store.WarehouseError("warehouse descriptor leaves the source folder")
                data = json.loads(path.read_text(encoding="utf-8"))
                if (data.get("version") != store.VERSION or data.get("source") != expected
                        or not data.get("ident") or path.parent.name != data["ident"]):
                    raise store.WarehouseError("invalid or misidentified warehouse descriptor")
                result.items.append({"source": expected, "ident": data["ident"],
                                     "character": data.get("character", path.parent.parent.name),
                                     "status": data.get("status"), "path": path,
                                     "sha256": store.digest(path)})
            except (OSError, ValueError) as exc:
                result.problems.append({"path": str(path), "error": str(exc)})
    return result
