"""Once, on the first start of the packaged Polylode: the extractor's data from where it lived when the tool was
part of IslandLife (``%LOCALAPPDATA%/IslandLife``), so a user keeps their settings, the Endfield asset maps
(tens of minutes to build), the GF2 bundle index and the stores, instead of making them again.

Each item is moved only when it is there and Polylode has none of its own yet; nothing else of IslandLife's
folder is touched (the game's own data stays).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..core.paths import _platform_user_root, frozen

#: what the extractor wrote under the old folder, relative to it
ITEMS = ("extractor", "assets_cache/extractor", "assets_cache/endfield/meta", "assets_cache/endfield/map_build",
         "assets_cache/wuthering_waves")


def from_islandlife(log=print, old: Path | None = None, new: Path | None = None) -> list[str]:
    """Move :data:`ITEMS` from *old* (IslandLife's folder) to *new* (this program's); what was moved."""
    if old is None and not frozen():
        return []
    new = new or _platform_user_root()
    old = old or new.parent / "IslandLife"
    moved = []
    for item in ITEMS:
        source, target = old / item, new / item
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append(item)
    if moved:
        log(f"moved from {old}: {', '.join(moved)}")
    return moved
