"""Where a game is installed: the folder holding its data, from a folder the user named or from Steam's libraries (§7.1454).

A source names its game by the data it reads (:attr:`PathSpec.marker`, e.g.
``Client/Content/Paks``), never by a folder or Steam app name: launchers
and Steam put the same data under different parents (``Wuthering Waves/
Wuthering Waves Game/…``, ``steamapps/common/Arknights Endfield/…``).  So
the folder a user picks may be the game folder, one above it or one inside
its data; :func:`game_root` finds the folder the marker hangs from, and says
nothing (``None``) rather than picking when two games' data are below it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

#: how far below a named folder the data may start (``steamapps/common/<game>/<launcher folder>/<data>``)
DEPTH = 3


def _marked(folder: Path, marker: str) -> bool:
    try:
        return (folder / marker).is_dir()
    except OSError:
        return False


def candidates(folder: Path, marker: str, depth: int = DEPTH, up: bool = True) -> list[Path]:
    """Every folder at or below *folder* (to *depth* levels) that *marker* hangs from; with *up*, first the folder
    itself or one above it up to the marker's own depth (a folder picked inside the game's data)."""
    folder = Path(folder)
    for parent in [folder, *(list(folder.parents)[:len(Path(marker).parts)] if up else [])]:
        if _marked(parent, marker):
            return [parent]
    found, level = [], [folder]
    for _ in range(depth):
        below = []
        for here in level:
            try:
                below += sorted(p for p in here.iterdir() if p.is_dir())
            except OSError:
                continue
        found += [p for p in below if _marked(p, marker)]
        level = [p for p in below if not _marked(p, marker)]
    return found


def game_root(folder: Path, marker: str) -> Path | None:
    """The one folder at, above or below *folder* that holds the game's data; ``None`` when none or several do."""
    found = candidates(folder, marker)
    return found[0] if len(found) == 1 else None


def libraries_from_vdf(text: str) -> list[Path]:
    """The library folders a ``steamapps/libraryfolders.vdf`` lists (its ``"path"`` values)."""
    return [Path(value.replace("\\\\", "\\")) for value in re.findall(r'"path"\s+"((?:[^"\\]|\\.)*)"', text)]


def steam_root() -> Path | None:
    """Steam's own folder, as the registry records it."""
    if sys.platform != "win32":
        return None
    import winreg
    for hive, key, value in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                             (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
                             (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath")):
        try:
            with winreg.OpenKey(hive, key) as handle:
                path = Path(winreg.QueryValueEx(handle, value)[0])
        except OSError:
            continue
        if path.is_dir():
            return path
    return None


def steam_libraries(root: Path | None = None) -> list[Path]:
    """The ``steamapps/common`` folders of every Steam library on this machine."""
    root = steam_root() if root is None else root
    if root is None:
        return []
    libraries = [root]
    vdf = root / "steamapps" / "libraryfolders.vdf"
    try:
        libraries += libraries_from_vdf(vdf.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        pass
    common, seen = [], set()
    for library in libraries:
        folder = library / "steamapps" / "common"
        key = str(folder).casefold()
        if key not in seen and folder.is_dir():
            seen.add(key)
            common.append(folder)
    return common


def steam_installs(marker: str, root: Path | None = None) -> list[Path]:
    """The game folders under the Steam libraries that hold *marker*: each installed game is looked into."""
    found = []
    for common in steam_libraries(root):
        try:
            games = sorted(p for p in common.iterdir() if p.is_dir())
        except OSError:
            continue
        for game in games:
            found += candidates(game, marker, DEPTH - 1, up=False)
    return found
