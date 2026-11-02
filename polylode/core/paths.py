"""UserData directory layout, mirroring the folder structure Koikatsu uses.

Two roots, and the split is the whole point (著作權.md 六 選項 B):

``ROOT``
    Where the code is.  In a release that is the frozen payload, and it must
    contain nothing derived from Illusion's data — not the bundles, and not our
    own converted format either, because converting does not change who owns
    the content.

``derived_root()``
    Where anything converted from *the user's own install* lands:
    ``assets_cache`` (their bundles), ``assets_baked`` (our format, baked from
    those bundles on their machine) and their ``UserData``.  Never shipped,
    never in version control.

In a source checkout the two are the same directory, so every tool, every
measurement in ``docs/updates/`` and every path written down over the last
hundred-odd sections keeps working unchanged.  They only separate once frozen,
which is exactly when ``ROOT`` stops being writable and starts being the thing
we hand to somebody else.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from . import swallowed

APP_NAME = "Polylode"

# Project root = parent of the `polylode` package.
ROOT = Path(__file__).resolve().parents[2]

DERIVED_ROOT_ENV = "ISLANDLIFE_DERIVED_ROOT"


def frozen() -> bool:
    """True when running from a packaged build rather than a checkout.

    Not "is ``polylode/__init__.py`` there": a bundle ships the package, so
    ROOT looks exactly like a checkout from the inside.

    **Two signals, because the packagers disagree.**  Measured on this machine
    (§7.151): PyInstaller sets ``sys.frozen = True``; **Nuitka does not set it
    at all** — it injects ``__compiled__`` into every compiled module's globals
    instead.  Reading only ``sys.frozen`` under Nuitka returns False, which
    sends ``derived_root()`` back to ``ROOT`` and writes the user's converted
    assets into the release directory.  Nothing reports that; it just quietly
    undoes §7.145.

    ``globals()`` here is this module's, so ``__compiled__`` answers for the
    build that compiled *paths.py*, not for whatever happens to call it.
    """
    return bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()


def _platform_user_root() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def derived_root() -> Path:
    """Root for everything converted from the user's own Illusion install."""
    env = os.environ.get(DERIVED_ROOT_ENV)
    if env:
        return Path(env)
    if not frozen():
        return ROOT
    return _platform_user_root()


USERDATA = derived_root() / "UserData"

CHARA_FEMALE = USERDATA / "chara" / "female"
CHARA_MALE = USERDATA / "chara" / "male"
#: Cards of converted characters (``chara.card.save_actor``): kept apart, the card lists of the other screens read only card characters
CHARA_CONVERTED = USERDATA / "chara" / "converted"
#: The source game's cards kept as test sources for the tools and as the source cards converted
#: bodies were built from (owner, 2026-10-16): in the development tree only, outside ``UserData``,
#: so the screens neither list nor open them and a package does not carry them
SOURCE_CARDS = ROOT / "reference" / "cards"
#: the test source the tools use when none is named: the card the reference shots were taken with
SOURCE_DEFAULTS = {0: "帰国子女.png", 1: "BetterRepack_BR-Kun.png"}
COORDINATE = USERDATA / "coordinate"
STUDIO_SCENE = USERDATA / "studio" / "scene"
STUDIO_POSE = USERDATA / "studio" / "pose"
#: Free H scenes recorded for the studio: one folder per recording, a pose track per body (D3)
STUDIO_RECORDING = USERDATA / "studio" / "recording"
CAPTURE = USERDATA / "cap"
CONFIG_DIR = USERDATA / "config"
ALL_DIRS = (CHARA_FEMALE, CHARA_MALE, COORDINATE, STUDIO_SCENE,
            STUDIO_POSE, STUDIO_RECORDING, CAPTURE, CONFIG_DIR)
#: settings and caches an earlier version kept under other names: this version's → that version's
LEGACY_CONFIG = {"origin.json": "kkassets.json", "origin_rig.json": "kk_rig.json",
                 "origin_lists.json": "kk_lists.json", "origin_sideloader.json": "kk_sideloader.json",
                 **{f"origin_studio_{kind}.json": f"kk_studio_{kind}.json"
                    for kind in ("items", "maps", "bones", "anime", "hand")}}


def migrate_config_names(folder: Path = CONFIG_DIR) -> list[str]:
    """Give the files :data:`LEGACY_CONFIG` names their current names; the ones moved.

    A file already under its current name is kept and the old one left
    alone (it is what this version wrote since).
    """
    moved = []
    for new, old in LEGACY_CONFIG.items():
        source, target = folder / old, folder / new
        if source.is_file() and not target.exists():
            try:
                source.replace(target)
                moved.append(old)
            except OSError as exc:
                swallowed.note(f"core.paths.migrate_config_names: {old}", exc)
    return moved


migrate_config_names()


# Subdirectories of ``UserData`` that are **by definition not defaults**:
# somebody's cards, and this machine's own output.  A save, a screenshot or a
# character card cannot be a default, so seeding must never carry one — and
# that has to be true here and not only in ``IslandLife.spec``, because any
# other path that copies ``UserData`` would otherwise have to remember it
# separately.  It is not hypothetical: simulating a frozen build inside a
# checkout makes ``ROOT`` the development tree, and the first time that was
# done this function copied **1,388 character cards (~1.6 GB)** into
# ``%LOCALAPPDATA%`` (§7.149).
_NOT_DEFAULTS = ("chara", "cap", "save", "studio", "coordinate")


def _seed_userdata() -> None:
    """Copy the release's bundled ``UserData`` defaults out to the user's copy.

    Only reached when frozen, and only for files that are not there yet: the
    shipped tree is defaults (``config/startup.json``, sounds we authored), the
    user's copy is theirs.  Overwriting would throw away a save every launch.
    """
    src = ROOT / "UserData"
    dest = USERDATA
    if not src.is_dir() or src.resolve() == dest.resolve():
        return
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        if rel.parts and rel.parts[0] in _NOT_DEFAULTS:
            continue
        out = dest / rel
        if out.exists():
            continue
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
        except OSError:
            continue


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    if frozen():
        _seed_userdata()


def chara_dir(sex: int | str = 0) -> Path:
    """sex: 0/'female' -> female folder, 1/'male' -> male folder."""
    female = sex in (0, "female", "f")
    return CHARA_FEMALE if female else CHARA_MALE


def unique_path(folder: Path, stem: str, suffix: str = ".png") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in stem if c not in '<>:"/\\|?*').strip() or "untitled"
    p = folder / f"{safe}{suffix}"
    i = 1
    while p.exists():
        p = folder / f"{safe}_{i}{suffix}"
        i += 1
    return p


def write_json_atomic(path: Path, data, **dumps) -> Path:
    """Serialise *data* to *path* so a failure never leaves a half-written file.

    ``Path.write_text`` truncates the target and then writes, so anything that
    fails in between — a ``MemoryError`` while encoding, a killed process, a
    full disk — leaves an **empty** file where a good one was.  That is how
    ``assets_baked/h_anim/hface.json`` ended up at 0 bytes (§7.357), and the
    same pattern wrote the save slots, the user's settings and every bake
    manifest.  Here the bytes go to a sibling ``.tmp`` first and
    ``os.replace`` swaps it in, which is atomic on one volume: readers see the
    old file or the new one, never neither.

    ``dumps`` goes to ``json.dumps`` (``ensure_ascii=False`` unless given).
    Creates the parent directory.  Returns *path*.

    Windows refuses the swap when another process holds *path* open without
    ``FILE_SHARE_DELETE`` — a reader that shares writing but not deleting; one
    ``h_anim`` clip stayed that way for hours and failed every H rebake at the
    same file (§7.521).  The bytes are complete in the ``.tmp`` by then, so they
    go into the existing file in place instead; that one write is not atomic,
    and it is counted (``swallowed.note``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dumps.setdefault("ensure_ascii", False)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, **dumps), encoding="utf-8")
        try:
            os.replace(tmp, path)
        except PermissionError as exc:
            if not path.is_file():
                raise
            swallowed.note("core.paths.write_json_atomic: os.replace denied, written in place", exc)
            with open(path, "r+b") as fh:
                fh.write(tmp.read_bytes())
                fh.truncate()
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def _resolve_project_path(named: str) -> Path | None:
    """Resolve config path: absolute, or relative to project ROOT."""
    if not named:
        return None
    p = Path(named)
    if p.is_file():
        return p
    # relative to project root (portable — no drive letter)
    rel = ROOT / named
    if rel.is_file():
        return rel
    # strip leading project name mistakes
    try:
        # if stored as absolute under this ROOT, still works via Path
        if p.exists():
            return p
    except OSError:
        pass
    return None


def default_card(sex: int = 0) -> Path | None:
    """The card the maker / Free H opens with for ``sex`` (0 female / 1 male).

    Order:
    * female: ``startup.default_card`` → newest local female character card of ours
    * male:   ``startup.default_card_male`` → newest local male character card of ours
    Project tree only — no external install path.
    """
    cfg = Config("startup")
    key = "default_card" if sex in (0, "female", "f") else "default_card_male"
    named = cfg.get(key)
    if named:
        p = _resolve_project_path(str(named))
        if p is not None:
            return p
    # female fallback: older configs only had default_card
    if sex not in (0, "female", "f"):
        pass
    from ..chara.card import own_cards
    cards = own_cards(chara_dir(sex))     # never an original-format card: those are test material (§7.1174)
    if cards:
        return max(cards, key=lambda p: p.stat().st_mtime)
    return None


def set_default_card(path: Path | None, sex: int = 0) -> None:
    key = "default_card" if sex in (0, "female", "f") else "default_card_male"
    if path is None:
        Config("startup").set(key, "")
        return
    p = Path(path)
    try:
        rel = p.resolve().relative_to(ROOT.resolve())
        Config("startup").set(key, rel.as_posix())
    except ValueError:
        Config("startup").set(key, str(p))


def list_cards(folder: Path) -> list[Path]:
    """PNG cards in ``folder`` (top-level and one nested level)."""
    if not folder.exists():
        return []
    # Flattened copies live at the top level; also pick up any subfolders.
    found = [p for p in folder.rglob("*.png") if p.is_file()]
    return sorted(found, key=lambda p: p.name.lower())


def source_dir(sex: int | str = 0) -> Path:
    """The test source cards of one sex (:data:`SOURCE_CARDS`)."""
    return SOURCE_CARDS / ("female" if sex in (0, "female", "f") else "male")


def source_cards(sex: int | str = 0) -> list[Path]:
    """Every test source card of one sex."""
    return list_cards(source_dir(sex))


def source_card(sex: int = 0) -> Path | None:
    """The test source card a tool uses when it is given none: :data:`SOURCE_DEFAULTS`, else the first listed."""
    female = sex in (0, "female", "f")
    named = source_dir(sex) / SOURCE_DEFAULTS[0 if female else 1]
    if named.is_file():
        return named
    cards = source_cards(sex)
    return cards[0] if cards else None


def stored_card(path: str | Path | None) -> Path | None:
    """A card path stored before the source cards left ``UserData/chara`` (§7.1124): the same card where it is now.

    A path that exists is returned as it is; one under the old card folders is
    looked for under the same name among the test source cards.  None when the
    card is in neither place.
    """
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return p
    parts = [part.lower() for part in p.parts]
    if len(parts) >= 3 and parts[-3] == "chara" and parts[-2] in ("female", "male"):
        moved = source_dir(parts[-2]) / p.name
        if moved.is_file():
            return moved
    return None


class Config:
    """Tiny JSON-backed settings store."""

    def __init__(self, name: str = "config"):
        self.path = CONFIG_DIR / f"{name}.json"
        self.data: dict = {}
        self.load()

    def load(self) -> None:
        try:
            if self.path.exists():
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            swallowed.note("core.paths.Config.load: if self.path.exists():", exc)
            self.data = {}

    def save(self) -> None:
        # A settings file that cannot be written must not take the caller down
        # (every screen saves on change), but it is said, not swallowed.
        try:
            write_json_atomic(self.path, self.data, indent=2)
        except (OSError, TypeError, ValueError) as exc:
            print(f"[config] 寫不進 {self.path}: {type(exc).__name__}: {exc}")

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value
        self.save()


ensure_dirs()
