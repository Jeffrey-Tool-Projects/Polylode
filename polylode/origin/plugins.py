"""The install's BepInEx plugins that change a character: there or not, and how set.

A plugin decides things a card cannot: whether it runs at all is the install's
(``BepInEx/plugins``), and several read their switches from the player's own
settings file (``BepInEx/config/<guid>.cfg``), not from the card.  Runtime
reads the project cache only (``CLAUDE.md`` 四之一), so :func:`cache` — run by
the sync, like ``install.cache_etc_config`` for ``system.xml`` — records both
into ``cache_dir()/config/plugins.json``.  A cache that lacks a registered
plugin (no file yet, or written before the plugin was added to
:data:`PLUGINS`) is refilled from the source install when there is one; with
neither, that plugin reads as absent — the vanilla game.
"""
from __future__ import annotations

import json
from pathlib import Path

#: guid -> (DLL file name under ``BepInEx/plugins``, settings file under ``BepInEx/config``)
PLUGINS: dict[str, tuple[str, str]] = {
    "kokaiinum.KKExpandMaleMaker": ("KK_ExpandMaleMaker.dll",
                                    "kokaiinum.KKExpandMaleMaker.cfg"),
    "com.deathweasel.bepinex.colliders": ("KK_Colliders.dll",
                                          "com.deathweasel.bepinex.colliders.cfg"),
    # IllusionFixes; no settings file
    "KK_Fix_HairShadows": ("KK_Fix_HairShadows.dll", "KK_Fix_HairShadows.cfg"),
    "KK_Fix_ModdedHeadEyeliner": ("KK_Fix_ModdedHeadEyeliner.dll",
                                  "KK_Fix_ModdedHeadEyeliner.cfg"),
    # ``General.Migration enabled`` decides whether a card's UAR rows migrate (§7.488)
    "com.bepis.bepinex.sideloader": ("Sideloader.dll", "com.bepis.bepinex.sideloader.cfg"),
    # which uncensored body / penis / balls a card gets (``uncensor_choice``, #93)
    "com.deathweasel.bepinex.uncensorselector": ("KK_UncensorSelector.dll",
                                                 "com.deathweasel.bepinex.uncensorselector.cfg"),
}

_STATE: dict = {}


def cache_path() -> Path:
    from .install import cache_dir
    return cache_dir() / "config" / "plugins.json"


def parse_cfg(text: str) -> dict[str, str]:
    """A BepInEx ``.cfg`` as ``{"Section.Key": "value"}`` (comments dropped)."""
    out: dict[str, str] = {}
    section = ""
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1].strip()
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            out[f"{section}.{k.strip()}"] = v.strip()
    return out


def scan(src_root: str | Path) -> dict:
    """``{guid: {"dll": relative path or None, "cfg": {...}}}`` for an install."""
    root = Path(src_root)
    plugins = root / "BepInEx" / "plugins"
    config = root / "BepInEx" / "config"
    dlls = {p.name.lower(): p for p in plugins.rglob("*.dll")} if plugins.is_dir() else {}
    out = {}
    for guid, (dll, cfg) in PLUGINS.items():
        hit = dlls.get(dll.lower())
        settings = {}
        f = config / cfg
        if f.is_file():
            settings = parse_cfg(f.read_text(encoding="utf-8", errors="replace"))
        out[guid] = {"dll": str(hit.relative_to(root)) if hit else None,
                     "cfg": settings}
    return out


def cache(src_root: str | Path) -> dict:
    """Write :func:`scan` of *src_root* into the project cache (sync step)."""
    data = scan(src_root)
    out = cache_path()
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "reason": str(exc)}
    _STATE.clear()
    return {"ok": True, "dest": str(out),
            "installed": sorted(g for g, v in data.items() if v["dll"])}


def _load() -> dict:
    hit = _STATE.get("v")
    if hit is not None:
        return hit
    f = cache_path()
    data = _read(f)
    if any(g not in data for g in PLUGINS):
        # no cache yet, or one written before a plugin was registered here
        from .install import source_install
        src = source_install()
        if src.valid:
            cache(src.root)
            data = _read(f)
    _STATE["v"] = data
    return data


def _read(f: Path) -> dict:
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def installed(guid: str) -> bool:
    """Does the install run *guid*?"""
    return bool((_load().get(guid) or {}).get("dll"))


def setting(guid: str, key: str, default: str) -> str:
    """*guid*'s ``Section.Key`` setting; *default* (the plugin's own) when unset."""
    return str(((_load().get(guid) or {}).get("cfg") or {}).get(key, default))


def settings(guid: str) -> dict:
    """Every ``Section.Key`` setting *guid*'s file has (empty when unset or not installed)."""
    return dict((_load().get(guid) or {}).get("cfg") or {})


def flag(guid: str, key: str, default: bool) -> bool:
    """A boolean setting, BepInEx-style (``true`` / ``false``)."""
    return setting(guid, key, "true" if default else "false").lower() == "true"
