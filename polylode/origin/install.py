"""Koikatsu data for this project — always under the project tree.

Runtime never depends on a hard-coded game path.  Assets live in
``assets_cache/`` (git-ignored); character cards live in ``UserData/chara/``.

An external install may be passed **once** to ``sync_from_install()`` to fill
the cache.  After that, ``configured()`` resolves only project-local paths.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..core.i18n import tr
from ..core.paths import Config, ROOT, USERDATA, chara_dir, derived_root, ensure_dirs

# Optional one-shot discovery only (no BetterRepack / drive-letter hardcoding).
# Prefer env var; otherwise leave unset so the project uses assets_cache alone.
ENV_INSTALL = "ISLANDLIFE_ORIGIN_INSTALL"
#: the name an earlier version read; still read when the current one is not set
LEGACY_ENV_INSTALL = "ISLANDLIFE_KK_INSTALL"


def _env_install() -> str:
    return os.environ.get(ENV_INSTALL, "") or os.environ.get(LEGACY_ENV_INSTALL, "")


def cache_dir() -> Path:
    """The user's own game data (bundles + zipmods), copied out of their install.

    Under ``derived_root()``, not ``ROOT``: these are Illusion's bundles
    verbatim, so they never travel with the code (四之一).  Same directory in a
    checkout, a per-user one once frozen.
    """
    env = os.environ.get("ISLANDLIFE_ASSET_CACHE")
    if env:
        return Path(env)
    return derived_root() / "assets_cache"


def project_mods_dir() -> Path:
    return cache_dir() / "mods"


@dataclass
class Install:
    root: Path
    abdata: Path
    valid: bool = False
    reason: str = ""
    local: bool = False          # True → root is project assets_cache layout

    @property
    def lists(self) -> Path:
        return self.abdata / "list" / "characustom"

    @property
    def chara(self) -> Path:
        return self.abdata / "chara"

    @property
    def userdata(self) -> Path:
        if self.local:
            return USERDATA
        return self.root / "UserData"

    def userdata_chara(self, sex: int | str = 0) -> Path:
        female = sex in (0, "female", "f")
        return self.userdata / "chara" / ("female" if female else "male")

    def bundle(self, rel: str) -> Path:
        """Resolve ``chara/xxx.unity3d`` under the local cache first."""
        rel = rel.replace("\\", "/").lstrip("/")
        # Always prefer project cache
        cached = cache_dir() / rel
        if cached.exists():
            return cached
        # local install's abdata *is* the cache root layout
        under = self.abdata / rel
        if under.exists():
            return under
        return cached   # destination for future sync

    def describe(self) -> str:
        if not self.valid:
            return tr("未就緒（{reason}）").format(reason=self.reason)
        if self.local:
            n = len(list(self.chara.glob("*.unity3d"))) if self.chara.exists() else 0
            m = project_mods_dir()
            nm = len(list(m.rglob("*.zipmod"))) if m.is_dir() else 0
            return tr("專案資產庫 assets_cache（chara {n} 包 · zipmod {mods}）").format(n=n, mods=nm)
        n = len(list(self.chara.glob("*.unity3d"))) if self.chara.exists() else 0
        return tr("外部安裝 {root}（chara {n} 包）— 請執行同步到專案").format(root=self.root, n=n)


def _looks_like_abdata(ab: Path) -> bool:
    return ab.is_dir() and (ab / "list" / "characustom").is_dir()


def local_install() -> Install:
    """Install handle that reads only ``assets_cache/`` under the project."""
    ab = cache_dir()
    inst = Install(ROOT, ab, local=True)
    if _looks_like_abdata(ab):
        inst.valid = True
    elif (ab / "chara").is_dir() and any((ab / "chara").glob("*.unity3d")):
        # partial cache still usable for some paths
        inst.valid = True
        inst.reason = tr("assets_cache 不完整（缺 list/characustom 時 id 解析會失敗）")
    else:
        inst.reason = tr("assets_cache 尚未同步")
    return inst


def check(root: str | Path) -> Install:
    """Validate an *external* game install (only used when syncing in)."""
    root = Path(root)
    ab = root / "abdata"
    inst = Install(root, ab, local=False)
    if not root.exists():
        inst.reason = tr("路徑不存在")
        return inst
    if not ab.exists():
        inst.reason = tr("找不到 abdata 資料夾")
        return inst
    if not (ab / "list" / "characustom").exists():
        inst.reason = tr("找不到 abdata/list/characustom")
        return inst
    inst.valid = True
    return inst


def source_install() -> Install:
    """External install path from config / env — for sync only, not runtime."""
    cfg = Config("origin")
    p = cfg.get("install_path", "") or cfg.get("source_install", "")
    if p:
        return check(p)
    env = _env_install()
    if env:
        return check(env)
    return Install(Path(""), Path(""), False, tr("未設定來源安裝（僅同步時需要）"))


def configured() -> Install:
    """Runtime data source: **project assets_cache only**.

    Falls back to an external install only if the local cache is empty *and*
    a source path is still configured (so the first sync can run).
    """
    local = local_install()
    if local.valid and (local.lists.is_dir() or local.chara.is_dir()):
        # Prefer local whenever anything useful is present
        if local.lists.is_dir() or any(local.chara.glob("*.unity3d")):
            return local
    src = source_install()
    if src.valid:
        return src
    return local


def set_path(path: str | Path | None) -> Install:
    """Record external install as **sync source** only (not a runtime hard-code)."""
    cfg = Config("origin")
    cfg.set("install_path", str(path) if path else "")
    cfg.set("source_install", str(path) if path else "")
    return source_install()


def set_use_local_only() -> None:
    """After a successful sync: stop reading the external install at runtime."""
    cfg = Config("origin")
    # Keep source_install for optional re-sync; clear active install_path so
    # configured() stays on assets_cache.
    cfg.set("install_path", "")
    cfg.set("prefer_local", True)


def autodetect() -> Install | None:
    """No drive-letter guessing. Env var only."""
    env = _env_install()
    if env:
        inst = check(env)
        if inst.valid:
            return inst
    local = local_install()
    return local if local.valid else None


def enabled() -> bool:
    cfg = Config("origin")
    if not bool(cfg.get("enabled", True)):
        return False
    return configured().valid


def _mirror(src: Path, dest: Path, pattern: str = "*",
            only_missing: bool = True) -> tuple[int, int]:
    if not src.is_dir():
        return 0, 0
    n = total = 0
    for f in src.rglob(pattern):
        if not f.is_file():
            continue
        out = dest / f.relative_to(src)
        try:
            if only_missing and out.exists() and out.stat().st_size == f.stat().st_size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            n += 1
            total += f.stat().st_size
        except OSError:
            continue
    return n, total


def etc_config_path() -> Path:
    """Where the cached copy of the install's ``system.xml`` lives."""
    return cache_dir() / "config" / "system.xml"


def cache_etc_config(src_root: str | Path) -> dict:
    """Copy the install's ``UserData/config/system.xml`` into the cache.

    Runtime reads the project cache only (§four), and this file is the one
    place the game records which of the seven toon ramps the player picked.
    It is a settings file rather than a game asset, and it is copied rather
    than parsed here so that a later reader can take more than the four values
    :mod:`figure.toon` currently wants.
    """
    src = Path(src_root) / "UserData" / "config" / "system.xml"
    if not src.is_file():
        return {"ok": False, "reason": tr("沒有 {path}").format(path=src)}
    out = etc_config_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, out)
    except OSError as exc:
        return {"ok": False, "reason": str(exc)}
    return {"ok": True, "dest": str(out), "bytes": out.stat().st_size}


def sync_from_install(
    source: str | Path | None = None,
    *,
    only_missing: bool = True,
    include_mods: bool = True,
    include_studio: bool = True,
    include_h: bool = True,
    include_chara_cards: bool = True,
) -> dict:
    """Copy everything needed from an external Origin install into the project.

    Destination layout (all under the repo)::

        assets_cache/chara/
        assets_cache/list/characustom/
        assets_cache/studio/
        assets_cache/map/
        assets_cache/h/
        assets_cache/mods/
        UserData/chara/{female,male}/   # cards

    Returns a report dict.  On success, runtime uses only project paths.
    """
    ensure_dirs()
    if source is not None:
        set_path(source)
    src = source_install()
    if not src.valid:
        # last chance: env
        auto = autodetect()
        if auto is not None and auto.valid and not auto.local:
            src = auto
            set_path(auto.root)
    report: dict = {"ok": False, "reason": "", "parts": {}}
    if not src.valid or src.local:
        report["reason"] = src.reason or tr("未設定外部來源安裝")
        # if local already good, that's fine
        loc = local_install()
        if loc.valid and loc.lists.is_dir():
            report["ok"] = True
            report["reason"] = tr("已使用專案 assets_cache")
            report["local"] = str(cache_dir())
        return report

    dest = cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    parts = report["parts"]

    jobs = [
        ("chara", src.abdata / "chara", dest / "chara", "*.unity3d"),
        ("list", src.abdata / "list" / "characustom",
         dest / "list" / "characustom", "*"),
        # ``customshape`` (which curve each slider plays) and ``shapecorrect``
        # (the offsets the bone-type body adds) live outside ``characustom``,
        # so the two jobs above miss them and shaping silently falls back to
        # "no table" on a freshly synced cache.
        ("list_root", src.abdata / "list", dest / "list", "*.unity3d"),
        ("shapecorrect", src.abdata / "list" / "shapecorrect",
         dest / "list" / "shapecorrect", "*"),
    ]
    if include_studio:
        jobs.append(("studio", src.abdata / "studio", dest / "studio", "*"))
        jobs.append(("map_list", src.abdata / "map" / "list",
                     dest / "map" / "list", "*"))
    if include_h:
        jobs.append(("h_anim", src.abdata / "h" / "anim",
                     dest / "h" / "anim", "*.unity3d"))
        # ``h/list`` holds ``AnimationInfo_{mode}`` — one row per selectable H
        # action, naming the pack + AnimatorController each role plays.  Without
        # it the packs are just 6,057 loose clips with no grouping, which is how
        # the menu ended up guessing families from pack filenames.
        jobs.append(("h_list", src.abdata / "h" / "list",
                     dest / "h" / "list", "*.unity3d"))
    if include_mods:
        jobs.append(("mods", src.root / "mods", dest / "mods", "*.zipmod"))

    grand = 0
    for label, s, d, pat in jobs:
        n, b = _mirror(s, d, pat, only_missing=only_missing)
        parts[label] = {"copied": n, "bytes": b, "dest": str(d)}
        grand += b

    # The colour grading LUT is not a character asset and not in abdata -- it
    # is a built-in resource beside the executable -- so the mirror jobs above
    # cannot reach it.  Extracted here instead of at runtime, because runtime
    # reads the project cache only (§four) and pulling one texture out of a
    # 600 MB sharedassets is not something to do while a screen is opening.
    try:
        from .grading import extract_from
        parts["grading"] = extract_from(src.root)
    except Exception as exc:                         # noqa: BLE001
        parts["grading"] = {"ok": False, "reason": str(exc)}

    # ``Config.EtcData`` for the same reason.  Three of the four globals it
    # feeds the character shaders match ``EtceteraSystem.Init``'s defaults on
    # every install, but ``rampId`` is a player setting and it decides whether
    # skin gets a binary terminator or a smooth one (§7.61).  Assuming the
    # constructor default is how we shipped a hard shadow edge round every
    # limb that the game does not draw.
    parts["etc_config"] = cache_etc_config(src.root)
    # The plugins whose switches the install, not the card, holds (which are
    # there, and ``BepInEx/config``) — same reason, see :mod:`plugins`.
    from .plugins import cache as cache_plugins
    parts["plugins"] = cache_plugins(src.root)

    if include_chara_cards:
        card_rep = copy_install_chara(
            sexes=(0, 1), only_missing=only_missing, source=src)
        parts["cards"] = card_rep

    # Prefer local-only at runtime
    set_use_local_only()
    # Rebuild zipmod index against project mods
    try:
        from .sideloader import refresh_index
        refresh_index()
        parts["sideloader"] = "rebuilt"
    except Exception as exc:  # noqa: BLE001
        parts["sideloader"] = f"rebuild failed: {exc}"

    report["ok"] = True
    report["source"] = str(src.root)
    report["local"] = str(dest)
    report["bytes"] = grand
    return report


def _safe_card_name(src: Path, root: Path) -> str:
    try:
        rel = src.relative_to(root)
    except ValueError:
        rel = Path(src.name)
    parts = [p for p in rel.parts[:-1] if p not in (".",)]
    stem = rel.stem if rel.suffix else src.stem
    raw = "_".join(parts + [stem]) if parts else stem
    safe = "".join(c for c in raw if c not in '<>:"/\\|?*').strip()
    safe = safe.replace("[", "").replace("]", "").strip(" ._") or "card"
    if len(safe) > 120:
        safe = safe[:120]
    return safe + ".png"


def copy_install_chara(
    *,
    sexes: tuple[int, ...] = (0, 1),
    limit_per_sex: int | None = None,
    only_missing: bool = True,
    source: Install | None = None,
) -> dict:
    """Copy female/male cards into project ``UserData/chara``."""
    ensure_dirs()
    inst = source or source_install()
    if not inst.valid:
        auto = autodetect()
        if auto is not None and auto.valid and not getattr(auto, "local", False):
            inst = auto
    report: dict = {"ok": False, "reason": "", "female": {}, "male": {}}
    # If only local, nothing to copy from outside
    if not inst.valid or getattr(inst, "local", False):
        # report local inventory
        for sex, label in ((0, "female"), (1, "male")):
            d = chara_dir(sex)
            n = len(list(d.rglob("*.png"))) if d.is_dir() else 0
            report[label] = {"copied": 0, "skipped": n, "dest": str(d),
                             "local_only": True}
        report["ok"] = True
        report["reason"] = tr("已使用專案 UserData/chara")
        return report

    report["ok"] = True
    report["source"] = str(inst.root / "UserData" / "chara")

    for sex in sexes:
        female = sex in (0, "female", "f")
        label = "female" if female else "male"
        src_root = inst.userdata_chara(sex)
        dest_root = chara_dir(sex)
        dest_root.mkdir(parents=True, exist_ok=True)
        copied = skipped = 0
        if not src_root.is_dir():
            report[label] = {"copied": 0, "skipped": 0, "source": str(src_root),
                             "missing": True}
            continue
        cards = [p for p in src_root.rglob("*.png") if p.is_file()]
        cards.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if limit_per_sex is not None:
            cards = cards[: max(0, int(limit_per_sex))]
        for src in cards:
            name = _safe_card_name(src, src_root)
            dest = dest_root / name
            try:
                if only_missing and dest.is_file():
                    if dest.stat().st_size == src.stat().st_size:
                        skipped += 1
                        continue
                shutil.copy2(src, dest)
                copied += 1
            except OSError:
                skipped += 1
        report[label] = {
            "copied": copied,
            "skipped": skipped,
            "total_src": len(cards),
            "dest": str(dest_root),
            "source": str(src_root),
        }
    return report
