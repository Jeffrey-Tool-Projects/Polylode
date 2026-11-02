"""Sideloader (.zipmod) support.

Modded items don't live in ``abdata`` — each ``.zipmod`` is a zip holding its own
``manifest.xml`` (with a GUID), CSV list files and asset bundles.  A card refers
to them indirectly: it stores a runtime slot number, and its
``com.bepis.sideloader.universalautoresolver`` extended data maps that slot back
to (GUID, real slot, property).

This builds a GUID -> category -> slot index once (a few seconds over thousands
of mods) and caches it, then loads bundles straight out of the zip.
"""
from __future__ import annotations

import csv as _csv
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ..core.paths import CONFIG_DIR, write_json_atomic
from . import lists as L
from ..core import swallowed

CACHE = CONFIG_DIR / "origin_sideloader.json"
# 1: bundle + asset only
# 2: the whole list row per item (ModRef.extra) — a modded head needs its own
#    ShapeAnime/material/textures, not just a prefab name (§7.41)
# 3: CATEGORY_NO transcribed from ChaListDefine.CategoryNo — 121–130 are the
#    accessories, not a second clothes block (§7.66).  The category name is
#    stored per item, so an index built under the old table has to be redone.
# 4: only the archive Sideloader loads per GUID (§7.479)
INDEX_VERSION = 5          # 5: the manifests' migration rows and the loaded GUIDs (§7.488)
GUID_RE = re.compile(rb"<guid>(.*?)</guid>", re.S)
VERSION_RE = re.compile(rb"<version>(.*?)</version>", re.S)

# basename -> resolved Path, filled lazily when install paths move
_zip_resolve_cache: dict[str, Path | None] = {}


def _mods_fingerprint(root: Path) -> tuple[int, float]:
    """``(zipmod count, newest mtime)`` under *root*, cheaply.

    ``os.scandir`` rather than ``Path.rglob`` because the mtime comes back with
    the directory entry on Windows — 4,291 local zipmods cost one walk instead
    of one walk plus 4,291 ``stat`` calls.
    """
    import os
    n, newest = 0, 0.0
    stack = [str(root)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.name.lower().endswith(".zipmod"):
                            n += 1
                            m = e.stat().st_mtime
                            if m > newest:
                                newest = m
                    except OSError:
                        continue
        except OSError:
            continue
    return n, newest


def mods_search_roots() -> list[Path]:
    """Directories that may contain ``.zipmod`` files — **project only**.

    Zipmods are mirrored into ``assets_cache/mods`` by ``sync_from_install``.
    Runtime never walks a hard-coded game drive path.
    """
    from .install import cache_dir, source_install
    roots: list[Path] = []
    cached = cache_dir() / "mods"
    if cached.is_dir():
        roots.append(cached)
    # During first-time sync only, also allow the external source mods tree
    src = source_install()
    if src.valid and not src.local:
        for name in ("mods", "Mods"):
            p = src.root / name
            if p.is_dir():
                roots.append(p)
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r).lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def install_drift() -> str:
    """Why ``assets_cache/mods`` no longer matches the install's ``mods``, or ``""``.

    The index notices a change *inside the cache* (:meth:`SideloaderIndex.is_stale`),
    but a zipmod added to the game after the last sync never reaches the cache
    at all — and then Origin's own rules make the card dress in something else
    without a word (``ResolveStructure`` keeps a garment's id for the base row,
    and drops an accessory, §7.72).  This compares the two folders with the
    same fingerprint the index uses: the count, then the newest mtime (the
    mirror copies with ``copy2``, so an in-sync pair has equal mtimes).

    ``""`` also when there is nothing to compare — no external install (a
    release machine), or no cache yet (that case has its own message).
    """
    from .install import project_mods_dir, source_install
    src = source_install()
    if not src.valid or src.local:
        return ""
    live = next((src.root / n for n in ("mods", "Mods")
                 if (src.root / n).is_dir()), None)
    cache = project_mods_dir()
    if live is None or not cache.is_dir():
        return ""
    n_live, t_live = _mods_fingerprint(live)
    n_cache, t_cache = _mods_fingerprint(cache)
    if n_live != n_cache:
        return f"安裝有 {n_live} 個 zipmod，assets_cache/mods 有 {n_cache} 個"
    if t_live > t_cache + 1.0:
        return "安裝裡有 zipmod 比 assets_cache/mods 的新"
    return ""


def resolve_zipmod_path(stored: str | Path) -> Path | None:
    """Locate a zipmod even when the cached absolute path is stale.

    The sideloader index stores absolute paths from the machine that built it.
    If the project was moved or the cache pointed at a missing ``assets_cache``,
    every mod load failed with "無法開啟".  We re-find by file name under the
    live install's ``mods/`` tree.
    """
    if not stored:
        return None
    # Cache the *hit* as well, keyed by what was asked for.  The index holds
    # ~27,000 refs across ~4,000 zipmods and ``repath()`` walks all of them, so
    # stat-ing before consulting the cache cost 51,000 syscalls — over a second
    # on the first card load — to re-derive 4,000 answers.
    skey = str(stored).lower()
    if skey in _zip_resolve_cache:
        return _zip_resolve_cache[skey]
    p = Path(stored)
    if p.is_file():
        _zip_resolve_cache[skey] = p
        return p
    key = p.name.lower()
    if key in _zip_resolve_cache:
        _zip_resolve_cache[skey] = _zip_resolve_cache[key]
        return _zip_resolve_cache[key]
    found: Path | None = None
    # Prefer path that preserves Sideloader Modpack relative tail
    tail = None
    s = str(stored).replace("\\", "/")
    for marker in ("/mods/", "/Mods/"):
        if marker in s:
            tail = s.split(marker, 1)[1]
            break
    for root in mods_search_roots():
        if tail:
            cand = root / Path(tail)
            if cand.is_file():
                found = cand
                break
        # basename search (last resort — first hit wins)
        hits = list(root.rglob(p.name))
        if hits:
            found = hits[0]
            break
    _zip_resolve_cache[key] = found
    _zip_resolve_cache[skey] = found
    return found

# ``ChaListDefine.CategoryNo`` → our category names, transcribed member by
# member from the enum (``origin_il_dump.py --enum ChaListDefine.CategoryNo``).  A
# zipmod's list CSV writes that number in its first cell, so this is the same
# key ``ChaListControl.GetCategoryInfo`` is called with — nothing here is a
# guess about "which block some pack happens to use".
#
# The names are the ones ``lists.CATEGORY`` gives the base table of the same
# enum member (``ao_head`` → ``acc_head``), so a modded item and a base item of
# the same kind land in the same bucket.
#
# Guessing here was expensive.  The old table read 121–129 as a second copy of
# the clothes block (top/bottom/bra/…) and invented an accessory block at
# 141–150 that the enum has no members for.  121–130 is **ao_\*** — the
# accessories — so 11,731 modded accessories (45% of every indexed item) were
# filed as garments: 4,850 hats under ``bottom``.  That let a hat answer a
# lookup for a skirt, and it made ``find_any`` useless for accessories.
CATEGORY_NO = {
    100: "head",
    101: "hair_back", 102: "hair_front", 103: "hair_side", 104: "hair_option",
    105: "top", 106: "bottom", 107: "bra", 108: "panty",
    109: "gloves", 110: "legwear", 111: "socks", 112: "shoes",
    # 120 ``ao_none`` is the empty slot and never names an asset.
    121: "acc_hair", 122: "acc_head", 123: "acc_face", 124: "acc_neck",
    125: "acc_body", 126: "acc_waist", 127: "acc_leg", 128: "acc_arm",
    129: "acc_hand", 130: "acc_kokan",
    # Uniform top sub-parts — a ``p_dummy`` top's real meshes.
    200: "cpo_sailor_a", 201: "cpo_sailor_b", 202: "cpo_sailor_c",
    210: "cpo_jacket_a", 211: "cpo_jacket_b", 212: "cpo_jacket_c",
    # Texture-only lists: the row *is* the map.
    400: "face_detail", 401: "eyeshadow", 402: "cheek", 403: "lip",
    404: "lipline", 405: "face_paint", 406: "eyebrow", 407: "eye_white",
    408: "eye", 409: "eye_gradation", 410: "eye_hi_up", 411: "eye_hi_down",
    412: "eyeline_up", 413: "eyeline_down", 414: "nose", 415: "mole",
    416: "face_preset",
    420: "body_detail", 421: "body_paint", 422: "sunburn", 423: "nip",
    424: "underhair", 430: "pattern", 431: "emblem", 432: "ramp",
    433: "hairgloss",
    # Placement presets (rows of numbers, no asset).
    3: "bodypaint_layout", 7: "facepaint_layout", 8: "mole_layout",
}
#: Our category name back to ``CategoryNo`` (the table above is one-to-one).
CATEGORY_NO_OF = {v: k for k, v in CATEGORY_NO.items()}


#: ``ChaListDefine.CategoryNo`` by member name (``origin_il_dump --enum
#: ChaListDefine.CategoryNo``, all 71).  A manifest's migration row names its
#: category this way and ``Manifest.LoadMigrationInfo`` reads it with
#: ``Enum.Parse`` (§7.488).
CATEGORY_ENUM = {
    "cha_sample_f": 0, "cha_sample_m": 1, "cha_eyeset": 2, "bodypaint_layout": 3,
    "cha_default_cos_f": 4, "cha_default_cos_m": 5, "cha_sample_voice": 6,
    "facepaint_layout": 7, "mole_layout": 8, "cha_sample_m_ex": 9,
    "bo_head": 100, "bo_hair_b": 101, "bo_hair_f": 102, "bo_hair_s": 103, "bo_hair_o": 104,
    "co_top": 105, "co_bot": 106, "co_bra": 107, "co_shorts": 108, "co_gloves": 109,
    "co_panst": 110, "co_socks": 111, "co_shoes": 112,
    "ao_none": 120, "ao_hair": 121, "ao_head": 122, "ao_face": 123, "ao_neck": 124,
    "ao_body": 125, "ao_waist": 126, "ao_leg": 127, "ao_arm": 128, "ao_hand": 129,
    "ao_kokan": 130,
    "cpo_sailor_a": 200, "cpo_sailor_b": 201, "cpo_sailor_c": 202,
    "cpo_jacket_a": 210, "cpo_jacket_b": 211, "cpo_jacket_c": 212,
    "mt_face_detail": 400, "mt_eyeshadow": 401, "mt_cheek": 402, "mt_lip": 403,
    "mt_lipline": 404, "mt_face_paint": 405, "mt_eyebrow": 406, "mt_eye_white": 407,
    "mt_eye": 408, "mt_eye_gradation": 409, "mt_eye_hi_up": 410, "mt_eye_hi_down": 411,
    "mt_eyeline_up": 412, "mt_eyeline_down": 413, "mt_nose": 414, "mt_mole": 415,
    "face_preset": 416,
    "mt_body_detail": 420, "mt_body_paint": 421, "mt_sunburn": 422, "mt_nip": 423,
    "mt_underhair": 424, "mt_pattern": 430, "mt_emblem": 431, "mt_ramp": 432,
    "mt_hairgloss": 433,
    "ex_bo_body": 500, "ex_bo_head": 501, "ex_bo_hair": 502, "ex_co_clothes": 503,
    "ex_co_shoes": 504,
}
#: ``Sideloader.AutoResolver.MigrationType``.
MIGRATE, MIGRATE_ALL, STRIP_ALL = 0, 1, 2
_MIGRATION_TYPES = {"Migrate": MIGRATE, "MigrateAll": MIGRATE_ALL, "StripAll": STRIP_ALL}
_NET_INT = re.compile(r"\s*[+-]?\d+\s*")
#: Sideloader's own plugin GUID — its settings file is ``<guid>.cfg``.
SIDELOADER_GUID = "com.bepis.bepinex.sideloader"


@dataclass(frozen=True)
class Migration:
    """One ``MigrationInfo`` a loaded manifest declares (``Manifest.LoadMigrationInfo``).

    ``MigrateAll`` / ``StripAll`` rows carry no category or ids; the
    constructor leaves both at their default, 0 — which is what the
    ``Migrate`` search in :meth:`SideloaderIndex.migrate` compares against.
    """
    kind: int
    guid_old: str
    guid_new: str
    cat_no: int = 0
    id_old: int = 0
    id_new: int = 0


def _net_enum(value: str, names: dict) -> int | None:
    """``Enum.Parse(type, value)``: a member name (case-sensitive) or an integer, surrounding blanks ignored."""
    s = str(value).strip()
    if _NET_INT.fullmatch(s):
        return int(s)
    return names.get(s)


def read_migrations(manifest_xml: bytes, guid: str) -> list:
    """The :class:`Migration` rows of one ``manifest.xml``, the way ``Manifest.LoadMigrationInfo`` reads them.

    Transcribed (``Sideloader.dll``)::

        foreach info in Root.Element("migrationInfo").Elements("info"):
            type    = migrationType attribute, blank -> Migrate (Enum.Parse)
            guidOld = its attribute                    (empty -> the line is skipped)
            guidNew = its attribute, blank -> this manifest's GUID
            MigrateAll / StripAll: (type, guidOld, guidNew) and done
            else: category (Enum.Parse on CategoryNo), idOld, idNew (Int32.TryParse;
                  a failed parse throws only for a Migrate row)

    Every failure throws inside a ``try`` whose ``catch`` logs "Could not load
    migration data for …, skipping line" — so a bad line costs that line only.
    Only the **first** ``migrationInfo`` element is read (``Element``, not
    ``Elements``).
    """
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(manifest_xml)
    except ET.ParseError:
        return []
    node = root.find("migrationInfo")
    out: list = []
    for info in (node.findall("info") if node is not None else []):
        a = info.attrib
        raw_type = a.get("migrationType")
        kind = (MIGRATE if raw_type is None or not raw_type.strip()
                else _net_enum(raw_type, _MIGRATION_TYPES))
        old, new = a.get("guidOld"), a.get("guidNew")
        if new is None or not new.strip():
            new = guid
        if kind is None or not old or (not new and kind == MIGRATE):
            continue
        if kind in (MIGRATE_ALL, STRIP_ALL):
            out.append(Migration(kind, old.strip(), (new or "").strip()))
            continue
        cat = a.get("category")
        cat_no = None if cat is None else _net_enum(cat, CATEGORY_ENUM)
        ids = []
        for key in ("idOld", "idNew"):
            v = a.get(key)
            if v is not None and _NET_INT.fullmatch(v):
                ids.append(int(v))
            elif kind == MIGRATE:
                break
            else:
                ids.append(0)
        if cat_no is None or len(ids) < 2:
            continue
        out.append(Migration(kind, old.strip(), new.strip(), cat_no, ids[0], ids[1]))
    return out


def migration_enabled() -> bool:
    """Sideloader's ``General.Migration enabled`` setting on this install (its default is on)."""
    from . import plugins
    return plugins.flag(SIDELOADER_GUID, "General.Migration enabled", True)


@dataclass
class ModRef:
    guid: str
    zipmod: str
    category: str
    id: int
    name: str
    bundle: str          # path *inside* the zipmod's abdata
    asset: str
    parent: str = ""
    cat_no: int = 0      # Illusion's own category number, matched against the
                         # card's accessory `type` so lookups can't cross-match
    # The rest of the row, by column name.  A mod's list row carries the same
    # columns the base game's does, and for some categories the item is not
    # just a prefab: a head names its own ``ShapeAnime`` curve asset, material
    # and face maps (§7.41).  Keeping only bundle+asset meant a modded head
    # had to borrow head 00's curves, which is a different face.
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"g": self.guid, "z": self.zipmod, "c": self.category,
                "i": self.id, "n": self.name, "b": self.bundle,
                "a": self.asset, "p": self.parent, "k": self.cat_no,
                "x": self.extra}

    def texture_ref(self, ab_key: str, tex_key: str) -> tuple[str, str] | None:
        """``(bundle, texture)`` this mod's list row points at.

        **A zipmod's row is a list row like any other.**  ``ChaControl`` does
        not branch on where an item came from: ``ChangeCustomClothes`` reads
        ``MainTex`` / ``ColorMaskTex`` off whatever ``GetListInfo`` returned,
        and Sideloader's whole job is to make its own rows answer that call.
        Skipping this for mods left every modded garment whose material binds
        no ``_MainTex`` — the normal case, since the base game's do not either
        — with no albedo at all, and the loader painted it a flat colour.

        The bundle names a path *inside the zipmod* unless the row points at a
        base-game one; :func:`load_mesh._ref_texture` tries the zipmod first
        and the installed bundles second, because both happen.
        """
        return L.texture_ref(self.extra, self.bundle, ab_key, tex_key)

    @staticmethod
    def from_dict(d: dict) -> "ModRef":
        return ModRef(d["g"], d["z"], d["c"], int(d["i"]), d["n"], d["b"],
                      d["a"], d.get("p", ""), int(d.get("k", 0)),
                      dict(d.get("x") or {}))

    def column(self, key: str) -> str:
        return str(self.extra.get(key, "") or "")


def _read_manifest(z: zipfile.ZipFile) -> tuple[str, str]:
    """``(guid, version)`` off an archive's ``manifest.xml``; empty strings when absent."""
    try:
        raw = z.read("manifest.xml")
    except KeyError:
        return "", ""
    g, v = GUID_RE.search(raw), VERSION_RE.search(raw)
    return (g.group(1).decode("utf-8", "replace").strip() if g else "",
            v.group(1).decode("utf-8", "replace").strip() if v else "")


def _read_manifest_guid(z: zipfile.ZipFile) -> str:
    return _read_manifest(z)[0]


def _has_guid(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return bool(_read_manifest(z)[0])
    except (zipfile.BadZipFile, OSError):
        return False


# ``ManifestVersionComparer`` (Sideloader.dll), transcribed.  ``Tokenize``
# splits on these (the array literal ``$field-2F24B699…``) and then into runs
# of digits and non-digits; ``CompareVersions`` first trims these off the front
# (``$field-509C9721…``).
_VERSION_SPLIT = re.compile(r"[. \-,_]")
_VERSION_TRIM = "vVrR "


def _version_tokens(text: str) -> list:
    """``ManifestVersionComparer.Tokenize``: runs of digits become ints (``Int32.TryParse``)."""
    out: list = []
    for part in _VERSION_SPLIT.split(text.strip()):
        if not part:
            out.append(0)
            continue
        runs = re.findall(r"\d+|\D+", part)
        for run in runs:
            try:
                value = int(run) if run.isdigit() and int(run) <= 2**31 - 1 else run
            except ValueError:
                value = run
            out.append(value)
    return out


def _culture_cmp(a: str, b: str) -> int:
    """``String.CompareTo`` (culture-sensitive), by the same key as :func:`guid_order_key`."""
    ka, kb = guid_order_key(a), guid_order_key(b)
    return (ka > kb) - (ka < kb)


def compare_versions(a: str | None, b: str | None) -> int:
    """``ManifestVersionComparer.CompareVersions``: <0, 0, >0 like ``IComparer``.

    Leading ``v``/``r`` and blanks are dropped, an empty version reads "0"; the
    token lists are compared element by element (a missing element is ``0``),
    ints numerically and strings culture-sensitively.  An int against a string
    throws inside ``IComparable.CompareTo`` and the catch compares the two
    texts instead: "0" sorts below anything else, otherwise ordinal (§7.479).
    """
    a = (a or "").strip().lstrip(_VERSION_TRIM) or "0"
    b = (b or "").strip().lstrip(_VERSION_TRIM) or "0"
    if a == b:
        return 0
    ta, tb = _version_tokens(a), _version_tokens(b)
    for i in range(max(len(ta), len(tb))):
        x = ta[i] if i < len(ta) else 0
        y = tb[i] if i < len(tb) else 0
        if type(x) is type(y):
            r = (x > y) - (x < y) if isinstance(x, int) else _culture_cmp(x, y)
        else:
            sx, sy = str(x), str(y)
            if sx == "0" and sy != "0":
                return -1
            if sy == "0" and sx != "0":
                return 1
            r = (sx > sy) - (sx < sy)
        if r:
            return r
    return 0


def loaded_archives(mods_dir: Path) -> list:
    """``[(guid, archive)]`` Sideloader loads from *mods_dir*, in its load order.

    One archive per GUID (:func:`loaded_archive`), the GUIDs sorted the way
    ``LoadModsFromDirectories`` sorts them (:func:`guid_order_key`).  Archives
    without a GUID are not loaded at all.  Read off the manifests only.
    """
    from collections import defaultdict
    copies: dict = defaultdict(list)
    for p in sorted(Path(mods_dir).rglob("*.zipmod")):
        try:
            with zipfile.ZipFile(p) as z:
                guid, version = _read_manifest(z)
        except (zipfile.BadZipFile, OSError):
            continue
        if guid:
            copies[guid].append((version, p, p.stat().st_mtime))
    return [(g, loaded_archive(copies[g])) for g in sorted(copies, key=guid_order_key)]


def bundle_path_of(entry: str) -> str:
    """The bundle path an archive entry registers (``ZipmodInfo.LoadAllLists``).

    Every entry ending in ``.unity3d`` (``OrdinalIgnoreCase``) is registered
    under its name with the first path segment removed — ``abdata/chara/x.unity3d``
    → ``chara/x.unity3d``.  The key is compared case-sensitively
    (``BundleManager.Bundles`` is a plain ``Dictionary<string, …>``).
    """
    return entry[entry.index("/") + 1:] if "/" in entry else entry


_REGISTRY: dict = {}


def bundle_registry(mods_dir: Path | None = None) -> dict:
    """``{bundle path: [(archive, entry), …]}`` — Sideloader's ``BundleManager.Bundles``.

    In load order, so the first element is the one ``AssetBundleLoadingHook``
    uses when the install has no such file and the first to be asked by
    ``RedirectHook`` for an asset (§7.480).  Cached per mods directory.
    """
    return _registries(mods_dir)[0]


def png_registry(mods_dir: Path | None = None) -> dict:
    """``{entry name: archive}`` — Sideloader's ``PngList``, the loose PNGs.

    ``ZipmodInfo.LoadAllLists`` collects every entry that starts with
    ``abdata/`` and ends in ``.png`` (both ``OrdinalIgnoreCase``) under its
    name **as spelled**; ``BuildPngFolderList`` adds them to one
    ``Dictionary<string, ZipmodInfo>`` (default comparer ⇒ case-sensitive)
    archive by archive in load order, and a name already there is a
    "Duplicate .png asset detected!" warning — so the first archive wins
    (§7.485).  Cached with :func:`bundle_registry`, one pass over the archives.
    """
    return _registries(mods_dir)[1]


def png_bytes(bundle_path: str, asset: str) -> bytes | None:
    """The loose PNG ``GetPng`` hands out for *asset* of *bundle_path*, or ``None``.

    Asked before any bundle for a texture (``RedirectHook``), whichever list
    row — stock or mod — makes the request (§7.485, §7.486).
    """
    asset = str(asset or "").strip()
    if not asset:
        return None
    key = png_request(bundle_path, asset)
    archive = png_registry().get(key)
    if archive is None:
        return None
    try:
        with zipfile.ZipFile(archive) as z:
            return z.read(key)
    except (zipfile.BadZipFile, OSError, KeyError) as exc:
        from .meshimport import note_load_failure
        note_load_failure(f"{Path(archive).name}!{key}", exc)
        return None


def png_request(bundle_path: str, asset: str) -> str:
    """The key ``RedirectHook`` asks ``GetPng`` for when asset *asset* of bundle *bundle_path* loads.

    ``"abdata/" + bundle.name.Replace(".unity3d", "", OrdinalIgnoreCase) + "/"
    + name + ".png"``, and ``AssetBundleLoadingHook`` sets ``bundle.name`` to
    the path as the game asked for it — so the row's ``MainAB`` and asset name,
    spelled as the row spells them (every occurrence of ``.unity3d`` goes,
    ``String.Replace`` is not a suffix strip).
    """
    p = re.sub(r"\.unity3d", "", str(bundle_path).replace("\\", "/").lstrip("/"),
               flags=re.IGNORECASE)
    return f"abdata/{p}/{asset}.png"


def _registries(mods_dir: Path | None) -> tuple:
    """``(bundles, pngs, png folders)`` for one mods directory, cached.

    No mods directory at all (a machine that never synced any) is three empty
    tables, not a scan of the working directory.
    """
    root = Path(mods_dir) if mods_dir is not None else _default_mods_dir()
    if root is None:
        return {}, {}, set()
    key = str(root)
    if key not in _REGISTRY:
        bundles: dict = {}
        pngs: dict = {}
        for _guid, archive in loaded_archives(root):
            try:
                with zipfile.ZipFile(archive) as z:
                    names = z.namelist()
            except (zipfile.BadZipFile, OSError):
                continue
            for n in names:
                low = n.lower()
                if low.endswith(".unity3d"):
                    bundles.setdefault(bundle_path_of(n), []).append((archive, n))
                elif low.startswith("abdata/") and low.endswith(".png"):
                    pngs.setdefault(n, archive)
        _REGISTRY[key] = (bundles, pngs, {n.rsplit("/", 1)[0] + "/" for n in pngs})
    return _REGISTRY[key]


def _default_mods_dir() -> Path | None:
    """The mods root the index is built from (the first search root that exists), or ``None``."""
    for root in mods_search_roots():
        if Path(root).is_dir():
            return Path(root)
    return None


def loaded_archive(candidates: list) -> "Path":
    """The one archive of a GUID Sideloader loads, from ``(version, path, mtime)`` triples.

    ``LoadModsFromDirectories``: when every copy has a version,
    ``OrderByDescending(Manifest.Version, ManifestVersionComparer)``
    ``.ThenByDescending(FileName.Length)``; otherwise
    ``OrderByDescending(LastWriteTime)``.  The first is loaded and the rest are
    reported as "Multiple versions detected" (§7.479).  ``OrderBy`` is stable,
    so ties keep the order the copies were found in.
    """
    from functools import cmp_to_key
    if len(candidates) == 1:
        return candidates[0][1]
    if all(v for v, _p, _t in candidates):
        def cmp(x, y):
            return -compare_versions(x[0], y[0]) or (len(str(y[1])) - len(str(x[1])))
        return sorted(candidates, key=cmp_to_key(cmp))[0][1]
    return sorted(candidates, key=lambda c: -c[2])[0][1]


def _parse_list_csv(data: bytes):
    """Sideloader list CSV: categoryNo, distribution, filePath, columns, rows."""
    text = data.decode("utf-8-sig", "replace")
    rows = [r for r in _csv.reader(io.StringIO(text)) if r and any(c.strip() for c in r)]
    if len(rows) < 4:
        return None, [], []
    try:
        cat_no = int(rows[0][0].strip())
    except (ValueError, IndexError):
        return None, [], []
    cols = [c.strip() for c in rows[3]]
    return cat_no, cols, rows[4:]


def scan_zipmod(path: Path) -> list[ModRef]:
    out: list[ModRef] = []
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        return out
    with z:
        guid = _read_manifest_guid(z)
        if not guid:
            return out
        for name in z.namelist():
            low = name.lower()
            if not (low.endswith(".csv") and "list/characustom" in low):
                continue
            try:
                cat_no, cols, rows = _parse_list_csv(z.read(name))
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.sideloader.scan_zipmod: z.read", exc)
                continue
            if not cols:
                continue
            cat = CATEGORY_NO.get(cat_no, f"cat{cat_no}")
            try:
                i_id, i_name = cols.index("ID"), cols.index("Name")
            except ValueError:
                continue
            # A texture-only list names its map in its own column (EyeTex,
            # EyelineUpTex, NipTex …) and has no MainData at all, so insisting
            # on that pair skipped every one of them.
            i_ab = next((cols.index(c) for c in L.AB_COLUMNS if c in cols), -1)
            i_data = next((cols.index(c) for c in L.DATA_COLUMNS if c in cols), -1)
            if i_ab < 0 or i_data < 0:
                continue
            i_parent = cols.index("Parent") if "Parent" in cols else -1
            for row in rows:
                if len(row) <= max(i_id, i_name, i_ab, i_data):
                    continue
                try:
                    rid = int(row[i_id].strip())
                except ValueError:
                    continue
                bundle = row[i_ab].strip()
                asset = row[i_data].strip()
                if not bundle or not asset or bundle in ("0", "-"):
                    continue
                out.append(ModRef(guid, str(path), cat, rid,
                                  row[i_name].strip(), bundle, asset,
                                  row[i_parent].strip() if i_parent >= 0 else "",
                                  cat_no,
                                  {c: v.strip() for c, v in zip(cols, row)}))
    return out


def guid_order_key(guid: str) -> tuple:
    """Sort key for the order Sideloader loads zipmods in — and so registers them.

    ``Sideloader.LoadModsFromDirectories`` groups the archives by GUID and walks
    the groups through ``OrderBy(kv => kv.Key)`` (``<LoadModsFromDirectories>
    b__53_6`` returns ``kv.Key``), keeping the newest version of each.  Every
    list row a mod registers becomes a ``ResolutionInfo`` in that order, and
    ``UniversalAutoResolver.TryGetResolutionInfo(slot, property, categoryNo)``
    answers with ``FirstOrDefault`` over them — so among mods that number an
    item alike, **the one whose GUID sorts first** is what a card with no UAR
    row for that field gets (§7.478).

    ``OrderBy`` on strings uses the default comparer, which in .NET is the
    culture-sensitive one: letters compare case-insensitively first (lowercase
    ahead of uppercase only as a tie-break), punctuation sorts ahead of digits
    and digits ahead of letters, and ``-`` / ``'`` are ignored ("word sort").
    That is what this key reproduces; ordinal comparison would put every
    upper-case GUID ahead of every lower-case one.
    """
    prim = []
    for c in str(guid):
        if c in "-'":
            continue
        if c.isalpha():
            prim.append((2, c.casefold()))
        elif c.isdigit():
            prim.append((1, c))
        else:
            prim.append((0, c))
    case = tuple(0 if c.islower() else 1 for c in str(guid) if c.isalpha())
    return (tuple(prim), case, str(guid))


class SideloaderIndex:
    def __init__(self):
        self._by_cat: dict | None = None     # category -> id -> ref (lazy)
        self._by_cat_no: dict | None = None  # (cat_no, id) -> ref (lazy)
        # guid -> category -> id -> ModRef
        self.by_guid: dict[str, dict[str, dict[int, ModRef]]] = {}
        self.mods = 0
        self.items = 0
        # every loaded manifest's migration rows, in load order, and the GUIDs
        # ``Sideloader.GetManifest`` answers for (§7.488)
        self.migrations: list[Migration] = []
        self.loaded_guids: set[str] = set()
        self._mig_by_guid: dict | None = None
        self._mig_by_id: dict | None = None

    # ------------------------------------------------------------------
    def build(self, mods_dir: Path, progress=None) -> "SideloaderIndex":
        """Index every zipmod Sideloader would load.

        Only the manifest and the list CSVs are read — the asset bundles inside
        are never decompressed here, so the scan stays flat in memory however
        many gigabytes of mods are installed.

        **One archive per GUID** (:func:`loaded_archive`): Sideloader loads the
        newest copy and skips the rest.  This used to index every copy and let
        whichever came later in path order overwrite item by item, so an item
        only an old version had stayed in the index — 11 local GUIDs have two
        archives, and ``WW_WeaponAccs`` mixed v2.4 with v1.0 (§7.479).
        """
        import gc
        self._by_cat = self._by_cat_no = None
        self._mig_by_guid = self._mig_by_id = None
        self.by_guid = {}
        self.mods = 0
        self.items = 0
        found = sorted(Path(mods_dir).rglob("*.zipmod"))
        loaded = loaded_archives(mods_dir)
        files = sorted(p for _g, p in loaded)
        self.loaded_guids = {g.strip() for g, _p in loaded}
        self.migrations = []
        for g, p in loaded:                      # load order, as the lookup keeps it
            try:
                with zipfile.ZipFile(p) as z:
                    raw = z.read("manifest.xml")
            except (zipfile.BadZipFile, OSError, KeyError) as exc:
                swallowed.note("origin.sideloader.build: manifest.xml", exc)
                continue
            if b"migrationInfo" in raw:
                self.migrations.extend(read_migrations(raw, g))
        chosen = set(files)
        self.skipped_copies = sorted(
            str(p) for p in found
            if p not in chosen and _has_guid(p))
        # What the index was built *from*, so a later run can tell whether the
        # folder has moved on.  ``self.mods`` cannot serve: it counts the
        # zipmods that yielded rows, so a mod with no list entry is invisible
        # to it and dropping one in would look like nothing changed.
        self.src_root = str(mods_dir)
        self.src_files, self.src_mtime = _mods_fingerprint(mods_dir)
        for i, p in enumerate(files):
            refs = scan_zipmod(p)
            if refs:
                self.mods += 1
                for r in refs:
                    self.by_guid.setdefault(r.guid, {}).setdefault(
                        r.category, {})[r.id] = r
                    self.items += 1
            if i % 400 == 0:
                gc.collect()
                if progress:
                    progress(i, len(files))
        gc.collect()
        return self

    def resolve(self, guid: str, category: str, slot: int) -> ModRef | None:
        return self.by_guid.get(guid, {}).get(category, {}).get(int(slot))

    def find(self, guid: str, slot: int, cat_no: int | None = None) -> ModRef | None:
        """Resolve (guid, slot), optionally pinned to Illusion's category number.

        Without the category a slot can cross-match a different item in the same
        mod — that is how a hairstyle ended up as a pair of animal ears — so
        callers that know the card's ``type`` should always pass it.  A pack that
        ships a whole outfit, or hair *and* clothes, numbers them from one run,
        and the unpinned walk answers with whichever category comes first in
        dictionary order.  Every caller in ``figure.resolve`` passes one now;
        three of them did not, and it cost a military jacket worn on a boy's head
        and 17.5% of cards wearing at least one wrong garment (§7.66 五).
        """
        cats = self.by_guid.get(guid) or {}
        slot = int(slot)
        if cat_no is not None:
            for items in cats.values():
                r = items.get(slot)
                if r is not None and r.cat_no == int(cat_no):
                    return r
            return None
        for items in cats.values():
            r = items.get(slot)
            if r is not None:
                return r
        return None

    def _in_load_order(self):
        """``by_guid`` items in the order Sideloader registers them (:func:`guid_order_key`)."""
        return sorted(self.by_guid.items(), key=lambda kv: guid_order_key(kv[0]))

    def find_any(self, category: str, slot: int) -> "ModRef | None":
        """``CompatibilityResolve`` for a list category: the first claimant of (category, slot).

        A card field with no UAR row — older saves, or a property the resolver
        of the day did not track — goes through
        ``UniversalAutoResolver.CompatibilityResolve``: the base list first
        (the caller asks it), then ``TryGetResolutionInfo(slot, property,
        categoryNo)``, which is ``FirstOrDefault`` over the mods in load order.
        So when several mods number an item alike the game does **not** give up;
        it takes the one whose GUID sorts first (§7.478).  This used to answer
        only a single claimant and report the rest missing, on the belief that
        the game draws nothing — the IL says otherwise.

        Indexed rather than scanned: there are ~4,000 guids, and a miss is the
        common case (every id that turns out to be base-game or 「なし」 lands
        here), so a linear walk cost more than the bundle read it precedes.
        """
        if self._by_cat is None:
            table: dict = {}
            for _guid, cats in self._in_load_order():
                for cat, items in cats.items():
                    bucket = table.setdefault(cat, {})
                    for rid, ref in items.items():
                        bucket.setdefault(rid, ref)
            self._by_cat = table
        try:
            hit = self._by_cat.get(category, {}).get(int(slot))
        except (TypeError, ValueError):
            return None
        return hit if hit is not None else self._migrated_claimant(
            CATEGORY_NO_OF.get(category), slot)

    def find_by_cat_slot(self, cat_no: int, slot: int) -> "ModRef | None":
        """:meth:`find_any` keyed by ``ChaListDefine.CategoryNo`` — the accessory case.

        An accessory's category is its part's ``type``, so the pair
        ``(CategoryNo, id)`` is what ``CompatibilityResolve`` matches on.
        Zipmod ids are not globally unique (1,780 of 19,203 pairs on this
        install have two claimants or more, 258 at ``ao_head`` alone), and the
        answer the game gives is the claimant whose GUID sorts first — not
        "whichever the dictionary met first", which is what put a pirate hat and
        a detective hat on a nurse in §7.66, and not "nothing", which is what
        §7.66 replaced it with and the IL does not do (§7.478).
        """
        if self._by_cat_no is None:
            table: dict = {}
            for _guid, cats in self._in_load_order():
                for items in cats.values():
                    for rid, ref in items.items():
                        table.setdefault((int(ref.cat_no), int(rid)), ref)
            self._by_cat_no = table
        try:
            hit = self._by_cat_no.get((int(cat_no), int(slot)))
        except (TypeError, ValueError):
            return None
        return hit if hit is not None else self._migrated_claimant(cat_no, slot)

    # ------------------------------------------------------------------ migration
    def _migration_tables(self) -> tuple:
        """``GetMigrationInfo``'s two lookups: by old GUID and by old id, each in load order."""
        if self._mig_by_guid is None:
            by_guid: dict = {}
            by_id: dict = {}
            for m in self.migrations:
                by_guid.setdefault(m.guid_old, []).append(m)
                by_id.setdefault(m.id_old, []).append(m)
            self._mig_by_guid, self._mig_by_id = by_guid, by_id
        return self._mig_by_guid, self._mig_by_id

    def migrate(self, guid: str, slot: int, cat_no: int) -> tuple:
        """``UniversalAutoResolver.MigrateData`` for one UAR row: ``(guid, slot)`` after migration.

        Transcribed::

            if GUID is blank: return
            rows = GetMigrationInfo(GUID)
            if rows.Any(StripAll): GUID = ""; return
            foreach r in rows.Where(IDOld == Slot && Category == CategoryNo):
                if GetManifest(r.GUIDNew) != null: GUID, Slot = r.GUIDNew, r.IDNew; return
            foreach r in rows.Where(MigrationType == MigrateAll):
                if GetManifest(r.GUIDNew) != null: GUID = r.GUIDNew; return

        It runs before the row is looked up, whether or not the old GUID is
        installed, so a mod that moved its items into another GUID takes every
        card naming the old one with it.  A blank answer means stripped:
        ``ResolveStructure`` then treats the field as if the card had no row
        (``CompatibilityResolve``).
        """
        if not str(guid or "").strip():
            return guid, slot
        rows = self._migration_tables()[0].get(str(guid).strip(), [])
        if any(m.kind == STRIP_ALL for m in rows):
            return "", slot
        for m in rows:
            if m.id_old == int(slot) and m.cat_no == int(cat_no) and m.guid_new in self.loaded_guids:
                return m.guid_new, m.id_new
        for m in rows:
            if m.kind == MIGRATE_ALL and m.guid_new in self.loaded_guids:
                return m.guid_new, slot
        return guid, slot

    def _migrated_claimant(self, cat_no, slot) -> "ModRef | None":
        """``CompatibilityResolve``'s last step: the first migration row of this old id and category.

        ``GetMigrationInfo(slot).FirstOrDefault(x => x.Category == category)``
        and then ``TryGetResolutionInfo(IDNew, property, category, GUIDNew)`` —
        only that first row is tried, whether or not its new GUID is loaded.
        """
        if cat_no is None:
            return None
        try:
            slot, cat_no = int(slot), int(cat_no)
        except (TypeError, ValueError):
            return None
        first = next((m for m in self._migration_tables()[1].get(slot, [])
                      if m.cat_no == cat_no), None)
        return None if first is None else self.find(first.guid_new, first.id_new, cat_no)

    def stats(self) -> dict:
        return {"mods": self.mods, "items": self.items,
                "guids": len(self.by_guid)}

    # ------------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = Path(path or CACHE)
        flat = [r.to_dict() for cats in self.by_guid.values()
                for items in cats.values() for r in items.values()]
        return write_json_atomic(path, {
            "version": INDEX_VERSION,
            "mods": self.mods,
            "src_root": getattr(self, "src_root", ""),
            "src_files": getattr(self, "src_files", 0),
            "src_mtime": getattr(self, "src_mtime", 0.0),
            "loaded_guids": sorted(self.loaded_guids),
            "migrations": [[m.kind, m.guid_old, m.guid_new, m.cat_no, m.id_old, m.id_new]
                           for m in self.migrations],
            "refs": flat})

    def load(self, path: Path | None = None) -> bool:
        path = Path(path or CACHE)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if int(data.get("version", 1)) < INDEX_VERSION:
            # An older index is wrong in ways nothing downstream can see: v1 has
            # no per-row columns (a modded head silently falls back to head 00's
            # curves), v2 files every modded accessory under a clothing
            # category, v3 mixes every copy of a GUID (§7.479).  Say so rather
            # than serving it -- §7.33's lesson about a version mismatch that
            # was quietly swallowed.  ``index()`` rebuilds on this answer.
            print(f"[sideloader] 索引格式 {int(data.get('version', 1))} 較舊"
                  f"（本版 {INDEX_VERSION}），重建中…")
            return False
        self.by_guid = {}
        self._by_cat = self._by_cat_no = None
        self.items = 0
        missing = 0
        for d in data.get("refs", []):
            r = ModRef.from_dict(d)
            # rewrite stale absolute paths onto the live install
            fixed = resolve_zipmod_path(r.zipmod)
            if fixed is not None:
                r.zipmod = str(fixed)
            else:
                missing += 1
            self.by_guid.setdefault(r.guid, {}).setdefault(
                r.category, {})[r.id] = r
            self.items += 1
        self.mods = int(data.get("mods", 0))
        self.loaded_guids = {str(g) for g in data.get("loaded_guids", [])}
        self.migrations = [Migration(int(k), str(o), str(n), int(c), int(a), int(b))
                           for k, o, n, c, a, b in data.get("migrations", [])]
        self._mig_by_guid = self._mig_by_id = None
        self.src_root = str(data.get("src_root", "") or "")
        self.src_files = int(data.get("src_files", 0) or 0)
        self.src_mtime = float(data.get("src_mtime", 0.0) or 0.0)
        # If most zip paths are dead, force a rebuild from the install
        if self.items and missing / max(self.items, 1) > 0.5:
            return False
        return True

    def is_stale(self) -> str:
        """Why the mods folder no longer matches this index, or ``""``.

        Dropping a ``.zipmod`` into ``assets_cache/mods`` used to change
        nothing: the loader served whatever JSON was on disk and the new item
        simply did not exist, with no error anywhere — a card asking for it fell
        through to the base list and wore something else (§一之一).  The
        fingerprint is the file count plus the newest mtime under the root the
        index was built from, which catches both "a mod appeared" and "a mod was
        replaced in place".

        An index with no fingerprint recorded is **not** called stale: it was
        written before this existed, and rebuilding every such index on sight
        would cost every user a full rescan for nothing.
        """
        if not self.src_root or not self.src_files:
            return ""
        root = Path(self.src_root)
        if not root.is_dir():
            return ""            # the root moved; ``repath`` handles that
        files, mtime = _mods_fingerprint(root)
        if files != self.src_files:
            return f"zipmod 數量 {self.src_files} → {files}"
        if mtime > self.src_mtime + 1.0:
            return "有 zipmod 被換過（mtime 較新）"
        return ""

    def repath(self) -> int:
        """Rewrite every ModRef.zipmod onto existing files. Returns fixed count."""
        n = 0
        for cats in self.by_guid.values():
            for items in cats.values():
                for r in items.values():
                    fixed = resolve_zipmod_path(r.zipmod)
                    if fixed is not None and str(fixed) != r.zipmod:
                        r.zipmod = str(fixed)
                        n += 1
                    elif fixed is not None:
                        r.zipmod = str(fixed)
        return n


_index: SideloaderIndex | None = None


def index(mods_dir: Path | None = None, refresh: bool = False,
          progress=None) -> SideloaderIndex:
    global _index
    if _index is not None and not refresh:
        return _index
    idx = SideloaderIndex()
    if not refresh and idx.load():
        why = idx.is_stale()
        if why:
            # Rebuilding is blocking and has no progress UI, so say what is
            # happening rather than appearing to hang.  Silently serving the
            # old index is the alternative and it is worse: the new mod is not
            # missing, it does not exist, and the card wearing it dresses in
            # something else without a word.
            print(f"[sideloader] mods 資料夾變了（{why}），重建索引…")
        else:
            idx.repath()
            _index = idx
            return idx
    # Prefer live Origin install mods/ (thousands of zipmods on BetterRepack)
    roots = mods_search_roots()
    if mods_dir is not None:
        roots = [Path(mods_dir)] + [r for r in roots if r != Path(mods_dir)]
    for root in roots:
        if root.is_dir():
            idx.build(root, progress)
            if idx.items:
                idx.save()
                break
    _index = idx
    return idx


def refresh_index(progress=None) -> SideloaderIndex:
    """Force re-scan of install mods and rewrite the JSON cache."""
    global _index
    _index = None
    _zip_resolve_cache.clear()
    return index(refresh=True, progress=progress)


# Sideloader accepts a plain image where the game expects an asset, so these
# are the extensions a list row's ``MainData`` may actually carry.
def load_mod_file(ref: ModRef) -> bytes | None:
    """The loose PNG Sideloader hands out for a row's ``(MainAB, asset)``, if any mod ships one.

    A texture-only mod is **not required to build a Unity bundle**: the map can
    live at ``abdata/<MainAB>/<MainData>.png``, and ``RedirectHook`` asks
    ``GetPng`` for that key before any bundle is asked for a texture.  The
    answer is Sideloader's ``PngList`` (:func:`png_registry`): **any** loaded
    archive, the first in load order, the key matched exactly — not only the
    row's own archive, and not case-blind or without the ``abdata/`` prefix,
    as this used to look (§7.485).
    """
    return png_bytes(ref.bundle, ref.asset)


# --------------------------------------------------------------------------
def bundle_sources(bundle_path: str) -> list:
    """Where a request for bundle *bundle_path* is answered from, in the order the game asks.

    ``RedirectHook`` asks every archive that registered the path, in load
    order (:func:`bundle_registry`), for the asset before the bundle itself is
    used — and the bundle itself is the install's file when there is one
    (``AssetBundleLoadingHook``'s ``File.Exists``).  So: the registering
    archives, then the install.  ``[("mod", archive, entry) | ("install", path, "")]``;
    the path is matched as the game spells it (case-sensitive, §7.480).
    """
    from .install import configured
    p = str(bundle_path).replace("\\", "/").lstrip("/")
    out = [("mod", a, e) for a, e in bundle_registry().get(p, [])]
    inst = configured()
    if inst.valid and inst.bundle(p).exists():
        out.append(("install", inst.bundle(p), ""))
    return out


def _open_source(kind: str, where, entry: str):
    """The UnityPy env of one :func:`bundle_sources` entry (cached), or ``None``."""
    from .meshimport import (BundleTooLarge, _load_bundle, load_bytes_bundle,
                             note_load_failure)
    try:
        if kind == "install":
            return _load_bundle(Path(where))
        with zipfile.ZipFile(where) as z:
            data = z.read(entry)
        return load_bytes_bundle(f"{where}!{entry}", data)
    except BundleTooLarge as exc:
        print(f"[sideloader] {exc}")
        note_load_failure(f"{Path(str(where)).name} {entry}", exc)
    except (zipfile.BadZipFile, OSError, KeyError) as exc:
        note_load_failure(f"{Path(str(where)).name} {entry}", exc)
    except Exception as exc:                         # noqa: BLE001
        # Counted, not just swallowed: a ``MemoryError`` here is why a modded
        # cheek sheet goes missing without anything saying so (§7.363).
        note_load_failure(f"{Path(str(where)).name} {entry}", exc)
    return None


def _contains(env, asset: str) -> bool:
    """``AssetBundle.Contains(name)`` for a bare name: a container key's file name, case-blind."""
    want = str(asset).lower()
    for path in env.container.keys():
        base = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        if (base.rsplit(".", 1)[0] if "." in base else base).lower() == want:
            return True
    return False


def load_mod_bundle(ref: ModRef, bundle: str | None = None, asset: str | None = None):
    """The bundle a zipmod row's asset is loaded from — Sideloader's rule, not "the row's own archive".

    Among :func:`bundle_sources` for the row's bundle path (*bundle*, default
    ``ref.bundle``), the first whose bundle contains *asset* (default
    ``ref.asset``) — the order ``RedirectHook`` asks in.  That is usually the
    row's own archive, but not always: 299 local rows name a bundle the install
    has (a mod re-listing stock assets) and 108 a path another mod registered
    first, and all of those came back "no bundle" when only the row's archive
    was searched (§7.481).  This used to fall back to a case-blind path and to
    guessing an archive entry by file-name ending or substring; the game does
    neither (``Bundles`` is case-sensitive), and every row those guesses rescued
    is one the game cannot load.

    Envs come from :mod:`meshimport`'s cache (``load_bytes_bundle`` /
    ``_load_bundle``), so a second asset out of the same bundle costs nothing.
    """
    zpath = resolve_zipmod_path(ref.zipmod)
    if zpath is not None:
        ref.zipmod = str(zpath)          # keep the resolved path for later
    return open_asset_bundle(bundle if bundle is not None else ref.bundle,
                             asset if asset is not None else ref.asset)


def open_asset_bundle(bundle_path: str, asset: str):
    """The loaded bundle a request for *asset* out of *bundle_path* is answered from, or ``None``.

    :func:`bundle_sources` in order, the first whose bundle contains *asset*
    (``RedirectHook`` → ``TryGetObjectFromName`` → the bundle itself).  The
    rule does not depend on who asks: a stock row's request is answered the
    same way, so a mod that registered a stock path overrides the stock asset
    of that name (§7.486).  An empty *asset* takes the first source.
    """
    for kind, where, entry in bundle_sources(bundle_path):
        env = _open_source(kind, where, entry)
        if env is not None and (not asset or _contains(env, asset)):
            return env
    return None


def overrides_at(bundle_path: str) -> bool:
    """Can anything but the bundle itself answer a request out of *bundle_path*?

    True when a loaded archive registered that path, or ships loose PNGs in
    its folder (``abdata/<path without .unity3d>/``) — the two things
    ``RedirectHook`` asks before the bundle.  False for almost every stock
    bundle, which then keeps its one install read (§7.486).
    """
    p = str(bundle_path).replace("\\", "/").lstrip("/")
    bundles, _pngs, folders = _registries(None)
    if bundles.get(p):
        return True
    return png_request(p, "").rsplit("/", 1)[0] + "/" in folders


# --------------------------------------------------------------------------
def card_resolutions(card) -> dict:
    """Decode a card's UAR table into {(property, local_slot): (guid, slot)}.

    Each row is migrated first when the install has migration on
    (:meth:`SideloaderIndex.migrate` — ``ResolveStructure`` calls
    ``MigrateData`` before it looks anything up), so the answer names the GUID
    and slot the game resolves.  A row migration strips is left out: the game
    treats that field as unnamed (§7.488).  The keys stay the card's own
    numbers — the field still holds the id it was saved with.
    """
    from ..chara.cardimport import _unpack
    ex = card.sections.get("KKEx") or {}
    uar = ex.get("com.bepis.sideloader.universalautoresolver")
    out: dict = {}
    if not isinstance(uar, list):
        return out
    migrate = index().migrate if migration_enabled() else None
    for entry in uar:
        info = entry.get("info") if isinstance(entry, dict) else None
        if not isinstance(info, list):
            continue
        for blob in info:
            if not isinstance(blob, (bytes, bytearray)):
                continue
            try:
                d = _unpack(bytes(blob))
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.sideloader.card_resolutions: _unpack", exc)
                continue
            if not isinstance(d, dict):
                continue
            prop = d.get("Property")
            local = d.get("LocalSlot")
            guid = d.get("ModID")
            slot = d.get("Slot")
            if prop is None or local is None or guid is None or slot is None:
                continue
            got = (str(guid), int(slot))
            if migrate is not None:
                try:
                    got = migrate(got[0], got[1], int(d.get("CategoryNo") or 0))
                except (TypeError, ValueError) as exc:
                    swallowed.note("origin.sideloader.card_resolutions: CategoryNo", exc)
                if not str(got[0]).strip():
                    continue                         # stripped: as if unnamed
            # a card may store either the mod's own slot or the runtime local
            # slot depending on how it was saved, so index both
            out[(str(prop), int(local))] = got
            out[(str(prop), int(slot))] = got
            out[(str(prop), None)] = got
    return out
