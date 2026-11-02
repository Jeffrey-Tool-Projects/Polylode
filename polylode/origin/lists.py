"""Read Illusion's ChaListData tables: asset ID -> bundle + asset name.

Each ``abdata/list/characustom/*.unity3d`` holds TextAssets that are MessagePack
maps of the form::

    {mark: "【ChaListData】", categoryNo: 101, lstKey: [...column names...],
     dictList: {row_index: [values...]}}

The table name says what it is (``bo_hair_b`` = back hair, ``co_top`` = tops),
and the columns name what each value means, so nothing here is guesswork.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..chara.cardimport import _unpack
from ..core.paths import CONFIG_DIR, write_json_atomic
from .install import Install
from ..core import swallowed

CACHE = CONFIG_DIR / "origin_lists.json"
# 2: rows keep their remaining columns (MainTex/ColorMask refs, NotBra, masks)
# 3: the face/eye texture lists (mt_eye, mt_eye_gradation, mt_eyeline_*, …)
# 4: rows that name no asset are remembered as "none" rather than dropped
# 5: the *_layout tables (they name no asset at all, only numbers)
# 6: mt_ramp / mt_hairgloss — the two globals the character shaders sample
# 7: bo_head — the head is a list item, not a constant (§7.41)
# 8: mt_emblem — the chest badge is a list item too (ChangeCustomEmblem)
# 9: which categories' tables carry a MainAB column (§7.493)
CACHE_VERSION = 9

# table prefix -> what it customises
CATEGORY = {
    # The head is a list item like everything else -- ``ChaControl.ChangeHead``
    # is ``ChangeHeadAsync(fileFace.headId)``, which does ``GetListInfo`` on
    # CategoryNo 100.  The row names five things that all change together:
    # the prefab, the **ShapeAnime** curve asset the face sliders play, the
    # face material, its albedo and its colour mask.  This was missing for a
    # long time and every card got ``p_cf_head_00`` with head 00's curves
    # (§7.41); locally that is a quarter of them.
    "bo_head": "head",
    "bo_hair_b": "hair_back", "bo_hair_f": "hair_front",
    "bo_hair_s": "hair_side", "bo_hair_o": "hair_option",
    "co_top": "top", "co_bot": "bottom", "co_bra": "bra",
    "co_shorts": "panty", "co_gloves": "gloves", "co_panst": "legwear",
    "co_socks": "socks", "co_shoes": "shoes", "co_swim": "swim",
    "co_swim_top": "swim_top", "co_swim_bot": "swim_bottom",
    # Uniform top sub-parts (ChaControl.ChangeClothesTopParts).  The co_top
    # entry is p_dummy; real meshes are looked up by subPartsId in these
    # three categories — same CategoryNo values as Illusion's enum.
    "cpo_sailor_a": "cpo_sailor_a", "cpo_sailor_b": "cpo_sailor_b",
    "cpo_sailor_c": "cpo_sailor_c",
    "cpo_jacket_a": "cpo_jacket_a", "cpo_jacket_b": "cpo_jacket_b",
    "cpo_jacket_c": "cpo_jacket_c",
    # Texture-only lists: the row *is* the map.  CreateBodyTexture /
    # CreateFaceTexture composite these over the white base albedo.
    "mt_pattern": "pattern",
    # The two maps every character shader reads as globals.  ``mt_ramp`` is
    # the diffuse toon ramp (``Character.UpdateGlobalShader`` -> _RampG, id
    # from EtceteraSystem.rampId) and ``mt_hairgloss`` is the strand highlight
    # sheet (``ChaControl.LoadHairGlossMask`` -> the card's ``hair.glossId``).
    "mt_ramp": "ramp", "mt_hairgloss": "hairgloss",
    # The chest badge.  It is *not* part of the garment's own maps:
    # ``ChangeCustomEmblem`` looks the card's ``emblemeId`` up in CategoryNo
    # 431 and assigns the result to ``ChaClothesComponent.rendEmblem01/02``'s
    # ``_MainTex``, which is why the emblem quad came out wearing the blouse.
    "mt_emblem": "emblem",
    "mt_body_detail": "body_detail", "mt_face_detail": "face_detail",
    "mt_nip": "nip", "mt_sunburn": "sunburn", "mt_underhair": "underhair",
    "mt_body_paint": "body_paint", "mt_face_paint": "face_paint",
    "mt_cheek": "cheek", "mt_lipline": "lipline", "mt_lip": "lip",
    "mt_mole": "mole", "mt_eyeshadow": "eyeshadow",
    "mt_eye_white": "eye_white", "mt_eye_gradation": "eye_gradation",
    # Every map ChangeSettingEye* / ChangeSettingEyeline* / ChangeSettingNose
    # pulls through GetListInfo.  Without these the loader had to guess the
    # bundle and asset name from a hard-coded table, so a modded iris or
    # eyeline silently fell back to the stock one.
    "mt_eye": "eye", "mt_eye_hi_up": "eye_hi_up",
    "mt_eye_hi_down": "eye_hi_down", "mt_eyebrow": "eyebrow",
    "mt_eyeline_up": "eyeline_up", "mt_eyeline_down": "eyeline_down",
    "mt_nose": "nose",
    # Placement presets.  These name no asset — the row *is* a set of numbers
    # (``CenterX``/``MoveX``/…), which ``CreateBodyTexture`` turns into the
    # Lerp range for the card's own paintLayout slider.  They therefore go in
    # ``AssetIndex.layouts`` rather than the ref table.
    "bodypaint_layout": "bodypaint_layout",
    "facepaint_layout": "facepaint_layout",
    "mole_layout": "mole_layout",
    "ao_hair": "acc_hair", "ao_head": "acc_head", "ao_face": "acc_face",
    "ao_kami": "acc_hair", "ao_neck": "acc_neck", "ao_body": "acc_body",
    "ao_arm": "acc_arm", "ao_hand": "acc_hand", "ao_leg": "acc_leg",
    "ao_waist": "acc_waist", "ao_back": "acc_back",
}


# Which columns name the bundle and the asset inside it.  Order matters:
# ``MainData`` first because a garment row also carries ``MainTex`` and the
# prefab is what a clothing lookup needs; ``ColorMaskTex`` last because only
# ``mt_eye_gradation`` has nothing else.  Shared with the zipmod scanner so a
# modded iris or eyeline is indexed the same way the base list is.
# The 「なし」 rows of the texture lists, recorded so a machine with no
# Illusion install still knows them.
#
# A row that exists and names no asset is Illusion's "the player picked none"
# — the same trick ``p_dummy`` plays for garments — and the live index puts
# those in :attr:`AssetIndex.empty`.  That works while the lists are there to
# read, which is exactly what F0 does not have: the generated sheets in
# :mod:`polylode.mesh.facetex` / :mod:`polylode.mesh.parttex` are drawn on
# machines with nothing converted, and ``_list_texture`` asks the provider
# **before** it asks the list, so a generator that answers is the last word.
# Measured with them missing: our own path came back with 11,818 texels of
# blush, 14,696 of eyeshadow, 18,970 of lip, 118 of lip line and 58 of mole on
# a card that asked for none of it, where the Illusion path returns ``None``
# for every one — 194 of 200 sampled female cards get at least one such
# drawing, and only 6 name a real row on all six layers.
#
# **The id is not always 0**, which is why this is measured per category and
# not a rule: ``hairgloss`` calls 5 「なし」 and ``nose`` calls 4, while the
# features everyone has (``eye``, ``eyeline_up``, ``eyebrow``, ``nip`` …) have
# no such row at all and their id 0 is タイプ１ — rejecting *that* drops the
# game's own default off every untouched card (``CLAUDE.md`` 七).
#
# ``tools/face_tex_perid.py --none`` re-derives this from the installed lists
# and exits non-zero on disagreement, so the copy has a check behind it rather
# than a comment.  The ids are list facts (``著作權.md`` 四), not a drawing.
NONE_IDS: dict = {
    "body_paint": (0,), "cheek": (0,), "emblem": (0,), "eyeshadow": (0,),
    "face_detail": (0,), "face_paint": (0,), "hairgloss": (5,), "lip": (0,),
    "lipline": (0,), "mole": (0,), "nose": (4,), "pattern": (0,),
    "sunburn": (0,), "underhair": (0,),
}

#: 「なし」 written the **other** way: the row names an asset, and the asset is
#: empty.
#:
#: ``eyeline_down`` 21 is ``cw_t_eyeline_down_none`` — a **16×16 texture with
#: every alpha byte zero**.  It is not in :attr:`AssetIndex.empty`, because the
#: list row does name a bundle and an asset; the "draw nothing" lives in the
#: asset's content instead.  Swept over every baked row in the library it is
#: the **only** one (1 of 3,518), which is why a rule that only knew the list
#: form missed it — and it is the most-named row we get wrong: **32 of 200
#: sampled female cards (16%)** choose it, i.e. they turned the lower lash line
#: off, and the generator drew one for every single one of them.
#:
#: Recorded rather than derived because deriving it means decoding the asset,
#: which the machine F0 exists for cannot do.  ``face_tex_perid.py --none``
#: re-derives it from the baked PNGs (alpha max 0) and fails on disagreement,
#: the same check ``NONE_IDS`` has.
#:
#: Kept apart from ``NONE_IDS`` because the two have different answers
#: downstream: a row that names no asset resolves to nothing, so declining is
#: right; a row that names an **empty** asset has content, and the honest thing
#: for a generator standing in for it is to produce that content — an empty
#: sheet.  Returning ``None`` instead would leave the renderer on the prefab's
#: own ``_MainTex`` (white, for a cut-out lash quad) on any machine with no
#: conversion to fall back to.
EMPTY_ASSET_IDS: dict = {"eyeline_down": (21,)}


def draws_nothing(category: str, ident) -> bool:
    """Does this id mean "draw nothing" — in **either** of Illusion's forms?

    The row names no asset (:data:`NONE_IDS` / :attr:`AssetIndex.empty`), or it
    names one that is empty (:data:`EMPTY_ASSET_IDS`).  Callers want the
    product question, not the encoding, and asking only the first form is how
    the lower lash line's なし row went on being drawn (§7.197).
    """
    try:
        ident = int(ident)
    except (TypeError, ValueError):
        return False
    return (index().is_none(str(category), ident)
            or ident in EMPTY_ASSET_IDS.get(str(category), ()))


AB_COLUMNS = ("MainAB", "MainTexAB", "ColorMaskAB")
DATA_COLUMNS = (
    "MainData", "MainTex", "NipTex", "PaintTex", "SunburnTex",
    "UnderhairTex", "CheekTex", "LipTex", "LiplineTex", "MoleTex",
    "EyeshadowTex", "EyeWhiteTex", "EyeTex", "EyeHiUpTex", "EyeHiDownTex",
    "EyebrowTex", "EyelineUpTex", "EyelineDownTex", "NoseTex", "LineMask",
    "NormallMapDetail", "ColorMaskTex",
)


@dataclass
class AssetRef:
    id: int
    name: str
    bundle: str
    asset: str
    table: str
    category: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "bundle": self.bundle,
                "asset": self.asset, "table": self.table,
                "category": self.category, "extra": self.extra}

    @staticmethod
    def from_dict(d: dict) -> "AssetRef":
        return AssetRef(int(d["id"]), d["name"], d["bundle"], d["asset"],
                        d["table"], d["category"], dict(d.get("extra") or {}))

    # -- ChaListControl fields the loader acts on ------------------------
    def flag(self, key: str) -> int:
        try:
            return int(str(self.extra.get(key, 0)))
        except (TypeError, ValueError):
            return 0

    def texture_ref(self, ab_key: str, tex_key: str) -> tuple[str, str] | None:
        """``(bundle, texture)`` a list row points at, if it names one."""
        return texture_ref(self.extra, self.bundle, ab_key, tex_key)


def texture_ref(extra: dict, own_bundle: str,
                ab_key: str, tex_key: str) -> tuple[str, str] | None:
    """``(bundle, texture)`` a list row points at, if it names one.

    A blank ``*AB`` column means "same bundle as the item" — that is the
    common case for base-game clothing, where only the texture name is
    spelled out.

    This lives outside :class:`AssetRef` because a **zipmod's** row carries the
    same columns and has to be read by the same rule: ``SideloaderIndex`` keeps
    the whole row in ``ModRef.extra`` (:class:`sideloader.ModRef`), and giving
    each class its own copy of the sentinel handling is how one of them ends up
    a version behind.  ``"0"`` is the column's own "unset" — Sideloader writes
    it where the base tables leave the cell empty — so both spellings mean the
    item's own bundle.
    """
    tex = str((extra or {}).get(tex_key, "") or "")
    if not tex or tex in ("0", "-"):
        return None
    bundle = str((extra or {}).get(ab_key, "") or "") or own_bundle
    if bundle in ("0", "-", "abdata"):
        bundle = own_bundle
    return bundle, tex


def _classify(table: str) -> str | None:
    for prefix, cat in sorted(CATEGORY.items(), key=lambda kv: -len(kv[0])):
        if table.startswith(prefix):
            return cat
    return None


def _read_bundle_tables(path: Path) -> dict:
    import UnityPy
    out = {}
    try:
        env = UnityPy.load(str(path))
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.lists._read_bundle_tables: env = UnityPy.load(str(path))", exc)
        return out
    for obj in env.objects:
        if obj.type.name != "TextAsset":
            continue
        try:
            d = obj.read()
            raw = d.m_Script if hasattr(d, "m_Script") else d.script
            if isinstance(raw, str):
                raw = raw.encode("utf-8", "surrogateescape")
            table = _unpack(raw)
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.lists._read_bundle_tables: obj.read", exc)
            continue
        if isinstance(table, dict) and "dictList" in table:
            out[d.m_Name] = table
    return out


def rows_by_category_no(category_no: int, inst: Install | None = None) -> dict:
    """``{row id: {column: value}}`` for one ``ChaListDefine.CategoryNo``.

    For the tables whose row *is* the answer — every column is a number or a
    flag and nothing in it names a prefab, so :meth:`AssetIndex._ingest` would
    file the whole table under ``empty`` and throw the columns away.
    ``cha_eyeset`` (CategoryNo 2) is the case this exists for: the card's
    ``eyesPtn`` is a row id here and the number ``ChaControl`` actually hands
    to ``FBSCtrlEyes`` is this row's ``EyesPtn`` column.

    Split tables merge the way ``GlobalMethod.LoadAllListText`` plus its
    readers do — sort by name, then **overwrite by id** (§7.130 一).  An
    append would give one id two rows and the second would never be reached.
    """
    from .install import configured

    inst = inst or configured()
    if not getattr(inst, "valid", False):
        return {}
    merged: dict[int, dict] = {}
    for p in sorted(inst.lists.rglob("*.unity3d")):
        for name, table in sorted(_read_bundle_tables(p).items()):
            try:
                if int(table.get("categoryNo")) != int(category_no):
                    continue
            except (TypeError, ValueError):
                continue
            cols = table.get("lstKey") or []
            if "ID" not in cols:
                continue
            i_id = cols.index("ID")
            for row in (table.get("dictList") or {}).values():
                if not isinstance(row, list) or len(row) <= i_id:
                    continue
                try:
                    merged[int(row[i_id])] = dict(zip(cols, row))
                except (TypeError, ValueError):
                    continue
    return merged


class AssetIndex:
    """category -> {id: AssetRef}, built from every characustom list."""

    def __init__(self):
        self.by_category: dict[str, dict[int, AssetRef]] = {}
        # ids whose row exists but names no asset — Illusion's 「なし」.  Kept
        # apart from the refs so callers can tell "the card asked for nothing"
        # from "we could not find what the card asked for"; reporting the two
        # the same way buries real failures (see §7.14).
        self.empty: dict[str, set] = {}
        # category -> id -> {column: value} for the *_layout preset tables
        self.layouts: dict[str, dict] = {}
        # categories whose tables have a ``MainAB`` column: Sideloader's
        # not-installed rule reads that column, and five texture lists name
        # their map in ``MainTexAB`` / ``ColorMaskAB`` instead (§7.493)
        self.main_ab: set[str] = set()
        self.built = False
        self.tables = 0

    # ------------------------------------------------------------------
    def build(self, inst: Install, extra_dirs=()) -> "AssetIndex":
        self.by_category = {}
        self.empty = {}
        self.layouts = {}
        self.main_ab = set()
        self.tables = 0
        folders = [inst.lists] + [Path(d) for d in extra_dirs]
        for folder in folders:
            if not folder.exists():
                continue
            for p in sorted(folder.rglob("*.unity3d")):
                for name, table in _read_bundle_tables(p).items():
                    cat = _classify(name)
                    if cat is None:
                        continue
                    self._ingest(name, cat, table)
        self.built = True
        return self

    def _ingest(self, table_name: str, category: str, table: dict) -> None:
        cols = table.get("lstKey") or []
        try:
            i_id = cols.index("ID")
            i_name = cols.index("Name")
        except ValueError:
            return
        if category.endswith("_layout"):
            bucket = self.layouts.setdefault(category, {})
            for row in (table.get("dictList") or {}).values():
                if not isinstance(row, list) or len(row) <= i_id:
                    continue
                try:
                    bucket[int(row[i_id])] = dict(zip(cols, row))
                except (TypeError, ValueError):
                    continue
            self.tables += 1
            return
        if "MainAB" in cols:
            self.main_ab.add(category)
        # Texture-only lists (mt_pattern, mt_nip, mt_body_detail …) have no
        # prefab: their "asset" is the map itself, and every table names that
        # column differently.  Skipping them left ColorInfo.pattern and the
        # skin detail maps permanently unresolvable.
        i_ab = i_data = -1
        for ab_key in AB_COLUMNS:
            if ab_key in cols:
                i_ab = cols.index(ab_key)
                break
        for data_key in DATA_COLUMNS:
            if data_key in cols:
                i_data = cols.index(data_key)
                break
        if i_ab < 0 or i_data < 0:
            return
        bucket = self.by_category.setdefault(category, {})
        for row in (table.get("dictList") or {}).values():
            if not isinstance(row, list) or len(row) <= max(i_id, i_name, i_ab, i_data):
                continue
            try:
                rid = int(row[i_id])
            except (TypeError, ValueError):
                continue
            bundle = str(row[i_ab])
            asset = str(row[i_data])
            if not bundle or not asset or bundle in ("0", "-"):
                self.empty.setdefault(category, set()).add(rid)
                continue
            # Keep the rest of the row.  ChaListControl reads far more than the
            # prefab name from here: which texture and colour mask the item
            # uses (``MainTexAB``/``MainTex``/``ColorMaskAB``/``ColorMaskTex``,
            # and the 02 pair for a second renderer), whether a top counts as
            # its own bra (``NotBra``), and the body/bra alpha masks it wants.
            # Dropping the columns forced texture lookups to guess from the
            # material name, which is why some zipmod shoes came out untextured.
            extra = {}
            for ci, key in enumerate(cols):
                if ci in (i_id, i_name, i_ab, i_data) or ci >= len(row):
                    continue
                val = row[ci]
                if val in ("", "0", "-", None):
                    continue
                extra[str(key)] = val
            bucket[rid] = AssetRef(rid, str(row[i_name]), bundle, asset,
                                   table_name, category, extra)
        self.tables += 1

    # ------------------------------------------------------------------
    def get(self, category: str, asset_id: int) -> AssetRef | None:
        return self.by_category.get(category, {}).get(int(asset_id))

    def layout(self, category: str, ident) -> dict:
        """A row of one of the ``*_layout`` preset tables (may be empty)."""
        try:
            return self.layouts.get(category, {}).get(int(ident)) or {}
        except (TypeError, ValueError):
            return {}

    def is_none(self, category: str, asset_id: int) -> bool:
        """Did the list say this id is 「なし」 rather than not saying anything?

        Falls back to :data:`NONE_IDS` for a category this index knows nothing
        about, which is the whole library on a machine with no Illusion
        install.  The live lists still win wherever they exist — a recorded
        copy that could override them would be a second owner of the same fact
        (``CLAUDE.md`` 七).
        """
        try:
            ident = int(asset_id)
        except (TypeError, ValueError):
            return False
        if category in self.by_category or category in self.empty:
            return ident in self.empty.get(category, ())
        return ident in NONE_IDS.get(str(category), ())

    def count(self, category: str) -> int:
        return len(self.by_category.get(category, {}))

    def categories(self) -> dict:
        return {k: len(v) for k, v in sorted(self.by_category.items())}

    def entries(self, category: str) -> list[AssetRef]:
        return sorted(self.by_category.get(category, {}).values(),
                      key=lambda r: r.id)

    # ------------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = Path(path or CACHE)
        data = {"version": CACHE_VERSION, "tables": self.tables,
                "categories": {c: [r.to_dict() for r in v.values()]
                               for c, v in self.by_category.items()},
                "empty": {c: sorted(v) for c, v in self.empty.items()},
                "layouts": self.layouts,
                "main_ab": sorted(self.main_ab)}
        return write_json_atomic(path, data)

    def load(self, path: Path | None = None) -> bool:
        path = Path(path or CACHE)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        # a cache written before the list rows carried their extra columns has
        # no texture / NotBra data, so rebuild rather than silently degrade
        if int(data.get("version", 0)) < CACHE_VERSION:
            return False
        self.by_category = {
            c: {int(r["id"]): AssetRef.from_dict(r) for r in rows}
            for c, rows in (data.get("categories") or {}).items()}
        self.empty = {c: {int(i) for i in ids}
                      for c, ids in (data.get("empty") or {}).items()}
        self.layouts = {c: {int(i): r for i, r in rows.items()}
                        for c, rows in (data.get("layouts") or {}).items()}
        self.main_ab = {str(c) for c in data.get("main_ab") or []}
        self.tables = int(data.get("tables", 0))
        self.built = True
        return True


_index: AssetIndex | None = None


def index(inst: Install | None = None, refresh: bool = False) -> AssetIndex:
    """The shared asset index; cached to disk because a scan takes ~30 s."""
    global _index
    if _index is not None and not refresh:
        return _index
    idx = AssetIndex()
    if not refresh and idx.load():
        _index = idx
        return idx
    from .install import configured
    inst = inst or configured()
    if inst.valid:
        idx.build(inst)
        idx.save()
    _index = idx
    return idx
