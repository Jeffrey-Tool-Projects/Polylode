"""Studio item / map / FK bone / animation catalogues from a user-owned Koikatsu install.

Item lists live as ``ItemList_*`` MonoBehaviours in ``abdata/studio/info/*.unity3d``,
the FK bone table as ``Bone_*`` and Studio's own animations as ``Anime_*`` in the
same bundles.  Map rows live in
``abdata/map/list/mapinfo/*.unity3d``.  All are read from the configured
install and cached under ``UserData/config``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..core.paths import CONFIG_DIR, write_json_atomic
from .install import Install, configured, autodetect, set_path
from ..core import swallowed

ITEM_CACHE = CONFIG_DIR / "origin_studio_items.json"
MAP_CACHE = CONFIG_DIR / "origin_studio_maps.json"
BONE_CACHE = CONFIG_DIR / "origin_studio_bones.json"
ANIME_CACHE = CONFIG_DIR / "origin_studio_anime.json"
HAND_CACHE = CONFIG_DIR / "origin_studio_hand.json"

# Studio's Map_* rows name their collision mesh ``map_col_<game map No>``; see
# the long note in _scan_maps for why that number, and not 管理番号, is the key.
_MAP_COL = re.compile(r"map_col_(\d+)")


@dataclass
class StudioItemRef:
    group: int
    category: int
    no: int
    name: str
    bundle: str
    asset: str
    manifest: str = ""

    def key(self) -> tuple[int, int, int]:
        return (self.group, self.category, self.no)

    def to_dict(self) -> dict:
        return {
            "group": self.group, "category": self.category, "no": self.no,
            "name": self.name, "bundle": self.bundle, "asset": self.asset,
            "manifest": self.manifest,
        }

    @staticmethod
    def from_dict(d: dict) -> "StudioItemRef":
        return StudioItemRef(
            int(d["group"]), int(d["category"]), int(d["no"]),
            str(d.get("name") or ""), str(d.get("bundle") or ""),
            str(d.get("asset") or ""), str(d.get("manifest") or ""))


@dataclass
class MapRef:
    no: int
    name: str
    bundle: str
    asset: str
    thumb_bundle: str = ""
    thumb_asset: str = ""
    is_2d: bool = False

    def to_dict(self) -> dict:
        return {
            "no": self.no, "name": self.name, "bundle": self.bundle,
            "asset": self.asset, "thumb_bundle": self.thumb_bundle,
            "thumb_asset": self.thumb_asset, "is_2d": self.is_2d,
        }

    @staticmethod
    def from_dict(d: dict) -> "MapRef":
        return MapRef(
            int(d["no"]), str(d.get("name") or ""),
            str(d.get("bundle") or ""), str(d.get("asset") or ""),
            str(d.get("thumb_bundle") or ""), str(d.get("thumb_asset") or ""),
            bool(d.get("is_2d")))


@dataclass
class FKBone:
    """One row of Studio's FK bone table (``Studio.Info.LoadBoneInfo``).

    Columns in order: 管理番号 (``no``, the scene's ``dicKey``), 取得先 (the
    transform name ``FindLoop`` looks for), 名称, 分類 (``group``), 操作
    (``level``).  A later list's row replaces an earlier one's (``set_Item``).
    """
    no: int
    bone: str
    name: str
    group: int
    level: int

    def to_dict(self) -> dict:
        return {"no": self.no, "bone": self.bone, "name": self.name,
                "group": self.group, "level": self.level}

    @staticmethod
    def from_dict(d: dict) -> "FKBone":
        return FKBone(int(d["no"]), str(d.get("bone") or ""), str(d.get("name") or ""),
                      int(d.get("group") or 0), int(d.get("level") or 0))


@dataclass
class StudioAnime:
    """One ``Anime_*`` or ``HAnime_*`` row (``Studio.Info``, §7.520／§7.521).

    Both lists fill ``dicAnimeLoadInfo[group][category][no]``.  An ``Anime_``
    row (``LoadAnimeLoadInfo``) is 表示名, バンドルパス, ファイル名 (the
    ``RuntimeAnimatorController``) and クリップ名 — the **state**
    ``CharAnimeCtrl.Play`` hands to ``Animator.Play``, not a clip name.

    An ``HAnime_`` row (``HAnimeLoadInfo..ctor``, *kind* ``"h"``) names a base
    controller (columns 4, 5), the H pack's override controller (6, 7 —
    *bundle* / *controller* here, the one whose states are baked), the state
    (8), ``breastLayer`` (9), ``dynamic`` (10, 11), ``isMotion`` (12), ``pv``
    (13‥16) and the yure list (17, 18).  ``OCIChar.LoadAnime`` sets
    ``motion`` to the scene's ``animePattern`` only when ``isMotion``.
    """
    group: int
    category: int
    no: int
    name: str
    bundle: str
    controller: str
    state: str
    kind: str = "anime"
    base_bundle: str = ""
    base_controller: str = ""
    breast_layer: int = -1
    is_motion: bool = False

    def key(self) -> tuple[int, int, int]:
        return (self.group, self.category, self.no)

    def to_dict(self) -> dict:
        d = {"group": self.group, "category": self.category, "no": self.no, "name": self.name,
             "bundle": self.bundle, "controller": self.controller, "state": self.state}
        if self.kind != "anime":
            d.update(kind=self.kind, base_bundle=self.base_bundle,
                     base_controller=self.base_controller, breast_layer=self.breast_layer,
                     is_motion=self.is_motion)
        return d

    @staticmethod
    def from_dict(d: dict) -> "StudioAnime":
        return StudioAnime(int(d["group"]), int(d["category"]), int(d["no"]), str(d.get("name") or ""),
                           str(d.get("bundle") or ""), str(d.get("controller") or ""),
                           str(d.get("state") or ""), str(d.get("kind") or "anime"),
                           str(d.get("base_bundle") or ""), str(d.get("base_controller") or ""),
                           int(d.get("breast_layer", -1)), bool(d.get("is_motion", False)))


@dataclass
class HandAnime:
    """One ``HandAnime_<hand>_<n>`` row (``Studio.Info.LoadHandAnimeInfo``, E1d).

    ``dicHandAnime[hand][no]`` — *hand* 0 is ``HandKind.Left``, 1 ``Right`` —
    and the row names the controller ``HandAnimeCtrl.LoadAnime`` puts on that
    hand's own ``Animator`` and the state it plays (``Animator.Play(clip)``).
    """
    hand: int
    no: int
    name: str
    bundle: str
    controller: str
    state: str

    def key(self) -> tuple[int, int]:
        return (self.hand, self.no)

    def to_dict(self) -> dict:
        return {"hand": self.hand, "no": self.no, "name": self.name, "bundle": self.bundle,
                "controller": self.controller, "state": self.state}

    @staticmethod
    def from_dict(d: dict) -> "HandAnime":
        return HandAnime(int(d["hand"]), int(d["no"]), str(d.get("name") or ""), str(d.get("bundle") or ""),
                         str(d.get("controller") or ""), str(d.get("state") or ""))


@dataclass
class StudioCatalog:
    items: dict[tuple[int, int, int], StudioItemRef] = field(default_factory=dict)
    maps: dict[int, MapRef] = field(default_factory=dict)
    bones: dict[int, FKBone] = field(default_factory=dict)
    anime: dict[tuple[int, int, int], StudioAnime] = field(default_factory=dict)
    hands: dict[tuple[int, int], HandAnime] = field(default_factory=dict)
    built: bool = False

    def lookup_item(self, group: int, category: int, no: int
                    ) -> StudioItemRef | None:
        return self.items.get((int(group), int(category), int(no)))

    def lookup_map(self, no: int) -> MapRef | None:
        return self.maps.get(int(no))

    # ------------------------------------------------------------------ build
    def build(self, inst: Install | None = None) -> "StudioCatalog":
        inst = inst or _install()
        if inst is None or not inst.valid:
            self.built = False
            return self
        self.items = _scan_items(inst)
        self.bones = _scan_bones(inst)
        self.anime = _scan_anime(inst)
        self.hands = _scan_hands(inst)
        self.maps = _scan_maps(inst)
        self.built = True
        self._save()
        return self

    def ensure(self, inst: Install | None = None) -> "StudioCatalog":
        if self.built and self.items:
            return self
        if self._load():
            return self
        return self.build(inst)

    def _save(self) -> None:
        write_json_atomic(ITEM_CACHE, [r.to_dict() for r in self.items.values()])
        write_json_atomic(MAP_CACHE, [r.to_dict() for r in self.maps.values()])
        write_json_atomic(BONE_CACHE, [r.to_dict() for r in self.bones.values()])
        write_json_atomic(ANIME_CACHE, [r.to_dict() for r in self.anime.values()])
        write_json_atomic(HAND_CACHE, [r.to_dict() for r in self.hands.values()])

    def _load(self) -> bool:
        try:
            # a cache written before the FK table was part of it is rebuilt (§7.518)
            if not all(f.exists() for f in (ITEM_CACHE, BONE_CACHE, ANIME_CACHE, HAND_CACHE)):
                return False
            items = json.loads(ITEM_CACHE.read_text(encoding="utf-8"))
            self.items = {}
            for d in items:
                r = StudioItemRef.from_dict(d)
                self.items[r.key()] = r
            self.bones = {}
            for d in json.loads(BONE_CACHE.read_text(encoding="utf-8")):
                r = FKBone.from_dict(d)
                self.bones[r.no] = r
            self.anime = {}
            for d in json.loads(ANIME_CACHE.read_text(encoding="utf-8")):
                r = StudioAnime.from_dict(d)
                self.anime[r.key()] = r
            self.hands = {}
            for d in json.loads(HAND_CACHE.read_text(encoding="utf-8")):
                r = HandAnime.from_dict(d)
                self.hands[r.key()] = r
            self.maps = {}
            if MAP_CACHE.exists():
                for d in json.loads(MAP_CACHE.read_text(encoding="utf-8")):
                    r = MapRef.from_dict(d)
                    self.maps[r.no] = r
            self.built = bool(self.items)
            return self.built
        except Exception as exc:  # noqa: BLE001
            swallowed.note("origin.studio_catalog.StudioCatalog._load: if not ITEM_CACHE.exists():", exc)
            return False


_catalog: StudioCatalog | None = None


def catalog(rebuild: bool = False) -> StudioCatalog:
    global _catalog
    if _catalog is None:
        _catalog = StudioCatalog()
    if rebuild or not _catalog.built:
        if not rebuild and _catalog._load():
            return _catalog
        _catalog.build()
    return _catalog


def _install() -> Install | None:
    inst = configured()
    if not inst.valid:
        auto = autodetect()
        if auto is not None and auto.valid:
            set_path(auto.root)
            return auto
        return None
    return inst


def _scan_items(inst: Install) -> dict[tuple[int, int, int], StudioItemRef]:
    import UnityPy
    out: dict[tuple[int, int, int], StudioItemRef] = {}
    info = inst.abdata / "studio" / "info"
    if not info.is_dir():
        return out
    for bp in sorted(info.glob("*.unity3d")):
        try:
            env = UnityPy.load(str(bp))
        except Exception as exc: # noqa: BLE001
            swallowed.note("origin.studio_catalog._scan_items: UnityPy.load", exc)
            continue
        for o in env.objects:
            if o.type.name != "MonoBehaviour":
                continue
            try:
                tree = o.read_typetree()
            except Exception as exc: # noqa: BLE001
                swallowed.note("origin.studio_catalog._scan_items: o.read_typetree", exc)
                continue
            name = tree.get("m_Name") or ""
            if not str(name).startswith("ItemList_"):
                continue
            rows = tree.get("list") or []
            if len(rows) < 2:
                continue
            headers = rows[0].get("list") if isinstance(rows[0], dict) else None
            if not headers or "管理番号" not in headers:
                continue
            for row in rows[1:]:
                vals = row.get("list") if isinstance(row, dict) else None
                if not vals or len(vals) < 7:
                    continue
                d = dict(zip(headers, vals))
                try:
                    g = int(float(d.get("大きい項目")))
                    c = int(float(d.get("中間項目")))
                    n = int(float(d.get("管理番号")))
                except Exception as exc: # noqa: BLE001
                    swallowed.note("origin.studio_catalog._scan_items: d.get", exc)
                    continue
                out[(g, c, n)] = StudioItemRef(
                    g, c, n,
                    str(d.get("名称") or ""),
                    str(d.get("バンドルパス") or "").replace("\\", "/"),
                    str(d.get("ファイルパス") or ""),
                    str(d.get("マニフェスト") or ""),
                )
    return out


def _scan_bones(inst: Install) -> dict[int, FKBone]:
    """``Bone_*`` rows of every ``studio/info`` bundle, in the order Studio loads them.

    ``LoadBoneInfo`` skips the header row, needs an integer 管理番号 and a
    non-empty 取得先, and assigns by number — so a later bundle's row wins.
    """
    import UnityPy
    out: dict[int, FKBone] = {}
    info = inst.abdata / "studio" / "info"
    if not info.is_dir():
        return out
    for bp in sorted(info.glob("*.unity3d")):
        try:
            env = UnityPy.load(str(bp))
        except Exception as exc:  # noqa: BLE001
            swallowed.note("origin.studio_catalog._scan_bones: UnityPy.load", exc)
            continue
        for o in env.objects:
            if o.type.name != "MonoBehaviour" or not o.peek_name().startswith("Bone_"):
                continue
            for row in (o.read_typetree().get("list") or [])[1:]:
                vals = row.get("list") if isinstance(row, dict) else None
                if not vals or len(vals) < 5 or not str(vals[1]):
                    continue
                try:
                    no = int(vals[0])
                    out[no] = FKBone(no, str(vals[1]), str(vals[2]), int(vals[3]), int(vals[4]))
                except ValueError:
                    continue
    return out


def _scan_anime(inst: Install) -> dict[tuple[int, int, int], StudioAnime]:
    """``Anime_*`` and ``HAnime_*`` rows the way ``Studio.Info`` reads them.

    Bundle by bundle, the ``Anime_`` lists and then the ``HAnime_`` lists
    (``<LoadExcelDataCoroutine>``), into the same table.  Both loaders skip two
    rows (a blank one and the header), keep a row only when column 4 names a
    bundle that exists (``FileCheck.Check``), take the key from the row's own
    管理番号／大きい項目／中間項目 columns and assign — a later row wins.
    """
    import UnityPy
    out: dict[tuple[int, int, int], StudioAnime] = {}
    info = inst.abdata / "studio" / "info"
    if not info.is_dir():
        return out
    for bp in sorted(info.glob("*.unity3d")):
        try:
            env = UnityPy.load(str(bp))
        except Exception as exc:  # noqa: BLE001
            swallowed.note("origin.studio_catalog._scan_anime: UnityPy.load", exc)
            continue
        lists = {"Anime": [], "HAnime": []}
        for o in env.objects:
            if o.type.name != "MonoBehaviour":
                continue
            m = re.fullmatch(r"(H?Anime)_\d*_\d*_\d*", o.peek_name())
            if m:
                lists[m.group(1)].append(o)
        for kind, objs in (("anime", lists["Anime"]), ("h", lists["HAnime"])):
            for o in objs:
                for row in (o.read_typetree().get("list") or [])[2:]:
                    v = row.get("list") if isinstance(row, dict) else None
                    if not v or len(v) < 7 or not (inst.abdata / str(v[4])).is_file():
                        continue
                    r = _anime_row(kind, v)
                    if r is not None:
                        out[r.key()] = r
    return out


def _scan_hands(inst: Install) -> dict[tuple[int, int], HandAnime]:
    """``HandAnime_<hand>_<n>`` rows the way ``LoadHandAnimeInfo`` reads them.

    Per ``studio/info`` bundle the lists are grouped by *hand* and taken in
    ``<n>`` order; each skips one header row, needs an integer 管理番号 and
    assigns by it — a later row wins.  No bundle check (unlike ``Anime_``).
    """
    import UnityPy
    out: dict[tuple[int, int], HandAnime] = {}
    info = inst.abdata / "studio" / "info"
    if not info.is_dir():
        return out
    for bp in sorted(info.glob("*.unity3d")):
        try:
            env = UnityPy.load(str(bp))
        except Exception as exc:  # noqa: BLE001
            swallowed.note("origin.studio_catalog._scan_hands: UnityPy.load", exc)
            continue
        lists: dict[int, dict[int, object]] = {}
        for o in env.objects:
            if o.type.name != "MonoBehaviour":
                continue
            m = re.fullmatch(r"HandAnime_(\d+)_(\d+)", o.peek_name(), flags=re.I)
            if m:
                lists.setdefault(int(m.group(1)), {})[int(m.group(2))] = o
        for hand, files in sorted(lists.items()):
            for _, o in sorted(files.items()):
                for row in (o.read_typetree().get("list") or [])[1:]:
                    v = row.get("list") if isinstance(row, dict) else None
                    if not v or len(v) < 5:
                        continue
                    try:
                        no = int(v[0])
                    except ValueError:
                        continue
                    out[(hand, no)] = HandAnime(hand, no, str(v[1]), str(v[2]), str(v[3]), str(v[4]))
    return out


def _anime_row(kind: str, v: list) -> StudioAnime | None:
    """One list row as a :class:`StudioAnime`, ``None`` when its key does not parse."""
    try:
        if kind == "anime":
            return StudioAnime(int(v[1]), int(v[2]), int(v[0]), str(v[3]), str(v[4]),
                               str(v[5]), str(v[6]))
        if len(v) < 9:
            return None
        get = lambda i: str(v[i]) if i < len(v) else ""        # noqa: E731 - SafeGet
        try:
            layer = int(get(9))
        except ValueError:
            layer = -1                                          # TryParse keeps the ctor's -1
        return StudioAnime(int(v[1]), int(v[2]), int(v[0]), get(3), get(6), get(7), get(8),
                           kind="h", base_bundle=get(4), base_controller=get(5),
                           breast_layer=layer, is_motion=get(12).strip().lower() == "true")
    except ValueError:
        return None


def _scan_maps(inst: Install) -> dict[int, MapRef]:
    import UnityPy
    out: dict[int, MapRef] = {}
    # 1) Game mapinfo (action maps with real scene bundles + thumbs)
    folder = inst.abdata / "map" / "list" / "mapinfo"
    if folder.is_dir():
        for bp in sorted(folder.glob("*.unity3d")):
            try:
                env = UnityPy.load(str(bp))
            except Exception as exc: # noqa: BLE001
                swallowed.note("origin.studio_catalog._scan_maps: UnityPy.load", exc)
                continue
            for o in env.objects:
                if o.type.name != "MonoBehaviour":
                    continue
                try:
                    tree = o.read_typetree()
                except Exception as exc: # noqa: BLE001
                    swallowed.note("origin.studio_catalog._scan_maps: o.read_typetree", exc)
                    continue
                for e in tree.get("param") or []:
                    if not isinstance(e, dict):
                        continue
                    try:
                        no = int(e.get("No"))
                    except Exception as exc: # noqa: BLE001
                        swallowed.note("origin.studio_catalog._scan_maps: e.get", exc)
                        continue
                    out[no] = MapRef(
                        no=no,
                        name=str(e.get("MapName") or f"map{no}"),
                        bundle=str(e.get("AssetBundleName") or "").replace("\\", "/"),
                        asset=str(e.get("AssetName") or ""),
                        thumb_bundle=str(e.get("ThumbnailBundle") or "").replace("\\", "/"),
                        thumb_asset=str(e.get("ThumbnailAsset") or ""),
                        is_2d=bool(e.get("is2D")),
                    )
    # 2) Studio Map_* lists.
    #
    # These rows are keyed by a **studio-local** index, not by the game's map
    # number.  Using 管理番号 as if it were ``No`` renamed 24 of the 25
    # overlapping maps to some other map's name — '教室1-1' ended up on the 3F
    # bundle — and the ten rows that carry no path at all landed on real game
    # numbers 28 and 33..41 and renamed those too.  Same integer, different
    # thing (CLAUDE.md 七).
    #
    # The link back to the game map is in the row's own asset name: every
    # bundled row points at ``h/list/map_col_NN``, and NN *is* the game's
    # ``No`` — 29 of 32 names then agree exactly and the other three are
    # wording variants of the same room ('女子トイレ覗き' / '女子トイレ個室'),
    # with none left over.  So key by that.
    #
    # These rows can never supply geometry either: 32 of the 42 point at a
    # collision mesh under ``h/list/`` and the remaining 10 have both path
    # columns empty.  The old "fill in a bundle from here" branch could not
    # fire.  What is left is a name, and the game's own table is the better
    # authority for one, so this only fills a name in that is missing.
    unplaceable = 0
    info = inst.abdata / "studio" / "info"
    if info.is_dir():
        for bp in sorted(info.glob("*.unity3d")):
            try:
                env = UnityPy.load(str(bp))
            except Exception as exc: # noqa: BLE001
                swallowed.note("origin.studio_catalog._scan_maps: UnityPy.load", exc)
                continue
            for o in env.objects:
                if o.type.name != "MonoBehaviour":
                    continue
                try:
                    tree = o.read_typetree()
                except Exception as exc: # noqa: BLE001
                    swallowed.note("origin.studio_catalog._scan_maps: o.read_typetree", exc)
                    continue
                name = tree.get("m_Name") or ""
                if not str(name).startswith("Map_"):
                    continue
                rows = tree.get("list") or []
                headers = None
                for row in rows:
                    vals = row.get("list") if isinstance(row, dict) else None
                    if not vals:
                        continue
                    if vals[0] == "管理番号":
                        headers = vals
                        continue
                    if headers is None or len(vals) < 4:
                        continue
                    d = dict(zip(headers, vals + [""] * max(0, len(headers) - len(vals))))
                    asset = str(d.get("ファイルパス") or "")
                    m = _MAP_COL.fullmatch(asset)
                    if m is None:
                        # A studio map whose bundle this table does not carry.
                        # Counted, not invented: dropping it silently and
                        # filing it under its studio index are both wrong, and
                        # the second one is what corrupted the real rows.
                        unplaceable += 1
                        continue
                    no = int(m.group(1))
                    existing = out.get(no)
                    if existing is None:
                        out[no] = MapRef(no=no, name=str(d.get("名称") or f"map{no}"))
                    elif not existing.name and d.get("名称"):
                        existing.name = str(d.get("名称"))
    if unplaceable:
        print(f"[studio_catalog] {unplaceable} studio map rows carry no bundle "
              f"path and no map_col id — not listed")
    return out
