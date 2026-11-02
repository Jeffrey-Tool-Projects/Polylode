"""GF2: the install's playable characters' root prefabs, exported through ``polylode.gf2``.

Scanning indexes the install's bundles once (resumable: an unchanged bundle
is not read again) and lists the root prefabs of every family the install's
own character table declares (``gf2.playable``), women and men alike: these
are the game's own models for the user's own use (owner 2026-11-02, §7.1455),
so scans and exports run in ``gf2.gender.all_playable`` and in a GF2 store of
this program's own (:func:`own_scope`), never the one the game publishes
from.  Exporting converts that root and writes it (``gf2.export``).
"""
from __future__ import annotations

import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

from ..core import progress
from .sources import ExtractorError, Log, Model, PathSpec, require

INSTALL = PathSpec("gf2", "GF2 安裝資料夾", "dir", "遊戲資料夾或它的上一層（官方啟動器或 Steam）", install=True,
                   marker="GF2_Exilium_Data")
BLENDER = PathSpec("blender", "Blender 執行檔", "file")
#: the scan's screen of roots (``batch.skinned_roots``): a kept list of another screen is screened again
#: (2: a whole human skeleton too, not only the character's own skinned mesh; 3: a renderer that points at
#: a mesh, §7.1455)
SCREEN = 3


@contextmanager
def own_scope():
    """Every playable character (``gender.all_playable``), in this program's own GF2 store (``ISLANDLIFE_GF2_DIR``)."""
    from ..core.paths import derived_root
    from ..gf2 import gender
    saved = os.environ.get("ISLANDLIFE_GF2_DIR")
    os.environ["ISLANDLIFE_GF2_DIR"] = str(derived_root() / "assets_cache" / "extractor" / "gf2_store")
    try:
        with gender.all_playable():
            yield
    finally:
        if saved is None:
            os.environ.pop("ISLANDLIFE_GF2_DIR", None)
        else:
            os.environ["ISLANDLIFE_GF2_DIR"] = saved


def _outfit(name: str, character: str) -> str:
    """The version a root name carries after the character's name: ``SR01``, ``Dorm``, ``SSR0101``…"""
    match = re.match(r"^c_" + re.escape(character) + r"((?:SSR|SR|Dorm)\d*)?_?(.*)$", name, re.IGNORECASE)
    if not match:
        return name
    return " ".join(part for part in (match.group(1) or "", match.group(2)) if part)


class GF2Source:
    id = "gf2"
    label = "GF2（少女前線 2）"
    paths = (INSTALL,)

    def _catalog(self, settings, log: Log, index: bool = True):
        """The install, its catalog and roster; *index* brings the bundle index up to date first (a scan),
        without it the index a scan made is used, refused when the bundles changed since."""
        import json
        from ..gf2 import playable, store
        from ..gf2.bundles import bundle_paths, scan, scan_directories
        from ..gf2.catalog import Catalog
        install = require(settings, INSTALL)
        root = store.root()
        root.mkdir(parents=True, exist_ok=True)
        files = [p.stat() for p in bundle_paths(install)]
        stamp = json.dumps([str(install), len(files), sum(f.st_size for f in files), max((f.st_mtime_ns for f in files), default=0)])
        recorded = root / "extractor_index_stamp.json"
        if index:
            log("建立 bundle 索引（第一次會花很久；之後只讀有變動的檔案）…")
            progress.step("建立 bundle 索引")
            scan_directories(install, root / "cab_index.jsonl")
            scan(install, root / "bundle_index.jsonl", 0)
            recorded.write_text(stamp, encoding="utf-8")
        elif not recorded.is_file() or recorded.read_text(encoding="utf-8") != stamp:
            raise ExtractorError("GF2 的安裝在掃描之後有變動（或還沒掃描過），請先重新掃描")
        return install, Catalog(install, root / "bundle_index.jsonl"), playable.roster(install)

    def scan(self, settings, log: Log) -> list[Model]:
        with own_scope():
            return self._scan(settings, log)

    def export(self, model: Model, settings, output: Path, formats: list[str], log: Log) -> list[Path]:
        with own_scope():
            return self._export(model, settings, output, formats, log)

    def _scan(self, settings, log: Log) -> list[Model]:
        import json
        from ..gf2 import playable, store
        from ..gf2.batch import declared_entries, load_results, save_results, skinned_roots
        from ..gf2.selection import roots_only
        install, catalog, declared = self._catalog(settings, log)
        # the roots with a skinned mesh of their own, kept for the bundle index they were found in: an
        # unchanged install is not read again (the first pass reads every root, tens of minutes)
        stamp = (store.root() / "extractor_index_stamp.json").read_text(encoding="utf-8")
        cache = store.root() / "extractor_models.json"
        saved = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
        if saved.get("stamp") == stamp and saved.get("screen") == SCREEN:
            entries = saved["entries"]
        else:
            roots = roots_only(catalog, declared_entries(catalog, declared), store.root() / "model_inventory.json")
            results = load_results()
            log(f"{len(roots)} 個根節點；檢查哪些有自己的蒙皮網格…")
            entries = skinned_roots(catalog, roots, declared, results, log)
            save_results(install, results)
            cache.write_text(json.dumps({"stamp": stamp, "screen": SCREEN, "entries": entries}), encoding="utf-8")
        known = load_results()
        models = []
        for entry in entries:
            if known.get(f'{entry["cab"]}/{entry["id"]}', {}).get("status") in ("not_model", "excluded_scope"):
                continue
            family = playable.families(entry["name"], declared)[0]
            models.append(Model("gf2", f'{entry["name"]}@{entry["cab"][-8:]}', family["name"],
                                _outfit(entry["name"], family["name"]), {"entry": entry}))
        log(f"可匯出的模型 {len(models)} 個（沒有自己蒙皮網格的根節點不列）")
        return models

    def _export(self, model: Model, settings, output: Path, formats: list[str], log: Log) -> list[Path]:
        from ..gf2.batch import convert_entry, load_results, save_results
        from ..gf2.export import export_one, scene_document
        from .sources import blender_for, write_own
        blender = blender_for(settings, formats, BLENDER)
        install, catalog, declared = self._catalog(settings, log, index=False)
        entry = model.detail["entry"]
        results = load_results()
        progress.step("從安裝轉換模型")
        outcome = convert_entry(catalog, entry, declared, results, log=log)
        save_results(install, results)
        row = results[f'{entry["cab"]}/{entry["id"]}']
        if outcome != "converted":
            raise ExtractorError(f"{model.ident} 沒有轉換：{row.get('reason', outcome)}")
        written = []
        if "blend" in formats:
            log(f"以 Blender 建 .blend 並讀回 {row['ident']}（每個模型幾分鐘）…")
            progress.step("以 Blender 建 .blend 並讀回")
            exported = export_one(blender, row["ident"], declared)
            if not exported.get("ok"):
                raise ExtractorError(f"{row['ident']} 讀回沒有通過：{'; '.join(exported.get('problems') or [])}")
            written += copy_out(Path(exported["fbx"]), exported.get("blend"), output, ["blend"])
        if any(kind in formats for kind in ("glb", "fbx")):
            # into the job's output: the store's exports/<ident>.fbx is the Blender-checked one export_one recorded
            progress.step("組合場景")
            _scene, _selection, _layout, _folder, document = scene_document(row["ident"], declared)
            # glTF and FBX cannot multiply or add the eye layers: their effect is baked into the eye's map
            # (gf2.eye_bake, owner 2026-11-02); the .blend above keeps them as layers
            from ..gf2.eye_bake import bake
            progress.step("把眼睛疊層烘進貼圖")
            written += write_own(formats, bake(document, log), output, row["ident"], blender, log)
        return written


def copy_out(fbx: Path, blend, output: Path, formats: list[str]) -> list[Path]:
    """Copy the requested files: the FBX with its ``textures`` folder beside it, and the ``.blend``."""
    written = []
    if "fbx" in formats:
        written.append(Path(shutil.copy2(fbx, output / fbx.name)))
        textures = fbx.parent / "textures"
        if textures.is_dir():
            shutil.copytree(textures, output / "textures", dirs_exist_ok=True)
            written.append(output / "textures")
    if "blend" in formats:
        blend = blend if blend and Path(blend).is_file() else fbx.with_suffix(".blend")
        if not Path(blend).is_file():
            raise ExtractorError("這個模型沒有 .blend（只有已核實的女性可玩角色才寫 .blend）")
        written.append(Path(shutil.copy2(blend, output / Path(blend).name)))
    return written
