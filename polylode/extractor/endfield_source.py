"""Arknights: Endfield: the install's playable characters, converted by ``polylode.endfield`` and written as FBX and ``.blend``.

Reading the install takes AnimeStudio (the Endfield research kit's build;
the packaged program carries it, ``bundled``) and the install's CAB and
asset maps, which the first scan makes from the install itself
(``endfield.mapbuild``: the current slots in checked batches) into
``assets_cache/endfield/meta``, and makes again when the install's VFS
files change; a verified batch of an unchanged chunk is kept.  Scanning
lists the prefabs of the playable characters (``pipeline.KINDS["chara"]``;
the NPCs are other kinds).

Every playable prefab is listed and converted: these are the game's own
models as it draws them, for the user's own use (owner 2026-11-02, §7.1455).
So the conversions go into a store of this program's own (:func:`own_store`),
never the Endfield store the game reads (``Charas`` and the shared store),
whose scope — who is converted, the age table, the completed bodies — stays
the game's.
"""
from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from ..core import progress
from .sources import ExtractorError, Log, Model, PathSpec, require

INSTALL = PathSpec("endfield", "終末地安裝資料夾", "dir", "遊戲資料夾或它的上一層（官方啟動器或 Steam）", install=True,
                   marker="Endfield_Data/StreamingAssets/VFS")
EXPORTER = PathSpec("animestudio", "AnimeStudio.CLI.exe（終末地版）", "file", "留空：用內建的 AnimeStudio")
BLENDER = PathSpec("blender", "Blender 執行檔", "file")


def data_root(install: Path) -> Path:
    """``Endfield_Data`` of the folder the user named (that folder itself, or the one holding it)."""
    from ..endfield.mapbuild import data_layers
    for candidate in (install, install / "Endfield_Data"):
        if data_layers(candidate):
            return candidate
    raise ExtractorError(f"這不是終末地的安裝資料夾（找不到 Endfield_Data/StreamingAssets/VFS）：{install}")


def vfs_stamp(game: Path) -> list:
    """Count, size and newest change of the VFS files of every layer: another stamp means the game changed."""
    from ..endfield.mapbuild import data_layers
    files = [p for layer in data_layers(game) for p in (game / layer / "VFS").rglob("*") if p.is_file()]
    return [str(game), len(files), sum(p.stat().st_size for p in files), max((p.stat().st_mtime_ns for p in files), default=0)]


def ensure_maps(exe: Path, install: Path, log: Log) -> Path:
    """The ``meta`` folder of maps made from *install*: kept while its VFS files are unchanged, else made
    (again) with ``endfield.mapbuild`` — resumable, a verified batch of an unchanged chunk is kept."""
    import json
    from ..endfield import mapbuild
    from ..endfield.pipeline import stage_root
    game = data_root(install)
    meta, work = stage_root() / "meta", stage_root() / "map_build"
    stamp_path = meta / "install_stamp.json"
    stamp = vfs_stamp(game)
    if stamp_path.is_file() and (meta / "build.json").is_file() \
            and json.loads(stamp_path.read_text(encoding="utf-8")) == stamp \
            and json.loads((meta / "build.json").read_text(encoding="utf-8")).get("types") == list(mapbuild.TYPES):
        return meta
    log("從遊戲安裝建立資產對照表（第一次或遊戲更新後；分批進行，可中斷續跑）…")
    stamp_path.unlink(missing_ok=True)
    try:
        summary = mapbuild.build(exe, game, work, meta, log)
    except mapbuild.MapBuildError as failure:
        raise ExtractorError(f"資產對照表建立失敗：{failure}") from None
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")
    log(f"資產對照表完成：{summary['slots']} 個資料槽、{summary['batches']} 批")
    return meta


@contextmanager
def own_store():
    """Convert into and read from this program's own Endfield store, one folder with no ``Charas`` lookup
    (``store.DIR_ENV``): what the extractor converts is never in the game's store."""
    from ..core.paths import derived_root
    from ..endfield import store
    saved = {name: os.environ.get(name) for name in (store.DIR_ENV, store.CHARAS_ENV)}
    os.environ[store.DIR_ENV] = str(derived_root() / "assets_cache" / "extractor" / "endfield_store")
    os.environ.pop(store.CHARAS_ENV, None)
    try:
        yield store.root()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def character_name(ident: str) -> str:
    """``chr_0036_jsspsi_postmodel`` → ``jsspsi``: the code name between the number and the kind."""
    parts = ident.split("_")
    if len(parts) >= 4 and parts[0] == "chr" and parts[1].isdigit():
        return "_".join(parts[2:-1])
    return ident


class EndfieldSource:
    id = "endfield"
    label = "明日方舟：終末地"
    paths = (INSTALL, EXPORTER)

    def _pipeline(self, settings, log: Log):
        from ..endfield import pipeline as PL
        install = require(settings, INSTALL)
        from . import bundled
        exe = require(settings, EXPORTER, bundled.animestudio())
        meta = ensure_maps(exe, install, log)
        try:
            sources = PL.sources_from_export(exe, meta)
        except FileNotFoundError as missing:
            raise ExtractorError(f"資產對照表缺 {missing}") from None
        log("讀資產對照表（第一次會建索引，要幾分鐘）…")
        progress.step("讀資產對照表")
        pipeline = PL.Pipeline(sources, log)
        elsewhere = pipeline.cabs.outside(install)
        if elsewhere:
            raise ExtractorError(f"對照表不是從這個安裝建的：{elsewhere} / {len(pipeline.cabs)} 個區塊不在 {install} 底下")
        return pipeline

    def scan(self, settings, log: Log) -> list[Model]:
        from ..endfield import pipeline as PL
        from ..endfield import store
        pipeline = self._pipeline(settings, log)
        models = []
        with own_store():
            for job in PL.select(pipeline.assets, ["chara"]):
                known = store.has("chara", job.ident)
                models.append(Model("endfield", job.ident, character_name(job.ident),
                                    "已轉換" if known else "", {"container": job.container}))
        log(f"可玩角色預製物 {len(models)} 個")
        return models

    def export(self, model: Model, settings, output: Path, formats: list[str], log: Log) -> list[Path]:
        from ..endfield import pipeline as PL
        from ..endfield import poses as P
        from ..endfield import store
        from .sources import blender_for, shared_writes
        blender = blender_for(settings, formats, BLENDER)
        with own_store(), shared_writes("endfield", log):
            # what the store shares between models — a prefab converted, a character's clips decoded — is written by
            # one job at a time; the rest of the export runs beside the others
            if not store.has("chara", model.ident):
                pipeline = self._pipeline(settings, log)
                job = next((j for j in PL.select(pipeline.assets, ["chara"]) if j.ident == model.ident), None)
                if job is None:
                    raise ExtractorError(f"安裝裡沒有 {model.ident}")
                progress.step("從安裝轉換預製物")
                result = pipeline.run([job])
                key = f"chara/{model.ident}"
                if key in result.failed:
                    raise ExtractorError(f"{model.ident} 轉換失敗：{result.failed[key]}")
                if key in result.empty:
                    raise ExtractorError(f"{model.ident} 沒有網格")
            if settings.poses:                            # the clips decoded only when the poses are asked for
                P.ensure(str((store.read_extra("chara", model.ident) or {}).get("model") or ""),
                         lambda: self._pipeline(settings, log), log)
        with own_store():
            return write_files(model.ident, blender, output, formats, log,
                               shaders=lambda names: self._locked_shaders(settings, names, log),
                               poses=(lambda: self._pipeline(settings, log))
                               if settings.poses or settings.expressions else None,
                               pose_clips=settings.poses, expressions=settings.expressions,
                               export_shaders=settings.export_shaders, parallel=True)

    def _locked_shaders(self, settings, names: list[str], log: Log) -> None:
        from .sources import shared_writes
        with shared_writes("endfield", log):
            self._fetch_shaders(settings, names, log)

    def _fetch_shaders(self, settings, names: list[str], log: Log) -> None:
        """The shaders *names* taken from the installation into the store's cache (once)."""
        from ..endfield import shader_code
        log(f"從安裝取出 shader：{'、'.join(names)}")
        progress.step("取出 shader")
        shader_code.fetch(self._pipeline(settings, log), names, log)


#: the most processes one export's poses are worked out in; fewer on a computer of fewer cores (half of them)
POSE_WORKERS = 6


def frames_in_workers(ident: str, ids: list[str], bones: list[str], log: Log) -> dict:
    """The frames of the stored clips *ids* on *bones* of the stored *ident*, worked out by worker processes of this
    program side by side (``--pose-frames``, :func:`run_pose_frames`): ``{clip id: (frames, bones, 4, 4)}``.  A clip a
    worker did not give back is left out (the caller works it out itself); with too few clips or cores, ``{}``.

    One clip after another, jsspsi's 25 took a minute; the clips are independent of each other."""
    import json
    import shutil
    import subprocess
    import tempfile
    import numpy as np
    from . import jobs
    count = min(len(ids), max(1, (os.cpu_count() or 2) // 2), POSE_WORKERS)
    if count < 2:
        return {}
    scratch = Path(tempfile.mkdtemp(prefix="polylode-poses-"))
    try:
        started = []
        for k in range(count):
            chunk = ids[k::count]
            folder = scratch / str(k)
            folder.mkdir()
            request = folder / "request.json"
            request.write_text(json.dumps({"ident": ident, "ids": chunk, "bones": bones}), encoding="utf-8")
            command = jobs._start("--pose-frames", str(request))
            started.append((subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)), chunk, folder))
        log(f"以 {count} 個行程同時計算 {len(ids)} 個姿勢")
        found = {}
        for process, chunk, folder in started:
            said, _ = process.communicate()
            if process.returncode != 0:
                tail = said.decode("utf-8", "replace").strip().splitlines()[-1:] if said else []
                log(f"  一組姿勢平行計算失敗，改在本行程計算：{tail[0] if tail else process.returncode}")
                continue
            for number, clip in enumerate(chunk):
                path = folder / f"{number}.npy"
                if path.is_file():
                    found[clip] = np.load(path)
        return found
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run_pose_frames(request: str) -> int:
    """``--pose-frames <request.json>``: a worker of :func:`frames_in_workers` — the frames of the clips it names, each
    saved beside the request as ``<k>.npy``."""
    import json
    import numpy as np
    from ..endfield import actor, clips
    from ..endfield import poses as P
    from ..endfield import store
    task = json.loads(Path(request).read_text(encoding="utf-8"))
    if not task.get("ids"):
        return 0                                    # nothing asked: what --selftest starts, to see a worker run
    with own_store():
        item = store.read_item("chara", task["ident"])
        extra = store.read_extra("chara", task["ident"]) or {}
        skeleton = actor.build_skeleton(item.nodes, task["ident"])
        for number, ident in enumerate(task["ids"]):
            frames = P.frames(clips.read(ident), skeleton, task["bones"], extra.get("paths") or {})
            np.save(Path(request).parent / f"{number}.npy", frames)
    return 0


def write_files(ident: str, blender: Path | None, output: Path, formats: list[str], log: Log,
                shaders=None, poses=None, export_shaders=False, parallel=False, pose_clips=True,
                expressions=True) -> list[Path]:
    """Write the stored *ident* in *formats* and read each back; the files, or why not.

    ``.glb`` and FBX are written here (``interchange``), the ``.blend`` by Blender; the images are inside
    every file, and ``<name>.json`` beside them keeps the bone paths and each material's native values.
    """
    from ..endfield import fbx_export as FX
    from ..endfield import model as M
    from ..endfield import rigscene as RS
    from ..endfield import store
    from .sources import write_own
    extra = store.read_extra("chara", ident) or {}
    # the rig's folder, then the prefab's own code name: the outfits of one character share the model's asset
    # folder (``girl/endminf`` holds endminf, endminfink and endminfunmasked) and would write over each other
    rig = M.folder_of(str(extra.get("model") or "")).rpartition("/")[0] or "other"
    folder = f"{rig}/{character_name(ident)}"
    target = FX.Target(folder, (str(extra.get("model") or ""), ident), f"chara/{ident}", ident=ident,
                       stem=character_name(ident))
    written = []
    if "blend" in formats:
        log(f"以 Blender 建 .blend 並讀回 {folder}（每個模型幾分鐘）…")
        progress.step("以 Blender 建 .blend 並讀回")
        row, _report = FX.write_one(target, output, blender, None, blend=True, shaders=shaders, log=log)
        if not row["success"]:
            raise ExtractorError(f"{ident} 讀回沒有通過：{'; '.join(row['problems'])}")
        Path(row["fbx"]).unlink()           # Blender's FBX was for its read-back; FBX is written below
        written.append(Path(row["fbx"]).with_suffix(".blend"))
    if export_shaders or any(kind in formats for kind in ("glb", "fbx")):
        progress.step("組合場景")
        _scene, document = FX.scene_document(target, output, None, shaders, log,
                                             poses if "glb" in formats or export_shaders else None,
                                             (lambda ids, bones: frames_in_workers(ident, ids, bones, log))
                                             if parallel else None, clips=pose_clips, expressions=expressions)
        try:
            written += write_own(formats, document, output / folder, target.stem, blender, log, export_shaders=export_shaders)
        finally:
            (output / folder / f"{target.stem}.npz").unlink(missing_ok=True)
    shutil.rmtree(output / folder / RS.TEXTURE_DIR, ignore_errors=True)
    shutil.rmtree(output / folder / "baked", ignore_errors=True)        # vfx_bake's maps: inside every file
    return written
