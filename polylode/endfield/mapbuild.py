"""The CAB and asset maps of an Endfield install, made from the install itself with AnimeStudio (§7.1070, §7.1452).

A single AnimeStudio map pass over the current game runs out of memory, and
without the current slots as ``--filter_data`` it also reads bundle slots
an update has replaced.  So the maps are made the way §7.1070 made them:

1. **index** — ``AnimeStudio vfs-index`` over the downloaded layer
   (``Persistent``, the installed one as its fallback; ``StreamingAssets``
   alone when nothing was downloaded) lists every current bundle slot: the
   chunk file it is in, its offset and length (:func:`run_index`).
2. **plan** — the slots of each chunk in batches of at most
   :data:`BATCH_BYTES` and :data:`BATCH_FILES` (:func:`plan`).
3. **shards** — each batch through AnimeStudio's map stage with exactly
   its slots as ``--filter_data``; a batch that runs out of memory is split
   in half and run again (:func:`run_shard`).  A shard is kept only when
   its log has no error and no CAB collision, its CAB map holds exactly
   the batch's slots and nothing outside its chunk, and every asset entry
   is from those slots; its receipt binds the chunk's size, time and MD5.
4. **merge** — every shard, after checking they cover each chunk's slots
   exactly once and no two give one CAB name different places, into the
   two layers' ``meta/cab_map/endfield_<layer>_assets.bin`` and
   ``meta/<Layer>/asset_map/endfield_<layer>_assets.json``
   (``pipeline.sources_from_export``'s layout), each read back (:func:`merge`).

Work is resumable: a verified shard of an unchanged chunk is not run again.
Moved from the dev scripts of the 1070 extraction (``probe_current_endfield``,
``build_effective_maps``, ``merge_current_maps``) without changing what they
check.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import shutil
import struct
import subprocess
from contextlib import ExitStack
from pathlib import Path

from ..core import progress
from .assetmap import iter_entries
from .cabmap import read_cabmap

LAYERS = ("StreamingAssets", "Persistent")
GAME = "ArknightsEndfield"
#: a batch's compressed bytes and file count (§7.1070's first plan; a batch out of memory is halved)
BATCH_BYTES, BATCH_FILES = 512 * (1 << 20), 10000
TYPES = ("AssetBundle:Parse", "GameObject", "Animator", "Mesh", "Avatar", "Material", "Shader", "Texture2D",
         "AnimationClip")
#: what :func:`run_shard` returns when AnimeStudio ran out of memory on the batch
NEEDS_SPLIT = "needs split"


class MapBuildError(RuntimeError):
    """The maps cannot be made as checked; the reason says what failed."""


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path: Path, value) -> None:
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def data_layers(game: Path) -> list[str]:
    """The install's layers that hold a VFS, installed first."""
    return [layer for layer in LAYERS if (Path(game) / layer / "VFS").is_dir()]


# -- 1. the index ---------------------------------------------------------------------------------------------

def run_index(exe: Path, game: Path, work: Path, log=print) -> Path:
    """``bundle_vfs_index.jsonl`` of *game* (``Endfield_Data``), checked to be complete; its path."""
    layers = data_layers(game)
    if not layers:
        raise MapBuildError(f"{game} has no StreamingAssets/VFS or Persistent/VFS")
    layer = layers[-1]                                  # the downloaded layer, the installed one beneath it
    output = Path(work) / "bundle_vfs_index.jsonl"
    argv = [str(exe), "vfs-index", "-s", str(Path(game) / layer)]
    if len(layers) == 2:
        argv += ["--fallback-assets", str(Path(game) / layers[0])]
    argv += ["-b", "bundle", "-b", "initial-bundle", "--jsonl", "-o", str(output)]
    log(f"列出 {layer} 的現行資料槽…")
    progress.step("列出現行資料槽")
    done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if done.returncode != 0 or not output.is_file():
        raise MapBuildError(f"vfs-index exit {done.returncode}: {(done.stdout + done.stderr)[-2000:]}")
    counts, last = collections.Counter(), None
    with output.open(encoding="utf-8-sig") as stream:
        for line in stream:
            last = json.loads(line)
            counts[last["recordType"]] += 1
    if not last or last["recordType"] != "summary" or last["fileCount"] != counts["file"] \
            or last["chunkCount"] != counts["chunk"] or last["missingChunkCount"] or last["missingBlockCount"]:
        raise MapBuildError(f"the index is incomplete: {last}, {dict(counts)}")
    return output


# -- 2. the plan ------------------------------------------------------------------------------------------------

def read_index(index: Path) -> list[dict]:
    """Each chunk with slots: its record with ``offsets`` and ``slot_lengths``, smallest first."""
    chunks, last = {}, None
    with Path(index).open(encoding="utf-8-sig") as stream:
        for line in stream:
            last = json.loads(line)
            if last["recordType"] == "chunk":
                chunks[last["chunkId"]] = {**last, "offsets": [], "slot_lengths": []}
            elif last["recordType"] == "file":
                chunks[last["chunkId"]]["offsets"].append(last["offset"])
                chunks[last["chunkId"]]["slot_lengths"].append(last["length"])
    if last is None or last["recordType"] != "summary" or last["missingChunkCount"]:
        raise MapBuildError("the index does not end in a summary without missing chunks")
    rows = sorted((r for r in chunks.values() if r["offsets"]), key=lambda r: (r["length"], r["chunkId"]))
    for row in rows:
        if not row["exists"] or len(row["offsets"]) != row["fileCount"] or len(set(row["offsets"])) != len(row["offsets"]):
            raise MapBuildError(f"chunk {row['fileName']}: missing, or its slots do not add up")
        row["key"] = row["blockName"] + "_" + Path(row["fileName"]).stem
    return rows


def _task(row: dict, offsets: list[int], parent: str | None = None) -> dict:
    key = (parent or row["key"]) + "_part_" + hashlib.sha256(json.dumps(offsets).encode()).hexdigest()[:12]
    return {**{k: v for k, v in row.items() if k not in ("slot_lengths", "offsets")},
            "parent_key": parent or row["key"], "key": key, "offsets": offsets}


def plan(rows: list[dict]) -> list[dict]:
    """The batches: each chunk's slots in offset order, cut at :data:`BATCH_BYTES` or :data:`BATCH_FILES`."""
    tasks = []
    for row in rows:
        group, total = [], 0
        for offset, length in sorted(zip(row["offsets"], row["slot_lengths"])):
            if group and (total + length > BATCH_BYTES or len(group) >= BATCH_FILES):
                tasks.append(_task(row, group))
                group, total = [], 0
            group.append(offset)
            total += length
        if group:
            tasks.append(_task(row, group))
    return tasks


def split(tasks: list[dict], key: str) -> list[dict]:
    """*tasks* with batch *key* in two halves; a batch of one slot cannot be split."""
    at = next(i for i, task in enumerate(tasks) if task["key"] == key)
    task = tasks[at]
    if len(task["offsets"]) < 2:
        raise MapBuildError(f"one bundle slot of {task['fileName']} needs more memory than this machine has")
    middle = len(task["offsets"]) // 2
    halves = [_task(task, offsets, task["parent_key"]) for offsets in (task["offsets"][:middle], task["offsets"][middle:])]
    return tasks[:at] + halves + tasks[at + 1:]


def check_partition(rows: list[dict], tasks: list[dict]) -> None:
    """Every chunk's slots in exactly one batch."""
    expected = {r["absolutePath"]: set(r["offsets"]) for r in rows}
    covered = collections.defaultdict(set)
    for task in tasks:
        selected = set(task["offsets"])
        if covered[task["absolutePath"]] & selected:
            raise MapBuildError(f"two batches share slots of {task['fileName']}")
        covered[task["absolutePath"]] |= selected
    if dict(covered) != expected:
        raise MapBuildError("the batches do not cover the index's slots exactly")


# -- 3. the shards --------------------------------------------------------------------------------------------

def _log_ok(text: str) -> bool:
    collisions = re.findall(r"(\d+) collisions found", text)
    return "[Error]" not in text and "Unable to load object" not in text and bool(collisions) \
        and all(int(n) == 0 for n in collisions)


def _chunk_md5(source: Path, expected: str, cache: Path) -> None:
    """The chunk's content MD5 against the index's (byte-reversed there), once per size and time."""
    stat = source.stat()
    identity = {"source": str(source), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "md5": expected}
    if cache.is_file() and _read(cache) == identity:
        return
    digest = hashlib.md5()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise MapBuildError(f"{source.name}: its content is not what the index says (MD5)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    _write(cache, identity)


def run_shard(exe: Path, task: dict, work: Path, identity: dict) -> str:
    """Make and check one batch's maps under ``work/shards/<key>``; ``"verified"``, ``"kept"`` or :data:`NEEDS_SPLIT`."""
    target = Path(work) / "shards" / task["key"]
    source = Path(task["absolutePath"])
    stat = source.stat()
    own = {**identity, "source": str(source), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    receipt, assets, cab = target / "receipt.json", target / "current_shard.json", target / "cab_map" / "current_shard.bin"
    if receipt.is_file():
        previous = _read(receipt)
        if previous.get("identity") == own and sha(assets) == previous["asset_map_sha256"] \
                and sha(cab) == previous["cab_map_sha256"]:
            return "kept"
    if stat.st_size != task["length"]:
        raise MapBuildError(f"{source.name} is not the size the index says")
    _chunk_md5(source, bytes.fromhex(task["contentMd5"])[::-1].hex(), Path(work) / "chunk_md5" / f"{task['parent_key']}.json")
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    slots = target / "effective_slots.json"
    slots.write_text(json.dumps([{"Source": str(source), "Offset": offset, "Name": "", "PathID": 0,
                                  "Type": "AssetBundle"} for offset in task["offsets"]]), encoding="utf-8")
    argv = [str(exe), str(source), str(target), "--game", GAME, "--logger_flags", "Info", "Warning", "Error",
            "--cab_map_dir", str(target / "cab_map"), "--map_op", "Both", "--map_type", "JSON",
            "--map_name", "current_shard", "--filter_data", str(slots), "--types", *TYPES]
    done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    text = done.stdout + done.stderr
    (target / "log.txt").write_text(text, encoding="utf-8")
    if "OutOfMemoryException" in text or "out of memory" in text.lower():
        return NEEDS_SPLIT
    if done.returncode != 0 or not _log_ok(text) or not assets.is_file() or not cab.is_file():
        raise MapBuildError(f"{task['key']}: AnimeStudio exit {done.returncode}, or errors or CAB collisions: {text[-1500:]}")
    entries = read_cabmap(cab)
    offsets = {entry.offset for entry in entries.values()}
    if offsets != set(task["offsets"]):
        raise MapBuildError(f"{task['key']}: the CAB map holds other slots than the batch "
                            f"(stale {sorted(offsets - set(task['offsets']))[:5]}, unread {sorted(set(task['offsets']) - offsets)[:5]})")
    if any(Path(str(entry.source)).resolve() != source.resolve() for entry in entries.values()):
        raise MapBuildError(f"{task['key']}: a CAB from another chunk")
    count = 0
    for item in iter_entries(assets):
        if Path(item["Source"]).resolve() != source.resolve() or int(item["Offset"]) not in set(task["offsets"]):
            raise MapBuildError(f"{task['key']}: an asset entry from outside the batch")
        count += 1
    _write(receipt, {"identity": own, "slots": len(offsets), "cabs": len(entries), "asset_entries": count,
                     "asset_map_sha256": sha(assets), "cab_map_sha256": sha(cab)})
    return "verified"


# -- 4. the merge ---------------------------------------------------------------------------------------------

def _string(stream, value: str) -> None:
    data = value.encode("utf-8")
    n = len(data)
    while n >= 128:
        stream.write(bytes([(n & 127) | 128]))
        n >>= 7
    stream.write(bytes([n]))
    stream.write(data)


def write_cabmap(path: Path, root: Path, entries: dict) -> None:
    """A CAB map (``cabmap.read_cabmap``'s layout): the layer's root, then each entry by name."""
    with Path(path).open("wb") as stream:
        _string(stream, str(root))
        stream.write(struct.pack("<i", len(entries)))
        for _name, entry in sorted(entries.items()):
            _string(stream, entry.name)
            _string(stream, str(Path(entry.source).resolve().relative_to(Path(root).resolve())))
            stream.write(struct.pack("<qi", entry.offset, len(entry.deps)))
            for dep in entry.deps:
                _string(stream, dep)


def merge(game: Path, tasks: list[dict], rows: list[dict], work: Path, meta: Path) -> dict:
    """The two layers' maps under *meta* from every verified shard; what was written."""
    check_partition(rows, tasks)
    game = Path(game).resolve()
    cabs = {layer: {} for layer in LAYERS}
    every: dict = {}
    shards = []
    for task in tasks:
        folder = Path(work) / "shards" / task["key"]
        receipt = _read(folder / "receipt.json")
        source = Path(task["absolutePath"]).resolve()
        layer = next((layer for layer in LAYERS if source.is_relative_to(game / layer)), None)
        if layer is None:
            raise MapBuildError(f"{source} is outside {game}")
        for entry in read_cabmap(folder / "cab_map" / "current_shard.bin").values():
            name = entry.name.casefold()
            place = (Path(str(entry.source)).resolve(), entry.offset, entry.deps)
            seen = every.get(name)
            if seen is not None and (Path(str(seen.source)).resolve(), seen.offset, seen.deps) != place:
                raise MapBuildError(f"CAB {entry.name} has two places in the current install")
            every[name] = cabs[layer][name] = entry
        shards.append((task, receipt, folder / "current_shard.json", layer, source))
    written = {}
    with ExitStack() as stack:
        handles, first = {}, {layer: True for layer in LAYERS}
        for layer in LAYERS:
            stem = f"endfield_{layer.lower()}_assets"
            asset = meta / layer / "asset_map" / f"{stem}.json"
            asset.parent.mkdir(parents=True, exist_ok=True)
            handles[layer] = stack.enter_context(asset.with_suffix(".tmp").open("w", encoding="utf-8"))
            handles[layer].write('{"GameType":"ArknightsEndfield","AssetEntries":[\n')
        for task, receipt, asset, layer, source in shards:
            count = 0
            for item in iter_entries(asset):
                if not first[layer]:
                    handles[layer].write(",\n")
                handles[layer].write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
                first[layer] = False
                count += 1
            if count != receipt["asset_entries"]:
                raise MapBuildError(f"{task['key']}: {count} asset entries for {receipt['asset_entries']}")
        for handle in handles.values():
            handle.write("\n]}\n")
    for layer in LAYERS:
        stem = f"endfield_{layer.lower()}_assets"
        cab = meta / "cab_map" / f"{stem}.bin"
        cab.parent.mkdir(parents=True, exist_ok=True)
        temporary = cab.with_suffix(".tmp")
        write_cabmap(temporary, game / layer, cabs[layer])
        reopened = read_cabmap(temporary)
        if len(reopened) != len(cabs[layer]) or any(
                (Path(str(e.source)).resolve(), e.offset, e.deps)
                != (Path(str(cabs[layer][n.casefold()].source)).resolve(), cabs[layer][n.casefold()].offset,
                    cabs[layer][n.casefold()].deps) for n, e in reopened.items()):
            raise MapBuildError(f"the {layer} CAB map does not read back as written")
        temporary.replace(cab)
        asset = meta / layer / "asset_map" / f"{stem}.json"
        asset.with_suffix(".tmp").replace(asset)
        written[layer] = {"cabs": len(cabs[layer]), "cab_map": str(cab), "asset_map": str(asset)}
    return written


# -- all of it ------------------------------------------------------------------------------------------------

def build(exe: Path, game: Path, work: Path, meta: Path, log=print) -> dict:
    """The maps of *game* in *meta*, made (or resumed) in *work*; the merge's summary, written to
    ``meta/build.json`` with the index and AnimeStudio it was made from."""
    exe, work, meta = Path(exe), Path(work), Path(meta)
    work.mkdir(parents=True, exist_ok=True)
    index = run_index(exe, game, work, log)
    identity = {"index_sha256": sha(index), "cli_sha256": sha(exe.with_suffix(".dll") if exe.with_suffix(".dll").is_file() else exe)}
    rows = read_index(index)
    plan_path = work / "plan.json"
    saved = _read(plan_path) if plan_path.is_file() else {}
    tasks = saved["tasks"] if {k: saved.get(k) for k in identity} == identity else plan(rows)
    check_partition(rows, tasks)
    _write(plan_path, {**identity, "tasks": tasks})
    done = 0
    while done < len(tasks):
        progress.report(done, len(tasks), "建立資產對照表")
        task = tasks[done]
        outcome = run_shard(exe, task, work, identity)
        if outcome == NEEDS_SPLIT:
            tasks = split(tasks, task["key"])
            _write(plan_path, {**identity, "tasks": tasks})
            log(f"記憶體不足，{task['fileName']} 的這一批分成兩半重跑")
            continue
        done += 1
        if done % 10 == 0 or done == len(tasks):
            log(f"資產對照表：{done} / {len(tasks)} 批")
    progress.report(len(tasks), len(tasks), "建立資產對照表")
    progress.step("合併資產對照表")
    summary = {**identity, "game": str(Path(game).resolve()), "batches": len(tasks),
               "slots": sum(len(r["offsets"]) for r in rows), "layers": merge(game, tasks, rows, work, meta)}
    _write(meta / "build.json", summary)
    return summary
