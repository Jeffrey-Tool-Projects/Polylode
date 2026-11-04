"""Making an install's maps in checked batches (``endfield/mapbuild.py``), with a stand-in for AnimeStudio."""
import json
import sys
import textwrap
from pathlib import Path

import pytest

from polylode.endfield import mapbuild as MB
from polylode.endfield.cabmap import CabIndex
from polylode.endfield.pipeline import sources_from_export

#: the stand-in: ``vfs-index`` lists two chunks; a map run writes the batch's slots (and a stale one when
#: given slot 999), or runs out of memory when given more than three
FAKE = textwrap.dedent("""
    import json, struct, sys
    from pathlib import Path
    sys.path.insert(0, {root!r})
    from polylode.endfield.mapbuild import write_cabmap
    from polylode.endfield.cabmap import CabEntry
    args = sys.argv[1:]
    if args[0] == "vfs-index":
        source = Path(args[2])
        out = Path(args[args.index("-o") + 1])
        rows = [dict(recordType="header", generatedAtEpoch=__import__("time").time_ns())]
        for n, (name, count) in enumerate((("a.chk", 2), ("b.chk", 5))):
            path = source / "VFS" / name
            rows.append(dict(recordType="chunk", chunkId=n, absolutePath=str(path), fileName=name, blockName="bundle",
                             length=path.stat().st_size, fileCount=count, exists=True,
                             contentMd5=bytes.fromhex(__import__("hashlib").md5(path.read_bytes()).hexdigest())[::-1].hex()))
            rows += [dict(recordType="file", chunkId=n, offset=100 * k, length=10) for k in range(count)]
        rows.append(dict(recordType="summary", fileCount=7, chunkCount=2, missingChunkCount=0, missingBlockCount=0))
        out.write_text("\\n".join(json.dumps(r) for r in rows) + "\\n", encoding="utf-8")
        sys.exit(0)
    source, target = Path(args[0]), Path(args[1])
    slots = json.loads(Path(args[args.index("--filter_data") + 1]).read_text(encoding="utf-8"))
    if any(s["Offset"] == 999 for s in slots):      # a superseded slot read as well (no filter applied)
        slots = slots + [dict(slots[0], Offset=777)]
    if len(slots) > 3:
        print("System.OutOfMemoryException"); sys.exit(1)
    (target / "cab_map").mkdir(parents=True, exist_ok=True)
    root = source.parent.parent
    entries = {{f"cab-{{source.stem}}-{{s['Offset']}}": CabEntry(f"CAB-{{source.stem}}-{{s['Offset']}}", root, source, s["Offset"], ())
               for s in slots}}
    write_cabmap(target / "cab_map" / "current_shard.bin", root, entries)
    assets = [dict(Source=str(source), Offset=s["Offset"], Container=f"assets/{{source.stem}}/{{s['Offset']}}.prefab",
                   Name="x", PathID=1, Type="GameObject") for s in slots]
    (target / "current_shard.json").write_text(json.dumps(dict(GameType="ArknightsEndfield", AssetEntries=assets)),
                                               encoding="utf-8")
    print("0 collisions found")
""")


@pytest.fixture
def install(tmp_path):
    game = tmp_path / "Endfield_Data"
    for layer, name in (("StreamingAssets", "a.chk"), ("Persistent", "b.chk")):
        (game / layer / "VFS").mkdir(parents=True)
    # the downloaded layer's index lists chunks of both layers: here both live under Persistent for the stand-in
    for name in ("a.chk", "b.chk"):
        (game / "Persistent" / "VFS" / name).write_bytes(name.encode() * 100)
    script = tmp_path / "fake_cli.py"
    script.write_text(FAKE.format(root=str(Path(__file__).resolve().parents[1])), encoding="utf-8")
    exe = tmp_path / "fake_cli.cmd"
    exe.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    return game, exe


def test_the_maps_are_made_in_batches_halved_when_out_of_memory_and_read_back(install, tmp_path):
    game, exe = install
    log = []
    summary = MB.build(exe, game, tmp_path / "work", tmp_path / "meta", log.append)
    assert summary["slots"] == 7
    assert any("記憶體不足" in line for line in log)                 # b.chk's five slots split into 2 + 3
    plan = json.loads((tmp_path / "work" / "plan.json").read_text(encoding="utf-8"))
    assert sorted(len(t["offsets"]) for t in plan["tasks"]) == [2, 2, 3]
    sources = sources_from_export(exe, tmp_path / "meta")
    cabs = CabIndex.load(sources.cab_maps)
    assert len(cabs) == 7 and cabs.outside(game) == 0
    entry = cabs.entry("CAB-b-400")
    assert entry.offset == 400 and Path(entry.source).name == "b.chk"
    receipts = {p: p.stat().st_mtime_ns for p in (tmp_path / "work" / "shards").glob("*/receipt.json")}
    again = MB.build(exe, game, tmp_path / "work", tmp_path / "meta", log.append)   # resumed: nothing re-run
    assert again["slots"] == 7
    assert {p: p.stat().st_mtime_ns for p in receipts} == receipts


def test_a_batch_whose_map_holds_other_slots_or_a_changed_chunk_is_refused(install, tmp_path):
    game, exe = install
    work = tmp_path / "work"
    work.mkdir()
    rows = MB.read_index(MB.run_index(exe, game, work, print))
    tasks = MB.plan(rows)
    MB.check_partition(rows, tasks)
    with pytest.raises(MB.MapBuildError, match="exactly"):
        MB.check_partition(rows, tasks[:-1])
    small = next(t for t in tasks if len(t["offsets"]) == 2)
    wrong = dict(small, offsets=[0, 999])
    with pytest.raises(MB.MapBuildError, match="other slots"):
        MB.run_shard(exe, wrong, work, {"index_sha256": "x", "cli_sha256": "y"})
    Path(small["absolutePath"]).write_bytes(b"changed" * 50)
    with pytest.raises(MB.MapBuildError, match="size|MD5"):
        MB.run_shard(exe, small, work, {"index_sha256": "x", "cli_sha256": "y"})
    with pytest.raises(MB.MapBuildError, match="more memory"):
        MB.split([dict(small, offsets=[0])], small["key"])


def test_added_asset_types_rebuild_cached_maps(install, tmp_path, monkeypatch):
    game, exe = install
    work = tmp_path / "work"
    work.mkdir()
    rows = MB.read_index(MB.run_index(exe, game, work, lambda line: None))
    task = next(t for t in MB.plan(rows) if len(t["offsets"]) == 2)
    identity = {"index_sha256": "x", "cli_sha256": "y"}
    current = MB.TYPES
    assert "MonoBehaviour" in current
    monkeypatch.setattr(MB, "TYPES", tuple(t for t in current if t != "MonoBehaviour"))
    assert MB.run_shard(exe, task, work, identity) == "verified"
    assert MB.run_shard(exe, task, work, identity) == "kept"
    monkeypatch.setattr(MB, "TYPES", current)
    assert MB.run_shard(exe, task, work, identity) == "verified"
    assert MB.run_shard(exe, task, work, identity) == "kept"
