"""A GF2 outfit's display poses, written into its exported scene as animations.

An outfit's clips are named after it (``c_NikketaSSR0102_…``).  Those with ``_slg_`` are the combat model's (run,
fire, cover…); the others are what the outfit is shown in off the battlefield: ``BarrackIdle``, ``Cloth_Idle``,
``EndingIdle``, ``Bedroom_0N_Idle``, and the transitions between them.  Every one is written, the transitions too
(the owner wants every pose, 2026-11-04); the held ones — the clips the game loops (``m_LoopTime``, as Endfield's:
on Nikketa's SSR0102 the eight idles) — come first.  A clip whose transform curves mostly miss the outfit's nodes
(fewer than :data:`BOUND`: the ``…_Gen`` ones bind none, a ``…_Camera`` one moves the camera) is not one of its
poses, nor is a cutscene's clip that pulls the body apart where its camera does not look (:data:`TORN`).

The bundle index already lists every clip (``bundles.scan``: one ``AnimationClip`` container each) but not by name;
:func:`index` names them once, opening only the bundles that hold clips, and again only for a bundle that changed.
A clip is read from its own bundle member (``clips``: its curves need nothing outside it).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import UnityPy

from . import bundles, clips

#: the share of a clip's transform curves that must drive the outfit's own nodes
BOUND = 0.5
#: how far a skeleton bone may move off its place under its parent before a clip is taken for a cutscene's, drawn for
#: its camera and not to be seen whole (``clips.strays``): ``Upgrade_*`` move Clukay SSR01's shoulder blade 1.0 m and
#: Cheyanne SSR0101's neck 0.46 m, Clukay's ``Class_*`` a shoulder blade 0.54 m; every other clip of the two moves its
#: skeleton 0.13 m at most (Cheyanne's ``Class_*``, a knee; Clukay's ``Love_4``, a shoulder 0.11 m)
TORN = 0.4
INDEX = "clip_index.jsonl"


def outfit_of(ident: str) -> str:
    """``c_NikketaSSR0102_slg_skin_48e81872d7`` → ``NikketaSSR0102``: what its clips are named after."""
    match = re.match(r"c_(.+?)_(?:slg_)?skin(?:_[0-9a-f]{10})?$", ident, re.IGNORECASE)
    return match[1] if match else ""


def index(install: Path, bundle_index: Path, out: Path, log=print) -> list[dict]:
    """Every clip by name: ``[{name, path, offset, cab, id}]``, kept in *out* per bundle and its stamp."""
    from ..core import progress
    known = {}
    if out.is_file():
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            known[row["path"]] = row
    wanted = []
    for line in bundle_index.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        held = [(m["offset"], e) for m in row.get("members", []) for e in m["containers"] if e["type"] == "AnimationClip"]
        if held:
            wanted.append((row["path"], row["stamp"], held))
    todo = [w for w in wanted if known.get(w[0], {}).get("stamp") != w[1]]
    if todo:
        log(f"為 {len(todo)} 個 bundle 裡的動作建立名稱索引（第一次要幾分鐘）")
        bundles.configure_unity(install)
    for number, (path, stamp, held) in enumerate(todo, 1):
        progress.report(number, len(todo), "建立動作索引")
        names = []
        by_offset = {}
        for offset, entry in held:
            by_offset.setdefault(offset, []).append(entry)
        try:
            for offset, data in bundles.members(install / path):
                if offset not in by_offset:
                    continue
                env = UnityPy.load(data)
                objects = {}
                for file in env.files.values():
                    for sub in (getattr(file, "files", None) or {}).values():
                        objects.update(getattr(sub, "objects", None) or {})
                    objects.update(getattr(file, "objects", None) or {})
                for entry in by_offset[offset]:
                    obj = objects.get(entry["id"])
                    if obj is not None:
                        names.append({"name": obj.peek_name(), "offset": offset, "cab": entry["cab"], "id": entry["id"]})
        except (ValueError, OSError, EOFError, KeyError) as failure:
            log(f"  {path}：{failure}")
        known[path] = {"path": path, "stamp": stamp, "clips": names}
    if todo:
        out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in known.values()), encoding="utf-8")
    return [dict(clip, path=row["path"]) for row in known.values() for clip in row["clips"]]


def candidates(rows: list[dict], outfit: str) -> list[dict]:
    """The outfit's own clips off the battlefield, by name."""
    prefix = f"c_{outfit}_".lower()
    found = {row["name"]: row for row in rows
             if row["name"].lower().startswith(prefix) and "_slg_" not in row["name"].lower()}
    return [found[name] for name in sorted(found)]


def read(install: Path, row: dict) -> dict:
    """The clip's typetree, from its own bundle member."""
    for offset, data in bundles.members(install / row["path"]):
        if offset != row["offset"]:
            continue
        env = UnityPy.load(data)
        for file in env.files.values():
            for objects in [getattr(file, "objects", None) or {}] + [
                    getattr(sub, "objects", None) or {} for sub in (getattr(file, "files", None) or {}).values()]:
                obj = objects.get(row["id"])
                if obj is not None:
                    return obj.read_typetree()
    raise LookupError(f"{row['name']} is not in {row['path']}")


def bound(tree: dict, paths: dict) -> float:
    """The share of the clip's transform curves that drive one of the outfit's nodes (*paths*)."""
    _times, by_hash = clips.transforms(tree)
    if not by_hash:
        return 0.0
    ours = {clips.path_hash(path) for path in paths.values()}
    return sum(1 for value in by_hash if value in ours) / len(by_hash)


def add(document_path: Path, scene, nodes: dict, extra: dict, ident: str, install: Path, rows: list[dict],
        log=print) -> int:
    """The outfit's display poses, held ones first, added to the document at *document_path* as animations
    (``endfield.poses.add`` writes the same fields); how many."""
    import numpy as np
    paths = extra.get("paths") or {}
    held = []
    for row in candidates(rows, outfit_of(ident)):
        try:
            tree = read(install, row)
        except (LookupError, ValueError, OSError) as failure:
            log(f"  {row['name']}：{failure}")
            continue
        try:
            if bound(tree, paths) < BOUND:
                continue
        except clips.ClipError as failure:
            log(f"  {row['name']}：{failure}")
            continue
        held.append((row["name"], tree))
    held.sort(key=lambda item: not item[1]["m_MuscleClip"].get("m_LoopTime"))
    if not held:
        return 0
    from ..endfield.poses import labels
    document_path = Path(document_path)
    document = json.loads(document_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(document_path.parent / document["arrays"]))
    bones = [bone.node for bone in scene.bones]
    rest = np.stack([bone.rest for bone in scene.bones]).astype(np.float64)
    parents = [bone.parent for bone in scene.bones]
    document["animations"] = []
    for (name, tree), label in zip(held, labels([n for n, _t in held])):
        frames = clips.poses(tree, nodes, paths, bones)
        moved, bone = clips.strays(frames, rest, parents, bones)
        if moved > TORN:
            log(f"  {label}：過場動作，{bones[bone]} 離開原位 {moved * 100:.0f} cm（只給鏡頭看的部分），不寫")
            continue
        log(f"寫入姿勢 {label}（{len(frames)} 格）")
        arrays[f"anim{len(document['animations'])}"] = frames
        document["animations"].append({"name": label, "clip": name, "rate": float(tree["m_SampleRate"]) or 30.0,
                                       "frames": len(frames)})
    if not document["animations"]:
        return 0
    np.savez(document_path.parent / document["arrays"], **arrays)
    document_path.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(document["animations"])
