"""A character's display poses, written into its exported scene as animations.

The game's character screens play an actor's own clips from its ``animations/ui`` folder: transitions between the
screens (``ui_overview_to_equip``) and the poses held on each (``ui_overview_loop``, ``ui_relax_loop``,
``uiteam_idle_01``…).  Every one of them is written, the transitions too (the owner wants every pose, 2026-11-04);
the held ones — the clips the game loops, the clip's own ``m_LoopTime`` (:attr:`clips.Stored.loop`), not its name:
on endminm 10 of the 24 ``ui`` clips, every ``*_loop`` and ``uiteam_idle_*`` — come first, so a viewer opens on a
pose rather than a move between two.

Each is played on the item's own skeleton the way the runtime plays it (:func:`actor.build_skeleton`,
:func:`clips.to_animation`, ``Skeleton.evaluate``), and every frame is kept as the exported bones' frames: the
transform the runtime skins with (:func:`rigscene.skinned`: the evaluated world after the inverse of the rest world,
on the scene's ``bone_rest``, which :func:`rigscene.rest_matrix` makes from the same nodes — at rest the two agree
exactly, 0 on endminm's 400 bones).  ``rest_matrix`` of the evaluated world alone took a clip's scale out: the face
bones a clip scales (jsspsi's lips to 0.35, her pupils to 0.16 and eye highlights to 0.27; endminf's teeth to 0.32
and tongue to 0.41) kept their rest size, the face 8–12 mm off the game's.  These clips drive the trunk and limbs
through their transform tracks (``Bip001_Pelvis``, ``Bip001_Spine``…: 321 of 332 tracks find a bone), so no
humanoid retargeting is involved.

The clips are decoded from the installation once (:meth:`pipeline.Pipeline.convert_clips`, through the exporter's
own ACL library) into the store, where later exports find them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import actor, clips
from . import rigscene as RS

#: the clips' folder below an actor's ``animations``
GROUP = "ui"


def folder_of(model: str) -> str:
    """Where *model*'s own display clips are filed (``boy/endminm/animations/ui/``); empty when it is not an
    actor's model."""
    if not model.startswith(clips.ACTOR_ROOT) or "/models/" not in model:
        return ""
    return f"{model[len(clips.ACTOR_ROOT):].split('/models/')[0]}/animations/{GROUP}/"


def stored(model: str) -> list[str]:
    """The ids of *model*'s own display clips already in the store, by name."""
    prefix = folder_of(model)
    return sorted(key for key in clips.index() if prefix and key.startswith(prefix)) if prefix else []


def ensure(model: str, fetch, log=print) -> list[str]:
    """*model*'s display clips, decoded from the installation when the store has none; *fetch* gives the
    pipeline (opened only then)."""
    found = stored(model)
    if found or not folder_of(model) or fetch is None:
        return found
    pipeline = fetch()
    prefix = clips.ACTOR_ROOT + folder_of(model)
    rows = [row for row in pipeline.clip_rows([model], [GROUP]) if row.container.startswith(prefix)]
    if not rows:
        return []
    log(f"從安裝解碼 {len(rows)} 個介面動作")
    result = pipeline.convert_clips(rows)
    for ident, why in result.failed.items():
        log(f"  {ident}：{why}")
    return stored(model)


def labels(names: list[str]) -> list[str]:
    """The clips' names without the prefix they all share (``A_actor_endminm_``): what a viewer lists."""
    if len(names) < 2:
        return list(names)
    shared = os.path.commonprefix(names)
    shared = shared[:shared.rfind("_") + 1]
    return [name[len(shared):] or name for name in names]


def frames(clip: clips.Stored, skeleton, nodes: list[str], paths: dict) -> np.ndarray:
    """*clip* played on *skeleton*: the frames of the bones on *nodes*, ``(frames, bones, 4, 4)`` in the scene's
    space (as ``bone_rest``)."""
    animation, _matched = clips.to_animation(clip, skeleton, paths)
    rows = [skeleton.index[node] for node in nodes]
    rest = skeleton.evaluate(None)
    out = np.empty((clip.frames, len(nodes), 4, 4), dtype=np.float32)
    for frame in range(clip.frames):
        world = skeleton.evaluate(animation.sample(frame / clip.rate))
        out[frame] = np.stack([RS.skinned(world[row], rest[row]) for row in rows])
    return out


def add(document_path: Path, scene: RS.Scene, nodes: dict, extra: dict, fetch=None, log=print) -> int:
    """The display clips of the scene's model, held ones first, added to the document at *document_path* as
    animations (``document["animations"]``: name, rate, frames; ``arrays["anim<k>"]`` their bone frames); how many."""
    model = str(extra.get("model") or "")
    ids = ensure(model, fetch, log)
    held = sorted((clips.read(ident) for ident in ids), key=lambda clip: not clip.loop)
    held = [clip for clip in held if clip.frames >= 1]
    if not held:
        return 0
    document_path = Path(document_path)
    document = json.loads(document_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(document_path.parent / document["arrays"]))
    skeleton = actor.build_skeleton(nodes, scene.name)
    bones = [bone.node for bone in scene.bones]
    paths = extra.get("paths") or {}
    document["animations"] = []
    for k, (clip, label) in enumerate(zip(held, labels([c.name for c in held]))):
        log(f"寫入姿勢 {label}（{clip.frames} 格）")
        arrays[f"anim{k}"] = frames(clip, skeleton, bones, paths)
        document["animations"].append({"name": label, "clip": clip.name, "rate": clip.rate, "frames": clip.frames})
    np.savez(document_path.parent / document["arrays"], **arrays)
    document_path.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(held)
