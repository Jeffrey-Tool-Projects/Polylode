"""Endfield animation clips: decoded once, stored in the store, played on an actor.

A clip binds each curve to a transform by the CRC-32 of its path below the
Animator's root, and keeps its bone curves in one ACL buffer
(:mod:`.acl`).  What was measured about the layout (§7.564):

* **ACL track *k* is the *k*-th rotation binding** (``Transform``, attribute 2)
  in ``genericBindings`` order — every animated bone has one; on a 273-track
  clip the decoded translations sit within 0.01 mm of the prefab's rest pose
  (median) under that pairing and nowhere near it under any other;
* a decoded sample is rotation ``xyzw``, translation, scale — local to the
  parent, in Unity's space.

A stored clip is addressed by the hashes, not by names: it plays on any actor
whose nodes sit at those paths, which is what lets a body type's shared clips
(``…/actor/girl/common/animations``) drive every character of that type.
Tracks are split into the ones that move and the ones that hold one value, so
a walk cycle does not store 273 bones × 60 frames of the same number.

**The body is not in those tracks.**  The transform buffer holds the bones the
humanoid rig does not own — hair, fingers, twist and corrective bones; the
trunk and limbs are Unity humanoid *muscle* curves in the clip's scalar buffer
(Animator bindings whose ``attribute`` is the muscle index: motion 0‥6, root
7‥13, the four limb goals, then the body's degrees of freedom and the
fingers').  They are decoded and stored here by that index, with the
root-motion buffer beside them; :func:`to_animation` turns them into the
trunk's and limbs' rotations with the actor's own Avatar definition
(:mod:`.muscle`, §7.568) when it is given one, and poses only the transform
buffer's bones when it is not.
"""
from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..anim.clip import AnimationClip
from ..core import math3d as m3
from ..core.paths import write_json_atomic
from . import acl, muscle, store

#: Where the clips sit in the store, and their index.
FOLDER = "clips"
INDEX = "index.json"
#: The asset folder every actor's art lives under.
ACTOR_ROOT = "assets/beyond/arts/entity/actor/"
#: A channel that never leaves its first value by more than this is constant.
STILL = 1e-6
_CHANNELS = (("rot", slice(0, 4)), ("pos", slice(4, 7)), ("scl", slice(7, 10)))
#: Unity → this project (X mirrored): quaternion (x, −y, −z, w), vector (−x, y, z).
_FLIP = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 dtype=np.float32)
_ROTATION = 2          # ``genericBindings[].attribute`` of a rotation curve
#: ``genericBindings[].customType`` of a humanoid muscle curve.
MUSCLE = 8


class ClipError(ValueError):
    """A staged clip whose buffer and bindings do not describe each other."""


class NotAcl(ClipError):
    """A clip kept in Unity's own streamed / dense / constant form.

    About one in sixty of the actors' clips (cameras, cloth open/close, the
    ``curve_collection`` clips, a few battle moves) has an empty ACL block and
    its curves in ``m_MuscleClip.m_Clip`` instead.  This module does not decode
    that form; such a clip is reported as what it is rather than as a failure.
    """


@dataclass
class Stored:
    """A clip as stored: per-track path hashes and the three channels."""
    name: str
    rate: float
    frames: int
    loop: bool
    hashes: np.ndarray                 # (tracks,) uint32
    #: channel → (indices of constant tracks, their value, indices of moving
    #: tracks, their samples ``(frames, n, width)``)
    channels: dict
    events: list
    #: the Animator curves: ``(attribute, customType)`` per scalar track
    #: (:data:`MUSCLE` ⇒ the attribute is a muscle index) and their samples
    #: ``(samples, tracks)`` — ``None`` when the clip has no scalar buffer.
    #: The sample count is that buffer's own, which may be one off ``frames``.
    float_keys: np.ndarray | None = None
    floats: np.ndarray | None = None
    #: the root-motion buffer's samples ``(samples, tracks)``, or ``None``
    root: np.ndarray | None = None

    @property
    def duration(self) -> float:
        return max(self.frames - 1, 0) / self.rate


def folder() -> Path:
    return store.root() / FOLDER


def index() -> dict:
    """``{clip id: {name, container, frames, rate, tracks}}`` of the stored clips."""
    path = folder() / INDEX
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def clip_id(container: str, name: str) -> str:
    """Where a clip is filed: its asset folder below the actor root, and its name.

    ``girl/pelica/animations/3c/A_actor_pelica_walk_loop``.
    """
    below = container[len(ACTOR_ROOT):] if container.startswith(ACTOR_ROOT) \
        else container
    return f"{below.rsplit('/', 1)[0]}/{name}"


def for_model(model: str) -> list[str]:
    """The stored clips made for the item whose model asset is *model*.

    A clip is filed by its asset folder (:func:`clip_id`) and an item's model
    sits in ``…/<name>/models/``: the clips filed under ``…/<name>/`` are its
    own and come first.  An actor's folder is ``<body type>/<name>``, and the
    clips under ``<body type>/common/`` are shared by every actor of that body
    type; they follow.  A model that is not in a ``models`` folder has none.
    """
    if "/models/" not in model:
        return []
    actor = model.startswith(ACTOR_ROOT)
    own = (model[len(ACTOR_ROOT):] if actor else model).split("/models/")[0]
    listed = sorted(index())
    out = [key for key in listed if key.startswith(own + "/")]
    shared = f"{own.split('/')[0]}/common"
    if actor and shared != own:
        out += [key for key in listed if key.startswith(shared + "/")]
    return out


def path_hash(path: str) -> int:
    """Unity's binding hash of a transform path."""
    return zlib.crc32(path.encode("utf-8"))


# -- import side ---------------------------------------------------------------
def track_hashes(doc: dict) -> np.ndarray:
    """The path hash of each ACL track: the rotation bindings, in order."""
    bindings = (doc.get("m_ClipBindingConstant") or {}).get("genericBindings") or []
    return np.array([int(b["path"]) for b in bindings
                     if b.get("typeID") == "Transform"
                     and int(b.get("attribute", 0)) == _ROTATION], dtype=np.uint32)


def split(samples: np.ndarray) -> dict:
    """``(frames, tracks, 10)`` → the constant and the moving tracks per channel."""
    out = {}
    for name, cols in _CHANNELS:
        data = samples[:, :, cols]
        still = np.abs(data - data[:1]).max(axis=(0, 2)) <= STILL
        out[name] = (np.flatnonzero(still).astype(np.uint16), data[0, still],
                     np.flatnonzero(~still).astype(np.uint16), data[:, ~still])
    return out


def float_keys(doc: dict) -> np.ndarray:
    """``(attribute, customType)`` of each scalar track: the Animator bindings."""
    bindings = (doc.get("m_ClipBindingConstant") or {}).get("genericBindings") or []
    return np.array([(int(b["attribute"]), int(b.get("customType", 0)))
                     for b in bindings if b.get("typeID") == "Animator"],
                    dtype=np.uint32).reshape(-1, 2)


def _scalars(decoder: acl.Decoder, packed: str | None,
             tracks: int | None, what: str) -> np.ndarray | None:
    """One scalar buffer as ``(its own frames, tracks)``; ``None`` when empty.

    Each buffer carries its own sample count, and it is not always the
    transform buffer's: 3 of one character's 55 locomotion clips are one
    sample apart (150 against 151, 900 against 901).  Nothing is resampled
    here — the stored array keeps the length the game gave it.
    """
    buffer = base64.b64decode(packed or "")
    if not buffer:
        return None
    got_tracks, _frames, _rate = acl.header(buffer)
    if tracks is not None and got_tracks != tracks:
        raise ClipError(f"{what} buffer has {got_tracks} tracks for {tracks} "
                        f"Animator bindings")
    return decoder.decode(buffer, acl.SCALAR)[:, :, 0]


def convert(doc: dict, decoder: acl.Decoder) -> Stored:
    """A staged ``AnimationClip`` as a :class:`Stored` clip in project space.

    Raises :class:`NotAcl` when the clip has no ACL transform buffer, and
    :class:`ClipError` when a buffer's track count is not the number of
    bindings it is paired with — rotation bindings for the transform buffer,
    Animator bindings for the scalar one.  Those pairings are what this
    reading rests on.
    """
    packed = doc.get("m_AclCompressedBuffer") or {}
    buffer = base64.b64decode(packed.get("TransformBufferData") or "")
    if not buffer:
        raise NotAcl("curves are in Unity's streamed/dense form, not ACL")
    hashes = track_hashes(doc)
    tracks, frames, rate = acl.header(buffer)
    if tracks != len(hashes):
        raise ClipError(f"{tracks} ACL tracks for {len(hashes)} rotation bindings")
    samples = decoder.decode(buffer, acl.TRANSFORM) * _FLIP
    muscle = doc.get("m_MuscleClip") or {}
    events = [{"time": float(e["time"]), "function": str(e["functionName"]),
               "data": str(e.get("data") or "")} for e in doc.get("m_Events") or []]
    keys = float_keys(doc)
    floats = _scalars(decoder, packed.get("FloatBufferData"), len(keys), "scalar")
    root = _scalars(decoder, packed.get("RootMotionBufferData"), None,
                    "root-motion")
    return Stored(str(doc.get("m_Name") or ""), rate, frames,
                  bool(muscle.get("m_LoopTime")), hashes, split(samples), events,
                  keys if floats is not None else None, floats, root)


def write(ident: str, container: str, clip: Stored) -> Path:
    """Store *clip* under *ident* and list it in the index."""
    path = folder() / f"{ident}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"hashes": clip.hashes}
    for name, (still, value, moving, data) in clip.channels.items():
        arrays[f"{name}_still"] = still
        arrays[f"{name}_value"] = value.astype(np.float32)
        arrays[f"{name}_moving"] = moving
        arrays[f"{name}_data"] = data.astype(np.float32)
    for name in ("float_keys", "floats", "root"):
        value = getattr(clip, name)
        if value is not None:
            arrays[name] = value
    meta = {"name": clip.name, "rate": clip.rate, "frames": clip.frames,
            "loop": clip.loop, "events": clip.events}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp.replace(path)
    listing = index()
    listing[ident] = {"name": clip.name, "container": container,
                      "frames": clip.frames, "rate": clip.rate,
                      "tracks": int(len(clip.hashes)),
                      "floats": 0 if clip.floats is None else int(clip.floats.shape[1])}
    write_json_atomic(folder() / INDEX, listing, indent=1)
    return path


# -- run time ------------------------------------------------------------------
def read(ident: str) -> Stored:
    """The stored clip *ident*; ``LookupError`` when it was never converted."""
    path = folder() / f"{ident}.npz"
    if not path.is_file():
        raise LookupError(f"clip {ident} is not in the store")
    with np.load(path) as arch:
        meta = json.loads(bytes(arch["meta"]).decode("utf-8"))
        channels = {name: (arch[f"{name}_still"], arch[f"{name}_value"],
                           arch[f"{name}_moving"], arch[f"{name}_data"])
                    for name, _cols in _CHANNELS}
        scalars = [arch[name] if name in arch else None
                   for name in ("float_keys", "floats", "root")]
        return Stored(meta["name"], float(meta["rate"]), int(meta["frames"]),
                      bool(meta["loop"]), arch["hashes"], channels,
                      meta["events"], *scalars)


def _dense(clip: Stored, name: str, width: int) -> np.ndarray:
    """One channel for every track and frame: ``(frames, tracks, width)``."""
    still, value, moving, data = clip.channels[name]
    out = np.empty((clip.frames, len(clip.hashes), width), dtype=np.float64)
    out[:, still] = value
    out[:, moving] = data
    return out


def _between(skeleton, bone, ancestor: str) -> np.ndarray:
    """The rest transform of the bones strictly between *bone* and *ancestor*."""
    chain = []
    index = bone.parent
    while index >= 0 and skeleton.bones[index].name != ancestor:
        chain.append(skeleton.bones[index])
        index = skeleton.bones[index].parent
    if index < 0:
        raise ClipError(f"{bone.name} does not sit under {ancestor} on this actor")
    out = m3.identity()
    for link in reversed(chain):
        out = out @ m3.compose(link.local_pos, link.local_rot, link.local_scale)
    return out


def body_curves(clip: Stored, skeleton, human: dict, out: AnimationClip,
                travel: bool = False) -> int:
    """Add the humanoid body's curves to *out*; returns how many bones they pose.

    *human* is the item's ``extra["human"]`` (the Avatar's definition and the
    node at each of its human-skeleton entries).  The muscle curves are solved
    into the definition's skeleton (:class:`.muscle.Human`) and each of its
    nodes is then written against the bone of that name: the hips with their
    position, every other node with its rotation only, so the actor keeps its
    own bone lengths.  ``travel=False`` plays the body as seen from the clip's
    motion node, so a clip that walks somewhere stays where it is put.

    Returns 0 — and adds nothing — for a clip without the body's curves.
    """
    if clip.float_keys is None or clip.floats is None:
        return 0
    columns = {int(attr): i for i, (attr, kind) in enumerate(clip.float_keys) if kind == MUSCLE}
    where = muscle.curves(columns)
    if where is None:
        return 0
    solver = muscle.Human(human["definition"], human["nodes"])
    data = clip.floats.astype(np.float64)
    root = data[:, where["root"]]
    root_t, root_q = root[:, :3], root[:, 3:]
    if not travel and where["motion"] is not None:
        motion = data[:, where["motion"]]
        root_t, root_q = muscle.in_place(root_t, root_q, motion[:, :3], motion[:, 3:])
    local_t, local_q = muscle.project_pose(
        *solver.relative(*solver.solve(data[:, where["dof"]], root_t, root_q)))
    times = np.arange(len(data), dtype=np.float64) / clip.rate
    hips = solver.node("Hips")
    posed = 0
    for node, name in enumerate(solver.nodes):
        parent = solver.parent[node]
        if parent < 0 or name is None or not skeleton.has(name):
            continue
        bone = skeleton.bones[skeleton.index[name]]
        between = _between(skeleton, bone, solver.nodes[parent])
        rots, poss = [], []
        for t, q in zip(local_t[:, node], local_q[:, node]):
            local = np.linalg.inv(between) @ m3.compose(t, q, np.ones(3))
            pos, rot, _scale = m3.decompose(local)
            rots.append(m3.quat_mul(m3.quat_conj(bone.local_rot), rot))
            poss.append(pos - bone.local_pos)
        out.set_curve(name, times, np.array(rots), np.array(poss) if node == hips else None)
        posed += 1
    return posed


def to_animation(clip: Stored, skeleton, paths: dict, human: dict | None = None,
                 travel: bool = False) -> tuple[AnimationClip, int]:
    """*clip* as offsets from *skeleton*'s rest, and how many tracks found a bone.

    *paths* is ``node name → path below the root`` (the item's *extra*).  The
    project's poses are deltas — ``rest · rotation``, ``rest + position``,
    ``rest × scale`` (``Skeleton.evaluate``) — while a Unity curve is the local
    transform itself, so each track is re-expressed against the bone it lands
    on.  A track whose path this actor has no node at is not played.

    With *human* (the item's ``extra["human"]``) the trunk and limbs are posed
    too, from the clip's muscle curves (:func:`body_curves`); the count
    returned is still the transform tracks'.
    """
    by_hash = {path_hash(path): node for node, path in paths.items()}
    rot = _dense(clip, "rot", 4)
    pos = _dense(clip, "pos", 3)
    scl = _dense(clip, "scl", 3)
    moving = {name: set(clip.channels[name][2].tolist()) for name, _ in _CHANNELS}
    times = np.arange(clip.frames, dtype=np.float64) / clip.rate
    out = AnimationClip(clip.name, clip.duration, clip.loop)
    matched = 0
    for track, value in enumerate(clip.hashes.tolist()):
        node = by_hash.get(int(value))
        if node is None or not skeleton.has(node):
            continue
        matched += 1
        bone = skeleton.bones[skeleton.index[node]]
        still = not any(track in moving[name] for name, _ in _CHANNELS)
        rows = slice(0, 1) if still else slice(None)
        inverse = m3.quat_conj(bone.local_rot)
        rots = np.array([m3.quat_mul(inverse, q) for q in rot[rows, track]])
        out.set_curve(node, times[rows], rots,
                      pos[rows, track] - bone.local_pos,
                      scl[rows, track] / bone.local_scale)
    if human is not None:
        body_curves(clip, skeleton, human, out, travel)
    return out, matched


def has_body(ident: str) -> bool:
    """Does the stored clip *ident* carry the body's muscle curves?"""
    clip = read(ident)
    if clip.float_keys is None:
        return False
    columns = {int(attr): i for i, (attr, kind) in enumerate(clip.float_keys) if kind == MUSCLE}
    return muscle.curves(columns) is not None
