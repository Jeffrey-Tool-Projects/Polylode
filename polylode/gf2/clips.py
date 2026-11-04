"""A GF2 animation clip read from its own Unity data: the transform curves of a generic rig, sampled per frame.

The game's clips are Unity ``AnimationClip`` objects whose curves are in the muscle clip's three blocks, numbered
one after another: the streamed curves (each key holds the cubic of the segment that starts at it,
``((a·t + b)·t + c)·t + d`` for ``t`` from the key's time), then the dense ones (sampled at the block's rate from its
begin time), then the constant ones (one value each).  ``m_ClipBindingConstant.genericBindings`` says what they drive,
in the same order: a Transform's position (3 curves), rotation (a quaternion, 4), scale (3) or Euler angles (3); any
other binding takes one.  The count is checked against ``m_ValueArrayDelta`` (one entry a curve), so a layout this
reading does not match is refused rather than misread.  A binding names its transform by the CRC-32 of the path
below the Animator's root, as Endfield's clips do.

GF2's clips bind the bones themselves (a generic rig: the trunk and limbs are Transform curves, measured on the
character clips), so a frame is the bones' local transforms in Unity's space; :func:`poses` turns them into the
exported bones' frames the way the scene's rest is made (``endfield.convert.to_project`` of ``prefab.local_matrix``,
the root at the origin, ``rigscene.rest_matrix`` of the world).

``rest_matrix`` takes the scale out, so a posed frame is not ``rest_matrix`` of the clip's world but the transform Unity
skins with (:func:`skinned`: the world after the inverse of the rest world), scale and shear and all.  Display clips
scale bones (``CommandCenterBackGround`` sets Clukay SSR01's ``root`` to 1.202; a shoulder chain to 0.931, the rig's
``…_Ctrl_Jnt_skin`` helpers unevenly up to 2×, Cheyanne SSR0101's skirt chains unevenly at three levels); with the scale
taken out every bone below a scaled one stood that much further from its parent on a skin drawn at its rest size —
the eyes, on bones of their own, sat up off the face, and the body lost its shape.
"""
from __future__ import annotations

import re
import struct
import zlib

import numpy as np

from ..endfield import prefab as P
from ..endfield import rigscene as RS
from ..endfield.unitymesh import to_project

#: ``genericBindings[].typeID`` of a Transform, and how many curves each of its attributes takes
TRANSFORM = 4
DIMENSION = {1: 3, 2: 4, 3: 3, 4: 3}
POSITION, ROTATION, SCALE, EULER = 1, 2, 3, 4


class ClipError(ValueError):
    """The clip's blocks are not laid out as this reading expects."""


def path_hash(path: str) -> int:
    return zlib.crc32(path.encode("utf-8"))


def _streamed(words: list[int]) -> dict[int, list[tuple[float, tuple]]]:
    """The streamed block as each curve's keys: ``(time, (a, b, c, d))``."""
    data = struct.pack(f"<{len(words)}I", *words)
    keys: dict[int, list] = {}
    at = 0
    while at + 8 <= len(data):
        time, count = struct.unpack_from("<fi", data, at)
        at += 8
        for _ in range(count):
            index, a, b, c, d = struct.unpack_from("<i4f", data, at)
            at += 20
            keys.setdefault(index, []).append((time, (a, b, c, d)))
    return keys


def _evaluate(keys: list[tuple[float, tuple]], times: np.ndarray) -> np.ndarray:
    starts = np.array([time for time, _coeffs in keys])
    coeffs = np.array([c for _time, c in keys])
    at = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(keys) - 1)
    dt = times - starts[at]
    a, b, c, d = coeffs[at].T
    return ((a * dt + b) * dt + c) * dt + d


#: where a clip's decoded curves are kept in its own typetree: the pose check, the frames and the expression curves
#: all read them, and decoding a clip four times was a tenth of an outfit's export
DECODED = "\0polylode curves"


def curves(tree: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(times, values)``: the clip's frames at its sample rate and every curve's value on them, ``(frames,
    curves)``; decoded once a clip (:data:`DECODED`)."""
    if DECODED not in tree:
        tree[DECODED] = _decode(tree)
    return tree[DECODED]


def _decode(tree: dict) -> tuple[np.ndarray, np.ndarray]:
    muscle = tree["m_MuscleClip"]
    blocks = muscle["m_Clip"]["data"]
    streamed, dense, constant = blocks["m_StreamedClip"], blocks["m_DenseClip"], blocks["m_ConstantClip"]["data"]
    count = int(streamed["curveCount"]) + int(dense["m_CurveCount"]) + len(constant)
    if count != len(muscle["m_ValueArrayDelta"]):
        raise ClipError(f"{count} curves in the blocks for {len(muscle['m_ValueArrayDelta'])} values")
    rate = float(tree["m_SampleRate"]) or 30.0
    length = max(float(muscle["m_StopTime"]) - float(muscle["m_StartTime"]), 0.0)
    times = float(muscle["m_StartTime"]) + np.arange(int(round(length * rate)) + 1) / rate
    values = np.zeros((len(times), count))
    for index, keys in _streamed(streamed["data"]).items():
        if 0 <= index < int(streamed["curveCount"]):
            values[:, index] = _evaluate([k for k in keys if np.isfinite(k[0])], times)
    width = int(dense["m_CurveCount"])
    if width:
        samples = np.asarray(dense["m_SampleArray"], dtype=np.float64).reshape(-1, width)
        position = np.clip((times - float(dense["m_BeginTime"])) * float(dense["m_SampleRate"]), 0, len(samples) - 1)
        low = np.floor(position).astype(int)
        high = np.minimum(low + 1, len(samples) - 1)
        mix = (position - low)[:, None]
        first = int(streamed["curveCount"])
        values[:, first:first + width] = samples[low] * (1 - mix) + samples[high] * mix
    values[:, count - len(constant):] = np.asarray(constant, dtype=np.float64)
    return times - times[0], values


def transforms(tree: dict) -> tuple[np.ndarray, dict[int, dict[int, np.ndarray]]]:
    """``(times, {path hash: {attribute: (frames, n)}})``: the clip's Transform curves by the node they drive."""
    times, values = curves(tree)
    out: dict[int, dict[int, np.ndarray]] = {}
    at = 0
    for binding in tree["m_ClipBindingConstant"]["genericBindings"]:
        attribute = int(binding["attribute"])
        # a Transform binding takes its attribute's curves whatever its customType (an Euler one names its rotation
        # order there): counted as one, every binding after it read its neighbour's curves
        transform = int(binding["typeID"]) == TRANSFORM
        width = DIMENSION.get(attribute, 1) if transform else 1
        if transform and attribute in DIMENSION:
            curves_of = out.setdefault(int(binding["path"]), {})
            curves_of[attribute] = values[:, at:at + width]
            if attribute == EULER:
                curves_of[ROTATION] = euler_quaternions(values[:, at:at + width], int(binding["customType"]))
        at += width
    # curves past the last binding belong to none (54 on the outfits' display clips): left alone, as Unity's own
    # readers do; only bindings asking for more curves than there are mean another layout
    if at > values.shape[1]:
        raise ClipError(f"the bindings take {at} curves of {values.shape[1]}")
    return times, out


def shape_weights(tree: dict, parts: list[dict]) -> dict[int, np.ndarray]:
    """Unity blend-shape bindings matched by renderer path and the bare shape-name CRC."""
    times, values = curves(tree)
    targets = {(path_hash(part["path"]), path_hash(name)): (index, column)
               for index, part in enumerate(parts) if part.get("path")
               for column, name in enumerate(part["shapes"])}
    out, at = {}, 0
    for binding in tree["m_ClipBindingConstant"]["genericBindings"]:
        attribute, kind = int(binding["attribute"]), int(binding["typeID"])
        width = DIMENSION.get(attribute, 1) if kind == TRANSFORM else 1
        if at + width > values.shape[1]:
            raise ClipError(f"the bindings take more than {values.shape[1]} curves")
        target = targets.get((int(binding["path"]), attribute))
        if kind == 137 and int(binding.get("customType", 0)) == 20 and target is not None:
            index, column = target
            weights = out.setdefault(index, np.zeros((len(times), len(parts[index]["shapes"])), dtype=np.float32))
            weights[:, column] = values[:, at] / 100.0
        at += width
    return out


def scalar_curves(tree: dict, parts: list[dict]) -> dict:
    """Preserve non-transform source bindings, including controls without a glTF target."""
    _times, values = curves(tree)
    targets = {(path_hash(part["path"]), path_hash(name)) for part in parts if part.get("path")
               for name in part["shapes"]}
    bindings, columns, unbound, at = [], [], [], 0
    for binding in tree["m_ClipBindingConstant"]["genericBindings"]:
        kind, attribute = int(binding["typeID"]), int(binding["attribute"])
        width = DIMENSION.get(attribute, 1) if kind == TRANSFORM else 1
        if kind != TRANSFORM:
            bindings.append({key: int(binding.get(key, 0)) for key in ("path", "attribute", "typeID", "customType")})
            columns.append(at)
            if kind != 137 or int(binding.get("customType", 0)) != 20 or (int(binding["path"]), attribute) not in targets:
                unbound.append(len(bindings) - 1)
        at += width
    if at > values.shape[1]:
        raise ClipError(f"the bindings take {at} curves of {values.shape[1]}")
    return {"rate": float(tree["m_SampleRate"]) or 30.0, "bindings": bindings,
            "values": values[:, columns].T.tolist(), "unbound": unbound}


#: Unity's rotation orders (``RotationOrder``), what an Euler binding's customType names; 0 is the default, ZXY
ORDERS = {0: "ZXY", 1: "XYZ", 2: "XZY", 3: "YZX", 4: "ZXY", 5: "ZYX", 6: "YXZ"}


def euler_quaternions(degrees: np.ndarray, order: int) -> np.ndarray:
    """Euler curves (degrees, x y z) as quaternions ``x y z w``: turned about the axes in *order*, the first one
    first, as Unity applies them (its default ``ZXY`` is ``Quaternion.Euler``: z, then x, then y)."""
    from scipy.spatial.transform import Rotation
    axes = ORDERS.get(order, "ZXY")
    angles = np.asarray(degrees, dtype=np.float64)[:, ["XYZ".index(a) for a in axes]]
    return Rotation.from_euler(axes.lower(), angles, degrees=True).as_quat()


def _local(position, rotation, scale) -> np.ndarray:
    xyz = "XYZ"
    return to_project(P.local_matrix({
        "m_LocalPosition": dict(zip(xyz, position)), "m_LocalRotation": dict(zip("XYZW", rotation)),
        "m_LocalScale": dict(zip(xyz, scale))}))


def _locals(positions: np.ndarray, rotations: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """:func:`_local` of every frame at once: positions, quaternions (xyzw, normalised here as ``local_matrix`` does)
    and scales ``(frames, …)`` as ``(frames, 4, 4)`` in this project's space."""
    q = np.asarray(rotations, dtype=np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = (q / np.where(norm > 0, norm, 1.0)).T
    out = np.zeros((len(q), 4, 4))
    out[:, 0, :3] = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1)
    out[:, 1, :3] = np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1)
    out[:, 2, :3] = np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)
    out[:, :3, :3] *= np.asarray(scales, dtype=np.float64)[:, None, :]
    out[:, :3, 3] = positions
    out[:, 3, 3] = 1.0
    return to_project(out)


def _rest_trs(local: np.ndarray) -> tuple:
    """A stored (project-space) local back as Unity's position, rotation and scale: what a clip leaves alone."""
    unity = to_project(local)                       # the mirror is its own inverse
    scale = np.linalg.norm(unity[:3, :3], axis=0)
    from ..interchange.glb import _quaternions
    return unity[:3, 3], _quaternions(unity[:3, :3] / scale), scale


#: a skeleton bone of the rig's own naming (``Scapula_L``, ``Spine1_M``, ``IndexFinger2_R``): props, hair, cloth and
#: the rig's ``…_Ctrl_Jnt_skin`` helpers are named otherwise
SKELETON = re.compile(r"_[LMR]$")
#: further than this off its place, metres, a bone is hidden, not moved: the game sends a part it does not show far
#: away (Cheyanne SSR0101's shoes 217 m in her bedroom, Clukay SSR01's axe 100 km in class)
HIDDEN = 10.0


def strays(frames: np.ndarray, rest: np.ndarray, parents: list[int], names: list[str]) -> tuple[float, int]:
    """The most a skeleton bone (:data:`SKELETON`, below another) moves off its rest place under its parent over the
    clip's *frames* ``(frames, bones, 4, 4)``, in the parent's own (unscaled) frame: ``(metres, bone)``, ``(0, -1)``
    when none does.  The hips under the rig's root are not counted: a pose lowers them to crouch or lie down; nor is
    a frame that sends a bone further than :data:`HIDDEN` away, which hides it.

    A bone of a skeleton keeps its distance to its parent; a cutscene's clip does not, where its camera cannot see:
    GF2's ``Upgrade_*`` push both shoulder blades a metre off the chest while the camera is on the face."""
    worst, which = 0.0, -1
    for bone, parent in enumerate(parents):
        if parent < 0 or not (SKELETON.search(names[bone]) and SKELETON.search(names[parent])):
            continue
        placed = np.linalg.solve(frames[:, parent].astype(np.float64), frames[:, bone].astype(np.float64))[:, :3, 3]
        resting = np.linalg.solve(rest[parent], rest[bone])[:3, 3]
        off = np.linalg.norm(placed - resting, axis=1)
        moved = float(off[off < HIDDEN].max(initial=0.0))
        if moved > worst:
            worst, which = moved, bone
    return worst, which


#: how far a bone's place under its parent may jump in one frame before it is taken for snapped out of place, metres:
#: the display clips' own moves stay within 0.013 (Clukay SSR01's 42 clips), a camera trick jumps 0.11 (Love_4's left
#: shoulder) to 1.0 (Upgrade_*'s shoulder blades)
SNAP = 0.05


def unsnapped(positions: np.ndarray, rest: np.ndarray) -> tuple[np.ndarray, int]:
    """A bone's position curve ``(frames, 3)`` with the frames it is snapped out of place held at *rest*.  The curve is
    cut where it jumps more than :data:`SNAP` in one frame; a stretch between cuts lying (by its median) more than
    :data:`SNAP` off rest is snapped out — a snap may go out in steps, Class_* move a shoulder blade 0.33 m and then
    0.54 m — and so is the run of frames more than :data:`SNAP` off joined to it (Class_* slide the blade 0.26 m out
    before the jumps).  A curve with no such jump is a bone's own motion and is left whole, as is a stretch further
    than :data:`HIDDEN` away (a part hidden).  The curve and how many frames were held.

    GF2's display clips cheat for their camera with stepped keys: Love_4 pops Clukay SSR01's left shoulder 11 cm out of
    the shoulder blade from 2.4 s to 6.07 s, the Upgrade clips both shoulder blades a metre; seen whole, the arm came
    off.  Put back, the clip keeps every turn it makes."""
    positions = np.asarray(positions, dtype=np.float64)
    off = np.linalg.norm(positions - rest, axis=1)
    held = np.zeros(len(positions), dtype=bool)
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    near = np.maximum(off[1:], off[:-1]) < HIDDEN
    cuts = [0, *(np.flatnonzero((step > SNAP) & near) + 1).tolist(), len(positions)]
    if len(cuts) > 2:
        for start, stop in zip(cuts, cuts[1:]):
            stretch = off[start:stop]
            if len(stretch) and np.median(stretch) > SNAP and stretch.max() < HIDDEN:
                held[start:stop] = True
        if held.any():
            # each run of frames off rest that a snapped stretch is part of, slide out and back included
            away = (off > SNAP) & (off < HIDDEN)
            edges = np.flatnonzero(np.diff(np.r_[0, away.astype(np.int8), 0]))
            for start, stop in zip(edges[::2], edges[1::2]):
                if held[start:stop].any():
                    held[start:stop] = True
    out = positions.copy()
    out[held] = rest
    return out, int(held.sum())


#: a bone's frame as Unity skins with it, shared with Endfield's poses
skinned = RS.skinned


def poses(tree: dict, nodes: dict, paths: dict, bones: list[str]) -> np.ndarray:
    """The clip played on the stored *nodes* (``name → (parent, local)``, *paths* below the root): the frames of
    the nodes *bones*, ``(frames, bones, 4, 4)``, in the space of the scene's ``bone_rest``, as Unity skins with them
    (:func:`skinned`)."""
    times, by_hash = transforms(tree)
    root = next(name for name, (parent, _local) in nodes.items() if not parent)
    driven = {name: by_hash[path_hash(path)] for name, path in paths.items()
              if name != root and path_hash(path) in by_hash and name in nodes}
    order, seen = [], set()

    def visit(name):
        if name in seen:
            return
        parent = nodes[name][0]
        if parent:
            visit(parent)
        seen.add(name)
        order.append(name)

    for name in nodes:
        visit(name)
    rests = {name: _rest_trs(np.asarray(nodes[name][1], dtype=np.float64)) for name in driven}
    # a bone below the skeleton snapped out of place for a camera is put back (unsnapped); the hips under the rig's
    # root move the whole body and are left as they are
    placed = {name: unsnapped(curves_of[POSITION], rests[name][0])[0] for name, curves_of in driven.items()
              if POSITION in curves_of and SKELETON.search(nodes[name][0] or "")}
    resting: dict[str, np.ndarray] = {}
    for name in order:
        parent, local = nodes[name]
        local = np.asarray(local, dtype=np.float64)
        resting[name] = local if not parent else resting[parent] @ local
    # every frame of a bone at once: a node the clip leaves alone keeps one matrix, broadcast against its parent's
    # frames (one bone, one frame at a time, an outfit's 40 clips took 4 minutes — 2.1 million small SVDs)
    count = len(times)
    world: dict[str, np.ndarray] = {}
    for name in order:
        parent, local = nodes[name]
        local = np.asarray(local, dtype=np.float64)
        curves_of = driven.get(name)
        if curves_of is not None:
            position, rotation, scale = rests[name]
            positions = placed[name] if name in placed else \
                curves_of[POSITION] if POSITION in curves_of else np.broadcast_to(position, (count, 3))
            rotations = curves_of[ROTATION] if ROTATION in curves_of else np.broadcast_to(rotation, (count, 4))
            scales = RS.least(curves_of[SCALE]) if SCALE in curves_of else np.broadcast_to(scale, (count, 3))
            local = _locals(positions, rotations, scales)
        world[name] = local if not parent else world[parent] @ local
    out = np.empty((count, len(bones), 4, 4), dtype=np.float32)
    for k, name in enumerate(bones):
        out[:, k] = skinned(np.broadcast_to(world[name], (count, 4, 4)), resting[name])
    return out
