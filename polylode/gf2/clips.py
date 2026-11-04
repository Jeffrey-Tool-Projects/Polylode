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


def curves(tree: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(times, values)``: the clip's frames at its sample rate and every curve's value on them, ``(frames,
    curves)``."""
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
    resting: dict[str, np.ndarray] = {}
    for name in order:
        parent, local = nodes[name]
        local = np.asarray(local, dtype=np.float64)
        resting[name] = local if not parent else resting[parent] @ local
    out = np.empty((len(times), len(bones), 4, 4), dtype=np.float32)
    for frame in range(len(times)):
        world: dict[str, np.ndarray] = {}
        for name in order:
            parent, local = nodes[name]
            local = np.asarray(local, dtype=np.float64)
            curves_of = driven.get(name)
            if curves_of is not None:
                position, rotation, scale = rests[name]
                position = curves_of[POSITION][frame] if POSITION in curves_of else position
                rotation = curves_of[ROTATION][frame] if ROTATION in curves_of else rotation
                scale = RS.least(curves_of[SCALE][frame]) if SCALE in curves_of else scale
                local = _local(position, rotation, scale)
            world[name] = local if not parent else world[parent] @ local
        for k, name in enumerate(bones):
            out[frame, k] = skinned(world[name], resting[name])
    return out
