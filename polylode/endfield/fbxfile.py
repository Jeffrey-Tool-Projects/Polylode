"""A binary FBX's node transforms: read, and rewritten in place.

``tools/ef_fbx_blender.py`` has Blender write the file.  Blender takes a
node's rotation as Euler angles **out of a 32-bit matrix**, and where a bone's
local pitch is within a hundredth of a degree of ±90° the other two angles
are read off entries no larger than their own rounding error: measured on the
game's rigs, a node came out 0.36° off and the bones below it a millimetre
(``bone world`` from the file's nodes against the scene it was written from;
the skin's own bind matrices are whole matrices and were exact).  The values
are 64-bit in the file, so :func:`refine` computes each bone's local transform
again in 64 bits and writes it over what is there — nothing moves, no record
changes size.

Only what that needs is read: node records, their scalar and string
properties, where each ``Lcl`` value sits, and the skin clusters' bind
matrices (``TransformLink``: the bone's world matrix when the mesh was bound,
which is what a skin is deformed from — a node and its cluster have to agree).
The layout is FBX 7.4's (32-bit record headers); a 7.5 file is refused rather
than misread.

A node's local transform is its ``Lcl Translation``, ``Lcl Rotation`` (Euler
XYZ, degrees: ``Rz · Ry · Rx``) and ``Lcl Scaling`` and nothing else here —
the exporter this reads after writes no pre-rotation or pivot, and a node that
carries one raises :class:`FbxError`.
"""
from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
#: the last version with 32-bit record headers
VERSION_LIMIT = 7500
#: byte size of each scalar property type
_SCALAR = {b"Y": 2, b"C": 1, b"I": 4, b"F": 4, b"D": 8, b"L": 8}
_FORMAT = {b"Y": "<h", b"C": "<B", b"I": "<i", b"F": "<f", b"D": "<d", b"L": "<q"}
_ARRAYS = b"fdlib"
#: node properties this module applies or knows to be unrelated to a transform
_KNOWN = {"Lcl Translation", "Lcl Rotation", "Lcl Scaling", "DefaultAttributeIndex",
          "InheritType", "Visibility"}
#: the file's space as Blender exports it (Y up) → the scene's (Z up)
FILE_TO_SCENE = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
BONE = "LimbNode"
#: how far :func:`refine` will move a value: further than this is not rounding
REFINE_DEG, REFINE_M = 3.0, 1e-3
#: a value the file leaves out (it reads as zero) may be this far from zero
UNWRITTEN_DEG, UNWRITTEN_M = 1e-4, 1e-7


class FbxError(ValueError):
    """Not a binary FBX this module reads, or a node it would misread."""


@dataclass
class Node:
    """One ``Model`` record."""
    uid: int
    name: str
    kind: str                           # ``LimbNode``, ``Null``, ``Mesh`` …
    #: property name → (three values, the byte offset of each)
    lcl: dict = field(default_factory=dict)
    parent: int = 0                     # uid; 0 at the top


def _records(data: bytes, start: int, end: int):
    """``(name, properties, body start, body end)`` per record in a range.

    A property is ``(type, value, offset of the value)``; an array's value is
    ``(count, encoding, stored bytes)`` and :func:`_array` reads it.
    """
    position = start
    while position + 13 <= end:
        stop, count, _length = struct.unpack_from("<3I", data, position)
        if stop == 0:
            return                      # the empty record that closes a list
        size = data[position + 12]
        name = data[position + 13:position + 13 + size]
        cursor = position + 13 + size
        properties = []
        for _ in range(count):
            kind = data[cursor:cursor + 1]
            cursor += 1
            if kind in _SCALAR:
                properties.append((kind, struct.unpack_from(_FORMAT[kind], data, cursor)[0],
                                   cursor))
                cursor += _SCALAR[kind]
            elif kind in (b"S", b"R"):
                (length,) = struct.unpack_from("<I", data, cursor)
                properties.append((kind, data[cursor + 4:cursor + 4 + length], cursor + 4))
                cursor += 4 + length
            elif kind in _ARRAYS and kind:
                header = struct.unpack_from("<3I", data, cursor)
                properties.append((kind, header, cursor + 12))
                cursor += 12 + header[2]
            else:
                raise FbxError(f"property type {kind!r} at byte {cursor - 1}")
        yield name, properties, cursor, stop
        position = stop


def _array(data: bytes, prop: tuple) -> np.ndarray:
    """An array property's numbers (only doubles are needed here)."""
    kind, (count, encoding, stored), offset = prop
    if kind != b"d":
        raise FbxError(f"array of type {kind!r} where doubles were expected")
    raw = data[offset:offset + stored]
    if encoding == 1:
        raw = zlib.decompress(raw)
    elif encoding != 0:
        raise FbxError(f"array encoding {encoding}")
    return np.frombuffer(raw, dtype="<f8", count=count)


def _top(data: bytes) -> dict:
    if not data.startswith(MAGIC):
        raise FbxError("not a binary FBX")
    (version,) = struct.unpack_from("<I", data, len(MAGIC))
    if version >= VERSION_LIMIT:
        raise FbxError(f"FBX version {version}: 64-bit records are not read here")
    return {name: (start, stop)
            for name, _props, start, stop in _records(data, len(MAGIC) + 4, len(data))}


def read_nodes(data: bytes) -> dict[int, Node]:
    """Every ``Model`` of the file by uid, with its ``Lcl`` values and parent."""
    top = _top(data)
    if b"Objects" not in top or b"Connections" not in top:
        raise FbxError("no Objects or no Connections")
    nodes: dict[int, Node] = {}
    for name, props, start, stop in _records(data, *top[b"Objects"]):
        if name != b"Model":
            continue
        node = Node(int(props[0][1]), props[1][1].split(b"\x00")[0].decode("utf-8"),
                    props[2][1].decode("utf-8"))
        for held, _props, inner, inner_stop in _records(data, start, stop):
            if held != b"Properties70":
                continue
            for _p, values, _a, _b in _records(data, inner, inner_stop):
                key = values[0][1].decode("utf-8")
                if key not in _KNOWN:
                    raise FbxError(f"{node.name}: node property {key!r} is not applied here")
                if key.startswith("Lcl "):
                    node.lcl[key] = ([float(v[1]) for v in values[4:7]],
                                     [v[2] for v in values[4:7]])
        nodes[node.uid] = node
    for _name, props, _start, _stop in _records(data, *top[b"Connections"]):
        if props[0][1] == b"OO" and props[1][1] in nodes and props[2][1] in nodes:
            nodes[props[1][1]].parent = int(props[2][1])
    return nodes


def compose(degrees) -> np.ndarray:
    """Euler XYZ in degrees as a 3×3: ``Rz · Ry · Rx``."""
    x, y, z = (math.radians(a) for a in degrees)
    turn_x = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    turn_y = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    turn_z = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return turn_z @ turn_y @ turn_x


def euler_xyz(rotation: np.ndarray) -> tuple:
    """The degrees :func:`compose` turns back into *rotation*.

    At the pole (pitch ±90°) only the sum or the difference of the other two
    angles is determined; all of it goes to X and Z is 0.
    """
    r = np.asarray(rotation, dtype=np.float64)
    level = math.hypot(r[0, 0], r[1, 0])
    y = math.atan2(-r[2, 0], level)
    if level > 1e-12:
        x, z = math.atan2(r[2, 1], r[2, 2]), math.atan2(r[1, 0], r[0, 0])
    else:
        x, z = math.atan2(-r[1, 2], r[1, 1]), 0.0
    return math.degrees(x), math.degrees(y), math.degrees(z)


def angle(want: np.ndarray, have: np.ndarray) -> float:
    """Degrees between two rotations, from the sine as well as the cosine.

    The cosine of a small angle differs from 1 by less than a 32-bit matrix
    resolves, and an ``acos`` of the trace alone reads 0.03° where there is
    none.
    """
    relative = np.asarray(want, dtype=np.float64).T @ np.asarray(have, dtype=np.float64)
    sine = 0.5 * math.sqrt((relative[2, 1] - relative[1, 2]) ** 2
                           + (relative[0, 2] - relative[2, 0]) ** 2
                           + (relative[1, 0] - relative[0, 1]) ** 2)
    return math.degrees(math.atan2(sine, (np.trace(relative) - 1.0) / 2.0))


def _local(node: Node) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = compose(node.lcl.get("Lcl Rotation", ([0.0, 0.0, 0.0],))[0]) * np.array(
        node.lcl.get("Lcl Scaling", ([1.0, 1.0, 1.0],))[0])
    out[:3, 3] = node.lcl.get("Lcl Translation", ([0.0, 0.0, 0.0],))[0]
    return out


def bone_worlds(data: bytes) -> dict[str, np.ndarray]:
    """Bone name → world 4×4 in the scene's space, from the file's own nodes."""
    nodes = read_nodes(data)
    worlds: dict[int, np.ndarray] = {}

    def world(uid: int) -> np.ndarray:
        if uid not in worlds:
            node = nodes[uid]
            own = _local(node)
            worlds[uid] = own if not node.parent else world(node.parent) @ own
        return worlds[uid]

    return {node.name: FILE_TO_SCENE @ world(uid)
            for uid, node in nodes.items() if node.kind == BONE}


def bind_worlds(data: bytes) -> dict[str, list]:
    """Bone name → the ``TransformLink`` of each cluster bound to it, scene space.

    FBX writes a matrix row by row with the translation in the last row.
    """
    top = _top(data)
    nodes = read_nodes(data)
    clusters: dict[int, np.ndarray] = {}
    for name, props, start, stop in _records(data, *top[b"Objects"]):
        if name != b"Deformer":
            continue
        for held, values, _a, _b in _records(data, start, stop):
            if held == b"TransformLink":        # only a cluster has one
                clusters[int(props[0][1])] = FILE_TO_SCENE @ _array(data, values[0]).reshape(4, 4).T
    out: dict[str, list] = {}
    for _name, props, _start, _stop in _records(data, *top[b"Connections"]):
        if props[0][1] == b"OO" and props[1][1] in nodes and props[2][1] in clusters:
            out.setdefault(nodes[props[1][1]].name, []).append(clusters[props[2][1]])
    return out


def refine(path: Path, bones: list) -> tuple[float, float]:
    """Write each bone's local transform into the file again, in 64 bits.

    *bones* is ``(name, parent index or -1, rest 4×4)`` in the scene's space,
    parents first; a root bone's local transform is its rest (the armature's
    own node carries the change of axes).  Returns the largest correction made,
    ``(degrees, metres)``.  Raises :class:`FbxError` — and leaves the file as
    it was — when a bone is missing, hangs under another parent than the scene
    says, or would move further than rounding explains.
    """
    data = bytearray(Path(path).read_bytes())
    nodes = read_nodes(bytes(data))
    by_name = {node.name: node for node in nodes.values() if node.kind == BONE}
    edits, most_turn, most_move = [], 0.0, 0.0
    for name, parent, rest in bones:
        node = by_name.get(name)
        if node is None:
            raise FbxError(f"bone {name} is not in the file")
        above = nodes.get(node.parent)
        expected = bones[parent][0] if parent >= 0 else None
        if (above.name if above is not None and above.kind == BONE else None) != expected:
            raise FbxError(f"bone {name} is not under {expected}")
        rest = np.asarray(rest, dtype=np.float64)
        local = rest if parent < 0 else np.linalg.inv(
            np.asarray(bones[parent][2], dtype=np.float64)) @ rest
        degrees = euler_xyz(local[:3, :3])
        was = _local(node)
        turned = angle(local[:3, :3], was[:3, :3])
        moved = float(np.linalg.norm(local[:3, 3] - was[:3, 3]))
        if turned > REFINE_DEG or moved > REFINE_M:
            raise FbxError(f"bone {name}: the file is {turned:.3f} deg, "
                           f"{moved * 1000:.3f} mm from the scene")
        most_turn, most_move = max(most_turn, turned), max(most_move, moved)
        for key, values, least in (("Lcl Rotation", degrees, UNWRITTEN_DEG),
                                   ("Lcl Translation", local[:3, 3], UNWRITTEN_M)):
            if key in node.lcl:
                edits += list(zip(node.lcl[key][1], values))
            elif max(abs(float(v)) for v in values) > least:
                # the exporter leaves a property out when it is all zeros
                raise FbxError(f"bone {name}: no {key} in the file to write")
    for offset, value in edits:
        struct.pack_into("<d", data, offset, float(value))
    Path(path).write_bytes(bytes(data))
    return most_turn, most_move


def bone_errors(path: Path, bones: list) -> dict:
    """How far the file's bones are from *bones* (as for :func:`refine`).

    ``{"bones": count in the file, "missing": names, "offset_mm": …,
    "turn_deg": …, "worst": name, "binds": count, "bind_offset_mm": …,
    "bind_turn_deg": …, "bind_worst": name}`` — the largest position and axis
    difference over the bones both have: first of the nodes, then of the skin
    clusters' bind matrices.
    """
    data = Path(path).read_bytes()
    worlds, binds = bone_worlds(data), bind_worlds(data)
    out = {"bones": len(worlds), "missing": [], "offset_mm": 0.0, "turn_deg": 0.0,
           "worst": "", "binds": sum(len(held) for held in binds.values()),
           "bind_offset_mm": 0.0, "bind_turn_deg": 0.0, "bind_worst": ""}
    for name, _parent, rest in bones:
        have = worlds.get(name)
        if have is None:
            out["missing"].append(name)
            continue
        rest = np.asarray(rest, dtype=np.float64)
        for prefix, matrix in [("", have)] + [("bind_", m) for m in binds.get(name, [])]:
            out[f"{prefix}offset_mm"] = max(
                out[f"{prefix}offset_mm"],
                1000.0 * float(np.linalg.norm(matrix[:3, 3] - rest[:3, 3])))
            turned = angle(rest[:3, :3], matrix[:3, :3] / np.linalg.norm(matrix[:3, 0]))
            if turned > out[f"{prefix}turn_deg"]:
                out[f"{prefix}turn_deg"], out[f"{prefix}worst"] = turned, name
    return out
