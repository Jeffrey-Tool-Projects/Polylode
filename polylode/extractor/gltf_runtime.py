"""Match a loaded glTF scene's mesh nodes to Qt's imported models."""
from __future__ import annotations

import json
import struct
from pathlib import Path


def read(path: Path) -> tuple[dict, bytes]:
    data = Path(path).read_bytes()
    if data[:4] != b"glTF":
        raise ValueError("a binary glTF file is required")
    length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20:20 + length])
    binary = 20 + length
    size, kind = struct.unpack_from("<II", data, binary)
    if kind != 0x004E4942:
        raise ValueError("the glTF has no binary buffer")
    return document, data[binary + 8:binary + 8 + size]


def _collapsed(document: dict, binary: bytes, mesh: dict) -> bool:
    """Qt drops triangles whose three positions coincide, including GF2 billboards."""
    primitives = mesh.get("primitives", [])
    if not primitives:
        return False
    for primitive in primitives:
        if primitive.get("mode", 4) != 4 or primitive.get("targets"):
            return False
        position = document["accessors"][primitive["attributes"]["POSITION"]]
        if position.get("sparse") or position["componentType"] != 5126 or position["type"] != "VEC3":
            return False
        view = document["bufferViews"][position["bufferView"]]
        if view.get("buffer", 0) != 0:
            return False
        offset = view.get("byteOffset", 0) + position.get("byteOffset", 0)
        stride = view.get("byteStride", 12)
        if "indices" in primitive:
            accessor = document["accessors"][primitive["indices"]]
            formats = {5121: "B", 5123: "H", 5125: "I"}
            if accessor.get("sparse") or accessor["componentType"] not in formats or accessor["type"] != "SCALAR":
                return False
            indices_view = document["bufferViews"][accessor["bufferView"]]
            if indices_view.get("buffer", 0) != 0:
                return False
            fmt = "<" + formats[accessor["componentType"]]
            start = indices_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
            step = indices_view.get("byteStride", struct.calcsize(fmt))
            indices = [struct.unpack_from(fmt, binary, start + i * step)[0] for i in range(accessor["count"])]
        else:
            indices = range(position["count"])
        if not indices or len(indices) % 3:
            return False
        for i in range(0, len(indices), 3):
            triangle = indices[i:i + 3]
            if any(index >= position["count"] for index in triangle):
                raise ValueError("a glTF triangle index exceeds its vertex count")
            points = [struct.unpack_from("<3f", binary, offset + index * stride) for index in triangle]
            if not points[0] == points[1] == points[2]:
                return False
    return True


def mesh_nodes(document: dict, binary: bytes | None = None) -> list[int]:
    nodes = document.get("nodes", [])
    stack = list(reversed(document["scenes"][document.get("scene", 0)].get("nodes", [])))
    found, seen = [], set()
    while stack:
        index = stack.pop()
        if index in seen:
            raise ValueError("the glTF scene repeats a node")
        seen.add(index)
        node = nodes[index]
        if "mesh" in node and (binary is None or not _collapsed(document, binary, document["meshes"][node["mesh"]])):
            found.append(index)
        stack.extend(reversed(node.get("children", [])))
    return found


def models(root, document: dict, binary: bytes) -> dict:
    """Node index to imported Model, checked against material and morph counts."""
    from PySide6.QtCore import QObject
    from PySide6.QtQml import QQmlListReference
    loader = root.findChild(QObject, "loader")
    stack = list(reversed(loader.children())) if loader is not None else []
    imported = []
    while stack:
        node = stack.pop()
        kind = node.metaObject().className()
        if kind == "QQuick3DModel":
            imported.append(node)
        if kind not in ("QQuickKeyframeGroup", "QQuickKeyframe"):
            stack.extend(reversed(node.children()))
    indices = mesh_nodes(document, binary)
    if len(indices) != len(imported):
        raise ValueError("the imported mesh count differs from glTF")
    out = {}
    for index, model in zip(indices, imported):
        mesh = document["meshes"][document["nodes"][index]["mesh"]]
        primitives = mesh["primitives"]
        if QQmlListReference(model, "materials").count() != len(primitives):
            raise ValueError("the imported material count differs from glTF")
        count = len(primitives[0].get("targets", []))
        if any(len(p.get("targets", [])) != count for p in primitives) \
                or QQmlListReference(model, "morphTargets").count() not in (count, count * len(primitives)):
            raise ValueError("the imported morph count differs from glTF")
        out[index] = model
    return out
