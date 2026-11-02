"""A staged Unity ``Mesh`` as arrays in this project's space.

Unity is left-handed and this project right-handed, both facing +Z, so the
change of handedness mirrors **X** (``origin/prefab.py`` does the same for
Illusion's meshes): positions and normals negate X, a matrix ``M`` becomes
``S·M·S`` with ``S = diag(-1, 1, 1, 1)``, and triangle winding is reversed.
UVs are left as Unity's — the textures are stored top row first, which is what
:class:`TextureCache` expects.

The exporter writes a ``Matrix4x4`` as ``M{column}{row}``: the translation of a
bind pose sits in ``M30 M31 M32`` (measured, §7.562).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIRROR = np.diag([-1.0, 1.0, 1.0, 1.0])

#: Mesh stream → (exporter field, components kept).  UV sets may be exported
#: wider than two components; only the first two are a texture coordinate.
_STREAMS = (("uv1", "m_UV1", 2), ("uv2", "m_UV2", 2), ("uv3", "m_UV3", 2),
            ("vcolor", "m_Colors", 4))


class MeshError(ValueError):
    """A staged mesh whose arrays do not agree with each other."""


@dataclass
class UnityMesh:
    name: str
    positions: np.ndarray          # (N, 3)
    normals: np.ndarray | None     # (N, 3)
    uvs: np.ndarray | None         # (N, 2)
    streams: dict                  # uv1 / uv2 / uv3 / vcolor → (N, w)
    submeshes: list                # one flat index array per submesh
    skin: np.ndarray | None        # (N, 8): four weights, four bone indices
    binds: np.ndarray | None       # (B, 4, 4)
    shapes: list                   # (name, indices, deltas, normal deltas)


def to_project(matrix: np.ndarray) -> np.ndarray:
    """A Unity-space 4×4 (or a stack of them) in this project's space."""
    return MIRROR @ np.asarray(matrix, dtype=np.float64) @ MIRROR


def matrix_of(m: dict) -> np.ndarray:
    """The exporter's ``Matrix4x4`` object as a row-major 4×4, Unity space."""
    return np.array([[float(m[f"M{col}{row}"]) for col in range(4)]
                     for row in range(4)])


def _table(doc: dict, key: str, count: int) -> np.ndarray | None:
    """A flat per-vertex list as ``(count, width)``; ``None`` when absent."""
    flat = doc.get(key)
    if not isinstance(flat, list) or not flat:
        return None
    arr = np.asarray(flat, dtype=np.float32)
    if count == 0 or arr.size % count:
        raise MeshError(f"{doc.get('m_Name')}: {key} has {arr.size} values "
                        f"for {count} vertices")
    return arr.reshape(count, -1)


def _mirror_x(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    out = arr[:, :3].copy()
    out[:, 0] = -out[:, 0]
    return out


def _submeshes(doc: dict, count: int) -> list:
    """One reversed-winding index array per submesh.

    The exporter concatenates the submeshes' triangle lists in order, each
    ``indexCount`` long.  A submesh declares the vertex range it uses
    (``firstVertex``, ``vertexCount``); indices are checked against it, and a
    list that only fits after adding ``baseVertex`` gets it added — Unity's own
    rule for that field.
    """
    indices = np.asarray(doc.get("m_Indices") or [], dtype=np.int64)
    out, cursor = [], 0
    for sub in doc.get("m_SubMeshes") or []:
        if str(sub.get("topology")) != "Triangles":
            raise MeshError(f"{doc.get('m_Name')}: submesh topology "
                            f"{sub.get('topology')} is not a triangle list")
        n = int(sub["indexCount"])
        tri = indices[cursor:cursor + n]
        cursor += n
        if len(tri) != n or n % 3:
            raise MeshError(f"{doc.get('m_Name')}: submesh wants {n} indices, "
                            f"{len(tri)} left")
        lo = int(sub["firstVertex"])
        hi = lo + int(sub["vertexCount"])
        if n and not (lo <= tri.min() and tri.max() < hi):
            tri = tri + int(sub["baseVertex"])
            if not (lo <= tri.min() and tri.max() < hi <= count):
                raise MeshError(f"{doc.get('m_Name')}: submesh indices fall "
                                f"outside its vertex range {lo}..{hi}")
        out.append(tri.reshape(-1, 3)[:, ::-1].reshape(-1).astype(np.uint32))
    if cursor != len(indices):
        raise MeshError(f"{doc.get('m_Name')}: {len(indices) - cursor} indices "
                        f"belong to no submesh")
    return out


def _skin(doc: dict, count: int) -> np.ndarray | None:
    rows = doc.get("m_Skin")
    if not rows:
        return None
    if len(rows) != count:
        raise MeshError(f"{doc.get('m_Name')}: {len(rows)} skin rows for "
                        f"{count} vertices")
    out = np.zeros((count, 8), dtype=np.float64)
    for i, row in enumerate(rows):
        out[i, :4] = row["weight"]
        out[i, 4:] = row["boneIndex"]
    return out


def _xyz(v: dict) -> tuple:
    return (-float(v["X"]), float(v["Y"]), float(v["Z"]))


def _shapes(doc: dict) -> list:
    """``m_Shapes`` → ``[(name, indices, deltas, normal deltas | None)]``.

    The same reading as ``origin/prefab.py``'s: a channel names a run of
    frames, the last frame is the shape at full weight, and a frame is a slice
    of the shared sparse ``vertices`` list.
    """
    sh = doc.get("m_Shapes") or {}
    verts, frames = sh.get("vertices") or [], sh.get("shapes") or []
    out = []
    for ch in sh.get("channels") or []:
        last = int(ch["frameIndex"]) + int(ch["frameCount"]) - 1
        if not 0 <= last < len(frames):
            raise MeshError(f"{doc.get('m_Name')}: blend shape {ch.get('name')} "
                            f"names frame {last} of {len(frames)}")
        frame = frames[last]
        first = int(frame["firstVertex"])
        block = verts[first:first + int(frame["vertexCount"])]
        idx = np.array([int(v["index"]) for v in block], dtype=np.int32)
        delta = np.array([_xyz(v["vertex"]) for v in block],
                         dtype=np.float32).reshape(-1, 3)
        normal = None
        if frame.get("hasNormals"):
            normal = np.array([_xyz(v["normal"]) for v in block],
                              dtype=np.float32).reshape(-1, 3)
            if not np.any(normal):
                normal = None
        out.append((str(ch.get("name", "")), idx, delta, normal))
    return out


def read_mesh(doc: dict) -> UnityMesh:
    """Convert the exporter's ``Mesh`` JSON; raises :class:`MeshError` if it lies."""
    count = int(doc["m_VertexCount"])
    positions = _mirror_x(_table(doc, "m_Vertices", count))
    if positions is None:
        raise MeshError(f"{doc.get('m_Name')}: no vertices")
    uv0 = _table(doc, "m_UV0", count)
    streams = {}
    for name, key, width in _STREAMS:
        arr = _table(doc, key, count)
        if arr is not None:
            if arr.shape[1] < width:
                raise MeshError(f"{doc.get('m_Name')}: {key} is "
                                f"{arr.shape[1]} wide, need {width}")
            streams[name] = arr[:, :width].copy()
    binds = doc.get("m_BindPose") or None
    return UnityMesh(
        name=str(doc.get("m_Name") or ""),
        positions=positions,
        normals=_mirror_x(_table(doc, "m_Normals", count)),
        uvs=None if uv0 is None else uv0[:, :2].copy(),
        streams=streams,
        submeshes=_submeshes(doc, count),
        skin=_skin(doc, count),
        binds=(None if binds is None else
               to_project(np.stack([matrix_of(m) for m in binds]))),
        shapes=_shapes(doc))
