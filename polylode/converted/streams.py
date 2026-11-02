"""Lossless native streams in the engine's mesh/bone space, before source-specific material binding.

An assembled mesh is not a drawable actor. In particular, these streams must
not be given an unrelated game's material to bypass missing native passes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..endfield.axes import TO_Z_UP
from ..mesh.mesh import Mesh
from ..mesh.skin import validate_streams
from .store import WarehouseError


@dataclass(frozen=True)
class NativeMesh:
    """One original renderer with its triangle slots, all UV/color/TBN streams and sparse morphs."""
    mesh: Mesh
    slots: np.ndarray
    materials: tuple[str, ...]
    morphs: dict[str, tuple[np.ndarray, np.ndarray]]
    hidden: bool


def _array(arrays: dict, key: str, shape: tuple, *, integer: bool = False) -> np.ndarray:
    value = np.asarray(arrays[key])
    if value.shape != shape or not np.isfinite(value).all() or (integer and value.dtype.kind not in "iu"):
        raise WarehouseError(f"invalid native vertex stream: {key}")
    return value


def native_meshes(document: dict, arrays: dict, skeleton) -> list[NativeMesh]:
    """Convert the export basis once and map bone indices by names, never by skeleton ordering.

    Positive source weights are not pruned or normalized. An inactive padding
    joint alone may be replaced with a valid zero index. Sparse morph indices,
    triangle ordering/winding and tangent handedness remain unchanged.
    """
    names = [bone["name"] for bone in document["bones"]]
    if len(names) != len(set(names)) or set(names) - skeleton.index.keys():
        raise WarehouseError("native streams name missing or duplicate skeleton bones")
    bone_ids = np.array([skeleton.index[name] for name in names], dtype=np.int32)
    return [_native_mesh(index, part, arrays, bone_ids) for index, part in enumerate(document["parts"])]


def _native_mesh(index: int, part: dict, arrays: dict, bone_ids: np.ndarray) -> NativeMesh:
    prefix = f"p{index}_"
    positions = np.asarray(arrays[prefix + "co"])
    count = len(positions)
    mesh = Mesh(part["name"])
    turn = TO_Z_UP[:3, :3]
    mesh.positions = _array(arrays, prefix + "co", (count, 3)) @ turn
    mesh.normals = _array(arrays, prefix + "no", (count, 3)) @ turn
    # Engine tint is separate from the source shader's COLOR input below.
    mesh.colors = np.ones((count, 4), dtype=np.float32)
    triangles = np.asarray(arrays[prefix + "tri"])
    _array(arrays, prefix + "tri", (len(triangles), 3), integer=True)
    slots = _array(arrays, prefix + "slot", (len(triangles),), integer=True)
    if (not count or np.any(triangles < 0) or np.any(triangles >= count)
            or np.any(slots < 0) or np.any(slots >= len(part["materials"]))):
        raise WarehouseError(f"native triangle or material slot out of bounds: {part['name']}")
    mesh.indices = triangles.reshape(-1).astype(np.uint32)
    joints, weights = np.asarray(arrays[prefix + "jnt"]), np.asarray(arrays[prefix + "wgt"])
    validate_streams(joints, weights, count)
    active = weights > 0
    if np.any(joints[active] >= len(bone_ids)):
        raise WarehouseError(f"native skin joint out of bounds: {part['name']}")
    mesh.bone_idx = np.zeros_like(joints, dtype=np.int32)
    mesh.bone_idx[active] = bone_ids[joints[active].astype(np.int64)]
    mesh.bone_wt = weights.copy()
    for channel in part["uvs"]:
        if channel not in (0, 1, 2, 3):
            raise WarehouseError("unsupported native UV channel")
        name = "uvs" if channel == 0 else f"uv{channel}"
        setattr(mesh, name, _array(arrays, prefix + f"uv{channel}", (count, 2)).copy())
    if part["colors"]:
        mesh.vcolor = _array(arrays, prefix + "col", (count, 4)).copy()
    if part.get("tangents"):
        mesh.tangents = _array(arrays, prefix + "tan", (count, 4)).copy()
        mesh.tangents[:, :3] = mesh.tangents[:, :3] @ turn
    morphs = {}
    for number, name in enumerate(part["shapes"]):
        indices = np.asarray(arrays[prefix + f"s{number}_idx"])
        _array(arrays, prefix + f"s{number}_idx", (len(indices),), integer=True)
        delta = _array(arrays, prefix + f"s{number}_delta", (len(indices), 3))
        if name in morphs or np.any(indices < 0) or np.any(indices >= count) or len(np.unique(indices)) != len(indices):
            raise WarehouseError(f"invalid native sparse morph: {name}")
        morphs[name] = (indices.copy(), delta @ turn)
    return NativeMesh(mesh, slots.copy(), tuple(part["materials"]), morphs, bool(part["hidden"]))
