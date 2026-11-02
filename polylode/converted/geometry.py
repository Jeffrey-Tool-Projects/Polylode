"""Exact geometry and full-weight CPU skinning of consolidated conversion packages.

This is deliberately independent of material rendering: a readable model does
not imply its original shaders or its new body's appearance have been accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..endfield.actor import build_skeleton
from ..endfield.axes import TO_Z_UP
from ..mesh.skeleton import Skeleton
from ..mesh.skinning import posed_normals
from ..origin import converted_rig as CR
from . import store


def skeleton(document: dict, arrays: dict) -> Skeleton:
    """Restore the export's Z-up matrices into the engine's Y-up space, retaining affine locals."""
    worlds = np.linalg.inv(TO_Z_UP) @ arrays["bone_rest"]
    nodes = {}
    for index, row in enumerate(document["bones"]):
        parent = row["parent"]
        if parent >= index or parent < -1 or row["name"] in nodes:
            raise store.WarehouseError("native hierarchy is not unique and parent-first")
        above = document["bones"][parent]["name"] if parent >= 0 else ""
        local = np.linalg.solve(worlds[parent], worlds[index]) if parent >= 0 else worlds[index]
        nodes[row["name"]] = (above, local)
    return build_skeleton(nodes, document["name"])


def _dense_inputs(vectors: np.ndarray, weights: np.ndarray, matrices: np.ndarray) -> tuple:
    vectors, weights, matrices = (np.asarray(a, dtype=np.float64) for a in (vectors, weights, matrices))
    if (vectors.ndim != 2 or vectors.shape[1] != 3 or matrices.ndim != 3 or matrices.shape[1:] != (4, 4)
            or weights.shape != (len(vectors), len(matrices))):
        raise store.WarehouseError("dense skin dimensions disagree")
    if not all(np.isfinite(a).all() for a in (vectors, weights, matrices)) or np.any(weights < 0):
        raise store.WarehouseError("dense skin contains invalid values")
    if not np.allclose(weights.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise store.WarehouseError("dense skin weights are not normalized")
    return vectors, weights, matrices


def dense_normals(normals: np.ndarray, weights: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    """Carry authored normals through the full dense blend's cofactor, including mirrors/shear.

    Retains every influence and bounds scratch storage by vertex count, rather
    than allocating vertex × bone matrices. Degenerate results are errors.
    """
    normals, weights, matrices = _dense_inputs(normals, weights, matrices)
    frames = np.zeros((len(normals), 3, 3), dtype=np.float64)
    for column, matrix in enumerate(matrices):
        frames += weights[:, column, None, None] * matrix[:3, :3]
    result = posed_normals(frames, normals)
    if not np.isfinite(result).all() or np.any(np.linalg.norm(result, axis=1) == 0):
        raise store.WarehouseError("dense skin has degenerate posed normals")
    return result


def dense_deform(positions: np.ndarray, weights: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    """Skin every recorded influence, including the opaque body's 51 nonzero weights per vertex.

    Accumulation uses float64 and bounded temporary storage; no top-k reduction,
    zero-weight threshold or per-vertex renormalisation changes the source.
    """
    positions, weights, matrices = _dense_inputs(positions, weights, matrices)
    out = np.zeros_like(positions)
    for column, matrix in enumerate(matrices):
        out += weights[:, column, None] * (positions @ matrix[:3, :3].T + matrix[:3, 3])
    return out


@dataclass
class CharacterGeometry:
    """The verified arrays and source rig; creating this object draws no image."""
    folder: Path
    metadata: dict
    document: dict
    native: dict
    body: dict
    source_skeleton: Skeleton
    rig: CR.ConvertedRig

    def rebound(self) -> CR.Rebound:
        """Bind the native model and fitted-body helper joints to the original-game hierarchy."""
        return CR.Rebound(self.rig, self.source_skeleton, self.metadata["ident"])

    def native_meshes(self) -> list:
        """Prepare engine-space source streams, retaining materials/morphs for their own renderer."""
        from .streams import native_meshes
        return native_meshes(self.document, self.native, self.source_skeleton)

    def opaque_positions(self, native_world: np.ndarray) -> np.ndarray:
        """Evaluate the complete opaque body using all weights on the native skeleton."""
        return dense_deform(self.body["opaque_positions"], self.body["opaque_weights"],
                            self.opaque_matrices(native_world))

    def opaque_matrices(self, native_world: np.ndarray) -> np.ndarray:
        """Return the exact inner-surface palette shared by position and normal evaluation."""
        names = self.body["opaque_bone_names"].tolist()
        indices = [self.source_skeleton.index[name] for name in names]
        binds = np.stack([self.source_skeleton.bones[index].inv_bind for index in indices])
        return native_world[indices] @ binds


def load(folder: Path) -> CharacterGeometry:
    """Load a package's native streams, full opaque skin and donor fit; reject stale members.

    An own-body package (``warehouse.json`` ``body.kind == "own"``) has no ``body.npz``:
    its ``body`` is empty and the body is the native renderer the descriptor names.
    """
    import json
    folder = Path(folder)
    meta = store.read(folder)
    document = json.loads(store.child(folder, "native.json").read_text(encoding="utf-8"))
    with np.load(store.child(folder, "native.npz"), allow_pickle=False) as archive:
        native = {name: archive[name] for name in archive.files}
    body = {}
    # an own-body package's body is one of its native renderers; only a lent body has its own arrays
    if meta.get("body", {}).get("kind") != "own":
        with np.load(store.child(folder, "body.npz"), allow_pickle=False) as archive:
            body = {name: archive[name] for name in archive.files}
    source = skeleton(document, native)
    for name, (parent, local) in meta["helper_nodes"].items():
        source.add_matrix(name, parent, np.asarray(local, dtype=np.float64))
    source.finalize()
    rig = CR.from_payload(meta["rig"], source)
    return CharacterGeometry(folder, meta, document, native, body, source, rig)
