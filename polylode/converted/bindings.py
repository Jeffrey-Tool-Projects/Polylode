"""Self-contained original donor binds: fitted skin must be blended only once during animation."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..core.paths import write_json_atomic
from ..endfield.body import GRAFT, unblend
from ..mesh.mesh import Mesh
from ..mesh.skin import split_archive_skin
from . import store


def prepare(folder: Path, source: Path) -> None:
    """Retain exact donor bone frames, independently of optional source-model or game installations."""
    metadata = store.read(folder)
    expected = metadata["provenance"]["source_body_archive_sha256"]
    if store.digest(source) != expected or metadata["donor"]["source"] != "base/body:o_body_a":
        raise store.WarehouseError("donor binding source differs from the body's recorded source")
    if all(name in metadata["files"] for name in ("donor_binding.json", "donor_binding.npz")):
        read(folder, metadata)
        return
    with np.load(source, allow_pickle=False) as archive:
        document = json.loads(bytes(archive["meta"]).decode("utf-8"))
        matches = [(i, row) for i, row in enumerate(document["pieces"]) if row["name"] == "o_body_a"]
        if len(matches) != 1:
            raise store.WarehouseError("donor source body is missing or ambiguous")
        index, piece = matches[0]
        names = np.asarray(piece["bones"])
        worlds = archive[f"{index}_bworld"]
        binds = archive[f"{index}_binds"]
        if not archive[f"{index}_bworld_ok"].all():
            raise store.WarehouseError("donor source body has missing bone frames")
        weights, indices = split_archive_skin(archive[f"{index}_skin"], bone_count=len(names))
    with np.load(folder / "body.npz", allow_pickle=False) as body:
        if (not np.array_equal(weights, body["donor_weights"])
                or not np.array_equal(indices, body["donor_indices"])
                or not np.array_equal(np.array([GRAFT + name for name in names]), body["donor_bone_names"])):
            raise store.WarehouseError("donor source weights or bone identities changed")
    np.savez_compressed(folder / "donor_binding.npz", names=names, worlds=worlds, binds=binds)
    write_json_atomic(folder / "donor_binding.json", {
        "version": 1, "source": metadata["source"], "ident": metadata["ident"],
        "source_body_sha256": expected, "body_sha256": metadata["files"]["body.npz"]}, indent=1)
    for name in ("donor_binding.json", "donor_binding.npz"):
        metadata["files"][name] = store.digest(folder / name)
    read(folder, metadata)
    metadata["ready_for_display"] = False
    write_json_atomic(folder / "warehouse.json", metadata, indent=1)


def read(folder: Path, metadata: dict) -> SimpleNamespace:
    """Read certified source-only bone frames; reject a recipe for another body or warehouse."""
    for name in ("donor_binding.json", "donor_binding.npz"):
        path = store.child(folder, name)
        if metadata["files"].get(name) != store.digest(path):
            raise store.WarehouseError("uncertified donor binding recipe")
    record = json.loads((folder / "donor_binding.json").read_text(encoding="utf-8"))
    if (record.get("version") != 1 or record.get("source") != metadata["source"]
            or record.get("ident") != metadata["ident"] or record.get("body_sha256") != metadata["files"]["body.npz"]
            or record.get("source_body_sha256") != metadata["provenance"]["source_body_archive_sha256"]):
        raise store.WarehouseError("donor binding recipe belongs to another source body")
    with np.load(folder / "donor_binding.npz", allow_pickle=False) as data:
        names, worlds, binds = data["names"].tolist(), data["worlds"], data["binds"]
    if (len(names) != len(set(names)) or worlds.shape != (len(names), 4, 4) or not np.isfinite(worlds).all()
            or not np.allclose(worlds[:, 3], [0, 0, 0, 1], atol=1e-12, rtol=0)):
        raise store.WarehouseError("invalid original donor bone frames")
    if binds.shape != worlds.shape or not np.isfinite(binds).all():
        raise store.WarehouseError("invalid original donor inverse binds")
    return SimpleNamespace(bones=names, bone_world=worlds, binds=binds)


def donor_mesh(model, recipe: SimpleNamespace) -> Mesh:
    """Create full fitted donor geometry, retaining each helper's original inverse bind for one blend."""
    data = model.body
    names = data["donor_bone_names"].tolist()
    if names != [GRAFT + name for name in recipe.bones]:
        raise store.WarehouseError("donor binding recipe has different bone names")
    mesh = Mesh.from_arrays(data["donor_positions"], data["donor_faces"], data["donor_normals"],
                            data["donor_uv"], name="Fitted complete body")
    weights, indices = data["donor_weights"], data["donor_indices"].astype(np.int64)
    active = weights > 0
    mesh.bone_idx = np.zeros_like(indices, dtype=np.int32)
    mapped = np.full(len(names), -1, dtype=np.int32)
    for index in np.unique(indices[active]):
        if not model.source_skeleton.has(names[index]):
            raise store.WarehouseError(f"donor helper joint missing: {names[index]}")
        mapped[index] = model.source_skeleton.index[names[index]]
    mesh.bone_idx[active] = mapped[indices[active]]
    mesh.bone_wt = weights.copy()
    if not unblend(model.source_skeleton, mesh, recipe):
        raise store.WarehouseError("donor body did not retain its original helper bindings")
    return mesh
