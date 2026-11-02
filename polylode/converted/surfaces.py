"""Archive the delivered opaque inner surface without requiring Blender at game runtime."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from ..core.paths import write_json_atomic
from . import store

VERSION = 1


def corner_uvs(document: dict, arrays: dict, triangle_count: int) -> tuple[np.ndarray, ...]:
    """Retain every authored UV seam; refuse missing, nonfinite or differently indexed layers."""
    count = document["uv_channels"]
    if type(count) is not int or count < 0:
        raise store.WarehouseError("invalid retained inner-surface UV channel count")
    expected = {f"uv{index}_corners" for index in range(count)}
    if {name for name in arrays if name.endswith("_corners")} != expected:
        raise store.WarehouseError("retained inner-surface UV channels disagree")
    result = tuple(arrays[f"uv{index}_corners"] for index in range(count))
    if any(values.shape != (triangle_count * 3, 2) or not np.isfinite(values).all() for values in result):
        raise store.WarehouseError("retained inner-surface UV corners differ from its triangles")
    return result


def reference(folder: Path, metadata: dict) -> Path:
    """Resolve this exact delivered covered model; never substitute another character."""
    relative = f'{metadata["ident"]}_covered.blend'
    proof = metadata.get("reference_files", {}).get(relative)
    if proof is None:
        raise store.WarehouseError("covered source model has no reference proof")
    model = store.child(folder, relative)
    if store.digest(model) != proof["sha256"]:
        raise store.WarehouseError("covered source model changed")
    return model


def read(folder: Path, metadata: dict | None = None) -> tuple[dict, dict]:
    """Read the retained material graph and authored normal/UV arrays, bound to the current body."""
    folder = Path(folder)
    metadata = store.read(folder) if metadata is None else metadata
    for relative in ("inner_surface.json", "inner_surface.npz"):
        path = store.child(folder, relative)
        if metadata["files"].get(relative) != store.digest(path):
            raise store.WarehouseError(f"uncertified runtime surface: {relative}")
    document = json.loads((folder / "inner_surface.json").read_text(encoding="utf-8"))
    if (document.get("version") != VERSION or document.get("source") != metadata["source"]
            or document.get("ident") != metadata["ident"]
            or document.get("body_sha256") != metadata["files"]["body.npz"]):
        raise store.WarehouseError("runtime surface belongs to another body or character")
    with np.load(folder / "inner_surface.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    normals = arrays["normals"]
    if normals.ndim != 2 or normals.shape[1] != 3 or not np.isfinite(normals).all():
        raise store.WarehouseError("invalid retained inner-surface normals")
    return document, arrays


def prepare(folder: Path, blender: Path) -> None:
    """Read one covered model in a hidden process, retain its material/normal data, and verify it.

    No scene is saved and no rendering is performed. Successful archives are
    independent runtime members; the source blend remains an optional reference.
    """
    folder = Path(folder).resolve()
    metadata = store.read(folder)
    model = reference(folder, metadata)
    if all(name in metadata["files"] for name in ("inner_surface.json", "inner_surface.npz")):
        document, _arrays = read(folder, metadata)
        if document["model_sha256"] == store.digest(model):
            return
    worker = Path(__file__).with_name("surfaces_blender.py")
    log = folder / "inner_surface.log"
    command = [str(Path(blender).resolve()), "--background", "--factory-startup", "--disable-autoexec",
               "--threads", "6", "--python-exit-code", "1", "--python", str(worker), "--", str(folder)]
    with log.open("wb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise store.WarehouseError(f"opaque surface archive failed ({result.returncode}): {log}")
    for name in ("inner_surface.json", "inner_surface.npz"):
        metadata["files"][name] = store.digest(folder / name)
    read(folder, metadata)
    metadata["ready_for_display"] = False
    write_json_atomic(folder / "warehouse.json", metadata, indent=1)
