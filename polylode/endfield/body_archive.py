"""Lossless original Origin body archives for editable Endfield handoffs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def store_root(ident: str) -> Path:
    """The store holding model *ident* (its ``Charas`` folder, else the shared store)."""
    from . import store

    return store.home(ident)


def character_extra(ident: str) -> dict:
    from . import store

    return store.read_extra("chara", ident) or {}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _map_archive(root: Path, output: Path, metadata: dict) -> dict:
    """Copy already encoded source maps without loading texture pixels."""
    paths = set(metadata.get("extras", {}).values())
    for piece in metadata["pieces"]:
        paths.update(piece.get("maps", {}).values())
    result = {}
    for relative in sorted(paths):
        source = (root / relative).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file():
            raise FileNotFoundError(
                f"original Origin map missing or outside store: {relative}"
            )
        destination = output / "maps" / source.relative_to(root.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        result[relative] = {
            "source_file": relative,
            "archive_file": destination.relative_to(output).as_posix(),
            "sha256": file_digest(destination),
        }
    return result


def export_original_archive(ident: str, output: Path) -> dict:
    """Return job fields for the unchanged body_source NPZ and encoded maps."""
    from .body_scope import require_body_scope

    require_body_scope(character_extra(ident), ident)
    root = store_root(ident).resolve()
    index = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["items"]
    source_key = f"body_source/{ident}"
    entry = index.get(source_key)
    if not entry or not entry.get("file"):
        raise FileNotFoundError(f"original Origin source archive missing: {source_key}")
    source = (root / entry["file"]).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise FileNotFoundError(f"original Origin source archive missing: {source}")
    destination = output.resolve() / "original_source_archive"
    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / "source.npz"
    shutil.copyfile(source, arrays_path)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        metadata = json.loads(bytes(arrays["meta"]).decode("utf-8"))
        streams = list(arrays.files)
        hierarchy = bool(metadata.get("nodes")) and "nodes" in streams
    if metadata.get("category") != "body_source" or not metadata.get("pieces"):
        raise ValueError(f"not an original Origin body_source archive: {source}")
    body_entry = index.get(f"table/endfield/body/{ident}", {})
    body = (
        json.loads((root / body_entry["file"]).read_text(encoding="utf-8"))
        if body_entry.get("file")
        else {}
    )
    limits = [
        "Original mesh-space data is archived without fitting or Endfield rig binding.",
        "Source transforms are reference objects; no replacement deform rig is invented.",
    ]
    if not hierarchy:
        limits.append(
            "Original source hierarchy is unavailable; exact declared bind/world matrices remain in the embedded NPZ."
        )
    digest = file_digest(arrays_path)
    manifest = {
        "version": 1,
        "source_item": source_key,
        "source_archive": str(source),
        "source_archive_sha256": digest,
        "original_metadata": metadata,
        "streams": streams,
        "hierarchy_available": hierarchy,
        "limits": limits,
        "provenance": {
            key: body.get(key) for key in ("source_card", "source_body", "rig_family")
        },
        "map_files": _map_archive(root, destination, metadata),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "original_source_archive": {
            "version": 1,
            "arrays": str(arrays_path),
            "manifest": str(manifest_path),
            "sha256": digest,
        }
    }


def piece_arrays(arrays, index: int, metadata: dict) -> dict:
    """Read one exact piece; numeric skin indices address original bone slots."""
    names = {
        "positions": "pos",
        "indices": "idx",
        "normals": "nrm",
        "uvs": "uv",
        "skin": "skin",
        "binds": "binds",
        "bone_world": "bworld",
        "bone_world_present": "bworld_ok",
        "uv1": "uv1",
        "uv2": "uv2",
        "uv3": "uv3",
        "vcolor": "col",
    }
    result = {
        name: arrays[f"{index}_{suffix}"]
        for name, suffix in names.items()
        if f"{index}_{suffix}" in arrays
    }
    for required in ("positions", "indices", "normals", "uvs"):
        if required not in result:
            raise ValueError(f"original Origin piece {index}: missing {required}")
    count = len(result["positions"])
    for name, width in (
        ("positions", 3),
        ("normals", 3),
        ("uvs", 2),
        ("uv1", 2),
        ("uv2", 2),
        ("uv3", 2),
        ("vcolor", 4),
    ):
        if name in result and (
            result[name].shape != (count, width) or not np.isfinite(result[name]).all()
        ):
            raise ValueError(f"original Origin piece {index}: invalid {name} topology")
    indices = result["indices"]
    if (
        indices.ndim != 1
        or len(indices) % 3
        or not np.issubdtype(indices.dtype, np.integer)
        or (len(indices) and (indices.min() < 0 or indices.max() >= count))
    ):
        raise ValueError(f"original Origin piece {index}: invalid triangle indices")
    if "skin" in result:
        skin = result["skin"]
        if skin.shape != (count, 8) or not np.isfinite(skin).all():
            raise ValueError(f"original Origin piece {index}: invalid skin topology")
        weights, slots = skin[:, :4], skin[:, 4:]
        used = slots[weights > 0]
        if np.any(weights < 0) or np.any(weights > 1):
            raise ValueError(f"original Origin piece {index}: invalid bone weight")
        if (
            np.any(used != np.floor(used))
            or np.any(used < 0)
            or np.any(used >= len(metadata.get("bones", [])))
        ):
            raise ValueError(f"original Origin piece {index}: invalid bone index")
    result["shapes"] = []
    for shape, name in enumerate(metadata.get("shapes", [])):
        prefix = f"{index}_shp{shape}_"
        result["shapes"].append(
            {
                "name": name,
                "indices": arrays[prefix + "i"],
                "deltas": arrays[prefix + "d"],
                "normals": arrays.get(prefix + "n"),
            }
        )
    return result
