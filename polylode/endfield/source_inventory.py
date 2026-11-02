"""Audit each locally exported Endfield FBX against its converted prefab.

This is source accounting, not a claim that a model has a fitted body or a
verified outfit. In particular, mesh names do not establish skin ownership.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import store
from .sample_support import bone_aliases


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _validate_metadata(data) -> None:
    if not isinstance(data, dict):
        raise ValueError('expected an object')
    for key in ('bones', 'parts'):
        rows = data.get(key, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f'{key} must be a list of objects')
    for bone in data.get('bones') or []:
        if not isinstance(bone.get('name'), str) or not bone['name']:
            raise ValueError('bone name is missing')
    for key in ('humanoid', 'materials', 'textures'):
        if not isinstance(data.get(key) or {}, dict):
            raise ValueError(f'{key} must be an object')
    if any(not isinstance(name, str) or not name for name in (data.get('humanoid') or {}).values()):
        raise ValueError('humanoid mappings must contain bone names')
    for material in (data.get('materials') or {}).values():
        if not isinstance(material, dict) or not isinstance(material.get('maps') or {}, dict):
            raise ValueError('materials must contain objects with a map table')
        for mapping in (material.get('maps') or {}).values():
            if isinstance(mapping, dict) and 'file' in mapping and not isinstance(mapping['file'], str):
                raise ValueError('material map file must be a path string')
    for part in data.get('parts') or []:
        names = part.get('materials') or []
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError('part materials must contain names')


def inspect_fbx(root: Path, fbx: Path) -> dict:
    """Return evidence and explicit gaps for one FBX; never infer a prefab id."""
    relative = _relative_path(root, fbx)
    record = {"fbx": relative, "metadata": str(fbx.with_suffix(".json")),
              "source": None, "prefab": None, "array_metadata": None,
              "parts": [], "materials": [],
              "textures": [], "humanoid": {}, "cloth": [], "colliders": {},
              "body_item": None, "gaps": []}
    metadata_path = fbx.with_suffix(".json")
    if not metadata_path.is_file():
        record["gaps"].append("source JSON missing")
        return record
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_metadata(data)
    except (OSError, ValueError, UnicodeError) as exc:
        record["gaps"].append(f"source JSON invalid: {exc}")
        return record
    bones = _inspect_export_metadata(data, fbx, record)
    return _inspect_prefab_metadata(data, record, bones)


def _inspect_export_metadata(data: dict, fbx: Path, record: dict) -> set:
    metadata_path = fbx.with_suffix(".json")
    source = data.get("source")
    record["source"] = source
    # The original exporter used this intermediate archive to build the FBX;
    # the FBX is self-contained and the intermediate is not copied beside it.
    record["array_metadata"] = data.get("arrays")
    bones = {bone["name"] for bone in data.get("bones", [])}
    declared_aliases = [bone for bone in data.get("bones", []) if "node" in bone]
    record["bone_alias_metadata"] = str(metadata_path)
    record["bone_alias_provenance"] = declared_aliases
    try:
        record["bone_aliases"] = bone_aliases(declared_aliases, bones)
    except ValueError as exc:
        record["bone_aliases"] = {}
        record["gaps"].append(f"FBX bone alias table invalid: {exc}")
    humanoid = data.get("humanoid") or {}
    record["humanoid"] = humanoid
    for role, name in humanoid.items():
        if name not in bones:
            record["gaps"].append(f"humanoid bone absent: {role} -> {name}")
    if not humanoid:
        record["gaps"].append("source humanoid mapping missing")

    materials = data.get("materials") or {}
    textures = data.get("textures") or {}
    record["materials"] = sorted(materials)
    record["textures"] = sorted(textures)
    for texture in textures:
        if not (fbx.parent / texture).is_file():
            record["gaps"].append(f"source texture missing: {texture}")
    for name, material in materials.items():
        for slot, mapping in (material.get("maps") or {}).items():
            if not isinstance(mapping, dict):
                record["gaps"].append(f"material map invalid: {name}/{slot}")
                continue
            path = mapping.get("file")
            if not path or path not in textures:
                record["gaps"].append(f"material map unlisted: {name}/{slot}: {path}")
    for part in data.get("parts") or []:
        record["parts"].append({key: part.get(key) for key in
                                ("name", "materials", "hidden", "uvs", "colors",
                                 "shapes", "vertices", "triangles")})
        for name in part.get("materials") or []:
            if name not in materials:
                record["gaps"].append(f"part material missing: {part.get('name')}: {name}")
    if not record["parts"]:
        record["gaps"].append("source has no mesh parts")

    return bones


def _inspect_prefab_metadata(data: dict, record: dict, bones: set) -> dict:
    source = data.get("source")
    humanoid = data.get("humanoid") or {}
    if not isinstance(source, str) or not source.startswith("chara/"):
        record["gaps"].append("no converted character prefab source")
        return record
    ident = source.split("/", 1)[1]
    record["prefab"] = source
    if not store.has("chara", ident):
        record["gaps"].append(f"converted prefab missing: {source}")
        return record
    extra = store.read_extra("chara", ident)
    if extra is None:
        record["gaps"].append(f"converted prefab extra missing: {source}")
        return record
    record["prefab_humanoid"] = extra.get("humanoid") or {}
    record["resolved_prefab_humanoid"] = {
        role: record["bone_aliases"].get(name, name)
        for role, name in record["prefab_humanoid"].items()}
    if record["resolved_prefab_humanoid"] != humanoid:
        record["gaps"].append("FBX and prefab humanoid mappings differ")
    components = extra.get("components") or {}
    record["colliders"] = {key: len(value) for key, value in components.items()
                           if "Collider" in key}
    collider_ids = {str(item.get("id")) for key, values in components.items()
                    if "Collider" in key for item in values}
    for cloth in components.get("BeyondBoneCloth") or []:
        settings = cloth.get("serializeData") or {}
        roots = [item.get("node") for item in settings.get("rootBones") or []]
        renderers = settings.get("sourceRenderers") or []
        refs = ((settings.get("colliderCollisionConstraint") or {}).get("colliderList") or [])
        missing = [str(ref.get("component")) for ref in refs
                   if isinstance(ref, dict) and ref.get("component") is not None
                   and str(ref["component"]) not in collider_ids]
        record["cloth"].append({"node": cloth.get("node"), "cloth_type": settings.get("clothType"),
                                "root_bones": roots, "source_renderers": renderers,
                                "collider_references": sum(isinstance(ref, dict) for ref in refs),
                                "null_collider_slots": sum(ref is None for ref in refs),
                                "missing_colliders": missing})
        for name in roots:
            if name not in bones:
                record["gaps"].append(f"cloth root absent from FBX: {name}")
        if missing:
            record["gaps"].append(f"cloth collider data missing: {cloth.get('node')}: {missing}")
    if not record["cloth"]:
        record["gaps"].append("converted prefab has no BeyondBoneCloth")
    if store.has("body", ident):
        record["body_item"] = f"body/{ident}"
    else:
        record["gaps"].append("fitted body item missing")
    return record


def build_inventory(source_root: Path) -> dict:
    """Enumerate the actual FBX files, preserving multiple versions per folder."""
    root = source_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(root.rglob("*.fbx"))
    if not files:
        raise ValueError(f"no FBX files found in {root}")
    records = [inspect_fbx(root, path) for path in files]
    return {"version": 1, "source_root": str(root), "fbx_count": len(records),
            "prefab_count": sum(item["prefab"] is not None for item in records),
            "cloth_count": sum(len(item["cloth"]) for item in records),
            "records": records}


def write_inventory(source_root: Path, output: Path) -> dict:
    """Write a snapshot under the caller's local derived-content directory."""
    report = build_inventory(source_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
