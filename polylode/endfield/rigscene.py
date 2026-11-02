"""A converted item as a rigged scene another program can write to a file.

The store keeps a prefab the way this renderer draws it: one piece per
material slot, vertices in bind space, the prefab's whole node hierarchy.  A
model file wants it the other way round — one mesh per renderer, vertices at
rest, an armature of the bones only.  :func:`scene_of` does that regrouping
and nothing else; ``tools/ef_fbx_blender.py`` turns the result into an FBX.

* **Bones** are every node of each top-level branch that holds a bone some
  mesh is skinned to (:func:`rig_nodes`).  That keeps the attach points, IK
  targets and collider nodes that hang off the skeleton and leaves out the
  branches that are not it (the LOD mesh nodes, the shadow proxies).  A bone's
  rest is the node's world transform with its scale taken out; its axes are
  the node's own, so a pose or a clip made for the game's rig still fits.
* **A part** is one renderer: the vertices skinned once through
  ``bone world · bind`` (the prefab need not rest in the pose a mesh was bound
  in), one triangle list with the material slot of every triangle, and the
  four weights per vertex as indices into the scene's bones.  A material past
  the renderer's last submesh draws that submesh a second time; a file has one
  material per triangle, so such a material is listed on the part and owns no
  triangle.
* **Space**: right-handed, Z up, metres, the character facing −Y — this
  project's space (:data:`TO_Z_UP` is a rotation, not a mirror, so the
  triangle winding stands).

Nothing is invented: a part whose pieces carry no material gets none, and a
material's maps are listed under the shader's own property names with only the
base colour singled out (:func:`shading.base_map`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..mesh.skin import split_archive_skin
from ..mesh.tangent import directions, validate as validate_tangents

from . import shading
from .axes import TO_Z_UP  # noqa: F401 - this module's space, named here for its users

#: the longest bone name the writer's host keeps (bytes of UTF-8)
NAME_LIMIT = 63
#: a bone's drawn length, metres: bounds, and what a childless root gets
LENGTH_RANGE = (0.005, 0.25)
LENGTH_ALONE = 0.05
#: where a scene's textures go, relative to the scene file
TEXTURE_DIR = "textures"
#: shaders drawn as a tinted coverage multiplied onto what is behind them (the game's FAMILY_OVERLAY)
OVERLAY_SHADERS = frozenset({"HGRP/CharacterNPR_OverlayShadow"})
UV_STREAMS = ("uvs", "uv1", "uv2", "uv3")


class SceneError(ValueError):
    """Pieces and nodes that do not add up to one rigged scene."""


@dataclass
class Bone:
    name: str                  # unique, at most :data:`NAME_LIMIT` bytes
    node: str                  # the item's node it stands for
    parent: int                # index into ``Scene.bones``; -1 at a root
    rest: np.ndarray           # 4×4 world, rotation and translation only
    length: float


@dataclass
class Part:
    name: str
    positions: np.ndarray      # (N, 3) at rest
    normals: np.ndarray        # (N, 3)
    uvs: list                  # four entries, (N, 2) or None, by UV channel
    colors: np.ndarray | None  # (N, 4) vertex colours as stored
    triangles: np.ndarray      # (T, 3)
    slots: np.ndarray          # (T,) material slot of each triangle
    materials: list            # material name per slot ("" = none)
    joints: np.ndarray         # (N, 4) indices into ``Scene.bones``
    weights: np.ndarray        # (N, 4)
    shapes: list = field(default_factory=list)   # (name, indices, deltas)
    hidden: bool = False       # the prefab ships this renderer switched off
    tangents: np.ndarray | None = None  # authored rest xyz and separate source handedness


@dataclass
class Scene:
    name: str
    bones: list
    parts: list
    materials: dict            # name → description (:func:`_material`)
    textures: dict             # file name → (serialized file, texture name)


def _cut(text: str, limit: int) -> str:
    """*text* shortened to at most *limit* bytes of UTF-8, on a character."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[:limit].decode("utf-8", errors="ignore")


def unique_names(keys: list, limit: int = NAME_LIMIT) -> dict:
    """A distinct name of at most *limit* bytes for each of *keys*, in order.

    The name is the key's last path segment; the second and later holders of
    one name get ``.2``, ``.3`` … (a cousin the store keys by its path, or a
    name that only differed past the limit).
    """
    out, taken = {}, set()
    for key in keys:
        base = _cut(key.rpartition("/")[2], limit)
        name, count = base, 1
        while name in taken:
            count += 1
            suffix = f".{count}"
            name = _cut(base, limit - len(suffix)) + suffix
        taken.add(name)
        out[key] = name
    return out


def rig_nodes(nodes: dict, used: set) -> list:
    """The nodes that make the rig, parents before children.

    *nodes* is the item's ``name → (parent, local)`` table and *used* the
    nodes some mesh is skinned to.  The root stands for the object itself and
    is a bone only when a mesh is bound to it.
    """
    children: dict[str, list] = {}
    for node, (parent, _local) in nodes.items():
        children.setdefault(parent, []).append(node)
    roots = children.get("", [])
    if len(roots) != 1:
        raise SceneError(f"{len(roots)} root nodes, expected 1")
    out = [roots[0]] if roots[0] in used else []
    for top in sorted(children.get(roots[0], [])):
        branch, stack = [], [top]
        while stack:
            node = stack.pop()
            branch.append(node)
            stack.extend(sorted(children.get(node, []), reverse=True))
        if used.intersection(branch):
            out += branch
    stray = sorted(used.difference(out))
    if stray:
        raise SceneError(f"skinned to nodes outside the hierarchy: {stray[:3]}")
    return out


def world_matrices(nodes: dict) -> dict:
    """Node name → world 4×4 in this project's space, scale included."""
    out: dict[str, np.ndarray] = {}

    def world(node: str) -> np.ndarray:
        if node not in out:
            parent, local = nodes[node]
            local = np.asarray(local, dtype=np.float64)
            out[node] = local if not parent else world(parent) @ local
        return out[node]

    for node in nodes:
        world(node)
    return out


def rest_matrix(world: np.ndarray) -> np.ndarray:
    """*world* in the scene's space with its scale taken out.

    The rotation is the orthonormal matrix nearest the 3×3 (its polar factor),
    which is the 3×3 itself for a node that is only scaled uniformly.
    """
    moved = TO_Z_UP @ np.asarray(world, dtype=np.float64)
    u, _s, vt = np.linalg.svd(moved[:3, :3])
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] = -u[:, -1]
    out = np.eye(4)
    out[:3, :3] = u @ vt
    out[:3, 3] = moved[:3, 3]
    return out


def _bones(nodes: dict, kept: list) -> list:
    worlds = world_matrices(nodes)
    names = unique_names(kept)
    index = {node: i for i, node in enumerate(kept)}
    bones = [Bone(names[node], node, index.get(nodes[node][0], -1),
                  rest_matrix(worlds[node]), 0.0) for node in kept]
    reach: dict[int, list] = {}
    for bone in bones:
        if bone.parent >= 0:
            gap = float(np.linalg.norm(bone.rest[:3, 3] - bones[bone.parent].rest[:3, 3]))
            reach.setdefault(bone.parent, []).append(gap)
    for i, bone in enumerate(bones):
        # to its nearest child; a leaf takes after its parent
        gaps = [g for g in reach.get(i, []) if g > 1e-4]
        if gaps:
            length = min(gaps)
        elif bone.parent >= 0:
            length = bones[bone.parent].length
        else:
            length = LENGTH_ALONE
        bone.length = float(np.clip(length, *LENGTH_RANGE))
    return bones


def renderers(pieces: list) -> list:
    """*pieces* grouped per renderer: a run starts at material slot 0."""
    out: list[list] = []
    for piece in pieces:
        if piece.mat_slot == 0 or not out:
            out.append([])
        out[-1].append(piece)
    return out


def _stream(mesh, name: str):
    values = getattr(mesh, name, None)
    if values is None or len(values) != mesh.vertex_count or not mesh.vertex_count:
        return None
    return np.asarray(values, dtype=np.float32)


def _part(group: list, name: str, bone_index: dict, materials: list) -> Part:
    first = group[0]
    mesh = first.mesh
    rest = np.stack(first.bone_world) @ np.asarray(first.binds, dtype=np.float64)
    skin = np.asarray(first.skin, dtype=np.float64)
    weights, local = split_archive_skin(skin, bone_count=len(rest))
    blend = np.einsum("vk,vkij->vij", weights, rest[local])
    turn = TO_Z_UP[:3, :3]
    homogeneous = np.c_[mesh.positions.astype(np.float64), np.ones(mesh.vertex_count)]
    positions = np.einsum("vij,vj->vi", blend, homogeneous)[:, :3] @ turn.T
    normals = np.einsum("vij,vj->vi", blend[:, :3, :3],
                        mesh.normals.astype(np.float64)) @ turn.T
    size = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(size > 1e-12, size, 1.0)
    validate_tangents(mesh.tangents, mesh.vertex_count)
    tangents = None
    if len(mesh.tangents):
        moved = directions(mesh.tangents[:, :3].astype(np.float64), blend[:, :3, :3]) @ turn.T
        tangents = np.c_[moved, mesh.tangents[:, 3]].astype(np.float32)
    table = np.array([bone_index[b] for b in first.bones], dtype=np.int32)
    submeshes = int(first.shader["_submeshes"])
    triangles, slots = [], []
    for piece in group:
        if piece.mat_slot >= submeshes:
            continue                    # a second pass over the last submesh
        tri = np.asarray(piece.mesh.indices, dtype=np.int64).reshape(-1, 3)
        triangles.append(tri)
        slots.append(np.full(len(tri), piece.mat_slot, dtype=np.int32))
    shapes = []
    for shape in first.shapes or []:
        index = np.asarray(shape[1], dtype=np.int64)
        delta = np.einsum("vij,vj->vi", blend[index, :3, :3],
                          np.asarray(shape[2], dtype=np.float64)) @ turn.T
        shapes.append((str(shape[0]), index.astype(np.int32), delta.astype(np.float32)))
    shader = first.shader
    return Part(
        name=name, positions=positions.astype(np.float32),
        normals=normals.astype(np.float32),
        uvs=[_stream(mesh, s) for s in UV_STREAMS], colors=_stream(mesh, "vcolor"),
        triangles=(np.concatenate(triangles) if triangles
                   else np.zeros((0, 3), dtype=np.int64)).astype(np.int32),
        slots=(np.concatenate(slots) if slots else np.zeros(0, dtype=np.int32)),
        materials=materials, joints=np.where(weights > 0, table[local], 0).astype(np.int32),
        weights=weights.astype(np.float32), shapes=shapes,
        hidden=bool(shader.get("_offNode") or shader.get("_rendOff")), tangents=tangents)


def _material(shader: dict, files: dict) -> dict:
    """One material as the file's sidecar lists it; *files* names its maps."""
    maps = {prop: {"file": files[(entry["cab"], entry["name"])],
                   "srgb": bool(entry["srgb"]), "scale": list(entry["scale"]),
                   "offset": list(entry["offset"])}
            for prop, entry in sorted(shader["maps"].items())}
    record = {"shader": shader["_shader"], "queue": shader.get("_queue"),
              "keywords": list(shader.get("keywords") or []),
              "floats": dict(shader.get("floats") or {}),
              "colors": dict(shader.get("colors") or {}),
              "maps": maps, "base": shading.base_map(shader)}
    if shader["_shader"] in OVERLAY_SHADERS:
        # how the game draws it (gl/shaders_endfield.efOverlay): the base map's red as coverage when
        # _UseGrayAsAlpha, white as colour, times _BaseColor, multiplied onto what is behind it; said in
        # the writers' terms, so the .glb and FBX do not draw the mask itself as an opaque colour (§7.1455)
        record["tint"] = [float(v) for v in record["colors"].get("_BaseColor", (1.0, 1.0, 1.0, 1.0))]
        record["alpha_from"] = "r" if record["floats"].get("_UseGrayAsAlpha", 0.0) >= 0.5 else "a"
        record["blend"] = "multiply"
    return record


def _named(wanted: str, record: dict, taken: dict) -> str:
    """*wanted*, or it with ``.2`` … when another material already has the name."""
    name, count = wanted, 1
    while name in taken and taken[name] != record:
        count += 1
        name = f"{wanted}.{count}"
    taken[name] = record
    return name


def scene_of(pieces: list, nodes: dict, name: str) -> Scene:
    """The item as one armature and one part per renderer.

    Raises :class:`SceneError` for an item with no piece, or with a renderer
    skinned to a node the hierarchy does not hold (:func:`rig_nodes`).
    """
    if not pieces:
        raise SceneError(f"{name}: nothing to write")
    groups = renderers(pieces)
    used = {bone for group in groups for bone in group[0].bones}
    bones = _bones(nodes, rig_nodes(nodes, used))
    bone_index = {bone.node: i for i, bone in enumerate(bones)}
    files: dict[tuple, str] = {}
    by_file: dict[str, tuple] = {}
    for piece in pieces:
        for entry in piece.shader["maps"].values():
            key = (entry["cab"], entry["name"])
            if key in files:
                continue
            file = f"{entry['name']}.png"
            if file in by_file:         # two files each hold a texture of this name
                file = f"{entry['name']}~{entry['cab'][-8:]}.png"
            files[key] = f"{TEXTURE_DIR}/{file}"
            by_file[file] = key
    materials: dict[str, dict] = {}
    parts, seen = [], {}
    for group in groups:
        slots = []
        for piece in group:
            if not piece.shader["_material"]:
                slots.append("")
                continue
            record = _material(piece.shader, files)
            slots.append(_named(piece.shader["_material"], record, materials))
        base = group[0].shader["_rendName"]
        seen[base] = seen.get(base, 0) + 1
        label = base if seen[base] == 1 else f"{base}.{seen[base]}"
        parts.append(_part(group, label, bone_index, slots))
    return Scene(name, bones, parts, materials, by_file)


def write(scene: Scene, folder: Path, stem: str, extra: dict | None = None,
          source: str = "") -> Path:
    """``<stem>.json`` (structure) and ``<stem>.npz`` (arrays) in *folder*.

    The JSON is what stays beside the model file: bone → node path, the human
    bones, and every material with its shader, keywords, values and maps.
    *extra* is the item's stored ``extra`` (``paths``, ``humanoid``) and
    *source* says where the scene came from.  Returns the JSON's path.
    """
    extra = extra or {}
    paths = extra.get("paths") or {}
    by_node = {bone.node: bone.name for bone in scene.bones}
    arrays = {
        "bone_rest": np.stack([b.rest for b in scene.bones]),
        "bone_parent": np.array([b.parent for b in scene.bones], dtype=np.int32),
        "bone_length": np.array([b.length for b in scene.bones], dtype=np.float32)}
    parts = []
    for i, part in enumerate(scene.parts):
        arrays.update({f"p{i}_co": part.positions, f"p{i}_no": part.normals,
                       f"p{i}_tri": part.triangles, f"p{i}_slot": part.slots,
                       f"p{i}_jnt": part.joints, f"p{i}_wgt": part.weights})
        channels = []
        for channel, uv in enumerate(part.uvs):
            if uv is not None:
                arrays[f"p{i}_uv{channel}"] = uv
                channels.append(channel)
        if part.colors is not None:
            arrays[f"p{i}_col"] = part.colors
        if part.tangents is not None:
            validate_tangents(part.tangents, len(part.positions))
            arrays[f"p{i}_tan"] = part.tangents
        for k, (_name, index, delta) in enumerate(part.shapes):
            arrays[f"p{i}_s{k}_idx"] = index
            arrays[f"p{i}_s{k}_delta"] = delta
        parts.append({"name": part.name, "materials": part.materials,
                      "hidden": part.hidden, "uvs": channels,
                      "colors": part.colors is not None,
                      "tangents": part.tangents is not None,
                      "shapes": [s[0] for s in part.shapes],
                      "vertices": int(len(part.positions)),
                      "triangles": int(len(part.triangles))})
    document = {
        "name": scene.name,
        "source": source,
        "arrays": f"{stem}.npz",
        "bones": [{"name": b.name, "node": b.node, "path": paths.get(b.node, ""),
                   "parent": b.parent} for b in scene.bones],
        "humanoid": {human: by_node[node]
                     for human, node in (extra.get("humanoid") or {}).items()
                     if node in by_node},
        "parts": parts,
        "materials": scene.materials,
        "textures": {f"{TEXTURE_DIR}/{file}": list(key)
                     for file, key in sorted(scene.textures.items())},
    }
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{stem}.npz", **arrays)
    target = folder / f"{stem}.json"
    target.write_text(json.dumps(document, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    return target
