"""A staged prefab: its transform hierarchy, its renderers and its LOD levels.

Everything is read off the serialized objects — which mesh a renderer draws,
which bones skin it, which renderers are the full-detail set (``LODGroup``'s
level 0, not a node called ``lod0``) and which nodes are switched off.  All
matrices here are still **Unity's** (left-handed); :mod:`.convert` mirrors
them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .cabmap import CabIndex, Ref
from .staged import Staged

#: ``Renderer.m_CastShadows`` value for ``ShadowCastingMode.ShadowsOnly`` — the
#: renderer writes the shadow map and is never drawn into the picture.
SHADOWS_ONLY = 3

_ACTIVE = re.compile(r"^\s*bool m_IsActive = (True|False)\s*$", re.M)


class PrefabError(RuntimeError):
    """The staged objects do not add up to one prefab."""


@dataclass
class Renderer:
    """A ``SkinnedMeshRenderer`` or a ``MeshRenderer`` + ``MeshFilter`` pair."""
    skinned: bool
    mesh: Ref | None
    materials: list            # Ref | None per material slot
    bones: list                # Transform path ids, in the mesh's bone order
    enabled: bool
    cast_shadows: int


@dataclass
class Node:
    """One GameObject, keyed by its Transform's path id."""
    tid: int
    name: str
    parent: int                # Transform path id; 0 at the prefab root
    local: np.ndarray          # 4×4, Unity space
    active: bool
    components: frozenset      # path ids of every component on the GameObject
    renderer: Renderer | None = None
    children: list = field(default_factory=list)


@dataclass
class Prefab:
    """Every node of one serialized file's hierarchy."""
    cab: str
    root: int
    nodes: dict                # Transform path id → Node
    #: per ``LODGroup``: one set of renderer component path ids per level
    lods: list


def _vec(d: dict, keys: str) -> list:
    return [float(d[k]) for k in keys]


def local_matrix(transform: dict) -> np.ndarray:
    """``m_LocalPosition`` / ``m_LocalRotation`` / ``m_LocalScale`` as a 4×4."""
    x, y, z, w = _vec(transform["m_LocalRotation"], "XYZW")
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    out = np.eye(4)
    out[:3, :3] = rot * np.array(_vec(transform["m_LocalScale"], "XYZ"))
    out[:3, 3] = _vec(transform["m_LocalPosition"], "XYZ")
    return out


def _renderer(cabs: CabIndex, cab: str, go: dict) -> Renderer | None:
    smr = go.get("m_SkinnedMeshRenderer")
    if isinstance(smr, dict):
        return Renderer(
            True, cabs.pointer(cab, smr.get("m_Mesh")),
            [cabs.pointer(cab, m) for m in smr.get("m_Materials") or []],
            [int(b["m_PathID"]) for b in smr.get("m_Bones") or []],
            bool(smr.get("m_Enabled", True)), int(smr.get("m_CastShadows", 1)))
    rigid, holder = go.get("m_MeshRenderer"), go.get("m_MeshFilter")
    if isinstance(rigid, dict):
        mesh = holder.get("m_Mesh") if isinstance(holder, dict) else None
        return Renderer(
            False, cabs.pointer(cab, mesh),
            [cabs.pointer(cab, m) for m in rigid.get("m_Materials") or []],
            [], bool(rigid.get("m_Enabled", True)),
            int(rigid.get("m_CastShadows", 1)))
    return None


def _active(staged: Staged, ref: Ref) -> bool:
    found = _ACTIVE.search(staged.text(ref))
    if found is None:
        raise PrefabError(f"no m_IsActive in the dump of GameObject {ref}")
    return found.group(1) == "True"


def read_prefab(staged: Staged, cabs: CabIndex, cab: str) -> Prefab:
    """Assemble the hierarchy staged from serialized file *cab*.

    Raises :class:`PrefabError` unless the GameObjects form exactly one tree —
    a second root would be a second prefab, and there is no rule here for
    which one a caller meant.
    """
    owner: dict[int, int] = {}              # GameObject path id → Transform's
    for ref in staged.of_type("Transform", cab):
        owner[int(staged.json(ref)["m_GameObject"]["m_PathID"])] = ref.path_id
    nodes: dict[int, Node] = {}
    for ref in staged.of_type("GameObject", cab):
        go = staged.json(ref)
        tid = owner.get(ref.path_id)
        if tid is None:
            raise PrefabError(f"GameObject {go.get('m_Name')!r} in {cab} has "
                              f"no staged Transform")
        transform = go["m_Transform"]
        nodes[tid] = Node(
            tid, str(go["m_Name"]), int(transform["m_Father"]["m_PathID"]),
            local_matrix(transform), _active(staged, ref),
            frozenset(int(c["m_PathID"]) for c in go.get("m_Components") or []),
            _renderer(cabs, cab, go),
            [int(c["m_PathID"]) for c in transform.get("m_Children") or []])
    roots = [n.tid for n in nodes.values() if n.parent not in nodes]
    if len(roots) != 1:
        raise PrefabError(f"{cab}: {len(roots)} root GameObjects, expected 1")
    for node in nodes.values():
        # Unity's own child order (``m_Children``): it is what decides which of
        # two same-named siblings a path means
        fathered = {t for t, n in nodes.items() if n.parent == node.tid}
        if set(node.children) != fathered:
            raise PrefabError(f"{node.name}: m_Children lists {len(node.children)} "
                              f"nodes, {len(fathered)} name it as their father")
    lods = [[{int(r["renderer"]["m_PathID"]) for r in level.get("renderers") or []}
             for level in staged.json(ref).get("m_LODs") or []]
            for ref in staged.of_type("LODGroup", cab)]
    return Prefab(cab, roots[0], nodes, lods)


def lod_level(prefab: Prefab, node: Node) -> int | None:
    """The lowest ``LODGroup`` level that lists *node*'s renderer, or ``None``.

    ``None`` means no group owns the renderer, so Unity draws it at every
    distance.  A renderer may sit in several levels; the lowest is the most
    detailed one it is part of.
    """
    levels = [index for group in prefab.lods for index, members in enumerate(group)
              if members & node.components]
    return min(levels) if levels else None


def world_matrices(prefab: Prefab) -> dict[int, np.ndarray]:
    """Transform path id → world 4×4 (Unity space), the prefab root at identity.

    The root's own local transform is where the prefab is *placed*, not part of
    its shape, and an item is baked at the origin.
    """
    out = {prefab.root: np.eye(4)}
    stack = [prefab.root]
    while stack:
        tid = stack.pop()
        for child in prefab.nodes[tid].children:
            out[child] = out[tid] @ prefab.nodes[child].local
            stack.append(child)
    return out


def node_path(prefab: Prefab, tid: int) -> str:
    """The node's path below the prefab root, as Unity writes a binding path.

    ``Root/Bip001/Bip001_Pelvis``; the root itself is ``""``.  An animation
    clip addresses a transform by the CRC-32 of this string.
    """
    parts = []
    while tid != prefab.root:
        node = prefab.nodes[tid]
        parts.append(node.name)
        tid = node.parent
    return "/".join(reversed(parts))


def find_path(prefab: Prefab, path: str) -> int | None:
    """Transform path id of the node at *path* below the root, or ``None``.

    Where siblings share a name the first child wins, which is how Unity's own
    ``Transform.Find`` resolves it.
    """
    tid = prefab.root
    for name in (path.split("/") if path else []):
        match = [c for c in prefab.nodes[tid].children
                 if prefab.nodes[c].name == name]
        if not match:
            return None
        tid = match[0]
    return tid


def inactive_under(prefab: Prefab, node: Node) -> str:
    """Name of the nearest switched-off node at or above *node*, ``""`` if none."""
    while True:
        if not node.active:
            return node.name
        if node.parent not in prefab.nodes:
            return ""
        node = prefab.nodes[node.parent]
