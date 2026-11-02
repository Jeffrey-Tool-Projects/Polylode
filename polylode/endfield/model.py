"""An actor's model container: its meshes on the skeleton its Avatar carries.

``…/actor/<body type>/<name>/models/sk_*.fbx`` is what Unity made of an
imported model file.  In a built player it holds the meshes and the Avatar and
**no GameObject** — the hierarchy only exists in the prefabs that instantiate
the model.  The Avatar still carries that hierarchy whole:

* ``m_AvatarSkeleton`` — every node's parent and the CRC-32 of its path;
* ``m_TOS``            — hash → path;
* ``m_DefaultPose``    — every node's local transform in the pose the file
                         was imported in,

and a mesh names its bones by the same path hashes (``m_BoneNameHashes``, in
the order of ``m_BindPose``).  That is enough to bind the meshes without a
prefab, which is what the folders with no prefab of their own need (the NPC
body types: their meshes are a library other prefabs pick from).

What the container cannot say is what a mesh is drawn with: a material belongs
to a renderer and there is none here.  The pieces carry no material, and
nothing is made up for them.

A mesh need not have been bound in the Avatar's pose (the NPC bodies are not:
one body mesh is skinned onto skeletons of several proportions), so the shape
at rest is the skinned one — ``bone world · bind`` per vertex, the same as for
a prefab (``actor._rest_mesh``).
"""
from __future__ import annotations

import zlib

from ..origin.prefab import Piece
from . import components as K
from . import prefab as P
from .convert import ConvertError, node_names, nodes_table, slot_mesh
from .unitymesh import read_mesh, to_project

#: where the game keeps an actor's art, one folder per body type and name
ART = "assets/beyond/arts/entity/actor"
#: SQL ``LIKE`` pattern of every model container below :data:`ART`
MODELS = f"{ART}/%/models/%.fbx"
#: Project rig labels; original asset paths remain valid lookup keys.
RIG_LABELS = {"loli": "best_girl"}


class ModelError(ConvertError):
    """A model container whose Avatar and meshes do not add up to a rig."""


def folder_of(container: str) -> str:
    """Project ``<rig label>/<name>`` of a model, or ``""`` for other assets.

    Source ``loli`` rigs use ``best_girl`` in project labels and export folders.
    The container itself stays unchanged for model, clip and morph lookups.
    """
    if not container.startswith(ART + "/") or "/models/" not in container:
        return ""
    folder = container[len(ART) + 1:].split("/models/", 1)[0]
    family, separator, name = folder.partition("/")
    return RIG_LABELS.get(family, family) + separator + name


def skeleton(avatar: dict, name: str) -> P.Prefab:
    """The Avatar's transform hierarchy in its default pose, as a prefab.

    Node ids are the skeleton's indices plus one (0 is "no parent", as in a
    staged prefab); children keep the skeleton's order.  Raises
    :class:`ModelError` when a node's hash is not in ``m_TOS``, is not the
    CRC-32 of the path it stands for, or when the nodes are not one tree.
    """
    body = avatar["m_Avatar"]
    nodes = body["m_AvatarSkeleton"]["m_Node"]
    ids = [int(i) for i in body["m_AvatarSkeleton"]["m_ID"]]
    pose = body["m_DefaultPose"]["m_X"]
    paths = {int(k): str(v) for k, v in avatar["m_TOS"].items()}
    if not len(nodes) == len(ids) == len(pose):
        raise ModelError(f"{name}: {len(nodes)} nodes, {len(ids)} ids and "
                         f"{len(pose)} poses")
    out: dict[int, P.Node] = {}
    for index, (node, ident, x) in enumerate(zip(nodes, ids, pose)):
        path = paths.get(ident)
        if path is None or zlib.crc32(path.encode("utf-8")) != ident:
            raise ModelError(f"{name}: skeleton node {index} (hash {ident}) "
                             f"has no path in m_TOS")
        parent = int(node["m_ParentId"])
        if parent >= index:
            raise ModelError(f"{name}: node {index} comes before its parent")
        if parent >= 0 and path.rpartition("/")[0] != paths[ids[parent]]:
            raise ModelError(f"{name}: {path!r} is not a child of "
                             f"{paths[ids[parent]]!r}")
        local = P.local_matrix({"m_LocalPosition": x["t"],
                                "m_LocalRotation": x["q"],
                                "m_LocalScale": x["s"]})
        out[index + 1] = P.Node(index + 1, path.rpartition("/")[2] or name,
                                parent + 1, local, True, frozenset())
        if parent >= 0:
            out[parent + 1].children.append(index + 1)
    roots = [tid for tid, node in out.items() if node.parent == 0]
    if len(roots) != 1:
        raise ModelError(f"{name}: {len(roots)} root nodes, expected 1")
    return P.Prefab("", roots[0], out, [])


def _bones(mesh: dict, by_hash: dict[int, int]) -> list[int]:
    """Node id of each of the mesh's bones, in bind-pose order."""
    hashes = [int(h) for h in mesh.get("m_BoneNameHashes") or []]
    missing = [h for h in hashes if h not in by_hash]
    if missing:
        raise ModelError(f"{mesh.get('m_Name')}: {len(missing)} of its "
                         f"{len(hashes)} bones are not in the Avatar")
    return [by_hash[h] for h in hashes]


def build(avatar: dict, meshes: list[dict], name: str,
          container: str = "") -> tuple[list, dict, dict]:
    """``(pieces, nodes, extra)`` for a model container's decoded objects.

    One piece per submesh, in mesh-name order; ``shader`` holds the mesh's
    name and nothing a material would say.  *extra* is what a converted prefab
    stores beside its pieces: node paths and the human bones.
    """
    prefab = skeleton(avatar, name)
    names = node_names(prefab)
    worlds = P.world_matrices(prefab)
    ids = [int(i) for i in avatar["m_Avatar"]["m_AvatarSkeleton"]["m_ID"]]
    by_hash = {ident: index + 1 for index, ident in enumerate(ids)}
    pieces = []
    for doc in sorted(meshes, key=lambda d: str(d.get("m_Name"))):
        mesh = read_mesh(doc)
        if mesh.skin is None or mesh.binds is None:
            raise ModelError(f"{mesh.name}: a model's mesh with no skin")
        bones = _bones(doc, by_hash)
        if len(bones) != len(mesh.binds):
            raise ModelError(f"{mesh.name}: {len(bones)} bones for "
                             f"{len(mesh.binds)} bind poses")
        if not mesh.submeshes:
            raise ModelError(f"{mesh.name}: no submesh")
        for slot in range(len(mesh.submeshes)):
            pieces.append(Piece(
                mesh=slot_mesh(mesh, slot, mesh.name),
                bones=[names[tid] for tid in bones], skin=mesh.skin,
                binds=mesh.binds,
                bone_world=[to_project(worlds[tid]) for tid in bones],
                shader={"_rendName": mesh.name, "_material": "", "_shader": "",
                        "_submeshes": len(mesh.submeshes), "_rigid": 0,
                        "maps": {}},
                mat_slot=slot,
                shapes=mesh.shapes if slot == 0 and mesh.shapes else None))
    if not pieces:
        raise ModelError(f"{name}: the container holds no mesh")
    extra = {"paths": {names[tid]: P.node_path(prefab, tid)
                       for tid in prefab.nodes},
             "humanoid": K.humanoid(avatar, prefab, names),
             "human": K.human_definition(avatar, prefab, names)}
    if container:
        extra["model"] = container
    return pieces, nodes_table(prefab, names), extra
