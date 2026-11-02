"""Run-time side: a baked Endfield item as a ``Model`` on its own ``Skeleton``.

Reads only this project's format, out of the Endfield store (:mod:`.store`).  The skeleton is the prefab's whole node hierarchy, so the item's own
attach nodes and sway bones are there to pose.

**The shading is the game's character lighting as this renderer transcribes
it** (``gl/shaders_endfield.py``; the material table is :mod:`.shading`): the
four ``HGRP/CharacterNPR*`` surface shaders and the overlay shadow.  A piece
whose shader has no counterpart here (the effect shaders, anything unknown) is
**not drawn** and is listed in :attr:`Actor.undrawn`; nothing is drawn in its
place.  A piece drawn without some of its material's features is listed in
:attr:`Actor.partial` with the keywords left out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import math3d as m3
from ..origin import textures as T
from ..mesh import pose_corrective
from ..mesh.mesh import Mesh, Model, Part
from ..mesh.skeleton import Skeleton
from ..mesh.skin import split_archive_skin
from . import shading, store

#: The human bone whose frame the face is lit by.
HEAD = "Head"


class ActorError(LookupError):
    """The item is not in the bake, or is not a complete one."""


@dataclass
class Actor:
    """A loaded item, ready for ``DrawItem(actor.model, skin=…)``."""
    model: Model
    skeleton: Skeleton
    height: float
    #: ``"renderer/material: shader"`` for every piece that is not drawn
    undrawn: list = field(default_factory=list)
    #: what the prefab's components say (:mod:`.components`): ``humanoid``,
    #: ``model``, ``components``, and ``paths`` (node name → binding path)
    extra: dict = field(default_factory=dict)
    #: ``"renderer/material: without <keywords>"`` for every piece drawn
    #: without part of what its material asks for
    partial: list = field(default_factory=list)
    #: ``(part, corrective, rest positions)`` of the parts whose rest shape
    #: follows the pose (``mesh/pose_corrective.py``)
    pose_correctives: list = field(default_factory=list)
    #: Pose-dependent coverings, constructed from declared original source points.
    surface_constructors: list = field(default_factory=list)
    #: the drivers' rotation vectors each of those parts was last shaped for
    _shaped_for: dict = field(default_factory=dict)

    def pose_surfaces(self, skin: np.ndarray) -> None:
        """Give every pose-corrected part the rest shape of this pose.

        *skin* is the skeleton's skinning matrices.  The offsets go on the
        part's rest positions, which the skinning then carries; a part without
        a corrective is not touched, and neither is one whose driver bones
        stand as they did the last time — only a changed shape is uploaded
        again (``Part.stale``).
        """
        if self.surface_constructors:
            frames = self.skeleton.rest_frames()
            for constructor in self.surface_constructors:
                if not getattr(constructor, "after_correctives", False):
                    constructor.update(skin, frames)
        if not self.pose_correctives:
            for constructor in self.surface_constructors:
                if getattr(constructor, "after_correctives", False):
                    constructor.update(skin, self.skeleton.rest_frames())
            return
        frames = self.skeleton.rest_frames()
        parents = np.array([bone.parent for bone in self.skeleton.bones])
        for part, found, rest in self.pose_correctives:
            vectors = pose_corrective.rotation_vectors(skin, frames, parents, found.drivers)
            last = self._shaped_for.get(id(part))
            if last is not None and np.array_equal(last, vectors):
                continue
            self._shaped_for[id(part)] = vectors
            part.mesh.positions = found.corrected(rest, vectors).astype(np.float32)
            part.stale = True
        for constructor in self.surface_constructors:
            if getattr(constructor, "after_correctives", False):
                constructor.update(skin, frames)

    def head_frame(self, world: np.ndarray | None = None) -> np.ndarray | None:
        """The head's world matrix as a turn from its rest, or None without one.

        *world* is the skeleton's evaluated bone matrices (the rest pose when
        ``None``).  The face, eye and hair shading and the character light all
        read a head frame whose rest is the world's own axes
        (``gl/shaders_endfield.py``); the head bone itself rests on whatever
        axes its rig gave it, so the rest is taken out.
        """
        head = (self.extra.get("humanoid") or {}).get(HEAD)
        if head is None or not self.skeleton.has(head):
            return None
        bone = self.skeleton.bones[self.skeleton.index[head]]
        matrix = bone.bind_world if world is None else world[self.skeleton.index[head]]
        rest = m3.quat_conj(m3.decompose(bone.bind_world)[1])
        out = m3.quat_to_mat4(m3.quat_mul(m3.decompose(matrix)[1], rest))
        out[:3, 3] = matrix[:3, 3]
        return out


def build_skeleton(nodes: dict, name: str) -> Skeleton:
    """The item's node table as a skeleton, parents before children."""
    children: dict[str, list] = {}
    for node, (parent, _local) in nodes.items():
        children.setdefault(parent, []).append(node)
    skeleton = Skeleton(name)
    stack = sorted(children.get("", []), reverse=True)
    while stack:
        node = stack.pop()
        parent, local = nodes[node]
        skeleton.add_matrix(node, parent or None, local)
        stack.extend(sorted(children.get(node, []), reverse=True))
    if len(skeleton) != len(nodes):
        raise ActorError(f"{name}: {len(nodes) - len(skeleton)} nodes hang "
                         f"off no root")
    return skeleton.finalize()


def _rest_mesh(piece, skeleton: Skeleton) -> Mesh:
    """The piece's mesh in the skeleton's rest pose, bound to its bones.

    ``Skeleton.skinning_matrices`` is ``world · inverse(rest world)``, so the
    vertices it deforms are rest-pose world positions: each vertex is skinned
    once here through ``bone world · bind``.
    """
    rest = np.stack(piece.bone_world) @ np.asarray(piece.binds, dtype=np.float64)
    skin = np.asarray(piece.skin, dtype=np.float64)
    weights, index = split_archive_skin(skin, bone_count=len(rest))
    blend = np.einsum("vk,vkij->vij", weights, rest[index])
    src = piece.mesh
    pos = np.c_[src.positions.astype(np.float64), np.ones(src.vertex_count)]
    out = src.copy()
    out.positions = np.einsum("vij,vj->vi", blend, pos)[:, :3].astype(np.float32)
    normals = np.einsum("vij,vj->vi", blend[:, :3, :3],
                        src.normals.astype(np.float64))
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    out.normals = (normals / np.where(length > 1e-12, length, 1.0)).astype(np.float32)
    bone_ids = np.array([skeleton.index[b] for b in piece.bones], dtype=np.int32)
    out.bone_idx = bone_ids[index].astype(np.int32)
    out.bone_wt = weights.astype(np.float32)
    return out


def _texture_key(entry: dict, repeat: bool):
    """The store key of one ``maps`` entry; :class:`ActorError` if it is gone.

    *repeat* is the sampler the shader reads the map with — a ramp is clamped
    whatever the texture's own wrap mode says.
    """
    found = store.read_texture(entry["cab"], entry["name"])
    if found is None:
        raise ActorError(f"texture {entry['name']} ({entry['cab']}) is not in "
                         f"the store")
    return T.store(found.px, srgb=found.srgb, repeat=repeat)


def stored_once(read=None):
    """A ``maps`` reader that stores each image once, however many pieces name it.

    A converted source's images are named by their content
    (``converted.runtime_items``), and its renderers share them: a stored copy
    per piece held ten 4096² maps over 27 parts and pushed the texture budget
    into eviction.  *read* is the reader it shares (:func:`_texture_key`).
    """
    read = read or _texture_key
    shared: dict = {}

    def keyed(entry: dict, repeat: bool):
        found = (entry["cab"], entry["name"], repeat)
        if found not in shared:
            shared[found] = read(entry, repeat)
        return shared[found]
    return keyed


def load_actor(category: str, ident: str) -> Actor:
    """Load baked item ``category/ident``; raises :class:`ActorError` if absent."""
    item = store.read_item(category, ident)
    if item is None or not item.nodes:
        raise ActorError(f"{category}/{ident} is not in the store — convert it "
                         f"first (tools/import_endfield.py)")
    skeleton = build_skeleton(item.nodes, ident)
    model = Model(ident, skeleton)
    undrawn, partial = [], []
    content_keyed = stored_once()
    for index, piece in enumerate(item.pieces):
        shader = piece.shader
        part = Part(f"{shader['_rendName']}#{piece.mat_slot}",
                    _rest_mesh(piece, skeleton), group="endfield", layer=index)
        hidden = shader.get("_offNode") or shader.get("_rendOff")
        principled = shading.SE.FAMILIES.get(shader["_shader"]) == shading.SE.FAMILY_PRINCIPLED
        material, note = shading.material(shader, content_keyed if principled else _texture_key)
        label = f"{shader['_rendName']}/{shader['_material']}"
        if material is None:
            part.visible = False
            undrawn.append(f"{label}: {note}")
        else:
            if principled:
                shading.coverage_blended(material, part.mesh.uvs, part.mesh.indices.reshape(-1, 3))
            part.material = material
            part.visible = not hidden
            # the opaque pass draws in this order, and the stencil stamps
            # depend on it: the game's render queue
            part.layer = material.render_queue
            if note:
                partial.append(f"{label}: {note}")
        model.add(part)
    from ..mesh import outline_normals
    outline_normals.fill_parts(model.parts)     # each of the game's meshes has its own (``_OutlineAverageNormal``)
    lo, hi = model.bounds()
    return Actor(model, skeleton, float(hi[1] - lo[1]), undrawn,
                 store.read_extra(category, ident) or {}, partial)
