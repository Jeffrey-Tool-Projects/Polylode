"""Staged prefab → ``origin.prefab.Piece`` list → ``store.write_item``.

One piece per (renderer, material slot), in the bake's existing shape:

* ``mesh``      positions in the mesh's own (bind) space, mirrored into this
                project's handedness, with the slot's submesh as its indices;
* ``bones`` / ``skin`` / ``binds`` / ``bone_world``
                the renderer's bone nodes by name, the mesh's weights and bind
                poses, and each bone's matrix in the prefab at rest.  A rigid
                renderer is bound to its own node with weight 1 and an identity
                bind, so one code path poses both kinds;
* ``shader``    the material, uninterpreted: shader name, keywords, queue,
                every float and colour, and ``maps`` — each bound texture under
                the shader's own property name.

Textures go to the store's named textures once per texture (keyed by the serialized
file they came from and their name), not into the piece: a character's
materials share ramps and look-up tables, and one copy means one answer.
The item's ``nodes`` table is the prefab's whole hierarchy, so a loader can
rebuild the skeleton, attach points included.

A reference that cannot be followed raises :class:`ConvertError`.  Nothing is
substituted and a prefab that fails is not written at all.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..origin import colorspace as CS
from ..origin.prefab import Piece
from ..mesh.mesh import Mesh
from . import prefab as P
from . import store
from .cabmap import CabIndex
from .material import TextureInfo, UnityMaterial, read_material, read_texture_info
from .staged import Ref, Staged
from .unitymesh import UnityMesh, read_mesh, to_project

class ConvertError(RuntimeError):
    """A prefab that cannot be converted faithfully."""


class NothingToConvert(ConvertError):
    """A prefab with no mesh renderer at all — an effect, not a model.

    Kept apart from a failure: a pass over a folder that also holds effect
    prefabs would otherwise end in an error every time, and an exit status
    that is always 1 says nothing.
    """


@dataclass
class Report:
    """What one prefab turned into."""
    renderers: int = 0
    pieces: int = 0
    vertices: int = 0
    triangles: int = 0
    textures: int = 0
    #: renderers the prefab holds but Unity would not draw at full detail
    skipped: Counter = field(default_factory=Counter)
    #: largest ``|bone world · bind − mean|`` over a skinned renderer's bones:
    #: 0 when the prefab rests in the pose its meshes were bound in
    bind_spread: float = 0.0


def drawn(prefab: P.Prefab) -> tuple[list, Counter]:
    """The renderers Unity draws at full detail, and why the others are not.

    Out: a renderer that only writes the shadow map
    (``ShadowCastingMode.ShadowsOnly``) and one whose most detailed
    ``LODGroup`` level is not level 0.  A renderer no group lists is drawn at
    every distance and stays.
    """
    keep, skipped = [], Counter()
    for node in sorted(prefab.nodes.values(), key=lambda n: n.tid):
        if node.renderer is None:
            continue
        if node.renderer.cast_shadows == P.SHADOWS_ONLY:
            skipped["shadows only"] += 1
        elif (P.lod_level(prefab, node) or 0) > 0:
            skipped["lower LOD"] += 1
        else:
            keep.append(node)
    return keep, skipped


def node_names(prefab: P.Prefab) -> dict[int, str]:
    """A unique name per node.

    A GameObject's own name when no other node of the prefab shares it,
    otherwise its path from the root — the skeleton and the bake's ``nodes``
    table are keyed by name, and Unity allows cousins to repeat.  It allows
    siblings to repeat too: of several nodes at one path the first child keeps
    the path (it is the one a path resolves to, :func:`prefab.find_path`) and
    the others get ``#2``, ``#3`` … in child order.
    """
    count = Counter(node.name for node in prefab.nodes.values())
    paths: dict[int, str] = {prefab.root: prefab.nodes[prefab.root].name}
    stack = [prefab.root]
    while stack:
        tid = stack.pop()
        seen: Counter = Counter()
        for child in prefab.nodes[tid].children:
            name = prefab.nodes[child].name
            seen[name] += 1
            suffix = "" if seen[name] == 1 else f"#{seen[name]}"
            paths[child] = f"{paths[tid]}/{name}{suffix}"
            stack.append(child)
    return {tid: node.name if count[node.name] == 1 else paths[tid]
            for tid, node in prefab.nodes.items()}


def nodes_table(prefab: P.Prefab, names: dict[int, str]) -> dict:
    """``name → (parent name, local 4×4)`` for ``bake.write_item(nodes=…)``.

    The root is placed at the origin (see :func:`prefab.world_matrices`).
    """
    table = {}
    for tid, node in prefab.nodes.items():
        if tid == prefab.root:
            table[names[tid]] = ("", np.eye(4))
        else:
            table[names[tid]] = (names[node.parent], to_project(node.local))
    return table


class Assets:
    """The staged meshes, materials and textures, read once each.

    *shader_name* answers ``Ref → name`` (the asset map's rows).
    """

    def __init__(self, staged: Staged, cabs: CabIndex,
                 shader_name: Callable[[Ref], str | None]):
        self.staged = staged
        self.cabs = cabs
        self._shader_name = shader_name
        self._materials: dict[Ref, UnityMaterial] = {}
        self._infos: dict[Ref, TextureInfo] = {}
        self._texture_owner: dict[tuple, Ref] = {}
        self.written: set[tuple] = set()

    def mesh(self, ref: Ref) -> UnityMesh:
        """The staged mesh *ref*, converted (not cached: meshes are large)."""
        return read_mesh(self.staged.json(ref))

    def material(self, ref: Ref) -> UnityMaterial:
        """The staged material *ref*."""
        if ref not in self._materials:
            self._materials[ref] = read_material(
                self.staged.json(ref), self.cabs, ref.cab)
        return self._materials[ref]

    def shader(self, ref: Ref | None) -> str:
        """The shader's name; :class:`ConvertError` when the asset map has none."""
        name = None if ref is None else self._shader_name(ref)
        if not name:
            raise ConvertError(f"shader {ref} is not in the asset map")
        return name

    def texture_info(self, ref: Ref) -> TextureInfo:
        """Name, colour space and wrap of texture *ref*, from its dump.

        Raises :class:`ConvertError` when one serialized file holds two
        textures of the same name — the named store keys on that pair.
        """
        if ref not in self._infos:
            info = read_texture_info(self.staged.text(ref))
            owner = self._texture_owner.setdefault((ref.cab, info.name), ref)
            if owner != ref:
                raise ConvertError(f"{ref.cab} holds two textures called "
                                   f"{info.name!r}; the named store cannot "
                                   f"tell them apart")
            self._infos[ref] = info
        return self._infos[ref]

    def store_texture(self, ref: Ref, force: bool = False) -> TextureInfo:
        """Put *ref*'s pixels in the named store (once); returns its info."""
        info = self.texture_info(ref)
        key = (store.write_target(), ref)      # one copy per model store being written
        if key in self.written:
            return info
        if force or not store.has_texture(ref.cab, info.name):
            px = _read_png(self.staged.path(ref, ".png"))
            wrap = info.wrap_u if info.wrap_u == info.wrap_v else None
            store.write_texture(ref.cab, info.name, CS.Map(px, info.srgb, wrap))
        self.written.add(key)
        return info


def _read_png(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8)


def _maps(material: UnityMaterial, assets: Assets, force: bool) -> dict:
    out = {}
    for tex in material.textures:
        info = assets.store_texture(tex.ref, force)
        out[tex.prop] = {
            "cab": tex.ref.cab, "name": info.name, "srgb": info.srgb,
            "wrap": [info.wrap_u, info.wrap_v],
            "scale": list(tex.scale), "offset": list(tex.offset)}
    return out


def _shader_dict(material: UnityMaterial, assets: Assets, node: P.Node,
                 prefab: P.Prefab, submeshes: int, force: bool) -> dict:
    off = P.inactive_under(prefab, node)
    out = {
        "_shader": assets.shader(material.shader),
        "_material": material.name,
        "_rendName": node.name,
        "_submeshes": submeshes,
        "_rigid": 0 if node.renderer.skinned else 1,
        "_queue": material.queue,
        "keywords": list(material.keywords),
        "tags": dict(material.tags),
        "disabled_passes": list(material.disabled_passes),
        "floats": dict(material.floats),
        "colors": {k: list(v) for k, v in material.colors.items()},
        "maps": _maps(material, assets, force),
    }
    if off:
        out["_offNode"] = off        # under a node the prefab ships inactive
    if not node.renderer.enabled:
        out["_rendOff"] = 1          # ``Renderer.enabled`` false in the prefab
    return out


def slot_mesh(mesh: UnityMesh, slot: int, name: str) -> Mesh:
    """Material slot *slot*'s share of *mesh*.

    Unity draws submesh *i* with material *i*, and a material past the last
    submesh draws the last submesh again.
    """
    indices = mesh.submeshes[min(slot, len(mesh.submeshes) - 1)]
    out = Mesh.from_arrays(mesh.positions, indices, normals=mesh.normals,
                           uvs=mesh.uvs, name=name)
    for stream, values in mesh.streams.items():
        setattr(out, stream, values)
    return out


def _rig(node: P.Node, mesh: UnityMesh, names: dict, worlds: dict) -> tuple:
    """``(bones, skin, binds, bone_world)`` for a renderer."""
    renderer = node.renderer
    if not renderer.skinned:
        skin = np.zeros((len(mesh.positions), 8))
        skin[:, 0] = 1.0
        return ([names[node.tid]], skin, np.eye(4)[None],
                [to_project(worlds[node.tid])])
    if mesh.skin is None or mesh.binds is None:
        raise ConvertError(f"{node.name}: a SkinnedMeshRenderer whose mesh "
                           f"has no skin")
    if len(renderer.bones) != len(mesh.binds):
        raise ConvertError(f"{node.name}: {len(renderer.bones)} bones for "
                           f"{len(mesh.binds)} bind poses")
    missing = [tid for tid in renderer.bones if tid not in names]
    if missing:
        raise ConvertError(f"{node.name}: {len(missing)} bones are not nodes "
                           f"of the prefab")
    return ([names[tid] for tid in renderer.bones], mesh.skin, mesh.binds,
            [to_project(worlds[tid]) for tid in renderer.bones])


def _bind_spread(binds: np.ndarray, bone_world: list) -> float:
    rest = np.stack(bone_world) @ binds
    return float(np.abs(rest - rest.mean(axis=0)).max())


def build_pieces(prefab: P.Prefab, assets: Assets,
                 force: bool = False) -> tuple[list, dict, Report]:
    """Every drawn renderer of *prefab* as pieces, with the node table.

    Writes the textures the pieces name into the store as it goes.
    """
    names = node_names(prefab)
    worlds = P.world_matrices(prefab)
    keep, skipped = drawn(prefab)
    if not keep and not skipped:
        raise NothingToConvert("no MeshRenderer or SkinnedMeshRenderer")
    report = Report(skipped=skipped)
    seen: set[Ref] = set()
    pieces = []
    for node in keep:
        renderer = node.renderer
        if renderer.mesh is None:
            raise ConvertError(f"{node.name}: renderer with no mesh")
        mesh = assets.mesh(renderer.mesh)
        if not mesh.submeshes:
            raise ConvertError(f"{node.name}: mesh {mesh.name} has no submesh")
        bones, skin, binds, bone_world = _rig(node, mesh, names, worlds)
        report.renderers += 1
        report.vertices += len(mesh.positions)
        report.bind_spread = max(report.bind_spread,
                                 _bind_spread(binds, bone_world))
        for slot, mat_ref in enumerate(renderer.materials):
            if mat_ref is None:
                raise ConvertError(f"{node.name}: material slot {slot} is null")
            material = assets.material(mat_ref)
            shader = _shader_dict(material, assets, node, prefab,
                                  len(mesh.submeshes), force)
            seen.update(tex.ref for tex in material.textures)
            piece_mesh = slot_mesh(mesh, slot, node.name)
            report.triangles += piece_mesh.tri_count
            pieces.append(Piece(
                mesh=piece_mesh, bones=bones, skin=skin, binds=binds,
                bone_world=bone_world, shader=shader, mat_slot=slot,
                shapes=mesh.shapes if slot == 0 and mesh.shapes else None))
    if not pieces:
        raise ConvertError(f"it has renderers but none is drawn at full "
                           f"detail ({dict(skipped)})")
    report.pieces = len(pieces)
    report.textures = len(seen)
    return pieces, nodes_table(prefab, names), report
