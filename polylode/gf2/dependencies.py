"""Restore individually verified outfit dependencies from native counterparts.

Recipes identify exact serialized objects and payloads. Original scene poses
remain intact; missing nodes retain the counterpart's authored local transform.
This records dependency reconstruction, not runtime animation/shader parity.
"""
from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
from UnityPy.files.ObjectReader import ObjectReader

from ..endfield import prefab as P
from ..endfield.unitymesh import UnityMesh
from . import palette
from .catalog import Catalog

NativeRef = tuple[str, int]


def subtree_paths(prefab: P.Prefab, owner: NativeRef) -> dict[str, list[NativeRef]]:
    """Index paths within one actual instance, keeping ambiguous paths explicit."""
    result = {'': [owner]}
    stack = [('', owner)]
    while stack:
        path, key = stack.pop()
        for child in prefab.nodes[key].children:
            relative = '/'.join(v for v in (path, prefab.nodes[child].name) if v)
            result.setdefault(relative, []).append(child)
            stack.append((relative, child))
    return result


def resolved_materials(declared: list, native: list, submesh_count: int) -> list:
    """Keep populated slots; complete only an exact native prefix or all-null list."""
    if any(m is None for m in declared) and any(m is not None for m in declared):
        raise ValueError('source dependency has a partially null material declaration')
    empty = not declared or all(m is None for m in declared)
    short = len(declared) < submesh_count
    if not empty and not short:
        return declared.copy()
    if not native or any(m is None for m in native) or len(native) < submesh_count:
        raise ValueError('source dependency has incomplete native materials')
    if not empty and declared != native[:len(declared)]:
        raise ValueError('source dependency native materials disagree with declared prefix')
    return native.copy()


def restore(prefab: P.Prefab, node: P.Node, source: P.Prefab,
            counterpart: P.Node, mesh: UnityMesh, owner: NativeRef, hashes: list[int]) -> dict:
    """Restore exact native palette paths and missing local nodes atomically.

    Existing node transforms, mesh streams and nonnull materials are preserved.
    Source instance keys distinguish repeated scene rigs without inventing bones.
    """
    rd, native = node.renderer, counterpart.renderer
    if native is None or not native.skinned or len(native.bones) != len(hashes) or len(hashes) != len(mesh.binds):
        raise ValueError('source dependency palette/bind/hash counts disagree')
    paths = [P.node_path(source, key) for key in native.bones]
    if any(zlib.crc32(path.encode('utf8')) != h for path, h in zip(paths, hashes)):
        raise ValueError('source dependency bone hashes disagree with native full paths')
    error, budget = palette.verify_binds(source, native.bones, mesh.binds)
    source_paths = subtree_paths(source, source.root)
    target_paths = subtree_paths(prefab, owner)
    needed = set(paths)
    for path in paths:
        while path:
            path = path.rpartition('/')[0]
            needed.add(path)
    if any(len(source_paths.get(path, [])) != 1 for path in needed):
        raise ValueError('source dependency has ambiguous or absent native bone paths')
    if any(len(target_paths[path]) != 1 for path in needed if path in target_paths):
        raise ValueError('source dependency has ambiguous target instance paths')
    materials = resolved_materials(rd.materials, native.materials, len(mesh.submeshes))
    mapping = {path: keys[0] for path, keys in target_paths.items() if path in needed}
    additions, differences = [], []
    for path in sorted(needed, key=lambda p: (p.count('/'), p)):
        original = source.nodes[source_paths[path][0]]
        if path in mapping:
            if path:
                differences.append(float(np.max(np.abs(prefab.nodes[mapping[path]].local - original.local))))
            continue
        parent = mapping[path.rpartition('/')[0]]
        key = ('source-instance:' + repr(owner) + ':' + repr(original.tid), original.tid[1])
        if key in prefab.nodes:
            raise ValueError('source dependency instance identity collision')
        if not np.isfinite(original.local).all():
            raise ValueError('source dependency has nonfinite native local transform')
        additions.append(P.Node(key, original.name, parent, original.local.copy(), original.active, frozenset()))
        mapping[path] = key
    report = {'name': node.name, 'node': node.tid, 'owner': owner,
              'original_bones': rd.bones.copy(), 'resolved_bones': [mapping[p] for p in paths],
              'original_materials': rd.materials.copy(), 'native_bones': native.bones,
              'native_palette_paths': paths, 'max_native_bind_error': error,
              'source_float32_budget': budget, 'max_shared_local_difference': max(differences, default=0.),
              'added_nodes': [{'instance': n.tid, 'source': source_paths[path][0], 'path': path,
                               'local': n.local.tolist()} for path, key in mapping.items()
                              for n in additions if n.tid == key],
              'existing_transforms_unchanged': True, 'mesh_weights_binds_unchanged': True,
              'pose_status': 'existing scene pose retained; missing nodes use native outfit local rest'}
    for added in additions:
        prefab.nodes[added.tid] = added
        prefab.nodes[added.parent].children.append(added.tid)
    rd.bones = report['resolved_bones']
    rd.materials = materials
    report['resolved_materials'] = rd.materials.copy()
    return report


def digest(reader: ObjectReader) -> str:
    """Hash the original serialized bytes, including mesh channels and binds."""
    return hashlib.sha256(reader.get_raw_data()).hexdigest()


def validate_recipe(recipe: dict, prefab: P.Prefab, readers: dict, node: P.Node) -> None:
    """Reject stale recipes or a mesh binding outside its own authored owner."""
    from . import convert as C

    owner = tuple(recipe['owner'])
    paths = subtree_paths(prefab, owner)
    relative = [p for p, keys in paths.items() if node.tid in keys]
    if len(relative) != 1 or digest(readers[node.renderer.mesh]) != recipe['mesh_sha256']:
        raise ValueError('source dependency target mesh/owner proof changed')
    role = readers[owner].assets_file.environment.find_file(recipe['role_mesh_res'][0]).objects[
        recipe['role_mesh_res'][1]]
    go = readers[owner].read().m_GameObject.read()
    if C.identity(role) not in {C.identity(pair.component.deref()) for pair in go.m_Component}:
        raise ValueError('source dependency RoleMeshRes belongs to another instance')
    native = role.read()
    if native.m_Script.read().m_ClassName != 'RoleMeshRes' or digest(role) != recipe['role_sha256']:
        raise ValueError('source dependency RoleMeshRes proof changed')
    assets = [v['MeshResPath'] for v in role.read_typetree()['MeshResList']
              if v['TransfromPath'] == relative[0]]
    if assets != [recipe['asset_path']]:
        raise ValueError('source dependency authored outfit asset path changed')


def resolve(catalog: Catalog, entry: dict, prefab: P.Prefab, readers: dict,
            keep: list[P.Node], meshes: dict[NativeRef, UnityMesh]) -> list[dict]:
    """Apply only the measured native dependencies for this exact source entry."""
    from . import convert as C
    from . import playable

    recipes = json.loads(Path(__file__).with_name('source_dependencies.json').read_text(encoding='utf8'))
    recipes = [r for r in recipes if r['target']['cab'] == entry['cab'] and r['target']['id'] == entry['id']]
    if not recipes:
        return []
    env = next(iter(readers.values())).assets_file.environment
    roster = playable.roster(catalog.install)
    reports, sources = [], {}
    kept = {n.tid for n in keep}
    for recipe in recipes:
        key = tuple(recipe['node'])
        if key not in kept:
            continue
        node = prefab.nodes[key]
        validate_recipe(recipe, prefab, readers, node)
        original = recipe['source']
        if playable.families(original['name'], roster) != playable.families(entry['name'], roster):
            raise ValueError('source dependency belongs to another character')
        source_key = original['cab'], original['id']
        if source_key not in sources:
            sources[source_key] = C.hierarchy(env.find_file(original['cab']).objects[original['id']],
                                            playable.families(original['name'], roster))[:2]
        source, native_readers = sources[source_key]
        counterpart = source.nodes[tuple(recipe['source_node'])]
        mesh_reader = readers[node.renderer.mesh]
        if digest(native_readers[counterpart.renderer.mesh]) != recipe['mesh_sha256']:
            raise ValueError('source dependency counterpart mesh fingerprint changed')
        if node.renderer.mesh not in meshes:
            meshes[node.renderer.mesh] = C.read_mesh(mesh_reader)
        mesh = meshes[node.renderer.mesh]
        report = restore(prefab, node, source, counterpart, mesh, tuple(recipe['owner']),
                         mesh_reader.read_typetree()['m_BoneNameHashes'])
        report.update(source_entry=original, mesh_sha256=recipe['mesh_sha256'],
                      asset_path=recipe['asset_path'], role_mesh_res=recipe['role_mesh_res'],
                      role_sha256=recipe['role_sha256'], source_node=recipe['source_node'])
        for added in report['added_nodes']:
            added['transform_sha256'] = digest(native_readers[tuple(added['source'])])
        readers.update({mat: native_readers[mat] for mat in node.renderer.materials if mat not in readers})
        reports.append(report)
    return reports
