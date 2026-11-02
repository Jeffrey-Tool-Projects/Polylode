"""Complete source palettes only from exact own-Avatar paths and verified binds."""
from __future__ import annotations

import numpy as np

from ..endfield import prefab as P
from ..endfield.unitymesh import to_project


def complete(prefab: P.Prefab, node: P.Node, mesh: object, hashes: list,
             avatars: list[dict]) -> dict:
    """Append existing source nodes, preserving every declared renderer slot.

    No bone, bind or weight is created. Ambiguous paths, a different original
    prefix or a bind disagreement fail before the in-memory renderer changes.
    """
    original = node.renderer.bones.copy()
    if not original or not len(original) < len(mesh.binds) == len(hashes):
        raise ValueError(f'{node.name}: palette completion requires a short nonempty source prefix')
    paths = {key: P.node_path(prefab, key) for key in prefab.nodes}
    ancestors = [a for a in avatars if not a['owner_path'] or paths[node.tid].startswith(a['owner_path'] + '/')]
    if not ancestors:
        raise ValueError(f'{node.name}: palette has no own ancestor Avatar')
    closest = max(len(a['owner_path']) for a in ancestors)
    own = [a for a in ancestors if len(a['owner_path']) == closest]
    if len(own) != 1:
        raise ValueError(f'{node.name}: palette has ambiguous own Avatars')
    avatar = own[0]
    resolved = []
    for bone_hash in hashes:
        relative = avatar['paths'].get(bone_hash)
        if relative is None:
            raise ValueError(f'{node.name}: palette hash {bone_hash} absent from own Avatar')
        full = '/'.join(p for p in (avatar['owner_path'], relative) if p)
        candidates = [key for key, path in paths.items() if path == full]
        if len(candidates) != 1:
            raise ValueError(f'{node.name}: palette path {full} is missing or ambiguous')
        resolved.append(candidates[0])
    if resolved[:len(original)] != original:
        raise ValueError(f'{node.name}: palette Avatar disagrees with the declared source prefix')
    error, budget = verify_binds(prefab, resolved, mesh.binds)
    report = {'name': node.name, 'node': node.tid, 'avatar': avatar['identity'],
              'original_bones': original, 'resolved_bones': resolved, 'mesh_hashes': hashes,
              'max_bind_error': error, 'source_float32_budget': budget,
              'mesh_weights_binds_unchanged': True}
    node.renderer.bones = resolved
    return report


def verify_binds(prefab: P.Prefab, bones: list, binds: np.ndarray) -> tuple[float, float]:
    """Verify the native rest skeleton against every serialized bind matrix."""
    if not len(bones) == len(binds) or not bones or any(b not in prefab.nodes for b in bones):
        raise ValueError('palette has missing nodes or inconsistent bind count')
    worlds = P.world_matrices(prefab)
    products = [to_project(worlds[key]) @ bind for key, bind in zip(bones, binds)]
    if not all(np.isfinite(product).all() for product in products):
        raise ValueError('palette has nonfinite bind rest matrices')
    # A 4x4 product has four terms per entry. Use the float32 gamma_4
    # roundoff budget of the serialized matrices; do not relax export precision.
    epsilon = float(np.finfo(np.float32).eps)
    gamma = 4 * epsilon / (1 - 4 * epsilon)
    scale = max(1., *(float(np.max(np.abs(to_project(worlds[key])) @ np.abs(bind)))
                       for key, bind in zip(bones, binds)))
    error = max(float(np.max(np.abs(product - products[0]))) for product in products)
    if not np.isfinite(error) or error > gamma * scale:
        raise ValueError(f'palette bind rest disagreement {error} exceeds source float32 budget')
    return error, gamma * scale
