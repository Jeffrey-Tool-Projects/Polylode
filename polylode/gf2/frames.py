"""Exact translation frames for float32 hosts, with source joint identities.

Each mesh has one deformation rig. The separate animation controller keeps
the complete source hierarchy. Deformation joints use evaluated world poses
in a nearby translation frame, avoiding large parent-child cancellation.
Integer power-of-two origins are exactly representable in float32.
The source scene and archives stay in their original world coordinates.
"""
from __future__ import annotations

import math

import numpy as np


def plan(bones, parts):
    # Reserve three quarters of the 10 micrometre rest-position budget for
    # edit-bone storage and arithmetic. Inside a cell, float32 rounding in
    # three coordinates is at most sqrt(3) * eps * half_cell / 2.
    half_cell = 2 ** math.floor(math.log2((.01 / 1000 / 4) /
                                         (math.sqrt(3) * np.finfo(np.float32).eps / 2)))
    step = 2 * half_cell
    positions = np.stack([b.rest[:3, 3] for b in bones])
    cells, lookup = [], {}
    # Spatially overlapping rigs can have unrelated FK motions. Join ONLY
    # joints that actually influence the same renderer, including its LODs;
    # no source names or assumed character/weapon roles enter this partition.
    leaders = list(range(len(bones)))

    def component(source):
        while leaders[source] != source:
            leaders[source] = leaders[leaders[source]]
            source = leaders[source]
        return source

    influences, weighted = [], set()
    for part in parts:
        used = [int(i) for i in np.unique(part.joints[part.weights > 0])]
        influences.append(used)
        weighted.update(used)
        for source in used[1:]:
            leaders[component(source)] = component(used[0])
    descendant = {i: i for i in weighted}
    for source in reversed(range(len(bones))):
        parent = bones[source].parent
        if source in descendant and parent >= 0 and parent not in weighted:
            descendant[parent] = min(descendant.get(parent, descendant[source]), descendant[source])
    components = []
    for source in range(len(bones)):
        nearest = descendant.get(source)
        above = source
        while nearest is None and above >= 0:
            if above in weighted:
                nearest = above
                break
            above = bones[above].parent
        components.append(component(nearest) if nearest is not None else -1)

    def frame(position, group):
        origin = (np.rint(np.asarray(position) / step) * step).tolist()
        key = (*origin, group)
        if key not in lookup:
            lookup[key] = len(cells)
            cells.append({'origin': origin, 'component': group, 'sources': set()})
        return lookup[key]

    homes = [frame(p, group) for p, group in zip(positions, components)]
    mesh_frames = [frame((p.positions.min(0) + p.positions.max(0)) / 2,
                         component(used[0]) if used else -1) for p, used in zip(parts, influences)]

    def keep(fid, source):
        cells[fid]['sources'].add(source)

    for source, fid in enumerate(homes):
        keep(fid, source)
    for part, fid in zip(parts, mesh_frames):
        for source in np.unique(part.joints[part.weights > 0]):
            keep(fid, int(source))
    instances, names = [], {b.name for b in bones}
    for fid, cell in enumerate(cells):
        for source in sorted(cell.pop('sources')):
            bone = bones[source]
            canonical = homes[source] == fid
            prefix = f'f{fid}_'
            base = prefix + bone.name.encode('utf8')[:63-len(prefix)].decode('utf8', errors='ignore')
            name, n = base, 1
            while name in names:
                n += 1
                suffix = f'.{n}'
                name = base.encode('utf8')[:63-len(suffix)].decode('utf8', errors='ignore') + suffix
            names.add(name)
            instances.append({'name': name, 'source': source, 'frame': fid,
                              'canonical': canonical,
                              'parent': -1})
        cell['name'] = f'GF2_Frame_{fid}'
        held = [r['source'] for r in instances if r['frame'] == fid]
        cell['anchor'] = min(held, key=lambda i: np.linalg.norm(positions[i]-cell['origin']))
    return {'method': 'source_hierarchy_controls_with_local_deformation_frames',
            'step_m': step, 'frames': cells, 'instances': instances,
            'mesh_frames': mesh_frames}


def expanded_bones(scene, layout):
    return ([(b.name, b.parent, b.rest) for b in scene.bones] +
            [(r['name'], -1, scene.bones[r['source']].rest) for r in layout['instances']])
