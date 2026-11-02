"""Conservative Blender eligibility from the local playable tables and source rig."""
from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path

import numpy as np
from . import gender


def _varint(data, position):
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ValueError('truncated protobuf integer')
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, position
    raise ValueError('overlong protobuf integer')


def fields(data):
    """Decode the wire types in GF2's local table envelopes; reject truncation."""
    position = 0
    result = []
    while position < len(data):
        tag, position = _varint(data, position)
        number, wire = tag >> 3, tag & 7
        if not number:
            raise ValueError('protobuf field zero')
        if wire == 0:
            value, position = _varint(data, position)
        else:
            if wire == 2:
                length, position = _varint(data, position)
            elif wire in (1, 5):
                length = 8 if wire == 1 else 4
            else:
                raise ValueError(f'unsupported protobuf wire type {wire}')
            end = position + length
            if end > len(data):
                raise ValueError('truncated protobuf field')
            value, position = data[position:end], end
        result.append((number, wire, value))
    return result


def _rows(data):
    if len(data) < 4:
        raise ValueError('missing local table envelope')
    offset = 4 + struct.unpack_from('<I', data)[0]
    if offset > len(data):
        raise ValueError('truncated local table index')
    fields(data[4:offset])
    result = []
    for number, wire, payload in fields(data[offset:]):
        if number != 1 or wire != 2:
            raise ValueError('unexpected local table row envelope')
        result.append({number: value for number, _, value in fields(payload)})
    return result


def roster(install: Path):
    """Join character wire 4 to gun wire 5, with matching gun/character IDs.

    These are observed local table columns, not inferred humanoid flags. A
    character-only NPC row (Mayling in this installation) is not sufficient.
    """
    folder = Path(install) / 'GF2_Exilium_Data/LocalCache/Data/Table'
    paths = [folder / 'GunCharacterData.bytes', folder / 'GunData.bytes']
    data = [p.read_bytes() for p in paths]
    guns = {r[1]: r for r in _rows(data[1])}
    characters = []
    for row in _rows(data[0]):
        gun = guns.get(row.get(9))
        if gun is None or row.get(4) != gun.get(5):
            continue
        name, model = row[3].decode('utf8'), row[4].decode('utf8')
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9]*', name + model):
            raise ValueError('unexpected playable identifier in local table')
        characters.append(gender.classify({'id': row[1], 'gun_id': gun[1], 'name': name, 'model': model}))
    if not characters:
        raise ValueError('no confirmed playable table joins; Blender export refused')
    return {'scope': 'playable_only', 'gender_scope': gender.policy(), 'characters': characters,
            'sources': [{'file': str(p), 'sha256': hashlib.sha256(d).hexdigest()}
                        for p, d in zip(paths, data)]}


#: the nodes a playable character's weighted skeleton must reach (its body, hands, feet and eyes)
HUMAN_RIG = frozenset({'Root_M', 'Head_M', 'Wrist_L', 'Wrist_R', 'Ankle_L', 'Ankle_R', 'Face_Eye_L', 'Face_Eye_R'})


def has_human_rig(gameobject: object) -> bool:
    """Whether the root's transform tree has every :data:`HUMAN_RIG` node: the skeleton :func:`eligibility`
    requires, asked of the hierarchy alone before any mesh is read (a pet, a weapon or a dorm prop carrying the
    character's name has none of it, §7.1455).  Weights are still :func:`eligibility`'s to check."""
    seen, stack = set(), []
    transform = next((item.component.read() for item in gameobject.m_Component
                      if item.component.type.name == 'Transform'), None)
    if transform is not None:
        stack.append(transform)
    while stack:
        node = stack.pop()
        seen.add(node.m_GameObject.read().m_Name)
        if HUMAN_RIG <= seen:
            return True
        stack.extend(pointer.read() for pointer in node.m_Children)
    return False


def eligibility(extra, declared, scene=None):
    """Require a verified female playable family and a complete human skeleton.

    Require real weighted character geometry, not just an unused human rig.
    The scene may also contain enemy meshes; only owned character renderers
    and their required skeleton belong in the Blender file.
    """
    source = extra['entry']['name']
    matches = [gender.classify(c) for c in families(source, declared)]
    required = HUMAN_RIG
    selected, used, components = [], set(), set()
    candidates, coverage = [], {}
    def human_root(path):
        parts = path.split('/')
        return '/'.join(parts[:parts.index('Root_M') + 1]) if 'Root_M' in parts else None
    if scene is not None and matches:
        for part in scene.parts:
            if not owned_part(part.name, matches):
                continue
            vertices = np.unique(part.triangles)
            if not len(vertices):
                continue
            weighted = part.joints[vertices][part.weights[vertices] > 0]
            if not len(weighted):
                continue
            joints = {int(i) for i in weighted}
            roots = set()
            for i in joints:
                path = extra.get('paths', {}).get(scene.bones[i].node, '')
                root = human_root(path)
                if root is not None:
                    roots.add(root)
                    coverage.setdefault(root, set()).update(path.split('/'))
            candidates.append((part, joints, roots))
        human_roots = {root for root, nodes in coverage.items() if required <= nodes}
        for part, joints, roots in candidates:
            # Summoned animals can reuse a human limb skeleton. Require the
            # actual weighted facial rig on the SAME source Root_M subtree.
            # An owned weapon without Root_M may accompany a confirmed body.
            if (roots and roots <= human_roots) or (not roots and human_roots and part.name.startswith('cw_')):
                selected.append(part.name)
                used.update(joints)
        for i in used:
            components.update(extra.get('paths', {}).get(scene.bones[i].node, '').split('/'))
    missing = sorted(required - components)
    eligible = bool(matches) and not missing
    kept = set(used) if eligible else set()
    if eligible:
        # Preserve the character's whole native Root_M subtree, including
        # attach/IK nodes without vertex weights, plus owned weapon ancestors.
        roots = {human_root(extra['paths'][scene.bones[i].node])
                 for i in used if human_root(extra['paths'][scene.bones[i].node]) is not None}
        for i, bone in enumerate(scene.bones):
            path = extra.get('paths', {}).get(bone.node, '')
            if any(path == root or path.startswith(root + '/') for root in roots):
                kept.add(i)
        for i in list(kept):
            parent = scene.bones[i].parent
            while parent >= 0:
                kept.add(parent)
                parent = scene.bones[parent].parent
    return {'eligible': eligible, 'scope': 'playable_only', 'gender_scope': gender.policy(),
            'characters': matches, 'missing_source_rig_nodes': missing,
            'parts': selected if eligible else [],
            'bones': [scene.bones[i].name for i in sorted(kept)] if scene is not None else [],
            'excluded_parts': [p.name for p in scene.parts if not eligible or p.name not in selected] if scene else [],
            'reason': ('confirmed female playable family and weighted human character geometry' if eligible
                       else 'not a confirmed female playable character' if not matches
                       else 'source lacks weighted geometry for a complete playable character')}


def families(source, declared):
    """Match only reviewed female identities before decoding source assets."""
    return [c for c in declared['characters'] if gender.verified(c) and re.match(
        r'^c_' + re.escape(c['name']) + r'(?:(?:SSR|SR|Dorm)\d*)?(?:_|$)', source, re.IGNORECASE)]


def owned_part(name, matches):
    return any(re.match(r'^(?:c_|cw_)' + re.escape(c['name'])
               + r'(?:(?:SSR|SR|Dorm)\d*)?(?:_|$)', name, re.IGNORECASE) for c in matches)


def has_owned_skin(gameobject: object, matches: list[dict]) -> bool:
    """Check a necessary source condition before reading unrelated scene assets.

    Ownership and component declarations suffice to reject templates without any
    character skin. A positive result still requires the weighted-human checks;
    this gate neither reads renderer payloads nor certifies a complete character.
    """
    if not matches:
        return False
    components = gameobject.m_Component
    if owned_part(gameobject.m_Name, matches) and any(
            item.component.type.name == 'SkinnedMeshRenderer' for item in components):
        return True
    transform = next(item.component.read() for item in components
                     if item.component.type.name == 'Transform')
    return any(has_owned_skin(pointer.read().m_GameObject.read(), matches)
               for pointer in transform.m_Children)


def has_owned_mesh(gameobject: object, matches: list[dict]) -> bool:
    """Whether one of the character's own skinned renderers under the root points at a mesh: a ``_model``
    variant has the renderers with an empty mesh reference, and its conversion stops on "no full-detail
    skinned renderer with a source mesh" (``convert.geometry_renderers``, §7.1455).  The renderer is read,
    the mesh is not."""
    if not matches:
        return False
    stack = [gameobject]
    while stack:
        go = stack.pop()
        components = go.m_Component
        if owned_part(go.m_Name, matches):
            for item in components:
                if item.component.type.name == 'SkinnedMeshRenderer':
                    mesh = item.component.read().m_Mesh
                    if getattr(mesh, 'm_PathID', getattr(mesh, 'path_id', 0)):
                        return True
        transform = next(item.component.read() for item in components if item.component.type.name == 'Transform')
        stack.extend(pointer.read().m_GameObject.read() for pointer in transform.m_Children)
    return False


def stored_eligibility(ident, declared):
    """Read only the ownership/skin arrays needed for the export queue gate."""
    from types import SimpleNamespace

    from . import store
    extra = store.read_extra(ident)
    matches = families(extra['entry']['name'], declared)
    if not matches:
        return eligibility(extra, declared)
    document_path = store.root() / 'exports' / ident / f'{ident}.json'
    if not document_path.is_file():
        from . import publish
        document_path = publish.root() / ident / f'{ident}.json'
    if not document_path.is_file():
        from ..endfield.rigscene import scene_of
        item = store.read_item(ident)
        return eligibility(extra, declared, scene_of(item.pieces, item.nodes, ident))
    import json
    document = json.loads(document_path.read_text(encoding='utf8'))
    bones = [SimpleNamespace(**b) for b in document['bones']]
    parts = []
    with np.load(document_path.parent / document['arrays']) as arrays:
        for i, record in enumerate(document['parts']):
            part = SimpleNamespace(name=record['name'])
            if any(re.match(r'^(?:c_|cw_)' + re.escape(c['name'])
                      + r'(?:(?:SSR|SR|Dorm)\d*)?(?:_|$)', part.name, re.IGNORECASE) for c in matches):
                part.triangles, part.joints, part.weights = (arrays[f'p{i}_{key}'] for key in ('tri', 'jnt', 'wgt'))
            parts.append(part)
        return eligibility(extra, declared, SimpleNamespace(bones=bones, parts=parts))


def character_scene(scene, selection):
    """Keep exactly verified female character geometry for FBX and Blender."""
    from dataclasses import replace
    if not gender.selection_verified(selection):
        raise ValueError('not a confirmed female main playable human character')
    selected = set(selection['parts'])
    retained = set(selection['bones'])
    indices = [i for i, b in enumerate(scene.bones) if b.name in retained]
    mapping = {old: new for new, old in enumerate(indices)}
    bones = []
    for i in indices:
        bone = scene.bones[i]
        if bone.parent >= 0 and bone.parent not in mapping:
            raise ValueError('character scope lost a source bone parent')
        bones.append(replace(bone, parent=mapping.get(bone.parent, -1)))
    parts = []
    for part in scene.parts:
        if part.name not in selected:
            continue
        joints = np.zeros_like(part.joints)
        for old in np.unique(part.joints[part.weights > 0]):
            if int(old) not in mapping:
                raise ValueError('character scope lost a weighted source joint')
            active = (part.joints == old) & (part.weights > 0)
            joints[active] = mapping[int(old)]
        parts.append(replace(part, joints=joints))
    used_materials = {m for p in parts for m in p.materials if m}
    materials = {n: r for n, r in scene.materials.items() if n in used_materials}
    used_textures = {m['file'].removeprefix('textures/') for r in materials.values() for m in r['maps'].values()}
    textures = {n: t for n, t in scene.textures.items() if n in used_textures}
    return replace(scene, bones=bones, parts=parts, materials=materials, textures=textures)
