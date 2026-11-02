"""Assemble serialized GF2 prefab references into project-compatible pieces.

Geometry and material selection come from the renderers, never texture names.
The shared Endfield geometry/rig helpers define the same right-handed Y-up
coordinate conversion and bake schema used by IslandLife.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from UnityPy.helpers.MeshHelper import MeshHandler

from ..endfield import components as K
from ..endfield import prefab as P
from ..endfield.convert import drawn, node_names, nodes_table, slot_mesh
from ..endfield.unitymesh import UnityMesh, to_project
from ..origin import bake
from ..origin import colorspace as CS
from ..origin.prefab import Piece
from . import playable, store


class NoGeometryError(ValueError):
    """The authored root has no full-detail skinned mesh geometry to export."""


def geometry_renderers(prefab: P.Prefab, names: dict) -> tuple[list, dict, list]:
    """Preserve null mesh declarations, without trying to decode absent geometry."""
    candidates, skipped = drawn(prefab)
    keep, empty = [], []
    for node in candidates:
        rd = node.renderer
        if rd.mesh is not None:
            keep.append(node)
            continue
        skipped['null source mesh'] += 1
        empty.append({'name': names[node.tid], 'mesh': None, 'active': node.active,
                      'enabled': rd.enabled, 'skinned': rd.skinned,
                      'materials': [list(m) if m is not None else None for m in rd.materials],
                      'bones': [list(b) if b is not None else None for b in rd.bones]})
    if not any(node.renderer.skinned for node in keep):
        raise NoGeometryError('no full-detail skinned renderer with a source mesh')
    return keep, skipped, empty


def identity(reader):
    return (reader.assets_file.name.lower(), reader.path_id)


def ref(pointer):
    return identity(pointer.deref()) if pointer else None


def matrix(value):
    return np.array([[value[f'e{r}{c}'] for c in range(4)] for r in range(4)], dtype=np.float64)


def transform_matrix(tree):
    # The shared staged representation spells vector components in uppercase.
    return P.local_matrix({key: {k.upper(): v for k, v in tree[key].items()}
                           for key in ('m_LocalPosition', 'm_LocalRotation', 'm_LocalScale')})


def _declared_ref(pointer):
    """Record a native pointer without resolving a foreign scene dependency."""
    from .catalog import cab_key
    if not pointer:
        return None
    source = pointer.assetsfile
    name = source.name if pointer.m_FileID == 0 else source.externals[pointer.m_FileID - 1].path
    return [cab_key(name), pointer.m_PathID]


def _renderer_data(native, components, include):
    rd = native.read()
    skinned = native.type.name == 'SkinnedMeshRenderer'
    meshptr = rd.m_Mesh if skinned else next(o for o in components if o.type.name == 'MeshFilter').read().m_Mesh
    declaration = {'renderer': list(identity(native)), 'skinned': skinned, 'mesh': _declared_ref(meshptr),
                   'materials': [_declared_ref(p) for p in rd.m_Materials],
                   'bones': [_declared_ref(p) for p in rd.m_Bones] if skinned else [],
                   'enabled': bool(rd.m_Enabled), 'cast_shadows': int(rd.m_CastShadows)}
    if not include:
        return None, {}, declaration
    renderer = P.Renderer(skinned, ref(meshptr), [ref(p) for p in rd.m_Materials],
                          [ref(p) for p in rd.m_Bones] if skinned else [],
                          bool(rd.m_Enabled), int(rd.m_CastShadows))
    readers = {ref(ptr): ptr.deref() for ptr in [meshptr, *rd.m_Materials] if ptr}
    return renderer, readers, declaration


def hierarchy(gameobject, matches: list[dict] | None = None, declarations: list | None = None):
    """Read the full node hierarchy with optional character renderer ownership.

    An ownership projection retains foreign renderer pointer declarations while
    leaving their payloads unresolved. The default still reads the full scene.
    """
    nodes, readers, lods, animators = {}, {}, [], []

    def visit(go_reader, parent):
        go = go_reader.read()
        components = [item.component.deref() for item in go.m_Component]
        transform = next(o for o in components if o.type.name in ('Transform', 'RectTransform'))
        tid = identity(transform)
        if tid in nodes:
            raise ValueError(f'prefab repeats transform {tid}')
        value = transform.read()
        tree = transform.read_typetree()
        renderer = None
        renderers = [o for o in components if o.type.name in ('SkinnedMeshRenderer', 'MeshRenderer')]
        if len(renderers) > 1:
            raise ValueError(f'{go.m_Name}: multiple mesh renderers on one node')
        if renderers:
            included = matches is None or playable.owned_part(go.m_Name, matches)
            renderer, held, declaration = _renderer_data(renderers[0], components, included)
            readers.update(held)
            if declarations is not None:
                declarations.append(dict(declaration, node=list(tid), name=go.m_Name,
                                         active=bool(go.m_IsActive), owned=included))
        children = [ref(ptr) for ptr in value.m_Children]
        nodes[tid] = P.Node(tid, go.m_Name, parent, transform_matrix(tree),
                           bool(go.m_IsActive), frozenset(identity(o) for o in components),
                           renderer, children)
        readers[tid] = transform
        for component in components:
            if component.type.name == 'LODGroup':
                lods.append([{ref(r.renderer) for r in lod.renderers}
                             for lod in component.read().m_LODs])
            if component.type.name == 'Animator':
                animators.append(component)
        for ptr in value.m_Children:
            child = ptr.read()
            if ref(child.m_Father) != tid:
                raise ValueError(f'{go.m_Name}: child/parent references disagree')
            visit(child.m_GameObject.deref(), tid)
        return tid

    root = visit(gameobject, None)
    return P.Prefab(gameobject.assets_file.name, root, nodes, lods), readers, animators


def read_mesh(reader):
    data = reader.read()
    handler = MeshHandler(data)
    handler.process()
    positions = np.asarray(handler.m_Vertices, dtype=np.float32).copy()
    normals = np.asarray(handler.m_Normals, dtype=np.float32)[:, :3].copy()
    if positions.shape != normals.shape or positions.shape[1] != 3:
        raise ValueError(f'{data.m_Name}: missing or inconsistent positions/normals')
    positions[:, 0] *= -1
    normals[:, 0] *= -1
    submeshes = []
    for triangles in handler.get_triangles():
        tri = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        if tri.size and (tri.min() < 0 or tri.max() >= len(positions)):
            raise ValueError(f'{data.m_Name}: indices outside vertex range')
        submeshes.append(tri[:, ::-1].astype(np.uint32).reshape(-1))
    streams = {}
    authored = getattr(handler, 'm_Tangents', None)
    if authored is not None and len(authored):
        tangent = np.asarray(authored, dtype=np.float32).copy()
        if tangent.shape != (len(positions), 4) or not np.isfinite(tangent).all():
            raise ValueError(f'{data.m_Name}: incomplete or nonfinite authored tangents')
        tangent[:, (0, 3)] *= -1
        streams['tangents'] = tangent
    for attr, field in [('uv1', 'm_UV1'), ('uv2', 'm_UV2'), ('uv3', 'm_UV3'), ('vcolor', 'm_Colors')]:
        values = getattr(handler, field)
        if values is not None:
            streams[attr] = np.asarray(values, dtype=np.float32)
    if any(getattr(handler, f'm_UV{i}') is not None for i in range(4, 8)):
        raise ValueError(f'{data.m_Name}: UV4..7 cannot be represented by the current project schema')
    skin = None
    if handler.m_BoneIndices is not None:
        ids = np.asarray(handler.m_BoneIndices, dtype=np.float64)
        weights = handler.m_BoneWeights
        # A single serialized blend index has one normalized influence.
        # Preserve that explicit index; never substitute a named bone.
        if weights is None and ids.shape == (len(positions), 1):
            weights = np.ones_like(ids)
        if weights is None:
            raise ValueError(f'{data.m_Name}: blend indices without representable weights')
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != ids.shape or not 1 <= ids.shape[1] <= 4:
            raise ValueError(f'{data.m_Name}: weight/index channel dimensions disagree')
        skin = np.zeros((len(positions), 8))
        skin[:, :ids.shape[1]] = weights
        skin[:, 4:4 + ids.shape[1]] = ids
        if skin.shape != (len(positions), 8) or not np.isfinite(skin).all():
            raise ValueError(f'{data.m_Name}: invalid skin stream')
        weights = skin[:, :4]
        if weights.min() < 0 or np.max(np.abs(weights.sum(axis=1) - 1)) > 1e-4:
            raise ValueError(f'{data.m_Name}: invalid weight normalization')
    tree = reader.read_typetree()
    variable = tree.get('m_VariableBoneCountWeights') or {}
    if isinstance(variable, dict):
        variable = variable.get('m_Data') or []
    if variable:
        raise ValueError(f'{data.m_Name}: variable influence stream requires an explicit decoder')
    binds = tree.get('m_BindPose') or []
    bind_matrices = to_project(np.stack([matrix(m) for m in binds])) if binds else None
    if skin is not None and bind_matrices is not None:
        positive = skin[:, :4] > 0
        ids = skin[:, 4:]
        if np.any(ids[positive] != np.floor(ids[positive])) or np.any(ids[positive] >= len(binds)) or np.any(ids[positive] < 0):
            raise ValueError(f'{data.m_Name}: skin bone index outside bind palette')
    shapes = []
    sh = tree.get('m_Shapes') or {}
    for ch in sh.get('channels') or []:
        frame = sh['shapes'][ch['frameIndex'] + ch['frameCount'] - 1]
        block = sh['vertices'][frame['firstVertex']:frame['firstVertex'] + frame['vertexCount']]
        index = np.asarray([v['index'] for v in block], dtype=np.int32)
        def vectors(key, vertices=block):
            a = np.asarray([[v[key][c] for c in 'xyz'] for v in vertices], dtype=np.float32).reshape(-1, 3)
            a[:, 0] *= -1
            return a
        shapes.append((ch['name'], index, vectors('vertex'), vectors('normal') if frame['hasNormals'] else None))
    return UnityMesh(data.m_Name, positions, normals,
                     np.asarray(handler.m_UV0, dtype=np.float32) if handler.m_UV0 is not None else None,
                     streams, submeshes, skin, bind_matrices, shapes)


def texture_pixels(texture, key):
    """Keep original payloads and float precision; PNGs are editable previews."""
    from UnityPy.enums import TextureFormat

    folder = store.root() / 'source' / 'textures'
    folder.mkdir(parents=True, exist_ok=True)
    raw = texture.get_image_data()
    payload = folder / f'{key}.bin'
    if not payload.exists():
        payload.write_bytes(raw)
    fmt = TextureFormat(texture.m_TextureFormat).name
    info = {'format': fmt, 'width': texture.m_Width, 'height': texture.m_Height,
            'mip_count': texture.m_MipCount, 'payload': str(payload.relative_to(store.root()))}
    floating = {'RHalf': (1, '<f2'), 'RGHalf': (2, '<f2'), 'RGBAHalf': (4, '<f2'),
                'RFloat': (1, '<f4'), 'RGFloat': (2, '<f4'), 'RGBAFloat': (4, '<f4')}
    if fmt in floating:
        channels, dtype = floating[fmt]
        values = np.frombuffer(raw, dtype=dtype, count=texture.m_Width * texture.m_Height * channels)
        values = values.reshape(texture.m_Height, texture.m_Width, channels)[::-1].astype(np.float32)
        exact = folder / f'{key}.npz'
        np.savez_compressed(exact, pixels=values)
        info['float_pixels'] = str(exact.relative_to(store.root()))
        info['preview_quantized'] = True
        rgba = np.zeros((*values.shape[:2], 4), dtype=np.float32)
        rgba[:, :, 3] = 1
        rgba[:, :, :channels] = values
        return np.rint(np.clip(rgba, 0, 1) * 255).astype(np.uint8), info
    return np.asarray(texture.image.convert('RGBA'), dtype=np.uint8), info


def material(reader, renderer, prefab, submeshes):
    value = reader.read()
    shader_reader = value.m_Shader.deref()
    shader_tree = shader_reader.read_typetree()
    parsed = shader_tree.get('m_ParsedForm') or {}
    shader_name = parsed.get('m_Name') or shader_tree.get('m_Name')
    if not shader_name:
        raise ValueError(f'{value.m_Name}: no shader name')
    props = value.m_SavedProperties
    declarations = (parsed.get('m_PropInfo') or {}).get('m_Props') or []
    declared_main = [p['m_Name'] for p in declarations if int(p.get('m_Flags', 0)) & 128]
    if len(declared_main) > 1:
        raise ValueError(f'{shader_name}: multiple MainTexture declarations')
    maps = {}
    for prop, binding in props.m_TexEnvs:
        if not binding.m_Texture:
            continue
        texture = binding.m_Texture.read()
        tex_reader = binding.m_Texture.deref()
        tex_tree = tex_reader.read_typetree()
        cab = tex_reader.assets_file.name.lower()
        name = f'{texture.m_Name}@{tex_reader.path_id}'
        setting = tex_tree['m_TextureSettings']
        wrap_u, wrap_v = int(setting['m_WrapU']), int(setting['m_WrapV'])
        color_space = int(tex_tree['m_ColorSpace'])
        if color_space not in (0, 1):
            raise ValueError(f'{texture.m_Name}: unsupported texture color space {color_space}')
        srgb = color_space == 1
        tex_key = hashlib.sha256(str(identity(tex_reader)).encode()).hexdigest()
        pixels, texture_source = texture_pixels(texture, tex_key)
        if bake.named_entry(store.SOURCE, cab, name) is None:
            bake.write_named_texture(store.SOURCE, cab, name, CS.Map(pixels, srgb, wrap_u if wrap_u == wrap_v else None))
        maps[prop] = {'cab': cab, 'name': name, 'original_name': texture.m_Name,
                      'path_id': tex_reader.path_id, 'srgb': srgb, 'wrap': [wrap_u, wrap_v],
                      'scale': [binding.m_Scale.x, binding.m_Scale.y],
                      'offset': [binding.m_Offset.x, binding.m_Offset.y]}
        maps[prop]['source'] = texture_source
    tree = reader.read_typetree()
    out = {'_shader': shader_name, '_material': value.m_Name,
           '_rendName': renderer.name, '_submeshes': submeshes,
           '_rigid': int(not renderer.renderer.skinned), '_queue': value.m_CustomRenderQueue,
           'keywords': str(tree.get('m_ShaderKeywords') or '').split(),
           'floats': dict(props.m_Floats),
           'colors': {k: [v.r, v.g, v.b, v.a] for k, v in props.m_Colors},
           'maps': maps, 'tags': dict(tree.get('stringTagMap') or []),
           'disabled_passes': tree.get('disabledShaderPasses') or [],
           'source_material': list(identity(reader))}
    # Unity's serialized MainTexture flag; otherwise Material.mainTexture is
    # the declared _MainTex property. This is an editable preview binding,
    # not a transcription of the game's lighting.
    base = declared_main[0] if declared_main else '_MainTex'
    # This shader explicitly labels its serialized _BaseMap "Albedo Map".
    # Bind it only as an editable preview, without claiming shader parity.
    if not declared_main and shader_name == 'gf_shader/pbr/character/uber':
        declaration = next((p for p in declarations if p['m_Name'] == '_BaseMap'), None)
        if declaration and declaration['m_Description'] == 'Albedo Map (RGBA)':
            base = '_BaseMap'
    out['_base_property'] = base if base in maps else None
    off = P.inactive_under(prefab, renderer)
    if off:
        out['_offNode'] = off
    if not renderer.renderer.enabled:
        out['_rendOff'] = 1
    # Shader declarations and compiled payload remain local evidence.
    key = hashlib.sha256(str(identity(shader_reader)).encode()).hexdigest()
    source = store.root() / 'source' / 'shaders' / f'{key}.bin'
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(shader_reader.get_raw_data())
        bake.write_named_table(store.SOURCE, 'shader', key,
                               {'name': shader_name, 'identity': list(identity(shader_reader)),
                                'declarations': parsed.get('m_PropInfo'),
                                'serialized_file': str(source.relative_to(store.root()))})
    return out


def source_selection(prefab, names, readers, keep, meshes, entry, declared):
    """Confirm weighted human geometry before material/texture conversion."""
    from types import SimpleNamespace

    from ..mesh.skin import split_archive_skin
    tids = list(prefab.nodes)
    index = {tid: i for i, tid in enumerate(tids)}
    bones = [SimpleNamespace(name=names[tid], node=names[tid],
                            parent=index.get(prefab.nodes[tid].parent, -1)) for tid in tids]
    matches = playable.families(entry['name'], declared)
    parts = []
    for node in keep:
        rd = node.renderer
        part = SimpleNamespace(name=names[node.tid])
        if not rd.skinned or not playable.owned_part(part.name, matches):
            continue
        if rd.mesh not in meshes:
            meshes[rd.mesh] = read_mesh(readers[rd.mesh])
        mesh = meshes[rd.mesh]
        if mesh.skin is None or mesh.binds is None or len(rd.bones) != len(mesh.binds):
            raise ValueError(f'{node.name}: bone palette/bind/skin disagree')
        weights, joints = split_archive_skin(mesh.skin, bone_count=len(rd.bones))
        palette = np.array([index[b] for b in rd.bones])
        part.joints, part.weights = palette[joints], weights
        part.triangles = np.concatenate(mesh.submeshes).reshape(-1, 3)
        parts.append(part)
    extra = {'entry': entry, 'paths': {names[tid]: P.node_path(prefab, tid) for tid in tids}}
    selection = playable.eligibility(extra, declared, SimpleNamespace(bones=bones, parts=parts))
    if not selection['eligible']:
        return selection
    # Rigid accessories use their authored node transform, as convert() already
    # does. Keep owned weapons and attachments on the confirmed human subtree;
    # an unrelated rigid effect cannot establish a complete human character.
    for node in keep:
        rd = node.renderer
        name = names[node.tid]
        if rd.skinned or not playable.owned_part(name, matches):
            continue
        if name not in selection['bones'] and not name.startswith('cw_'):
            continue
        if rd.mesh not in meshes:
            meshes[rd.mesh] = read_mesh(readers[rd.mesh])
        mesh = meshes[rd.mesh]
        weights = np.zeros((len(mesh.positions), 4)); weights[:, 0] = 1
        joints = np.zeros_like(weights, dtype=np.int64); joints[:, 0] = index[node.tid]
        parts.append(SimpleNamespace(name=name, joints=joints, weights=weights,
                                     triangles=np.concatenate(mesh.submeshes).reshape(-1, 3)))
    return playable.eligibility(extra, declared, SimpleNamespace(bones=bones, parts=parts))


def complete_palettes(prefab, readers, animators, keep, meshes):
    """Resolve a short source palette only with its own exact Avatar and binds."""
    from . import palette

    pending = []
    for node in keep:
        rd = node.renderer
        if not rd.skinned:
            continue
        if rd.mesh not in meshes:
            meshes[rd.mesh] = read_mesh(readers[rd.mesh])
        mesh = meshes[rd.mesh]
        if mesh.skin is not None and mesh.binds is not None and len(rd.bones) < len(mesh.binds):
            pending.append(node)
    if not pending:
        return []
    avatars = []
    for reader in animators:
        value = reader.read()
        if not value.m_Avatar:
            continue
        go = value.m_GameObject.read()
        transform = next(c.component.deref() for c in go.m_Component if c.component.type.name == 'Transform')
        avatar = value.m_Avatar.deref()
        avatars.append({'identity': list(identity(avatar)),
                        'owner_path': P.node_path(prefab, identity(transform)),
                        'paths': dict(avatar.read_typetree()['m_TOS'])})
    return [dict(palette.complete(prefab, node, meshes[node.renderer.mesh],
                                  readers[node.renderer.mesh].read_typetree().get('m_BoneNameHashes', []), avatars),
                 mesh=list(node.renderer.mesh)) for node in pending]


def convert(catalog, entry):
    from . import dependencies

    declared = playable.roster(catalog.install)
    matches = playable.families(entry['name'], declared)
    if not matches:
        raise NoGeometryError('not a confirmed female playable character')
    env = catalog.environment()
    gameobject = env.find_file(entry['cab']).objects[entry['id']]
    declarations = []
    prefab, readers, animators = hierarchy(gameobject, matches, declarations)
    names = node_names(prefab)
    keep, skipped, empty = geometry_renderers(prefab, names)
    pieces, renderers = [], []
    meshes = {}
    source_dependencies = dependencies.resolve(catalog, entry, prefab, readers, keep, meshes)
    names = node_names(prefab)
    worlds = P.world_matrices(prefab)
    palette_completions = complete_palettes(prefab, readers, animators, keep, meshes)
    selection = source_selection(prefab, names, readers, keep, meshes, entry, declared)
    if not selection['eligible']:
        raise NoGeometryError(selection['reason'])
    selected = set(selection['parts'])
    keep = [node for node in keep if names[node.tid] in selected]
    for declaration in declarations:
        tid = tuple(declaration['node'])
        declaration.update(name=names[tid], path=P.node_path(prefab, tid), archived=names[tid] in selected)
    with store.opened():
        for node in keep:
            rd = node.renderer
            mesh = meshes.setdefault(rd.mesh, None)
            if mesh is None:
                mesh = meshes[rd.mesh] = read_mesh(readers[rd.mesh])
            if rd.skinned:
                if mesh.binds is None or mesh.skin is None or len(rd.bones) != len(mesh.binds):
                    raise ValueError(f'{node.name}: bone palette/bind/skin disagree: '
                                     f'{len(rd.bones)} bones, '
                                     f'{None if mesh.binds is None else len(mesh.binds)} binds, '
                                     f'{None if mesh.skin is None else mesh.skin.shape} skin')
                if any(b not in names for b in rd.bones):
                    raise ValueError(f'{node.name}: bone outside prefab hierarchy')
                bones, skin, binds = [names[b] for b in rd.bones], mesh.skin, mesh.binds
                bone_world = [to_project(worlds[b]) for b in rd.bones]
            else:
                bones, binds, bone_world = [names[node.tid]], np.eye(4)[None], [to_project(worlds[node.tid])]
                skin = np.zeros((len(mesh.positions), 8)); skin[:, 0] = 1
            if len(rd.materials) < len(mesh.submeshes):
                raise ValueError(f'{node.name}: fewer materials than submeshes')
            for slot, mat in enumerate(rd.materials):
                if mat is None:
                    raise ValueError(f'{node.name}: null material at slot {slot}')
                shader = material(readers[mat], node, prefab, len(mesh.submeshes))
                if shader['maps'] and mesh.uvs is None:
                    raise ValueError(f'{node.name}: textured mesh has no UV0')
                pieces.append(Piece(slot_mesh(mesh, slot, names[node.tid]), bones, skin, binds,
                                    bone_world=bone_world, shader=shader, mat_slot=slot,
                                    shapes=mesh.shapes if slot == 0 else None))
            renderers.append({'name': names[node.tid], 'mesh': list(rd.mesh),
                              'vertices': len(mesh.positions),
                              'triangles': sum(len(t) // 3 for t in mesh.submeshes),
                              'bones': len(bones), 'materials': [list(m) for m in rd.materials]})
        source_files = []
        for relative, offset in sorted(env.loaded):
            path = catalog.install / relative
            source_files.append({'path': relative, 'offset': offset,
                                 'sha256': file_digest(path)})
        ident = entry['name'] + '_' + hashlib.sha256(f'{entry["cab"]}/{entry["id"]}'.encode()).hexdigest()[:10]
        avatars = []
        humanoid = {}
        for animator in animators:
            value = animator.read()
            if value.m_Avatar:
                avatar = value.m_Avatar.deref()
                raw = avatar.read_typetree()
                avatars.append({'identity': list(identity(avatar)), 'data': raw})
                flat = unwrap_data(raw)
                flat['m_TOS'] = {str(key): value for key, value in raw['m_TOS']}
                mapping = K.humanoid(flat, prefab, names)
                for human, node in mapping.items():
                    if human in humanoid and humanoid[human] != node:
                        raise ValueError(f'Avatar human bone {human} maps to two nodes')
                    humanoid[human] = node
        extra = {'source': store.SOURCE, 'entry': entry, 'coordinates': 'right-handed Y-up; Unity units',
                 'archive_scope': 'main_playable_humans', 'renderer_declarations': declarations,
                 'gender_scope': selection['gender_scope'], 'characters': selection['characters'],
                 'palette_completions': palette_completions,
                 'source_dependencies': source_dependencies,
                 'paths': {names[tid]: P.node_path(prefab, tid) for tid in prefab.nodes},
                 'renderers': renderers, 'skipped': dict(skipped), 'source_files': source_files,
                 'empty_renderers': empty,
                 'humanoid': humanoid, 'avatars': avatars, 'animators': [a.read_typetree() for a in animators],
                 'runtime_shader_status': 'preserved; GF2 lighting has not been transcribed'}
        bake.write_item('chara', ident, pieces, label=entry['name'], source=store.SOURCE,
                        nodes=nodes_table(prefab, names))
        bake.write_named_table(store.SOURCE, 'chara', ident, extra)
        loaded = bake.read_item('chara', ident)
        validate_archive(pieces, nodes_table(prefab, names), loaded)
    return ident, extra


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unwrap_data(value):
    """UnityPy's type-tree structural `data` wrappers, absent in staged JSON."""
    if isinstance(value, dict):
        if set(value) == {'data'}:
            return unwrap_data(value['data'])
        return {key: unwrap_data(v) for key, v in value.items()}
    if isinstance(value, list):
        return [unwrap_data(v) for v in value]
    return value


def validate_archive(pieces, nodes, loaded):
    """Check exact stream and declaration preservation, including authored NaNs.

    Nonfinite material metadata remains archival source data; preserving it
    does not establish that the material is suitable for rendering.
    """
    if loaded is None or len(pieces) != len(loaded.pieces) or set(nodes) != set(loaded.nodes):
        raise ValueError('bake read-back lost pieces or nodes')
    for source, actual in zip(pieces, loaded.pieces):
        for attr in ('positions', 'indices', 'normals', 'uvs', 'uv1', 'uv2', 'uv3', 'vcolor', 'tangents'):
            a, b = getattr(source.mesh, attr), getattr(actual.mesh, attr)
            if a is None and b is None:
                continue
            np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg=f'bake {attr} changed')
        # The archive stores JSON markers for nonfinite authored properties.
        # NaN is unequal to itself, so compare the serialized declarations.
        source_shader = json.dumps(source.shader, sort_keys=True, allow_nan=True)
        actual_shader = json.dumps(actual.shader, sort_keys=True, allow_nan=True)
        if source.bones != actual.bones or source_shader != actual_shader:
            raise ValueError('bake read-back changed bone order or material declarations')
        for attr in ('skin', 'binds', 'bone_world'):
            np.testing.assert_allclose(getattr(source, attr), getattr(actual, attr), rtol=1e-6, atol=1e-7)
        if len(source.shapes or []) != len(actual.shapes or []):
            raise ValueError('bake read-back changed blend shape count')
        for left, right in zip(source.shapes or [], actual.shapes or []):
            if left[0] != right[0]:
                raise ValueError('bake read-back changed blend shape name')
            for a, b in zip(left[1:], right[1:]):
                if a is None and b is None:
                    continue
                np.testing.assert_allclose(a, b, rtol=0, atol=0)
    for name, (parent, local) in nodes.items():
        if loaded.nodes[name][0] != parent:
            raise ValueError(f'bake changed parent of {name}')
        np.testing.assert_allclose(local, loaded.nodes[name][1], rtol=1e-6, atol=1e-7)
