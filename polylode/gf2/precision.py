"""Write GF2's validated double precision rest transforms into Blender's FBX.

Blender stores edit bones as float32 head/tail/roll. FBX model scaling, cluster
matrices and BindPose records must all be written from the same source rest
matrices. Updating only Euler angles leaves their accumulated scaling and
bind disagreement intact. Mesh coordinates and weights are never modified.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ..endfield import fbxfile as F


def bone_errors(path, bones, file_to_scene=F.FILE_TO_SCENE):
    """Use the shared binary reader with the file's declared export axes."""
    change = F.FILE_TO_SCENE @ np.asarray(file_to_scene).T
    return F.bone_errors(path, [(name, parent, change @ rest) for name, parent, rest in bones])


def bone_worlds(path, file_to_scene=F.FILE_TO_SCENE):
    change = np.asarray(file_to_scene) @ F.FILE_TO_SCENE.T
    return {name: change @ rest for name, rest in F.bone_worlds(Path(path).read_bytes()).items()}


def serialized_worlds(path: Path, file_to_scene=None) -> dict[tuple[str, str], np.ndarray]:
    """Read model worlds in float64, applying the declared file axes once.

    Model kind remains part of the identity because a renderer and a source
    joint can legitimately share a name. No source archive supplies matrices.
    """
    axes = np.eye(4) if file_to_scene is None else np.asarray(file_to_scene, dtype=np.float64)
    if axes.shape != (4, 4) or not np.isfinite(axes).all():
        raise F.FbxError('invalid FBX file-to-scene transform')
    nodes = F.read_nodes(Path(path).read_bytes())
    matrices, visiting = {}, set()

    def world(uid):
        if uid not in matrices:
            if uid in visiting:
                raise F.FbxError('cyclic FBX model parents')
            visiting.add(uid)
            node = nodes[uid]
            matrices[uid] = (world(node.parent) if node.parent else np.eye(4)) @ F._local(node)
            visiting.remove(uid)
        return matrices[uid]

    result = {}
    for uid, node in nodes.items():
        key = (node.kind, node.name)
        if key in result:
            raise F.FbxError('ambiguous FBX model kind and name')
        result[key] = axes @ world(uid)
    return result


def serialized_rest_worlds(path: Path, file_to_scene=None) -> dict[tuple[str, str], np.ndarray]:
    """Read BindPose worlds, retaining model defaults for nodes without a pose.

    Animated FBX model defaults may describe a sampled pose rather than the
    skin's rest. BindPose matrices are already global; no parent composition
    applies to them. Conflicting rest declarations are rejected before use.
    """
    result = serialized_worlds(path, file_to_scene)
    axes = np.eye(4) if file_to_scene is None else np.asarray(file_to_scene, dtype=np.float64)
    data = Path(path).read_bytes()
    nodes = F.read_nodes(data)
    poses = {}
    for name, props, start, stop in F._records(data, *F._top(data)[b'Objects']):
        if name != b'Pose' or len(props) < 3 or props[2][1] != b'BindPose':
            continue
        for kind, _, inner, end in F._records(data, start, stop):
            if kind != b'PoseNode':
                continue
            fields = {key: values[0] for key, values, _, _ in F._records(data, inner, end)}
            if b'Node' not in fields or b'Matrix' not in fields:
                raise F.FbxError('incomplete FBX bind pose node')
            uid = fields[b'Node'][1]
            values = F._array(data, fields[b'Matrix'])
            if uid not in nodes or values.shape != (16,) or not np.isfinite(values).all():
                raise F.FbxError('invalid FBX bind pose node or matrix')
            matrix = values.reshape(4, 4).T
            if not np.array_equal(matrix[3], [0, 0, 0, 1]):
                raise F.FbxError('invalid homogeneous FBX bind pose matrix')
            if uid in poses and not np.array_equal(poses[uid], matrix):
                raise F.FbxError('conflicting FBX bind pose matrices')
            poses[uid] = matrix
    for uid, matrix in poses.items():
        node = nodes[uid]
        result[node.kind, node.name] = axes @ matrix
    return result


def write_rest(path: Path, bones, source_mesh_worlds=None, *,
               file_to_scene=F.FILE_TO_SCENE, armature_worlds=None):
    original = Path(path).read_bytes()
    data = bytearray(original)
    nodes = F.read_nodes(original)
    source = {name: np.asarray(rest, dtype=np.float64) for name, _, rest in bones}
    file_to_scene = np.asarray(file_to_scene, dtype=np.float64)
    file_rest = {name: file_to_scene.T @ rest for name, rest in source.items()}
    by_name = {n.name: n for n in nodes.values() if n.kind == F.BONE}
    if len(source) != len(bones) or set(source) != set(by_name):
        raise F.FbxError('source and FBX bone identities differ')
    before = bone_errors(path, bones, file_to_scene)
    edits = []

    def local_edits(node, matrix):
        for key, values, least in [('Lcl Rotation', F.euler_xyz(matrix[:3, :3]), F.UNWRITTEN_DEG),
                                   ('Lcl Translation', matrix[:3, 3], F.UNWRITTEN_M),
                                   ('Lcl Scaling', [1., 1., 1.], 0.)]:
            if key in node.lcl:
                edits.extend(zip(node.lcl[key][1], values))
            elif key != 'Lcl Scaling' and max(abs(float(v)) for v in values) > least:
                raise F.FbxError(f'{node.name}: missing {key} record')

    roots = {by_name[name].parent for name, parent, _ in bones if parent < 0}
    if armature_worlds is None and len(roots) != 1:
        raise F.FbxError('expected one source armature axes node')
    axis_worlds = {}
    for uid in roots:
        node = nodes.get(uid)
        if node is None or node.kind != 'Null' or node.parent:
            raise F.FbxError('unsupported armature axes hierarchy')
        world = np.eye(4) if armature_worlds is None else np.asarray(armature_worlds[node.name])
        if (world.shape != (4, 4) or not np.isfinite(world).all()
                or not np.array_equal(world[:3, :3], np.eye(3))
                or not np.array_equal(world[3], [0, 0, 0, 1])):
            raise F.FbxError('armature frame is not a translation')
        axis_worlds[uid] = world
    # Blender also rounds the global -90 degree axis change. At large source
    # coordinates that alone exceeds the position tolerance. This is the exact
    # same documented world-space conversion used by all meshes and clusters.
    for uid, world in axis_worlds.items():
        local_edits(nodes[uid], file_to_scene.T @ world)
    certified = source_mesh_worlds is not None
    if certified:
        mesh_nodes = {n.name: n for n in nodes.values() if n.kind == 'Mesh'}
        if set(mesh_nodes) != set(source_mesh_worlds):
            raise F.FbxError('source scene and FBX mesh identities differ')
        for name, matrix in source_mesh_worlds.items():
            matrix = np.asarray(matrix, dtype=np.float64)
            if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or not np.array_equal(matrix[:3, :3], np.eye(3))
                    or not np.array_equal(matrix[3], [0, 0, 0, 1])
                    or (armature_worlds is None and not np.array_equal(matrix, np.eye(4)))):
                raise F.FbxError('source scene mesh world is not the baked identity transform')
            node = mesh_nodes[name]
            if node.parent and node.parent not in axis_worlds:
                raise F.FbxError('unsupported source mesh parent')
            local_edits(node, np.linalg.inv(axis_worlds[node.parent]) @ matrix
                        if node.parent else file_to_scene.T @ matrix)
    for name, parent, rest in bones:
        node = by_name[name]
        above = nodes.get(node.parent)
        expected_parent = bones[parent][0] if parent >= 0 else None
        if (above.name if above and above.kind == F.BONE else None) != expected_parent:
            raise F.FbxError(f'{name}: source and FBX parents differ')
        rest = source[name]
        if (not np.isfinite(rest).all() or not np.allclose(rest[3], [0, 0, 0, 1], atol=1e-12)
                or not np.allclose(rest[:3, :3].T @ rest[:3, :3], np.eye(3), atol=1e-12)
                or np.linalg.det(rest[:3, :3]) < 0):
            raise F.FbxError(f'{name}: source rest is not rigid')
        local = (np.linalg.inv(axis_worlds[node.parent]) @ rest if parent < 0
                 else np.linalg.inv(source[expected_parent]) @ rest)
        local_edits(node, local)

    top = F._top(original)
    objects = list(F._records(original, *top[b'Objects']))
    clusters = {}
    for name, properties, start, stop in objects:
        if name == b'Deformer':
            matrices = {held: values[0] for held, values, _, _ in F._records(original, start, stop)
                        if held in (b'Transform', b'TransformLink')}
            if b'TransformLink' in matrices:
                clusters[int(properties[0][1])] = matrices
    links = {}
    object_parents = {}
    for _, props, _, _ in F._records(original, *top[b'Connections']):
        child, parent = props[1][1], props[2][1]
        if props[0][1] == b'OO':
            object_parents.setdefault(child, []).append(parent)
        if props[0][1] == b'OO' and child in nodes and parent in clusters:
            if nodes[child].kind != F.BONE or parent in links:
                raise F.FbxError('ambiguous cluster bone connection')
            links[parent] = nodes[child].name
    if set(links) != set(clusters) or not links:
        raise F.FbxError('unresolved skin cluster bones')

    def matrix_edit(prop, value):
        if prop[0] != b'd' or prop[1] != (16, 0, 128):
            raise F.FbxError('expected uncompressed double 4x4 FBX matrix')
        offset = prop[2]
        edits.extend((offset + i * 8, v) for i, v in enumerate(value.T.ravel()))

    # The scene builder bakes vertices into world space and gives every mesh an
    # identity world transform. FBX changes world axes but not those coordinates.
    for uid, matrices in clusters.items():
        old_link = F._array(original, matrices[b'TransformLink']).reshape(4, 4).T
        old_transform = F._array(original, matrices[b'Transform']).reshape(4, 4).T
        recovered_mesh = file_to_scene @ old_link @ old_transform
        if not certified and (np.linalg.norm(recovered_mesh[:3, 3]) * 1000 > .01
                or not np.allclose(recovered_mesh[:3, :3], np.eye(3), atol=1e-4)):
            raise F.FbxError('cluster mesh is not the baked identity scene transform')
        bone_world = file_rest[links[uid]]
        mesh_world = file_to_scene.T
        if certified:
            # Cluster -> Skin -> Geometry -> Mesh model. No association may
            # be inferred from names: the file's OO graph owns the binding.
            frontier = [uid]
            reached, mesh_uids = set(), set()
            while frontier:
                held = frontier.pop()
                if held in reached:
                    continue
                reached.add(held)
                node = nodes.get(held)
                if node is not None:
                    if node.kind == 'Mesh':
                        mesh_uids.add(held)
                    continue
                frontier.extend(object_parents.get(held, []))
            if len(mesh_uids) != 1:
                # Earlier callers certify an identity scene; the synthetic
                # reader fixtures have no geometry connection graph.
                if armature_worlds is not None:
                    raise F.FbxError('unresolved cluster mesh frame')
            else:
                mesh_world = file_to_scene.T @ np.asarray(source_mesh_worlds[nodes[next(iter(mesh_uids))].name])
        matrix_edit(matrices[b'TransformLink'], bone_world)
        matrix_edit(matrices[b'Transform'], np.linalg.inv(bone_world) @ mesh_world)
    poses = 0
    for name, _, start, stop in objects:
        if name != b'Pose':
            continue
        for held, _, inner, end in F._records(original, start, stop):
            if held != b'PoseNode':
                continue
            records = {key: props[0] for key, props, _, _ in F._records(original, inner, end)}
            node = nodes.get(records[b'Node'][1])
            if node and node.kind == F.BONE:
                matrix_edit(records[b'Matrix'], file_rest[node.name])
                poses += 1
            elif node and node.uid in axis_worlds:
                matrix_edit(records[b'Matrix'], file_to_scene.T @ axis_worlds[node.uid])
            elif node and node.kind == 'Mesh' and certified:
                matrix_edit(records[b'Matrix'], file_to_scene.T @ np.asarray(source_mesh_worlds[node.name]))
    if not poses:
        raise F.FbxError('missing source bone bind poses')
    for offset, value in edits:
        struct.pack_into('<d', data, offset, float(value))
    # No write occurs until every source identity, relationship and record passes.
    Path(path).write_bytes(data)
    return {'method': 'source_rest_float64_models_clusters_bindposes',
            'before': before, 'cluster_pairs': len(links), 'bind_pose_bones': poses,
            'mesh_transform_source': 'measured Blender source scene' if certified else 'original cluster pair',
            'exact_world_axes': True}
