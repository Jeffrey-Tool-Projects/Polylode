"""An explicitly declared affine source rig combined with its native skin.

The source's own converted joint program supplies its declared helper writes.
Original affine ancestry, all source weights and the registered transition
remain data. No pose fitting or character-specific branches are involved.
"""
from __future__ import annotations

import numpy as np

from ..core import math3d as m3
from ..mesh import skinning
from ..mesh.skeleton import Skeleton
from ..origin.joint_correct import Corrector


def load_program(descriptor, root, bone_index):
    """Load only the declared, hash-bound files inside this candidate store."""
    import json
    from .body_archive import file_digest
    from ..mesh.skin import validate_streams
    if descriptor.get('version') != 1 or descriptor.get('kind') != 'affine_source_ancestry':
        raise ValueError('unsupported declared affine source program')
    paths = {}
    for name in ('metadata', 'arrays'):
        entry = descriptor[name]
        path = (root / entry['file']).resolve()
        if (not path.is_relative_to(root.resolve()) or not path.is_file()
                or file_digest(path) != entry['sha256']):
            raise ValueError('declared affine source dependency is changed or outside its store')
        paths[name] = path
    metadata = json.loads(paths['metadata'].read_text('utf-8'))
    if (metadata.get('formal_sha256') != descriptor.get('scene_sha256')
            or metadata.get('arrays_sha256') != descriptor['arrays']['sha256']):
        raise ValueError('declared source program metadata belongs to another scene or array archive')
    with np.load(paths['arrays'], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    count = len(arrays['rest'])
    for idx, wt, bones in [('native_idx', 'native_wt', metadata['native_bones']),
                           ('source_idx', 'source_wt', metadata['source_bones'])]:
        validate_streams(arrays[idx], arrays[wt], count)
        if np.any(arrays[idx] >= len(bones)):
            raise ValueError('declared affine source weight palette is out of range')
    if (not all(np.isfinite(a).all() for a in arrays.values())
            or arrays['rest'].shape != (count, 3)
            or arrays['source_root_rest_position_986'].shape != (count, 3)
            or arrays['own_actual_rest_position_1058'].shape != (count, 3)
            or any(arrays[name].shape != (count,) for name in
                   ('source_scope_986', 'own_source_native_alpha_1058', 'own_source_native_support_1058'))
            or np.any(arrays['own_source_native_alpha_1058'] < 0)
            or np.any(arrays['own_source_native_alpha_1058'] > 1)):
        raise ValueError('invalid declared affine source geometry or transition')
    return AffineSourceMotion(metadata, arrays, bone_index)


def _proper(matrix):
    u, _, vt = np.linalg.svd(matrix)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt


class AffineSourceMotion:
    """Evaluate one pinned source program in its declared native bone frame."""

    def __init__(self, metadata, arrays, bone_index):
        self.meta, self.data = metadata, arrays
        self.native = np.array([bone_index[n] for n in metadata['native_bones']], np.int32)
        self.skeleton = Skeleton('declared affine source')
        hierarchy = metadata['source_hierarchy']
        pending = set(hierarchy)
        while pending:
            ready = sorted(n for n in pending if not hierarchy[n]['parent']
                           or self.skeleton.has(hierarchy[n]['parent']))
            if not ready:
                raise ValueError('declared source hierarchy is cyclic or incomplete')
            for name in ready:
                parent = hierarchy[name]['parent']
                t, q, scale = m3.decompose(np.asarray(hierarchy[name]['local']))
                self.skeleton.add(name, parent or None, t, q, scale)
                pending.remove(name)
        self.skeleton.finalize()
        self.worlds = np.asarray([hierarchy[b.name]['world'] for b in self.skeleton.bones])
        self.corrector = Corrector(self.skeleton, metadata['joint_program'])
        self.original = self._locals({})
        self.baseline = self._locals(self.corrector.apply({}))
        self.helpers = [self.skeleton.index[n] for n in metadata['helper_nodes']]
        self.source = [self.skeleton.index[n] for n in metadata['source_bones']]
        self.frame = np.asarray(metadata['native_to_source_frame'], np.float64)
        self.root = np.asarray(metadata['root_fit'], np.float64)
        self.inverse_root = np.linalg.inv(self.root)
        self.body = np.asarray(arrays['body_world'], np.float64)
        self.inverse_body = np.linalg.inv(self.body)
        self.rests = np.asarray(arrays['native_rest'], np.float64)
        self.inverse_rests = np.linalg.inv(self.rests)
        self.native_local = self.rests.copy()
        self.parents = np.asarray(arrays['native_parents'], np.int64)
        for i, parent in enumerate(self.parents):
            above = arrays['rig_world'] if parent < 0 else self.rests[parent]
            self.native_local[i] = np.linalg.inv(above) @ self.rests[i]
        self.inverse_native_local = np.linalg.inv(self.native_local)
        self.skin_rests = skinning.rest_frames(self.inverse_body @ self.rests)
        self.last = None
        self.result = None

    def _locals(self, pose):
        matrices = []
        for bone in self.skeleton.bones:
            entry = pose.get(bone.name, {})
            matrices.append(m3.compose(
                bone.local_pos + np.asarray(entry.get('pos', [0, 0, 0])),
                m3.quat_mul(bone.local_rot, np.asarray(entry.get('rot', [0, 0, 0, 1]))),
                bone.local_scale * np.asarray(entry.get('scale', [1, 1, 1]))))
        return np.asarray(matrices)

    def points(self, matrices):
        native_skin = self.frame @ matrices[self.native] @ np.linalg.inv(self.frame)
        if self.last is not None and np.array_equal(native_skin, self.last):
            return self.result
        world = native_skin @ self.rests
        basis = np.empty_like(world)
        for i, parent in enumerate(self.parents):
            above = self.data['rig_world'] if parent < 0 else world[parent]
            basis[i] = self.inverse_native_local[i] @ np.linalg.inv(above) @ world[i]
        source_pose = {}
        for driver in self.meta['drivers']:
            axes = np.asarray(driver['axes'])
            delta = np.eye(4)
            delta[:3, :3] = axes @ _proper(basis[driver['native_slot'], :3, :3]) @ axes.T
            source_pose[driver['source_node']] = {'rot': m3.decompose(delta)[1]}
        raw = self._locals(source_pose)
        corrected = self._locals(self.corrector.apply(source_pose))
        anchored = raw.copy()
        for i in self.helpers:
            anchored[i] = self.original[i] @ np.linalg.inv(self.baseline[i]) @ corrected[i]
        deltas = []
        for i, bone in enumerate(self.skeleton.bones):
            delta = (self.worlds[i] @ np.linalg.inv(self.original[i]) @ anchored[i]
                     @ np.linalg.inv(self.worlds[i]))
            deltas.append(delta if bone.parent < 0 else deltas[bone.parent] @ delta)
        hip = self.meta['common_hip_slot']
        hip_delta = self.inverse_body @ world[hip] @ self.inverse_rests[hip] @ self.body
        palette = np.asarray([hip_delta @ self.root @ deltas[i] @ self.inverse_root
                              for i in self.source])
        native_local_skin = self.inverse_body @ native_skin @ self.body
        d = self.data
        result = skinning.skinned(d['rest'], d['native_idx'], d['native_wt'], native_local_skin,
                                  self.meta['native_skinning'], self.skin_rests)

        def source_points(points):
            chosen = palette[d['source_idx']]
            moved = np.einsum('vbij,vj->vbi', chosen[:, :, :3, :3], points) + chosen[:, :, :3, 3]
            return np.sum(d['source_wt'][..., None] * moved, axis=1)

        scope = d['source_scope_986'].astype(bool)
        result[scope] = source_points(d['source_root_rest_position_986'])[scope]
        support = d['own_source_native_support_1058'].astype(bool)
        alpha = d['own_source_native_alpha_1058']
        full = source_points(d['own_actual_rest_position_1058'])
        result[support] += alpha[support, None] * (full[support] - result[support])
        # Body object space -> project space, the same exported point convention.
        transform = np.linalg.inv(self.frame) @ self.body
        result = result @ transform[:3, :3].T + transform[:3, 3]
        if not np.isfinite(result).all():
            raise ValueError('declared affine source program produced nonfinite positions')
        self.last, self.result = native_skin.copy(), result
        return result


class SourceSurface:
    """Upload the declared posed positions once, with GPU skinning disabled."""

    def __init__(self, part, motion, source_vertices):
        self.part, self.motion, self.source_vertices = part, motion, source_vertices

    def update(self, matrices, rests):
        positions = self.motion.points(matrices)[self.source_vertices].astype(np.float32)
        if self.part.skinned or not np.array_equal(self.part.mesh.positions, positions):
            self.part.mesh.positions = positions
            self.part.skinned = False
            self.part.stale = True
