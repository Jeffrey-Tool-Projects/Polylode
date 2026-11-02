"""Self-contained Blender rig evaluator, also embedded in each GF2 .blend.

Run the saved GF2_RigRuntime.py Text once when Blender disables script loading.
No external archives, imports from this checkout, or driver namespace is needed.
The control skeleton's source rest matrices remain float64 in the Text data.
"""
from __future__ import annotations

import json

import bpy
import numpy as np
from bpy.app.handlers import persistent
from mathutils import Matrix

_busy = False
_last = None


def source_pose(records, rests, channels):
    posed = []
    for i, record in enumerate(records):
        parent = record['parent']
        local = rests[i] if parent < 0 else np.linalg.inv(rests[parent]) @ rests[i]
        posed.append((np.eye(4) if parent < 0 else posed[parent]) @ local @ channels[i])
    return np.stack(posed)


def deform(layout, rests, posed, retained=None):
    """Move frame origins WITH the pose; float32 skinning remains near zero."""
    active = []
    for fid, frame in enumerate(layout['frames']):
        rig = bpy.data.objects.get(frame['name'])
        if rig is None:
            continue
        anchor = frame['anchor']
        origin = np.eye(4)
        origin[:3, 3] = frame['origin']
        world = posed[anchor] @ np.linalg.inv(rests[anchor]) @ origin
        holder = bpy.data.objects.get(f'GF2_FrameInstance_{fid}')
        if holder is None:
            raise ValueError('missing local deformation collection instance')
        # Object placement must be rigid. Skin matrices below retain the full
        # pose, including scale; polar decomposition removes accumulated shear
        # from the placement and leaves it in those local skin transforms.
        u, _, vt = np.linalg.svd(world[:3, :3])
        world[:3, :3] = u @ vt
        holder.matrix_world = Matrix(world.tolist())
        active.append((fid, rig, holder))
    bpy.context.view_layer.update()
    for fid, rig, holder in active:
        # Use the STORED float32 frame world for the inverse. This keeps the
        # double target world pose exact despite rounding an object's matrix.
        inverse = np.linalg.inv(np.asarray(holder.matrix_world, dtype=np.float64))
        for row in layout['instances']:
            if row['frame'] != fid or (retained is not None and row['source'] not in retained):
                continue
            pose = rig.pose.bones.get(row['name'])
            if pose is not None:
                pose.matrix = Matrix((inverse @ posed[row['source']]).tolist())


def update(force=False):
    global _busy, _last
    definition = bpy.data.texts.get('GF2_RigDefinition.json')
    controller = bpy.data.objects.get('GF2_AnimationControls')
    if _busy or definition is None or controller is None:
        return
    data = json.loads(definition.as_string())
    # Blender's channel matrix honors the active rotation mode and normalizes
    # quaternion channels. Interpolated RNA quaternions need not have unit norm.
    channels = []
    for record in data['bones']:
        bone = controller.pose.bones.get(record['name'])
        channels.append(np.eye(4) if bone is None else
                        np.asarray(bone.matrix_basis, dtype=np.float64))
    snapshot = (tuple(tuple(r) for r in controller.matrix_world),
                tuple(tuple(matrix.ravel()) for matrix in channels))
    if not force and snapshot == _last:
        return
    _busy = True
    try:
        rests = np.asarray(data['rests'], dtype=np.float64)
        posed = source_pose(data['bones'], rests, channels)
        posed = np.asarray(controller.matrix_world, dtype=np.float64) @ posed
        deform(data['layout'], rests, posed, set(data['retained']))
        _last = snapshot
    finally:
        _busy = False


@persistent
def gf2_rig_update(*_args):
    update()


def register():
    global _last
    _last = None
    for collection in (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.frame_change_post):
        for handler in list(collection):
            if handler.__name__ == 'gf2_rig_update':
                collection.remove(handler)
        collection.append(gf2_rig_update)
    update(force=True)


if __name__ == '__main__':
    register()
