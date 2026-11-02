"""Blender entry point reusing IslandLife's existing rigged scene writer."""
import json
import runpy
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from polylode.endfield.fbxfile import angle


def previews(folder):
    """Inspect a disabled source variant without changing its saved visibility."""
    scene = bpy.context.scene
    visible = [o for o in bpy.data.objects if o.type == 'MESH' and not o.hide_render]
    override = []
    if not visible:
        visible = [o for o in bpy.data.objects if o.type == 'MESH']
        override = [(o, o.hide_viewport, o.hide_render) for o in visible]
        for obj, _, _ in override:
            obj.hide_viewport = obj.hide_render = False
        bpy.context.view_layer.update()
    visible_names = {o.name for o in visible}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = [instance.matrix_world @ Vector(corner)
              for instance in depsgraph.object_instances
              if instance.object.type == 'MESH' and instance.object.original.name in visible_names
              for corner in instance.object.bound_box]
    if not points:
        raise ValueError('no visible geometry for inspection')
    bounds = np.asarray([tuple(p) for p in points])
    center = Vector(((bounds.min(axis=0) + bounds.max(axis=0)) * .5).tolist())
    span = float(np.max(np.ptp(bounds, axis=0)))
    camera = bpy.data.objects.new('InspectionCamera', bpy.data.cameras.new('InspectionCamera'))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = span * 1.2
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'TEXTURE'
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = False
    scene.display.shading.background_type = 'WORLD'
    scene.world = scene.world or bpy.data.worlds.new('InspectionWorld')
    scene.world.color = (.15, .15, .15)
    scene.render.resolution_x = scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    paths = []
    try:
        for name, direction in [('front', (0, -1, 0)), ('back', (0, 1, 0)),
                                ('left', (-1, 0, 0)), ('right', (1, 0, 0))]:
            camera.location = center + Vector(direction) * span * 3
            camera.rotation_euler = (center - camera.location).to_track_quat('-Z', 'Y').to_euler()
            path = folder / f'preview_{name}.png'
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            paths.append(path.name)
    finally:
        for obj, viewport, render in override:
            obj.hide_viewport, obj.hide_render = viewport, render
    return paths, [o.name for o, _, _ in override]

shared = Path(__file__).resolve().parents[2] / 'tools' / 'ef_fbx_blender.py'
if not shared.is_file():
    raise FileNotFoundError('FBX export needs the source checkout tools/ef_fbx_blender.py')
implementation = runpy.run_path(str(shared))
original_mesh = implementation['_mesh']
original_armature = implementation['_armature']


def precise_armature(document, arrays):
    # Tail minus head is a float32 direction. Its component error is bounded
    # by the sum of endpoint ULPs. Allocate half the existing angular tolerance
    # to that error; the other half covers matrix/roll storage. Length is only
    # an editor display extent and does not change a joint or a skin transform.
    rest = arrays['bone_rest']
    lengths = arrays['bone_length'].copy()
    extent = np.abs(rest[:, :3, 3]) + lengths[:, None]
    ulp = np.spacing(extent.astype(np.float32)).astype(np.float64)
    minimum = 2 * np.linalg.norm(ulp, axis=1) / np.sin(np.radians(.01 / 2))
    values = dict(arrays)
    values['bone_length'] = np.maximum(lengths, minimum)
    return original_armature(document, values)


def neutral_mesh(*args):
    counts = original_mesh(*args)
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                key.value = 0.0
    return counts


implementation['write'].__globals__['_mesh'] = neutral_mesh
implementation['write'].__globals__['_armature'] = precise_armature
implementation['_mesh'] = neutral_mesh
implementation['_armature'] = precise_armature
mode, document_path, target, report_path = sys.argv[sys.argv.index('--') + 1:]
document_path, target, report_path = map(Path, (document_path, target, report_path))
document = json.loads(document_path.read_text(encoding='utf8'))
with np.load(document_path.parent / document['arrays']) as arrays:
    if document.get('local_frames'):
        from polylode.gf2 import frame_worker
        if mode == 'write':
            report = frame_worker.write(document, arrays, document_path.parent, target, implementation, previews)
        elif mode == 'check':
            report = frame_worker.check(document, arrays, target, implementation)
        else:
            raise ValueError(mode)
    elif mode == 'write':
        report = implementation['write'](document, arrays, document_path.parent, target)
        report['source_mesh_worlds'] = {o.name: np.asarray(o.matrix_world).tolist()
                                       for o in bpy.data.objects if o.type == 'MESH'}
        if document.get('save_blend') is True:
            selected = set(document['blender_selection']['parts'])
            for obj in list(bpy.data.objects):
                if obj.type == 'MESH' and obj.name not in selected:
                    mesh = obj.data
                    bpy.data.objects.remove(obj, do_unlink=True)
                    if not mesh.users:
                        bpy.data.meshes.remove(mesh)
            armature = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='EDIT')
            retained = set(document['blender_selection']['bones'])
            for bone in list(armature.data.edit_bones):
                if bone.name not in retained:
                    armature.data.edit_bones.remove(bone)
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.file.pack_all()
            bpy.ops.wm.save_as_mainfile(filepath=str(target.with_suffix('.blend')))
        report['blend_saved'] = document.get('save_blend') is True
        report['previews'], report['preview_visibility_override'] = previews(target.parent)
    elif mode == 'check':
        from polylode.gf2 import frame_worker
        from polylode.endfield import fbxfile

        original_import = implementation['check'].__globals__['import_scene']

        def precise_import(**options):
            result = original_import(**options)
            frame_worker.restore_import_precision(Path(options['filepath']), fbxfile.FILE_TO_SCENE)
            return result

        implementation['check'].__globals__['import_scene'] = precise_import
        # A root translation also excites vertices located exactly on their
        # rotation pivot. Its distance is derived from the source scene bounds.
        bounds = np.concatenate([arrays[f'p{i}_co'] for i in range(len(document['parts']))])
        distance = float(np.max(np.ptp(bounds, axis=0))) / 10
        original_turn = implementation['_turn']

        def translated_turn(index):
            if document['bones'][index]['parent'] < 0:
                armature = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
                bone = armature.pose.bones[document['bones'][index]['name']]
                bone.location = bone.bone.matrix_local.to_3x3().inverted() @ Vector((distance, 0, 0))
            return original_turn(index)

        implementation['check'].__globals__['_turn'] = translated_turn
        report = implementation['check'](document, arrays, target)
        report['pose_root_translation_m'] = distance
        report['shape_state'] = {o.name: [(k.name, k.value, k.vertex_group) for k in o.data.shape_keys.key_blocks]
                                 for o in bpy.data.objects if o.type == 'MESH' and o.data.shape_keys}
        shape_errors = []
        for index, part in enumerate(document['parts']):
            if not part['shapes']:
                continue
            obj = bpy.data.objects[part['name']]
            keys = obj.data.shape_keys.key_blocks
            basis = np.asarray([tuple(v.co) for v in keys[0].data])
            for k, name in enumerate(part['shapes']):
                key = keys.get(name)
                if key is None:
                    raise ValueError(f'missing blend shape {name}')
                expected = np.zeros_like(basis)
                expected[arrays[f'p{index}_s{k}_idx']] = arrays[f'p{index}_s{k}_delta']
                actual = np.asarray([tuple(v.co) for v in key.data]) - basis
                shape_errors.append(float(np.linalg.norm(actual - expected, axis=1).max()) * 1000)
        report['shape_delta_mm'] = max(shape_errors, default=0)
        if document.get('save_blend') is True:
            bpy.ops.wm.open_mainfile(filepath=str(target.with_suffix('.blend')))
            armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
            if len(armatures) != 1:
                raise ValueError('saved Blender file does not contain one armature')
            armature = armatures[0]
            expected_bones = set(document['blender_selection']['bones'])
            actual_bones = {b.name for b in armature.data.bones}
            if actual_bones != expected_bones:
                raise ValueError('saved Blender character bone selection changed')
            expected_parts = set(document['blender_selection']['parts'])
            actual_parts = {o.name for o in bpy.data.objects if o.type == 'MESH'}
            if actual_parts != expected_parts:
                raise ValueError('saved Blender includes excluded meshes or loses character meshes')
            offsets, turns = [], []
            for index, record in enumerate(document['bones']):
                if record['name'] not in expected_bones:
                    continue
                bone = armature.data.bones[record['name']]
                have = np.asarray(armature.matrix_world) @ np.asarray(bone.matrix_local)
                want = arrays['bone_rest'][index]
                offsets.append(float(np.linalg.norm(have[:3, 3] - want[:3, 3])) * 1000)
                turns.append(angle(want[:3, :3], have[:3, :3]))
            report['blend_bone_offset_mm'] = max(offsets, default=0)
            report['blend_bone_turn_deg'] = max(turns, default=0)
            report['blend_visibility_changed'] = [p['name'] for p in document['parts'] if p['name'] in expected_parts
                if (bpy.data.objects[p['name']].hide_viewport != p['hidden']
                    or bpy.data.objects[p['name']].hide_render != p['hidden'])]
            report['blend_parts'] = sorted(actual_parts)
            report['blend_bones'] = len(actual_bones)
            mesh_errors = []
            for i, part in enumerate(document['parts']):
                if part['name'] not in expected_parts:
                    continue
                obj = bpy.data.objects[part['name']]
                actual = np.asarray([tuple(v.co) for v in obj.data.vertices])
                expected = arrays[f'p{i}_co']
                if actual.shape != expected.shape:
                    raise ValueError('saved Blender character vertex count changed')
                mesh_errors.append(float(np.max(np.linalg.norm(actual - expected, axis=1))) * 1000)
                if obj.data.shape_keys and any(k.value != 0 for k in obj.data.shape_keys.key_blocks):
                    raise ValueError('saved Blender character expression is not neutral')
            report['blend_rest_mm'] = max(mesh_errors, default=0)
    else:
        raise ValueError(mode)
report_path.write_text(json.dumps(report, indent=1), encoding='utf8')
