"""Blender operations on GF2 translation frames and their source joint map."""
from __future__ import annotations

import json
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

from polylode.endfield import fbxfile as F
from polylode.gf2 import precision, rig_runtime
from polylode.gf2.fbx_tangents import export_scene, import_scene


def frame_document(document, arrays, fid):
    layout = document['local_frames']
    rows = [(i, r) for i, r in enumerate(layout['instances']) if r['frame'] == fid]
    local = {global_index: i for i, (global_index, _) in enumerate(rows)}
    bones = [{'name': r['name'], 'parent': local.get(r['parent'], -1)} for _, r in rows]
    source = [r['source'] for _, r in rows]
    rest = arrays['bone_rest'][source].copy()
    rest[:, :3, 3] -= layout['frames'][fid]['origin']
    return dict(document, name=layout['frames'][fid]['name'], bones=bones), {
        'bone_rest': rest, 'bone_length': arrays['bone_length'][source]}, rows


def control_rig(document, arrays, implementation):
    source = list(range(len(document['bones'])))
    remap = {s: i for i, s in enumerate(source)}
    bones = [dict(document['bones'][s], parent=remap.get(document['bones'][s]['parent'], -1)) for s in source]
    values = {'bone_rest': arrays['bone_rest'][source], 'bone_length': arrays['bone_length'][source]}
    doc = dict(document, name='GF2_AnimationControls', bones=bones)
    controller = implementation['_armature'](doc, values)
    controller['gf2_role'] = 'animation_controls'
    controller.show_in_front = True
    for bone in controller.data.bones:
        bone.use_deform = False
    for bone in controller.pose.bones:
        bone.rotation_mode = 'QUATERNION'
    return controller


def localize_collections(document):
    """Keep the modifier itself at zero; place its collection as an instance.

    Armature modifiers invert object WORLD matrices in float32, even when
    mesh and rig have the same world transform. A collection instance applies
    placement AFTER that calculation and avoids the far-origin cancellation.
    """
    layout = document['local_frames']
    for fid, frame in enumerate(layout['frames']):
        rig = bpy.data.objects.get(frame['name'])
        if rig is None:
            continue
        collection = bpy.data.collections.new(f'GF2_LocalFrame_{fid}')
        holder = bpy.data.objects.new(f'GF2_FrameInstance_{fid}', None)
        holder.instance_type = 'COLLECTION'
        holder.instance_collection = collection
        holder.matrix_world = rig.matrix_world.copy()
        bpy.context.scene.collection.objects.link(holder)
        members = [rig] + [o for o in bpy.data.objects if o.type == 'MESH' and o.parent == rig]
        for obj in members:
            for owner in list(obj.users_collection):
                owner.objects.unlink(obj)
            collection.objects.link(obj)
        rig.matrix_world = Matrix.Identity(4)
        for obj in members[1:]:
            obj.matrix_parent_inverse = Matrix.Identity(4)
            obj.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def write(document, arrays, folder, target, implementation, previews):
    implementation['_empty_scene']()
    materials = {n: implementation['_material'](n, r, folder) for n, r in document['materials'].items()}
    layout = document['local_frames']
    rigs, maps, docs = {}, {}, {}
    counts = {'degenerate': {}, 'repeated': {}}
    for fid, frame in enumerate(layout['frames']):
        doc, values, rows = frame_document(document, arrays, fid)
        rig = implementation['_armature'](doc, values)
        rig.location = frame['origin']
        rig['gf2_frame'] = fid
        rigs[fid], docs[fid] = rig, doc
        maps[fid] = {r['source']: i for i, (_, r) in enumerate(rows)}
        for _, row in rows:
            bone = rig.data.bones[row['name']]
            bone['gf2_source_name'] = document['bones'][row['source']]['name']
            bone['gf2_canonical'] = row['canonical']
    for index, part in enumerate(document['parts']):
        fid = layout['mesh_frames'][index]
        # Load only this renderer's arrays. Keeping all parts duplicated while
        # constructing a scene would defeat the batch's memory cap.
        values = {k: arrays[k] for k in arrays.files if k.startswith(f'p{index}_')}
        values[f'p{index}_co'] = values[f'p{index}_co'].astype(np.float64) - layout['frames'][fid]['origin']
        joints, weights = values[f'p{index}_jnt'], values[f'p{index}_wgt']
        remapped = np.zeros_like(joints)
        for source in np.unique(joints[weights > 0]):
            remapped[joints == source] = maps[fid][int(source)]
        values[f'p{index}_jnt'] = remapped
        degenerate, repeated = implementation['_mesh'](index, part, docs[fid], values, rigs[fid], materials)
        obj = bpy.data.objects[part['name']]
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_basis = Matrix.Identity(4)
        if degenerate:
            counts['degenerate'][part['name']] = degenerate
        if repeated:
            counts['repeated'][part['name']] = repeated
    controller = control_rig(document, arrays, implementation)
    hidden = [(o, o.hide_viewport) for o in bpy.data.objects if o.type == 'MESH']
    for obj, _ in hidden:
        obj.hide_viewport = False
    bpy.context.view_layer.update()
    counts['source_mesh_worlds'] = {o.name: np.asarray(o.matrix_world).tolist() for o in bpy.data.objects if o.type == 'MESH'}
    counts['source_armature_worlds'] = {o.name: np.asarray(o.matrix_world).tolist() for o in [*rigs.values(), controller]}
    for obj, was_hidden in hidden:
        obj.hide_viewport = was_hidden
    export_scene(filepath=str(target), use_selection=False, use_visible=False,
        object_types={'ARMATURE', 'MESH'}, add_leaf_bones=False, bake_anim=False,
        axis_forward='Y', axis_up='Z', primary_bone_axis='Y', secondary_bone_axis='X',
        apply_scale_options='FBX_SCALE_ALL', mesh_smooth_type='OFF',
        colors_type='LINEAR', path_mode='COPY', embed_textures=True)
    if document.get('save_blend') is True:
        parts = set(document['blender_selection']['parts'])
        bones = set(document['blender_selection']['bones'])
        for obj in list(bpy.data.objects):
            if obj.type == 'MESH' and obj.name not in parts:
                mesh = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if not mesh.users:
                    bpy.data.meshes.remove(mesh)
        for rig in rigs.values():
            bpy.context.view_layer.objects.active = rig
            bpy.ops.object.mode_set(mode='EDIT')
            for bone in list(rig.data.edit_bones):
                if bone['gf2_source_name'] not in bones:
                    rig.data.edit_bones.remove(bone)
            bpy.ops.object.mode_set(mode='OBJECT')
            if not rig.data.bones:
                bpy.data.objects.remove(rig, do_unlink=True)
        bpy.context.view_layer.objects.active = controller
        bpy.ops.object.mode_set(mode='EDIT')
        for bone in list(controller.data.edit_bones):
            if bone.name not in bones:
                controller.data.edit_bones.remove(bone)
        bpy.ops.object.mode_set(mode='OBJECT')
        retained = [i for i, b in enumerate(document['bones']) if b['name'] in bones]
        # Anchor selection can include an excluded branch. Use a retained
        # deformation joint of that same frame for the saved character rig.
        saved_layout = json.loads(json.dumps(layout))
        for fid, frame in enumerate(saved_layout['frames']):
            available = [r['source'] for r in layout['instances'] if r['frame'] == fid and r['source'] in retained]
            if available and frame['anchor'] not in retained:
                frame['anchor'] = available[0]
        localize_collections(document)
        definition = {'bones': document['bones'], 'rests': arrays['bone_rest'].tolist(),
                      'layout': saved_layout, 'retained': retained}
        bpy.data.texts.new('GF2_RigDefinition.json').write(json.dumps(definition))
        text = bpy.data.texts.new('GF2_RigRuntime.py')
        text.write(Path(rig_runtime.__file__).read_text(encoding='utf8'))
        text.use_module = True
        controller['GF2_animation'] = 'Run GF2_RigRuntime.py Text once if Blender blocks script loading; animate these source bones.'
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(target.with_suffix('.blend')))
    else:
        localize_collections(document)
    counts['blend_saved'] = document.get('save_blend') is True
    counts['previews'], counts['preview_visibility_override'] = previews(target.parent)
    return counts


def file_worlds(target: Path, file_to_scene: np.ndarray | None = None) -> dict[tuple[str, str], np.ndarray]:
    """Read serialized model identities and double worlds in declared axes."""
    return precision.serialized_worlds(target, file_to_scene)


def restore_import_precision(target: Path, file_to_scene: np.ndarray | None = None) -> None:
    """Recover local rest frames from the FBX doubles before float32 skinning.

    Blender's importer first casts WORLD bind matrices to float32, then
    subtracts the armature origin. Read the serialized matrices independently
    and subtract in float64 first. No source archive is used for this step.
    Ordinary exports supply their Y-up to scene conversion; localized Z-up
    exports use identity. The declared conversion is applied exactly once.
    Stored BindPose worlds take precedence over animated model defaults.
    """
    worlds = precision.serialized_rest_worlds(target, file_to_scene)
    for obj in bpy.data.objects:
        if obj.type in ('ARMATURE', 'MESH'):
            kind = 'Null' if obj.type == 'ARMATURE' else 'Mesh'
            obj.matrix_world = Matrix(worlds[kind, obj.name].tolist())
    bpy.context.view_layer.update()
    for rig in [o for o in bpy.data.objects if o.type == 'ARMATURE']:
        inverse = np.linalg.inv(worlds['Null', rig.name])
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        for bone in rig.data.edit_bones:
            rest = inverse @ worlds[F.BONE, bone.name]
            ulp = np.spacing((np.abs(rest[:3, 3]) + bone.length).astype(np.float32)).astype(np.float64)
            bone.length = max(bone.length, 2 * np.linalg.norm(ulp) / np.sin(np.radians(.01 / 2)))
            bone.matrix = Matrix(rest.tolist())
            bone.align_roll(Vector(rest[:3, 2].tolist()))
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()
    # Imported meshes can retain a bone-parent correction derived from the
    # rounded bind pose. Restore their OBJECT parent and local model transform
    # from the serialized FBX after the rest bones have been rebuilt.
    for obj in [o for o in bpy.data.objects if o.type == 'MESH']:
        modifiers = [m for m in obj.modifiers if m.type == 'ARMATURE']
        if len(modifiers) != 1 or modifiers[0].object is None:
            raise ValueError('localized mesh has an ambiguous deformation rig')
        rig = modifiers[0].object
        obj.parent = rig
        obj.parent_type = 'OBJECT'
        obj.parent_bone = ''
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_basis = Matrix((np.linalg.inv(worlds['Null', rig.name]) @ worlds['Mesh', obj.name]).tolist())
    bpy.context.view_layer.update()


def measure(document, arrays, implementation, *, saved=False):
    layout = document['local_frames']
    selection = document['blender_selection']
    selected_bones = set(selection['bones']) if saved else {b['name'] for b in document['bones']}
    selected_parts = set(selection['parts']) if saved else {p['name'] for p in document['parts']}
    instances = [r for r in layout['instances'] if document['bones'][r['source']]['name'] in selected_bones]
    rigs = {o.name: o for o in bpy.data.objects if o.type == 'ARMATURE' and o.name != 'GF2_AnimationControls'}
    meshes = {o.name: o for o in bpy.data.objects if o.type == 'MESH'}
    expected_frames = {r['frame'] for r in instances}
    controller = bpy.data.objects.get('GF2_AnimationControls')
    if controller is None or {b.name for b in controller.data.bones} != selected_bones:
        raise ValueError('source animation control skeleton changed')
    report = {'armatures': len(rigs) + 1, 'expected_frame_armatures': len(expected_frames) + 1,
              'bones': sum(len(o.data.bones) for o in rigs.values()) + len(controller.data.bones),
              'expected_instance_bones': len(instances) + len(selected_bones), 'meshes': len(meshes),
              'bones_missing': [], 'parents_wrong': 0, 'parts': [], 'images': len(bpy.data.images)}
    if set(meshes) != selected_parts:
        raise ValueError('localized scene mesh selection changed')
    hidden = {n for n, o in meshes.items() if o.hide_viewport}
    for obj in meshes.values():
        obj.hide_viewport = False
    bpy.context.view_layer.update()
    def placed_world(obj, fid):
        holder = bpy.data.objects[f'GF2_FrameInstance_{fid}']
        return np.asarray(holder.matrix_world, dtype=np.float64) @ np.asarray(obj.matrix_world, dtype=np.float64)

    part_frames = {p['name']: layout['mesh_frames'][i] for i, p in enumerate(document['parts'])}
    neutral_worlds = {n: placed_world(o, part_frames[n]).copy() for n, o in meshes.items()}
    for index, bone in enumerate(document['bones']):
        if bone['name'] not in selected_bones:
            continue
        held = controller.data.bones[bone['name']]
        parent = document['bones'][bone['parent']]['name'] if bone['parent'] >= 0 else None
        report['parents_wrong'] += (held.parent.name if held.parent else None) != parent
    rests, offsets, turns = {}, [], []
    detail = []
    for row in instances:
        rig = rigs.get(layout['frames'][row['frame']]['name'])
        bone = rig.data.bones.get(row['name']) if rig else None
        if bone is None:
            report['bones_missing'].append(row['name'])
            continue
        parent = layout['instances'][row['parent']]['name'] if row['parent'] >= 0 else None
        report['parents_wrong'] += (bone.parent.name if bone.parent else None) != parent
        rests[row['name']] = np.asarray(bone.matrix_local, dtype=np.float64)
        if row['canonical']:
            have = placed_world(rig, row['frame']) @ rests[row['name']]
            want = arrays['bone_rest'][row['source']]
            offsets.append(float(np.linalg.norm(have[:3, 3] - want[:3, 3])) * 1000)
            turns.append(F.angle(want[:3, :3], have[:3, :3]))
            detail.append({'name': row['name'], 'frame': row['frame'], 'offset_mm': offsets[-1], 'turn_deg': turns[-1],
                           'have': have[:3, 3].tolist(), 'want': want[:3, 3].tolist(),
                           'local': rests[row['name']][:3, 3].tolist(), 'origin': np.asarray(rig.matrix_world)[:3, 3].tolist()})
    report['native_bone_offset_mm'] = max(offsets, default=0)
    report['native_bone_turn_deg'] = max(turns, default=0)
    report['native_bone_worst'] = sorted(detail, key=lambda r: max(r['offset_mm'], r['turn_deg']), reverse=True)[:5]
    # Bounds per renderer avoid using an unrelated parked variant's distance
    # as the root translation. This is still a real translation of every root.
    distance = max(float(np.max(np.ptp(arrays[f'p{i}_co'], axis=0)))
                   for i, p in enumerate(document['parts']) if p['name'] in selected_parts) / 10
    channels = []
    for source, bone in enumerate(document['bones']):
        driven = controller.pose.bones.get(bone['name'])
        if driven is None:
            channels.append(np.eye(4))
            continue
        driven.rotation_mode = 'QUATERNION'
        driven.rotation_quaternion = implementation['_turn'](source)
        if bone['parent'] < 0:
            # Channel axes belong to the SOURCE joint, not the controller's
            # drawn head or a frame's origin.
            rotation = arrays['bone_rest'][source, :3, :3]
            driven.location = Vector((rotation.T @ [distance, 0, 0]).tolist())
        matrix = np.eye(4)
        matrix[:3, :3] = np.asarray(driven.rotation_quaternion.to_matrix()) * np.asarray(driven.scale)
        matrix[:3, 3] = driven.location
        channels.append(matrix)
    source_poses = rig_runtime.source_pose(document['bones'], arrays['bone_rest'], channels)
    if saved:
        # Exercise the SELF-CONTAINED evaluator saved in the reopened file.
        namespace = {'__name__': '__main__'}
        exec(compile(bpy.data.texts['GF2_RigRuntime.py'].as_string(), 'GF2_RigRuntime.py', 'exec'), namespace)  # noqa: S102 -- execute the exporter-owned embedded rig module
        namespace['update'](force=True)
    else:
        rig_runtime.deform(layout, arrays['bone_rest'], source_poses)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    shape_errors = []
    report['shape_state'] = {}
    for index, part in enumerate(document['parts']):
        if part['name'] not in selected_parts:
            continue
        obj = meshes[part['name']]
        fid = layout['mesh_frames'][index]
        rig = rigs[layout['frames'][fid]['name']]
        mesh = obj.data
        skin = source_poses @ np.linalg.inv(arrays['bone_rest'])
        co = arrays[f'p{index}_co'].astype(np.float64)
        joints = arrays[f'p{index}_jnt']
        weights = arrays[f'p{index}_wgt'].astype(np.float64)
        total = weights.sum(1, keepdims=True)
        blend = np.einsum('vk,vkij->vij', weights / np.where(total > 0, total, 1), skin[joints])
        blend[total[:, 0] <= 0] = np.eye(4)
        expect = np.einsum('vij,vj->vi', blend, np.c_[co, np.ones(len(co))])[:, :3]
        actual_local = implementation['_world_positions'](obj, depsgraph)
        holder = bpy.data.objects[f'GF2_FrameInstance_{fid}']
        placement = np.asarray(holder.matrix_world, dtype=np.float64)
        actual = actual_local @ placement[:3, :3].T + placement[:3, 3]
        raw = np.asarray([tuple(v.co) for v in mesh.vertices])
        # Rest is read from the serialized mesh frame, before the frame moves
        # with the pose. Posed geometry uses the actual evaluated object world.
        neutral = neutral_worlds[part['name']]
        rest = raw @ neutral[:3, :3].T + neutral[:3, 3]
        if raw.shape != co.shape:
            raise ValueError('localized vertex count changed')
        row = {'name': part['name'], 'found': True, 'hidden': part['name'] in hidden,
               'vertices': len(mesh.vertices), 'triangles': len(mesh.polygons),
               'armature': any(m.type == 'ARMATURE' and m.object == rig for m in obj.modifiers),
               'weighted': sum(bool(v.groups) for v in mesh.vertices),
               'weighted_expected': int((total[:, 0] > 0).sum()),
               'shape_keys': len(mesh.shape_keys.key_blocks)-1 if mesh.shape_keys else 0,
               'rest_mm': float(np.linalg.norm(rest-co, axis=1).max()) * 1000,
               'posed_mm': float(np.linalg.norm(actual-expect, axis=1).max()) * 1000,
               'moved_mm': float(np.linalg.norm(expect-co, axis=1).max()) * 1000}
        if row['posed_mm'] > .05:
            row['frame_world'] = np.asarray(rig.matrix_world).tolist()
            row['mesh_world'] = np.asarray(obj.matrix_world).tolist()
            row['mesh_parent'] = obj.parent.name if obj.parent else None
            row['sample_actual'] = actual[0].tolist()
            row['sample_expected'] = expect[0].tolist()
            row['parent_inverse'] = np.asarray(obj.matrix_parent_inverse).tolist()
            row['parent_type'] = obj.parent_type
            row['parent_bone'] = obj.parent_bone
            row['basis'] = np.asarray(obj.matrix_basis).tolist()
            row['delta_rotation'] = list(obj.delta_rotation_euler)
            row['constraints'] = [c.type for c in obj.constraints]
        report['parts'].append(row)
        if mesh.shape_keys:
            keys = mesh.shape_keys.key_blocks
            report['shape_state'][part['name']] = [(k.name, k.value, k.vertex_group) for k in keys]
            basis = np.asarray([tuple(v.co) for v in keys[0].data])
            for k, name in enumerate(part['shapes']):
                expected = np.zeros_like(basis)
                expected[arrays[f'p{index}_s{k}_idx']] = arrays[f'p{index}_s{k}_delta']
                delta = np.asarray([tuple(v.co) for v in keys[name].data]) - basis
                shape_errors.append(float(np.linalg.norm(delta-expected, axis=1).max()) * 1000)
    report['shape_delta_mm'] = max(shape_errors, default=0)
    report['pose_root_translation_m'] = distance
    if saved:
        report['blend_bone_offset_mm'] = report['native_bone_offset_mm']
        report['blend_bone_turn_deg'] = report['native_bone_turn_deg']
        report['blend_rest_mm'] = max(p['rest_mm'] for p in report['parts'])
        report['blend_posed_mm'] = max(p['posed_mm'] for p in report['parts'])
        report['blend_parts'] = sorted(meshes)
        report['blend_bones'] = len(selected_bones)
        report['blend_animation_control_bones'] = len(controller.data.bones)
        report['blend_visibility_changed'] = [p['name'] for p in document['parts'] if p['name'] in selected_parts
                                               if (p['name'] in hidden) != p['hidden']
                                               or meshes[p['name']].hide_render != p['hidden']]
    return report


def check(document, arrays, target, implementation):
    implementation['_empty_scene']()
    import_scene(filepath=str(target))
    restore_import_precision(target)
    localize_collections(document)
    report = measure(document, arrays, implementation)
    report['import_precision_method'] = 'FBX float64 rest to local frame before float32 storage'
    if document.get('save_blend') is True:
        bpy.ops.wm.open_mainfile(filepath=str(target.with_suffix('.blend')))
        blend = measure(document, arrays, implementation, saved=True)
        report.update({k: v for k, v in blend.items() if k.startswith('blend_')})
    return report
