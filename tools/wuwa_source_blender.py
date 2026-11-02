"""Run inside Blender: build a Wuthering Waves display model's original clothed scene, or read it back.

    blender -b --factory-startup --python-exit-code 1 --python tools/wuwa_source_blender.py -- build <case> <ident>
    blender -b --factory-startup --python-exit-code 1 --python tools/wuwa_source_blender.py -- check <case> <ident>

``<case>`` holds ``source_document.json`` and its arrays
(``polylode/wuthering_waves/source_scene.py``).  **build** makes the
materials (the base colour map connected, every other native map loaded and
labelled, a mask cutoff where the material blends by mask), the armature
(:func:`armature`: Blender keeps a bone as head, tail and roll, so a virtual
tail is moved until each bone's rest rotation is the native one within
0.001°), the meshes, and saves ``<ident>.blend`` with its images packed and
``<ident>.fbx`` with them embedded; the FBX bones' rest transforms are then
written again in 64 bits.  **check** imports the FBX, turns every bone and
compares the deformed vertices with the same blend done from the native
frames, then opens the ``.blend`` and does the same; any bone or vertex over
its tolerance stops it.

Moved from ``dev/wuwa_source_blender_1121.py`` and ``dev/wuwa_armature_1121.py``.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import ef_fbx_blender as FB  # noqa: E402
from polylode.endfield import fbxfile  # noqa: E402
from polylode.endfield.fbxfile import angle  # noqa: E402
from polylode.gf2.fbx_tangents import export_scene  # noqa: E402
from polylode.gf2.precision import write_rest  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def armature(document, arrays):
    """The armature with every bone's rest rotation the native one (≤ 0.01°) and its head on the native joint."""
    native = arrays["bone_rest"]
    rig = FB._armature(document, arrays)
    goal = float(document.get("bone_display_precision", {}).get("angular_budget_deg", .001))
    maximum_tail_change = 0.
    initial_error = None
    for _ in range(np.finfo(np.float32).nmant + 1):
        bpy.context.view_layer.update()
        errors = []
        for index, record in enumerate(document["bones"]):
            wanted = native[index]
            actual = np.asarray(rig.data.bones[record["name"]].matrix_local)
            errors.append(dict(index=index, name=record["name"], turn_deg=angle(wanted[:3, :3], actual[:3, :3]),
                               offset_mm=float(np.linalg.norm(wanted[:3, 3] - actual[:3, 3]) * 1000), actual=actual))
        worst = max(errors, key=lambda row: row["turn_deg"])
        if initial_error is None:
            initial_error = worst["turn_deg"]
        if worst["turn_deg"] <= goal:
            break
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")
        moved = False
        for row in errors:
            if row["turn_deg"] <= goal:
                continue
            index = row["index"]
            rest = native[index]
            bone = rig.data.edit_bones[row["name"]]
            old = np.array(bone.tail)
            head = np.array(bone.head)
            direction = (old - head) / np.linalg.norm(old - head)
            # In Blender's near-antiparallel parent branch the display-vector
            # tilt can have gain two in the stored rest matrix. Invert that
            # measured mapping with a half-step, changing only a virtual tail.
            direction += .5 * (rest[:3, 1] - row["actual"][:3, 1])
            direction /= np.linalg.norm(direction)
            bone.head = Vector(rest[:3, 3].tolist())
            bone.tail = Vector((rest[:3, 3] + direction * arrays["bone_length"][index]).tolist())
            bone.align_roll(Vector(rest[:3, 2].tolist()))
            moved |= not np.array_equal(old, np.asarray(bone.tail))
            requested = rest[:3, 3] + rest[:3, 1] * arrays["bone_length"][index]
            maximum_tail_change = max(maximum_tail_change, float(np.linalg.norm(np.asarray(bone.tail) - requested)))
        bpy.ops.object.mode_set(mode="OBJECT")
        if not moved:
            break
    bpy.context.view_layer.update()
    # Some near-pole rolls have almost zero inverse gain. Move only their
    # virtual display direction just outside Blender's documented 2.5e-4
    # singularity, with the already-established float32 angular uncertainty.
    escape = []
    critical = 2.5e-4
    for index, record in enumerate(document["bones"]):
        actual = np.asarray(rig.data.bones[record["name"]].matrix_local)
        if angle(native[index, :3, :3], actual[:3, :3]) <= .01:
            continue
        parent = record["parent"]
        parent_rotation = (np.asarray(rig.data.bones[document["bones"][parent]["name"]].matrix_local)[:3, :3]
                           if parent >= 0 else np.eye(3))
        local = np.linalg.solve(parent_rotation, native[index, :3, 1])
        local /= np.linalg.norm(local)
        radius = float(np.linalg.norm(local[[0, 2]]))
        if local[1] < 0 and 0 < radius < critical:
            target_radius = np.nextafter(critical + 2 * np.deg2rad(goal), np.inf)
            local[[0, 2]] *= target_radius / radius
            local[1] = -np.sqrt(1 - target_radius ** 2)
            direction = parent_rotation @ local
            escape.append((index, direction / np.linalg.norm(direction)))
    if escape:
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")
        for index, direction in escape:
            rest = native[index]
            bone = rig.data.edit_bones[document["bones"][index]["name"]]
            bone.head = Vector(rest[:3, 3].tolist())
            bone.tail = Vector((rest[:3, 3] + direction * arrays["bone_length"][index]).tolist())
            bone.align_roll(Vector(rest[:3, 2].tolist()))
            requested = rest[:3, 3] + rest[:3, 1] * arrays["bone_length"][index]
            maximum_tail_change = max(maximum_tail_change, float(np.linalg.norm(np.asarray(bone.tail) - requested)))
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.update()
    final = []
    for index, record in enumerate(document["bones"]):
        wanted = native[index]
        actual = np.asarray(rig.data.bones[record["name"]].matrix_local)
        final.append(dict(name=record["name"], turn_deg=angle(wanted[:3, :3], actual[:3, :3]),
                          offset_mm=float(np.linalg.norm(wanted[:3, 3] - actual[:3, 3]) * 1000)))
    worst = max(final, key=lambda row: row["turn_deg"])
    if worst["turn_deg"] > .01 or max(row["offset_mm"] for row in final) > .01:
        raise ValueError(f"bone rest frames off after storage: {worst}")
    rig["native_rest_turn_error_deg"] = worst["turn_deg"]
    rig["native_rest_worst_bone"] = worst["name"]
    rig["native_rest_initial_error_deg"] = initial_error
    rig["native_virtual_tail_correction_m"] = maximum_tail_change
    rig["native_virtual_tail_pole_escapes"] = len(escape)
    return rig


def build(case, ident, document, arrays):
    FB._empty_scene()
    materials = {}
    for name, record in document["materials"].items():
        material = FB._material(name, record, case)
        material["native_parameters_json"] = json.dumps(record.get("native_material", {}))
        material["display_profile"] = record.get("profile", "unused")
        material["original_game_shader_recreated"] = False
        source = record.get("native_material", {})
        shader = material.node_tree.nodes.get("Principled BSDF")
        if record.get("constant_color") is not None:
            shader.inputs["Base Color"].default_value = record["constant_color"]
            material.diffuse_color = record["constant_color"]
            shader.inputs["Transmission Weight"].default_value = 1
        if record.get("native_surface_unmapped"):
            material["native_surface_unmapped"] = True
            material.diffuse_color = (0, 0, 0, 0)
            material.node_tree.nodes.clear()
            output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
            transparent = material.node_tree.nodes.new("ShaderNodeBsdfTransparent")
            material.node_tree.links.new(transparent.outputs[0], output.inputs["Surface"])
            materials[name] = material
            continue
        images = [node for node in material.node_tree.nodes if node.type == "TEX_IMAGE"]
        if images and source.get("BlendMode") == 1:
            threshold = source["Properties"].get("BasePropertyOverrides", {}).get("OpacityMaskClipValue")
            if threshold is None:
                raise ValueError(f"{name}: a masked material with no cutoff")
            compare = material.node_tree.nodes.new("ShaderNodeMath")
            compare.operation = "GREATER_THAN"
            compare.inputs[1].default_value = threshold
            material.node_tree.links.new(images[0].outputs["Alpha"], compare.inputs[0])
            material.node_tree.links.new(compare.outputs[0], shader.inputs["Alpha"])
            material.surface_render_method = "DITHERED"
        for key, value in record["maps"].items():
            if key == record.get("base"):
                continue
            image = bpy.data.images.load(str(case / value["file"]), check_existing=True)
            image.colorspace_settings.name = "sRGB" if value["srgb"] else "Non-Color"
            node = material.node_tree.nodes.new("ShaderNodeTexImage")
            node.image = image
            node.label = key
            node["native_parameter"] = key
        materials[name] = material
    rig = armature(document, arrays)
    counts = {"degenerate": {}, "repeated": {}}
    for index, part in enumerate(document["parts"]):
        degenerate, repeated = FB._mesh(index, part, document, arrays, rig, materials)
        bpy.data.objects[part["name"]].matrix_parent_inverse = rig.matrix_world.inverted()
        counts["degenerate"][part["name"]] = degenerate
        counts["repeated"][part["name"]] = repeated
    bpy.context.preferences.filepaths.save_version = 0
    bpy.context.scene["wuthering_waves_scope"] = "original clothing and source rig"
    text = bpy.data.texts.new("WutheringWaves_Source.json")
    text.write(json.dumps(document, ensure_ascii=False))
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(case / f"{ident}.blend"))
    export_scene(filepath=str(case / f"{ident}.fbx"), use_selection=False, use_visible=False,
                 object_types={"ARMATURE", "MESH"}, add_leaf_bones=False, bake_anim=False,
                 axis_forward="-Z", axis_up="Y", primary_bone_axis="Y", secondary_bone_axis="X",
                 apply_scale_options="FBX_SCALE_ALL", mesh_smooth_type="OFF", colors_type="LINEAR",
                 path_mode="COPY", embed_textures=True)
    bones = [(bone["name"], bone["parent"], arrays["bone_rest"][index]) for index, bone in enumerate(document["bones"])]
    target = case / f"{ident}.fbx"
    before = fbxfile.bone_errors(target, bones)
    refined = write_rest(target, bones)
    receipt = {"ident": ident, "source_document_sha256": sha(case / "source_document.json"),
               "source_arrays_sha256": sha(case / document["arrays"]), "blend_sha256": sha(case / f"{ident}.blend"),
               "fbx_sha256": sha(target), "bone_basis_before_refine": before, "refined": refined,
               "source_omissions": counts, "screenshots_created": 0}
    (case / "original_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def check(case, ident, document, arrays):
    bones = [(bone["name"], bone["parent"], arrays["bone_rest"][index]) for index, bone in enumerate(document["bones"])]
    basis = fbxfile.bone_errors(case / f"{ident}.fbx", bones)
    if basis["missing"] or basis["bones"] != len(bones):
        raise ValueError(f"FBX bones missing: {basis}")
    if max(basis["offset_mm"], basis["bind_offset_mm"]) > .01 or max(basis["turn_deg"], basis["bind_turn_deg"]) > .01:
        raise ValueError(f"FBX bone frames off: {basis}")
    result = FB.check(document, arrays, case / f"{ident}.fbx")
    if result["bones_missing"] or result["parents_wrong"]:
        raise ValueError("FBX bones missing or under another parent")
    for part in result["parts"]:
        if not part["found"] or part["rest_mm"] > .01 or part["posed_mm"] > .05:
            raise ValueError(f"FBX mesh off: {part}")
    # Compare actual evaluated vertices to the native double-precision frames,
    # rather than only to the imported armature's own possibly rounded frames.
    native_rest = arrays["bone_rest"]
    native_pose = []
    for index, bone in enumerate(document["bones"]):
        parent = bone["parent"]
        turn = np.eye(4)
        turn[:3, :3] = np.asarray(FB._turn(index).to_matrix())
        local = np.linalg.solve(native_rest[parent], native_rest[index]) if parent >= 0 else native_rest[index]
        native_pose.append((native_pose[parent] if parent >= 0 else np.eye(4)) @ local @ turn)
    skin = np.stack(native_pose) @ np.linalg.inv(native_rest)
    expected = {}
    for index, part in enumerate(document["parts"]):
        co = arrays[f"p{index}_co"]
        joints = arrays[f"p{index}_jnt"]
        weights = arrays[f"p{index}_wgt"].astype(float)
        total = weights.sum(1)
        share = weights / np.where(total > 0, total, 1)[:, None]
        points = np.zeros((len(co), 3))
        homogeneous = np.c_[co, np.ones(len(co))]
        for column in range(joints.shape[1]):
            points += share[:, column, None] * np.einsum("nij,nj->ni", skin[joints[:, column], :3], homogeneous)
        points[total <= 0] = co[total <= 0]
        expected[part["name"]] = points

    def native_points():
        rig, = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
        for index, bone in enumerate(document["bones"]):
            pose = rig.pose.bones[bone["name"]]
            pose.matrix_basis.identity()
            pose.rotation_mode = "QUATERNION"
            pose.rotation_quaternion = FB._turn(index)
        for name in expected:
            bpy.data.objects[name].hide_viewport = False
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        errors = {name: float(np.linalg.norm(FB._world_positions(bpy.data.objects[name], graph) - points, axis=1).max()
                              * 1000) for name, points in expected.items()}
        if max(errors.values()) > .05:
            raise ValueError(f"posed vertices off the native frames: {errors}")
        return errors

    imported = native_points()
    bpy.ops.wm.open_mainfile(filepath=str(case / f"{ident}.blend"), load_ui=False)
    rig, = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    turns = [angle(native_rest[index, :3, :3],
                   (np.asarray(rig.matrix_world) @ np.asarray(rig.data.bones[bone["name"]].matrix_local))[:3, :3])
             for index, bone in enumerate(document["bones"])]
    if max(turns) > .01:
        raise ValueError(f"saved .blend bone rest frames off by {max(turns)} deg")
    not_packed = sorted(i.name for i in bpy.data.images if i.source == "FILE" and i.packed_file is None)
    if not_packed:
        raise ValueError(f"images not inside the .blend: {not_packed}")
    saved = native_points()
    validation = {"measurements": result, "native_frame_imported_vertex_errors_mm": imported,
                  "native_frame_saved_blend_vertex_errors_mm": saved, "saved_blend_native_rest_turn_deg": max(turns),
                  "bone_basis": basis, "blend_sha256": sha(case / f"{ident}.blend"),
                  "fbx_sha256": sha(case / f"{ident}.fbx")}
    path = case / "original_validation.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main():
    mode, case, ident = sys.argv[sys.argv.index("--") + 1:][:3]
    case = Path(case)
    document = json.loads((case / "source_document.json").read_text(encoding="utf-8"))
    arrays = np.load(case / document["arrays"])
    if mode == "build":
        build(case, ident, document, arrays)
    elif mode == "check":
        check(case, ident, document, arrays)
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    print(f"{mode.upper()} {ident}", flush=True)


if __name__ == "__main__":
    main()
