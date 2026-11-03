"""Run inside Blender: write one rigged scene as an FBX, or read one back.

    blender -b --factory-startup --python tools/ef_fbx_blender.py -- \
        write <scene.json> <out.fbx> <report.json>
    blender -b --factory-startup --python tools/ef_fbx_blender.py -- \
        check <scene.json> <out.fbx> <report.json>

``scene.json`` and the ``.npz`` it names are ``polylode/endfield/rigscene``'s
(``tools/ef_export_fbx.py`` writes them and starts this script).  The scene is
already in Blender's space: right-handed, Z up, metres.

**write**: one armature whose bones keep the nodes' own axes, one mesh per part
with its weights as vertex groups, its normals, UV sets, vertex colours and
blend shapes, and one material per material name with the base colour map
connected.  Blender cannot hold two faces on the same three vertices, nor a
face that names a vertex twice.  The second kind is left out of the file and
counted (``degenerate``); the first kind is written, and an import drops it
(``repeated`` counts them, so the triangle count read back can be checked
against what an import keeps).

With ``write_blend`` set in the scene, **write** also saves the same scene
as ``<out>.blend`` with its images packed inside, and **blend** opens that
file again and says what it holds and which images are not inside it.

**check**: the file is imported into an empty scene; bone names and parents,
vertex counts and rest positions are compared with the scene it was written
from, and **every bone is turned** and the deformed vertices are compared with
the same blend done here from the scene's own weights — so a wrong vertex
group, a lost weight or a reordered vertex shows as a distance, not as a
picture someone has to look at.  The bones' rest *transforms* are not measured
here: Blender keeps a bone as head, tail and roll, and an import of an exact
file reads up to 0.08° off on a bone near the armature's −Y.  Those are read
from the file's own nodes (``polylode/endfield/fbxfile.py``).
"""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from polylode.gf2.fbx_tangents import export_scene, import_scene

#: the angle every bone is turned by in the read-back pose
TURN = math.radians(25.0)


def _empty_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _render_method(material, method: str) -> None:
    """Blended or dithered surfaces: ``surface_render_method`` from Blender 4.2, ``blend_method`` before it."""
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = method
    else:
        material.blend_method = {"BLENDED": "BLEND", "DITHERED": "HASHED"}[method]


def _material(name: str, record: dict, folder: Path):
    """The material (``_surface``): drawn clear when the writers could not map it (``native_surface_unmapped``),
    and giving off its ``emission`` — a map times a colour times a strength — whatever lights it."""
    if record.get("native_surface_unmapped"):
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
        clear = nodes.new("ShaderNodeBsdfTransparent")
        links.new(clear.outputs["BSDF"], output.inputs["Surface"])
        _render_method(material, "BLENDED")
        return material
    material = _surface(name, record, folder)
    emission = record.get("emission")
    entry = record["maps"].get((emission or {}).get("map") or "")
    shader = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if emission and shader is not None:
        nodes, links = material.node_tree.nodes, material.node_tree.links
        colour = [float(v) for v in emission.get("color", (1.0, 1.0, 1.0))[:3]] + [1.0]
        if entry is not None:
            image = bpy.data.images.load(str(folder / entry["file"]), check_existing=True)
            image.colorspace_settings.name = "sRGB" if entry.get("srgb", True) else "Non-Color"
            node = nodes.new("ShaderNodeTexImage")
            node.image = image
            tinted = nodes.new("ShaderNodeMix")
            tinted.data_type = "RGBA"
            tinted.blend_type = "MULTIPLY"
            tinted.inputs["Factor"].default_value = 1.0
            links.new(node.outputs["Color"], tinted.inputs["A"])
            tinted.inputs["B"].default_value = colour
            links.new(tinted.outputs["Result"], shader.inputs["Emission Color"])
        else:
            shader.inputs["Emission Color"].default_value = colour
        shader.inputs["Emission Strength"].default_value = float(emission.get("strength", 1.0))
    return material


def _surface(name: str, record: dict, folder: Path):
    """The material's base map, and how it covers (the writers' ``tint``, ``alpha_from``, ``blend``, ``cutoff``,
    §7.1455): a multiplied overlay is a Transparent BSDF filtering what is behind it by mix(white, tint,
    coverage) — the game's multiply, which glTF and FBX can only approximate — a blended one the tint by its
    coverage as alpha, a cut-out one its coverage compared with the cutoff."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    base = record["maps"].get(record.get("base") or "")
    if base is None:
        if record.get("blend") in ("add", "multiply"):
            # a layer added or multiplied with nothing to add or multiply by: kept, drawn clear
            output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
            clear = nodes.new("ShaderNodeBsdfTransparent")
            links.new(clear.outputs["BSDF"], output.inputs["Surface"])
            _render_method(material, "BLENDED")
        return material
    image = bpy.data.images.load(str(folder / base["file"]), check_existing=True)
    image.colorspace_settings.name = "sRGB" if base["srgb"] else "Non-Color"
    node = nodes.new("ShaderNodeTexImage")
    node.image = image
    tint = record.get("tint")
    alpha_from = record.get("alpha_from")
    blend = record.get("blend")
    if tint is None and blend is None and record.get("cutoff") is None:
        links.new(node.outputs["Color"], shader.inputs["Base Color"])
        return material
    tint = [float(v) for v in (tint or (1.0, 1.0, 1.0, 1.0))]
    if alpha_from in ("r", "g", "b"):
        channels = nodes.new("ShaderNodeSeparateColor")
        links.new(node.outputs["Color"], channels.inputs["Color"])
        coverage = channels.outputs["RGB".index(alpha_from.upper())]
        colour = None                                  # a coverage mask: the colour is the tint alone
    else:
        coverage, colour = node.outputs["Alpha"], node.outputs["Color"]
    scaled = nodes.new("ShaderNodeMath")
    scaled.operation = "MULTIPLY"
    links.new(coverage, scaled.inputs[0])
    scaled.inputs[1].default_value = tint[3]
    tint_rgba = (*tint[:3], 1.0)
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    layer = None                                   # a colour layer: the base map times the tint
    if colour is not None and blend in ("multiply", "add"):
        layer = nodes.new("ShaderNodeMix")
        layer.data_type = "RGBA"
        layer.blend_type = "MULTIPLY"
        layer.inputs["Factor"].default_value = 1.0
        links.new(colour, layer.inputs["A"])
        layer.inputs["B"].default_value = tint_rgba
    if blend == "multiply":
        # what is behind times mix(white, the layer, its strength); a coverage mask: mix(white, tint, coverage)
        filtered = nodes.new("ShaderNodeMix")
        filtered.data_type = "RGBA"
        filtered.inputs["A"].default_value = (1.0, 1.0, 1.0, 1.0)
        if layer is None:
            links.new(scaled.outputs["Value"], filtered.inputs["Factor"])
            filtered.inputs["B"].default_value = tint_rgba
        else:
            filtered.inputs["Factor"].default_value = float(record.get("strength", 1.0))
            links.new(layer.outputs["Result"], filtered.inputs["B"])
        transparent = nodes.new("ShaderNodeBsdfTransparent")
        links.new(filtered.outputs["Result"], transparent.inputs["Color"])
        links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
        _render_method(material, "BLENDED")
        return material
    if blend == "add":
        # what is behind plus the layer times its strength: an emission over a clear surface (the source's
        # main-light term, 0.2 + 0.8 * N.L, depends on its light and is taken at its full value here)
        glow = nodes.new("ShaderNodeEmission")
        links.new((layer or node).outputs["Result" if layer else "Color"], glow.inputs["Color"])
        glow.inputs["Strength"].default_value = float(record.get("strength", 1.0))
        clear = nodes.new("ShaderNodeBsdfTransparent")
        added = nodes.new("ShaderNodeAddShader")
        links.new(clear.outputs["BSDF"], added.inputs[0])
        links.new(glow.outputs["Emission"], added.inputs[1])
        links.new(added.outputs["Shader"], output.inputs["Surface"])
        _render_method(material, "BLENDED")
        return material
    if colour is None:
        shader.inputs["Base Color"].default_value = tint_rgba
    else:
        tinted = nodes.new("ShaderNodeMix")
        tinted.data_type = "RGBA"
        tinted.blend_type = "MULTIPLY"
        tinted.inputs["Factor"].default_value = 1.0
        links.new(colour, tinted.inputs["A"])
        tinted.inputs["B"].default_value = tint_rgba
        links.new(tinted.outputs["Result"], shader.inputs["Base Color"])
    if blend == "alpha":
        links.new(scaled.outputs["Value"], shader.inputs["Alpha"])
        _render_method(material, "BLENDED")
    elif record.get("cutoff") is not None:
        kept = nodes.new("ShaderNodeMath")
        kept.operation = "GREATER_THAN"           # kept at or above the cutoff: discarded only strictly below it
        links.new(scaled.outputs["Value"], kept.inputs[0])
        kept.inputs[1].default_value = float(record["cutoff"]) - 1e-6
        links.new(kept.outputs["Value"], shader.inputs["Alpha"])
        _render_method(material, "DITHERED")
    return material


def _armature(document: dict, arrays) -> bpy.types.Object:
    data = bpy.data.armatures.new(document["name"])
    armature = bpy.data.objects.new(document["name"], data)
    bpy.context.scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    made = []
    for index, bone in enumerate(document["bones"]):
        edit = data.edit_bones.new(bone["name"])
        if edit.name != bone["name"]:
            raise RuntimeError(f"bone {bone['name']!r} was renamed to {edit.name!r}")
        # the length first: assigning the matrix keeps it and sets head, axis and roll
        edit.head = (0.0, 0.0, 0.0)
        edit.tail = (0.0, float(arrays["bone_length"][index]), 0.0)
        rest = arrays["bone_rest"][index]
        edit.matrix = Matrix(rest.tolist())
        # A bone is kept as head, tail and roll.  The tail is rounded to 32
        # bits, and near the armature's -Y a rounded direction turns the
        # roll's reference by degrees (a 5 mm bone 0.04 deg from -Y came back
        # 0.9 deg off).  So the roll is taken again from the direction as kept.
        edit.align_roll(Vector(rest[:3, 2].tolist()))
        if bone["parent"] >= 0:
            edit.parent = made[bone["parent"]]
            edit.use_connect = False
        made.append(edit)
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def _mesh(index: int, part: dict, document: dict, arrays, armature, materials: dict) -> tuple:
    """Build part *index*; returns ``(degenerate, repeated)`` triangle counts."""
    co = arrays[f"p{index}_co"]
    tri, slot = arrays[f"p{index}_tri"], arrays[f"p{index}_slot"]
    # a triangle that names one vertex twice is not a face Blender can hold
    sound = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    tri, slot = tri[sound], slot[sound]
    mesh = bpy.data.meshes.new(part["name"])
    mesh.vertices.add(len(co))
    mesh.vertices.foreach_set("co", co.ravel())
    mesh.loops.add(tri.size)
    mesh.loops.foreach_set("vertex_index", tri.ravel())
    mesh.polygons.add(len(tri))
    mesh.polygons.foreach_set("loop_start", np.arange(0, tri.size, 3))
    mesh.polygons.foreach_set("loop_total", np.full(len(tri), 3))
    mesh.polygons.foreach_set("material_index", slot)
    mesh.polygons.foreach_set("use_smooth", np.ones(len(tri), dtype=bool))
    mesh.update(calc_edges=True)
    corner = tri.ravel()
    for channel in part["uvs"]:
        layer = mesh.uv_layers.new(name=f"UV{channel}")
        layer.data.foreach_set("uv", arrays[f"p{index}_uv{channel}"][corner].ravel())
    if part["colors"]:
        colors = mesh.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
        colors.data.foreach_set("color", arrays[f"p{index}_col"][corner].ravel())
    mesh.normals_split_custom_set_from_vertices(arrays[f"p{index}_no"].tolist())
    if part.get("tangents", False):
        authored = arrays[f"p{index}_tan"]
        if authored.shape != (len(co), 4) or not np.isfinite(authored).all():
            raise ValueError(f"{part['name']}: invalid authored tangent stream")
        attribute = mesh.attributes.new("SourceTangent", "FLOAT_VECTOR", "POINT")
        attribute.data.foreach_set("vector", authored[:, :3].ravel())
        attribute = mesh.attributes.new("SourceTangentSign", "FLOAT", "POINT")
        attribute.data.foreach_set("value", authored[:, 3].ravel())
        attribute = mesh.attributes.new("SourceNormal", "FLOAT_VECTOR", "POINT")
        attribute.data.foreach_set("vector", arrays[f"p{index}_no"].ravel())
    for name in part["materials"]:
        mesh.materials.append(materials[name] if name else None)

    obj = bpy.data.objects.new(part["name"], mesh)
    bpy.context.scene.collection.objects.link(obj)
    joints, weights = arrays[f"p{index}_jnt"], arrays[f"p{index}_wgt"]
    groups = {}
    for vertex, column in zip(*np.nonzero(weights > 0)):
        bone = int(joints[vertex, column])
        group = groups.get(bone)
        if group is None:
            group = groups[bone] = obj.vertex_groups.new(name=document["bones"][bone]["name"])
        group.add([int(vertex)], float(weights[vertex, column]), "ADD")
    obj.parent = armature
    obj.modifiers.new("Armature", "ARMATURE").object = armature
    if part["shapes"]:
        # The scene carries deltas, not an active expression. Blender 5.2
        # creates new keys at value 1, so set the neutral state explicitly.
        obj.shape_key_add(name="Basis", from_mix=False).value = 0.0
        for k, name in enumerate(part["shapes"]):
            moved = co.copy()
            moved[arrays[f"p{index}_s{k}_idx"]] += arrays[f"p{index}_s{k}_delta"]
            key = obj.shape_key_add(name=name, from_mix=False)
            key.value = 0.0
            key.data.foreach_set("co", moved.ravel())
    # FBX carries the viewport flag as the node's ``Visibility``
    obj.hide_viewport = obj.hide_render = bool(part["hidden"])
    distinct = len(np.unique(np.sort(tri, axis=1), axis=0)) if len(tri) else 0
    return int((~sound).sum()), len(tri) - distinct


def write(document: dict, arrays, folder: Path, target: Path) -> dict:
    """Build the scene and export it.

    Returns ``{"degenerate": …, "repeated": …}``, each part name → count: the
    triangles left out, and the ones written that share their three vertices
    with an earlier triangle.
    """
    _empty_scene()
    materials = {name: _material(name, record, folder)
                 for name, record in document["materials"].items()}
    armature = _armature(document, arrays)
    counts = {"degenerate": {}, "repeated": {}}
    for index, part in enumerate(document["parts"]):
        degenerate, repeated = _mesh(index, part, document, arrays, armature, materials)
        if degenerate:
            counts["degenerate"][part["name"]] = degenerate
        if repeated:
            counts["repeated"][part["name"]] = repeated
    target.parent.mkdir(parents=True, exist_ok=True)
    export_scene(
        filepath=str(target), use_selection=False, use_visible=False,
        object_types={"ARMATURE", "MESH"}, add_leaf_bones=False, bake_anim=False,
        axis_forward="-Z", axis_up="Y",
        # Y along the bone, X beside it: Blender's own, so the axes are written as set
        primary_bone_axis="Y", secondary_bone_axis="X",
        apply_scale_options="FBX_SCALE_ALL", mesh_smooth_type="OFF",
        colors_type="LINEAR", path_mode="COPY", embed_textures=True)
    if document.get("write_blend"):
        # the same scene as a Blender file, its images inside it (read back by ``blend``)
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(target.with_suffix(".blend")), compress=True)
    return counts


def blend(target: Path) -> dict:
    """Open the ``.blend`` written beside *target*: what it holds, and the images not inside it."""
    bpy.ops.wm.open_mainfile(filepath=str(target.with_suffix(".blend")))
    meshes = {o.name: {"vertices": len(o.data.vertices), "polygons": len(o.data.polygons),
                       "armature": any(m.type == "ARMATURE" and m.object for m in o.modifiers)}
              for o in bpy.data.objects if o.type == "MESH"}
    return {"armatures": sum(o.type == "ARMATURE" for o in bpy.data.objects), "meshes": meshes,
            "images": len([i for i in bpy.data.images if i.source == "FILE"]),
            "not_packed": sorted(i.name for i in bpy.data.images if i.source == "FILE" and i.packed_file is None)}


def _turn(bone_index: int) -> Quaternion:
    """A different axis per bone, the same angle: a pose no symmetry hides."""
    axis = ((1.0, 0.3, 0.2), (0.2, 1.0, 0.3), (0.3, 0.2, 1.0))[bone_index % 3]
    return Quaternion(axis, TURN if bone_index % 2 else -TURN)


def _world_positions(obj, depsgraph) -> np.ndarray:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    out = np.zeros(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", out)
    evaluated.to_mesh_clear()
    world = np.array(evaluated.matrix_world)
    return out.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]


def check(document: dict, arrays, target: Path, importer=None) -> dict:
    """Import *target* (FBX, or with *importer* any format it reads) into an empty scene and measure it."""
    _empty_scene()
    (importer or import_scene)(filepath=str(target))
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    meshes = {o.name: o for o in bpy.data.objects if o.type == "MESH"}
    report = {"armatures": len(armatures), "meshes": len(meshes), "parts": [],
              "bones": 0, "bones_missing": [], "images": len(bpy.data.images)}
    if len(armatures) != 1:
        return report
    armature = armatures[0]
    world = np.array(armature.matrix_world)
    names = [bone["name"] for bone in document["bones"]]
    report["bones"] = len(armature.data.bones)
    report["bones_missing"] = [n for n in names if n not in armature.data.bones][:5]
    rest, parents_wrong = {}, 0
    for index, name in enumerate(names):
        bone = armature.data.bones.get(name)
        if bone is None:
            continue
        rest[index] = world @ np.array(bone.matrix_local)
        parent = document["bones"][index]["parent"]
        above = bone.parent.name if bone.parent else None
        parents_wrong += above != (names[parent] if parent >= 0 else None)
    report["parents_wrong"] = parents_wrong
    # where each bone's head is against the scene's own joint (an importer may turn a bone's axes for display)
    heads = [np.linalg.norm(rest[i][:3, 3] - np.asarray(arrays["bone_rest"][i])[:3, 3]) for i in rest]
    report["bone_head_mm"] = float(max(heads, default=0.0)) * 1000.0

    for index, name in enumerate(names):
        pose = armature.pose.bones.get(name)
        if pose is not None:
            pose.rotation_mode = "QUATERNION"
            pose.rotation_quaternion = _turn(index)
    hidden = {name for name, obj in meshes.items() if obj.hide_viewport}
    for obj in meshes.values():
        obj.hide_viewport = False       # a hidden object is not evaluated
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    skin = {index: world @ np.array(armature.pose.bones[names[index]].matrix)
            @ np.linalg.inv(matrix) for index, matrix in rest.items()}
    skin_stack = np.stack([skin.get(i, np.eye(4)) for i in range(len(names))])
    for index, part in enumerate(document["parts"]):
        row = {"name": part["name"], "found": part["name"] in meshes}
        report["parts"].append(row)
        if not row["found"]:
            continue
        obj = meshes[part["name"]]
        row["hidden"] = part["name"] in hidden
        co = arrays[f"p{index}_co"].astype(np.float64)
        joints, weights = arrays[f"p{index}_jnt"], arrays[f"p{index}_wgt"].astype(np.float64)
        mesh = obj.data
        row.update(vertices=len(mesh.vertices), triangles=len(mesh.polygons),
                   uv_layers=len(mesh.uv_layers), colors=len(mesh.color_attributes),
                   materials=[m.name if m else None for m in mesh.materials],
                   groups=len(obj.vertex_groups),
                   shape_keys=len(mesh.shape_keys.key_blocks) - 1 if mesh.shape_keys else 0,
                   armature=any(m.type == "ARMATURE" and m.object == armature
                                for m in obj.modifiers))
        if len(mesh.vertices) != len(co):
            continue
        # the pose first (the object is posed now), then the rest
        posed = _world_positions(obj, depsgraph)
        total = weights.sum(axis=1, keepdims=True)
        share = weights / np.where(total > 0, total, 1.0)
        blend = np.einsum("vk,vkij->vij", share, skin_stack[joints])
        blend[total[:, 0] <= 0] = np.eye(4)
        expect = np.einsum("vij,vj->vi", blend, np.c_[co, np.ones(len(co))])[:, :3]
        row["posed_mm"] = float(np.linalg.norm(posed - expect, axis=1).max()) * 1000.0
        row["moved_mm"] = float(np.linalg.norm(expect - co, axis=1).max()) * 1000.0
        raw = np.zeros(len(co) * 3)
        mesh.vertices.foreach_get("co", raw)
        own = np.array(obj.matrix_world)
        at_rest = raw.reshape(-1, 3) @ own[:3, :3].T + own[:3, 3]
        row["rest_mm"] = float(np.linalg.norm(at_rest - co, axis=1).max()) * 1000.0
        counted = np.zeros(len(co), dtype=bool)
        for vertex in mesh.vertices:
            counted[vertex.index] = len(vertex.groups) > 0
        row["weighted"] = int(counted.sum())
        row["weighted_expected"] = int((total[:, 0] > 0).sum())
    return report


def main() -> None:
    mode, *paths = sys.argv[sys.argv.index("--") + 1:][:4]
    scene_path, target, report_path = (Path(p) for p in paths)
    document = json.loads(scene_path.read_text(encoding="utf-8"))
    arrays = np.load(scene_path.parent / document["arrays"])
    if mode == "write":
        report = write(document, arrays, scene_path.parent, target)
    elif mode == "check":
        report = check(document, arrays, target)
    elif mode == "blend":
        report = blend(target)
    elif mode == "glb":
        report = check(document, arrays, target, importer=lambda filepath: bpy.ops.import_scene.gltf(filepath=filepath))
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{mode.upper()} {target}")


if __name__ == "__main__":
    main()
