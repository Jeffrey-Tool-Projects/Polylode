"""The glTF binary writer (``interchange/glb.py``), read back by a reader written here from the spec."""
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from polylode.interchange import glb

PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
       b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82")


def test_native_lighting_maps_are_portable_and_gf2_multipliers_are_not_material_values(tmp_path, monkeypatch):
    from polylode.extractor.character_styles import descriptors
    monkeypatch.setenv("ISLANDLIFE_DERIVED_ROOT", str(tmp_path))
    document, arrays = _scene(tmp_path)
    skin = document["materials"]["skin"]
    skin.update(shader="gf_shader/pbr/character/uber",
                floats={"_UseRampMap": 1, "_Metallic": 1, "_Smoothness": 1})
    skin["maps"]["_RampMap"] = {"file": "textures/base.png", "srgb": False}
    path = tmp_path / "portable.glb"
    glb.write(document, arrays, tmp_path, path)
    (tmp_path / "textures/base.png").unlink()
    parsed, _, _ = _read(path)
    assert "texture" in parsed["materials"][0]["extras"]["maps"]["_RampMap"]
    desc = descriptors(path)[0][0]
    from PySide6.QtCore import QUrl
    assert Path(QUrl(desc["ramp"]).toLocalFile()).read_bytes() == PNG
    assert desc["metallic"] == 0 and desc["smoothness"] == .5


def test_wuwa_shading_keeps_authored_zero_values(tmp_path, monkeypatch):
    from polylode.extractor.character_styles import descriptors
    monkeypatch.setenv("ISLANDLIFE_DERIVED_ROOT", str(tmp_path))
    document, arrays = _scene(tmp_path)
    document["materials"]["skin"]["native_material"] = {"Scalars": {
        "ShadowProcess": 0, "FrontShadowProcessOffset": 0, "PBRRateIntensity": 0}}
    path = tmp_path / "wuwa.glb"
    glb.write(document, arrays, tmp_path, path)
    desc = descriptors(path)[0][0]
    assert desc["shadowProcess"] == desc["frontShadowOffset"] == desc["pbrRate"] == 0


def _read(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data)
    assert (magic, version, length) == (0x46546C67, 2, len(data))
    size, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A
    gltf = json.loads(data[20:20 + size])
    blob_size, blob_kind = struct.unpack_from("<II", data, 20 + size)
    assert blob_kind == 0x004E4942
    blob = data[28 + size:28 + size + blob_size]

    def view(index):
        record = gltf["bufferViews"][index]
        return blob[record["byteOffset"]:record["byteOffset"] + record["byteLength"]]

    def accessor(index):
        record = gltf["accessors"][index]
        dtype = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}[record["componentType"]]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[record["type"]]
        values = (np.frombuffer(view(record["bufferView"]), dtype=dtype).reshape(-1, width).copy()
                  if "bufferView" in record else np.zeros((record["count"], width), dtype))
        sparse = record.get("sparse")
        if sparse:
            at = np.frombuffer(view(sparse["indices"]["bufferView"]), dtype=np.uint32)
            values[at] = np.frombuffer(view(sparse["values"]["bufferView"]), dtype=np.float32).reshape(-1, width)
        return values
    return gltf, accessor, view


def _scene(tmp_path):
    (tmp_path / "textures").mkdir(exist_ok=True)
    (tmp_path / "textures" / "base.png").write_bytes(PNG)
    rest = np.stack([np.eye(4), np.eye(4)])
    rest[1, :3, 3] = (0.0, 0.2, 1.0)                                  # 1 m up, 0.2 m back (Z up)
    rest[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]              # turned 90° about Z
    arrays = {"bone_rest": rest,
              "p0_co": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1], [0, 0, 2]], np.float32),
              "p0_no": np.array([[0, 0, 1]] * 4, np.float32),
              "p0_tan": np.array([[1, 0, 0, -1]] * 4, np.float32),
              "p0_tri": np.array([[0, 1, 2], [0, 2, 3], [1, 1, 3]]),
              "p0_slot": np.array([0, 1, 1]),
              "p0_jnt": np.array([[0, 1], [1, 0], [1, 0], [0, 0]]),
              "p0_wgt": np.array([[.25, .75], [1, 0], [2, 2], [0, 0]], np.float32),
              "p0_uv0": np.array([[0, 0], [1, 0], [0, 1], [.5, .25]], np.float32),
              "p0_s0_idx": np.array([3, 1]), "p0_s0_delta": np.array([[0, 0, .5], [0, .1, 0]], np.float32)}
    materials = {"skin": {"maps": {"_MainTex": {"file": "textures/base.png", "srgb": True, "scale": [2, 1],
                                                "offset": [0, .5]},
                                   "_Normal": {"file": "textures/base.png", "srgb": False}},
                          "base": "_MainTex", "floats": {"_Cutoff": .5}},
                 "cloth": {"maps": {}, "base": None}}
    document = {"name": "Rig", "bones": [{"name": "root", "parent": -1}, {"name": "spine", "parent": 0}],
                "humanoid": {"Hips": "root"}, "materials": materials, "arrays": "x.npz",
                "parts": [{"name": "body", "materials": ["skin", "cloth"], "uvs": [0], "colors": False,
                           "tangents": True, "shapes": ["smile"], "hidden": False}]}
    return document, arrays


def test_expression_weights_survive_hidden_parts_material_splitting_and_the_shared_first_frame(tmp_path):
    import copy
    from polylode.interchange.split import by_material
    from polylode.extractor.share import posed
    document, arrays = _scene(tmp_path)
    document["parts"][0]["shapes"] = ["smile", "blink"]
    arrays["p0_s1_idx"] = np.array([3])
    arrays["p0_s1_delta"] = np.array([[0, 0, .2]], np.float32)
    hidden = copy.deepcopy(document["parts"][0])
    hidden.update(name="hidden", hidden=True)
    document["parts"].insert(0, hidden)
    for key in list(arrays):
        if key.startswith("p0_"):
            arrays["p1_" + key[3:]] = arrays[key].copy()
    document["animations"] = [{"name": "face", "rate": 30, "frames": 3}]
    arrays["anim0"] = np.repeat(arrays["bone_rest"][None], 3, axis=0)
    arrays["p1_anim0_weights"] = np.array([[.4, .2], [.7, .2], [1, .2]])
    document, arrays = by_material(document, arrays)
    target = tmp_path / "expressions.glb"
    glb.write(document, arrays, tmp_path, target)
    gltf, accessor, _ = _read(target)
    animation = gltf["animations"][0]
    assert len(gltf["meshes"]) == 2
    for channel in animation["channels"]:
        assert channel["target"]["path"] == "weights"
        node = gltf["nodes"][channel["target"]["node"]]
        names = gltf["meshes"][node["mesh"]]["extras"]["targetNames"]
        values = accessor(animation["samplers"][channel["sampler"]]["output"]).reshape(-1, len(names))
        np.testing.assert_allclose(values[:, names.index("smile")], [.4, 1])
        if "blink" in names:
            np.testing.assert_allclose(values[:, names.index("blink")], [.2, .2])
    held, _, _ = _read(posed(target, tmp_path / "held.glb"))
    for node in held["nodes"]:
        if "mesh" in node:
            names = held["meshes"][node["mesh"]]["extras"]["targetNames"]
            np.testing.assert_allclose(node["weights"], [.4 if name == "smile" else .2 for name in names])


def test_source_shape_names_are_selectable_expression_animations(tmp_path):
    from polylode.interchange.expressions import add_shapes
    document, arrays = _scene(tmp_path)
    assert add_shapes(document, arrays) == 1
    glb.write(document, arrays, tmp_path, tmp_path / "expressions.glb")
    gltf, accessor, _ = _read(tmp_path / "expressions.glb")
    animation = gltf["animations"][0]
    assert animation["name"] == "expression/smile"
    channel = next(c for c in animation["channels"] if c["target"]["path"] == "weights")
    np.testing.assert_allclose(accessor(animation["samplers"][channel["sampler"]]["output"]), 1)


def test_expression_sidecar_preserves_unbound_scalar_samples_and_authored_controls(tmp_path):
    from polylode.interchange.expressions import write_sidecar
    source = {"names": ["smile", "UI_Eyes_Control"], "rate": 30,
              "values": [[.1, .3], [1, 0]], "unbound": ["UI_Eyes_Control"]}
    document = {"source": "game/model", "animations": [
        {"name": "idle", "clip": "ui_idle", "rate": 60, "frames": 2, "source_curves": source}],
        "expression_controls": [{"kind": 0, "params": {"blush": ["_Blush", 0]}}]}
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(document))
    result = write_sidecar(path, tmp_path / "model.expressions.json")
    back = json.loads(result.read_text(encoding="utf-8"))
    assert back["animations"][0]["source_curves"] == source
    assert back["expression_controls"] == document["expression_controls"]


def test_the_file_is_the_scene_in_y_up_with_its_skin_shapes_and_base_map(tmp_path):
    document, arrays = _scene(tmp_path)
    report = glb.write(document, arrays, tmp_path, tmp_path / "out.glb")
    gltf, accessor, view = _read(tmp_path / "out.glb")
    assert report["degenerate"] == {"body": 1} and report["unweighted"] == {"body": 1}

    nodes = gltf["nodes"]
    root, bone_root, bone_spine = nodes[0], nodes[1], nodes[2]
    assert root["name"] == "Rig" and root["children"] == [1] and bone_root["children"] == [2]
    world_spine = (np.array(bone_root["matrix"]).reshape(4, 4).T @ np.array(bone_spine["matrix"]).reshape(4, 4).T)
    assert np.allclose(world_spine, glb.TO_Y_UP @ arrays["bone_rest"][1])
    assert np.allclose(world_spine[:3, 3], [0.0, 1.0, -0.2])          # Z up → Y up
    local_spine = np.array(bone_spine["matrix"]).reshape(4, 4).T
    assert np.allclose(local_spine, np.linalg.inv(arrays["bone_rest"][0]) @ arrays["bone_rest"][1])  # native local
    assert gltf["skins"][0]["skeleton"] == 0                           # the rig node, every bone's common root
    inverse = accessor(gltf["skins"][0]["inverseBindMatrices"])[1].reshape(4, 4).T
    assert np.allclose(inverse @ world_spine, np.eye(4), atol=1e-6)

    mesh = gltf["meshes"][0]
    attributes = mesh["primitives"][0]["attributes"]
    assert np.allclose(accessor(attributes["POSITION"])[2], [0, 1, -1])
    assert np.allclose(accessor(attributes["TEXCOORD_0"])[3], [.5, .75])  # v flipped
    assert np.allclose(accessor(attributes["TANGENT"])[0], [1, 0, 0, -1])
    weights = accessor(attributes["WEIGHTS_0"])
    assert np.allclose(weights[0], [.25, .75, 0, 0]) and np.allclose(weights[2], [.5, .5, 0, 0])
    assert np.allclose(weights[3], 0)                                    # held by no bone: stays
    assert [accessor(p["indices"]).ravel().tolist() for p in mesh["primitives"]] == [[0, 1, 2], [0, 2, 3]]
    assert [p.get("material") for p in mesh["primitives"]] == [0, 1]
    delta = accessor(mesh["primitives"][0]["targets"][0]["POSITION"])
    assert np.allclose(delta[3], [0, .5, 0]) and np.allclose(delta[1], [0, 0, -.1]) and np.allclose(delta[0], 0)
    assert mesh["extras"]["targetNames"] == ["smile"] and mesh["weights"] == [0.0]

    skin = gltf["materials"][0]
    texture = skin["pbrMetallicRoughness"]["baseColorTexture"]
    image = gltf["images"][gltf["textures"][texture["index"]]["source"]]
    assert view(image["bufferView"]) == PNG and image["mimeType"] == "image/png"
    assert texture["extensions"]["KHR_texture_transform"] == {"scale": [2.0, 1.0], "offset": [0.0, -0.5]}
    assert skin["extras"]["maps"]["_Normal"]["file"] == "textures/base.png"   # kept by name, not reinterpreted
    assert "baseColorTexture" not in gltf["materials"][1]["pbrMetallicRoughness"]
    body = next(n for n in nodes if n.get("mesh") == 0)
    assert body["skin"] == 0 and "extensions" not in body


def test_a_masked_material_without_a_cutoff_and_a_bone_before_its_parent_are_refused(tmp_path):
    document, arrays = _scene(tmp_path)
    document["materials"]["skin"]["native_material"] = {"BlendMode": 1, "Properties": {}}
    with pytest.raises(glb.GlbError, match="cutoff"):
        glb.write(document, arrays, tmp_path, tmp_path / "out.glb")
    document, arrays = _scene(tmp_path)
    document["bones"][0]["parent"] = 1
    with pytest.raises(glb.GlbError, match="before its parent"):
        glb.write(document, arrays, tmp_path, tmp_path / "out.glb")


def test_the_readback_judges_bones_vertices_weights_and_shapes():
    document = {"bones": [{"name": "root", "parent": -1}],
                "parts": [{"name": "body", "shapes": ["smile"]}]}
    arrays = {"p0_co": np.zeros((4, 3))}
    good = {"armatures": 1, "bones": 1, "bones_missing": [], "parents_wrong": 0, "bone_head_mm": 0.001,
            "parts": [{"found": True, "vertices": 4, "armature": True, "weighted": 4, "weighted_expected": 4,
                       "rest_mm": .001, "posed_mm": .001, "shape_keys": 1}]}
    assert glb.readback_problems(document, arrays, good) == []
    assert "heads" in glb.readback_problems(document, arrays, dict(good, bone_head_mm=.5))[0]
    worse = dict(good, parts=[dict(good["parts"][0], posed_mm=.2)])
    assert "posed off" in glb.readback_problems(document, arrays, worse)[0]
    assert "shapes" in glb.readback_problems(document, arrays, dict(good, parts=[dict(good["parts"][0], shape_keys=0)]))[0]


def test_an_overlay_is_its_tint_laid_on_by_its_maps_red_channel(tmp_path):
    """§7.1455: an eye shadow's map is a coverage mask, not a colour; drawn opaque it painted the mask itself."""
    import io
    from PIL import Image
    document, arrays = _scene(tmp_path)
    document["materials"]["cloth"] = {"maps": {"_BaseMap": {"file": "textures/base.png", "srgb": False}},
                                      "base": "_BaseMap", "tint": [0.8, 0.6, 0.7, 1.0], "alpha_from": "r",
                                      "blend": "multiply"}
    glb.write(document, arrays, tmp_path, tmp_path / "out.glb")
    gltf, _accessor, view = _read(tmp_path / "out.glb")
    cloth = next(m for m in gltf["materials"] if m["name"] == "cloth")
    assert cloth["alphaMode"] == "BLEND"
    factor = cloth["pbrMetallicRoughness"]["baseColorFactor"]
    assert factor[3] == 1.0 and abs(factor[0] - ((0.8 + 0.055) / 1.055) ** 2.4) < 1e-6          # sRGB → linear
    image = gltf["images"][gltf["textures"][cloth["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]]
    pixels = Image.open(io.BytesIO(view(image["bufferView"]))).convert("RGBA").getpixel((0, 0))
    red = Image.open(tmp_path / "textures" / "base.png").convert("RGB").getpixel((0, 0))[0]
    assert pixels == (255, 255, 255, red)
    skin = next(m for m in gltf["materials"] if m["name"] == "skin")
    assert "alphaMode" not in skin and skin["pbrMetallicRoughness"]["baseColorTexture"]["index"] != \
        cloth["pbrMetallicRoughness"]["baseColorTexture"]["index"]                    # the colour map stays its own


def test_gf2_coverage_is_its_verified_surface_rule_and_other_shaders_say_they_are_not():
    """§7.1455: ubertrans blends, an alpha-tested uber is cut at _Cutoff; eye blends are not guessed from names."""
    from polylode.gf2 import presentation
    from polylode.gf2.export import coverage_fields
    colour = {"_BaseColor": [1.0, 0.5, 0.5, 0.8]}
    cut = {"shader": presentation.UBER, "keywords": ["_ALPHATEST_ON"], "floats": {"_Cutoff": 0.4}, "colors": colour}
    coverage_fields(cut)
    assert cut["cutoff"] == 0.4 and cut["tint"] == [1.0, 1.0, 1.0, 0.8] and "blend" not in cut
    glass = {"shader": presentation.TRANS, "keywords": [], "colors": colour,
             "floats": {"_BlendMode": 5, "_PreMulAlpha": 0}}
    coverage_fields(glass)
    assert glass["blend"] == "alpha" and glass["alpha_from"] == "a"
    shade = {"shader": "gf_shader/pbr/character/eyeblend_multiply", "keywords": [],
             "floats": {"_SpecularIntensity": 0.437}, "colors": {"_MainColor": [0.9, 0.8, 0.9, 0.0]}}
    coverage_fields(shade)                                  # read from the shader's render state and fragment
    assert shade["blend"] == "multiply" and shade["strength"] == 0.437 and shade["tint"] == [0.9, 0.8, 0.9, 1.0]
    glint = {"shader": "gf_shader/pbr/character/eyeblend_add", "keywords": [], "floats": {}, "colors": {}}
    coverage_fields(glint)
    assert glint["blend"] == "add" and "alpha_from" not in glint
    eye = {"shader": "gf_shader/pbr/character/eye", "keywords": [], "floats": {}, "colors": {}}
    coverage_fields(eye)
    assert "blend" not in eye and "opaque" in eye["coverage"]                    # Blend One Zero, ZWrite on
    effect = {"shader": "gf_shader/vfx/gf_sfx_holographic3", "keywords": [], "floats": {}, "colors": {}}
    coverage_fields(effect)
    assert "blend" not in effect and "not verified" in effect["coverage"]



def test_a_colour_layer_multiplied_or_added_onto_what_is_behind_is_kept_but_drawn_invisible(tmp_path):
    """§7.1455: glTF cannot multiply or add a colour layer (GF2's eye blends); opaque it was a solid square."""
    document, arrays = _scene(tmp_path)
    document["materials"]["cloth"] = {"maps": {"_MainTex": {"file": "textures/base.png", "srgb": True}},
                                      "base": "_MainTex", "tint": [1.0, 1.0, 1.0, 1.0], "strength": 0.5,
                                      "blend": "add"}
    glb.write(document, arrays, tmp_path, tmp_path / "out.glb")
    gltf, _accessor, _view = _read(tmp_path / "out.glb")
    cloth = next(m for m in gltf["materials"] if m["name"] == "cloth")
    assert cloth["alphaMode"] == "BLEND" and cloth["pbrMetallicRoughness"]["baseColorFactor"][3] == 0.0
    assert cloth["extras"]["blend"] == "add" and cloth["extras"]["strength"] == 0.5


def test_a_sources_vertex_colours_are_kept_as_data_and_never_multiply_the_base_colour(tmp_path):
    # GF2 keeps outline normals in its vertex colours, about (0.5, 0.5, 1): as COLOR_0 every viewer would
    # multiply them into the base colour, and Qt's loader drew the characters not at all
    document, arrays = _scene(tmp_path)
    document["parts"][0]["colors"] = True
    arrays["p0_col"] = np.array([[.5, .5, 1, 1]] * 4, np.float32)
    glb.write(document, arrays, tmp_path, tmp_path / "out.glb")
    gltf, accessor, _view = _read(tmp_path / "out.glb")
    attributes = gltf["meshes"][0]["primitives"][0]["attributes"]
    assert "COLOR_0" not in attributes and glb.VERTEX_DATA.startswith("_")
    assert np.allclose(accessor(attributes[glb.VERTEX_DATA]), [.5, .5, 1, 1])


def test_a_part_of_several_materials_splits_into_a_part_per_section_with_its_vertices_and_shapes():
    import pytest
    from polylode.interchange.split import by_material, section_name
    document = {"parts": [{"name": "Rig", "materials": ["MI_RigHair", "MI_RigFace", "MI_Unused"], "uvs": [0],
                           "shapes": ["blink", "hair_only"], "hidden": False}], "materials": {}}
    arrays = {"bone_rest": np.eye(4)[None],
              "p0_co": np.arange(18, dtype=np.float32).reshape(6, 3), "p0_uv0": np.arange(12.0).reshape(6, 2),
              "p0_jnt": np.zeros((6, 4), int), "p0_wgt": np.ones((6, 4)),
              "p0_tri": np.array([[0, 1, 2], [3, 4, 5], [2, 1, 0]]), "p0_slot": np.array([0, 1, 0]),
              "p0_s0_idx": np.array([4, 5]), "p0_s0_delta": np.ones((2, 3)),        # the face's vertices
              "p0_s1_idx": np.array([1]), "p0_s1_delta": np.full((1, 3), 2.0)}      # the hair's
    split, out = by_material(document, arrays)
    assert [p["name"] for p in split["parts"]] == ["Rig_Hair", "Rig_Face"]
    assert [p["materials"] for p in split["parts"]] == [["MI_RigHair"], ["MI_RigFace"]]
    assert out["p0_tri"].tolist() == [[0, 1, 2], [2, 1, 0]] and out["p1_tri"].tolist() == [[0, 1, 2]]
    assert np.array_equal(out["p1_co"], arrays["p0_co"][3:]) and np.array_equal(out["p1_uv0"], arrays["p0_uv0"][3:])
    assert split["parts"][0]["shapes"] == ["hair_only"] and out["p0_s0_idx"].tolist() == [1]
    assert split["parts"][1]["shapes"] == ["blink"] and out["p1_s0_idx"].tolist() == [1, 2]
    assert out["bone_rest"] is arrays["bone_rest"]                       # not a part's: carried as it is
    assert by_material({"parts": [document["parts"][0]]}, {**arrays, "p0_slot": np.zeros(3, int)})[0]["parts"][0]["name"] == "Rig"
    with pytest.raises(ValueError):
        by_material(document, {**arrays, "p0_mystery": np.zeros(6)})   # a stream it cannot split: refused
    assert section_name("R2T1ChunMd10011", "MI_R2T1ChunMd1001Eye") == "R2T1ChunMd10011_Eye"
    assert section_name("R2T1ChunMd10011", "MI_5XingStar") == "R2T1ChunMd10011_5XingStar"


def test_a_baked_effect_is_its_coverage_with_no_colour_and_its_colour_given_off(tmp_path):
    """An effect (vfx_bake): the base map's alpha its coverage, black its own colour, the map its emission —
    over 1 as KHR_materials_emissive_strength."""
    document, arrays = _scene(tmp_path)
    document["materials"]["cloth"] = {"maps": {"_Baked": {"file": "textures/base.png", "srgb": True}},
                                      "base": "_Baked", "tint": [0.0, 0.0, 0.0, 1.0], "blend": "alpha",
                                      "alpha_from": "a", "emission": {"map": "_Baked", "color": [1, 1, 1],
                                                                      "strength": 2.5}}
    glb.write(document, arrays, tmp_path, tmp_path / "out.glb")
    gltf, _accessor, _view = _read(tmp_path / "out.glb")
    cloth = next(m for m in gltf["materials"] if m["name"] == "cloth")
    assert cloth["alphaMode"] == "BLEND" and cloth["pbrMetallicRoughness"]["baseColorFactor"][:3] == [0, 0, 0]
    assert cloth["emissiveTexture"]["index"] == cloth["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    assert cloth["emissiveFactor"] == [1, 1, 1]
    assert cloth["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"] == 2.5
    assert "KHR_materials_emissive_strength" in gltf["extensionsUsed"]


def test_a_part_hidden_at_rest_is_left_out_of_the_glb_with_the_materials_only_it_used(tmp_path):
    """Blender (5.2) ignores KHR_node_visibility: a hidden LOD level or outfit version was drawn over the shown one."""
    document, arrays = _scene(tmp_path)
    document["parts"].insert(0, dict(document["parts"][0], name="body_lod1", materials=["lod"], hidden=True))
    document["materials"]["lod"] = {"maps": {}, "base": None}
    arrays = {(f"p{int(k[1]) + 1}_{k.split('_', 1)[1]}" if k.startswith("p0_") else k): v for k, v in arrays.items()}
    arrays.update({k.replace("p1_", "p0_", 1): v for k, v in arrays.items() if k.startswith("p1_")})
    glb.write(document, arrays, tmp_path, tmp_path / "out.glb")
    gltf, accessor, _view = _read(tmp_path / "out.glb")
    assert [m["name"] for m in gltf["meshes"]] == ["body"] and "lod" not in [m["name"] for m in gltf["materials"]]
    assert "KHR_node_visibility" not in gltf.get("extensionsUsed", [])
    position = gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
    assert len(accessor(position)) == 4                                # the shown part's own vertices


def _turn(angle):
    """A rotation of *angle* radians about Z, as a 4×4 (the scene is Z up)."""
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    return m


def test_a_scenes_poses_are_animations_of_its_bones_and_the_bones_are_written_as_trs(tmp_path):
    document, arrays = _scene(tmp_path)
    rest = arrays["bone_rest"]
    # two frames: the spine turned 30° then 60° about its parent's Z; the root holds its rest
    frames = np.stack([np.stack([rest[0], rest[0] @ np.linalg.solve(rest[0], rest[1]) @ _turn(a)])
                       for a in (np.pi / 6, np.pi / 3)]).astype(np.float32)
    arrays["anim0"] = frames
    document["animations"] = [{"name": "relax", "rate": 30.0, "frames": 2}]
    glb.write(document, arrays, tmp_path, tmp_path / "posed.glb")
    gltf, accessor, _view = _read(tmp_path / "posed.glb")
    spine = gltf["nodes"][2]
    assert "matrix" not in spine and {"translation", "rotation"} <= set(spine)   # glTF animates only TRS nodes
    rest_local = np.linalg.solve(glb.TO_Y_UP @ rest[0], glb.TO_Y_UP @ rest[1])
    assert np.allclose(spine["rotation"], glb._quaternions(rest_local[:3, :3]))
    (animation,) = gltf["animations"]
    assert animation["name"] == "relax"
    channels = {(c["target"]["node"], c["target"]["path"]): animation["samplers"][c["sampler"]]
                for c in animation["channels"]}
    assert set(channels) == {(2, "rotation")}                       # only what the pose moves off its rest
    sampler = channels[(2, "rotation")]
    assert np.allclose(accessor(sampler["input"]).ravel(), [0, 1 / 30])
    posed = glb.TO_Y_UP @ frames[1].astype(np.float64)
    expected = glb._quaternions(np.linalg.solve(posed[0], posed[1])[:3, :3])
    assert np.allclose(accessor(sampler["output"])[1], expected, atol=1e-6)
    # without poses the bones stay matrices
    del document["animations"]
    glb.write(document, arrays, tmp_path, tmp_path / "still.glb")
    still, _a, _v = _read(tmp_path / "still.glb")
    assert "matrix" in still["nodes"][2] and "animations" not in still


def test_a_pose_that_grows_a_bone_writes_its_scale_and_its_children_keep_their_rest_offsets(tmp_path):
    """GF2's CommandCenterBackGround sets Clukay SSR01's root to 1.202: with the scale left out, every bone below
    stood 1.2 times further from its parent on a skin of its rest size, the eyes up off the face."""
    from polylode.endfield import rigscene as RS
    from polylode.gf2.clips import skinned
    document, arrays = _scene(tmp_path)
    rest = arrays["bone_rest"].astype(np.float64)
    big = np.diag([1.202, 1.202, 1.202, 1.0])
    # the root grown about its own origin: the spine carried 1.202 times further out, and grown with it
    root = rest[0] @ big
    spine = root @ np.linalg.solve(rest[0], rest[1])
    arrays["anim0"] = np.stack([np.stack([root, spine])] * 2).astype(np.float32)
    document["animations"] = [{"name": "big", "rate": 30.0, "frames": 2}]
    glb.write(document, arrays, tmp_path, tmp_path / "big.glb")
    gltf, accessor, _view = _read(tmp_path / "big.glb")
    (animation,) = gltf["animations"]
    channels = {(c["target"]["node"], c["target"]["path"]): animation["samplers"][c["sampler"]]
                for c in animation["channels"]}
    assert set(channels) == {(1, "scale")}                          # the spine neither moves nor turns from its parent
    assert np.allclose(accessor(channels[(1, "scale")]["output"]), 1.202, atol=1e-5)
    # the frame Unity skins with: rest_matrix of the world for a bone only moved and turned, whatever its rest's own
    # scale; the clip's growth on top where it scales it; tiny, not singular, where it scales it to nothing
    stretched = np.diag([1.0, 2.0, 0.5, 1.0])
    moved = _turn(0.7) @ stretched
    moved[:3, 3] = [0.1, 0.2, 0.3]
    assert np.allclose(skinned(moved, stretched), RS.rest_matrix(moved))
    frame = skinned(big @ stretched, stretched)
    assert np.allclose(frame[:3, :3], 1.202 * RS.rest_matrix(stretched)[:3, :3])
    assert np.linalg.svd(skinned(np.diag([0.0, 1, 1, 1]), np.eye(4))[:3, :3], compute_uv=False).min() > 0
    # a frame with shear (a child turned below an unevenly scaled parent) comes out the TRS nearest it
    sheared = np.diag([1.0, 2.0, 0.5, 1.0]) @ _turn(0.6)
    t, q, s = glb._fitted(sheared[None])
    fit = glb._matrices(t, q, s)[0]
    assert np.linalg.norm(fit[:3, :3] - sheared[:3, :3]) < np.linalg.norm(
        glb._matrices(*glb._trs(sheared[None]))[0][:3, :3] - sheared[:3, :3])


def test_below_an_unevenly_scaled_bone_every_joint_still_lands_on_its_place():
    """Measured against the frames, the shear a child's turn takes below an unevenly scaled bone (which no TRS holds)
    added up down a chain: Cheyanne SSR0101's skirt tips sat 106 mm off.  Against the parent as glTF composes it, each
    joint is exact."""
    parents = np.array([-1, 0, 1, 2])
    frames = np.stack([np.eye(4)] * 4)[None].copy()
    frames[0, 0, :3, :3] = np.diag([1.0, 2.0, 0.5])                    # the root scaled unevenly
    for bone in (1, 2, 3):                                             # each next one turned 40° and 0.3 m further
        turned = np.eye(4)
        turned[:3, :3] = _turn(0.7)[:3, :3] * (1.0 + 0.4 * bone)
        frames[0, bone] = turned
        frames[0, bone, :3, 3] = [0.3 * bone, 0.1 * bone, 0.05 * bone]
    translation, rotation, scale = glb._local_trs(frames, parents)
    made = glb._matrices(translation, rotation, scale)[0]
    placed = [made[0]]
    for bone in (1, 2, 3):
        placed.append(placed[parents[bone]] @ made[bone])
    for bone in range(4):
        assert np.allclose(placed[bone][:3, 3], frames[0, bone, :3, 3], atol=1e-12)
    # rigid frames give the plain parent-relative locals
    rigid = np.stack([np.eye(4), _turn(0.3)])[None].copy()
    rigid[0, 1, :3, 3] = [0, 0.5, 0]
    t, q, s = glb._local_trs(rigid, np.array([-1, 0]))
    assert np.allclose(s, 1) and np.allclose(t[0, 1], [0, 0.5, 0]) and np.allclose(q[0, 1], glb._quaternions(_turn(0.3)[:3, :3]))


def test_rotation_matrices_become_the_quaternions_that_give_them_back():
    from scipy.spatial.transform import Rotation
    turns = Rotation.random(200, random_state=7)
    quats = glb._quaternions(turns.as_matrix())
    assert np.allclose(Rotation.from_quat(quats).as_matrix(), turns.as_matrix(), atol=1e-9)
    assert np.allclose(glb._quaternions(np.eye(3)), [0, 0, 0, 1])
    half = Rotation.from_euler("x", 180, degrees=True).as_matrix()       # w = 0: another branch picks it
    assert np.allclose(Rotation.from_quat(glb._quaternions(half)).as_matrix(), half)


def test_only_the_keys_linear_interpolation_needs_are_kept_and_the_rest_lie_within_tolerance():
    from polylode.interchange import glb as G
    straight = np.linspace([0, 0, 0], [1, 2, 3], 50)
    assert G.keys(straight, 1e-4).tolist() == [0, 49]                  # a straight move is its two ends
    frames = np.arange(400)
    curve = np.stack([np.sin(frames / 120) * .1, np.cos(frames / 90) * .05, frames / 4000], axis=1)
    kept = G.keys(curve, 1e-4)
    assert kept[0] == 0 and kept[-1] == 399 and len(kept) < len(curve) / 2
    rebuilt = np.stack([np.interp(frames, kept, curve[kept, c]) for c in range(3)], axis=1)
    assert np.abs(rebuilt - curve).max() <= 1e-4                       # what glTF draws between them
    assert G.keys(curve[:2], 1e-4).tolist() == [0, 1]


def test_a_shape_that_moves_nothing_on_a_part_is_still_written_with_sparse_values():
    """An accessor of neither a buffer view nor sparse values is all zeros by the spec, but Qt's importer never finished
    loading pelica.glb, whose eyebrows each carry the other side's two shapes."""
    builder = glb._Builder()
    empty = builder.sparse(260, np.zeros(0, dtype=np.uint32), np.zeros((0, 3), dtype=np.float32))
    moved = builder.sparse(4, np.array([2, 0]), np.array([[0, .1, 0], [.2, 0, 0]], dtype=np.float32))
    record = builder.accessors[empty]
    assert record["count"] == 260 and record["sparse"]["count"] == 1 and record["min"] == record["max"] == [0, 0, 0]
    assert builder.accessors[moved]["sparse"]["count"] == 2 and builder.accessors[moved]["max"][1] == pytest.approx(.1)
