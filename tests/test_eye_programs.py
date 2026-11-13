"""The three games' eye programs, baked for glTF / FBX and stated for the sidecar and the preview (IslandLife §7.1491):
Endfield's iris terms (``endfield/eye_bake.py``), Wuthering Waves' base pass (``wuthering_waves/eye_bake.py``) and
what the preview reads of them (``extractor/character_styles.eye_style``)."""
import json

import numpy as np
import pytest
from PIL import Image

from polylode.endfield import eye_bake as EB
from polylode.extractor import character_styles as CS
from polylode.wuthering_waves import eye_bake as WB
from polylode.wuthering_waves.shaders import vm


def _iris(**colours):
    colors = {"_EyeHighLightColor": [3.0, 2.0, 1.0, 1.0], "_EyeScatteringColor": [1.0, 1.0, 1.0, 1.0],
              "_MatcapColor": [0.0, 0.0, 0.0, 0.0]} | colours
    return {"shader": EB.PROGRAM, "keywords": ["_DIFF_RAMP_ON", "_EYE_HIGHLIGHT", "_MATCAP_ON"], "base": "_BaseMap",
            "floats": {"_ParallaxScale": 0.03, "_MatcapNormalScale": 0.9}, "colors": colors,
            "maps": {"_BaseMap": {"file": "textures/iris.png", "srgb": True},
                     "_MatcapTex": {"file": "textures/matcap.png", "srgb": False}}}


def test_the_iris_sprite_outside_the_disc_is_brightened_and_the_scattering_follows_the_alpha():
    record = _iris(_EyeScatteringColor=[0.5, 0.5, 0.5, 1.0])
    base = np.full((8, 8, 4), 0.2)
    base[..., 3] = 0.0
    base[0, 0, 3] = 1.0                                        # a corner texel: outside the disc, alpha 1
    colour, clipped = EB.bake_map(base, record, EB.terms(record), np.zeros((4, 4, 4)))
    assert np.allclose(colour[4, 4, :3], 0.2)                  # the disc's centre: untouched
    assert np.allclose(colour[0, 1, :3], [0.6, 0.4, 0.2])      # outside: × _EyeHighLightColor
    assert np.allclose(colour[0, 0, :3], [0.3, 0.2, 0.1])      # × lerp(1, _EyeScatteringColor, alpha)
    assert clipped == 0
    hot = _iris(_EyeHighLightColor=[10.0, 1.0, 1.0, 1.0])
    _colour, clipped = EB.bake_map(base, hot, EB.terms(hot), np.zeros((4, 4, 4)))
    assert clipped > 0                                          # past 1: clipped and counted


def test_the_iris_matcap_is_read_by_the_dome_normal_head_on():
    record = _iris(_MatcapColor=[0.0, 0.0, 1.0, 0.5])
    matcap = np.zeros((16, 16, 4))
    matcap[..., 0] = np.linspace(0, 1, 16)[None, :]            # red grows to the right
    matcap[..., 3] = 1.0
    base = np.zeros((32, 32, 4))
    colour, _ = EB.bake_map(base, record, EB.terms(record), matcap)
    # rgb × _MatcapColor.a + a × _MatcapColor.rgb; at the disc's centre the dome faces the camera (the middle)
    assert colour[16, 16, 2] == pytest.approx(1.0) and colour[16, 16, 0] == pytest.approx(0.25, abs=0.05)
    # the dome leans the normal against q (× −_MatcapNormalScale): left of centre reads further right
    assert colour[16, 4, 0] > colour[16, 16, 0] > colour[16, 28, 0]


def test_an_iris_missing_a_value_its_program_reads_is_refused_and_another_material_is_not_an_iris():
    record = _iris()
    del record["colors"]["_EyeScatteringColor"]
    with pytest.raises(EB.EyeError):
        EB.terms(record)
    assert EB.terms(dict(_iris(), keywords=["_DIFF_RAMP_ON"])) is None          # a brow: no highlight terms
    assert EB.terms(dict(_iris(), shader="HGRP/CharacterNPR")) is None


def test_the_endfield_document_keeps_the_unbaked_map_and_states_the_terms(tmp_path):
    (tmp_path / "textures").mkdir()
    Image.new("RGBA", (8, 8), (100, 100, 100, 255)).save(tmp_path / "textures" / "iris.png")
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(tmp_path / "textures" / "matcap.png")
    document = {"materials": {"iris": _iris()}, "parts": [{"name": "eye", "materials": ["iris"]}]}
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    baked = EB.bake(path, log=lambda _line: None)
    assert baked != path and json.loads(path.read_text())["materials"]["iris"]["base"] == "_BaseMap"
    record = json.loads(baked.read_text())["materials"]["iris"]
    assert record["base"] == EB.BAKED_KEY and record["maps"]["_BaseMap"]["file"] == "textures/iris.png"
    eye = record["eye"]
    assert eye["status"] == "baked" and eye["source_base"] == "_BaseMap"
    assert eye["maps"] == {"source_base": "_BaseMap", "matcap": "_MatcapTex"}
    assert eye["colors"]["_EyeHighLightColor"] == [3.0, 2.0, 1.0, 1.0] and eye["floats"]["_MatcapNormalScale"] == 0.9
    pixels = np.asarray(Image.open(tmp_path / record["maps"][EB.BAKED_KEY]["file"]).convert("RGB"))
    assert pixels[0, 0, 0] > pixels[4, 4, 0] == 100            # the sprite brightened, the disc as it was


def test_bit_field_instructions_run_as_direct3d_defines_them():
    v0 = np.array([[0, 0, 0, 0]], np.float32)
    asm = """
ubfe r0.x, l(2), l(16), l(0x00030000)
ubfe r0.y, l(0), l(4), l(0xffffffff)
ubfe r0.z, l(8), l(28), l(0xf0000000)
bfi r1.x, l(3), l(5), l(7), l(12)
imad r2.x, l(3), l(4), l(-5)
ibfe r3.x, l(4), l(0), l(0x0000000f)
"""
    machine, _ = run(asm, {0: v0})
    words = lambda k, c: int(machine.registers[k][0, c])
    assert words(0, 0) == 3 and words(0, 1) == 0 and words(0, 2) == 0xF       # past bit 31: shifted by the offset
    assert words(1, 0) == (7 << 5) | 12 and words(2, 0) == 7
    assert machine.registers[3].view(np.int32)[0, 0] == -1                    # sign-extended


def run(asm, inputs, buffers=None, textures=None, clamp=None):
    machine = WB._Machine(vm.parse(asm), len(next(iter(inputs.values()))), inputs, buffers or {}, textures or {},
                          clamp=clamp)
    return machine, machine.run()


def test_a_clamped_texture_is_read_at_its_edge_past_it_and_a_wrapped_one_wraps():
    image = np.zeros((1, 4, 4), np.float32)
    image[0, :, 0] = [0.1, 0.2, 0.3, 0.9]
    v0 = np.array([[1.4, 0.5, 0, 0]], np.float32)                # past the right edge
    asm = "sample_indexable(texture2d)(float,float,float,float) r0.x, v0.xyxx, t0.xyzw, s0\n"
    clamped, _ = run(asm, {0: v0}, textures={0: image}, clamp={0: (True, True)})
    wrapped, _ = run(asm, {0: v0}, textures={0: image})
    assert clamped.registers[0].view(np.float32)[0, 0] == pytest.approx(0.9)
    assert wrapped.registers[0].view(np.float32)[0, 0] == pytest.approx(0.21)  # 1.4 wraps to 0.4: texel 1.1


def test_overlapping_eye_islands_drawn_differently_get_a_map_each():
    """Both eyes on one island, the vertex colour moving the highlight: the texels agree but where it lies."""
    rows = np.array([0, 0, 0, 0, 1, 1])
    columns = np.array([0, 1, 0, 1, 0, 0])
    which = np.array([0, 0, 1, 1, 2, 3])
    values = np.zeros((6, 6))
    values[2, :3] = 1.0                                           # triangle 1 draws a highlight on texel (0, 0)
    values[5, 3] = np.nan                                         # a discarded sample agrees with anything
    takes, layer = WB._layers(rows, columns, which, values, 4, (2, 2))
    assert layer.tolist() == [0, 1, 0, 0] and len(takes) == 2
    assert takes[1].tolist() == [False, False, True, True, False, False]


def test_the_wuwa_eye_is_drawn_from_its_base_pass_maps_and_states_them(tmp_path):
    (tmp_path / "src.png").write_bytes(b"")
    record = {"base": "MainTex", "maps": {"MainTex": {"file": "src.png", "srgb": True, "sha256": "x"}}}
    baked = {"base": np.full((2, 2, 4), 0.5), "emission": np.full((2, 2, 3), 2.0)}
    eye = WB.apply("MI_Eye", record, baked, {"ran": 1}, tmp_path)
    assert record["base"] == WB.BASE_KEY and record["maps"]["MainTex"]["file"] == "src.png"
    assert record["emission"] == {"map": WB.EMISSION_KEY, "color": [1.0, 1.0, 1.0], "strength": 2.0}
    assert eye["base_color"]["lit"] is True and eye["emission"]["lit"] is False and eye["source_base"] == "MainTex"
    emitted = np.asarray(Image.open(tmp_path / record["maps"][WB.EMISSION_KEY]["file"]))
    assert (emitted == 255).all()                                 # 2.0 at strength 2: the map holds 1
    assert "sha256" not in record["maps"][WB.BASE_KEY]


def test_the_preview_reads_each_games_eye_statement():
    endfield = CS.eye_style({"eye": {"game": "endfield", "status": "baked", "keywords": ["_EYE_HIGHLIGHT"],
                                     "colors": {"_EyeHighLightColor": [2, 2, 2, 1]}, "floats": {"_ParallaxScale": .5},
                                     "maps": {"source_base": "_BaseMap"}},
                            "maps": {"_BaseMap": {"srgb": True}}})
    assert endfield["eyeMode"] == 1 and endfield["eyeHighlight"] and not endfield["eyeMatcapOn"]
    assert endfield["maps"] == {"eyeBase": "_BaseMap"} and endfield["eyeParallax"] == .5
    gf2 = CS.eye_style({"eye": {"game": "gf2", "status": "baked", "add_scale": 1.5,
                                "values": {"_ShadowIntensity": .3, "_SpecularIntensity": 0.9, "_FinalTint": None},
                                "maps": {"source_base": "_MainTex", "add": "A", "multiply": "M"}},
                        "maps": {"A": {"srgb": True}, "M": {"srgb": False}}})
    assert gf2["eyeMode"] == 2 and gf2["eyeShadowIntensity"] == .3 and gf2["eyeAddScale"] == 1.5
    assert gf2["eyeAddSrgb"] and not gf2["eyeMultiplySrgb"] and gf2["eyeFinalTint"] == [1, 1, 1, 1]
    wuwa = CS.eye_style({"eye": {"game": "wuwa", "status": "baked", "maps": {"emission": "E"}},
                         "emission": {"map": "E", "strength": 1.7}, "maps": {"E": {"srgb": True}}})
    assert wuwa["eyeMode"] == 3 and wuwa["maps"] == {"eyeEmission": "E"} and wuwa["eyeEmissionStrength"] == 1.7
    assert CS.eye_style({"eye": {"game": "wuwa", "status": None, "reason": "x"}})["eyeMode"] == 0
    assert CS.eye_style({})["eyeMode"] == 0


def test_the_sidecar_carries_the_eye_statement_and_the_glb_embeds_the_maps_it_names(tmp_path):
    from test_interchange_glb import _scene
    from test_interchange_sidecar import _written
    from polylode.interchange import glb
    (tmp_path / "out").mkdir()
    document, arrays = _scene(tmp_path)
    eye = {"game": "gf2", "status": "baked", "source_base": "_MainTex", "maps": {"source_base": "_Normal"}}
    document["materials"]["skin"]["eye"] = eye
    _path, data = _written(tmp_path, document, arrays)
    assert data["materials"]["skin"]["eye"] == eye and "eye" not in data["materials"]["skin"]["native"]
    glb.write(document, arrays, tmp_path, tmp_path / "out" / "model.glb")
    _document, _blob, materials = CS.read_materials(tmp_path / "out" / "model.glb", tmp_path / "cache")
    assert materials[0]["eyeBase"].startswith("file:") and materials[0]["eyeMode"] == 2


def test_parameter_collections_take_the_shader_maps_order_not_the_root_materials():
    """The uniform expression set lists the collections it reads by StateId, one after another: that order numbers
    MaterialCollection<i> (椿's eye: the global collection first, though the root lists the eye's first)."""
    import struct
    from polylode.wuthering_waves.shader_bake import collections_in_map
    guid = lambda text: struct.pack("<4I", *(int(w, 16) for w in text.split("-")))  # noqa: E731
    eyes = {"StateId": "1A6B5811-43528A6C-6AFB37B5-DF037E63", "name": "eyes"}
    world = {"StateId": "C7A8A13A-4BFC1A3B-40B67186-7C79461E", "name": "global"}
    unused = {"StateId": "8071EBCB-42907E48-F974AF8D-5C69D18F", "name": "sequence"}
    uexp = b"\x00" * 9 + guid(world["StateId"]) + guid(eyes["StateId"]) + b"\x01" * 7 + guid(world["StateId"])
    assert [c["name"] for c in collections_in_map(uexp, [eyes, world, unused])] == ["global", "eyes"]
    assert collections_in_map(b"\x00" * 64, [eyes]) == []
