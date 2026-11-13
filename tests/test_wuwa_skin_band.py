"""Wuthering Waves' skin band read off the base pass's own code (``wuthering_waves/skin_band``)."""
import numpy as np
import pytest

from polylode.wuthering_waves import skin_band
from polylode.wuthering_waves.shaders import dxbc

#: the Material buffer is cb4; SkinColor is its slot 7; the TypeMask map is t1 (Material resource 2), sampled at
#: TEXCOORD0 (v1.xy); the vertex colour is v2
SKIN = {7}
HEAD = """ps_5_0
dcl_output o0.xyzw
"""
LERP = """add r1.xyz, cb4[7].xyzx, l(-1.000000, -1.000000, -1.000000, 0.000000)
mad r2.xyz, r0.xxxx, r1.xyzx, l(1.000000, 1.000000, 1.000000, 0.000000)
mov o0.xyz, r2.xyzx
ret
"""
SAMPLE = "sample_indexable(texture2d)(float,float,float,float) r3.xyzw, v1.xyxx, t1.xyzw, s1\n"


def shader(body: str) -> dxbc.Shader:
    return dxbc.Shader(asm=HEAD + body, textures={1: (4, 2), 2: (4, 4)}, samplers={}, buffers={"Material": 4},
                       inputs={1: [("TEXCOORD", 0, "xy")], 2: [("COLOR", 0, "xyzw")]})


def read(body: str, buffer=None) -> dict:
    material = np.zeros((16, 4))
    material[7] = (0.8, 0.6, 0.5, 1.0)
    if buffer:
        for slot, value in buffer.items():
            material[slot] = value
    return skin_band._variant(shader(body), {4: material}, 4, SKIN, 1)


def test_a_band_on_the_type_masks_red_strictly_above_its_literal_is_read_with_its_comparison():
    found = read(SAMPLE + "lt r0.x, l(0.900000), r3.x\nand r0.x, r0.x, l(0x3f800000)\n" + LERP)
    assert (found["source"], found["channel"], found["map"], found["uv_set"]) == ("TypeMask", "R", "TypeMask", 0)
    assert (found["comparison"], found["threshold"]) == (">", 0.9)
    assert (found["factor_below"], found["factor_above"]) == (0.0, 1.0)
    # the same literal compared with ge is at or above it: the sweep holds 0.9 itself and the float past it
    found = read(SAMPLE + "ge r0.x, r3.x, l(0.900000)\nand r0.x, r0.x, l(0x3f800000)\n" + LERP)
    assert (found["comparison"], found["threshold"]) == (">=", 0.9)


def test_a_band_on_the_vertex_colour_is_read_from_the_colour_input():
    found = read("lt r0.x, l(0.900000), v2.x\nand r0.x, r0.x, l(0x3f800000)\n" + LERP)
    assert (found["source"], found["channel"], found["comparison"], found["threshold"]) == ("COLOR0", "R", ">", 0.9)
    assert "LayerElementColor" in found["in_fbx"]


def test_a_factor_following_no_input_is_the_materials_constant():
    found = read("mov r0.x, cb4[9].y\n" + LERP, {9: (0, 0.25, 0, 0)})
    assert found == {"source": None, "constant_factor": 0.25, "lerp_instruction": 2}


def test_a_pass_that_never_reads_skin_color_draws_no_skin():
    assert read("mov o0.xyz, l(1.0, 1.0, 1.0, 0)\nret\n") == {"source": None, "skin_color_read": False}


def test_the_lerp_left_unfused_as_a_multiply_and_an_add_is_found():
    body = (SAMPLE + "lt r0.x, l(0.900000), r3.x\nand r0.x, r0.x, l(0x3f800000)\n"
            "add r1.xyz, cb4[7].xyzx, l(-1.000000, -1.000000, -1.000000, 0.000000)\n"
            "mul r2.xyz, r0.xxxx, r1.xyzx\nadd r2.xyz, r2.xyzx, l(1.000000, 1.000000, 1.000000, 0.000000)\n"
            "mov o0.xyz, r2.xyzx\nret\n")
    assert read(body)["source"] == "TypeMask"


def test_a_factor_that_also_follows_another_map_is_refused_not_read_as_a_band():
    body = (SAMPLE + "sample_indexable(texture2d)(float,float,float,float) r4.xyzw, v1.xyxx, t2.xyzw, s2\n"
            "lt r0.x, r4.x, r3.x\nand r0.x, r0.x, l(0x3f800000)\n" + LERP)
    with pytest.raises(skin_band.BandError, match="other maps"):
        read(body)


def test_the_switches_and_scalars_say_where_their_values_come_from():
    from polylode.wuthering_waves.shaders.inline import Uniforms
    record = {"native_material": {"Switches": {"UseSkinMask": True}, "Scalars": {"UseSkinMask_DYN": 1.0}}}
    uniforms = Uniforms([], [], [("DefaultSkinFactor", 0.0)], [], [])
    found = skin_band.parameters(record, uniforms)
    # a switch the instance does not set is its parent's (compiled in), not off (IslandLife §7.1492)
    assert found["UseSkinMask"]["value"] is True and found["UseTypeMask"]["value"] is None
    assert "parent" in found["UseTypeMask"]["from"]
    assert found["UseSkinMask_DYN"] == {"value": 1.0, "from": "the instance's scalar"}
    assert found["DefaultSkinFactor"]["value"] == 0.0 and "default" in found["DefaultSkinFactor"]["from"]
    assert found["UseFaceAni_SkinMask"]["value"] is None


def test_every_toon_common_material_but_the_tacet_mark_has_its_band_read():
    from polylode.interchange.sidecar import slot_role
    toon = lambda **extra: {"native_material": {"Switches": {}, "Properties": {                       # noqa: E731
        "BasePropertyOverrides": {"ShadingModel": "MSM_ToonCommon"}}}, **extra}
    assert skin_band.is_target(toon(), slot_role)                   # sets no UseSkinMask: read all the same
    assert skin_band.is_target(toon(skin_band={"status": "no_skin_color"}), slot_role)   # a case read before
    assert not skin_band.is_target(toon(profile="tacet_mark_texture"), slot_role)
    hair = {"native_material": {"Properties": {"BasePropertyOverrides": {"ShadingModel": "MSM_ToonHair"}}}}
    assert not skin_band.is_target(hair, slot_role) and not skin_band.is_target({}, slot_role)


def test_a_shader_map_without_skin_colour_draws_no_skin():
    from polylode.wuthering_waves.shaders.inline import Uniforms

    class Store:
        def map_index(self, digest):
            return 0 if digest == b"m" * 20 else None

    uniforms = Uniforms([], [], [], [("BaseColor", (1, 1, 1, 1))], [])
    original = skin_band.inline.read
    skin_band.inline.read = lambda uexp: [uniforms]
    try:
        found = skin_band.analyse({"native_material": {}}, Store(), b"x" + b"m" * 20, None, [])
    finally:
        skin_band.inline.read = original
    assert found["status"] == "no_skin_color" and found["skin_color_read"] is False
    assert "SkinColor" in found["reason"] and found["evidence"]["shader_map_sha1"] == ("6d" * 20)


def test_a_translucent_map_is_read_off_the_passes_that_tint_toward_skin_colour(monkeypatch):
    # no seven-target pass: the map's Material-reading pixel shaders, those reading SkinColor (§7.1492)
    from polylode.wuthering_waves import shader_bake
    from polylode.wuthering_waves.shaders.inline import Uniforms

    class Store:
        def map_index(self, digest):
            return 0 if digest == b"m" * 20 else None

    uniforms = Uniforms([], [], [], [("SkinColor", (1, 1, 1, 1))], [])
    declared = "dcl_constantbuffer CB4[16], immediateIndexed\n"
    tinting, depth = shader(declared + LERP), shader(declared + "mov o0.xyzw, l(0, 0, 0, 0)\nret\n")
    monkeypatch.setattr(skin_band.inline, "read", lambda uexp: [uniforms])
    monkeypatch.setattr(skin_band, "_slots", lambda *a: set(SKIN))
    monkeypatch.setattr(shader_bake, "_frozen_values", lambda *a: np.zeros((16, 4)))
    monkeypatch.setattr(skin_band, "base_passes", lambda store, index: [])
    monkeypatch.setattr(skin_band, "shading_passes", lambda store, index: [(1, tinting), (2, depth)])
    found = skin_band.analyse({"native_material": {}}, Store(), b"x" + b"m" * 20, None, [])
    assert found["status"] == "constant" and found["constant_factor"] == 0.0
    assert found["evidence"]["translucent_tinting_pixel_shaders"] == [1]
    monkeypatch.setattr(skin_band, "shading_passes", lambda store, index: [(2, depth)])
    found = skin_band.analyse({"native_material": {}}, Store(), b"x" + b"m" * 20, None, [])
    assert found["status"] == "no_skin_color" and found["evidence"]["translucent_passes"] == [2]
    # an opaque map's base passes are its seven-target ones, as before: a pass without the lerp is not dropped
    monkeypatch.setattr(skin_band, "base_passes", lambda store, index: [(1, tinting), (2, depth)])
    assert skin_band.analyse({"native_material": {}}, Store(), b"x" + b"m" * 20, None, [])["status"] == "variants_differ"


def test_a_factor_that_follows_only_another_map_is_refused_not_read_as_a_constant():
    # an ID decoded from another map and compared with a range (PlayerFemale / Lucy 10011's SkinMin, SkinMax;
    # IslandLife §7.1492): at the two constant fills it is the same (0.25 → ID 63, 0.75 → ID 191, neither 4),
    # so without sweeping that map it read as the constant 0
    body = ("sample_indexable(texture2d)(float,float,float,float) r4.xyzw, v1.xyxx, t2.xyzw, s2\n"
            "mul r4.x, r4.x, l(255.000000)\nftoi r4.x, r4.x\nieq r0.x, r4.x, l(4)\n"
            "and r0.x, r0.x, l(0x3f800000)\n" + LERP)
    with pytest.raises(skin_band.BandError, match="another of the material's maps"):
        read(body)


def test_a_map_one_of_whose_variants_follows_another_map_is_refused_whatever_the_others_read(monkeypatch):
    # Lucy 10011's Up (§7.1492): some base-pass variants read the constant 0, the others an ID map
    from polylode.wuthering_waves import shader_bake
    from polylode.wuthering_waves.shaders.inline import Uniforms

    class Store:
        def map_index(self, digest):
            return 0 if digest == b"m" * 20 else None

    declared = "dcl_constantbuffer CB4[16], immediateIndexed\n"
    constant = shader(declared + "mov r0.x, l(0)\n" + LERP)
    by_id = shader(declared + "sample_indexable(texture2d)(float,float,float,float) r4.xyzw, v1.xyxx, t2.xyzw, s2\n"
                   "mul r4.x, r4.x, l(255.000000)\nftoi r4.x, r4.x\nieq r0.x, r4.x, l(4)\n"
                   "and r0.x, r0.x, l(0x3f800000)\n" + LERP)
    monkeypatch.setattr(skin_band.inline, "read", lambda uexp: [Uniforms([], [], [], [("SkinColor", (1, 1, 1, 1))], [])])
    monkeypatch.setattr(skin_band, "_slots", lambda *a: set(SKIN))
    monkeypatch.setattr(shader_bake, "_frozen_values", lambda *a: np.zeros((16, 4)))
    monkeypatch.setattr(skin_band, "base_passes", lambda store, index: [(1, constant), (2, by_id)])
    with pytest.raises(skin_band.BandError, match="another of the material's maps"):
        skin_band.analyse({"native_material": {}}, Store(), b"x" + b"m" * 20, None, [])
    monkeypatch.setattr(skin_band, "base_passes", lambda store, index: [(1, constant)])
    assert skin_band.analyse({"native_material": {}}, Store(), b"x" + b"m" * 20, None, [])["status"] == "constant"
