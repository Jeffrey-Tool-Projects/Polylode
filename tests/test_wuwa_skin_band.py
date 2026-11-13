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
    assert found["UseSkinMask"]["value"] is True and found["UseTypeMask"]["value"] is False
    assert found["UseSkinMask_DYN"] == {"value": 1.0, "from": "the instance's scalar"}
    assert found["DefaultSkinFactor"]["value"] == 0.0 and "default" in found["DefaultSkinFactor"]["from"]
    assert found["UseFaceAni_SkinMask"]["value"] is None
