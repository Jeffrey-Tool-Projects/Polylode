"""Endfield materials the writers would not draw (``endfield/vfx_bake``): baked from the game's shader, their
emission baked, or marked invisible with why — never drawn grey."""
import json

import numpy as np
from PIL import Image

from polylode.endfield import shader_code as SC, vfx_bake as VB


def document(tmp_path, materials):
    arrays = {"p0_co": np.zeros((3, 3), np.float32), "p0_tri": np.array([[0, 1, 2]]),
              "p0_slot": np.array([0]), "p0_uv0": np.array([[0, 0], [1, 0], [0, 1]], np.float32)}
    np.savez(tmp_path / "x.npz", **arrays)
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"materials": materials, "arrays": "x.npz",
                                "parts": [{"name": "wing", "materials": list(materials)}]}), encoding="utf-8")
    return path


def test_a_material_with_nothing_to_draw_and_no_shader_to_run_is_marked_invisible_with_why(tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "cache_dir", lambda: tmp_path / "shaders")
    asked = []
    path = document(tmp_path, {"M_fx_wing": {"shader": "HGRP/CharacterNPR_VFX", "maps": {}, "base": None}})
    refused = VB.complete(path, asked.extend, log=lambda *_: None)
    record = json.loads(path.read_text(encoding="utf-8"))["materials"]["M_fx_wing"]
    assert asked == ["HGRP/CharacterNPR_VFX"]                                   # fetched once, then
    assert record["native_surface_unmapped"] is True and "HGRP/CharacterNPR_VFX" in record["unmapped_reason"]
    assert len(refused) == 1


def test_the_main_shaders_emission_map_and_parallax_crystal_are_baked_as_its_emission(tmp_path):
    (tmp_path / "t").mkdir()
    base = np.zeros((4, 4, 4), np.uint8); base[..., 3] = 0; base[:2, :, 3] = 255     # the crystal: the top half
    Image.fromarray(base, "RGBA").save(tmp_path / "t/base.png")
    emission = np.zeros((4, 4, 4), np.uint8); emission[..., 3] = 255; emission[:, :2, :3] = 255
    Image.fromarray(emission, "RGBA").save(tmp_path / "t/e.png")
    Image.fromarray(np.full((2, 2, 4), 255, np.uint8), "RGBA").save(tmp_path / "t/p.png")
    record = {"shader": "HGRP/CharacterNPR", "keywords": ["_EMISSION", "_PARALLAX_MAP"], "base": "_BaseMap",
              "floats": {"_EmissionBrightness": 2.0, "_UseParallax": 1.0},
              "colors": {"_EmissionColor": [1, 1, 1, 1], "_ParallaxColor": [1, 0, 0, 1], "_BaseColor": [1, 1, 1, 1]},
              "maps": {"_BaseMap": {"file": "t/base.png", "srgb": True}, "_EmissionMap": {"file": "t/e.png", "srgb": True},
                       "_ParallaxTex": {"file": "t/p.png", "srgb": True, "scale": [0, 0], "offset": [0, 0]}}}
    out = VB.npr_emission("horn", record, tmp_path)
    assert out["emission"]["strength"] == 3.0                           # 2 × the map, + 1 × the red crystal
    stored = np.asarray(Image.open(tmp_path / out["maps"]["_EmissionBaked"]["file"]), float) / 255
    light = np.where(stored <= 0.04045, stored / 12.92, ((stored + 0.055) / 1.055) ** 2.4) * 3   # linear, unscaled
    assert np.allclose(light[0, 0], [3, 2, 2], atol=0.05)              # the map's white × 2, + the red crystal
    assert np.allclose(light[0, 3], [1, 0, 0], atol=0.02)              # the crystal alone
    assert np.allclose(light[3, 0], [2, 2, 2], atol=0.05)              # the map alone (no crystal below)
    assert np.allclose(light[3, 3], [0, 0, 0], atol=0.02)
    assert VB.npr_emission("x", dict(record, keywords=[]), tmp_path) == dict(record, keywords=[])


def test_the_hair_shaders_emission_is_baked_but_a_parallax_term_only_where_it_was_read(tmp_path):
    (tmp_path / "t").mkdir()
    Image.fromarray(np.full((2, 2, 4), 255, np.uint8), "RGBA").save(tmp_path / "t/base.png")
    Image.fromarray(np.full((2, 2, 4), 255, np.uint8), "RGBA").save(tmp_path / "t/e.png")
    record = {"shader": "HGRP/CharacterNPR_Hair", "keywords": ["_EMISSION", "_PARALLAX_MAP"], "base": "_BaseMap",
              "floats": {"_EmissionBrightness": 0.5}, "colors": {"_EmissionColor": [1, 1, 1, 1], "_ParallaxColor": [1, 0, 0, 1]},
              "maps": {"_BaseMap": {"file": "t/base.png", "srgb": True}, "_EmissionMap": {"file": "t/e.png", "srgb": True},
                       "_ParallaxTex": {"file": "t/e.png", "srgb": True}}}
    out = VB.npr_emission("hair", record, tmp_path)
    stored = np.asarray(Image.open(tmp_path / out["maps"]["_EmissionBaked"]["file"]), float) / 255
    assert out["emission"]["strength"] == 1.0 and np.allclose(stored[0, 0], 0.735, atol=0.01)   # 0.5, sRGB-encoded


def test_a_depth_only_shader_and_an_unbound_base_are_drawn_as_the_game_draws_them():
    depth = {"m_ParsedForm": {"m_SubShaders": [{"m_Passes": [{"m_State": {"rtBlend": [{"colMask": {"val": 0.0}}]}}]}],
                              "m_PropInfo": {"m_Props": [{"m_Name": "_ColorMask", "m_DefValue": [15.0, 0, 0, 0]}]}}}
    assert VB._draws_no_colour(depth, {})
    driven = json.loads(json.dumps(depth))
    driven["m_ParsedForm"]["m_SubShaders"][0]["m_Passes"][0]["m_State"]["rtBlend"][0]["colMask"]["name"] = "_ColorMask"
    assert not VB._draws_no_colour(driven, {})                        # the property's default: all colours
    assert VB._draws_no_colour(driven, {"floats": {"_ColorMask": 0.0}})
    npr = {"m_ParsedForm": {"m_SubShaders": [{"m_Passes": []}],
                            "m_PropInfo": {"m_Props": [{"m_Name": "_BaseMap", "m_DefTexture": {"m_DefaultName": "white"}}]}}}
    record = {"shader": "HGRP/CharacterNPR", "keywords": ["_ALPHABLEND_ON"], "maps": {"_MainTex": {}},
              "colors": {"_BaseColor": [1.0, 1.0, 1.0, 0.0]}}
    assert VB.unbound_base(record, npr)["constant_color"] == [1.0, 1.0, 1.0, 0.0]   # Purrche's: clear
    opaque = VB.unbound_base(dict(record, keywords=[]), npr)
    assert opaque["constant_color"] == [1.0, 1.0, 1.0, 1.0]                          # alpha only when it blends
    assert VB.drawable({"tint": [1, 1, 1, 1], "blend": "multiply", "maps": {}})       # an overlay's tint layer


def test_light_added_without_covering_covers_by_its_brightness_and_premultiplied_colour_is_divided_out():
    colour = np.array([[0.25, 0.5, 0.0, 0.5],      # premultiplied: (0.5, 1, 0) at half coverage
                       [0.6, 0.3, 0.0, 0.0],       # _BlendMode 1: light added, alpha 0
                       [0.0, 0.0, 0.0, 0.0]])      # nothing
    rgb, alpha = VB.straight(colour, "premultiplied")
    assert np.allclose(alpha, [0.5, 0.6, 0.0]) and np.allclose(rgb[0], [0.5, 1.0, 0.0])
    assert np.allclose(rgb[1] * alpha[1], [0.6, 0.3, 0.0]) and not rgb[2].any()   # the light it adds, kept
    rgb, alpha = VB.straight(colour, "straight")
    assert alpha[1] == 0.0 and not rgb[1].any()          # a straight blend at alpha 0 draws nothing, as the game
