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
