"""GF2's eye blend layers baked into the eye's map for glTF and FBX (``gf2/eye_bake.py``, §7.1455)."""
import json

import numpy as np
from PIL import Image

from polylode.gf2 import eye_bake
from polylode.gf2.export import EYE, EYE_ADD, EYE_MULTIPLY


def _quad(z: float):
    """Two triangles over a 1 cm square at height *z*, UVs spanning the whole map."""
    co = np.array([[0, 0, z], [0.01, 0, z], [0.01, 0.01, z], [0, 0.01, z]], np.float32)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    return co, uv, np.array([[0, 1, 2], [0, 2, 3]])


def _document(tmp_path, add_z: float, multiply_z: float):
    (tmp_path / "textures").mkdir()
    Image.new("RGBA", (8, 8), (128, 128, 128, 255)).save(tmp_path / "textures" / "eye.png")
    Image.new("RGBA", (8, 8), (255, 255, 255, 255)).save(tmp_path / "textures" / "glint.png")
    Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(tmp_path / "textures" / "shade.png")
    arrays, parts = {}, []
    for index, (name, z) in enumerate((("eye", 0.0), ("glint", add_z), ("shade", multiply_z))):
        co, uv, tri = _quad(z)
        arrays.update({f"p{index}_co": co, f"p{index}_uv0": uv, f"p{index}_tri": tri, f"p{index}_slot": np.zeros(2, int)})
        parts.append({"name": name, "materials": [name]})
    np.savez(tmp_path / "x.npz", **arrays)
    material = lambda shader, file, **more: {"shader": shader, "base": "_MainTex",
                                            "maps": {"_MainTex": {"file": f"textures/{file}"}}, **more}
    document = {"arrays": "x.npz", "parts": parts, "materials": {
        "eye": material(EYE, "eye.png"),
        "glint": material(EYE_ADD, "glint.png", tint=[1, 1, 1, 1], strength=0.1, blend="add"),
        "shade": material(EYE_MULTIPLY, "shade.png", tint=[1, 1, 1, 1], strength=0.5, blend="multiply")}}
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _eye(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    record = document["materials"]["eye"]
    return np.asarray(Image.open(path.parent / record["maps"][record["base"]]["file"]).convert("RGB"), float), record


def test_the_add_layer_then_the_multiply_layer_are_baked_in_linear_light(tmp_path):
    original = _document(tmp_path, add_z=0.0004, multiply_z=0.0005)
    baked = eye_bake.bake(original, log=lambda _line: None)
    assert baked != original and json.loads(original.read_text())["materials"]["eye"]["maps"]["_MainTex"]["file"] \
        == "textures/eye.png"                                       # the .blend's document is left alone
    pixels, record = _eye(baked)
    assert record["eye_layers_baked"] == ["glint", "shade"]
    assert record["base"] == eye_bake.BAKED_KEY and record["maps"]["_MainTex"]["file"] == "textures/eye.png"
    lin = eye_bake._linear(np.array(128 / 255))
    expected = eye_bake._srgb((lin + 1.0 * 0.1) * (1.0 + 0.5 * (0.0 - 1.0))) * 255   # add, then multiply
    assert np.allclose(pixels, expected, atol=1.0)


def test_a_layer_further_than_the_reach_is_not_the_eyes(tmp_path):
    original = _document(tmp_path, add_z=0.02, multiply_z=0.02)                  # 2 cm away: another surface
    pixels, _record = _eye(eye_bake.bake(original, log=lambda _line: None))
    assert np.allclose(pixels, 128)


def _map(path, record, role):
    return np.asarray(Image.open(path.parent / record["maps"][record["eye"]["maps"][role]]["file"]).convert("RGB"),
                      float) / 255


def test_the_layers_are_also_kept_as_maps_on_the_eyes_uv_with_the_programs_terms(tmp_path):
    """A reader that lights the eye itself (the add layer by 0.2 + 0.8 N.L, IslandLife §7.1491) gets the add
    colour at its scale and the multiply factor on the eye's UV, and the unbaked map."""
    baked = eye_bake.bake(_document(tmp_path, add_z=0.0004, multiply_z=0.0005), log=lambda _line: None)
    _pixels, record = _eye(baked)
    eye = record["eye"]
    assert eye["game"] == "gf2" and eye["status"] == "baked" and eye["source_base"] == "_MainTex"
    assert eye["maps"] == {"source_base": "_MainTex", "add": eye_bake.ADD_KEY, "multiply": eye_bake.MULTIPLY_KEY}
    assert eye["add_scale"] == 1.0 and [row["material"] for row in eye["layers"]] == ["glint", "shade"]
    added = eye_bake._linear(_map(baked, record, "add")) * eye["add_scale"]
    assert np.allclose(added, 1.0 * 0.1, atol=2 / 255)                 # the glint's white at strength 0.1
    assert np.allclose(_map(baked, record, "multiply"), 1.0 + 0.5 * (0.0 - 1.0), atol=1 / 255)
    assert record["maps"][eye_bake.ADD_KEY]["srgb"] and not record["maps"][eye_bake.MULTIPLY_KEY]["srgb"]


def test_the_eyes_own_specular_map_is_laid_on_fully_lit(tmp_path):
    """``_Specularmap`` × ``_SpecularIntensity`` × N.L (taken as 1), before the layers."""
    path = _document(tmp_path, add_z=0.02, multiply_z=0.02)                       # no layer reaches the eye
    Image.new("RGBA", (8, 8), (64, 64, 64, 255)).save(tmp_path / "textures" / "spec.png")
    document = json.loads(path.read_text(encoding="utf-8"))
    eye = document["materials"]["eye"]
    eye["maps"]["_Specularmap"] = {"file": "textures/spec.png"}
    eye["floats"] = {"_SpecularIntensity": 0.5, "_ShadowIntensity": 0.25}
    path.write_text(json.dumps(document), encoding="utf-8")
    pixels, record = _eye(eye_bake.bake(path, log=lambda _line: None))
    lin = eye_bake._linear(np.array(128 / 255)) + eye_bake._linear(np.array(64 / 255)) * 0.5
    assert np.allclose(pixels, eye_bake._srgb(lin) * 255, atol=1.0)
    assert record["eye"]["maps"] == {"source_base": "_MainTex", "specular": "_Specularmap"}
    assert record["eye"]["values"]["_ShadowIntensity"] == 0.25 and "eye_layers_baked" not in record
