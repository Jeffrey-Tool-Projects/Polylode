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
    return np.asarray(Image.open(path.parent / record["maps"]["_MainTex"]["file"]).convert("RGB"), float), record


def test_the_add_layer_then_the_multiply_layer_are_baked_in_linear_light(tmp_path):
    original = _document(tmp_path, add_z=0.0004, multiply_z=0.0005)
    baked = eye_bake.bake(original, log=lambda _line: None)
    assert baked != original and json.loads(original.read_text())["materials"]["eye"]["maps"]["_MainTex"]["file"] \
        == "textures/eye.png"                                       # the .blend's document is left alone
    pixels, record = _eye(baked)
    assert record["eye_layers_baked"] == ["glint", "shade"]
    lin = eye_bake._linear(np.array(128 / 255))
    expected = eye_bake._srgb((lin + 1.0 * 0.1) * (1.0 + 0.5 * (0.0 - 1.0))) * 255   # add, then multiply
    assert np.allclose(pixels, expected, atol=1.0)


def test_a_layer_further_than_the_reach_is_not_the_eyes(tmp_path):
    original = _document(tmp_path, add_z=0.02, multiply_z=0.02)                  # 2 cm away: another surface
    pixels, _record = _eye(eye_bake.bake(original, log=lambda _line: None))
    assert np.allclose(pixels, 128)
