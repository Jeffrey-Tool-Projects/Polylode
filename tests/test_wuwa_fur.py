"""Wuthering Waves' parallax fur built as strand layers (``wuthering_waves/fur_shells``), and the shader input
signature's Used column (``shaders/dxbc.used_inputs``) the coverage bake checks against the mesh."""
import numpy as np
import pytest
from PIL import Image

from polylode.wuthering_waves import fur_shells
from polylode.wuthering_waves.shaders import dxbc


def fur_case(tmp_path, alpha=0.5):
    base = np.zeros((8, 8, 4), np.uint8)
    base[..., 0], base[..., 3] = 200, 255
    Image.fromarray(base, "RGBA").save(tmp_path / "base.png")
    height = np.zeros((4, 4, 4), np.uint8)
    height[1, 1, 0], height[..., 3] = 255, 255
    Image.fromarray(height, "RGBA").save(tmp_path / "height.png")
    record = {"base": "Base", "maps": {"Base": {"file": "base.png"}, "HeightMap": {"file": "height.png"}},
              "native_material": {"Switches": {"UseFurryParallax": True},
                                  "Scalars": {"MaxStep": 4, "Height": 50.0, "HeightMapSize": 2.0}}}
    document = {"materials": {"Fur": record, "Skin": {"maps": {}}},
                "parts": [{"name": "Tail", "materials": ["Fur", "Skin"], "shapes": []}]}
    co = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]], np.float32)
    arrays = {"p0_co": co, "p0_no": np.tile([0, 0, 1], (6, 1)).astype(np.float32),
              "p0_uv0": np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 1], [1, 0]], np.float32),
              "p0_col": np.array([[1, 1, 1, alpha]] * 4 + [[1, 1, 1, 1]] * 2, np.float32),
              "p0_tri": np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5]]), "p0_slot": np.array([0, 0, 1])}
    return document, arrays


def test_fur_cards_go_to_the_ground_and_strands_stand_above_them(tmp_path):
    document, arrays = fur_case(tmp_path)
    document, arrays = fur_shells.build(document, arrays, tmp_path, log=lambda *_: None)
    assert arrays["p0_tri"].tolist() == [[1, 4, 5]]                    # the cards left their part; the skin stays
    layers = document["parts"][1]
    assert layers["materials"] == ["Fur", "Fur_Strands1", "Fur_Strands2", "Fur_Strands3"]
    assert len(arrays["p1_tri"]) == 2 * 4 and sorted(set(arrays["p1_slot"].tolist())) == [0, 1, 2, 3]
    z = arrays["p1_co"][:, 2].reshape(4, 4)
    # one world unit per UV unit here: Height 50 × 0.01 = 0.5 deep, the ground at −0.5 and up by ⅛ each layer
    assert z[:, 0] == pytest.approx([-0.5, -0.375, -0.25, -0.125])
    assert [document["materials"][f"Fur_Strands{k}"]["cutoff"] for k in (1, 2, 3)] == [0.25, 0.5, 0.75]
    strands = np.asarray(Image.open(tmp_path / document["materials"]["Fur_Strands1"]["maps"]["Base_Strands"]["file"]))
    assert strands[..., 3].max() > 150 and (strands[..., 3] == 0).mean() > 0.5   # strands only where the field stands


def test_material_without_parallax_fur_or_cards_is_left_alone(tmp_path):
    document, arrays = fur_case(tmp_path, alpha=1.0)                   # no vertex in the card band
    before = {k: v.copy() for k, v in arrays.items()}
    document, arrays = fur_shells.build(document, arrays, tmp_path, log=lambda *_: None)
    assert len(document["parts"]) == 1 and all(np.array_equal(arrays[k], before[k]) for k in before)
    assert not fur_shells.is_fur({"base": "B", "maps": {"HeightMap": {}}, "native_material": {"Switches": {}}})


SIGNATURE = """// Input signature:
//
// Name                 Index   Mask Register SysValue  Format   Used
// -------------------- ----- ------ -------- -------- ------- ------
// TEXCOORD                 0   xyzw        0     NONE   float   xy w
// TEXCOORD                 1   xyz         1     NONE   float
// COLOR                    0   xyzw        2     NONE   float   x  w
//
// Output signature:
"""


def test_used_inputs_read_the_signatures_used_column():
    assert dxbc.used_inputs(SIGNATURE) == {("TEXCOORD", 0, "x"), ("TEXCOORD", 0, "y"), ("TEXCOORD", 0, "w"),
                                           ("COLOR", 0, "x"), ("COLOR", 0, "w")}
