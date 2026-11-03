"""Wuthering Waves' parallax fur built as strand layers (``wuthering_waves/fur_shells``), and the shader input
signature's Used column (``shaders/dxbc.used_inputs``) the coverage bake checks against the mesh."""
import numpy as np
import pytest
from PIL import Image

from polylode.wuthering_waves import fur_shells
from polylode.wuthering_waves.shaders import dxbc


def fur_case(tmp_path, alpha=0.5, flow=None):
    base = np.zeros((8, 8, 4), np.uint8)
    base[..., 0], base[..., 3] = 200, 255
    Image.fromarray(base, "RGBA").save(tmp_path / "base.png")
    height = np.zeros((4, 4, 4), np.uint8)
    height[1, 1, 0], height[..., 3] = 255, 255
    Image.fromarray(height, "RGBA").save(tmp_path / "height.png")
    scalars = {"MinStep": 4, "MaxStep": 4, "Height": 50.0, "HeightMapSize": 2.0, "FlowMapInt": 0.0,
               "FurRootInv": 1.0, "ParallaxRimFadeMin": 0.0, "ParallaxRimFadeMax": 1.0}
    maps = {"Base": {"file": "base.png"}, "HeightMap": {"file": "height.png"}}
    if flow is not None:                                               # one flow everywhere, as RG in 0..1
        Image.fromarray(np.full((2, 2, 4), [*flow, 0, 255], np.uint8), "RGBA").save(tmp_path / "flow.png")
        maps["FlowmapMap"] = {"file": "flow.png", "srgb": False}
        scalars["FlowMapInt"] = 0.1
    record = {"base": "Base", "maps": maps,
              "native_material": {"Switches": {"UseFurryParallax": True}, "Scalars": scalars}}
    document = {"materials": {"Fur": record, "Skin": {"maps": {}}},
                "parts": [{"name": "Tail", "materials": ["Fur", "Skin"], "shapes": []}]}
    co = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]], np.float32)
    arrays = {"p0_co": co, "p0_no": np.tile([0, 0, 1], (6, 1)).astype(np.float32),
              "p0_uv0": np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 1], [1, 0]], np.float32),
              "p0_col": np.array([[1, 1, 1, alpha]] * 4 + [[1, 1, 1, 1]] * 2, np.float32),
              "p0_tri": np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5]]), "p0_slot": np.array([0, 0, 1])}
    return document, arrays


def test_fur_cards_stay_as_the_ground_and_strands_stand_above_them(tmp_path):
    document, arrays = fur_case(tmp_path)
    before = arrays["p0_tri"].copy()
    document, arrays = fur_shells.build(document, arrays, tmp_path, log=lambda *_: None)
    assert np.array_equal(arrays["p0_tri"], before)                    # the card is the ground: kept as it is
    layers = document["parts"][1]
    assert layers["materials"] == ["Fur_Strands1", "Fur_Strands2", "Fur_Strands3"]
    assert len(arrays["p1_tri"]) == 2 * 3 and sorted(set(arrays["p1_slot"].tolist())) == [0, 1, 2]
    z = arrays["p1_co"][:, 2].reshape(3, 4)
    # one world unit per UV unit here: Height 50 × 0.01 = 0.5 above the card, up by ⅛ each layer
    assert z[:, 0] == pytest.approx([0.125, 0.25, 0.375])
    shells = [document["materials"][f"Fur_Strands{k}"] for k in (1, 2, 3)]
    assert [m["cutoff"] for m in shells] == [0.25, 0.5, 0.75] and shells[0]["fur_rim_fade"] == [0.0, 1.0]
    strands = np.asarray(Image.open(tmp_path / shells[0]["maps"]["Base_Strands"]["file"]))
    assert strands[..., 3].max() > 150 and (strands[..., 3] == 0).mean() > 0.5   # strands only where the field stands
    assert np.array_equal(arrays["p1_uv0"].reshape(3, 4, 2)[0], arrays["p0_uv0"][:4])   # no flow map: no lean


def test_the_flow_map_leans_each_layer_by_its_steps_offset(tmp_path):
    document, arrays = fur_case(tmp_path, flow=(255, 128))              # +u, about 0 along v
    document, arrays = fur_shells.build(document, arrays, tmp_path, log=lambda *_: None)
    shift = (arrays["p1_uv0"].reshape(3, 4, 2) - arrays["p0_uv0"][:4])[:, 0]
    n, tiles, offset = 4, 2.0, 1.0 * 0.1 / 4                           # (255 / 255 × 2 − 1) × FlowMapInt / n
    # layer k (height k/n) is step j = n − k: m_j = FurRootInv × (n + 2 − 2(j − 1)) + (j − 1)
    expected = [fur_shells.lean(n - k, n, 1.0) * offset / tiles for k in (1, 2, 3)]
    assert fur_shells.lean(3, 4, 1.0) == 4 and fur_shells.lean(3, 4, 0.0) == 2
    assert shift[:, 0] == pytest.approx(expected, rel=1e-3)
    assert np.abs(shift[:, 1]).max() < 1e-3                            # v: 128 is just over the middle


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
