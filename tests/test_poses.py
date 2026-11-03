"""A GF2 outfit's display poses read from the game's own Unity clips (``gf2/clips``, ``gf2/poses``)."""
import struct

import numpy as np
from scipy.spatial.transform import Rotation

from polylode.gf2 import clips as G
from polylode.gf2 import poses as GP


def _words(frames):
    """A streamed block: per frame its time, its key count, then each key's curve index and cubic."""
    data = b""
    for time, keys in frames:
        data += struct.pack("<fi", time, len(keys))
        for index, coeffs in keys:
            data += struct.pack("<i4f", index, *coeffs)
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def _clip(bindings, streamed, dense, constant, stop=1.0, rate=2.0):
    count = streamed[1] + dense[1] + len(constant)
    return {"m_SampleRate": rate, "m_ClipBindingConstant": {"genericBindings": bindings},
            "m_MuscleClip": {"m_StartTime": 0.0, "m_StopTime": stop, "m_LoopTime": True,
                             "m_ValueArrayDelta": [{}] * count,
                             "m_Clip": {"data": {
                                 "m_StreamedClip": {"data": streamed[0], "curveCount": streamed[1]},
                                 "m_DenseClip": {"m_CurveCount": dense[1], "m_SampleArray": dense[0], "m_BeginTime": 0.0,
                                                 "m_SampleRate": rate, "m_FrameCount": len(dense[0]) // max(dense[1], 1)},
                                 "m_ConstantClip": {"data": constant}}}}}


def test_a_unity_clips_curves_are_its_streamed_cubics_dense_samples_and_constants_in_binding_order():
    root, arm = G.path_hash("Hips"), G.path_hash("Hips/Arm")
    bindings = [{"path": root, "attribute": 1, "typeID": 4, "customType": 0},          # position: 3 curves
                {"path": arm, "attribute": 4, "typeID": 4, "customType": 5},           # Euler, an order named: 3
                {"path": 99, "attribute": 7, "typeID": 137, "customType": 20}]         # not a Transform: 1
    # curve 0 (position x) rises 0 → 1 over the second: a key at the sentinel time, then t at 0
    streamed = (_words([(-3.4e38, [(0, (0, 0, 0, 0))]), (0.0, [(0, (0, 0, 1.0, 0))]), (1.0, [(0, (0, 0, 0, 1.0))])]), 1)
    dense = ([0.5, 0.75, 1.0], 1)                                                     # position y, 2 frames a second
    constant = [2.0, 0.0, 30.0, 0.0, 7.0, 5.0, 6.0]    # position z, the Euler, the other; two curves no binding takes
    times, by_hash = G.transforms(_clip(bindings, streamed, dense, constant))
    assert np.allclose(times, [0, .5, 1])
    assert np.allclose(by_hash[root][G.POSITION], [[0, .5, 2], [.5, .75, 2], [1, 1, 2]])
    assert np.allclose(by_hash[arm][G.EULER][0], [0, 30, 0])
    turned = Rotation.from_quat(by_hash[arm][G.ROTATION][0])
    assert np.allclose(turned.as_euler("zyx", degrees=True), [0, 30, 0], atol=1e-9)
    assert 99 not in by_hash


def test_an_outfits_poses_are_its_own_clips_off_the_battlefield():
    rows = [{"name": n} for n in ("c_NikketaSSR0102_BarrackIdle", "c_NikketaSSR0102_slg_Run", "c_NikketaSSR0103_Cloth_Idle",
                                  "c_NikketaSSR0102_Cloth_Idle")]
    assert GP.outfit_of("c_NikketaSSR0102_slg_skin_48e81872d7") == "NikketaSSR0102"
    assert GP.outfit_of("c_VepleyDorm_skin") == "VepleyDorm"
    assert [r["name"] for r in GP.candidates(rows, "NikketaSSR0102")] == ["c_NikketaSSR0102_BarrackIdle",
                                                                           "c_NikketaSSR0102_Cloth_Idle"]
