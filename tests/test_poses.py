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


def test_shape_weights_bind_the_renderer_and_bare_shape_name_and_keep_constant_expressions():
    bindings = [{"path": G.path_hash("Rig"), "attribute": 1, "typeID": 4, "customType": 0},
                {"path": G.path_hash("Face"), "attribute": G.path_hash("smile"), "typeID": 137, "customType": 20},
                {"path": G.path_hash("Face"), "attribute": G.path_hash("blink"), "typeID": 137, "customType": 20}]
    tree = _clip(bindings, ([], 0), ([], 0), [1, 2, 3, 38, 100])
    parts = [{"path": "Other", "shapes": ["smile", "blink"]},
             {"path": "Face", "shapes": ["blink", "smile", "unused"]}]
    weights = G.shape_weights(tree, parts)
    assert list(weights) == [1]
    np.testing.assert_allclose(weights[1], [[1, .38, 0]] * 3)


def test_endfield_exports_its_own_face_controls_ears_and_authored_presets(monkeypatch):
    from polylode.endfield import actor, morph, poses
    skeleton = actor.build_skeleton({"root": ("", np.eye(4)), "face": ("root", np.eye(4)),
                                     "ear": ("root", np.eye(4))}, "test")
    face = {"kind": 0, "bones": ["face"], "base": [[0, 0, 0, 0, 0, 0, 1, 1, 1]],
            "controls": {"smile": {"bones": [[0, [.1, 0, 0, 0, 0, 0, 0, 0, 0]]]}}, "params": {}}
    ear = {"kind": 1, "bones": ["ear"], "base": [[0, 0, 0, 0, 0, 0, 1, 1, 1]],
           "controls": {"ear_up": {"bones": [[0, [0, .2, 0, 0, 0, 0, 0, 0, 0]]]}}, "params": {}}
    monkeypatch.setattr(morph, "for_actor", lambda extra: [face, ear])
    monkeypatch.setattr(morph, "emotions", lambda: {"happy": {"controls": {"smile": .5}, "additive": False}})
    monkeypatch.setattr(morph, "poses", lambda: {"other_actor": {"controls": {"unknown": 1}, "additive": False}})
    document, arrays = {"animations": []}, {}
    assert poses.add_expressions(document, arrays, skeleton, ["root", "face", "ear"], {}, log=lambda _: None) == 3
    assert [record["name"] for record in document["animations"]] == [
        "expression/control/smile", "expression/control/ear_up", "expression/emotion/happy"]
    np.testing.assert_allclose(arrays["anim0"][:, 1, 0, 3], -.1)
    np.testing.assert_allclose(arrays["anim2"][:, 1, 0, 3], -.05)
    np.testing.assert_allclose(arrays["anim1"][:, 2, 2, 3], .2)


def test_endfield_material_only_expressions_are_retained_as_source_data(monkeypatch):
    from polylode.endfield import actor, morph, poses
    skeleton = actor.build_skeleton({"root": ("", np.eye(4))}, "test")
    table = {"kind": 0, "bones": [], "base": [], "controls": {}, "params": {"blush": ["_Blush", 0]}}
    monkeypatch.setattr(morph, "for_actor", lambda extra: [table])
    monkeypatch.setattr(morph, "emotions", lambda: {"shy": {"controls": {"blush": .7}, "additive": True}})
    monkeypatch.setattr(morph, "poses", lambda: {})
    document = {"animations": []}
    assert poses.add_expressions(document, {}, skeleton, ["root"], {}, log=lambda _: None) == 0
    assert document["animations"] == []
    preset = document["expression_presets"][1]
    assert preset["controls"] == {"blush": .7} and preset["material_values"] == {"_Blush": .7}
    assert preset["additive"] is True


def test_endfield_existing_config_does_not_hide_missing_expression_tables(monkeypatch):
    from types import SimpleNamespace
    from polylode.endfield import actor, morph, poses
    skeleton = actor.build_skeleton({"root": ("", np.eye(4))}, "test")
    table = {"kind": 0, "bones": [], "base": [], "controls": {}, "params": {}}
    monkeypatch.setattr(morph, "for_actor", lambda extra: [table])
    monkeypatch.setattr(morph, "emotions", lambda: {})
    monkeypatch.setattr(morph, "poses", lambda: {})
    calls = []
    pipeline = SimpleNamespace(convert_morphs=lambda: calls.append(True) or SimpleNamespace(failed={}, unread={}))
    poses.add_expressions({"animations": []}, {}, skeleton, ["root"], {"model": "own"},
                          fetch=lambda: pipeline, log=lambda _: None)
    assert calls == [True]


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


def test_a_cutscene_clip_that_pulls_the_skeleton_apart_is_measured_by_its_skeleton_alone():
    """GF2's Upgrade_* push both shoulder blades a metre off the chest while their camera is on the face: seen whole,
    the body tears.  A crouch (the hips lowered under the root) and a prop sent away are not the skeleton's."""
    names = ["root", "Root_M", "Chest_M", "Scapula_L", "axe_skin", "Shoes01_R"]
    parents = [-1, 0, 1, 2, 2, 1]
    rest = np.stack([np.eye(4)] * 6)
    for bone, height in enumerate([0.0, 0.9, 1.3, 1.35, 1.4, 0.1]):
        rest[bone, 1, 3] = height
    frames = np.stack([rest] * 3).copy()
    frames[1, 1:, 1, 3] -= 0.6                                # crouched: everything from the hips down 60 cm
    frames[2, 4, 0, 3] = 100.0                                # the axe hidden 100 m away
    frames[2, 5, 0, 3] = 217.0                                # shoes the skeleton's naming, hidden the same way
    assert GP.TORN > G.strays(frames, rest, parents, names)[0]
    frames[2, 3, 2, 3] += 1.0                                 # the shoulder blade a metre off the chest
    moved, bone = G.strays(frames, rest, parents, names)
    assert np.isclose(moved, 1.0) and names[bone] == "Scapula_L" and moved > GP.TORN
    # a grown chest carries the blade with it: measured in the parent's own frame, not torn
    grown = np.stack([rest] * 2).copy()
    grown[1, 2, :3, :3] *= 1.5
    grown[1, 3] = grown[1, 2] @ np.linalg.solve(rest[2], rest[3])
    assert G.strays(grown, rest, parents, names)[0] < 1e-9


def test_a_bone_snapped_out_of_place_for_a_camera_is_put_back_and_its_moves_kept():
    """GF2's Love_4 pops Clukay SSR01's left shoulder 11 cm out of the shoulder blade from 2.4 s to 6.07 s (stepped
    keys, a camera trick): seen whole, her arm came off."""
    rest = np.array([0.0689, 0.0, 0.0])
    curve = np.tile(rest, (10, 1))
    curve[1, 0] += 0.004                                       # its own small moves are kept
    curve[3:7] = [0.1486, -0.0731, -0.0331]                    # snapped out for four frames
    fixed, held = G.unsnapped(curve, rest)
    assert held == 4 and np.allclose(fixed[3:7], rest) and np.allclose(fixed[[0, 1, 2, 7, 8, 9]], curve[[0, 1, 2, 7, 8, 9]])
    # out at the start and back later; out to the end
    start = np.tile(rest, (6, 1))
    start[:2] += [0.2, 0, 0]
    assert G.unsnapped(start, rest)[1] == 2
    end = np.tile(rest, (6, 1))
    end[4:] += [0, 0.3, 0]
    assert G.unsnapped(end, rest)[1] == 2
    # a part sent 100 km away is hidden, not snapped; a smooth reach of 30 cm is a move
    # out in two steps (Class_*: 0.33 m, then 0.54 m) and back: every step out put back
    steps = np.tile(rest, (8, 1))
    steps[2:4] += [0.325, 0, 0]
    steps[4:6] += [0.542, 0, 0]
    fixed, held = G.unsnapped(steps, rest)
    assert held == 4 and np.allclose(fixed, rest)
    # slid out (no jump) before it jumps: the slide is put back too, as far as it is more than SNAP off
    slide = np.tile(rest, (10, 1))
    slide[2:6] += np.array([0.03, 0.12, 0.2, 0.25])[:, None] * [1, 0, 0]
    slide[6:8] += [0.54, 0, 0]
    fixed, held = G.unsnapped(slide, rest)
    assert held == 5 and np.allclose(fixed[3:8], rest) and np.allclose(fixed[2], slide[2])
    hidden = np.tile(rest, (5, 1))
    hidden[2:] = [1e5, 0, 0]
    assert G.unsnapped(hidden, rest)[1] == 0
    reach = rest + np.linspace(0, 0.3, 30)[:, None] * [1, 0, 0]
    assert G.unsnapped(reach, rest)[1] == 0
