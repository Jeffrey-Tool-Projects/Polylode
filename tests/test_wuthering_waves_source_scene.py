"""The original clothed scene's numbers (``wuthering_waves/source_scene.py``): rest frames, socket placement, materials."""
import numpy as np
import pytest

from polylode.wuthering_waves import source_scene as S

BONES = [{"name": "root", "parent": -1}, {"name": "child", "parent": 0}]


def _world(scale=1.0, shear=0.0):
    world = np.stack([np.eye(4), np.eye(4)])
    world[1, :3, 3] = (0, 0, .5)
    world[1, :3, :3] = np.diag([scale, scale, scale])
    world[1, 0, 1] = shear
    return world


def test_a_uniform_rest_scale_bakes_to_rigid_frames_on_the_same_joints():
    world = _world(scale=1.25)
    rigid, proof, native, baked = S.bake_uniform(world, BONES)
    assert np.allclose(rigid[:, :3, :3], np.eye(3)) and np.allclose(rigid[:, :3, 3], world[:, :3, 3])
    assert proof["uniform_rest_scale_exact"] and proof["all_bone_rotation_skin_matrix_error"] < 1e-9
    points = np.array([[0, 0, .6], [0, .1, .4]])
    joints = np.array([[1], [1]])
    S.validate_skin_bound(proof, native, baked, points, joints, np.ones((2, 1), dtype=np.float32))
    assert proof["all_native_vertices_rotation_skin_error_mm"] <= S.SKIN_BOUND_MM


def test_a_sheared_rest_frame_that_moves_skinned_vertices_is_refused():
    rigid, proof, native, baked = S.bake_uniform(_world(shear=.3), BONES)
    with pytest.raises(ValueError, match="move a vertex"):
        S.validate_skin_bound(proof, native, baked, np.array([[0., .2, .6]]), np.array([[1]]),
                              np.ones((1, 1), dtype=np.float32))


def test_a_mirrored_socket_placement_flips_the_winding_and_the_tangent_sign():
    arrays = {"p0_co": np.array([[1., 0, 0]], dtype=np.float32), "p0_no": np.array([[0., 1, 0]], dtype=np.float32),
              "p0_tan": np.array([[1., 0, 0, 1]], dtype=np.float32), "p0_tri": np.array([[0, 1, 2]]),
              "p0_s0_delta": np.array([[0., 1, 0]], dtype=np.float32)}
    placement = np.diag([1., -1., 1., 1.])
    placement[:3, 3] = (0, 0, 2)
    moved = S.place_streams(arrays, placement)
    assert np.allclose(moved["p0_co"], [[1, 0, 2]]) and np.allclose(moved["p0_no"], [[0, -1, 0]])
    assert moved["p0_tri"].tolist() == [[0, 2, 1]] and moved["p0_tan"][0, 3] == -1
    assert np.allclose(moved["p0_s0_delta"], [[0, -1, 0]])


def test_a_parts_material_with_other_maps_under_a_taken_name_gets_its_own():
    record = lambda digest: {"native_material": {"x": 1}, "maps": {"MainTex": {"sha256": digest, "srgb": True}},
                             "source_package": f"/Game/{digest}"}
    document = {"materials": {"M": record("a")}}
    part = {"materials": ["M"]}
    S.merge_materials(document, {"materials": {"M": record("b")}}, part)
    assert part["materials"][0] != "M" and part["materials"][0].startswith("M_") and len(document["materials"]) == 2
    same = {"materials": ["M"]}
    S.merge_materials(document, {"materials": {"M": record("a")}}, same)
    assert same["materials"] == ["M"] and len(document["materials"]) == 2


def test_the_biped_must_chain_its_limbs():
    names = ["Bip001Pelvis", "Bip001Spine", "Bip001Spine1", "Bip001Spine2", "Bip001Neck", "Bip001Head"]
    for prefix in ("Bip001L", "Bip001R"):
        names += [prefix + n for n in ("Clavicle", "UpperArm", "Forearm", "Hand", "Thigh", "Calf", "Foot", "Toe0")]
    bones = [{"name": name, "parent": index - 1} for index, name in enumerate(names)]
    assert S.humanoid(set(names), bones)["LeftFoot"] == "Bip001LFoot"
    foot = names.index("Bip001LFoot")
    bones[foot]["parent"] = names.index("Bip001LThigh")       # the foot under the thigh, the calf skipped
    with pytest.raises(ValueError, match="is under"):
        S.humanoid(set(names), bones)
    with pytest.raises(ValueError, match="not the native biped"):
        S.humanoid({"Bip001Pelvis"}, bones)


def test_a_part_on_a_subset_of_the_primarys_bones_follows_them_by_name():
    # Lili's bottle (ELiliPingziMd10011): 18 of her 367 bones, matched by name as a leader pose matches them
    primary = [{"name": "root", "parent": -1}, {"name": "spine", "parent": 0}, {"name": "arm", "parent": 1},
               {"name": "twist", "parent": 2}, {"name": "flap", "parent": 3}]
    rests = np.stack([np.eye(4)] * 5)
    rests[:, :3, 3] = [(0, 0, 0), (0, 0, .5), (.2, 0, .6), (.3, 0, .6), (.35, 0, .6)]
    part = [{"name": "root", "parent": -1}, {"name": "arm", "parent": 0}, {"name": "flap", "parent": 1}]
    remap, proof = S.leader_subset("bottle", part, rests[[0, 2, 4]], primary, rests)
    assert remap.tolist() == [0, 2, 4] and proof["leader_pose_bones"] == 3 and proof["joint_offset_mm"] == 0
    with pytest.raises(ValueError, match="not on the primary's skeleton"):        # a bone she does not have
        S.leader_subset("bottle", part + [{"name": "lid", "parent": 2}], rests[[0, 2, 4, 4]], primary, rests)
    with pytest.raises(ValueError, match="ancestors"):                            # flap is not under spine
        S.leader_subset("bottle", [{"name": "flap", "parent": -1}, {"name": "spine", "parent": 0}],
                        rests[[4, 1]], primary, rests)
    moved = rests[[0, 2, 4]].copy()
    moved[1, 0, 3] += .001
    with pytest.raises(ValueError, match="mm off"):
        S.leader_subset("bottle", part, moved, primary, rests)
    turned = rests[[0, 2, 4]].copy()
    turned[2, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="rest frames differ"):
        S.leader_subset("bottle", part, turned, primary, rests)


def test_a_liquid_draws_its_middle_band_and_keeps_the_gradient(tmp_path):
    # Lili's bottle liquid: its diffuse slots hold the engine's white default, its colour is the WaterColor_* bands
    import json
    colour = lambda r, g, b: {"R": r, "G": g, "B": b, "A": 1.0}                 # noqa: E731
    material = {"BlendMode": 1, "Textures": {}, "Properties": {},
                "Colors": {"WaterColor_Down": colour(.85, .77, .09), "WaterColor_Middle": colour(.9, .6, .15),
                           "WaterColor_High": colour(.98, .6, .02), "FresnelColorAdd": colour(1, .92, 0)},
                "Scalars": {"WaterColor_MiddleProcess": .57, "Water_HighWidth": 1.0, "Roughness": .05}}
    (tmp_path / "Pkg").mkdir()
    (tmp_path / "Pkg" / "MI_Liquid.json").write_text(json.dumps(material), encoding="utf-8")
    (tmp_path / "texture_metadata.json").write_text("[]", encoding="utf-8")
    record = S.material_record({"Material": {"ObjectPath": "Pkg/MI_Liquid.0"}}, tmp_path, tmp_path, True)
    assert record["profile"] == "authored_liquid_color_preview" and record["constant_color"] == [.9, .6, .15, 1.0]
    assert record["liquid"]["drawn"] == "WaterColor_Middle" and set(record["liquid"]["colors"]) ==         {"WaterColor_Down", "WaterColor_Middle", "WaterColor_High"}
    assert record["liquid"]["placing"] == {"WaterColor_MiddleProcess": .57, "Water_HighWidth": 1.0}
    del material["Colors"]["WaterColor_High"]                                    # not the liquid's bands: refused
    (tmp_path / "Pkg" / "MI_Liquid.json").write_text(json.dumps(material), encoding="utf-8")
    with pytest.raises(ValueError, match="no verified base colour binding"):
        S.material_record({"Material": {"ObjectPath": "Pkg/MI_Liquid.0"}}, tmp_path, tmp_path, True)

