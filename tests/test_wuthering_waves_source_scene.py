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
