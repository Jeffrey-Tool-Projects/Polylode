"""The sidecar's ``physics`` section (``docs/sidecar_physics.md``, IslandLife §7.1512): Endfield's bone cloth and
Wuthering Waves' KawaiiPhysics nodes said against the FBX's bones."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from polylode.endfield import physics as EP
from polylode.endfield.rigscene import Bone, rest_matrix, world_matrices
from polylode.interchange import sidecar
from polylode.wuthering_waves import kawaii


def _translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _endfield_item():
    # root → Bip001_Pelvis (Hips) → Bip001_L_Thigh (LeftUpperLeg); pelvis → skirt_0 → skirt_1; root → MC_Skirt
    nodes = {"root": ("", np.eye(4)),
             "Bip001_Pelvis": ("root", _translate(0, 1.0, 0)),
             "Bip001_L_Thigh": ("Bip001_Pelvis", _translate(-0.1, 0, 0)),
             "skirt_0": ("Bip001_Pelvis", _translate(0.1, 0, 0.1)),
             "skirt_1": ("skirt_0", _translate(0, -0.2, 0)),
             "MC_Skirt": ("root", np.eye(4))}
    worlds = world_matrices(nodes)
    kept = ["Bip001_Pelvis", "Bip001_L_Thigh", "skirt_0", "skirt_1"]
    bones = [Bone(n, n, -1, rest_matrix(worlds[n]), 0.1) for n in kept]
    capsule = {"id": "7", "node": "Bip001_L_Thigh", "enabled": True, "center": {"x": 0.0, "y": -0.2, "z": 0.0},
               "size": {"x": 0.05, "y": 0.04, "z": 0.4}, "direction": 1, "reverseDirection": 0,
               "radiusSeparation": 1, "alignedOnCenter": 1}
    on_skirt = {"id": "8", "node": "skirt_1", "enabled": True, "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                "size": {"x": 0.03, "y": 0.0, "z": 0.0}}
    cloth = {"id": "1", "node": "MC_Skirt", "enabled": True,
             "serializeData": {"clothType": 1, "connectionMode": 0, "rootBones": [{"node": "skirt_0"}],
                               "ignoreFromRootBones": [], "gravity": 5.0,
                               "gravityDirection": {"x": 0.0, "y": -1.0, "z": 0.0},
                               "inertiaConstraint": {"anchor": {"node": "Bip001_L_Thigh"}},
                               "colliderCollisionConstraint": {"mode": 1, "colliderList": [
                                   {"component": "7", "node": "Bip001_L_Thigh"}, None,
                                   {"component": "8", "node": "skirt_1"}]}},
             "selectionData": {"positions": [{"x": -0.1, "y": 1.0, "z": 0.1}, {"x": -0.1, "y": 0.8, "z": 0.1}],
                               "attributes": [{"Value": 1}, {"Value": 2}], "maxConnectionDistance": 0.01}}
    extra = {"components": {"BeyondBoneCloth": [cloth], "BeyondBoneCapsuleCollider": [capsule],
                            "BeyondBoneSphereCollider": [on_skirt]},
             "humanoid": {"Hips": "Bip001_Pelvis", "LeftUpperLeg": "Bip001_L_Thigh"}}
    return SimpleNamespace(bones=bones), nodes, extra, worlds


def test_endfield_chain_rows_name_fbx_bones_and_resolve_the_painting():
    scene, nodes, extra, _worlds = _endfield_item()
    out = EP.physics(scene, nodes, extra)
    assert out["status"] == "read" and out["system"] == EP.SYSTEM
    chain, = out["chains"]
    assert chain["roots"] == ["skirt_0"] and chain["bones"] == ["skirt_0", "skirt_1"]
    # the cloth node is a child of the root, not a bone: the model's root, in the FBX's world
    assert chain["frame"]["bone"] is None and np.allclose(chain["frame"]["local"], np.eye(4))
    assert chain["colliders"] == ["7", "8"] and chain["empty_slots"] == 1
    assert chain["states"] == [{"node": "skirt_0", "state": "fixed"}, {"node": "skirt_1", "state": "move"}]
    assert chain["params"]["inertiaConstraint"]["anchor"] == {"node": "Bip001_L_Thigh"}
    assert "colliderList" not in chain["params"]["colliderCollisionConstraint"]
    assert np.allclose(chain["resolved"]["gravity_world"], [0, -5, 0])


def test_endfield_collider_geometry_lands_where_the_game_puts_it():
    scene, nodes, extra, worlds = _endfield_item()
    out = EP.physics(scene, nodes, extra)
    rows = {row["id"]: row for row in out["colliders"]}
    thigh = rows["7"]
    assert thigh["kind"] == "capsule" and thigh["bone"] == "Bip001_L_Thigh"
    assert thigh["radius"] == [0.05, 0.04]
    assert thigh["host"] == {"humanoid": "LeftUpperLeg", "on_chain": False}
    assert rows["8"]["host"] == {"humanoid": "Hips", "on_chain": True}
    # bone-local in the FBX's frame, back to the world: the centre is 0.2 below the thigh joint (y −0.2,
    # Unity's x mirrored), the axis along y, reach z − both radii
    bone = next(b for b in scene.bones if b.name == "Bip001_L_Thigh")
    from polylode.endfield.rigscene import TO_Z_UP
    to_world = np.linalg.inv(TO_Z_UP) @ bone.rest
    a = (to_world @ np.append(thigh["start"], 1))[:3]
    b = (to_world @ np.append(thigh["end"], 1))[:3]
    joint = worlds["Bip001_L_Thigh"][:3, 3]
    assert np.allclose((a + b) / 2, joint + [0, -0.2, 0], atol=1e-6)
    assert math.isclose(np.linalg.norm(a - b), 0.4 - 0.05 - 0.04, abs_tol=1e-6)


def test_endfield_item_without_cloth_is_read_and_empty():
    scene, nodes, extra, _worlds = _endfield_item()
    out = EP.physics(scene, nodes, {"components": {}, "humanoid": extra["humanoid"]})
    assert out["status"] == "read" and out["chains"] == [] and "reason" in out


def test_sidecar_without_a_physics_reading_says_unread():
    assert sidecar.physics({})["status"] == "unread"
    assert sidecar.physics({"physics": {"status": "read", "chains": []}})["status"] == "read"


def test_rotator_quaternion_is_unreals():
    assert np.allclose(kawaii.rotator_quat({"Pitch": 0, "Yaw": 0, "Roll": 0}), [0, 0, 0, 1])
    # a yaw of 90° turns +X to +Y about +Z
    from scipy.spatial.transform import Rotation
    q = kawaii.rotator_quat({"Pitch": 0, "Yaw": 90, "Roll": 0})
    assert np.allclose(Rotation.from_quat(q).apply([1, 0, 0]), [0, 1, 0], atol=1e-9)


def _wuwa_model():
    # Unreal centimetres: pelvis at z 90, thigh 10 cm to the side, its +X pointing down the leg
    turn = [0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]          # +X → −Z about +Y
    source = [{"name": "Bip001Pelvis", "parent": -1, "translation": [0, 0, 90], "rotation_xyzw": [0, 0, 0, 1],
               "scale": [1, 1, 1]},
              {"name": "Bip001LThigh", "parent": 0, "translation": [0, 10, 0], "rotation_xyzw": turn,
               "scale": [1, 1, 1]},
              {"name": "Bone_Skirt001", "parent": 0, "translation": [5, 10, 0], "rotation_xyzw": turn,
               "scale": [1, 1, 1]},
              {"name": "Bone_Skirt002", "parent": 2, "translation": [20, 0, 0], "rotation_xyzw": [0, 0, 0, 1],
               "scale": [1, 1, 1]},
              {"name": "Bone_Skirt003", "parent": 3, "translation": [20, 0, 0], "rotation_xyzw": [0, 0, 0, 1],
               "scale": [1, 1, 1]}]
    from polylode.wuthering_waves.source_scene import native_world
    rest = native_world(source)
    document = {"source_bones": source, "bones": [{"name": b["name"], "parent": b["parent"]} for b in source],
                "humanoid": {"Hips": "Bip001Pelvis", "LeftUpperLeg": "Bip001LThigh"}}
    node = {"RootBones": [{"BoneName": "Bone_Skirt001"}], "ExcludeBones": [{"BoneName": "Bone_Skirt003"}],
            "PhysicsSettings": {"Damping": 0.2, "Stiffness": 0.2, "Radius": 3.0, "LimitAngle": 35.0,
                                "Shape": "EPhyCollisionShape::Planar", "HalfWidth": 4.0},
            "DummyBoneLength": 10.0, "BoneForwardAxis": "EBoneForwardAxis::Y_Positive",
            "Gravity": {"X": 0.0, "Y": 0.0, "Z": -980.0}, "TeleportDistanceThreshold": 300.0,
            "AlphaInputType": "EAnimAlphaInputType::Curve", "Alpha": 0.5, "AlphaCurveName": "Kawaii_alpha_x",
            "KuroLODMask": 1, "StiffnessCurve": {"ObjectPath": "Client/Content/C/kawaii_S.0"},
            "CapsuleLimits": [{"Radius": 7.0, "Length": 20.0, "DrivingBone": {"BoneName": "Bip001LThigh"},
                               "OffsetLocation": {"X": 11.0, "Y": 0.0, "Z": 0.0},
                               "OffsetRotation": {"Pitch": -90.0, "Yaw": 0.0, "Roll": 0.0}}],
            "SphericalLimits": [], "PlanarLimits": []}
    return document, rest, node


def test_kawaii_limit_rides_its_driving_bone_offset_in_its_frame():
    document, rest, node = _wuwa_model()
    out = kawaii.physics(document, rest, "ABP_test", [("AnimGraphNode_KawaiiPhysicsGroup", node)],
                         {"Client/Content/C/kawaii_S.0": [{"Time": 0.0, "Value": 1.0}]})
    capsule, = out["colliders"]
    assert capsule["bone"] == "Bip001LThigh" and capsule["host"] == {"humanoid": "LeftUpperLeg", "on_chain": False}
    thigh = rest[1]
    a, b = ((thigh @ np.append(capsule[k], 1))[:3] for k in ("start", "end"))
    # 11 cm down the thigh (its +X points down), not above the hip
    centre = (a + b) / 2
    assert np.allclose(centre, thigh[:3, 3] + [0, 0, -0.11], atol=1e-6)
    assert math.isclose(np.linalg.norm(a - b), 0.20, abs_tol=1e-6) and capsule["radius"] == [0.07, 0.07]


def test_kawaii_chain_alpha_dummy_and_unread_fields():
    document, rest, node = _wuwa_model()
    out = kawaii.physics(document, rest, "ABP_test", [("N", node)], {"Client/Content/C/kawaii_S.0": None})
    chain, = out["chains"]
    assert chain["bones"] == ["Bone_Skirt001", "Bone_Skirt002"]        # the excluded bone ends the branch
    resolved = chain["resolved"]
    assert resolved["alpha"] is None                                  # an animation curve drives it
    assert resolved["radius_m"] == pytest.approx(0.03)
    assert resolved["dummy"] == {"length_m": pytest.approx(0.1), "forward": [0.0, -1.0, 0.0]}
    assert np.allclose(resolved["gravity_world"], [0, -9.8, 0])
    assert resolved["curves"]["StiffnessCurve"]["keys"] is None
    assert out["unread"]["N"]["PhysicsSettings.Shape"] == "EPhyCollisionShape::Planar"
    assert out["unread"]["N"]["AlphaCurveName"] == "Kawaii_alpha_x"


def test_kawaii_refuses_a_driving_bone_the_model_lacks():
    document, rest, node = _wuwa_model()
    node["CapsuleLimits"][0]["DrivingBone"]["BoneName"] = "Bip001Missing"
    with pytest.raises(kawaii.KawaiiError):
        kawaii.physics(document, rest, "ABP_test", [("N", node)], {})
