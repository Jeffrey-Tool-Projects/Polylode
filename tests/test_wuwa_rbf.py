"""Wuthering Waves' pose-space correctives in the FBX sidecar (``wuthering_waves.rbf``, ``docs/sidecar_rbf.md``)."""
from __future__ import annotations

import numpy as np
import pytest

from polylode.interchange import sidecar
from polylode.wuthering_waves import rbf
from polylode.wuthering_waves.source_scene import native_world

# a clavicle-like driver: rest along X, one raised target, the Y column a float-noise zero as the game's is
CLAVICLE = {"object": "RBFDriver_0", "name": "weightClavicle_L", "driver_bone": "Arm", "base_bone": "Clav",
            "attributes": list(rbf.ATTRIBUTES), "morphs": ["Up_U"],
            "targets": [{"input": [8.5, 4e-6, 7e-6], "weights": {"Up_U": 0.0}},
                        {"input": [6.0104, -9e-6, 6.0104], "weights": {"Up_U": 1.0}}],
            "weight_modifier": None}
# an elbow-like driver: several targets around a sphere, weights per morph
ARM = {"object": "RBFDriver_1", "name": "weightArm_L", "driver_bone": "Fore", "base_bone": "Arm",
       "attributes": list(rbf.ATTRIBUTES), "morphs": ["P", "F", "B"],
       "targets": [{"input": [20.0, 0.0, 0.0], "weights": {"P": 0.0, "F": 0.0, "B": 0.0}},
                   {"input": [11.47, 0.0, 16.38], "weights": {"P": 1.0, "F": 0.0, "B": 0.0}},
                   {"input": [3.47, 19.70, 0.0], "weights": {"P": 0.0, "F": 1.0, "B": 0.0}},
                   {"input": [0.0, -20.0, 0.0], "weights": {"P": 0.0, "F": 0.0, "B": 1.0}},
                   {"input": [12.86, 15.32, 0.0], "weights": {"P": 0.0, "F": 0.3, "B": 0.0}}],
       "weight_modifier": {"SrcWeight": "P", "DstDriverIndex": 0, "DstWeight": "Up_U",
                           "FactorCurve": {"EditorCurveData": {"Keys": [
                               {"Time": 0.0, "Value": 0.0, "InterpMode": "RCIM_Linear"},
                               {"Time": 1.0, "Value": 1.0, "InterpMode": "RCIM_Linear"}]}}}}


def _inputs(driver):
    return (np.array([t["input"] for t in driver["targets"]]),
            np.array([[t["weights"][m] for m in driver["morphs"]] for t in driver["targets"]]))


def _table(driver, limit_xy=60.0, limit_z=60.0, interval=4.0):
    radius = float(np.linalg.norm(driver["targets"][0]["input"]))
    table = {"interval": interval, "limit_xy": limit_xy, "limit_z": limit_z, "radius": radius,
             "lookup_xy": list(np.arange(-limit_xy, limit_xy + 1e-9, interval)),
             "lookup_z": list(np.arange(-limit_z, limit_z + 1e-9, interval))}
    table["weights"] = rbf.solve(*_inputs(driver), rbf.directions(table))
    return table


def test_solver_interpolates_its_targets_and_clips():
    for driver in (CLAVICLE, ARM):
        inputs, weights = _inputs(driver)
        assert np.abs(rbf.solve(inputs, weights, inputs) - weights).max() < 1e-9
    inputs, weights = _inputs(ARM)
    far = rbf.solve(inputs, weights, np.array([[-14.0, -14.0, 0.0], [0.0, 0.0, -20.0]]))
    assert far.min() >= 0.0 and far.max() <= 1.0


def test_a_noise_column_is_divided_by_one():
    # the clavicle's Y column is float noise around zero: divided by its own norm it would dominate the distance
    inputs, weights = _inputs(CLAVICLE)
    clean = inputs.copy()
    clean[:, 1] = 0.0
    probe = np.array([[8.0, 1.0, 2.0], [7.0, -0.5, 4.0]])
    assert np.allclose(rbf.solve(inputs, weights, probe), rbf.solve(clean, weights, probe), atol=1e-5)


def test_directions_put_node_ij_at_azimuth_and_elevation():
    table = _table(ARM, 160.0, 88.0)
    points = rbf.directions(table)
    i, j = 60, 22                       # azimuth -160 + 240 = 80, elevation -88 + 88 = 0
    assert np.allclose(points[i, j], 20.0 * np.array([np.cos(np.radians(80)), np.sin(np.radians(80)), 0.0]))
    i, j = 40, 36                       # azimuth 0, elevation 56
    assert np.allclose(points[i, j], 20.0 * np.array([np.cos(np.radians(56)), 0.0, np.sin(np.radians(56))]))


def _package(table, name):
    xy, z = table["lookup_xy"], table["lookup_z"]
    lookup = [{"Key": {"IndexXY": a, "IndexZ": b}, "Value": a * len(z) + b} for a in range(len(xy)) for b in range(len(z))]
    weights = [{"Data": list(map(float, row))} for row in table["weights"].reshape(len(xy) * len(z), -1)]
    return [{"Type": "RBFPrecomputations", "Properties": {"Precomputations": [{"Key": name, "Value": {
        "RotationLimitX": 0.0, "RotationLimitY": table["limit_xy"], "RotationLimitZ": table["limit_z"],
        "Interval": table["interval"], "PrecomputeRadius": table["radius"], "LookupAngleXY": xy,
        "LookupAngleZ": z, "WeightLookupTable": lookup, "PreWeights": weights}}]}}]


def test_tables_read_back_and_refuse_another_order():
    table = _table(CLAVICLE)
    read = rbf.tables(_package(table, "weightClavicle_L"))["weightClavicle_L"]
    assert np.allclose(read["weights"], table["weights"])
    swapped = _package(table, "weightClavicle_L")
    rows = swapped[0]["Properties"]["Precomputations"][0]["Value"]["WeightLookupTable"]
    rows[0]["Value"], rows[1]["Value"] = rows[1]["Value"], rows[0]["Value"]
    with pytest.raises(rbf.RbfError):
        rbf.tables(swapped)
    twisted = _package(table, "weightClavicle_L")
    twisted[0]["Properties"]["Precomputations"][0]["Value"]["RotationLimitX"] = 30.0
    with pytest.raises(rbf.RbfError):
        rbf.tables(twisted)


def _document(rest_cm=(8.5, 4e-6, 7e-6)):
    turn = [0.15424459, -0.47032696, -0.08061865, -0.86516]      # a base bone that is not at identity
    source = [{"name": "Root", "parent": -1, "rotation_xyzw": [0, 0, 0, 1], "translation": [0, 0, 100.0], "scale": [1, 1, 1]},
              {"name": "Clav", "parent": 0, "rotation_xyzw": turn, "translation": [10.0, 2.0, 1.0], "scale": [1, 1, 1]},
              {"name": "Arm", "parent": 1, "rotation_xyzw": [0, 0, 0, 1], "translation": list(rest_cm), "scale": [1, 1, 1]},
              {"name": "Fore", "parent": 2, "rotation_xyzw": [0, 0, 0, 1], "translation": [20.0, 0, 0], "scale": [1, 1, 1]}]
    document = {"bones": [{"name": b["name"], "parent": b["parent"]} for b in source], "source_bones": source,
                "parts": [{"name": "Up", "shapes": ["Up_U", "P", "F"]}, {"name": "Down", "shapes": ["Up_U"]}]}
    return document, native_world(source)


def test_driver_row_says_targets_in_the_fbx_frame_and_checks_the_rest():
    document, rest = _document()
    table = _table(CLAVICLE)
    row = rbf.driver_row(CLAVICLE, table, "Pre.uasset", document, rest, {0: "weightClavicle_L"})
    assert row["rest_translation"]["gap_cm"] < 1e-6 and row["rest_translation"]["base_frame_gap_deg"] < 1e-6
    assert row["targets"][1]["fbx_m"] == pytest.approx([0.060104, 9e-8, 0.060104])     # Y mirrored, cm → m
    assert row["rest_translation"]["scene_fbx_m"] == pytest.approx(row["rest_translation"]["fbx_m"], abs=1e-9)
    assert row["outputs"]["parts"] == {"Up_U": ["Up", "Down"]}
    assert row["table"]["matches_targets"] and row["table"]["max_error"] < 1e-9
    assert "problems" not in row
    moved, _ = _document((8.0, 0.0, 0.0))
    assert rbf.driver_row(CLAVICLE, table, "Pre.uasset", moved, _document((8.0, 0.0, 0.0))[1], {})["problems"]


def test_a_table_of_other_targets_is_flagged():
    document, rest = _document()
    other = dict(CLAVICLE, targets=[CLAVICLE["targets"][0],
                                    {"input": [6.0104, -1.0, 6.0104], "weights": {"Up_U": 1.0}}])
    row = rbf.driver_row(CLAVICLE, _table(other), "Pre.uasset", document, rest, {})
    assert not row["table"]["matches_targets"] and row["table"]["max_error"] > 0.01


def test_weight_modifier_is_written_as_stored():
    row = rbf._modifier(ARM["weight_modifier"], {0: "weightClavicle_L"})
    assert row["target_driver"] == "weightClavicle_L" and row["semantics"] is None
    assert [k["value"] for k in row["factor_curve"]] == [0.0, 1.0]


def _node(scale=0.5):
    return {"SourceBones": [{"BoneName": "Clav"}], "OnlyDriveBones": [{"BoneName": "Fore"}],
            "PoseTargets": [{"BoneTransforms": [{"TargetRotation": {"Pitch": 0.0, "Yaw": 30.0, "Roll": 0.0}}],
                             "TargetScale": scale, "DrivenName": "Pose_1"}],
            "PoseDriverOffsetType": "EPoseDriverOffsetType::KuroCachedBone", "RBFParams": {"Radius": 23.0},
            "DriveSource": "EPoseDriverSource::Rotation", "DriveOutput": "EPoseDriverOutput::DrivePoses",
            "bOnlyDriveSelectedBones": True, "PoseAsset": {"ObjectPath": "Game/PD.0"}}


def test_pose_driver_row_mirrors_rotations_and_keeps_the_driven_bones():
    document, _rest = _document()
    asset = {"additive": True, "poses": {"Pose_1": {
        "Fore": {"rotation": [0.0, 0.0, 0.2588190, 0.9659258], "translation": [1.0, 2.0, 3.0], "scale": [0, 0, 0]},
        "Arm": {"rotation": [0.1, 0.0, 0.0, 0.99498744], "translation": [0, 0, 0], "scale": [0, 0, 0]}}}}
    row = rbf.pose_driver_row("AnimGraphNode_PoseDriver", _node(), asset, "Game/PD.uasset", document)
    target = row["targets"][0]
    assert target["quat_game"][0] == pytest.approx([0.0, 0.0, 0.2588190, 0.9659258], abs=1e-6)
    assert target["quat_fbx"][0] == pytest.approx([0.0, 0.0, -0.2588190, 0.9659258], abs=1e-6)
    assert target["scale"] == 0.5
    pose = row["pose_asset"]["poses"]["Pose_1"]
    assert sorted(pose) == ["Fore"]                         # only the driven bones (and the sources)
    assert pose["Fore"]["translation_fbx_m"] == pytest.approx([0.01, -0.02, 0.03])
    with pytest.raises(rbf.RbfError):
        rbf.pose_driver_row("n", dict(_node(), SourceBones=[{"BoneName": "Nope"}]), asset, None, document)


def test_sidecar_without_rbf_says_unread():
    assert sidecar.rbf({})["status"] == "unread"
    assert sidecar.rbf({"rbf": {"format": 1, "status": "read", "drivers": [], "pose_drivers": []}})["status"] == "read"
