"""The FBX's sidecar (``interchange/sidecar.py``) and the sources' statements for it (``extractor/identity.py``)."""
import hashlib
import json

import numpy as np
import pytest

from polylode.interchange import fbx, sidecar
from test_interchange_glb import PNG, _scene


def _written(tmp_path, document, arrays, **statement):
    np.savez(tmp_path / "x.npz", **arrays)
    (tmp_path / "scene.json").write_text(json.dumps(document), encoding="utf-8")
    report, problems = fbx.write_checked(tmp_path / "scene.json", tmp_path / "out" / "model.fbx")
    assert not problems
    path = sidecar.write(tmp_path / "scene.json", tmp_path / "out" / "model.fbx", report,
                         identity=statement.get("identity", {"game": "test"}),
                         humanoid_statement=statement.get("humanoid", {"basis": "unity_avatar"}))
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_the_sidecar_names_its_schema_its_model_and_where_each_map_is(tmp_path):
    (tmp_path / "out").mkdir()
    document, arrays = _scene(tmp_path)
    path, data = _written(tmp_path, document, arrays)
    assert path.name == "model.islandlife.json" and (data["schema"], data["version"]) == (sidecar.SCHEMA, 1)
    model = tmp_path / "out" / "model.fbx"
    assert data["model"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    maps = data["materials"]["skin"]["maps"]
    # the base map is inside the FBX: named by its slot, not copied again
    assert maps["_MainTex"]["in_fbx"] == ["DiffuseColor"] and maps["_MainTex"]["file"] is None
    assert maps["_MainTex"]["scale"] == [2.0, 1.0] and maps["_MainTex"]["offset"] == [0.0, 0.5]
    # every other map is copied once beside the model, named by its content, with its colour space
    digest = hashlib.sha256(PNG).hexdigest()
    assert maps["_Normal"] == {"source_name": "base.png", "color_space": "linear", "scale": [1.0, 1.0],
                               "offset": [0.0, 0.0], "sha256": digest, "file": f"textures/{digest}.png"}
    assert (tmp_path / "out" / "textures" / f"{digest}.png").read_bytes() == PNG
    assert data["materials"]["skin"]["fbx"]["DiffuseColor"] == "_MainTex"
    assert data["materials"]["skin"]["native"]["floats"] == {"_Cutoff": .5}
    # each slot's triangles as the source has them (the degenerate one included), and a role it cannot decide
    slots = data["parts"][0]["slots"]
    assert [(s["material"], s["triangles"], s["role"]) for s in slots] == [("skin", 1, "unknown"),
                                                                          ("cloth", 2, "unknown")]


def test_a_map_of_a_material_no_triangle_draws_is_listed_and_not_copied(tmp_path):
    (tmp_path / "out").mkdir()
    document, arrays = _scene(tmp_path)
    arrays["p0_slot"] = np.array([1, 1, 1])
    _path, data = _written(tmp_path, document, arrays)
    normal = data["materials"]["skin"]["maps"]["_Normal"]
    assert normal["file"] is None and "no triangle" in normal["skipped"] and normal["sha256"]
    assert not (tmp_path / "out" / "textures").exists()


def test_the_human_bones_say_how_complete_they_are_and_refuse_a_bone_the_scene_has_not(tmp_path):
    (tmp_path / "out").mkdir()
    document, arrays = _scene(tmp_path)
    _path, data = _written(tmp_path, document, arrays)
    human = data["humanoid"]
    assert human["status"] == "partial" and human["bones"] == {"Hips": "root"}
    assert human["basis"] == "unity_avatar" and "Head" in human["required_missing"]
    document["humanoid"] = {}
    _path, data = _written(tmp_path, document, arrays, humanoid={"basis": "unity_avatar", "reason": "Generic"})
    assert data["humanoid"]["status"] == "missing" and data["humanoid"]["basis"] is None
    assert data["humanoid"]["reason"] == "Generic"
    document["humanoid"] = {"Hips": "pelvis"}
    with pytest.raises(sidecar.SidecarError, match="pelvis"):
        _written(tmp_path, document, arrays)


def test_bone_expressions_are_local_transforms_in_the_files_space_and_single_shapes_stay_in_the_fbx(tmp_path):
    (tmp_path / "out").mkdir()
    document, arrays = _scene(tmp_path)
    turned = np.array(arrays["bone_rest"], dtype=np.float64)
    turned[1, :3, 3] += (0.0, 0.0, 0.01)                         # the child 1 cm up (Z up in the scene)
    document["animations"] = [
        {"name": "expression/control/jaw", "clip": "control/jaw", "kind": "expression", "frames": 2,
         "controls": {"jaw": 1.0}, "material_values": {"_EmotionIndex": 2.0}},
        {"name": "expression/smile", "clip": "smile", "kind": "expression", "frames": 2}]
    arrays["anim0"] = np.repeat(turned[None], 2, axis=0)
    arrays["anim1"] = np.repeat(arrays["bone_rest"][None], 2, axis=0)
    arrays["p0_anim1_weights"] = np.array([[1.0], [1.0]])
    _path, data = _written(tmp_path, document, arrays)
    shapes = data["expressions"]["shape_keys"]["shapes"]
    poses = data["expressions"]["bone_poses"]["poses"]
    assert shapes == {"smile": ["body"]} and [p["name"] for p in poses] == ["expression/control/jaw"]
    assert set(poses[0]["bones"]) == {"spine"} and poses[0]["material_values"] == {"_EmotionIndex": 2.0}
    # the FBX's Lcl of the child at rest, moved 1 cm along its parent's local Z (the root is the scene turned Y up)
    rest_local = np.linalg.solve(arrays["bone_rest"][0], arrays["bone_rest"][1])
    moved = np.asarray(poses[0]["bones"]["spine"])
    np.testing.assert_allclose(moved[:3, :3], rest_local[:3, :3], atol=1e-7)
    np.testing.assert_allclose(moved[:, 3] - rest_local[:3, 3], [0.0, 0.0, 0.01], atol=1e-7)


def test_slot_roles_come_from_each_games_shader_data():
    role = lambda record: sidecar.slot_role(record)[0]                 # noqa: E731
    assert role({"shader": "HGRP/CharacterNPR_Skin", "keywords": ["_SDFLIGHTMAP"]}) == "face"
    assert role({"shader": "HGRP/CharacterNPR_Skin", "keywords": []}) == "body_skin"
    assert role({"shader": "HGRP/CharacterNPR_Hair"}) == "hair"
    assert role({"shader": "HGRP/CharacterNPR"}) == "clothing"
    assert role({"shader": "HGRP/Effect/VFXTransparentDepthOnly"}) == "effect"
    assert role({"shader": "HGRP/Lit"}) == "unknown"
    assert role({"shader": "gf_shader/pbr/character/eye"}) == "eye"
    assert role({"shader": "gf_shader/pbr/character/uber"}) == "unknown"
    wuwa = lambda model, **switches: {"native_material": {"Switches": switches, "Properties": {      # noqa: E731
        "BasePropertyOverrides": {"ShadingModel": model}}}}
    assert role(wuwa("MSM_ToonHair")) == "hair" and role(wuwa("MSM_ToonOutline")) == "outline"
    assert role(wuwa("MSM_ToonFace")) == "face" and role(wuwa("MSM_ToonFace", UseEyeSDF=True)) == "eye"
    assert role(wuwa("MSM_ToonCommon", UseSkinMask=True)) == "skin_and_clothing"
    assert role(wuwa("MSM_ToonCommon")) == "unknown"
    assert role({}) == "unknown"


def test_writing_an_fbx_writes_its_sidecar_and_says_so(tmp_path):
    from polylode.extractor.sources import write_own
    document, arrays = _scene(tmp_path)
    np.savez(tmp_path / "x.npz", **arrays)
    (tmp_path / "scene.json").write_text(json.dumps(document), encoding="utf-8")
    log = []
    files = write_own(["fbx"], tmp_path / "scene.json", tmp_path / "out", "model", None, log.append,
                      sidecar={"identity": {"game": "test", "polylode_ident": "m"}})
    assert [f.name for f in files] == ["model.fbx", "model.islandlife.json"]
    data = json.loads(files[1].read_text(encoding="utf-8"))
    assert data["identity"] == {"game": "test", "polylode_ident": "m"} and data["humanoid"]["basis"] is None
    assert any("model.islandlife.json" in line for line in log)
    files = write_own(["glb"], tmp_path / "scene.json", tmp_path / "glb", "model", None, log.append)
    assert [f.name for f in files] == ["model.glb"]                     # the .glb keeps all of it in extras


def test_gf2s_generic_avatar_is_said_to_have_no_human_bones(tmp_path):
    from types import SimpleNamespace
    from polylode.extractor.identity import gf2_statement
    avatar = {"m_Name": "a", "m_Avatar": {"m_Human": {"m_HumanBoneIndex": [-1] * 25}},
              "m_HumanDescription": {"m_Human": []}}
    model = SimpleNamespace(ident="c_X@1", character="X", outfit="", detail={})
    found = gf2_statement(model, tmp_path, "c_X_1", {"avatars": [{"data": avatar}], "source_files": []},
                          tmp_path / "none.npz", None)
    assert "Generic" in found["humanoid"]["reason"]
    assert found["humanoid"]["source"]["avatars"][0]["mapped_human_bones"] == 0
    assert found["identity"]["install"]["version"]["source"] is None
    assert found["identity"]["converted_archive"]["sha256"] is None


def test_a_steam_install_names_its_build_and_a_wuwa_skeleton_its_rig(tmp_path):
    from polylode.extractor.identity import steam_build, wuwa_skeleton
    game = tmp_path / "steamapps" / "common" / "Game" / "Game_Data"
    game.mkdir(parents=True)
    (tmp_path / "steamapps" / "appmanifest_7.acf").write_text(
        '"AppState"\n{\n\t"appid"\t\t"7"\n\t"installdir"\t\t"Game"\n\t"buildid"\t\t"42"\n}\n', encoding="utf-8")
    assert steam_build(game)["buildid"] == "42" and steam_build(tmp_path) is None
    mesh = [{"Type": "SkeletalMesh", "Properties": {"Skeleton": {"ObjectPath": "Game/X_Skeleton.1"}}}]
    skeleton = {"Type": "Skeleton", "AnimRetargetSources": {}, "Properties": {"Sockets": [1, 2]}}
    assert wuwa_skeleton(mesh, skeleton) == {"skeleton": "Game/X_Skeleton.1", "read": True, "rig_config": None,
                                             "anim_retarget_sources": 0, "sockets": 2}
