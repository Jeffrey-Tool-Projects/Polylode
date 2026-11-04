import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from polylode.extractor import shader_export
from polylode.extractor.gltf_runtime import read
from polylode.interchange import glb


def fixture(tmp_path, family="endfield"):
    from test_interchange_glb import _scene
    document, arrays = _scene(tmp_path)
    for material in document["materials"].values():
        if family == "wuwa":
            material["native_material"] = {"Scalars": {"ShadowProcess": .37}}
        else:
            material["shader"] = "HGRP/CharacterNPR" if family == "endfield" else "gf_shader/pbr/character/uber"
    path = tmp_path / "model.glb"
    glb.write(document, arrays, tmp_path, path)
    return path


def test_bundle_is_self_contained_and_materials_have_stable_ids(tmp_path):
    source = fixture(tmp_path)
    bundle = shader_export.write_bundle(source, tmp_path / "model.shaders")
    source.unlink()
    moved = tmp_path / "moved"
    bundle.rename(moved)
    document, _ = read(moved / "model.glb")
    manifest = json.loads((moved / "materials.json").read_text(encoding="utf-8"))
    assert [m["extras"]["polylode_material_index"] for m in document["materials"]] == [0, 1]
    assert {m["id"] for m in manifest["materials"]} == {0, 1}
    assert {b["material"] for b in manifest["bindings"]} == {0, 1}
    assert str(tmp_path) not in json.dumps(manifest)
    assert "file:" not in json.dumps(manifest)
    for material in manifest["materials"]:
        for key in ("baseTexture", "surface", "ramp", "shadowLut"):
            if material[key]:
                assert not Path(material[key]).is_absolute()
                assert (moved / material[key]).is_file()
    assert "from polylode" not in (moved / "import_blender.py").read_text(encoding="utf-8")


def test_preview_never_builds_shader_companions_and_setting_roundtrips(tmp_path):
    from polylode.extractor.settings import Settings, load, save
    from polylode.extractor.preview import preview_settings
    assert not Settings().export_shaders                       # written only when the user ticks it
    settings = Settings(export_shaders=True, tools={"blender": "blender.exe"})
    assert not preview_settings(settings).export_shaders
    assert settings.export_shaders
    path = tmp_path / "settings.json"
    settings.export_shaders = False
    save(settings, path)
    assert not load(path).export_shaders


def test_shader_export_without_blender_returns_portable_bundle(tmp_path):
    from polylode.extractor.sources import write_own
    from test_interchange_glb import _scene
    import numpy as np
    document, arrays = _scene(tmp_path)
    document["materials"]["skin"]["shader"] = "HGRP/CharacterNPR"
    source = tmp_path / "scene.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    np.savez(tmp_path / document["arrays"], **arrays)
    log = []
    output = tmp_path / "export"
    output.mkdir()
    files = write_own([], source, output, "character", None, log.append, export_shaders=True)
    assert output / "character.shaders" / "model.glb" in files
    assert output / "character.shaders" / "import_blender.py" in files
    assert not list(output.glob("*.shaded.blend"))
    assert any("未設定 Blender" in line for line in log)


@pytest.mark.parametrize("family", ["endfield", "gf2", "wuwa"])
def test_relocated_bundle_builds_packed_blend_and_renders(tmp_path, family):
    blender = os.environ.get("BLENDER_EXE") or shutil.which("blender")
    if not blender:
        pytest.skip("BLENDER_EXE is needed for the real Blender shader check")
    source = fixture(tmp_path, family)
    bundle = shader_export.write_bundle(source, tmp_path / "model.shaders")
    moved = tmp_path / "relocated shaders"
    bundle.rename(moved)
    source.unlink()
    output = tmp_path / "character.shaded.blend"
    shader_export.write_blend(moved, Path(blender), output)
    check = tmp_path / "check.py"
    check.write_text('''
import bpy
from pathlib import Path
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
assert set(bpy.context.scene["polylode_shader_materials"]) == {0, 1}
assert all(m.get("polylode_character_style") for m in bpy.data.materials)
assert all(i.packed_file for i in bpy.data.images if i.source == "FILE" and i.size[0])
scene = bpy.context.scene
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
for obj in scene.objects:
    if obj.type == "MESH":
        evaluated = obj.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            projected = world_to_camera_view(scene, scene.camera, evaluated.matrix_world @ Vector(corner))
            assert 0 <= projected.x <= 1 and 0 <= projected.y <= 1, tuple(projected)
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.filepath = str(Path(bpy.data.filepath).with_suffix(".png"))
bpy.ops.render.render(write_still=True)
assert Path(scene.render.filepath).stat().st_size > 200
''', encoding="utf-8")
    done = subprocess.run([blender, "--background", str(output), "--python-exit-code", "1", "--python", str(check)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    assert done.returncode == 0, done.stdout + done.stderr
