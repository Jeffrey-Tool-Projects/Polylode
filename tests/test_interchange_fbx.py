"""The binary FBX writer (``interchange/fbx.py``), read back with ``endfield/fbxfile.py``'s record reader."""
import struct
import zlib

import numpy as np
import pytest

from polylode.endfield import fbxfile
from polylode.interchange import fbx
from test_interchange_glb import PNG, _scene


def _children(data, start, stop):
    return {name.decode(): (props, a, b) for name, props, a, b in fbxfile._records(data, start, stop)}


def _objects(data):
    top = fbxfile._top(data)
    out = []
    for name, props, start, stop in fbxfile._records(data, *top[b"Objects"]):
        out.append((name.decode(), props, start, stop))
    return out


def _value(data, prop):
    """A property's value; an array of any type decoded here (``fbxfile._array`` reads doubles only)."""
    kind, value = prop[0], prop[1]
    if kind not in b"fdlib" or len(kind) != 1:
        return value
    count, encoding, stored = value
    raw = data[prop[2]:prop[2] + stored]
    raw = zlib.decompress(raw) if encoding == 1 else raw
    return np.frombuffer(raw, dtype={b"f": "<f4", b"d": "<f8", b"l": "<i8", b"i": "<i4", b"b": "u1"}[kind], count=count)


def test_the_file_holds_the_skeleton_skin_polygons_materials_and_shapes(tmp_path):
    document, arrays = _scene(tmp_path)
    report = fbx.write(document, arrays, tmp_path, tmp_path / "out.fbx")
    data = (tmp_path / "out.fbx").read_bytes()
    assert data[:23] == fbxfile.MAGIC and struct.unpack_from("<I", data, 23)[0] == 7400
    assert data.endswith(fbx._FOOT_MAGIC) and report["degenerate"] == {"body": 1}

    bones = [(b["name"], b["parent"], arrays["bone_rest"][i]) for i, b in enumerate(document["bones"])]
    errors = fbxfile.bone_errors(tmp_path / "out.fbx", bones)
    assert errors["bones"] == 2 and not errors["missing"]
    assert errors["offset_mm"] < 1e-6 and errors["turn_deg"] < 1e-6
    assert errors["binds"] == 2 and errors["bind_offset_mm"] < 1e-6 and errors["bind_turn_deg"] < 1e-6

    objects = _objects(data)
    kinds = [(name, _value(data, props[2]) if len(props) > 2 else None) for name, props, _a, _b in objects]
    assert ("Model", b"Null") in kinds and kinds.count(("Model", b"LimbNode")) == 2 and ("Model", b"Mesh") in kinds
    geometry = next(o for o in objects if o[0] == "Geometry" and _value(data, o[1][2]) == b"Mesh")
    held = _children(data, geometry[2], geometry[3])
    corners = _value(data, held["PolygonVertexIndex"][0][0])
    assert corners.tolist() == [0, 1, -3, 0, 2, -4]                          # the degenerate triangle left out
    vertices = _value(data, held["Vertices"][0][0]).reshape(-1, 3)
    assert np.allclose(vertices[2], [0, 1, -1])                              # Y up
    materials = _children(data, *held["LayerElementMaterial"][1:])
    assert _value(data, materials["Materials"][0][0]).tolist() == [0, 1]
    uv = _children(data, *held["LayerElementUV"][1:])
    assert np.allclose(_value(data, uv["UV"][0][0]).reshape(-1, 2)[3], [.5, .25])   # FBX keeps v up
    assert _value(data, uv["UVIndex"][0][0]).tolist() == [0, 1, 2, 0, 2, 3]

    clusters = [o for o in objects if o[0] == "Deformer" and _value(data, o[1][2]) == b"Cluster"]
    weights = {}
    for _name, props, start, stop in clusters:
        held = _children(data, start, stop)
        weights[_value(data, props[1]).split(b"\x00")[0]] = (_value(data, held["Indexes"][0][0]).tolist(),
                                                             _value(data, held["Weights"][0][0]).tolist())
    assert weights[b"root"] == ([0, 2], [.25, 2.0]) and weights[b"spine"] == ([0, 1, 2], [.75, 1.0, 2.0])
    shape = next(o for o in objects if o[0] == "Geometry" and _value(data, o[1][2]) == b"Shape")
    held = _children(data, shape[2], shape[3])
    assert _value(data, held["Indexes"][0][0]).tolist() == [1, 3]
    assert np.allclose(_value(data, held["Vertices"][0][0]).reshape(-1, 3), [[0, 0, -.1], [0, .5, 0]])
    video = next(o for o in objects if o[0] == "Video")
    assert _value(data, _children(data, video[2], video[3])["Content"][0][0]) == PNG


def test_a_sheared_bone_and_a_bone_before_its_parent_are_refused(tmp_path):
    document, arrays = _scene(tmp_path)
    arrays["bone_rest"] = arrays["bone_rest"].copy()
    arrays["bone_rest"][1, 1, 1] += .3                # the second axis leans on the first
    with pytest.raises(fbx.FbxWriteError, match="sheared"):
        fbx.write(document, arrays, tmp_path, tmp_path / "out.fbx")
    document, arrays = _scene(tmp_path)
    document["bones"][0]["parent"] = 1
    with pytest.raises(fbx.FbxWriteError, match="before its parent"):
        fbx.write(document, arrays, tmp_path, tmp_path / "out.fbx")


def test_a_base_maps_tiling_is_its_textures_scaling_and_one_image_is_stored_once(tmp_path):
    document, arrays = _scene(tmp_path)
    document["materials"]["cloth"] = {"maps": {"_MainTex": {"file": "textures/base.png", "srgb": True,
                                                            "scale": [1, 1], "offset": [0, 0]}}, "base": "_MainTex"}
    fbx.write(document, arrays, tmp_path, tmp_path / "out.fbx")
    data = (tmp_path / "out.fbx").read_bytes()
    objects = _objects(data)
    assert len([o for o in objects if o[0] == "Video"]) == 1                    # the bytes once
    textures = [o for o in objects if o[0] == "Texture"]
    assert len(textures) == 2                                                   # two placements of one image
    placed = {}
    for _name, _props, start, stop in textures:
        held = _children(data, start, stop)
        if "Properties70" not in held:
            continue
        for _n, props, _a, _b in fbxfile._records(data, *held["Properties70"][1:]):
            placed[_value(data, props[0]).decode()] = [_value(data, p) for p in props[4:6]]
    assert placed == {"Translation": [0.0, 0.5], "Scaling": [2.0, 1.0]}       # Unity's offset and tiling as they are
    assert fbx._FILE_ID in data and fbx._TIME_ID.encode() in data                # the pair the footer code belongs to


def test_a_slot_naming_a_missing_material_is_refused(tmp_path):
    document, arrays = _scene(tmp_path)
    document["parts"][0]["materials"] = ["skin", "gone"]
    with pytest.raises(fbx.FbxWriteError, match="gone"):
        fbx.write(document, arrays, tmp_path, tmp_path / "out.fbx")


def test_an_overlays_coverage_image_is_its_transparency_too(tmp_path):
    """§7.1455: the overlay's white-with-coverage image, linked as diffuse and transparent colour."""
    document, arrays = _scene(tmp_path)
    document["materials"]["cloth"] = {"maps": {"_BaseMap": {"file": "textures/base.png", "srgb": False}},
                                      "base": "_BaseMap", "tint": [0.8, 0.6, 0.7, 1.0], "alpha_from": "r",
                                      "blend": "multiply"}
    fbx.write(document, arrays, tmp_path, tmp_path / "out.fbx")
    data = (tmp_path / "out.fbx").read_bytes()
    assert b"base_r_alpha.png" in data and b"TransparentColor" in data
    assert len([o for o in _objects(data) if o[0] == "Video"]) == 2               # the colour map and the coverage
