import struct

import pytest

from polylode.extractor.gltf_runtime import mesh_nodes


def scene(points, indices=None, component=5123, stride=12):
    binary = b"pad!" + b"".join(struct.pack("<3f", *point) + b"\0" * (stride - 12) for point in points)
    document = {
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "type": "VEC3", "count": len(points)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 4, "byteLength": len(binary) - 4, "byteStride": stride}],
    }
    if indices is not None:
        fmt = {5121: "B", 5123: "H", 5125: "I"}[component]
        packed = struct.pack("<" + fmt * len(indices), *indices)
        document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(packed)})
        document["accessors"].append({"bufferView": 1, "componentType": component, "type": "SCALAR", "count": len(indices)})
        document["meshes"][0]["primitives"][0]["indices"] = 1
        binary += packed
    return document, binary


@pytest.mark.parametrize("component", [5121, 5123, 5125])
@pytest.mark.parametrize("stride", [12, 16])
def test_fully_collapsed_indexed_billboard_is_omitted(component, stride):
    document, binary = scene([(1, 2, 3)] * 3, [0, 1, 2], component, stride)
    assert mesh_nodes(document) == [0]
    assert mesh_nodes(document, binary) == []


def test_nonindexed_collapsed_mesh_is_omitted():
    document, binary = scene([(0, 0, 0)] * 3)
    assert mesh_nodes(document, binary) == []


def test_mixed_collapsed_and_real_triangles_are_retained():
    document, binary = scene([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 0, 0, 0, 1, 2])
    assert mesh_nodes(document, binary) == [0]


def test_collinear_triangle_is_not_assumed_to_disappear():
    document, binary = scene([(0, 0, 0), (1, 0, 0), (2, 0, 0)], [0, 1, 2])
    assert mesh_nodes(document, binary) == [0]


def test_collapsed_parent_does_not_hide_real_child():
    document, binary = scene([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 0, 0])
    document["nodes"][0]["children"] = [1]
    document["nodes"].append({"mesh": 1})
    document["meshes"].append({"primitives": [{"attributes": {"POSITION": 0}}]})
    assert mesh_nodes(document, binary) == [1]


def test_morph_mesh_is_retained_even_when_base_is_collapsed():
    document, binary = scene([(0, 0, 0)] * 3)
    document["meshes"][0]["primitives"][0]["targets"] = [{"POSITION": 0}]
    assert mesh_nodes(document, binary) == [0]
