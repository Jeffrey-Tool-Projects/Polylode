"""A GF2 renderer palette longer than its mesh's binds, trimmed only with the mesh's bone hashes as proof
(``gf2.palette.trim``; Cheeta SR01's ``cloth_lod0``: 117 bones for the 3.0 mesh's 116 binds)."""
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from polylode.endfield import prefab as P
from polylode.gf2 import palette


def skeleton():
    """root → a → b, and c beside a: identity locals, so an identity bind is every bone's rest."""
    nodes = {0: P.Node(0, "root", None, np.eye(4), True, frozenset(), children=[1, 3]),
             1: P.Node(1, "a", 0, np.eye(4), True, frozenset(), children=[2]),
             2: P.Node(2, "b", 1, np.eye(4), True, frozenset()),
             3: P.Node(3, "c", 0, np.eye(4), True, frozenset())}
    renderer = P.Renderer(True, ("cab", 9), [("cab", 5)], [1, 2, 3], True, 1)
    node = P.Node(4, "c_X_slg_cloth_lod0", 0, np.eye(4), True, frozenset(), renderer)
    nodes[4] = node
    nodes[0].children.append(4)
    return P.Prefab("cab", 0, nodes, []), node


def crc(prefab, keys):
    return [zlib.crc32(P.node_path(prefab, key).encode("utf8")) for key in keys]


def test_a_palette_longer_than_the_binds_loses_its_tail_when_the_hashes_prove_the_rest():
    prefab, node = skeleton()
    mesh = SimpleNamespace(binds=np.stack([np.eye(4)] * 2))
    report = palette.trim(prefab, node, mesh, crc(prefab, [1, 2]))
    assert node.renderer.bones == [1, 2] and report["dropped"] == [3]
    assert report["original_bones"] == [1, 2, 3] and report["mesh_weights_binds_unchanged"]


def test_an_unproven_palette_is_left_as_it_is():
    prefab, node = skeleton()
    mesh = SimpleNamespace(binds=np.stack([np.eye(4)] * 2))
    with pytest.raises(ValueError, match="hashes disagree"):         # the binds name other bones
        palette.trim(prefab, node, mesh, crc(prefab, [1, 3]))
    with pytest.raises(ValueError, match="hash per bind"):
        palette.trim(prefab, node, mesh, crc(prefab, [1]))
    assert node.renderer.bones == [1, 2, 3]
    with pytest.raises(ValueError, match="longer palette"):           # not longer: nothing to trim
        palette.trim(prefab, node, SimpleNamespace(binds=np.stack([np.eye(4)] * 3)), crc(prefab, [1, 2, 3]))
