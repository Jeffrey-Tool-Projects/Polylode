"""GF2 3.0 ``_model`` prefabs: RoleMeshRes paths resolved through the Addressables catalog (IslandLife §7.1510)."""
import struct
from types import SimpleNamespace

import pytest

from polylode.gf2 import addressables, batch, playable, role_mesh
from polylode.gf2 import convert as C

MESH_PATH = "Assets/ArtsResource/Player/Asteria/AsteriaSSR01/Models/c_AsteriaSSR01_slg_body_lod0.mesh"


class Buffer:
    """A tiny Addressables binary catalog, laid out as GF2 3.0's ``catalog_main_29003.bin`` is."""

    def __init__(self):
        self.data = bytearray(32)

    def alloc(self, payload: bytes) -> int:
        self.data += struct.pack("<I", len(payload))
        at = len(self.data)
        self.data += payload
        return at

    def ascii(self, text):
        return self.alloc(text.encode("ascii"))

    def unicode(self, text):
        return self.alloc(text.encode("utf-16-le")) | addressables.UNICODE

    def dynamic(self, parts):
        following = addressables.NONE
        for part in reversed(parts):
            following = self.alloc(struct.pack("<2I", self.ascii(part), following)) | addressables.DYNAMIC
        return following

    def type(self, assembly, name):
        return self.alloc(struct.pack("<2I", self.dynamic(assembly.split(",")), self.dynamic(name.split("."))))

    def location(self, primary, internal, provider, kind, dependencies=()):
        deps = self.alloc(struct.pack(f"<{len(dependencies)}I", *dependencies)) if dependencies else addressables.NONE
        return self.alloc(struct.pack("<7I", self.ascii(primary), internal, self.dynamic(provider.split(".")),
                                      deps, 0, addressables.NONE, self.type("Unity", kind)))

    def finish(self, keys: dict) -> bytes:
        string = self.type("mscorlib", "System.String")
        other = self.type("mscorlib", "System.Int32")
        table = []
        for key, locations in keys.items():
            kind, value = (other, self.alloc(struct.pack("<i", key))) if isinstance(key, int) else \
                (string, self.alloc(struct.pack("<I", self.ascii(key))))
            table += [self.alloc(struct.pack("<2I", kind, value)),
                      self.alloc(struct.pack(f"<{len(locations)}I", *locations))]
        at = self.alloc(struct.pack(f"<{len(table)}I", *table))
        struct.pack_into("<8I", self.data, 0, addressables.MAGIC, 1, at, 0, 0, 0, 0, 0)
        return bytes(self.data)


def catalog_file(tmp_path, kind="UnityEngine.Mesh", provider=role_mesh.PROVIDER, guid="0ee472299169de34cb4ec0e22ca5fe57"):
    buf = Buffer()
    bundle = buf.location("d226945aee30d9508494376d5b544449.bundle", buf.ascii("\\d226945aee30d9508494376d5b544449.bundle"),
                          role_mesh.BUNDLE_PROVIDER, "UnityEngine.ResourceManagement.ResourceProviders.IAssetBundleResource")
    mesh = buf.location(addressables.address(MESH_PATH), buf.unicode(guid), provider, kind, [bundle])
    path = tmp_path / "catalog_main_29003.bin"
    path.write_bytes(buf.finish({addressables.address(MESH_PATH): [mesh],
                                 "d226945aee30d9508494376d5b544449.bundle": [bundle], 7: [bundle]}))
    return path


def test_the_key_of_a_path_is_the_md5_of_its_utf16_bytes_as_bitconverter_writes_it():
    # GF2 3.0's catalog lists this key for the path RoleMeshRes names (catalog_main_29003.bin)
    assert addressables.address(MESH_PATH) == "FB-AC-A3-53-0A-B5-63-20-38-4C-47-D3-13-4A-A7-62"


def test_a_mesh_path_resolves_to_its_guid_and_bundle_file(tmp_path):
    content = addressables.ContentCatalog(catalog_file(tmp_path))
    assert role_mesh.mesh_location(content, MESH_PATH) == ("0ee472299169de34cb4ec0e22ca5fe57",
                                                           ["d226945aee30d9508494376d5b544449.bundle"])
    assert content.other_keys == 1                       # the Int32 key is not read as a string
    with pytest.raises(role_mesh.RoleMeshError):          # a path the catalog lacks is not matched by name
        role_mesh.mesh_location(content, MESH_PATH.replace("lod0", "lod1"))


def test_a_location_that_is_not_a_bundled_mesh_is_refused(tmp_path):
    content = addressables.ContentCatalog(catalog_file(tmp_path, kind="UnityEngine.Texture2D"))
    with pytest.raises(role_mesh.RoleMeshError, match="not a bundled Mesh"):
        role_mesh.mesh_location(content, MESH_PATH)


def test_the_catalog_is_the_one_of_the_current_resource_version(tmp_path):
    folder = tmp_path / addressables.BUNDLES
    folder.mkdir(parents=True)
    (tmp_path / addressables.CONFIG).write_text("CacheResVersion:3.0.5388.0.29003\n", encoding="utf-8")
    (folder / "catalog_main_28279.bin").write_bytes(b"old")
    with pytest.raises(addressables.CatalogError, match="29003"):
        addressables.catalog_path(tmp_path)
    (folder / "catalog_main_29003.bin").write_bytes(b"new")
    assert addressables.catalog_path(tmp_path).name == "catalog_main_29003.bin"


def test_the_guid_is_the_container_key_of_the_one_bundle_that_holds_the_mesh(tmp_path):
    rows = {f"{addressables.BUNDLES}/dep.bundle": {"members": [{"containers": [
                {"container": "cc", "cab": "CAB-d", "id": 1, "type": "Material"}]}]},
            f"{addressables.BUNDLES}/b.bundle": {"members": [{"containers": [
                {"container": "aa", "cab": "CAB-x", "id": 5, "type": "Mesh"},
                {"container": "bb", "cab": "CAB-x", "id": 6, "type": "Texture2D"}]}]}}
    catalog = SimpleNamespace(rows=rows, install=tmp_path)
    # a location lists the bundles loaded with the asset; the one whose container key is the GUID holds it
    assert role_mesh.bundle_entry(catalog, MESH_PATH, "aa", ["dep.bundle", "b.bundle"]) ==         ("CAB-x", 5, f"{addressables.BUNDLES}/b.bundle")
    with pytest.raises(role_mesh.RoleMeshError, match="Texture2D"):
        role_mesh.bundle_entry(catalog, MESH_PATH, "bb", ["b.bundle"])
    with pytest.raises(role_mesh.RoleMeshError, match="not downloaded"):
        role_mesh.bundle_entry(catalog, MESH_PATH, "zz", ["dep.bundle", "missing.bundle"])
    (tmp_path / addressables.BUNDLES).mkdir(parents=True)
    (tmp_path / addressables.BUNDLES / "missing.bundle").write_bytes(b"")
    with pytest.raises(role_mesh.RoleMeshError, match="scan again"):
        role_mesh.bundle_entry(catalog, MESH_PATH, "aa", ["missing.bundle"])


def test_a_guid_the_catalogs_bundles_do_not_hold_is_found_by_the_whole_index(tmp_path):
    # GF2 3.0: the catalog's bundle for a _model prefab's face is not downloaded, and the mesh with that GUID is in
    # the install under a bundle file name the catalog does not list (Vector SSR01's face: c69de0ff… → f08f4902…)
    rows = {f"{addressables.BUNDLES}/dep.bundle": {"members": [{"containers": [
                {"container": "cc", "cab": "CAB-d", "id": 1, "type": "Material"}]}]},
            f"{addressables.BUNDLES}/old.bundle": {"members": [{"containers": [
                {"container": "aa", "cab": "CAB-o", "id": 7, "type": "Mesh"},
                {"container": "tt", "cab": "CAB-o", "id": 8, "type": "Texture2D"}]}]},
            f"{addressables.BUNDLES}/copy.bundle": {"members": [{"containers": [
                {"container": "aa", "cab": "CAB-o", "id": 7, "type": "Mesh"}]}]},
            f"{addressables.BUNDLES}/two.bundle": {"members": [{"containers": [
                {"container": "dd", "cab": "CAB-p", "id": 1, "type": "Mesh"},
                {"container": "dd", "cab": "CAB-q", "id": 2, "type": "Mesh"}]}]},
            f"{addressables.BUNDLES}/broken.bundle": {"error": "GFF is not a Unity bundle"}}
    catalog = SimpleNamespace(rows=rows, install=tmp_path)
    cab, path_id, bundle = role_mesh.bundle_entry(catalog, MESH_PATH, "aa", ["dep.bundle", "missing.bundle"])
    assert (cab, path_id) == ("CAB-o", 7)                 # the same Mesh object in two files is one mesh
    with pytest.raises(role_mesh.RoleMeshError, match="not downloaded"):      # only a Texture2D has the GUID
        role_mesh.bundle_entry(catalog, MESH_PATH, "tt", ["missing.bundle"])
    with pytest.raises(role_mesh.RoleMeshError, match="2 meshes"):            # two meshes: not settled
        role_mesh.bundle_entry(catalog, MESH_PATH, "dd", ["missing.bundle"])


# -- fake prefab objects ------------------------------------------------------------------------------------------
class Ref:
    def __init__(self, target=None, kind=None):
        self.target, self.kind = target, kind
        self.m_PathID = 0 if target is None else 1

    def __bool__(self):
        return self.target is not None

    @property
    def type(self):
        return SimpleNamespace(name=self.kind)

    def read(self):
        return self.target

    def deref(self):
        return self.target

    def read_typetree(self):
        return self.target.tree


def node(name, children=(), components=()):
    transform = SimpleNamespace(m_Children=[])
    go = SimpleNamespace(m_Name=name, m_Component=[SimpleNamespace(component=Ref(transform, "Transform"))])
    transform.m_GameObject = Ref(go)
    for child in children:
        transform.m_Children.append(Ref(child.m_Component[0].component.target))
    go.m_Component += [SimpleNamespace(component=c) for c in components]
    return go


class Reader(SimpleNamespace):
    """An ObjectReader: ``read`` gives the parsed object, without insisting on consuming all its bytes."""

    def read(self, check_read=True):
        if check_read and getattr(self, "short", False):
            raise ValueError("Expected to read 256 bytes, but only read 212 bytes")
        return self


def behaviour(script, rows, short=False):
    value = Reader(m_Script=Ref(SimpleNamespace(m_ClassName=script)), short=short,
                   tree={"MeshResList": [{"TransfromPath": a, "MeshResPath": b} for a, b in rows]})
    return Ref(value, "MonoBehaviour")


def skinned(mesh=None):
    return Ref(Reader(m_Mesh=Ref(mesh)), "SkinnedMeshRenderer")


def test_role_mesh_res_paths_are_paths_below_its_own_node():
    body = node("c_X_slg_body_lod0", components=[skinned()])
    nested = node("c_X_slg_body_lod0")
    group = node("group", [nested])
    root = node("c_X_slg_skin_model", [body, group], [behaviour("RoleMeshRes", [("group/c_X_slg_body_lod0", "a")])])
    assert role_mesh.declared(root) == [{"node": "group/c_X_slg_body_lod0", "path": "a"}]
    transform = root.m_Component[0].component.target
    assert role_mesh.find_node(transform, "group/c_X_slg_body_lod0") is nested.m_Component[0].component.target
    with pytest.raises(role_mesh.RoleMeshError):          # not a direct child: the name alone does not find it
        role_mesh.find_node(transform, "missing")
    twin = node("c_X_slg_body_lod0")
    transform.m_Children.append(Ref(twin.m_Component[0].component.target))
    with pytest.raises(role_mesh.RoleMeshError, match="2 children"):
        role_mesh.find_node(transform, "c_X_slg_body_lod0")
    assert role_mesh.declared(node("plain", components=[behaviour("SomethingElse", [("a", "b")])])) is None
    # another script whose typetree does not cover its data (Colphne SR01 Gacha's face) is still told apart
    assert role_mesh.declared(node("face", components=[behaviour("SomethingElse", [], short=True)])) is None


def test_a_renderer_with_its_own_mesh_keeps_it_and_its_row_is_not_resolved():
    # the 2.x prefabs (Mosinnagant SSR01 Class) carry a RoleMeshRes whose paths the 3.0 catalog lacks; their
    # renderers point at meshes of their own, and nothing is looked up, not even the content catalog
    def unopened():
        raise AssertionError("the content catalog was opened")
    own = node("c_X_slg_body_lod0", components=[skinned(SimpleNamespace(name="own"))])
    root = node("c_X_Class", [own])
    transform = root.m_Component[0].component.target
    row = {"node": "c_X_slg_body_lod0", "path": MESH_PATH.replace("body", "boby")}
    assert role_mesh._resolve(None, None, unopened, transform, row) is None
    empty = node("c_X_slg_body_lod0", components=[skinned()])
    transform = node("c_X_slg_skin_model", [empty]).m_Component[0].component.target
    with pytest.raises(AssertionError, match="opened"):   # an empty one is resolved
        role_mesh._resolve(None, None, unopened, transform, row)


def test_the_scan_counts_a_renderer_a_role_mesh_res_gives_a_mesh():
    matches = [{"name": "X"}]
    body = node("c_X_slg_body_lod0", components=[skinned()])
    without = node("c_X_slg_skin_model", [body])
    assert not playable.has_owned_mesh(without, matches)
    body = node("c_X_slg_body_lod0", components=[skinned()])
    root = node("c_X_slg_skin_model", [body], [behaviour("RoleMeshRes", [("c_X_slg_body_lod0", MESH_PATH)])])
    assert playable.has_owned_mesh(root, matches)
    other = node("c_X_slg_body_lod0", components=[skinned()])
    elsewhere = node("c_X_slg_skin_model", [other], [behaviour("RoleMeshRes", [("c_X_slg_hair_lod0", MESH_PATH)])])
    assert not playable.has_owned_mesh(elsewhere, matches)
    # a path the game's catalog lacks gives the renderer nothing (the 3.0 shadow _model copies name only such paths)
    asked = []
    assert not playable.has_owned_mesh(root, matches, lambda path: asked.append(path) or False)
    assert asked == [MESH_PATH] and playable.has_owned_mesh(root, matches, lambda path: True)


def test_the_screen_asks_the_install_catalog_whether_a_path_can_be_loaded(tmp_path):
    folder = tmp_path / addressables.BUNDLES
    folder.mkdir(parents=True)
    (tmp_path / addressables.CONFIG).write_text("CacheResVersion:3.0.5388.0.29003\n", encoding="utf-8")
    catalog_file(folder)
    has = role_mesh.loadable(tmp_path)
    assert has(MESH_PATH) and not has(MESH_PATH.replace("Player", "Effect"))
    assert role_mesh.loadable(tmp_path / "elsewhere")(MESH_PATH)    # no catalog: listed, the export says why


def test_a_meshless_verdict_from_before_role_mesh_res_was_read_is_checked_again():
    old = {"status": "not_model", "reason": batch.MESHLESS}
    assert not batch.settled(old)
    assert batch.settled(dict(old, role_mesh_res=True))
    assert batch.settled({"status": "not_model", "reason": "no owned skinned mesh renderer"})
    assert batch.settled({"status": "excluded_scope"})
    assert not batch.settled({"status": "failed"})


def test_the_assigned_mesh_takes_the_place_of_the_empty_mesh_reference():
    rd = SimpleNamespace(m_Mesh=Ref(), m_Materials=[], m_Bones=[], m_Enabled=1, m_CastShadows=1)
    native = SimpleNamespace(type=SimpleNamespace(name="SkinnedMeshRenderer"), read=lambda: rd,
                             assets_file=SimpleNamespace(name="CAB-Prefab"), path_id=9)
    mesh = SimpleNamespace(assets_file=SimpleNamespace(name="CAB-Mesh"), path_id=-3)
    provenance = {"path": MESH_PATH}
    renderer, readers, declaration = C._renderer_data(native, [], True, {("cab-prefab", 9): (mesh, provenance)})
    assert renderer.mesh == ("cab-mesh", -3) and readers[("cab-mesh", -3)] is mesh
    assert declaration["mesh"] == ["cab-mesh", -3] and declaration["mesh_source"] == provenance
    renderer, _readers, declaration = C._renderer_data(native, [], True, {})
    assert renderer.mesh is None and declaration["mesh"] is None
