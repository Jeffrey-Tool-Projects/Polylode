"""GF2 3.0's ``_model`` prefabs: the meshes ``RoleMeshRes`` assigns to their renderers (IslandLife §7.1510).

3.0 ships a character as ``c_<character><outfit>_slg_skin_model``: the skeleton, the
renderers with their bones and materials, and an empty ``m_Mesh`` on every
SkinnedMeshRenderer.  The root carries a ``RoleMeshRes`` MonoBehaviour (MonoScript
``RoleMeshRes``, ``Assembly-CSharp.dll``; the typetree is in the bundle) whose
``MeshResList`` pairs a ``TransfromPath`` (sic) with a ``MeshResPath``, an asset path such as
``Assets/ArtsResource/Player/Asteria/AsteriaSSR01/Models/c_AsteriaSSR01_slg_body_lod0.mesh``.
The meshes live in other bundles; the game loads each path through Addressables and puts it
on the renderer at run time.

Here the same thing is done before conversion: the path is resolved through the install's
content catalog (:mod:`.addressables`: path → key → the mesh's GUID and its bundle file → that
bundle's container entry), and the renderer of the node the ``TransfromPath`` names, a path
below the root, gets that mesh.  Bones, bind poses and materials stay the prefab's own.
Anything that does not resolve — a path the catalog lacks, a location that is not a ``Mesh``,
a bundle the index does not have, a node that is not there or has no renderer, a renderer
that already has a different mesh — is an error; nothing is matched by name.
"""
from __future__ import annotations

from .addressables import BUNDLES, CatalogError, address, opened

SCRIPT = "RoleMeshRes"
PROVIDER = "UnityEngine.ResourceManagement.ResourceProviders.BundledAssetProvider"
BUNDLE_PROVIDER = "UnityEngine.ResourceManagement.ResourceProviders.AssetBundleProvider"
MESH_TYPE = "UnityEngine.Mesh"


class RoleMeshError(ValueError):
    """A ``RoleMeshRes`` entry the install does not resolve."""


def script_name(component) -> str | None:
    """The class name of a MonoBehaviour component's MonoScript (None when it has none)."""
    behaviour = component.read()
    script = getattr(behaviour, "m_Script", None)
    if not script:
        return None
    return script.read().m_ClassName


def declared(gameobject) -> list[dict] | None:
    """The ``MeshResList`` of the ``RoleMeshRes`` on *gameobject* (a GameObject read), as
    ``[{'node': TransfromPath, 'path': MeshResPath}, …]``; None when it has no ``RoleMeshRes``."""
    found = None
    for item in gameobject.m_Component:
        component = item.component
        if component.type.name != "MonoBehaviour" or script_name(component) != SCRIPT:
            continue
        if found is not None:
            raise RoleMeshError(f"{gameobject.m_Name}: two RoleMeshRes components")
        tree = component.read_typetree()
        if "MeshResList" not in tree:
            raise RoleMeshError(f"{gameobject.m_Name}: RoleMeshRes without a MeshResList in its typetree")
        found = [{"node": row["TransfromPath"], "path": row["MeshResPath"]} for row in tree["MeshResList"]]
    return found


def mesh_location(content, path: str) -> tuple[str, list[str]]:
    """``(GUID, bundle file names)`` of the mesh asset at *path* in the content catalog *content*: the location's
    dependencies are the bundles loaded with it — its own and the ones that one depends on (a prop's mesh lists
    several) — so which of them holds the asset is read from the bundles (:func:`bundle_entry`)."""
    key = address(path)
    locations = content.locate(key)
    if len(locations) != 1:
        raise RoleMeshError(f"{path}: the catalog has {len(locations)} locations for its key {key}")
    location = locations[0]
    if location.provider != PROVIDER or location.type != MESH_TYPE:
        raise RoleMeshError(f"{path}: catalog location is a {location.type} from {location.provider}, not a bundled Mesh")
    if not location.dependencies or any(d.provider != BUNDLE_PROVIDER for d in location.dependencies):
        raise RoleMeshError(f"{path}: catalog location depends on {len(location.dependencies)} locations, "
                            f"not all of them bundles")
    names = []
    for bundle in location.dependencies:
        name = (bundle.internal_id or "").lstrip("\\/")
        if not name or "/" in name or "\\" in name or not name.endswith(".bundle"):
            raise RoleMeshError(f"{path}: catalog bundle id {bundle.internal_id!r} is not a bundle file name")
        names.append(name)
    return location.internal_id, names


def bundle_entry(catalog, path: str, guid: str, bundles: list[str]) -> tuple[str, int, str]:
    """``(CAB, path id, bundle path)`` of the ``Mesh`` container *guid* in the one of the indexed *bundles* that holds
    it.  A bundle the install does not have (the game downloads them as it needs them) is reported when the
    others do not hold the asset."""
    holders, absent, unindexed = [], [], []
    for bundle in bundles:
        relative = f"{BUNDLES}/{bundle}"
        row = catalog.rows.get(relative)
        if row is None or row.get("error"):
            (unindexed if (catalog.install / relative).is_file() else absent).append(bundle)
            continue
        holders += [(entry, relative) for member in row["members"] for entry in member["containers"]
                    if entry["container"] == guid]
    if len(holders) == 1 and holders[0][0]["type"] == "Mesh":
        return holders[0][0]["cab"], holders[0][0]["id"], holders[0][1]
    if not holders and absent:
        raise RoleMeshError(f"{path}: not in the install: bundle {', '.join(absent)} is not downloaded")
    if not holders and unindexed:
        raise RoleMeshError(f"{path}: bundle {', '.join(unindexed)} is not in the bundle index (scan again)")
    raise RoleMeshError(f"{path}: {len(holders)} containers {guid} in its bundles"
                        + (f", a {holders[0][0]['type']}" if len(holders) == 1 else ""))


def find_node(root_transform, node_path: str):
    """The Transform at *node_path* below *root_transform*, one child name per ``/`` step; a step that matches no
    child or several is an error (the game's own lookup is not readable, so an ambiguous name is not settled)."""
    transform = root_transform
    for step in node_path.split("/"):
        children = [pointer.read() for pointer in transform.m_Children]
        named = [child for child in children if child.m_GameObject.read().m_Name == step]
        if len(named) != 1:
            raise RoleMeshError(f"{node_path}: {len(named)} children named {step!r}")
        transform = named[0]
    return transform


def assignments(catalog, env, gameobject) -> dict:
    """``{(CAB, path id) of the renderer: (mesh ObjectReader, provenance)}`` for every ``RoleMeshRes`` in the prefab
    of the root *gameobject* (a GameObject read from *env*); empty when it has none.

    A ``RoleMeshRes`` names its nodes by path below its own node: on a ``_model`` root that is the root, and the 2.x
    cut scenes had one on a nested outfit instance (``source_dependencies.json``, Alva SSR01_Upgrade).  *catalog* is
    the bundle catalog (:class:`.catalog.Catalog`), whose install's content catalog resolves the asset paths and
    whose index finds the bundle entries."""
    found = {}
    content = None
    stack = [gameobject]
    while stack:
        go = stack.pop()
        transform = next(item.component.read() for item in go.m_Component if item.component.type.name == "Transform")
        stack.extend(pointer.read().m_GameObject.read() for pointer in transform.m_Children)
        rows = declared(go)
        if not rows:
            continue
        if content is None:
            try:
                content = opened(catalog.install)
            except CatalogError as exc:
                raise RoleMeshError(f"{go.m_Name}: RoleMeshRes paths cannot be resolved: {exc}") from exc
        for row in rows:
            mesh, provenance, renderer = _resolve(catalog, env, content, transform, row)
            key = (renderer.assets_file.name.lower(), renderer.path_id)
            if key in found:
                raise RoleMeshError(f"{row['node']}: RoleMeshRes names the renderer twice")
            found[key] = (mesh, dict(provenance, owner=go.m_Name))
    return found


def _resolve(catalog, env, content, owner, row):
    guid, bundles = mesh_location(content, row["path"])
    cab, path_id, bundle = bundle_entry(catalog, row["path"], guid, bundles)
    node = find_node(owner, row["node"]).m_GameObject.read()
    renderers = [item.component.deref() for item in node.m_Component
                 if item.component.type.name == "SkinnedMeshRenderer"]
    if len(renderers) != 1:
        raise RoleMeshError(f"{row['node']}: {len(renderers)} SkinnedMeshRenderers on the node RoleMeshRes names")
    renderer = renderers[0]
    mesh = env.find_file(cab).objects[path_id]
    if mesh.type.name != "Mesh":
        raise RoleMeshError(f"{row['path']}: {cab}/{path_id} is a {mesh.type.name}")
    held = renderer.read().m_Mesh
    if held.m_PathID and (held.deref().assets_file.name.lower(), held.m_PathID) != (cab.lower(), path_id):
        raise RoleMeshError(f"{row['node']}: the renderer already has another mesh")
    return mesh, {"node": row["node"], "path": row["path"], "address": address(row["path"]), "guid": guid,
                  "bundle": bundle, "cab": cab, "id": path_id, "catalog": content.path.name}, renderer


def covered(gameobject, path: tuple = ()) -> set[tuple]:
    """The node paths (tuples of names below *gameobject*) some ``RoleMeshRes`` in its subtree gives a mesh, read for
    the scan's screen: nothing is resolved."""
    out = set()
    rows = declared(gameobject) or []
    out.update(path + tuple(row["node"].split("/")) for row in rows)
    transform = next(item.component.read() for item in gameobject.m_Component
                     if item.component.type.name == "Transform")
    for pointer in transform.m_Children:
        child = pointer.read().m_GameObject.read()
        out |= covered(child, path + (child.m_Name,))
    return out
