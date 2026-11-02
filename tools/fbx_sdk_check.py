"""Read an FBX with Autodesk's FBX SDK and compare it with the interchange scene it was written from (§7.1448).

    <python with the fbx wheel> tools/fbx_sdk_check.py <file.fbx> <scene.json>

The FBX SDK is Autodesk's (licensed by them, not shipped or tracked here): install its Python wheel
(``fbx-2020.3.11-cp311-none-win_amd64.whl``, inside ``fbx2020311_fbxpythonsdk_win.exe`` from Autodesk's
FBX SDK page) into a Python 3.11 of its own, with numpy.  Prints one JSON line: whether the SDK imported
the file and with what status, its axes and unit, every bone's global rest against the scene's
(``bone_head_mm``, ``bone_turn_deg``, parents), each part's control points, polygons, skin, weighted
vertices, shapes, UV sets and normals (``parts_bad`` lists the ones that differ), the skin clusters'
bind matrices (``bind_mm``), the textures, and the embedded images the SDK extracted (``<file>.fbm``).
A base map's tiling and offset are printed per material (``placements``).
"""
import json
import sys
from pathlib import Path

import fbx
import numpy as np

TO_Y_UP = np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


def matrix(m) -> np.ndarray:
    """An FbxAMatrix (row vectors, translation in row 3) as a column-vector 4×4."""
    return np.array([[m.Get(r, c) for c in range(4)] for r in range(4)]).T


def angle(a, b) -> float:
    r = a[:3, :3].T @ b[:3, :3]
    sine = 0.5 * np.linalg.norm([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return float(np.degrees(np.arctan2(sine, (np.trace(r) - 1) / 2)))


path, scene_path = Path(sys.argv[1]), Path(sys.argv[2])
document = json.loads(scene_path.read_text(encoding="utf-8"))
arrays = np.load(scene_path.parent / document["arrays"])
manager = fbx.FbxManager.Create()
settings = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
manager.SetIOSettings(settings)
importer = fbx.FbxImporter.Create(manager, "")
report = {"file": path.name}
if not importer.Initialize(str(path), -1, settings):
    print(json.dumps({"file": path.name, "initialize": False, "error": importer.GetStatus().GetErrorString()}))
    sys.exit(1)
report["file_version"] = list(importer.GetFileVersion())
scene = fbx.FbxScene.Create(manager, "")
report["imported"] = bool(importer.Import(scene))
report["status"] = importer.GetStatus().GetErrorString()
axes = scene.GetGlobalSettings().GetAxisSystem()
report["up_axis"] = axes.GetUpVector()[0]
report["unit_cm"] = scene.GetGlobalSettings().GetSystemUnit().GetScaleFactor()

nodes = {}
def walk(node):
    nodes[node.GetName()] = node
    for i in range(node.GetChildCount()):
        walk(node.GetChild(i))
walk(scene.GetRootNode())

rest = np.einsum("ij,njk->nik", TO_Y_UP, np.asarray(arrays["bone_rest"], dtype=np.float64))
missing, head, turn, parents = [], 0.0, 0.0, 0
for index, bone in enumerate(document["bones"]):
    node = nodes.get(bone["name"])
    attribute = node.GetNodeAttribute() if node else None
    if node is None or attribute is None or attribute.GetAttributeType() != fbx.FbxNodeAttribute.EType.eSkeleton:
        missing.append(bone["name"])
        continue
    world = matrix(node.EvaluateGlobalTransform())
    head = max(head, float(np.linalg.norm(world[:3, 3] - rest[index][:3, 3])) * 1000)
    turn = max(turn, angle(world, rest[index]))
    above = node.GetParent().GetName()
    expected = document["bones"][bone["parent"]]["name"] if bone["parent"] >= 0 else document["name"]
    parents += above != expected
report.update(bones=len(document["bones"]) - len(missing), bones_missing=missing[:5], bone_head_mm=head,
              bone_turn_deg=turn, parents_wrong=parents)

parts, bind_worst = [], 0.0
for index, part in enumerate(document["parts"]):
    node = nodes.get(part["name"])
    mesh = node.GetMesh() if node else None
    if mesh is None:
        parts.append({"name": part["name"], "found": False})
        continue
    tri = np.asarray(arrays[f"p{index}_tri"])
    sound = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    co = np.asarray(arrays[f"p{index}_co"], dtype=np.float64) @ TO_Y_UP[:3, :3].T
    points = np.array([[mesh.GetControlPointAt(i)[k] for k in range(3)] for i in range(mesh.GetControlPointsCount())])
    skins = mesh.GetDeformerCount(fbx.FbxDeformer.EDeformerType.eSkin)
    clusters, weighted = 0, set()
    if skins:
        skin = mesh.GetDeformer(0, fbx.FbxDeformer.EDeformerType.eSkin)
        clusters = skin.GetClusterCount()
        for c in range(clusters):
            cluster = skin.GetCluster(c)
            link = fbx.FbxAMatrix()
            cluster.GetTransformLinkMatrix(link)
            name = cluster.GetLink().GetName()
            bone = next(i for i, b in enumerate(document["bones"]) if b["name"] == name)
            bind_worst = max(bind_worst, float(np.linalg.norm(matrix(link)[:3, 3] - rest[bone][:3, 3])) * 1000)
            weighted.update(cluster.GetControlPointIndices())
    weights = np.asarray(arrays[f"p{index}_wgt"])
    shapes = 0
    for b in range(mesh.GetDeformerCount(fbx.FbxDeformer.EDeformerType.eBlendShape)):
        shapes += mesh.GetDeformer(b, fbx.FbxDeformer.EDeformerType.eBlendShape).GetBlendShapeChannelCount()
    uv_sets = mesh.GetElementUVCount()
    parts.append({"name": part["name"], "found": True,
                  "vertices_ok": len(points) == len(co), "vertex_mm": float(np.abs(points - co).max() * 1000) if len(points) == len(co) else None,
                  "polygons_ok": mesh.GetPolygonCount() == int(sound.sum()),
                  "skins": skins, "weighted_ok": len(weighted) == int((weights > 0).any(axis=1).sum()),
                  "shapes_ok": shapes == len(part["shapes"]), "uv_sets": uv_sets,
                  "uv_ok": uv_sets == len(part["uvs"]), "normals": mesh.GetElementNormalCount(),
                  "materials": node.GetMaterialCount()})
report["parts"] = len(parts)
report["parts_bad"] = [p for p in parts if not (p.get("found") and p["vertices_ok"] and p["polygons_ok"] and p["skins"] == 1
                                                and p["weighted_ok"] and p["shapes_ok"] and p["uv_ok"] and p["normals"])]
report["vertex_mm"] = max((p["vertex_mm"] or 0) for p in parts if p.get("found"))
report["bind_mm"] = bind_worst
textures = [scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxFileTexture.ClassId), i)
            for i in range(scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxFileTexture.ClassId)))]
report["textures"] = len(textures)
report["materials"] = scene.GetMaterialCount()
media = path.with_suffix(".fbm")
report["embedded_extracted"] = sorted(p.name for p in media.iterdir()) if media.is_dir() else []
placements = {}
for i in range(scene.GetMaterialCount()):
    material = scene.GetMaterial(i)
    diffuse = material.FindProperty(fbx.FbxSurfaceMaterial.sDiffuse)
    for t in range(diffuse.GetSrcObjectCount()):
        texture = diffuse.GetSrcObject(t)
        if (texture.GetScaleU(), texture.GetScaleV(), texture.GetTranslationU(), texture.GetTranslationV()) != (1, 1, 0, 0):
            placements[material.GetName()] = [texture.GetScaleU(), texture.GetScaleV(),
                                              texture.GetTranslationU(), texture.GetTranslationV()]
report["placements"] = placements
print(json.dumps(report, ensure_ascii=False, default=str))
