"""A converted scene as one binary FBX 7.4 file: skeleton, skinned meshes, shapes, materials with embedded base maps.

The file is the 7.4 binary layout (32-bit record headers; ``endfield/fbxfile.py``
reads the same): a header, the top-level records, a null record and the
footer.  Its space is Y up, metres (``GlobalSettings``: up Y, front Z, unit
scale 100 cm), the same turn as the ``.glb`` (:data:`glb.TO_Y_UP`, frames
turned as a whole so every bone keeps its native local axes).

What goes in:

* one ``Null`` model named after the rig at the top, every bone a
  ``LimbNode`` model under it with a ``Skeleton`` attribute; a bone's local
  transform is ``Lcl Translation``, ``Lcl Rotation`` (Euler XYZ, degrees:
  ``Rz · Ry · Rx``) and ``Lcl Scaling`` in doubles — a frame with shear cannot
  be written that way and is refused;
* one ``Mesh`` model per part under the rig, its geometry with polygon
  corners, normals, every UV set, colours, tangents and binormals (all by
  corner, direct), and the material of each triangle by slot; triangles that
  name a vertex twice are left out and counted;
* a ``Skin`` per part with a ``Cluster`` per bone it uses (``Transform`` the
  inverse of the bone's world rest, ``TransformLink`` that rest), and one
  ``BindPose`` for all of them;
* a ``BlendShape`` per part with shapes, one channel and one ``Shape``
  (sparse deltas) per shape, all at 0;
* a ``Material`` per material, its base colour map as a ``Texture`` with the
  image bytes in its ``Video``'s ``Content``; a material's other native maps
  and values are not reinterpreted (the ``.glb`` keeps them in ``extras``).
  A part hidden at rest has ``Visibility`` 0.  A base map's Unity tiling and offset are its texture's
  ``Scaling`` and ``Translation`` (FBX's UVs are v up like Unity's, so no flip as in the ``.glb``).
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from ..endfield.fbxfile import euler_xyz
from .glb import TO_Y_UP, _image

VERSION = 7400
MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
#: A file id, creation time and footer code that belong together: Autodesk's FBX SDK checks the footer
#: code against the other two and calls a file whose pair does not match corrupted (format constants;
#: Blender and other writers use the same three).
_FILE_ID = bytes.fromhex("28b32aebb624ccc2bfc8b02aa92bfcf1")
_TIME_ID = "1970-01-01 10:00:00:000"
_FOOT_ID = bytes.fromhex("fabcab09d0c8d466b176fb831cf7267e")
_FOOT_MAGIC = bytes.fromhex("f85a8c6adef5d97eece90ce3758f290b")
#: arrays at least this many bytes are stored zlib-compressed
_COMPRESS = 128


class FbxWriteError(ValueError):
    """A scene this writer cannot put in a file as it is."""


# -- the record layer -----------------------------------------------------------------------------------------

class Rec:
    """One record: a name, typed properties ``(code, value)``, child records."""

    __slots__ = ("name", "props", "children")

    def __init__(self, name: str, *props, children=None):
        self.name, self.props, self.children = name, props, list(children or [])

    def add(self, name: str, *props, children=None) -> "Rec":
        record = Rec(name, *props, children=children)
        self.children.append(record)
        return record


def _prop(code: str, value) -> bytes:
    if code in "YCIFDL":
        fmt = {"Y": "<h", "C": "<B", "I": "<i", "F": "<f", "D": "<d", "L": "<q"}[code]
        return code.encode() + struct.pack(fmt, int(value) if code in "YCIL" else float(value))
    if code in "SR":
        data = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        return code.encode() + struct.pack("<I", len(data)) + data
    dtype = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "u1"}[code]
    array = np.ascontiguousarray(np.asarray(value).ravel(), dtype=dtype)
    raw = array.tobytes()
    if len(raw) >= _COMPRESS:
        packed = zlib.compress(raw, 6)
        return code.encode() + struct.pack("<III", len(array), 1, len(packed)) + packed
    return code.encode() + struct.pack("<III", len(array), 0, len(raw)) + raw


def _encode(record: Rec, offset: int) -> bytes:
    props = b"".join(_prop(code, value) for code, value in record.props)
    name = record.name.encode("utf-8")
    head = 13 + len(name)
    body, at = [], offset + head + len(props)
    for child in record.children:
        encoded = _encode(child, at)
        body.append(encoded)
        at += len(encoded)
    if record.children or not record.props:      # a nested list, or nothing at all, ends with a null record
        body.append(b"\x00" * 13)
        at += 13
    return struct.pack("<IIIB", at, len(record.props), len(props), len(name)) + name + props + b"".join(body)


def _footer(length: int) -> bytes:
    pad = (16 - (length + len(_FOOT_ID)) % 16) % 16 or 16
    return _FOOT_ID + b"\x00" * pad + struct.pack("<I", 0) + struct.pack("<I", VERSION) + b"\x00" * 120 + _FOOT_MAGIC


def encode(records: list[Rec]) -> bytes:
    out = bytearray(MAGIC + struct.pack("<I", VERSION))
    for record in records:
        out += _encode(record, len(out))
    out += b"\x00" * 13
    out += _footer(len(out))
    return bytes(out)


# -- the scene ------------------------------------------------------------------------------------------------

def _p(parent: Rec, name: str, kind: str, label: str, flags: str, *values) -> None:
    codes = {"int": "I", "enum": "I", "bool": "I", "double": "D", "Number": "D", "KString": "S"}
    if kind in ("Lcl Translation", "Lcl Rotation", "Lcl Scaling", "Vector3D", "Vector", "ColorRGB", "Color"):
        props = [("D", v) for v in values]
    else:
        props = [(codes.get(kind, "D"), v) for v in values]
    parent.add("P", ("S", name), ("S", kind), ("S", label), ("S", flags), *props)


def _named(name: str, kind: str) -> tuple:
    return ("S", f"{name}\x00\x01{kind}")


def _matrix(matrix: np.ndarray) -> tuple:
    """FBX keeps a matrix row by row with the translation in the last row (the transpose)."""
    return ("d", np.asarray(matrix, dtype=np.float64).T.ravel())


def _trs(local: np.ndarray, name: str) -> tuple:
    linear = local[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    if np.linalg.det(linear) <= 0 or np.any(scale <= 0):
        raise FbxWriteError(f"bone {name}: a mirrored or collapsed rest frame")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise FbxWriteError(f"bone {name}: a sheared rest frame has no Lcl transform")
    return local[:3, 3], euler_xyz(rotation), scale


class _Ids:
    def __init__(self):
        self.next = 1_000_000

    def __call__(self) -> int:
        self.next += 1
        return self.next


def write(document: dict, arrays, folder: Path, target: Path) -> dict:
    """Write the scene (*document*, *arrays*; maps relative to *folder*) to *target*; what was left out."""
    folder, target = Path(folder), Path(target)
    uid = _Ids()
    objects = Rec("Objects")
    links: list[tuple] = []
    counts = {"Model": 0, "NodeAttribute": 0, "Geometry": 0, "Material": 0, "Texture": 0, "Video": 0,
              "Deformer": 0, "Pose": 0}

    def link(child: int, parent: int, prop: str | None = None) -> None:
        links.append((child, parent, prop))

    def model(name: str, kind: str, local: np.ndarray | None, visible: bool = True) -> tuple[int, Rec]:
        ident = uid()
        record = objects.add("Model", ("L", ident), _named(name, "Model"), ("S", kind))
        record.add("Version", ("I", 232))
        props = record.add("Properties70")
        if local is not None:
            translation, rotation, scale = _trs(local, name)
            _p(props, "Lcl Translation", "Lcl Translation", "", "A", *translation)
            _p(props, "Lcl Rotation", "Lcl Rotation", "", "A", *rotation)
            _p(props, "Lcl Scaling", "Lcl Scaling", "", "A", *scale)
        _p(props, "DefaultAttributeIndex", "int", "Integer", "", 0)
        _p(props, "InheritType", "enum", "", "", 1)
        if not visible:
            _p(props, "Visibility", "Visibility", "", "A", 0.0)
        record.add("Shading", ("C", 1))
        record.add("Culling", ("S", "CullingOff"))
        counts["Model"] += 1
        return ident, record

    bones = document["bones"]
    rest = np.einsum("ij,njk->nik", TO_Y_UP, np.asarray(arrays["bone_rest"], dtype=np.float64))
    if len(rest) != len(bones):
        raise FbxWriteError(f"{len(rest)} rest frames for {len(bones)} bones")
    rig, _ = model(document["name"], "Null", np.eye(4))
    attribute = uid()
    objects.add("NodeAttribute", ("L", attribute), _named(document["name"], "NodeAttribute"), ("S", "Null"),
                children=[Rec("TypeFlags", ("S", "Null"))])
    counts["NodeAttribute"] += 1
    link(attribute, rig)
    link(rig, 0)
    bone_ids = []
    for index, bone in enumerate(bones):
        parent = bone["parent"]
        if not -1 <= parent < index:
            raise FbxWriteError(f"bone {bone['name']} comes before its parent")
        local = rest[index] if parent < 0 else np.linalg.solve(rest[parent], rest[index])
        ident, _ = model(bone["name"], "LimbNode", local)
        attribute = uid()
        objects.add("NodeAttribute", ("L", attribute), _named(bone["name"], "NodeAttribute"), ("S", "LimbNode"),
                    children=[Rec("TypeFlags", ("S", "Skeleton"))])
        counts["NodeAttribute"] += 1
        link(attribute, ident)
        link(ident, rig if parent < 0 else bone_ids[parent])
        bone_ids.append(ident)

    images: dict = {}
    videos: dict = {}
    materials: dict[str | None, int] = {}

    def material(name: str | None) -> int:
        if name in materials:
            return materials[name]
        if name and name not in document["materials"]:
            raise FbxWriteError(f"a part's slot names material {name}, which the scene does not have")
        record = document["materials"][name] if name else {"maps": {}, "base": None}
        ident = uid()
        rec = objects.add("Material", ("L", ident), _named(name or "none", "Material"), ("S", ""))
        rec.add("Version", ("I", 102))
        rec.add("ShadingModel", ("S", "Phong"))
        rec.add("MultiLayer", ("I", 0))
        props = rec.add("Properties70")
        # the source's tint (an overlay's colour, as Unity's material holds it) or a constant glass colour
        colour = record.get("tint") or record.get("constant_color") or [1.0, 1.0, 1.0, 1.0]
        _p(props, "DiffuseColor", "Color", "", "A", *colour[:3])
        _p(props, "DiffuseFactor", "Number", "", "A", 1.0)
        # a colour layer multiplied or added onto what is behind cannot be drawn by FBX: kept, invisible (as in glTF)
        unblendable = record.get("blend") in ("multiply", "add") and not record.get("alpha_from")
        if record.get("native_surface_unmapped") or record.get("constant_color") is not None \
                or record.get("blend") in ("multiply", "alpha", "add"):
            _p(props, "Opacity", "double", "Number", "",
               0.0 if record.get("native_surface_unmapped") or unblendable else colour[3])
        counts["Material"] += 1
        base = record["maps"].get(record.get("base") or "")
        alpha_from = record.get("alpha_from") if record.get("alpha_from") in ("r", "g", "b") else None

        def texture_of(entry: dict, alpha_from) -> int:
            path = (folder / entry["file"]).resolve()
            scale = tuple(float(v) for v in (entry.get("scale") or [1, 1]))
            offset = tuple(float(v) for v in (entry.get("offset") or [0, 0]))
            if (path, alpha_from) not in videos:          # the image bytes once per file (and coverage channel)
                data, _mime = _image(path, alpha_from)
                file_name = path.name if data[:4] == b"\x89PNG" or path.suffix.lower() in (".png", ".jpg", ".jpeg") \
                    else path.with_suffix(".png").name
                if alpha_from:
                    file_name = f"{path.stem}_{alpha_from}_alpha.png"
                video = uid()
                vid = objects.add("Video", ("L", video), _named(path.stem, "Video"), ("S", "Clip"))
                vid.add("Type", ("S", "Clip"))
                vid.add("Properties70", children=[Rec("P", ("S", "Path"), ("S", "KString"), ("S", "XRefUrl"),
                                                      ("S", ""), ("S", file_name))])
                vid.add("UseMipMap", ("I", 0))
                vid.add("Filename", ("S", file_name))
                vid.add("RelativeFilename", ("S", file_name))
                vid.add("Content", ("R", data))
                counts["Video"] += 1
                videos[(path, alpha_from)] = (video, file_name)
            key = (path, alpha_from, scale, offset)
            if key not in images:           # a texture per file and placement: two materials may tile one image apart
                video, file_name = videos[(path, alpha_from)]
                texture = uid()
                tex = objects.add("Texture", ("L", texture), _named(path.stem, "Texture"), ("S", ""))
                tex.add("Type", ("S", "TextureVideoClip"))
                tex.add("Version", ("I", 202))
                tex.add("TextureName", _named(path.stem, "Texture"))
                tex.add("Media", _named(path.stem, "Video"))
                tex.add("FileName", ("S", file_name))
                tex.add("RelativeFilename", ("S", file_name))
                tex.add("ModelUVTranslation", ("D", 0.0), ("D", 0.0))
                tex.add("ModelUVScaling", ("D", 1.0), ("D", 1.0))
                tex.add("Texture_Alpha_Source", ("S", "None"))
                tex.add("Cropping", ("I", 0), ("I", 0), ("I", 0), ("I", 0))
                if scale != (1.0, 1.0) or offset != (0.0, 0.0):
                    # Unity's tiling and offset: FBX keeps UVs with v up as Unity does, so they go in as they are
                    props = tex.add("Properties70")
                    _p(props, "Translation", "Vector", "", "A", offset[0], offset[1], 0.0)
                    _p(props, "Scaling", "Vector", "", "A", scale[0], scale[1], 1.0)
                link(video, texture)
                counts["Texture"] += 1
                images[key] = texture
            return images[key]

        if base is not None:
            texture = texture_of(base, alpha_from)
            if any(float(v) for v in colour[:3]):
                # a black tint is a surface with no colour of its own (an effect drawn as its emission): its map
                # is only its coverage
                link(texture, ident, "DiffuseColor")
            if alpha_from or record.get("blend") in ("multiply", "alpha", "add") or record.get("cutoff") is not None:
                # the coverage is the image's alpha: Unity and Blender read a material as transparent from this
                link(texture, ident, "TransparentColor")
        emission = record.get("emission")
        if emission:
            _p(props, "EmissiveColor", "Color", "", "A", *[float(v) for v in emission.get("color", (1, 1, 1))[:3]])
            _p(props, "EmissiveFactor", "Number", "", "A", float(emission.get("strength", 1.0)))
            entry = record["maps"].get(emission.get("map") or "")
            if entry is not None:
                link(texture_of(entry, None), ident, "EmissiveColor")
        materials[name] = ident
        return ident

    pose_nodes = [(bone_ids[i], rest[i]) for i in range(len(bones))]
    report = {"degenerate": {}, "parts": len(document["parts"]), "bones": len(bones)}
    turn = TO_Y_UP[:3, :3]
    for index, part in enumerate(document["parts"]):
        p = f"p{index}_"
        co = np.asarray(arrays[p + "co"], dtype=np.float64) @ turn.T
        tri = np.asarray(arrays[p + "tri"])
        slots = np.asarray(arrays[p + "slot"])
        sound = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
        if (~sound).any():
            report["degenerate"][part["name"]] = int((~sound).sum())
        tri, slots = tri[sound], slots[sound]
        corner = tri.ravel()
        mesh_id, _ = model(part["name"], "Mesh", np.eye(4), visible=not part.get("hidden"))
        link(mesh_id, rig)
        pose_nodes.append((mesh_id, np.eye(4)))
        geometry = uid()
        geo = objects.add("Geometry", ("L", geometry), _named(part["name"], "Geometry"), ("S", "Mesh"))
        counts["Geometry"] += 1
        link(geometry, mesh_id)
        geo.add("Vertices", ("d", co))
        polygons = tri.astype(np.int64).copy()
        polygons[:, 2] = -polygons[:, 2] - 1                       # the last corner of a polygon is ~index
        geo.add("PolygonVertexIndex", ("i", polygons))
        geo.add("GeometryVersion", ("I", 124))
        normals = np.asarray(arrays[p + "no"], dtype=np.float64) @ turn.T
        layer_elements = [("LayerElementNormal", 0)]
        element = geo.add("LayerElementNormal", ("I", 0))
        element.add("Version", ("I", 101))
        element.add("Name", ("S", ""))
        element.add("MappingInformationType", ("S", "ByPolygonVertex"))
        element.add("ReferenceInformationType", ("S", "Direct"))
        element.add("Normals", ("d", normals[corner]))
        if part.get("tangents") and p + "tan" in arrays:
            tangent = np.asarray(arrays[p + "tan"], dtype=np.float64)
            along = tangent[:, :3] @ turn.T
            binormal = np.sign(tangent[:, 3:4]) * np.cross(normals, along)
            for name, values in (("LayerElementTangent", along), ("LayerElementBinormal", binormal)):
                element = geo.add(name, ("I", 0))
                element.add("Version", ("I", 101))
                element.add("Name", ("S", "UV0"))
                element.add("MappingInformationType", ("S", "ByPolygonVertex"))
                element.add("ReferenceInformationType", ("S", "Direct"))
                element.add("Tangents" if name == "LayerElementTangent" else "Binormals", ("d", values[corner]))
                layer_elements.append((name, 0))
        for slot, channel in enumerate(part["uvs"]):
            uv = np.asarray(arrays[f"{p}uv{channel}"], dtype=np.float64)
            element = geo.add("LayerElementUV", ("I", slot))
            element.add("Version", ("I", 101))
            element.add("Name", ("S", f"UV{channel}"))
            element.add("MappingInformationType", ("S", "ByPolygonVertex"))
            element.add("ReferenceInformationType", ("S", "IndexToDirect"))
            element.add("UV", ("d", uv))
            element.add("UVIndex", ("i", corner))
            layer_elements.append(("LayerElementUV", slot))
        if part.get("colors"):
            element = geo.add("LayerElementColor", ("I", 0))
            element.add("Version", ("I", 101))
            element.add("Name", ("S", "Col"))
            element.add("MappingInformationType", ("S", "ByPolygonVertex"))
            element.add("ReferenceInformationType", ("S", "IndexToDirect"))
            element.add("Colors", ("d", np.asarray(arrays[p + "col"], dtype=np.float64)))
            element.add("ColorIndex", ("i", corner))
            layer_elements.append(("LayerElementColor", 0))
        # the model's materials in slot order: a triangle's material index is its slot's place in that order
        slot_names = list(part["materials"])
        used = sorted(set(int(s) for s in slots))
        if used and used[-1] >= len(slot_names):
            raise FbxWriteError(f"{part['name']}: a triangle in slot {used[-1]} of {len(slot_names)}")
        connected: list[str | None] = []
        place = {}
        for slot in range(len(slot_names)):
            name = slot_names[slot] or None
            if name not in connected:
                connected.append(name)
            place[slot] = connected.index(name)
        for name in connected:
            link(material(name), mesh_id)
        element = geo.add("LayerElementMaterial", ("I", 0))
        element.add("Version", ("I", 101))
        element.add("Name", ("S", ""))
        element.add("MappingInformationType", ("S", "ByPolygon"))
        element.add("ReferenceInformationType", ("S", "IndexToDirect"))
        element.add("Materials", ("i", np.array([place[int(s)] for s in slots], dtype=np.int32)))
        layer_elements.append(("LayerElementMaterial", 0))
        layers: dict[int, list] = {}
        for name, typed in layer_elements:
            layers.setdefault(typed if name == "LayerElementUV" else 0, []).append((name, typed))
        for number, elements in sorted(layers.items()):
            layer = geo.add("Layer", ("I", number))
            layer.add("Version", ("I", 100))
            for name, typed in elements:
                layer.add("LayerElement", children=[Rec("Type", ("S", name)), Rec("TypedIndex", ("I", typed))])

        joints = np.asarray(arrays[p + "jnt"])
        weights = np.asarray(arrays[p + "wgt"], dtype=np.float64)
        if joints.size and joints[weights > 0].max(initial=0) >= len(bones):
            raise FbxWriteError(f"{part['name']}: a weight on a bone the rig does not have")
        skin = uid()
        objects.add("Deformer", ("L", skin), _named(part["name"], "Deformer"), ("S", "Skin"),
                    children=[Rec("Version", ("I", 101)), Rec("Link_DeformAcuracy", ("D", 50.0))])
        counts["Deformer"] += 1
        link(skin, geometry)
        for bone in np.unique(joints[weights > 0]):
            vertices, columns = np.nonzero((joints == bone) & (weights > 0))
            order = np.argsort(vertices, kind="stable")
            cluster = uid()
            objects.add("Deformer", ("L", cluster), _named(bones[int(bone)]["name"], "SubDeformer"),
                        ("S", "Cluster"),
                        children=[Rec("Version", ("I", 100)), Rec("UserData", ("S", ""), ("S", "")),
                                  Rec("Indexes", ("i", vertices[order])),
                                  Rec("Weights", ("d", weights[vertices, columns][order])),
                                  Rec("Transform", _matrix(np.linalg.inv(rest[int(bone)]))),
                                  Rec("TransformLink", _matrix(rest[int(bone)]))])
            counts["Deformer"] += 1
            link(cluster, skin)
            link(bone_ids[int(bone)], cluster)

        if part["shapes"]:
            blend = uid()
            objects.add("Deformer", ("L", blend), _named(part["name"], "Deformer"), ("S", "BlendShape"),
                        children=[Rec("Version", ("I", 100))])
            counts["Deformer"] += 1
            link(blend, geometry)
            for k, name in enumerate(part["shapes"]):
                channel = uid()
                objects.add("Deformer", ("L", channel), _named(name, "SubDeformer"), ("S", "BlendShapeChannel"),
                            children=[Rec("Version", ("I", 100)), Rec("DeformPercent", ("D", 0.0)),
                                      Rec("FullWeights", ("d", np.array([100.0])))])
                counts["Deformer"] += 1
                shape = uid()
                indices = np.asarray(arrays[f"{p}s{k}_idx"])
                deltas = np.asarray(arrays[f"{p}s{k}_delta"], dtype=np.float64) @ turn.T
                order = np.argsort(indices, kind="stable")
                objects.add("Geometry", ("L", shape), _named(name, "Geometry"), ("S", "Shape"),
                            children=[Rec("Version", ("I", 100)), Rec("Indexes", ("i", indices[order])),
                                      Rec("Vertices", ("d", deltas[order]))])
                counts["Geometry"] += 1
                link(shape, channel)
                link(channel, blend)

    pose = uid()
    rec = objects.add("Pose", ("L", pose), _named("BindPose", "Pose"), ("S", "BindPose"))
    rec.add("Type", ("S", "BindPose"))
    rec.add("Version", ("I", 100))
    rec.add("NbPoseNodes", ("I", len(pose_nodes)))
    for node, matrix in pose_nodes:
        rec.add("PoseNode", children=[Rec("Node", ("L", node)), Rec("Matrix", _matrix(matrix))])
    counts["Pose"] = 1

    records = _header(document["name"])
    definitions = Rec("Definitions", children=[Rec("Version", ("I", 100)),
                                               Rec("Count", ("I", sum(counts.values()) + 1))])
    definitions.add("ObjectType", ("S", "GlobalSettings"), children=[Rec("Count", ("I", 1))])
    for kind, count in counts.items():
        if count:
            definitions.add("ObjectType", ("S", kind), children=[Rec("Count", ("I", count))])
    connections = Rec("Connections")
    for child, parent, prop in links:
        if prop is None:
            connections.add("C", ("S", "OO"), ("L", child), ("L", parent))
        else:
            connections.add("C", ("S", "OP"), ("L", child), ("L", parent), ("S", prop))
    records += [definitions, objects, connections, Rec("Takes", children=[Rec("Current", ("S", ""))])]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encode(records))
    return report


def _header(name: str) -> list[Rec]:
    stamp = Rec("CreationTimeStamp", children=[Rec("Version", ("I", 1000))] + [
        Rec(key, ("I", value)) for key, value in (("Year", 2000), ("Month", 1), ("Day", 1), ("Hour", 0),
                                                 ("Minute", 0), ("Second", 0), ("Millisecond", 0))])
    header = Rec("FBXHeaderExtension", children=[Rec("FBXHeaderVersion", ("I", 1003)), Rec("FBXVersion", ("I", VERSION)),
                                                 Rec("EncryptionType", ("I", 0)), stamp,
                                                 Rec("Creator", ("S", "Polylode interchange"))])
    settings = Rec("GlobalSettings", children=[Rec("Version", ("I", 1000))])
    props = settings.add("Properties70")
    for key, value in (("UpAxis", 1), ("UpAxisSign", 1), ("FrontAxis", 2), ("FrontAxisSign", 1), ("CoordAxis", 0),
                       ("CoordAxisSign", 1), ("OriginalUpAxis", 1), ("OriginalUpAxisSign", 1)):
        _p(props, key, "int", "Integer", "", value)
    _p(props, "UnitScaleFactor", "double", "Number", "", 100.0)
    _p(props, "OriginalUnitScaleFactor", "double", "Number", "", 100.0)
    document = Rec("Documents", children=[Rec("Count", ("I", 1))])
    scene = document.add("Document", ("L", 1), ("S", ""), ("S", "Scene"))
    scene.add("Properties70", children=[Rec("P", ("S", "SourceObject"), ("S", "object"), ("S", ""), ("S", "")),
                                        Rec("P", ("S", "ActiveAnimStackName"), ("S", "KString"), ("S", ""), ("S", ""),
                                            ("S", ""))])
    scene.add("RootNode", ("L", 0))
    return [header, Rec("FileId", ("R", _FILE_ID)), Rec("CreationTime", ("S", _TIME_ID)),
            Rec("Creator", ("S", f"Polylode interchange ({name})")), settings, document, Rec("References")]


#: our own reader's tolerances for the bones and the skin's bind matrices (``endfield/fbx_export.py``'s)
BONE_MM, BONE_DEG = 0.01, 0.01


def write_checked(document_path: Path, target: Path, blender: Path | None = None) -> tuple[dict, list[str]]:
    """Write the scene of *document_path* as *target* and read it back; the report and the problems found.

    Always: the file's own nodes and skin clusters, read by ``endfield/fbxfile.py``, against the scene's
    rest frames.  With *blender*: Blender's FBX import, measured as the ``.glb``'s
    (``glb.readback_problems``).
    """
    import json
    import subprocess

    from ..endfield import fbxfile
    from .glb import CHECKER, readback_problems
    document_path, target = Path(document_path), Path(target)
    document = json.loads(document_path.read_text(encoding="utf-8"))
    arrays = np.load(document_path.parent / document["arrays"])
    report = write(document, arrays, document_path.parent, target)
    bones = [(bone["name"], bone["parent"], np.asarray(arrays["bone_rest"][i]))
             for i, bone in enumerate(document["bones"])]
    measured = fbxfile.bone_errors(target, bones)
    report["file"] = measured
    problems = []
    if measured["missing"] or measured["bones"] != len(bones):
        problems.append(f"{measured['bones']} bones in the file for {len(bones)}")
    if measured["offset_mm"] > BONE_MM or measured["turn_deg"] > BONE_DEG:
        problems.append(f"bones off by {measured['offset_mm']:.4f} mm, {measured['turn_deg']:.4f} deg")
    if measured["bind_offset_mm"] > BONE_MM or measured["bind_turn_deg"] > BONE_DEG:
        problems.append(f"bind matrices off by {measured['bind_offset_mm']:.4f} mm, {measured['bind_turn_deg']:.4f} deg")
    report["readback"] = None
    if blender is None:
        return report, problems
    out = target.with_name(target.name + ".check.json")
    out.unlink(missing_ok=True)
    done = subprocess.run([str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
                           "--python", str(CHECKER), "--", "check", str(document_path), str(target), str(out)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if done.returncode != 0 or not out.is_file():
        raise FbxWriteError(f"Blender FBX read-back exit {done.returncode}: {(done.stdout + done.stderr)[-1500:]}")
    report["readback"] = json.loads(out.read_text(encoding="utf-8"))
    out.unlink()
    return report, problems + readback_problems(document, arrays, report["readback"])
