"""A converted scene as one glTF 2.0 binary file (``.glb``): skeleton, skinned meshes, shapes, embedded base colour maps.

Space: the scene is right-handed, Z up, metres; glTF is right-handed, Y up,
metres, so every point and frame goes through :data:`TO_Y_UP` (``x, z, -y``).
A frame is turned as a whole (``TO_Y_UP · rest``), not re-expressed in the
new axes: every bone keeps its own native local axes, and every bone but a
root keeps its native local transform exactly (the FBX Blender writes is the
same, ``endfield/fbxfile.FILE_TO_SCENE``).

Bones are nodes under one root node named after the rig; each keeps its local
rest transform as a ``matrix`` (the native frames are not always rigid and a
matrix loses nothing to a decomposition), and the skin's inverse bind matrices
are the inverses of the world rest frames.  A part is one mesh with one
primitive per material slot it uses; the primitives share the part's vertex
streams.  Triangles that name a vertex twice are left out (as the Blender
writer does) and counted.  UVs flip ``v`` (glTF's origin is the top left).
Joints and weights go four columns per set; a vertex no bone holds keeps zero
weights and stays where it is.  Shapes are sparse ``POSITION`` targets named
in ``extras.targetNames``, all at weight 0.

Materials: the base colour map is embedded (PNG or JPEG as they are; any other
image is converted to PNG) with its Unity scale and offset as
``KHR_texture_transform``; the material's other native maps and values are
kept in ``extras`` by name, not reinterpreted.  Where a scene says how a
material blends (Wuthering Waves: masked with a cutoff, a constant glass
colour, an empty pass), that becomes ``alphaMode``.  A part hidden at rest is
``KHR_node_visibility`` hidden and ``extras.hidden``.
"""
from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np

#: scene (Z up) → glTF (Y up)
TO_Y_UP = np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
FLOAT, UNSIGNED_SHORT, UNSIGNED_INT = 5126, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
TYPES = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}
#: the rules a file was written by, kept in ``asset.extras.rules``: a viewer says a file of older rules
#: should be exported again (1: coverage — overlays, transparency, cut-outs, eye layers; 2: GF2's eye layers
#: baked into the eye's map — §7.1455)
RULES = 2


class GlbError(ValueError):
    """A scene this writer cannot put in a file as it is."""


class _Builder:
    def __init__(self):
        self.blob = bytearray()
        self.views, self.accessors = [], []

    def view(self, data: bytes, target: int | None = None) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        view = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.blob += data
        self.views.append(view)
        return len(self.views) - 1

    def accessor(self, values: np.ndarray, component: int, target: int | None = None, bounds: bool = False,
                 normalized: bool = False) -> int:
        dtype = {FLOAT: np.float32, UNSIGNED_SHORT: np.uint16, UNSIGNED_INT: np.uint32}[component]
        values = np.ascontiguousarray(values, dtype=dtype)
        width = 1 if values.ndim == 1 else int(np.prod(values.shape[1:]))
        record = {"bufferView": self.view(values.tobytes(), target), "componentType": component,
                  "count": int(len(values)), "type": TYPES[width]}
        if normalized:
            record["normalized"] = True
        if bounds:
            flat = values.reshape(len(values), -1)
            record["min"] = flat.min(axis=0).tolist()
            record["max"] = flat.max(axis=0).tolist()
        self.accessors.append(record)
        return len(self.accessors) - 1

    def sparse(self, count: int, indices: np.ndarray, values: np.ndarray) -> int:
        """A VEC3 float accessor of *count* zeros but *values* at *indices* (a shape's deltas)."""
        order = np.argsort(indices, kind="stable")
        indices = np.ascontiguousarray(indices[order], dtype=np.uint32)
        values = np.ascontiguousarray(values[order], dtype=np.float32)
        dense_min = np.minimum(values.min(axis=0), 0.0) if len(values) else np.zeros(3)
        dense_max = np.maximum(values.max(axis=0), 0.0) if len(values) else np.zeros(3)
        record = {"componentType": FLOAT, "count": int(count), "type": "VEC3",
                  "min": dense_min.astype(float).tolist(), "max": dense_max.astype(float).tolist()}
        if len(indices):
            record["sparse"] = {"count": int(len(indices)),
                                "indices": {"bufferView": self.view(indices.tobytes()),
                                            "componentType": UNSIGNED_INT},
                                "values": {"bufferView": self.view(values.tobytes())}}
        self.accessors.append(record)
        return len(self.accessors) - 1


def _column_major(matrix: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(matrix, dtype=np.float64).T.ravel()]


def _linear(value: float) -> float:
    """One sRGB channel as glTF's linear factor."""
    value = min(max(float(value), 0.0), 1.0)
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _image(path: Path, alpha_from: str | None = None) -> tuple[bytes, str]:
    """*path*'s bytes as a PNG or JPEG; with *alpha_from* (``"r"`` …) a white PNG whose alpha is that channel."""
    if alpha_from in ("r", "g", "b"):
        from PIL import Image
        out = io.BytesIO()
        with Image.open(path) as image:
            coverage = image.convert("RGBA").getchannel("RGB".index(alpha_from.upper()))
        white = Image.new("RGBA", coverage.size, (255, 255, 255, 255))
        white.putalpha(coverage)
        white.save(out, format="PNG")
        return out.getvalue(), "image/png"
    data = path.read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return data, "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return data, "image/jpeg"
    from PIL import Image
    out = io.BytesIO()
    with Image.open(path) as image:
        image.save(out, format="PNG")
    return out.getvalue(), "image/png"


def _texture_transform(record: dict) -> dict | None:
    """A map's Unity scale and offset in glTF's flipped-v space; ``None`` when they change nothing."""
    scale, offset = record.get("scale") or [1, 1], record.get("offset") or [0, 0]
    if list(scale) == [1, 1] and list(offset) == [0, 0]:
        return None
    return {"scale": [float(scale[0]), float(scale[1])],
            "offset": [float(offset[0]), float(1 - scale[1] - offset[1])]}


def _material(name: str, record: dict, folder: Path, builder: _Builder, gltf: dict,
              images: dict) -> dict:
    material = {"name": name, "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.5},
                "doubleSided": True}
    extras = {key: value for key, value in record.items() if key not in ("maps",)}
    extras["maps"] = {key: {k: v for k, v in value.items() if k != "sha256"} for key, value in record["maps"].items()}
    base = record["maps"].get(record.get("base") or "")
    alpha_from = record.get("alpha_from") if record.get("alpha_from") in ("r", "g", "b") else None
    if base is not None:
        path = (folder / base["file"]).resolve()
        key = (path, alpha_from)
        if key not in images:
            data, mime = _image(path, alpha_from)
            gltf["images"].append({"name": path.stem + (f"_{alpha_from}_alpha" if alpha_from else ""),
                                   "bufferView": builder.view(data), "mimeType": mime})
            gltf["textures"].append({"source": len(gltf["images"]) - 1, "sampler": 0})
            images[key] = len(gltf["textures"]) - 1
        info = {"index": images[key]}
        transform = _texture_transform(base)
        if transform is not None:
            info["extensions"] = {"KHR_texture_transform": transform}
            gltf["extensionsUsed"].add("KHR_texture_transform")
        material["pbrMetallicRoughness"]["baseColorTexture"] = info
    native = record.get("native_material") or {}
    if record.get("tint") is not None:
        # a colour the base map is multiplied by, as the source gives it (sRGB); glTF's factor is linear
        tint = [float(v) for v in record["tint"]]
        material["pbrMetallicRoughness"]["baseColorFactor"] = [_linear(v) for v in tint[:3]] + [tint[3]]
    if record.get("cutoff") is not None and record.get("blend") is None and base is not None:
        material["alphaMode"] = "MASK"
        material["alphaCutoff"] = float(record["cutoff"])
    if record.get("blend") in ("multiply", "alpha", "add"):
        # glTF blends by coverage only: a multiplied overlay (an eye shadow) comes out as its tint laid on
        # by its coverage, not darkening what is under it; extras keep "blend" for a reader that can
        material["alphaMode"] = "BLEND"
    if record.get("blend") in ("multiply", "add") and not record.get("alpha_from"):
        # a colour layer multiplied or added onto what is behind (GF2's eye blends) has no coverage to lay on:
        # glTF cannot draw it, and drawn opaque it was a solid square, so it is kept but drawn invisible
        factor = material["pbrMetallicRoughness"].setdefault("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        factor[3] = 0.0
    if record.get("constant_color") is not None:
        material["pbrMetallicRoughness"]["baseColorFactor"] = [float(v) for v in record["constant_color"]]
        material["alphaMode"] = "BLEND"
    if record.get("native_surface_unmapped"):
        material["pbrMetallicRoughness"]["baseColorFactor"] = [0.0, 0.0, 0.0, 0.0]
        material["alphaMode"] = "BLEND"
    elif base is not None and native.get("BlendMode") == 1:
        cutoff = native.get("Properties", {}).get("BasePropertyOverrides", {}).get("OpacityMaskClipValue")
        if cutoff is None:
            raise GlbError(f"{name}: a masked material with no cutoff")
        material["alphaMode"] = "MASK"
        material["alphaCutoff"] = float(cutoff)
    material["extras"] = json.loads(json.dumps(extras, default=str))
    return material


def write(document: dict, arrays, folder: Path, target: Path) -> dict:
    """Write the scene (*document*, *arrays*; maps relative to *folder*) to *target*; what was left out."""
    folder, target = Path(folder), Path(target)
    builder = _Builder()
    gltf = {"asset": {"version": "2.0", "generator": "Polylode interchange", "extras": {"rules": RULES}}, "scene": 0,
            "scenes": [{"name": document["name"], "nodes": [0]}], "nodes": [], "meshes": [], "skins": [],
            "materials": [], "textures": [], "images": [], "samplers": [{"wrapS": 10497, "wrapT": 10497}],
            "extensionsUsed": set()}
    bones = document["bones"]
    rest = np.einsum("ij,njk->nik", TO_Y_UP, np.asarray(arrays["bone_rest"], dtype=np.float64))
    if len(rest) != len(bones):
        raise GlbError(f"{len(rest)} rest frames for {len(bones)} bones")
    root = {"name": document["name"], "children": [], "extras": {"humanoid": document.get("humanoid", {})}}
    gltf["nodes"].append(root)
    first = 1
    for index, bone in enumerate(bones):
        parent = bone["parent"]
        if not -1 <= parent < index:
            raise GlbError(f"bone {bone['name']} comes before its parent")
        local = rest[index] if parent < 0 else np.linalg.solve(rest[parent], rest[index])
        node = {"name": bone["name"], "matrix": _column_major(local)}
        if bone.get("native_name"):
            node["extras"] = {"native_name": bone["native_name"]}
        gltf["nodes"].append(node)
        if parent < 0:
            root["children"].append(first + index)
        else:
            gltf["nodes"][first + parent].setdefault("children", []).append(first + index)
    inverse = np.stack([np.linalg.inv(m) for m in rest])
    # the rig node: the common root of every bone, however many roots the rig has
    gltf["skins"].append({"name": document["name"], "skeleton": 0, "joints": list(range(first, first + len(bones))),
                          "inverseBindMatrices": builder.accessor(
                              np.stack([np.asarray(m, dtype=np.float64).T for m in inverse]).reshape(-1, 16), FLOAT)})
    if len(bones) > 65535:
        raise GlbError(f"{len(bones)} bones do not fit 16-bit joint indices")

    images: dict = {}
    material_index = {}
    for name, record in document["materials"].items():
        gltf["materials"].append(_material(name, record, folder, builder, gltf, images))
        material_index[name] = len(gltf["materials"]) - 1

    report = {"degenerate": {}, "unweighted": {}, "parts": len(document["parts"]), "bones": len(bones)}
    turn = TO_Y_UP[:3, :3]
    for index, part in enumerate(document["parts"]):
        p = f"p{index}_"
        co = np.asarray(arrays[p + "co"], dtype=np.float64) @ turn.T
        normals = np.asarray(arrays[p + "no"], dtype=np.float64) @ turn.T
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, length, out=np.zeros_like(normals), where=length > 0)
        attributes = {"POSITION": builder.accessor(co, FLOAT, ARRAY_BUFFER, bounds=True)}
        if (length > 0).all():
            attributes["NORMAL"] = builder.accessor(normals, FLOAT, ARRAY_BUFFER)
        if part.get("tangents") and p + "tan" in arrays:
            tangent = np.asarray(arrays[p + "tan"], dtype=np.float64).copy()
            tangent[:, :3] = tangent[:, :3] @ turn.T
            size = np.linalg.norm(tangent[:, :3], axis=1, keepdims=True)
            if (size > 0).all() and np.isin(np.sign(tangent[:, 3]), (-1, 1)).all():
                tangent[:, :3] /= size
                tangent[:, 3] = np.sign(tangent[:, 3])
                attributes["TANGENT"] = builder.accessor(tangent, FLOAT, ARRAY_BUFFER)
        for slot, channel in enumerate(part["uvs"]):
            uv = np.asarray(arrays[f"{p}uv{channel}"], dtype=np.float64).copy()
            uv[:, 1] = 1.0 - uv[:, 1]
            attributes[f"TEXCOORD_{slot}"] = builder.accessor(uv, FLOAT, ARRAY_BUFFER)
        if part.get("colors"):
            attributes["COLOR_0"] = builder.accessor(np.asarray(arrays[p + "col"], dtype=np.float64), FLOAT,
                                                     ARRAY_BUFFER)
        joints = np.asarray(arrays[p + "jnt"])
        weights = np.asarray(arrays[p + "wgt"], dtype=np.float64)
        weights = np.where(weights > 0, weights, 0.0)
        total = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
        joints = np.where(weights > 0, joints, 0)
        if joints.max(initial=0) >= len(bones):
            raise GlbError(f"{part['name']}: a weight on a bone the rig does not have")
        columns = -(-joints.shape[1] // 4) * 4
        joints = np.pad(joints, ((0, 0), (0, columns - joints.shape[1])))
        weights = np.pad(weights, ((0, 0), (0, columns - weights.shape[1])))
        for set_index in range(columns // 4):
            span = slice(4 * set_index, 4 * set_index + 4)
            attributes[f"JOINTS_{set_index}"] = builder.accessor(joints[:, span], UNSIGNED_SHORT, ARRAY_BUFFER)
            attributes[f"WEIGHTS_{set_index}"] = builder.accessor(weights[:, span], FLOAT, ARRAY_BUFFER)
        unweighted = int((total[:, 0] <= 0).sum())
        if unweighted:
            report["unweighted"][part["name"]] = unweighted
        targets = [{"POSITION": builder.sparse(len(co), np.asarray(arrays[f"{p}s{k}_idx"]),
                                               np.asarray(arrays[f"{p}s{k}_delta"], dtype=np.float64) @ turn.T)}
                   for k in range(len(part["shapes"]))]
        tri = np.asarray(arrays[p + "tri"])
        slots = np.asarray(arrays[p + "slot"])
        sound = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
        if (~sound).any():
            report["degenerate"][part["name"]] = int((~sound).sum())
        tri, slots = tri[sound], slots[sound]
        primitives = []
        for slot in np.unique(slots):
            primitive = {"attributes": attributes,
                         "indices": builder.accessor(tri[slots == slot].ravel(), UNSIGNED_INT, ELEMENT_ARRAY_BUFFER),
                         "mode": 4}
            name = part["materials"][int(slot)] if int(slot) < len(part["materials"]) else None
            if name:
                primitive["material"] = material_index[name]
            if targets:
                primitive["targets"] = targets
            primitives.append(primitive)
        mesh = {"name": part["name"], "primitives": primitives}
        if targets:
            mesh["weights"] = [0.0] * len(targets)
            mesh["extras"] = {"targetNames": list(part["shapes"])}
        gltf["meshes"].append(mesh)
        node = {"name": part["name"], "mesh": len(gltf["meshes"]) - 1, "skin": 0}
        if part.get("hidden"):
            node["extras"] = {"hidden": True}
            node["extensions"] = {"KHR_node_visibility": {"visible": False}}
            gltf["extensionsUsed"].add("KHR_node_visibility")
        gltf["nodes"].append(node)
        gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)

    for key in ("textures", "images", "materials"):
        if not gltf[key]:
            del gltf[key]
    if "textures" not in gltf:
        del gltf["samplers"]
    gltf["extensionsUsed"] = sorted(gltf["extensionsUsed"])
    if not gltf["extensionsUsed"]:
        del gltf["extensionsUsed"]
    gltf["buffers"] = [{"byteLength": len(builder.blob)}]
    gltf["bufferViews"], gltf["accessors"] = builder.views, builder.accessors
    while len(builder.blob) % 4:
        builder.blob.append(0)
    gltf["buffers"][0]["byteLength"] = len(builder.blob)
    text = json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        out.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(builder.blob)))
        out.write(struct.pack("<II", len(text), 0x4E4F534A))
        out.write(text)
        out.write(struct.pack("<II", len(builder.blob), 0x004E4942))
        out.write(builder.blob)
    return report


#: the read-back tolerances, in millimetres: the same as the FBX files' (``endfield/fbx_export.py``)
HEAD_MM, REST_MM, POSED_MM = 0.01, 0.01, 0.05
#: Blender's side of the read-back (run as a ``.py`` file)
CHECKER = Path(__file__).resolve().parents[2] / "tools" / "ef_fbx_blender.py"


def readback_problems(document: dict, arrays, report: dict) -> list[str]:
    """What Blender's glTF import of the file says is wrong with it (empty when nothing)."""
    names = [bone["name"] for bone in document["bones"]]
    out = []
    if report["armatures"] != 1:
        return [f"{report['armatures']} armatures"]
    if report["bones"] != len(names) or report["bones_missing"] or report["parents_wrong"]:
        out.append(f"{report['bones']} bones for {len(names)}, {len(report['bones_missing'])} missing, "
                   f"{report['parents_wrong']} under another parent")
    if report.get("bone_head_mm", 0.0) > HEAD_MM:
        out.append(f"bone heads off by {report['bone_head_mm']:.4f} mm")
    for index, (part, row) in enumerate(zip(document["parts"], report["parts"])):
        if not row["found"]:
            out.append(f"{part['name']}: not in the file")
            continue
        if row["vertices"] != len(arrays[f"p{index}_co"]):
            out.append(f"{part['name']}: {row['vertices']} vertices for {len(arrays[f'p{index}_co'])}")
            continue
        if not row["armature"] or row["weighted"] != row["weighted_expected"]:
            out.append(f"{part['name']}: {row['weighted']} of {row['weighted_expected']} vertices weighted")
        if row["rest_mm"] > REST_MM or row["posed_mm"] > POSED_MM:
            out.append(f"{part['name']}: rest off {row['rest_mm']:.4f} mm, posed off {row['posed_mm']:.4f} mm")
        if row["shape_keys"] != len(part["shapes"]):
            out.append(f"{part['name']}: {row['shape_keys']} shapes for {len(part['shapes'])}")
    return out


def write_checked(document_path: Path, target: Path, blender: Path | None = None) -> tuple[dict, list[str]]:
    """Write the scene of *document_path* as *target*; with *blender*, import it there and measure it.

    Returns the write's report (``readback`` holds Blender's, or ``None`` when no Blender was given) and
    the problems the read-back found.
    """
    import subprocess
    document_path, target = Path(document_path), Path(target)
    document = json.loads(document_path.read_text(encoding="utf-8"))
    arrays = np.load(document_path.parent / document["arrays"])
    report = write(document, arrays, document_path.parent, target)
    report["readback"] = None
    if blender is None:
        return report, []
    measured = target.with_name(target.name + ".check.json")
    measured.unlink(missing_ok=True)
    done = subprocess.run([str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
                           "--python", str(CHECKER), "--", "glb", str(document_path), str(target), str(measured)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if done.returncode != 0 or not measured.is_file():
        raise GlbError(f"Blender glb read-back exit {done.returncode}: {(done.stdout + done.stderr)[-1500:]}")
    report["readback"] = json.loads(measured.read_text(encoding="utf-8"))
    measured.unlink()
    return report, readback_problems(document, arrays, report["readback"])
