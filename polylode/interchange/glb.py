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
colour, an empty pass), that becomes ``alphaMode``.  A part hidden at rest is not written (``shown``).

Animations: a scene that carries clips (``document["animations"]``, each one's bone frames per frame in
``arrays["anim<k>"]``, in the space of ``bone_rest``: an Endfield character's display poses, ``endfield/poses``)
writes each as a glTF animation, the bones' local translation, rotation and (a bone a clip grows, shrinks or
stretches, ``gf2.clips.skinned``) scale.  A key is kept only where leaving it
out would move its bone by more than :data:`KEY_METRES` or turn it by more than :data:`KEY_QUATERNION` from the
line glTF draws between the keys kept (:func:`keys`): written whole, every bone at every frame, a GF2 outfit's 34
poses came to 105 MB, over GitHub's 100 MB a file.
glTF animates only a node given as translation, rotation and scale, so its bones are written that way instead of
as a ``matrix`` — exact for those frames, which are rigid (``rigscene.rest_matrix``); a channel is written only for
a bone the clip moves off its rest.
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
#: baked into the eye's map — §7.1455; 3: the sources' vertex colours as ``_SOURCE_COLOR``, not ``COLOR_0``, and GF2's
#: lower LOD levels hidden; 4: Wuthering Waves' one mesh as a part per material section)
RULES = 4
#: where a source's vertex colours go: an application attribute (the spec's leading underscore), not
#: ``COLOR_0``, which every glTF viewer multiplies into the base colour.  The three games keep shader data
#: there (GF2's are outline normals, about (0.5, 0.5, 1)), never a colour their materials multiply by; as
#: ``COLOR_0`` they tinted a model blue in a viewer that follows the spec, and Qt's loader drew GF2's
#: characters not at all
VERTEX_DATA = "_SOURCE_COLOR"


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
    elif base is not None and native.get("BlendMode") == 1 and record.get("blend") is None \
            and record.get("cutoff") is None:
        cutoff = native.get("Properties", {}).get("BasePropertyOverrides", {}).get("OpacityMaskClipValue")
        if cutoff is None:
            raise GlbError(f"{name}: a masked material with no cutoff")
        material["alphaMode"] = "MASK"
        material["alphaCutoff"] = float(cutoff)
    emission = record.get("emission")
    if emission:
        # light the surface gives off whatever lights it (an unlit effect, an emission map): its map as
        # emissiveTexture, its colour as the factor, a strength over 1 as KHR_materials_emissive_strength
        entry = record["maps"].get(emission.get("map") or "")
        if entry is not None:
            path = (folder / entry["file"]).resolve()
            key = (path, None)
            if key not in images:
                data, mime = _image(path, None)
                gltf["images"].append({"name": path.stem, "bufferView": builder.view(data), "mimeType": mime})
                gltf["textures"].append({"source": len(gltf["images"]) - 1, "sampler": 0})
                images[key] = len(gltf["textures"]) - 1
            material["emissiveTexture"] = {"index": images[key]}
        material["emissiveFactor"] = [min(1.0, float(v)) for v in emission.get("color", (1.0, 1.0, 1.0))[:3]]
        strength = float(emission.get("strength", 1.0))
        if strength > 1.0:
            material.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {"emissiveStrength": strength}
            gltf["extensionsUsed"].add("KHR_materials_emissive_strength")
    material["extras"] = json.loads(json.dumps(extras, default=str))
    return material


#: how far a channel may stay from the bone's rest and not be written: metres, and quaternion components
STILL = 1e-6
#: how far a dropped key may sit from the line between the keys kept: metres of translation, and quaternion
#: components (1e-4 is about 0.01°)
KEY_METRES, KEY_QUATERNION = 1e-4, 1e-4
#: the same two for a bone's scale, a factor: the frames come as float32, whose rigid 3×3 is 1 within ~1e-7
STILL_SCALE, KEY_SCALE = 1e-5, 1e-4


def keys(values: np.ndarray, tolerance: float) -> np.ndarray:
    """The frames of *values* ``(frames, n)`` to keep as keys: the first, the last, and every frame whose leaving out
    would put the line between the keys kept further than *tolerance* from it (Ramer–Douglas–Peucker, each
    component's distance on its own); the frames left out are within *tolerance* of that line."""
    count = len(values)
    if count <= 2:
        return np.arange(count)
    kept = np.zeros(count, dtype=bool)
    kept[[0, -1]] = True
    spans = [(0, count - 1)]
    while spans:
        a, b = spans.pop()
        if b - a < 2:
            continue
        t = (np.arange(a + 1, b) - a)[:, None] / (b - a)
        line = values[a] + t * (values[b] - values[a])
        error = np.abs(values[a + 1:b] - line).max(axis=1)
        worst = int(np.argmax(error))
        if error[worst] > tolerance:
            middle = a + 1 + worst
            kept[middle] = True
            spans += [(a, middle), (middle, b)]
    return np.flatnonzero(kept)


def _quaternions(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` as unit quaternions ``(..., 4)``, ``x y z w`` (glTF's order)."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    # each of the four forms is taken where its divisor is largest (Shepperd), so none divides by a small number
    candidates = np.stack([
        np.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1], 1 + trace], -1),
        np.stack([1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0],
                  m[..., 2, 1] - m[..., 1, 2]], -1),
        np.stack([m[..., 0, 1] + m[..., 1, 0], 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], m[..., 1, 2] + m[..., 2, 1],
                  m[..., 0, 2] - m[..., 2, 0]], -1),
        np.stack([m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
                  m[..., 1, 0] - m[..., 0, 1]], -1)], -2)
    pick = np.argmax(np.stack([trace, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1), axis=-1)
    q = np.take_along_axis(candidates, pick[..., None, None], axis=-2)[..., 0, :]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., 3:] < 0, -q, q)


def _trs(local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local frames ``(..., 4, 4)`` as translation, rotation (``xyzw``) and scale (each axis's length)."""
    scale = np.linalg.norm(local[..., :3, :3], axis=-2)
    return local[..., :3, 3], _quaternions(local[..., :3, :3] / scale[..., None, :]), scale


def _trs_node(local: np.ndarray) -> dict:
    translation, rotation, scale = _trs(local)
    node = {"translation": translation.tolist(), "rotation": rotation.tolist()}
    if np.abs(scale - 1).max() > STILL:
        node["scale"] = scale.tolist()
    return node


def _matrices(translation: np.ndarray, rotation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``(..., 4, 4)`` from translation, rotation (``xyzw``) and scale, as glTF composes a node: T·R·S."""
    x, y, z, w = np.moveaxis(rotation, -1, 0)
    out = np.zeros(translation.shape[:-1] + (4, 4))
    out[..., 0, :3] = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1)
    out[..., 1, :3] = np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1)
    out[..., 2, :3] = np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)
    out[..., :3, :3] *= scale[..., None, :]
    out[..., :3, 3] = translation
    out[..., 3, 3] = 1.0
    return out


def _fitted(local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local frames ``(..., 4, 4)`` as the translation, rotation and scale that come nearest them: the rotation nearest
    the 3×3 (its polar factor) and the scale along that rotation's axes.  Exact for a frame that is a rotation and
    a scale; for one with shear (below a bone scaled unevenly), the TRS closest to it, where :func:`_trs`'s axis
    lengths and the rotation of the columns divided by them drifted off with each level of a stretched chain."""
    m = np.asarray(local, dtype=np.float64)[..., :3, :3]
    u, _size, vt = np.linalg.svd(m)
    u[..., :, -1] *= np.where(np.linalg.det(u @ vt) < 0, -1.0, 1.0)[..., None]
    turn = u @ vt
    scale = np.einsum("...ji,...ji->...i", turn, m)
    return np.asarray(local, dtype=np.float64)[..., :3, 3], _quaternions(turn), scale


def _local_trs(frames: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bones' local translation, rotation and scale ``(frames, bones, …)`` that put each bone at its frame.

    Each is taken against its parent as glTF will place it (its own TRS composed down the chain), not the parent's
    frame itself: below a bone scaled unevenly, a child's turn carries a shear no TRS holds, and measured against
    the frames that small miss added up down every chain — Cheyanne SSR0101's skirt tips sat 106 mm off.  Taken so,
    each bone stands exactly at its place and only its own turn is approximate.  Rigid frames come out the same."""
    count = frames.shape[1]
    placed = frames.copy()
    translation = np.empty(frames.shape[:2] + (3,))
    rotation = np.empty(frames.shape[:2] + (4,))
    scale = np.empty(frames.shape[:2] + (3,))
    done = np.zeros(count, dtype=bool)

    def visit(bone: int) -> None:
        if done[bone]:
            return
        parent = parents[bone]
        if parent >= 0:
            visit(parent)
            local = np.linalg.solve(placed[:, parent], frames[:, bone])
        else:
            local = frames[:, bone]
        translation[:, bone], rotation[:, bone], scale[:, bone] = _fitted(local)
        made = _matrices(translation[:, bone], rotation[:, bone], scale[:, bone])
        placed[:, bone] = made if parent < 0 else placed[:, parent] @ made
        done[bone] = True

    for bone in range(count):
        visit(bone)
    return translation, rotation, scale


def _animations(document: dict, arrays, rest: np.ndarray, builder: _Builder, first: int) -> list[dict]:
    """``document["animations"]`` as glTF animations of the bone nodes (numbered from *first*)."""
    parents = np.array([bone["parent"] for bone in document["bones"]])
    child = parents >= 0
    rest_local = rest.copy()
    rest_local[child] = np.linalg.solve(rest[parents[child]], rest[child])
    rest_t, rest_q, _rest_s = _trs(rest_local)
    out = []
    for k, record in enumerate(document["animations"]):
        frames = np.einsum("ij,fnjk->fnik", TO_Y_UP, np.asarray(arrays[f"anim{k}"], dtype=np.float64))
        if frames.shape[1:] != rest.shape:
            raise GlbError(f"animation {record['name']}: {frames.shape[1]} bones for {len(rest)}")
        translation, rotation, scale = _local_trs(frames, parents)
        # one hemisphere along the clip, so interpolation between frames takes the short way
        for f in range(1, len(rotation)):
            flip = np.sum(rotation[f] * rotation[f - 1], axis=-1) < 0
            rotation[f, flip] *= -1
        seconds = np.arange(len(frames)) / float(record["rate"])
        times = {}                                          # one time accessor per set of keys kept
        channels, samplers = [], []
        for bone in range(len(rest)):
            moved_t = np.abs(translation[:, bone] - rest_t[bone]).max() > STILL
            near = np.minimum(np.abs(rotation[:, bone] - rest_q[bone]), np.abs(rotation[:, bone] + rest_q[bone]))
            moved_s = np.abs(scale[:, bone] - 1).max() > STILL_SCALE
            for path, values, moved, tolerance in (("translation", translation[:, bone], moved_t, KEY_METRES),
                                                   ("rotation", rotation[:, bone], near.max() > STILL, KEY_QUATERNION),
                                                   ("scale", scale[:, bone], moved_s, KEY_SCALE)):
                if moved:
                    kept = keys(values, tolerance)
                    marks = kept.tobytes()
                    if marks not in times:
                        times[marks] = builder.accessor(seconds[kept], FLOAT, bounds=True)
                    samplers.append({"input": times[marks], "output": builder.accessor(values[kept], FLOAT),
                                     "interpolation": "LINEAR"})
                    channels.append({"sampler": len(samplers) - 1, "target": {"node": first + bone, "path": path}})
        if channels:
            out.append({"name": record["name"], "channels": channels, "samplers": samplers})
    return out


def shown(document: dict, arrays) -> tuple[dict, dict]:
    """*document* and *arrays* with only the parts shown at rest, numbered again, and the materials they use.

    A part hidden at rest (a lower LOD level, an outfit's other version) is not written to a ``.glb``: glTF's
    only way to hide one, ``KHR_node_visibility``, is ignored by Blender (5.2) and most viewers, which drew every
    hidden part over the shown one — nodes in a second scene or in none are imported all the same.  The FBX and
    the ``.blend`` keep them, hidden (Blender honours their hidden state)."""
    parts = [(index, part) for index, part in enumerate(document["parts"]) if not part.get("hidden")]
    if len(parts) == len(document["parts"]):
        return document, arrays
    out = {key: value for key, value in arrays.items() if not key.startswith("p") or "_" not in key
           or not key.split("_", 1)[0][1:].isdigit()}
    for new, (old, _part) in enumerate(parts):
        prefix = f"p{old}_"
        for key in arrays.keys():
            if key.startswith(prefix):
                out[f"p{new}_{key[len(prefix):]}"] = arrays[key]
    used = {name for _index, part in parts for name in part["materials"]}
    document = dict(document, parts=[part for _index, part in parts],
                    materials={name: record for name, record in document["materials"].items() if name in used})
    return document, out


def write(document: dict, arrays, folder: Path, target: Path) -> dict:
    """Write the scene (*document*, *arrays*; maps relative to *folder*) to *target*; what was left out.
    Only the parts shown at rest are written (``shown``)."""
    folder, target = Path(folder), Path(target)
    document, arrays = shown(document, arrays)
    builder = _Builder()
    gltf = {"asset": {"version": "2.0", "generator": "Polylode interchange", "extras": {"rules": RULES}}, "scene": 0,
            "scenes": [{"name": document["name"], "nodes": [0]}], "nodes": [], "meshes": [], "skins": [],
            "materials": [], "textures": [], "images": [], "samplers": [{"wrapS": 10497, "wrapT": 10497}],
            "extensionsUsed": set()}
    bones = document["bones"]
    rest = np.einsum("ij,njk->nik", TO_Y_UP, np.asarray(arrays["bone_rest"], dtype=np.float64))
    animated = bool(document.get("animations"))
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
        node = {"name": bone["name"], **(_trs_node(local) if animated else {"matrix": _column_major(local)})}
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
            attributes[VERTEX_DATA] = builder.accessor(np.asarray(arrays[p + "col"], dtype=np.float64), FLOAT,
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
        gltf["nodes"].append(node)
        gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)

    if animated:
        gltf["animations"] = _animations(document, arrays, rest, builder, first)
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
    document_path, target = Path(document_path), Path(target)
    document = json.loads(document_path.read_text(encoding="utf-8"))
    arrays = np.load(document_path.parent / document["arrays"])
    document, arrays = shown(document, dict(arrays))
    report = write(document, arrays, document_path.parent, target)
    report["readback"] = None
    if blender is None:
        return report, []
    if len(document["parts"]) != len(json.loads(document_path.read_text(encoding="utf-8"))["parts"]):
        # the read-back measures the file against the parts written: those, as a document of their own
        checked = document_path.with_name(document_path.stem + ".shown.json")
        np.savez(checked.with_suffix(".npz"), **arrays)
        checked.write_text(json.dumps(dict(document, arrays=checked.with_suffix(".npz").name), ensure_ascii=False),
                           encoding="utf-8")
        try:
            return _readback(checked, target, blender, report, document, arrays)
        finally:
            checked.unlink(missing_ok=True)
            checked.with_suffix(".npz").unlink(missing_ok=True)
    return _readback(document_path, target, blender, report, document, arrays)


def _readback(document_path: Path, target: Path, blender: Path, report: dict, document: dict,
              arrays) -> tuple[dict, list[str]]:
    import subprocess
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
