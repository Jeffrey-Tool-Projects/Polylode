"""Wuthering Waves' masked materials as glTF, FBX and Blender can carry them: the coverage the game's own shader
gives each texel, baked into a copy of the base map's alpha.

A masked material's coverage is not its base map's alpha — the fox tails of 心 read a type mask, a fur mask,
the vertex colour, a dither, even a lit colour's luminance, each variant of the shader its own way — so it is
not written as a rule here: the game's compiled shader is run (``shaders``).  For each masked material:

1. its instance's inline shader map gives the uniform expressions, run with the instance's parameters
   (``shaders.inline``, ``shaders.preshader``); the shader map's hash finds its compiled shaders in
   ``ShaderArchive-Client_Chunk-PCD3D_SM5``, and the smallest pixel shader of its shadow depth pass that
   discards is the one that says where the material is (``shaders.archive``, ``shaders.dxbc``);
2. every texel of the base map its triangles cover gets the point of the surface there — the UV sets
   (Unreal's, v down), the vertex colour, the tangent frame — interpolated (``_texels``);
3. the shader is run at each such point (``shaders.vm``) as seen head-on: the view along the surface's normal
   (the camera-relative position −N and, per point, a view matrix with N toward the camera), at ten screen
   positions spanning the five dither patterns;
4. the share of runs that are not discarded is the texel's alpha, drawn blended where coverage is partial and
   cut at 0.5 where it is all or nothing.

The runtime's own buffers are given what a lit, dry, unfogged frame holds: ``DiffuseOverrideParameter``
(0, 0, 0, 1), the primitive's determinant 1, wetness 0, mip bias and frame index 0; a material parameter
collection gives its defaults.  A material whose shader cannot be run here (an instruction, an opcode or a
buffer not read) is left as the source declares it and the log says why — never given a coverage made up.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .shaders import archive as A
from .shaders import dxbc, inline, preshader, vm

#: screen positions for the dither: (x + 2y) mod 5 runs through all five patterns, the noise tile varies
SCREEN = [(3.5 + 7 * k, 5.5 + 11 * k) for k in range(10)]
#: lanes run at once
CHUNK = 1 << 18
#: the largest coverage map worked out: a larger base map takes it scaled up (a coverage is an outline)
MOST = 1024


class BakeError(ValueError):
    pass


def _linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _image(path: Path, srgb: bool) -> np.ndarray:
    with Image.open(path) as opened:
        data = np.asarray(opened.convert("RGBA"), np.float32) / 255.0
    if srgb:                                           # an sRGB texture is read linear on the GPU
        data[..., :3] = _linear(data[..., :3])
    return data


def _texels(uv: np.ndarray, size: tuple[int, int], unique: bool = True):
    """Every texel centre of a *size* map inside one of the triangles *uv* (T × 3 × 2, v down): row, column,
    triangle and barycentrics — each texel once, or (*unique* false) once for every triangle over it."""
    w, h = size
    rows, columns, triangles, weights = [], [], [], []
    for number, corners in enumerate(uv):
        px = corners * np.array([w, h])
        x0, x1 = int(np.floor(px[:, 0].min())), int(np.ceil(px[:, 0].max()))
        y0, y1 = int(np.floor(px[:, 1].min())), int(np.ceil(px[:, 1].max()))
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        centres = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
        a, b, c = px
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (centres[:, 0] - c[0]) + (c[0] - b[0]) * (centres[:, 1] - c[1])) / denominator
        l2 = ((c[1] - a[1]) * (centres[:, 0] - c[0]) + (a[0] - c[0]) * (centres[:, 1] - c[1])) / denominator
        inside = (l1 >= 0) & (l2 >= 0) & (1 - l1 - l2 >= 0)
        if inside.any():
            rows.append(np.mod(ys.ravel()[inside], h))
            columns.append(np.mod(xs.ravel()[inside], w))
            triangles.append(np.full(int(inside.sum()), number))
            weights.append(np.stack([l1[inside], l2[inside], 1 - l1[inside] - l2[inside]], axis=1))
    if not rows:
        raise BakeError("底色圖上沒有貼圖像素落在材質的三角形上")
    if not unique:
        return tuple(np.concatenate(parts) for parts in (rows, columns, triangles, weights))
    # a texel two triangles share (an island's seam) is taken once
    keys = np.concatenate(rows) * w + np.concatenate(columns)
    _unique, first = np.unique(keys, return_index=True)
    pick = lambda parts: np.concatenate(parts)[first]
    return pick(rows), pick(columns), pick(triangles), pick(weights)


def _frozen_values(document_material: dict, uniforms: inline.Uniforms, texture_size) -> np.ndarray:
    """The Material uniform buffer: each vector expression, then the scalars four to a float4."""
    native = document_material.get("native_material") or {}
    values, colours = native.get("Scalars") or {}, native.get("Colors") or {}
    scalars = [values.get(name, default) for name, default in uniforms.scalars]
    vectors = [tuple(colours[name][c] for c in "RGBA") if name in colours else default
               for name, default in uniforms.vectors]
    slots = [preshader.evaluate(code, scalars, vectors, texture_size) for code in uniforms.vector_code]
    packed = np.zeros((len(uniforms.scalar_code) + 3) // 4 * 4)
    for k, code in enumerate(uniforms.scalar_code):
        packed[k] = preshader.evaluate(code, scalars, vectors, texture_size)[0]
    return np.array(slots + list(packed.reshape(-1, 4)), dtype=np.float64)


def _collection(info: dict) -> np.ndarray:
    """A material parameter collection's uniform buffer from its defaults: scalars four to a float4, then
    vectors."""
    scalars = [p.get("DefaultValue", 0.0) for p in info.get("ScalarParameters", [])]
    vectors = [[p.get("DefaultValue", {}).get(c, 0.0) for c in "RGBA"] for p in info.get("VectorParameters", [])]
    packed = np.zeros((len(scalars) + 3) // 4 * 4)
    packed[:len(scalars)] = scalars
    return np.array(list(packed.reshape(-1, 4)) + vectors, dtype=np.float64).reshape(-1, 4)


#: the frame's own values a bake gives, each read off how the shaders use it against Unreal 4.26's View and
#: Primitive uniform buffers, as the game's default frame holds them, seen head-on:
#:   View [12..15] TranslatedWorldToView — per point, its normal toward the camera (``coverage``);
#:   View [36] ClipToView's first row — tan(horizontal FOV / 2): 1, a 90° view;
#:   View [67], [70] the camera's translation and origin — 0 (the position is camera-relative);
#:   View [134] the viewport's origin, [149] the mip bias, [158] the frame index, [148] the game time — 0;
#:   View [144] Diffuse/NormalOverrideParameter — (0, 0, 0, 1): no override;  [147].w — 1;
#:   Primitive [0..3] LocalToWorld — identity (the scene is in its own frame);  [4].w the determinant's sign — 1;
#:   Primitive [27] per-object custom data, [33] wetness — 0;
#:   MaterialCollection<i> — its asset's defaults (``_collection``).
#: A shader reading any other field is not baked: a value a bake would make up is a coverage made up.  (With
#: LocalToWorld left 0 the lower body of 心 took a distance of 0, a NaN, and lost its legs.)
RUNTIME = {"View": {12, 13, 14, 15, 36, 67, 70, 134, 144, 147, 148, 149, 158},
           "Primitive": {0, 1, 2, 3, 4, 27, 33}}
FRAME = {"View": {36: (1.0, 0.0, 0.0, 0.0), 144: (0.0, 0.0, 0.0, 1.0), 147: (0.0, 0.0, 0.0, 1.0)},
         "Primitive": {0: (1.0, 0.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0, 0.0), 2: (0.0, 0.0, 1.0, 0.0),
                       3: (0.0, 0.0, 0.0, 1.0), 4: (1.0, 1.0, 1.0, 1.0)}}


def runtime_reads(shader: dxbc.Shader) -> list[str]:
    """The frame values *shader* reads that a bake cannot stand for: none, for a bakeable shader."""
    names = {register: name for name, register in shader.buffers.items()}
    wrong = []
    for register, slot in {(int(a), int(b)) for a, b in re.findall(r"cb(\d+)\[(\d+)\]", shader.asm.split("ps_5_0", 1)[-1])}:
        name = names.get(register, f"cb{register}")
        if name in ("Material", "ShadowDepthPass") or name.startswith("MaterialCollection"):
            continue
        if slot not in RUNTIME.get(name, set()):
            wrong.append(f"{name}[{slot}]")
    return sorted(wrong)


def _sizes(asm: str) -> dict[int, int]:
    return {int(i): int(n) for i, n in re.findall(r"dcl_constantbuffer CB(\d+)\[(\d+)\]", asm)}


def depth_shader(store: A.Archive, map_index: int) -> dxbc.Shader:
    """The map's smallest shadow-depth pixel shader that discards: where the material is, and nothing else."""
    best = None
    for shader in store.shaders(map_index):
        if store.frequency(shader) != A.PIXEL:
            continue
        read = dxbc.read(store.code(shader))
        if "ShadowDepthPass" in read.buffers and "discard" in read.asm:
            size = read.asm.count("\n")
            if best is None or size < best[0]:
                best = (size, read)
    if best is None:
        raise BakeError("找不到會丟棄像素的陰影深度 shader")
    return best[1]


#: how many points the sensitivity test runs, and the share of them whose result may change
PROBE, SENSITIVE = 20000, 0.01


def unknown_inputs(shader: dxbc.Shader, uv_sets: int) -> list[tuple[int, int]]:
    """The (input register, component) the shader reads that the mesh does not have: an interpolant past its
    UV sets — a customised UV the game's vertex shader works out, which a bake has no value for."""
    packed = _packed(shader)
    used = dxbc.used_inputs(shader.asm)
    unknown = []
    for register, semantics in shader.inputs.items():
        for semantic, index, mask in semantics:
            if semantic != "TEXCOORD" or index >= 6:
                continue
            for c in mask:
                if (semantic, index, c) not in used:
                    continue
                k = "xyzw".index(c)
                uv = 2 * index + (k >= 2) if packed else (index if k < 2 else None)
                if uv is None or uv >= uv_sets:
                    unknown.append((register, k))
    return unknown


def _packed(shader: dxbc.Shader) -> bool:
    return any(mask == "xyzw" for semantic, index, mask in
               (x for register in shader.inputs.values() for x in register) if semantic == "TEXCOORD" and index == 0)


def coverage(shader: dxbc.Shader, buffers: dict[int, np.ndarray], textures: dict[int, np.ndarray],
             points: dict[str, np.ndarray], fill: dict | None = None, screens=None) -> np.ndarray:
    """Each point's share of the dither positions the shader keeps; *fill* gives inputs the mesh lacks a value
    ({(register, component): value})."""
    program = vm.parse(shader.asm)
    n = len(points["normal"])
    # without the screen position (no dither) one position says it all
    if screens is None:
        screens = SCREEN if re.search(r"dcl_input_ps_siv .*, position", shader.asm) else SCREEN[:1]
    packed = _packed(shader)
    kept = np.zeros(n)
    for start in range(0, n, CHUNK):
        part = {k: v[start:start + CHUNK] for k, v in points.items() if k != "uvs"}
        part["uvs"] = [uv[start:start + CHUNK] for uv in points["uvs"]]
        m = len(part["normal"])
        normal, tangent = part["normal"], part["tangent"]
        bitangent = np.cross(normal, tangent) * part["sign"][:, None]
        view = np.zeros((m, 4, 4))                    # rows 12..15 of View: world → view, N toward the camera
        view[:, :3, 0], view[:, :3, 1], view[:, :3, 2] = tangent, bitangent, -normal
        view[:, 3, 3] = 1.0
        for x, y in screens:
            inputs = {}
            for register, semantics in shader.inputs.items():
                value = np.zeros((m, 4))
                for semantic, index, mask in semantics:
                    source = _input(semantic, index, mask, part, packed, (x, y))
                    for c in mask:
                        value[:, "xyzw".index(c)] = source[:, "xyzw".index(c)]
                for (filled, component), given in (fill or {}).items():
                    if filled == register:
                        value[:, component] = given
                inputs[register] = value
            lane_buffers = dict(buffers)
            view_register = shader.buffers.get("View")
            if view_register is not None:
                each = np.broadcast_to(buffers[view_register], (m,) + buffers[view_register].shape).copy()
                each[:, 12:16] = view
                lane_buffers[view_register] = each
            machine = vm.Machine(program, m, inputs, lane_buffers, textures)
            kept[start:start + m] += ~machine.run()
    return kept / len(screens)


def _input(semantic: str, index: int, mask: str, part: dict, packed: bool, screen) -> np.ndarray:
    m = len(part["normal"])
    out = np.zeros((m, 4))
    if semantic == "TEXCOORD" and index == 10:
        out[:, :3] = part["tangent"]
    elif semantic == "TEXCOORD" and index == 11:
        out[:, :3], out[:, 3] = part["normal"], part["sign"]
    elif semantic == "COLOR" and index == 0:
        out[:] = part["colour"]
    elif semantic == "TEXCOORD" and index == 7:
        out[:, :3] = -part["normal"]                  # the camera-relative position: one unit along the normal
    elif semantic == "TEXCOORD" and index < 6:
        uvs = part["uvs"]
        first, second = (2 * index, 2 * index + 1) if packed else (index, None)
        if first < len(uvs):
            out[:, 0:2] = uvs[first]
        if second is not None and second < len(uvs):
            out[:, 2:4] = uvs[second]
    elif semantic == "SV_Position":
        out[:, 0], out[:, 1], out[:, 3] = screen[0], screen[1], 1.0
    return out


def bake_material(name: str, record: dict, document: dict, arrays: dict, case: Path, store: A.Archive,
                  uexp: bytes, root: dict, collections: list[dict], announce=lambda: None):
    """*name*'s coverage over its base map: maps (each H × W, 0..1) and each of its triangles' map, in the order
    of the parts and their triangles (``_layers``)."""
    resources = inline.read(uexp)
    if not resources:
        raise BakeError("材質實例的套件裡沒有內嵌的 shader map")
    digests = [uexp[i:i + 20] for i in range(len(uexp) - 19) if store.map_index(uexp[i:i + 20]) is not None]
    if not digests:
        raise BakeError("材質的 shader map 不在 SM5 封存檔裡")
    shader = depth_shader(store, store.map_index(digests[0]))
    unknown = runtime_reads(shader)
    if unknown:
        raise BakeError("覆蓋率讀了執行期的光照或視角（" + "、".join(unknown[:4])
                        + ("…" if len(unknown) > 4 else "") + "）")
    announce()
    uniforms = resources[0]
    maps = record["maps"]
    referenced = [Path(t.get("ObjectPath", "")).stem.split(".")[0] if isinstance(t, dict) else None
                  for t in root.get("ReferencedTextures", [])]

    def texture_of(reference: int, parameter: str | None):
        key = parameter if parameter in maps else (referenced[reference] if 0 <= reference < len(referenced) else None)
        if key not in maps:
            raise BakeError(f"材質的貼圖裡沒有 {parameter or reference}")
        return maps[key]

    def size_of(reference: int):
        named = next((p for p, r in uniforms.textures if r == reference), None)
        with Image.open(case / texture_of(reference, named)["file"]) as opened:
            return opened.size

    sizes = _sizes(shader.asm)
    buffers = {register: np.zeros((sizes.get(register, 1), 4)) for register in sizes}
    for buffer_name, values in FRAME.items():
        register = shader.buffers.get(buffer_name)
        if register is not None:
            for slot, value in values.items():
                if slot < len(buffers[register]):
                    buffers[register][slot] = value
    material = _frozen_values(record, uniforms, size_of)
    register = shader.buffers.get("Material")
    if register is None:
        raise BakeError("shader 沒有讀 Material buffer")
    buffers[register][:min(len(material), len(buffers[register]))] = material[:len(buffers[register])]
    for number, collection in enumerate(collections):
        target = shader.buffers.get(f"MaterialCollection{number}")
        if target is not None:
            values = _collection(collection)
            buffers[target][:min(len(values), len(buffers[target]))] = values[:len(buffers[target])]
    textures = {}
    for register_number, (buffer, resource) in shader.textures.items():
        if buffer != register or resource % 2 or resource // 2 >= len(uniforms.textures):
            raise BakeError(f"貼圖暫存器 t{register_number} 不是材質的 2D 貼圖")
        parameter, reference = uniforms.textures[resource // 2]
        map_record = texture_of(reference, parameter)
        textures[register_number] = _image(case / map_record["file"], bool(map_record.get("srgb")))

    base = _image(case / maps[record["base"]]["file"], False)
    full = (base.shape[1], base.shape[0])
    scale = min(1.0, MOST / max(full))
    size = (max(1, round(full[0] * scale)), max(1, round(full[1] * scale)))
    triangles = []
    for index, part in enumerate(document["parts"]):
        if name not in part["materials"]:
            continue
        p = f"p{index}_"
        chosen = arrays[p + "tri"][arrays[p + "slot"] == part["materials"].index(name)]
        uvs = [arrays[f"{p}uv{k}"].astype(np.float64) * [1, -1] + [0, 1] for k in part.get("uvs", [0])]  # v down
        colour = arrays[p + "col"] if p + "col" in arrays else np.ones((len(arrays[p + "co"]), 4))
        tangent = arrays[p + "tan"] if p + "tan" in arrays else np.zeros((len(arrays[p + "co"]), 4))
        triangles.append(dict(uvs=[uv[chosen] for uv in uvs], normal=arrays[p + "no"][chosen],
                              tangent=tangent[chosen][..., :3], sign=-tangent[chosen][..., 3],  # v down: sign turns
                              colour=colour[chosen]))
    if not triangles:
        raise BakeError("沒有部件用這個材質")
    joined = {k: np.concatenate([t[k] for t in triangles]) for k in ("normal", "tangent", "sign", "colour")}
    uv_sets = [np.concatenate([t["uvs"][k] for t in triangles]) for k in range(len(triangles[0]["uvs"]))]
    rows, columns, which, weights = _texels(uv_sets[0], size, unique=False)
    at = lambda values: (weights[:, :, None] * values[which]).sum(axis=1)
    normal = at(joined["normal"])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    tangent = at(joined["tangent"])
    tangent -= (tangent * normal).sum(1, keepdims=True) * normal
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    sign = np.where(at(joined["sign"][:, :, None])[:, 0] < 0, -1.0, 1.0)
    points = dict(normal=normal, tangent=tangent, sign=sign, colour=at(joined["colour"]),
                  uvs=[at(uv) for uv in uv_sets])
    unknown = unknown_inputs(shader, len(uv_sets))
    if unknown:
        # the coverage may not depend on what the mesh lacks: run a sample with it at 0 and at 1
        pick = np.random.default_rng(0).choice(len(normal), min(PROBE, len(normal)), replace=False)
        sample = _subset(points, pick)
        low = coverage(shader, buffers, textures, sample, {u: 0.0 for u in unknown}, SCREEN[:2])
        high = coverage(shader, buffers, textures, sample, {u: 1.0 for u in unknown}, SCREEN[:2])
        changed = float((np.abs(low - high) > 0.25).mean())
        if changed > SENSITIVE:
            raise BakeError(f"覆蓋率取決於網格沒有的輸入（頂點 shader 算出的自訂 UV；{changed:.0%} 的點會變）")
    fill = {u: 0.0 for u in unknown}
    covered = _covered(lambda pick: coverage(shader, buffers, textures, _subset(points, pick), fill),
                       rows, columns, which, len(joined["normal"]), size[0])
    maps, layer = _layers(rows, columns, which, covered, len(joined["normal"]), (size[1], size[0]))
    maps = [_fill(alpha) for alpha in maps]
    if size != full:
        maps = [np.asarray(Image.fromarray(alpha.astype(np.float32), "F").resize(full, Image.BILINEAR))
                for alpha in maps]
    return maps, layer



#: how many of a triangle's own samples are run to tell whether it agrees with the texels' first triangles
PROBES = 6


def _subset(points: dict, pick: np.ndarray) -> dict:
    sample = {k: v[pick] for k, v in points.items() if k != "uvs"}
    sample["uvs"] = [uv[pick] for uv in points["uvs"]]
    return sample


def _covered(run, rows, columns, which, triangles: int, width: int) -> np.ndarray:
    """Every sample's coverage, the shader run (``run(indices)``) only where it can differ: once per texel (its
    first triangle), at up to PROBES samples of each triangle, and over the whole of a triangle whose probes lie
    more than DISAGREE from the texels' first values.  The rest — a texel's other triangles that agree, mostly
    mirrored islands — take its first value."""
    keys = rows * width + columns
    _unique, inverse = np.unique(keys, return_inverse=True)
    first = np.full(int(inverse.max()) + 1, len(keys))
    np.minimum.at(first, inverse, np.arange(len(keys)))
    head = first[inverse]                                  # each sample's texel's first sample
    covered = np.full(len(keys), np.nan)
    done = np.zeros(len(keys), bool)

    def evaluate(indices):
        indices = indices[~done[indices]]
        if len(indices):
            covered[indices] = run(indices)
            done[indices] = True
    evaluate(np.unique(first))
    order = np.argsort(which, kind="stable")
    starts = np.searchsorted(which[order], np.arange(triangles))
    ends = np.searchsorted(which[order], np.arange(triangles), side="right")
    probes = np.concatenate([order[a:b][np.linspace(0, b - a - 1, min(PROBES, b - a)).astype(np.int64)]
                             for a, b in zip(starts, ends) if b > a] or [np.zeros(0, np.int64)])
    evaluate(probes[probes != head[probes]])
    distance = np.abs(covered[probes] - covered[head[probes]])
    suspect = np.zeros(triangles, bool)
    total = np.bincount(which[probes], distance, minlength=triangles)
    count = np.bincount(which[probes], minlength=triangles)
    suspect[total > DISAGREE * np.maximum(count, 1)] = True
    evaluate(np.flatnonzero(suspect[which]))
    rest = ~done
    covered[rest] = covered[head[rest]]
    return covered

#: the most coverage maps one material is split into, and how far (mean |Δ|) a triangle's own coverage may lie
#: from a map's before it goes to the next one
LAYERS, DISAGREE = 4, 0.25


def _layers(rows, columns, which, covered, triangles: int, shape) -> tuple[list[np.ndarray], np.ndarray]:
    """The coverage maps the samples need, and each triangle's map.

    One map holds one coverage per texel, but triangles can share texels with a different coverage: UV islands
    laid over each other whose vertex colour or other UVs differ (心's ear: opaque parallax-fur cards over the
    strand cards).  Each map is filled, first come, from the triangles still unplaced; those whose own samples lie
    on average more than DISAGREE from it go on to the next map.  Mirrored islands agree and stay together."""
    left = np.ones(triangles, bool)
    layer = np.zeros(triangles, np.int64)
    maps = []
    for number in range(LAYERS):
        if not left.any():
            break
        if number == LAYERS - 1:
            agree = left.copy()
        else:
            alpha = _first(rows, columns, covered, left[which], shape)
            take = left[which]
            distance = np.abs(covered - alpha[rows, columns])
            total = np.bincount(which[take], distance[take], minlength=triangles)
            count = np.bincount(which[take], minlength=triangles)
            agree = left & (total <= DISAGREE * np.maximum(count, 1))
            if not agree.any():
                agree = left.copy()
        layer[agree] = number
        left &= ~agree
        maps.append(_first(rows, columns, covered, agree[which], shape))
    return maps, layer


def _first(rows, columns, covered, take, shape) -> np.ndarray:
    """A map of the taken samples, each texel the first sample over it (NaN where none is)."""
    alpha = np.full(shape, np.nan)
    keys = rows[take] * shape[1] + columns[take]
    _unique, first = np.unique(keys, return_index=True)
    alpha[rows[take][first], columns[take][first]] = covered[take][first]
    return alpha


def _fill(alpha: np.ndarray) -> np.ndarray:
    """Texels no triangle covers take their nearest covered texel's value (bilinear filtering reads them)."""
    from scipy import ndimage
    empty = np.isnan(alpha)
    if empty.all():
        return np.zeros_like(alpha)
    _distance, (iy, ix) = ndimage.distance_transform_edt(empty, return_indices=True)
    return alpha[iy, ix]


#: raised when what a bake gives changes, so a coverage kept from an earlier rule is not used again
VERSION = 7
#: a texel's coverage counted as partial (drawn blended, not cut) between these
PARTIAL = (0.02, 0.98)
#: the share of a material's covered texels that must be partial for it to be drawn blended; below it the
#: partial ones are cut at 0.5 (a dither's edge, a strand's tip) and the rest drawn as the solid cards they are
BLENDED_SHARE = 0.5



def _move_layers(document: dict, arrays: dict, name: str, layer: np.ndarray) -> None:
    """*name*'s triangles whose coverage lies on map k > 0 moved to the material ``<name>_Overlap<k>``, in a part
    of their own (``<part>_Overlap<k>``: a part is one material, as ``split.by_material`` leaves them); *layer* is
    in the order ``bake_material`` reads them (the parts in order, each part's triangles in order)."""
    from ..interchange.split import by_material
    at = 0
    for index, part in enumerate(list(document["parts"])):
        if name not in part["materials"]:
            continue
        p = f"p{index}_"
        slots = arrays[p + "slot"].copy()
        mine = np.flatnonzero(slots == part["materials"].index(name))
        theirs = layer[at:at + len(mine)]
        at += len(mine)
        numbers = [int(n) for n in np.unique(theirs[theirs > 0])]
        if not numbers:
            continue
        part = dict(part, materials=list(part["materials"]))
        for number in numbers:
            part["materials"].append(f"{name}_Overlap{number}")
            slots[mine[theirs == number]] = len(part["materials"]) - 1
        own = {key[len(p):]: value for key, value in arrays.items() if key.startswith(p)}
        own["slot"] = slots
        pieces, piece_arrays = by_material({"parts": [part]}, {f"p0_{key}": value for key, value in own.items()})
        for key in [key for key in arrays if key.startswith(p)]:
            del arrays[key]
        placed = False
        for number, piece in enumerate(pieces["parts"]):
            material = piece["materials"][0]
            if material.startswith(f"{name}_Overlap"):
                piece["name"] = f"{part['name']}_Overlap{material[len(name) + len('_Overlap'):]}"
            elif len(pieces["parts"]) == len(numbers) + 1:
                piece["name"] = part["name"]                  # one material before: the part keeps its name
            if not placed and not material.startswith(f"{name}_Overlap"):
                target, placed = index, True
                document["parts"][index] = piece
            else:
                target = len(document["parts"])
                document["parts"].append(piece)
            for key, value in piece_arrays.items():
                if key.startswith(f"p{number}_"):
                    arrays[f"p{target}_{key[len(f'p{number}_'):]}"] = value
    if at != len(layer):
        raise BakeError("覆蓋率分層的三角形數和材質對不上")


def bake_case(case: Path, files, log=print, step=lambda text: None) -> None:
    """Every masked material of the scene in *case* given the coverage the game's shader gives it.

    *files* reads the game: ``archive()`` → the SM5 ``Client_Chunk`` shader archive's path, ``package(path)`` →
    a package's ``.uexp`` bytes, ``root_of(path)`` → the root material's ``CachedExpressionData`` and its
    parameter collections, ``cache`` → a folder for baked coverages.  The scene document is rewritten.
    """
    case = Path(case)
    document = json.loads((case / "source_document.json").read_text(encoding="utf-8"))
    masked = [name for name, record in document["materials"].items()
              if not record.get("unused") and record.get("base") and record.get("blend") is None
              and record.get("cutoff") is None and (record.get("native_material") or {}).get("BlendMode") == 1]
    if not masked:
        return
    arrays = dict(np.load(case / document["arrays"]))
    store = None
    changed = False
    import hashlib
    for name in masked:
        record = document["materials"][name]
        geometry = hashlib.sha256()
        for index, part in enumerate(document["parts"]):
            if name in part["materials"]:
                for key in ("tri", "slot", "uv0", "no", "tan", "col"):
                    if f"p{index}_{key}" in arrays:
                        geometry.update(np.ascontiguousarray(arrays[f"p{index}_{key}"]).tobytes())
        key = hashlib.sha256((f"{VERSION}|" + record.get("source_json_sha256", "") + name + geometry.hexdigest())
                             .encode()).hexdigest()[:24]
        cached = Path(files.cache) / f"{key}.npz"
        step("以遊戲 shader 烘焙裁切覆蓋率")
        try:
            if cached.is_file():
                with np.load(cached) as stored:
                    maps = [stored[f"map{k}"].astype(np.float32) / 255.0 for k in range(int(stored["count"]))]
                    layer = stored["layer"]
            else:
                if store is None:
                    store = A.read(files.archive())
                root, collections = files.root_of(record["source_package"])
                maps, layer = bake_material(name, record, document, arrays, case, store,
                                            files.package(record["source_package"]), root, collections,
                                            announce=lambda: log(f"{name}：以遊戲的 shader 烘焙覆蓋率（第一次要一兩分鐘）…"))
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cached, count=len(maps), layer=layer,
                                    **{f"map{k}": np.round(m * 255).astype(np.uint8) for k, m in enumerate(maps)})
        except (BakeError, vm.ShaderError, preshader.PreshaderError, OSError, KeyError, ValueError) as failure:
            log(f"{name}：遊戲的 shader 無法在這裡執行（{failure}），照材質宣告的裁切輸出")
            continue
        base_record = record["maps"][record["base"]]
        base = _image(case / base_record["file"], False)
        folder = case / "coverage"
        folder.mkdir(exist_ok=True)
        original = dict(record, maps=dict(record["maps"]))
        for number, alpha in enumerate(maps):
            target = folder / (f"{name}.png" if number == 0 else f"{name}_{number}.png")
            rgba = np.concatenate([base[:, :, :3], alpha[:, :, None]], axis=2)
            Image.fromarray(np.round(rgba * 255).astype(np.uint8), "RGBA").save(target)
            if number == 0:
                layered = record
            else:
                # triangles over the same texels with another coverage: a material of their own
                layered = dict(original, maps=dict(original["maps"]))
                document["materials"][f"{name}_Overlap{number}"] = layered
            covered = alpha[(alpha > 0)]
            partial = float(((covered > PARTIAL[0]) & (covered < PARTIAL[1])).mean()) if covered.size else 0.0
            coverage_key = original["base"] + "_Coverage"
            layered["maps"][coverage_key] = dict(base_record, file=target.relative_to(case).as_posix())
            layered.update(base=coverage_key, alpha_from="a",
                           coverage="the game's shadow-depth shader run over each texel, seen head-on (shader_bake)"
                           + (f"; overlapping UVs, map {number + 1} of {len(maps)}" if len(maps) > 1 else ""))
            if partial > BLENDED_SHARE:
                layered["blend"] = "alpha"
            else:
                layered["cutoff"] = 0.5
        if len(maps) > 1:
            _move_layers(document, arrays, name, layer)
            changed = True
            log(f"{name}：UV 重疊處的覆蓋率不同，分成 {len(maps)} 個材質")
    if changed:
        np.savez_compressed(case / document["arrays"], **arrays)
    (case / "source_document.json").write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
