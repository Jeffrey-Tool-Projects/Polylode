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


def _texels(uv: np.ndarray, size: tuple[int, int]):
    """Every texel centre of a *size* map inside one of the triangles *uv* (T × 3 × 2, v down): row, column,
    triangle and barycentrics."""
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


#: the frame's own values a bake may give (as a head-on, lit, dry frame holds them): View's viewport origin,
#: camera position and translation (the head-on view), handedness, mip bias and frame index; the primitive's
#: determinant and wetness.  A shader that reads any other — the view matrices, the diffuse override, a
#: parameter collection, the lights — draws its coverage from the frame's lighting or view, which a bake can
#: only make up: such a material is not baked (the 心 lower body's sheer cloth took its coverage from a lit
#: colour's luminance, and with made-up light it lost the legs).
RUNTIME = {"View": {67, 70, 134, 147, 149, 158}, "Primitive": {4, 33}}


def runtime_reads(shader: dxbc.Shader) -> list[str]:
    """The frame values *shader* reads that a bake cannot stand for: none, for a bakeable shader."""
    names = {register: name for name, register in shader.buffers.items()}
    wrong = []
    for register, slot in {(int(a), int(b)) for a, b in re.findall(r"cb(\d+)\[(\d+)\]", shader.asm.split("ps_5_0", 1)[-1])}:
        name = names.get(register, f"cb{register}")
        if name in ("Material", "ShadowDepthPass"):
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


def coverage(shader: dxbc.Shader, buffers: dict[int, np.ndarray], textures: dict[int, np.ndarray],
             points: dict[str, np.ndarray]) -> np.ndarray:
    """Each point's share of the dither positions the shader keeps."""
    program = vm.parse(shader.asm)
    n = len(points["normal"])
    # without the screen position (no dither) one position says it all
    screens = SCREEN if re.search(r"dcl_input_ps_siv .*, position", shader.asm) else SCREEN[:1]
    packed = any(mask == "xyzw" for semantic, index, mask in
                 (x for register in shader.inputs.values() for x in register) if semantic == "TEXCOORD" and index == 0)
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
                  uexp: bytes, root: dict, collections: list[dict], announce=lambda: None) -> np.ndarray:
    """*name*'s coverage over its base map (H × W, 0..1)."""
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
    view, primitive = shader.buffers.get("View"), shader.buffers.get("Primitive")
    if view is not None:
        buffers[view][144] = (0.0, 0.0, 0.0, 1.0)      # DiffuseOverrideParameter
        buffers[view][147, 3] = 1.0
    if primitive is not None:
        buffers[primitive][4, 3] = 1.0                 # the local-to-world determinant's sign
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
    rows, columns, which, weights = _texels(uv_sets[0], size)
    at = lambda values: (weights[:, :, None] * values[which]).sum(axis=1)
    normal = at(joined["normal"])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    tangent = at(joined["tangent"])
    tangent -= (tangent * normal).sum(1, keepdims=True) * normal
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    sign = np.where(at(joined["sign"][:, :, None])[:, 0] < 0, -1.0, 1.0)
    points = dict(normal=normal, tangent=tangent, sign=sign, colour=at(joined["colour"]),
                  uvs=[at(uv) for uv in uv_sets])
    covered = coverage(shader, buffers, textures, points)
    alpha = np.full((size[1], size[0]), np.nan)
    alpha[rows, columns] = covered
    alpha = _fill(alpha)
    if size != full:
        alpha = np.asarray(Image.fromarray(alpha.astype(np.float32), "F").resize(full, Image.BILINEAR))
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
VERSION = 2
#: a texel's coverage counted as partial (drawn blended, not cut) between these
PARTIAL = (0.02, 0.98)
#: the share of a material's covered texels that must be partial for it to be drawn blended; below it the
#: partial ones are cut at 0.5 (a dither's edge, a strand's tip) and the rest drawn as the solid cards they are
BLENDED_SHARE = 0.5


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
        cached = Path(files.cache) / f"{key}.png"
        step("以遊戲 shader 烘焙裁切覆蓋率")
        try:
            if cached.is_file():
                with Image.open(cached) as opened:
                    alpha = np.asarray(opened, np.float32) / 255.0
            else:
                if store is None:
                    store = A.read(files.archive())
                root, collections = files.root_of(record["source_package"])
                alpha = bake_material(name, record, document, arrays, case, store,
                                      files.package(record["source_package"]), root, collections,
                                      announce=lambda: log(f"{name}：以遊戲的 shader 烘焙覆蓋率（第一次要一兩分鐘）…"))
                cached.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.round(alpha * 255).astype(np.uint8)).save(cached)
        except (BakeError, vm.ShaderError, preshader.PreshaderError, OSError, KeyError, ValueError) as failure:
            log(f"{name}：遊戲的 shader 無法在這裡執行（{failure}），照材質宣告的裁切輸出")
            continue
        base_record = record["maps"][record["base"]]
        base = _image(case / base_record["file"], False)
        folder = case / "coverage"
        folder.mkdir(exist_ok=True)
        target = folder / f"{name}.png"
        rgba = np.concatenate([base[:, :, :3], alpha[:, :, None]], axis=2)
        Image.fromarray(np.round(rgba * 255).astype(np.uint8), "RGBA").save(target)
        covered = alpha[(alpha > 0)]
        partial = float(((covered > PARTIAL[0]) & (covered < PARTIAL[1])).mean()) if covered.size else 0.0
        coverage_key = record["base"] + "_Coverage"
        record["maps"][coverage_key] = dict(base_record, file=str(target.relative_to(case)))
        record.update(base=coverage_key, alpha_from="a",
                      coverage="the game's shadow-depth shader run over each texel, seen head-on (shader_bake)")
        if partial > BLENDED_SHARE:
            record["blend"] = "alpha"
        else:
            record["cutoff"] = 0.5
    (case / "source_document.json").write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
