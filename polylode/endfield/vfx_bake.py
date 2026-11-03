"""An Endfield material this program does not draw, run as the game compiled it and baked into its files.

The writers draw a material by its base map; a material with none — the effect shaders (``HGRP/CharacterNPR_VFX``:
Si's wings, a character's glow and trails) colour a surface from masks, flow maps and tints — came out as a plain
grey-white surface.  Here such a material is run instead: its shader's pass variant for its keywords
(``shader_code``), the pixel stage's assembly interpreted over each texel of its triangles in their first UV set
(``wuthering_waves.shaders.vm``), with its own values in its material buffer, and the result baked into a map of its
own that every format draws.

Only a pass the game draws **unlit** (``RenderType`` ``…Unlit…``) is baked: its colour output is the colour drawn,
seen from anywhere; a lit pass's would carry this program's choice of light.  The rest of the frame is the
game's at rest — time 0, no camera fade, no LOD cross-fade, exposure 1 (``FRAME``) — and a buffer variable the
pixel stage reads that is neither the material's nor in ``FRAME`` stops the bake.  The pixel inputs come from the
mesh where the vertex stage passes a vertex stream through (``mov o#, v#``: the UV sets, the vertex colour); any
other input is run at 0 and at 1 and must not change the colour.  The blend the pass declares says what the output
is: ``One, OneMinusSrcAlpha`` a premultiplied colour and coverage, ``SrcAlpha, OneMinusSrcAlpha`` a straight one;
an additive one has no form in glTF and is not baked.

The bake is written as glTF draws an unlit surface: the base map's alpha the coverage and its colour black (the
base colour factor), the colour as the emission — over 1 scaled into the map and the scale its strength.  A material
that cannot be baked is not drawn grey: it is marked ``native_surface_unmapped`` with the reason, which every
writer draws invisible, and the reason is logged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ..wuthering_waves.shaders import dxbc, vm
from . import shader_code as SC

#: the texel grid a bake is run on: the largest map the material samples, at most this
MOST = 1024
#: screen positions the pixel stage is run at (a dither or a screen-space read averages over them)
SCREEN = [(3.5 + 7 * k, 5.5 + 11 * k) for k in range(4)]
#: the frame's own values, by variable name: the game at rest (identity transforms, time 0, no fades, exposure 1)
WIDTH, HEIGHT = 1920.0, 1080.0
IDENTITY = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
FRAME = {
    "_ScreenSize": [WIDTH, HEIGHT, 1 / WIDTH, 1 / HEIGHT],
    "_ScreenParams": [WIDTH, HEIGHT, 1 + 1 / WIDTH, 1 + 1 / HEIGHT],
    "_ProjectionParams": [1.0, 0.1, 1000.0, 0.001],
    "_GlobalMipBias": [0.0], "_TaaJitterStrength": [0.0, 0.0, 0.0, 0.0],
    "_ExposureWithMiscParams": [1.0, 1.0, 1.0, 1.0], "_VFXParams0": [0.0, 0.0, 0.0, 0.0],
    "_WorldSpaceCameraPos_Internal": [0.0, 0.0, 0.0, 0.0], "_PrevCamPosRWS_Internal": [0.0, 0.0, 0.0, 0.0],
    "_ViewMatrix": IDENTITY, "_InvViewProjMatrix": IDENTITY, "_NonJitteredViewNoTransProjMatrix": IDENTITY,
    "_PrevNonJitteredViewNoTransProjMatrix": IDENTITY,
    "unity_ObjectToWorld": IDENTITY, "unity_MatrixPreviousM": IDENTITY,
    "unity_LODFade": [1.0, 0.0, 0.0, 0.0], "unity_WorldTransformParams": [0.0, 0.0, 0.0, 1.0],
    "unity_MotionVectorsParamsInternal": [0.0, 0.0, 0.0, 0.0],
    "unity_Float4x5_Param2": [0.0, 0.0, 0.0, 0.0], "unity_Float4x5_Param3": [0.0, 0.0, 0.0, 0.0],
    "unity_OrthoParams": [0.0, 0.0, 0.0, 0.0],                    # a perspective camera
    # D3D's reversed depth (UNITY_REVERSED_Z): x = −1 + far/near, y = 1, z = x/far, w = 1/far
    "_ZBufferParams": [9999.0, 1.0, 9.999, 0.001],
    "_FrameCount": [0.0],
    "_DisableAnimateVert": [1.0],                                  # the rest pose the export holds, not animated
    # no fog: every density and scattering term 0, the volumetric fog off (its gate is _VolumetricFogParams0.z > 0);
    # checked by baking a _USE_FOG material with and without that keyword (the same colour)
    **{f"_ExponentialFogParams{k}": [0.0, 0.0, 0.0, 0.0] for k in range(6)},
    **{f"_AtmosphereFogParams{k}": [0.0, 0.0, 0.0, 0.0] for k in range(6)},
    **{f"_VolumetricFogParams{k}": [0.0, 0.0, 0.0, 0.0] for k in range(5)},
}
#: the frame's own maps, by name, each one texel: nothing behind the effect (the far plane: 0 in reversed depth)
#: and no fog (no light scattered, all of it let through)
FRAME_MAPS = {"_CameraDepthTexture": [0.0, 0.0, 0.0, 0.0], "_IntegratedLightScattering": [0.0, 0.0, 0.0, 1.0]}
#: the buffers a material's values fill: its own, and the loose uniforms of its shader
MATERIAL_BUFFERS = ("UnityPerMaterial", "_Globals")
#: Unity's BlendMode values the pass's blend factors are read as
ZERO, ONE, SRC_ALPHA, ONE_MINUS_SRC_ALPHA = 0, 1, 5, 10
#: an input changing the colour by more than this when run at 0 and at 1 stops the bake
SENSITIVE = 0.02
#: how many texels the dependence on missing inputs and on the screen position is tried at
PROBE = 4096


class BakeError(ValueError):
    """Why a material is not baked."""


def drawable(record: dict) -> bool:
    """A material the writers draw as it is: by a base map, a constant colour, a tinted layer (an overlay's
    colour multiplied on, §7.1455), or marked invisible."""
    return bool(record.get("base") or record.get("constant_color") is not None
                or record.get("native_surface_unmapped")
                or (record.get("tint") is not None and record.get("blend") in ("multiply", "alpha")))


#: Unity's default textures (ShaderLab's ``white``, ``black``, ``gray``…): what a texture property declares it
#: samples when the material binds nothing to it
DEFAULT_TEXTURES = {"white": (1.0, 1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0, 1.0), "gray": (0.5, 0.5, 0.5, 1.0),
                    "grey": (0.5, 0.5, 0.5, 1.0), "red": (1.0, 0.0, 0.0, 1.0)}


def _state(node, record: dict, defaults: dict) -> float:
    """A render-state value: the material property it names (the material's, else the shader's default), or the
    value it holds."""
    if not isinstance(node, dict):
        return float(node)
    name = node.get("name")
    if name and name in (record.get("floats") or {}):
        return float(record["floats"][name])
    if name and name in defaults:
        return float(defaults[name][0])
    return float(node.get("val", 0.0))


def _draws_no_colour(doc: dict, record: dict) -> bool:
    """Every pass of the shader writes no colour for this material (a colour mask of 0): a depth-only effect.
    The mask is often a property (``_ColorMask``), so it is the material's value, not the state's placeholder."""
    passes = doc["m_ParsedForm"]["m_SubShaders"][0]["m_Passes"]
    defaults = SC.property_defaults(doc)

    def mask(chosen):
        node = ((chosen["m_State"].get("rtBlend") or [{}])[0] or {}).get("colMask")
        return 15.0 if node is None else _state(node, record, defaults)
    return bool(passes) and all(mask(chosen) == 0.0 for chosen in passes)


def unbound_base(record: dict, doc: dict) -> dict | None:
    """A lit material that binds no texture to its base property, drawn as the game draws it: the texture the
    shader declares for that property (Purrche's ``_BaseMap``: ``white``) times ``_BaseColor``, a constant —
    whose alpha counts when the material blends (``_ALPHABLEND_ON``); or ``None`` when the shader is not one
    whose base property is known here."""
    from .shading import base_property
    prop = base_property({"_shader": record.get("shader"), "maps": record.get("maps", {})})
    if prop is None or prop in record.get("maps", {}):
        return None
    declared = next((p for p in doc["m_ParsedForm"]["m_PropInfo"]["m_Props"] if p["m_Name"] == prop), None)
    default = DEFAULT_TEXTURES.get(((declared or {}).get("m_DefTexture") or {}).get("m_DefaultName", ""))
    if default is None:
        return None
    tint = (record.get("colors") or {}).get("_BaseColor", (1.0, 1.0, 1.0, 1.0))
    colour = [float(d) * float(t) for d, t in zip(default, tint)]
    if "_ALPHABLEND_ON" not in (record.get("keywords") or []):
        colour[3] = 1.0
    return dict(record, constant_color=colour,
                coverage=f"no {prop} bound: the shader's default ({declared['m_DefTexture']['m_DefaultName']}) "
                         f"× _BaseColor")


def _image(path: Path, srgb: bool) -> np.ndarray:
    """A map as the GPU samples it in Unity's UVs: linear, its bottom row first (row 0 at v = 0)."""
    with Image.open(path) as opened:
        data = np.asarray(opened.convert("RGBA"), np.float32) / 255.0
    if srgb:
        rgb = data[..., :3]
        data[..., :3] = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.ascontiguousarray(data[::-1])


def _value(record: dict, name: str, defaults: dict) -> list[float] | None:
    if name in (record.get("floats") or {}):
        return [float(record["floats"][name])]
    if name in (record.get("colors") or {}):
        return [float(v) for v in record["colors"][name]]
    if name.endswith("_ST") and name[:-3] in record.get("maps", {}):
        entry = record["maps"][name[:-3]]
        return [float(v) for v in (*entry.get("scale", (1.0, 1.0)), *entry.get("offset", (0.0, 0.0)))]
    if name in defaults:
        return defaults[name]
    return None


def _buffers(variant: SC.Variant, record: dict, defaults: dict, asm: str, stage: str) -> dict[int, np.ndarray]:
    """Each constant buffer a stage declares, filled by variable name; a read variable no source gives raises."""
    registers, _textures, _samplers = variant.registers(stage)
    used = _used_registers(asm)
    sizes = {int(m[1]): int(m[2]) for m in re.finditer(r"dcl_constantbuffer CB(\d+)\[(\d+)\]", asm)}
    out, missing = {}, []
    for register, rows in sizes.items():
        buffer = np.zeros((rows, 4))
        name, variables = registers.get(register, ("?", []))
        material = name in MATERIAL_BUFFERS
        for variable, offset, rows_, columns, array in variables:
            first = offset // 16
            span = range(first, first + max(rows_, 1) * max(array, 1)) if rows_ > 1 or array else range(first, first + 1)
            if not used.get(register, set()) & set(span):
                continue                            # declared, not read by this variant
            value = _value(record, variable, defaults) if material else FRAME.get(variable)
            if value is None and name == "UnityPerMaterial":
                value = [0.0]                       # set by no one: Unity's material buffer starts zeroed
            if value is None:
                missing.append(f"{name}.{variable}")
                continue
            # only the words the variable holds: scalars share a row (_MainTexUseDisturb .x, _UseMainTexAsAlpha .y)
            words = min(len(value), len(span) * 4 - offset % 16 // 4,
                        max(rows_, 1) * max(array, 1) * 4 if rows_ > 1 or array else columns)
            flat = buffer.reshape(-1)
            flat[offset // 4:offset // 4 + words] = value[:words]
        out[register] = buffer
    if missing:
        raise BakeError("shader 讀了沒有值的變數：" + "、".join(missing[:5]))
    return out


def _used_registers(asm: str) -> dict[int, set[int]]:
    used: dict[int, set[int]] = {}
    for buffer, slot in re.findall(r"cb(\d+)\[(\d+)\]", asm):
        used.setdefault(int(buffer), set()).add(int(slot))
    return used


def _blend(variant: SC.Variant, record: dict, defaults: dict) -> str:
    """What the pass's colour output is, by its blend factors: ``premultiplied`` or ``straight``."""
    blend = (variant.state.get("rtBlend") or [{}])[0] if isinstance(variant.state.get("rtBlend"), list) \
        else variant.state.get("rtBlend0") or {}
    def factor(key):
        return int(_state(blend.get(key) or {}, record, defaults))
    pair = (factor("srcBlend"), factor("destBlend"))
    if pair == (ZERO, ONE):
        raise BakeError("混合方式 (Zero, One) 不改變畫面，不畫顏色")
    if pair == (ONE, ONE_MINUS_SRC_ALPHA):
        return "premultiplied"
    if pair == (SRC_ALPHA, ONE_MINUS_SRC_ALPHA):
        return "straight"
    raise BakeError(f"混合方式 {pair} 在 glTF 裡沒有對應（只烘預乘與一般的 alpha 混合）")


def _texels(uv: np.ndarray, size: tuple[int, int]):
    """Each texel centre inside a triangle (row, column, triangle, barycentrics); triangles too small to hold a
    centre are sampled at their middle instead, so a tiny effect is baked rather than lost."""
    from ..wuthering_waves.shader_bake import BakeError as NoTexel, _texels as texels
    image = np.stack([uv[..., 0], 1.0 - uv[..., 1]], axis=-1)       # image rows top first, as the bake map is saved
    w, h = size
    try:
        rows, columns, which, weights = texels(image, size)
    except NoTexel:
        rows = columns = which = np.zeros(0, np.int64)
        weights = np.zeros((0, 3))
    missing = np.setdiff1d(np.arange(len(uv)), which)
    if len(missing):
        middle = image[missing].mean(axis=1)
        rows = np.concatenate([rows, np.mod(np.floor(middle[:, 1] * h).astype(np.int64), h)])
        columns = np.concatenate([columns, np.mod(np.floor(middle[:, 0] * w).astype(np.int64), w)])
        which = np.concatenate([which, missing])
        weights = np.concatenate([weights, np.full((len(missing), 3), 1.0 / 3.0)])
    return rows, columns, which, weights


#: the mesh streams a vertex input semantic reads
STREAMS = {"POSITION": "co", "NORMAL": "no", "TANGENT": "tan", "COLOR": "col"}


def _signature(asm: str, which: str) -> dict[int, list[tuple[str, int, str]]]:
    """A stage's input or output signature: register → [(semantic, index, mask)]."""
    if which == "input":
        text = asm.split("// Input signature:", 1)[-1].split("// Output signature:", 1)[0]
    else:
        text = asm.split("// Output signature:", 1)[-1]
    out: dict[int, list] = {}
    for semantic, index, mask, register in re.findall(r"//\s+(\w+)\s+(\d+)\s+([xyzw]+)\s+(\d+)\s", text):
        out.setdefault(int(register), []).append((semantic.upper(), int(index), mask))
    return out


def _vertex_run(variant: SC.Variant, record: dict, defaults: dict, corners: dict, fill: float,
                textures: dict) -> dict[tuple[str, int], np.ndarray]:
    """The vertex stage run on every corner of the material's triangles: (semantic, index) of each output →
    its values (corners × 4).  An input the mesh does not have, or a component of one it has fewer of, is *fill*."""
    asm = variant.vertex_asm
    if asm is None:
        raise BakeError("變體沒有頂點階段")
    n = len(corners["uv0"])
    if "col" not in corners:
        # a mesh with no colour stream reads white: Unity's answer for a missing COLOR, and the only value that
        # leaves Si's wings visible in the game (the colour's alpha is their coverage; black would hide them)
        corners = dict(corners, col=np.ones((n, 4)))
    inputs = {}
    for register, semantics in _signature(asm, "input").items():
        value = np.full((n, 4), fill)
        for semantic, index, mask in semantics:
            if semantic == "TEXCOORD":
                data = corners.get(f"uv{index}")
            else:
                data = corners.get(STREAMS.get(semantic, ""))   # skin weights and indices: skipped at rest
            if data is not None:
                lanes = "xyzw".index(mask[0])
                width = min(len(mask), data.shape[1])
                value[:, lanes:lanes + width] = data[:, :width]
        inputs[register] = value
    buffers = _buffers(variant, record, defaults, asm, "vs")
    _registers, names, _samplers = variant.registers("vs")
    machine = vm.Machine(vm.parse(asm), n, inputs, buffers,
                         {r: textures[t] for r, t in names.items() if t in textures})
    machine.run()
    out = {}
    for register, semantics in _signature(asm, "output").items():
        words = machine.outputs.get(register)
        if words is None:
            continue
        for semantic, index, _mask in semantics:
            out[(semantic, index)] = words.view(np.float32).astype(np.float64)
    return out


def straight(colour: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """The pass's colour output (n × 4) as a straight colour and its coverage, as glTF blends them.

    A premultiplied output is the colour times its coverage; where it covers nothing but still adds light
    (_BlendMode 1 writes alpha 0: the colour is added onto what is behind) glTF has no additive blend, so such a
    texel covers by its brightness and is that colour over it — what is behind dims by (1 − brightness) where
    the game leaves it as it is."""
    alpha = np.clip(colour[:, 3], 0.0, 1.0)
    added = np.maximum(colour[:, :3], 0.0)
    if kind == "premultiplied":
        lone = (alpha < 1e-3) & (added.max(axis=1) > 1e-3)
        alpha = np.where(lone, np.minimum(1.0, added.max(axis=1)), alpha)
        rgb = added / np.maximum(alpha, 1e-6)[:, None]
    else:
        rgb = added
    return np.where(alpha[:, None] > 1e-6, rgb, 0.0), alpha


def bake_material(name: str, record: dict, document: dict, arrays: dict, folder: Path, doc: dict,
                  log=print) -> dict:
    """*record* baked: a map of its own and how the writers draw it, or :class:`BakeError`."""
    variant = SC.variant(doc, record.get("keywords") or [])
    if "UNLIT" not in variant.tags.get("RENDERTYPE", "").upper():
        raise BakeError(f"{record.get('shader')} 是打光的 shader，烘下去會帶入這個程式自己的光照")
    defaults = SC.property_defaults(doc)
    kind = _blend(variant, record, defaults)
    pixel_asm = dxbc.disassemble(variant.pixel)
    program = vm.parse(pixel_asm)
    registers, texture_names, _samplers = variant.registers()
    textures, sizes, textures_by_name = {}, [], {}
    for register, texture in texture_names.items():
        entry = record["maps"].get(texture)
        if entry is None and texture in FRAME_MAPS:
            textures[register] = textures_by_name[texture] = \
                np.asarray(FRAME_MAPS[texture], np.float32).reshape(1, 1, 4)
            continue
        if entry is None:
            # bound to nothing: the GPU samples the texture the shader declares for the property (white, black…)
            declared = next((q for q in doc["m_ParsedForm"]["m_PropInfo"]["m_Props"] if q["m_Name"] == texture), None)
            default = DEFAULT_TEXTURES.get(((declared or {}).get("m_DefTexture") or {}).get("m_DefaultName", ""))
            if default is None:
                raise BakeError(f"材質沒有 shader 讀的貼圖 {texture}")
            textures[register] = textures_by_name[texture] = np.asarray(default, np.float32).reshape(1, 1, 4)
            continue
        image = _image(folder / entry["file"], bool(entry.get("srgb")))
        textures[register] = textures_by_name[texture] = image
        sizes.append(image.shape[:2])
    buffers = _buffers(variant, record, defaults, pixel_asm, "ps")
    height, width = (max(s[0] for s in sizes), max(s[1] for s in sizes)) if sizes else (256, 256)
    scale = min(1.0, MOST / max(width, height))
    grid = (max(1, round(width * scale)), max(1, round(height * scale)))
    # the corners of the material's triangles, each stream the mesh has
    gathered: dict[str, list] = {}
    for index, part in enumerate(document["parts"]):
        if name not in part["materials"]:
            continue
        p = f"p{index}_"
        tri = arrays[p + "tri"][arrays[p + "slot"] == part["materials"].index(name)]
        if not len(tri):
            continue
        for key in ("co", "no", "tan", "col", "uv0", "uv1", "uv2", "uv3"):
            if p + key in arrays:
                stream = np.asarray(arrays[p + key])
                gathered.setdefault(key, []).append(stream[tri].reshape(-1, stream.shape[-1]))
    if "uv0" not in gathered:
        raise BakeError("沒有部件用這個材質")
    corners = {k: np.concatenate(v).astype(np.float64) for k, v in gathered.items()
               if len(v) == len(gathered["uv0"])}
    triangles = len(corners["uv0"]) // 3
    rows, columns, which, weights = _texels(corners["uv0"].reshape(triangles, 3, -1)[..., :2], grid)
    n = len(rows)
    pixel_inputs = _signature(pixel_asm, "input")

    reached_by_fill: dict[float, dict] = {}

    def run(fill: float, pick: np.ndarray, screens) -> np.ndarray:
        """The colour at the texels *pick*, averaged over *screens*, with what the mesh lacks at *fill*."""
        if fill not in reached_by_fill:
            reached_by_fill[fill] = _vertex_run(variant, record, defaults, corners, fill, textures_by_name)
        k = len(pick)
        interpolated = {key: (weights[pick][:, :, None] * values.reshape(triangles, 3, 4)[which[pick]]).sum(axis=1)
                        for key, values in reached_by_fill[fill].items()}
        total = np.zeros((k, 4))
        for x, y in screens:
            inputs = {}
            for register, semantics in pixel_inputs.items():
                value = np.full((k, 4), fill)
                for semantic, index, mask in semantics:
                    if semantic == "SV_POSITION":
                        value[:, 0], value[:, 1], value[:, 2], value[:, 3] = x, y, 0.5, 1.0
                        continue
                    data = interpolated.get((semantic, index))
                    if data is not None:
                        lanes = ["xyzw".index(c) for c in mask]
                        value[:, lanes] = data[:, lanes]
                inputs[register] = value
            machine = vm.Machine(program, k, inputs, buffers, textures)
            discarded = machine.run()
            colour = machine.outputs.get(0, np.zeros((k, 4), np.uint32)).view(np.float32).astype(np.float64)
            colour[discarded] = 0.0
            total += np.nan_to_num(colour)
        return total / len(screens)

    # what the colour depends on, tried on a sample: an input the mesh lacks must not change it, and when the
    # screen position does not either (no dither, no screen-space read) one position is the whole bake
    sample = np.unique(np.linspace(0, n - 1, min(n, PROBE)).astype(np.int64))
    tried = [run(0.0, sample, [screen]) for screen in SCREEN]
    if np.abs(run(1.0, sample, SCREEN[:1]) - tried[0]).max() > SENSITIVE:
        raise BakeError("顏色取決於網格沒有的輸入")
    screens = SCREEN[:1] if all(np.array_equal(t, tried[0]) for t in tried[1:]) else SCREEN
    colour = run(0.0, np.arange(n), screens)
    rgb, alpha = straight(colour, kind)
    strength = max(1.0, float(np.percentile(rgb.max(axis=1)[alpha > 0.01], 99.5)) if (alpha > 0.01).any() else 1.0)
    picture = np.full((grid[1], grid[0], 4), np.nan)
    picture[rows, columns, :3] = np.clip(rgb / strength, 0.0, 1.0)
    picture[rows, columns, 3] = alpha
    # texels no triangle covers take the nearest covered one's value: filtering reads across an island's edge
    from ..wuthering_waves.shader_bake import _fill
    picture = np.stack([_fill(picture[..., c]) for c in range(4)], axis=-1)
    encoded = picture.copy()
    linear = encoded[..., :3]
    encoded[..., :3] = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    target = folder / "baked" / f"{name}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(np.clip(encoded, 0, 1) * 255).astype(np.uint8), "RGBA").save(target)
    key = "_Baked"
    out = dict(record, maps=dict(record["maps"]))
    out["maps"][key] = {"file": target.relative_to(folder).as_posix(), "srgb": True,
                        "scale": [1.0, 1.0], "offset": [0.0, 0.0]}
    out.update(base=key, tint=[0.0, 0.0, 0.0, 1.0], blend="alpha", alpha_from="a",
               emission={"map": key, "color": [1.0, 1.0, 1.0], "strength": strength},
               coverage=f"the game's {record.get('shader')} pass run over each texel at rest (vfx_bake, "
                        f"variant {' '.join(variant.keywords)})")
    return out


#: raised when what a bake gives changes, so a bake kept from an earlier rule is not used again
VERSION = 3


def _cache_dir() -> Path:
    from . import store
    return store.root() / "baked"


def _cache_key(name: str, record: dict, document: dict, arrays: dict, doc: dict) -> str:
    """The bake's inputs: this module's rule, the shader, the material, and the mesh it covers."""
    import hashlib
    digest = hashlib.sha256(f"{VERSION}|{doc['m_ParsedForm']['m_Name']}|".encode())
    digest.update(json.dumps(doc["m_ParsedForm"]["m_SubShaders"][0]["m_Passes"][0]["m_State"], sort_keys=True,
                             default=str).encode())
    digest.update(json.dumps({k: v for k, v in record.items() if k != "maps"}, sort_keys=True, default=str).encode())
    for entry in record["maps"].values():
        digest.update(entry["file"].encode())
    for index, part in enumerate(document["parts"]):
        if name in part["materials"]:
            for key in ("tri", "slot", "co", "no", "col", "uv0", "uv1", "uv2", "uv3"):
                if f"p{index}_{key}" in arrays:
                    digest.update(np.ascontiguousarray(arrays[f"p{index}_{key}"]).tobytes())
    return digest.hexdigest()[:24]


def _cached(key: str, name: str, folder: Path) -> dict | None:
    import shutil
    record_path, image_path = _cache_dir() / f"{key}.json", _cache_dir() / f"{key}.png"
    if not (record_path.is_file() and image_path.is_file()):
        return None
    baked = json.loads(record_path.read_text(encoding="utf-8"))
    target = folder / baked["maps"]["_Baked"]["file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image_path, target)
    return baked


def _keep(key: str, baked: dict, folder: Path) -> None:
    import shutil
    _cache_dir().mkdir(parents=True, exist_ok=True)
    shutil.copyfile(folder / baked["maps"]["_Baked"]["file"], _cache_dir() / f"{key}.png")
    (_cache_dir() / f"{key}.json").write_text(json.dumps(baked, ensure_ascii=False), encoding="utf-8")


def complete(document_path: Path, shaders, log=print) -> list[str]:
    """Every material of the document at *document_path* the writers would not draw, baked, or marked
    invisible with why; returns what was not baked.  *shaders(names)* fetches shaders not in the cache."""
    document_path = Path(document_path)
    document = json.loads(document_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(document_path.parent / document["arrays"]))
    # the materials some triangle is drawn with (a part can list one no slot uses)
    drawn = {part["materials"][int(slot)] for index, part in enumerate(document["parts"])
             for slot in np.unique(arrays[f"p{index}_slot"]) if int(slot) < len(part["materials"])}
    folder = document_path.parent
    for name, record in list(document["materials"].items()):
        if name in drawn:
            document["materials"][name] = npr_emission(name, record, folder)
    pending = {name: record for name, record in document["materials"].items()
               if not drawable(record) and name in drawn}
    if not pending:
        document_path.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
        return []
    wanted = sorted({record.get("shader") for record in pending.values() if record.get("shader")})
    missing = [name for name in wanted if SC.load(name) is None]
    if missing and shaders is not None:
        shaders(missing)
    refused = []
    for name, record in pending.items():
        try:
            doc = SC.load(record.get("shader") or "")
            if doc is None:
                raise BakeError(f"找不到 shader {record.get('shader')}")
            if _draws_no_colour(doc, record):
                raise BakeError(f"{record.get('shader')} 只寫深度，不畫顏色")
            constant = unbound_base(record, doc)
            if constant is not None:
                document["materials"][name] = constant
                log(f"{name}：沒有底色圖，照 shader 宣告的預設貼圖畫成定色")
                continue
            key = _cache_key(name, record, document, arrays, doc)
            baked = _cached(key, name, folder)
            if baked is None:
                baked = bake_material(name, record, document, arrays, folder, doc, log)
                _keep(key, baked, folder)
            document["materials"][name] = baked
            log(f"{name}：以遊戲的 {record.get('shader')} 烘焙")
        except (ValueError, vm.ShaderError, KeyError, OSError) as failure:    # BakeError, ShaderCodeError…
            document["materials"][name] = dict(record, native_surface_unmapped=True, unmapped_reason=str(failure))
            refused.append(f"{name}（{failure}）")
            log(f"{name}：沒辦法畫出，設為不可見（{failure}）")
    document_path.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    return refused


#: the character shaders whose emission is read here — each ForwardLit's _EMISSION variant adds
#: _EmissionMap.rgb (first UV set) × _EmissionColor × _EmissionBrightness onto its lit colour, whatever lights it
NPR_SHADERS = ("HGRP/CharacterNPR", "HGRP/CharacterNPR_Hair", "HGRP/CharacterNPR_Skin", "HGRP/CharacterNPR_Eye",
               "HGRP/CharacterNPR_LiquidAg")
#: and the ones whose _PARALLAX_MAP term is read (the crystal: _ParallaxTex.r × _ParallaxColor × base alpha)
PARALLAX_SHADERS = ("HGRP/CharacterNPR",)


def _linear(values) -> np.ndarray:
    """A material colour as the shader reads it (Unity's colours are sRGB, converted for a linear pipeline)."""
    c = np.asarray(values, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def npr_emission(name: str, record: dict, folder: Path) -> dict:
    """*record* with the light its shader adds onto its lit colour baked as its ``emission``, or as it was.

    Read from the pass (ForwardLit): ``_EmissionMap``.rgb at the first UV set × ``_EmissionColor`` ×
    ``_EmissionBrightness``, plus — with ``_PARALLAX_MAP`` — ``_ParallaxTex``.r × ``_ParallaxColor`` × the base
    map's alpha × ``_BaseColor``.a, the parallax map looked up where the view ray meets it: seen head-on, at its
    own tiling (``_ParallaxTex_ST``) with no march offset.  Si's horns are this: a dim emission map and the
    parallax crystal where the base map's alpha marks them."""
    if record.get("shader") not in NPR_SHADERS or record.get("emission") or not record.get("base"):
        return record
    keywords = set(record.get("keywords") or [])
    floats, colours, maps = record.get("floats") or {}, record.get("colors") or {}, record["maps"]
    use_emission = "_EMISSION" in keywords and "_EmissionMap" in maps
    use_parallax = (record.get("shader") in PARALLAX_SHADERS and "_PARALLAX_MAP" in keywords
                    and "_ParallaxTex" in maps and floats.get("_UseParallax", 1.0) != 0.0)
    if not (use_emission or use_parallax):
        return record
    base_entry = maps[record["base"]]
    base = _image(folder / base_entry["file"], bool(base_entry.get("srgb")))[::-1]   # rows top first
    height, width = base.shape[:2]
    light = np.zeros((height, width, 3))
    if use_emission:
        entry = maps["_EmissionMap"]
        image = _image(folder / entry["file"], bool(entry.get("srgb")))[::-1]
        if image.shape[:2] != (height, width):
            image = _resize(image, (height, width))
        light += image[..., :3] * _linear(colours.get("_EmissionColor", (1, 1, 1, 1))[:3]) \
            * float(floats.get("_EmissionBrightness", 1.0))
    if use_parallax:
        entry = maps["_ParallaxTex"]
        image = _image(folder / entry["file"], bool(entry.get("srgb")))
        scale, offset = entry.get("scale", (1.0, 1.0)), entry.get("offset", (0.0, 0.0))
        v, u = np.meshgrid(1.0 - (np.arange(height) + 0.5) / height, (np.arange(width) + 0.5) / width, indexing="ij")
        pu, pv = u * scale[0] + offset[0], v * scale[1] + offset[1]
        h, w = image.shape[:2]
        sample = image[np.mod(np.floor(pv * h).astype(int), h), np.mod(np.floor(pu * w).astype(int), w), 0]
        coverage = base[..., 3] * float(colours.get("_BaseColor", (1, 1, 1, 1))[3])
        light += (sample * coverage)[..., None] * _linear(colours.get("_ParallaxColor", (0, 0, 0, 1))[:3])
    if light.max() <= 0.0:
        return record
    strength = max(1.0, float(np.percentile(light.max(axis=2)[light.max(axis=2) > 0], 99.5)))
    scaled = np.clip(light / strength, 0.0, 1.0)
    encoded = np.where(scaled <= 0.0031308, scaled * 12.92, 1.055 * np.power(scaled, 1 / 2.4) - 0.055)
    target = folder / "baked" / f"{name}_emission.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(encoded * 255).astype(np.uint8), "RGB").save(target)
    out = dict(record, maps=dict(maps))
    out["maps"]["_EmissionBaked"] = {"file": target.relative_to(folder).as_posix(), "srgb": True,
                                     "scale": [1.0, 1.0], "offset": [0.0, 0.0]}
    out["emission"] = {"map": "_EmissionBaked", "color": [1.0, 1.0, 1.0], "strength": strength}
    return out


def _resize(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    channels = [np.asarray(Image.fromarray(image[..., c].astype(np.float32), "F").resize((shape[1], shape[0]),
                                                                                       Image.BILINEAR))
                for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)
