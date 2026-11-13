"""Wuthering Waves' eyes as the game's own base pass draws them, baked for every format.

An eye material (``MSM_ToonFace`` with the static switch ``UseEyeSDF``) draws its highlights in its base pass,
not as another material: the instance's ``HeightLightMap`` (``T_EyesSecondHeightLight*_EG``) is read at a
coordinate the pass works out from the first UV set, the material's position, size and turn parameters and the
vertex colour's G (which picks between two positions), and the eye's map ``EM`` gives the parallax height and a
second highlight shape.  What the pass writes is read off its compiled code (IslandLife §7.1491):

* ``SV_Target3`` (GBufferC) is the base colour the deferred toon lighting lights;
* ``SV_Target0`` is the scene colour the pass itself adds: Unreal's emissive, which no light changes.

The highlight map's channels lerp the eye's colour towards two highlight colours before it is split into those
two, and one channel is added to the emissive alone — so the highlight is partly lit and partly not.  Rather
than writing those terms down as a rule, the pass is run (``shaders.vm``) over every texel of the eye's map, as
:mod:`shader_bake` runs the shadow-depth pass: the surface there interpolated, seen head-on (the view along its
normal: the parallax walks no distance), at time 0, pre-exposure 1.  The two targets become the material's base
map and its emission map (``emission``: map × colour × strength), which glTF, FBX and Blender all carry; the
source's own map stays among the maps (``eye.source_base``) for a reader that runs the game's program itself.

Every frame value the pass reads is either stated here (:data:`FRAME`) or shown not to change the two targets:
each other slot of the View, Primitive, light and parameter-collection buffers it reads is given random values on
a sample, every input the mesh has no value for is given 0 and 1, and the screen position is moved; a target
that moves refuses the bake.  Every base-pass variant of the shader map must give the same targets.  The
contribution of each channel of the highlight map is measured (that channel at 0) and kept in the statement.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .shaders import archive as A
from .shaders import dxbc, inline, preshader, vm

#: bump when what a bake gives changes: the cache key carries it
VERSION = 2
#: the base pass's render targets read: the scene colour it adds (not lit) and GBufferC's base colour (lit)
EMISSION, BASE_COLOR = 0, 3
#: the frame a bake gives, against Unreal 4.26's View and Primitive buffers as the base pass reads them:
#:   View [44..47] SVPositionToTranslatedWorld — identity: a point's position is its SV_Position (−N per point,
#:   so the camera, at the origin, looks along the normal);  [134] the viewport's origin 0;  [135] its size and
#:   inverse 1;  [144] Diffuse/NormalOverrideParameter (0, 0, 0, 1);  [147].w 1;  [148] game time 0;
#:   [149] mip bias 0;  [285] pre-exposure 1 (the scene colour as the material gives it);
#:   Primitive [0..3] LocalToWorld identity, [4].w the determinant's sign 1.
FRAME = {"View": {44: (1.0, 0.0, 0.0, 0.0), 45: (0.0, 1.0, 0.0, 0.0), 46: (0.0, 0.0, 1.0, 0.0),
                  47: (0.0, 0.0, 0.0, 1.0), 134: (0.0, 0.0, 0.0, 0.0), 135: (1.0, 1.0, 1.0, 1.0),
                  144: (0.0, 0.0, 0.0, 1.0), 147: (0.0, 0.0, 0.0, 1.0), 148: (0.0, 0.0, 0.0, 0.0),
                  149: (0.0, 0.0, 0.0, 0.0), 285: (1.0, 1.0, 1.0, 1.0)},
         "Primitive": {0: (1.0, 0.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0, 0.0), 2: (0.0, 0.0, 1.0, 0.0),
                       3: (0.0, 0.0, 0.0, 1.0), 4: (1.0, 1.0, 1.0, 1.0)}}
#: points the sensitivity tests and the variant check run; lanes run at once
PROBE, CHUNK = 4096, 16384
#: how far a target may move (linear, any channel) and still count as unmoved
STILL = 1e-4
#: the most two triangles over one texel may disagree (a map holds one value per texel)
OVERLAP = 0.02
#: the values each read of the immediate constant buffer is given in turn (``skin_band.ICB``)
ICB = (0.0, 0.777778)
#: the map the highlights come from (the instance's texture parameter of that name)
HIGHLIGHT = "HeightLightMap"


class EyeError(ValueError):
    pass


def eye_materials(document: dict) -> list[str]:
    """The scene's eye materials a triangle draws with (``sidecar.slot_role``: ``MSM_ToonFace`` + ``UseEyeSDF``)."""
    from ..interchange.sidecar import slot_role
    return [name for name, record in document["materials"].items()
            if not record.get("unused") and record.get("base") and slot_role(record)[0] == "eye"]


def _linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


class _Machine(vm.Machine):
    """The interpreter with each texture read the way its asset says past its edge: ``clamp`` holds, per texture
    register, which axes are ``TA_Clamp`` (the interpreter itself wraps)."""

    def __init__(self, *args, clamp=None):
        super().__init__(*args)
        self.clamp = clamp or {}

    def sample(self, register, uv):
        axes = self.clamp.get(register)
        if axes and any(axes):
            uv = np.array(uv, dtype=np.float64)
            image = self.textures[register]
            h, w = image.shape[:2]
            # clamped: the coordinate held within the edge texels' centres, so the bilinear read stays on them
            for axis, size in ((0, w), (1, h)):
                if axes[axis]:
                    uv[:, axis] = np.clip(np.nan_to_num(uv[:, axis]), 0.5 / size, 1.0 - 0.5 / size)
        return super().sample(register, uv)


class _Pass:
    """One base-pass pixel shader with its buffers and textures, run over points (``run``)."""

    def __init__(self, shader: dxbc.Shader, buffers: dict, textures: dict, uv_sets: int, clamp: dict):
        from .shader_bake import _packed
        self.shader, self.buffers, self.textures, self.clamp = shader, buffers, textures, clamp
        # the immediate constant buffer (a dither table: it decides which pixels a dithered fade discards) is not
        # run: each read of it is given a constant, twice (``ICB``); the targets must not differ between the two
        asm = re.sub(r"dcl_immediateConstantBuffer \{.*?\}\s*\}", "", self.clean(shader.asm), flags=re.S)
        self.programs = [vm.parse(re.sub(r"icb\[[^\]]*\](\.[xyzw]+)?", f"l({value})", asm)) for value in ICB]
        self.program = self.programs[0]
        self.packed = _packed(shader)
        self.uv_sets = uv_sets

    @staticmethod
    def clean(asm: str) -> str:
        # ``[precise]`` only forbids reassociation (same value); a conservative depth output is a depth output
        asm = re.sub(r" ?\[precise(?:\([xyzw]+\))?\]", "", asm)
        return re.sub(r"oDepth(?:LE|GE)", "oDepth", asm)

    def run(self, points: dict, fill: dict | None = None, buffers: dict | None = None, textures: dict | None = None,
            shift=(0.0, 0.0), program=None) -> dict[int, np.ndarray]:
        """``{target: n × 4}`` of :data:`EMISSION` and :data:`BASE_COLOR` at each point; NaN where the pass
        discards the point (a dithered fade)."""
        from .shader_bake import _input
        buffers = buffers or self.buffers
        textures = textures or self.textures
        n = len(points["normal"])
        out = {EMISSION: np.zeros((n, 4)), BASE_COLOR: np.zeros((n, 4))}
        view = self.shader.buffers.get("View")
        for start in range(0, n, CHUNK):
            part = {k: v[start:start + CHUNK] for k, v in points.items() if k != "uvs"}
            part["uvs"] = [uv[start:start + CHUNK] for uv in points["uvs"]]
            m = len(part["normal"])
            inputs = {}
            for register, semantics in self.shader.inputs.items():
                value = np.zeros((m, 4))
                for semantic, index, mask in semantics:
                    if semantic == "SV_Position":
                        # the point's translated-world position (View [44..47] is the identity): one unit behind
                        # it along the normal is the camera; *shift* moves the screen position, View [47] moves
                        # back by as much
                        source = np.zeros((m, 4))
                        source[:, :3] = -part["normal"]
                        source[:, :2] += shift
                        source[:, 3] = 1.0
                    else:
                        source = _input(semantic, index, mask, part, self.packed, (0.0, 0.0))
                    for c in mask:
                        value[:, "xyzw".index(c)] = source[:, "xyzw".index(c)]
                for (filled, component), given in (fill or {}).items():
                    if filled == register:
                        value[:, component] = given
                inputs[register] = value
            lane_buffers = dict(buffers)
            if view is not None and shift != (0.0, 0.0):
                moved = np.array(buffers[view], dtype=np.float64)
                moved[47, :2] -= shift
                lane_buffers[view] = moved
            machine = _Machine(program or self.program, m, inputs, lane_buffers, textures, clamp=self.clamp)
            discarded = machine.run()
            for target in out:
                words = machine.outputs.get(target)
                if words is None:
                    raise EyeError(f"the base pass writes no SV_Target{target}")
                values = words.view(np.float32).astype(np.float64)
                values[discarded] = np.nan
                out[target][start:start + m] = values
        return out


def _moved(a: dict, b: dict) -> float:
    """The most a target moved between two runs, over the points neither run discards."""
    most = 0.0
    for t in a:
        both = ~(np.isnan(a[t][:, :3]).any(1) | np.isnan(b[t][:, :3]).any(1))
        if both.any():
            most = max(most, float(np.abs(a[t][both, :3] - b[t][both, :3]).max()))
    return most


def _sensitivity(run: _Pass, sample: dict, unknown: list) -> dict:
    """Refuse a bake whose targets move with anything this does not state; what was tried, for the evidence."""
    reference = run.run(sample, {u: 0.0 for u in unknown})
    tried = {}
    if unknown:
        tried["unknown_inputs"] = [f"v{r}.{'xyzw'[c]}" for r, c in unknown]
        moved = _moved(reference, run.run(sample, {u: 1.0 for u in unknown}))
        if moved > STILL:
            raise EyeError(f"the base pass's targets depend on inputs the mesh has no value for (moved {moved:.4g})")
    moved = _moved(reference, run.run(sample, {u: 0.0 for u in unknown}, shift=(0.37, 0.61)))
    if moved > STILL:
        raise EyeError(f"the base pass's targets depend on the screen position (moved {moved:.4g})")
    tried["screen_position"] = "moved by (0.37, 0.61): unmoved"
    names = {register: name for name, register in run.shader.buffers.items()}
    reads = sorted({(int(a), int(b)) for a, b in re.findall(r"cb(\d+)\[(\d+)\]", run.shader.asm.split("ps_5_0", 1)[-1])})
    rng = np.random.default_rng(1491)
    unmoved, collections = [], set()
    for register, slot in reads:
        name = names.get(register, f"cb{register}")
        if name.startswith("MaterialCollection"):
            collections.add(f"{name}[{slot}]")       # given its asset's defaults, as every bake gives it
            continue
        if name == "Material" or slot in FRAME.get(name, {}):
            continue
        buffers = dict(run.buffers)
        changed = np.array(buffers[register], dtype=np.float64)
        changed[slot] = rng.uniform(-2.0, 2.0, 4)
        buffers[register] = changed
        moved = _moved(reference, run.run(sample, {u: 0.0 for u in unknown}, buffers=buffers))
        if moved > STILL:
            raise EyeError(f"the base pass's targets depend on {name}[{slot}], which a bake has no value for "
                           f"(moved {moved:.4g})")
        unmoved.append(f"{name}[{slot}]")
    tried["frame_slots_unmoved"] = unmoved
    tried["collection_slots_at_defaults"] = sorted(collections)
    return tried


def _buffers(shader: dxbc.Shader, material_values, collections) -> dict:
    from .shader_bake import _collection, _sizes
    sizes = _sizes(shader.asm)
    buffers = {register: np.zeros((sizes.get(register, 1), 4)) for register in sizes}
    for buffer_name, values in FRAME.items():
        register = shader.buffers.get(buffer_name)
        if register is not None:
            for slot, value in values.items():
                if slot < len(buffers[register]):
                    buffers[register][slot] = value
    material = shader.buffers.get("Material")
    if material is None:
        raise EyeError("the base pass does not read the Material buffer")
    buffers[material][:min(len(material_values), len(buffers[material]))] = material_values[:len(buffers[material])]
    for number, collection in enumerate(collections):
        target = shader.buffers.get(f"MaterialCollection{number}")
        if target is not None:
            values = _collection(collection)
            buffers[target][:min(len(values), len(buffers[target]))] = values[:len(buffers[target])]
    return buffers


def bake_material(name: str, record: dict, document: dict, arrays: dict, case: Path, store: A.Archive,
                  uexp: bytes, root: dict, collections: list[dict], address) -> dict:
    """*name*'s two targets over its base map: ``{"layers": [{"base": H × W × 4 linear, "emission": H × W × 3
    linear}, …], "layer": each triangle's map, "statement": …}``.  *address* gives a texture package's
    ``(AddressX, AddressY)`` (``GameFiles.address``)."""
    from .shader_bake import _frozen_values, _image, _subset, collections_in_map, surface_points, unknown_inputs
    from .skin_band import base_passes
    resources = inline.read(uexp)
    if not resources:
        raise EyeError("the instance's package has no inline shader map")
    uniforms = resources[0]
    digests = [uexp[i:i + 20] for i in range(len(uexp) - 19) if store.map_index(uexp[i:i + 20]) is not None]
    if not digests:
        raise EyeError("the instance's shader map is not in the SM5 archive")
    passes = base_passes(store, store.map_index(digests[0]))
    if not passes:
        raise EyeError("the shader map has no base-pass pixel shader")
    maps = record["maps"]
    referenced = [Path(t.get("ObjectPath", "")).stem.split(".")[0] if isinstance(t, dict) else None
                  for t in root.get("ReferencedTextures", [])]
    packages = [t.get("ObjectPath", "").rsplit(".", 1)[0] if isinstance(t, dict) else None
                for t in root.get("ReferencedTextures", [])]
    textures_of = (record.get("native_material") or {}).get("Textures") or {}

    def texture_of(reference: int, parameter: str | None):
        key = parameter if parameter in maps else (referenced[reference] if 0 <= reference < len(referenced) else None)
        if key not in maps:
            raise EyeError(f"the material's maps have no {parameter or reference}")
        return key, maps[key]

    def size_of(reference: int):
        named = next((p for p, r in uniforms.textures if r == reference), None)
        with Image.open(case / texture_of(reference, named)[1]["file"]) as opened:
            return opened.size

    material_values = _frozen_values(record, uniforms, size_of)
    collections = collections_in_map(uexp, collections)
    base_record = maps[record["base"]]
    base = _image(case / base_record["file"], bool(base_record.get("srgb")))
    size = (base.shape[1], base.shape[0])
    points, rows, columns, which, count, uv_count = surface_points(name, document, arrays, size)
    if not len(rows):
        raise EyeError("no texel of the base map lies on the material's triangles")
    runs = []
    for index, shader in passes:
        textures, bound, clamp = {}, {}, {}
        for register, (buffer, resource) in shader.textures.items():
            if buffer != shader.buffers.get("Material") or resource % 2 or resource // 2 >= len(uniforms.textures):
                raise EyeError(f"texture register t{register} is not one of the material's 2D textures")
            parameter, reference = uniforms.textures[resource // 2]
            key, map_record = texture_of(reference, parameter)
            textures[register] = _image(case / map_record["file"], bool(map_record.get("srgb")))
            # the texture the instance binds to the parameter, else the one the root material references
            bound_object = textures_of.get(parameter) if parameter else None
            package = ((bound_object or {}).get("ObjectPath", "").rsplit(".", 1)[0]
                       or (packages[reference] if 0 <= reference < len(packages) else None))
            if not package:
                raise EyeError(f"no texture asset for {parameter or reference}: its address mode is unknown")
            modes = tuple(address(package))
            if any(mode not in ("TA_Wrap", "TA_Clamp") for mode in modes):
                raise EyeError(f"{package}: address mode {modes} is not read here")
            clamp[register] = tuple(mode == "TA_Clamp" for mode in modes)
            bound[register] = (parameter, key, list(modes))
        run = _Pass(shader, _buffers(shader, material_values, collections), textures, uv_count, clamp)
        runs.append((index, run, bound))
    pick = np.unique(np.linspace(0, len(rows) - 1, min(PROBE, len(rows))).astype(np.int64))
    sample = _subset(points, pick)
    # every variant, with the dither table at either value, gives the same targets where it draws; the bake runs
    # the one that discards least of the eye (a variant without the dithered fade discards nothing)
    results = []
    for index, run, bound in runs:
        given = {u: 0.0 for u in unknown_inputs(run.shader, uv_count)}
        for program in run.programs:
            results.append((index, run, bound, given, run.run(sample, given, program=program)))
    reference = results[0][4]
    differ = sorted({f"{index} ({_moved(reference, result):.3g})" for index, _r, _b, _g, result in results[1:]
                     if _moved(reference, result) > 1e-3})
    if differ:
        raise EyeError("the shader map's base-pass variants give different targets: " + ", ".join(differ))
    first_index, first, bound, fill, _result = min(results, key=lambda r: int(np.isnan(r[4][BASE_COLOR][:, 0]).sum()))
    tried = _sensitivity(first, sample, [u for u in fill])
    targets = first.run(points, fill)
    values = np.concatenate([targets[BASE_COLOR][:, :3], np.maximum(targets[EMISSION][:, :3], 0.0)], axis=1)
    shape = (size[1], size[0])
    takes, layer = _layers(rows, columns, which, values, count, shape)
    layers = [{"base": np.concatenate([_fill(rows[take], columns[take], values[take, :3], shape), base[:, :, 3:4]],
                                      axis=2),
               "emission": _fill(rows[take], columns[take], values[take, 3:], shape)} for take in takes]
    keys = rows * size[0] + columns
    statement = {"base_pass_pixel_shaders": [r[0] for r in runs], "ran": first_index,
                 "shader_map_sha1": digests[0].hex(), "texels": int(len(np.unique(keys))),
                 "samples": int(len(rows)), "discarded_samples": int(np.isnan(values[:, 0]).sum()),
                 "unknown_inputs_given": "0",
                 "textures": {parameter or key: {"map": key, "address": modes}
                              for parameter, key, modes in bound.values()}, **tried,
                 "highlight": _highlight(first, sample, fill, bound)}
    if len(layers) > 1:
        statement["overlapping_islands"] = (f"triangles over the same texels are drawn differently (the vertex colour "
                                            f"picks the highlight's position): {len(layers)} maps")
    return {"layers": layers, "layer": layer, "statement": statement}


#: the most maps one eye material is split into (``shader_bake.LAYERS``)
LAYERS = 4


def _layers(rows, columns, which, values, triangles: int, shape) -> tuple[list[np.ndarray], np.ndarray]:
    """The samples each map takes, and each triangle's map: one map holds one value per texel, but UV islands laid
    over each other may be drawn differently (both eyes on one island, the vertex colour moving the highlight).
    Each map is filled, first come, from the triangles still unplaced; those whose samples lie on average more
    than :data:`OVERLAP` from it go on to the next (``shader_bake._layers``, on colours).  A discarded sample
    (NaN) agrees with anything."""
    left = np.ones(triangles, bool)
    layer = np.zeros(triangles, np.int64)
    takes = []
    keys = rows * shape[1] + columns
    for number in range(LAYERS):
        if not left.any():
            break
        take = left[which]
        if number < LAYERS - 1:
            chosen = np.flatnonzero(take & ~np.isnan(values[:, 0]))
            _unique, first = np.unique(keys[chosen], return_index=True)
            table = np.full((shape[0] * shape[1], values.shape[1]), np.nan)
            table[keys[chosen][first]] = values[chosen][first]
            distance = np.nan_to_num(np.abs(values - table[keys]).max(axis=1), nan=0.0)
            total = np.bincount(which[take], distance[take], minlength=triangles)
            counted = np.bincount(which[take], minlength=triangles)
            agree = left & (total <= OVERLAP * np.maximum(counted, 1))
            if not agree.any():
                agree = left.copy()
        else:
            agree = left.copy()
        layer[agree] = number
        left &= ~agree
        takes.append(agree[which] & ~np.isnan(values[:, 0]))
    return takes, layer


def _highlight(run: _Pass, points: dict, fill: dict, bound: dict) -> dict:
    """What each channel of :data:`HIGHLIGHT` does to the two targets at *points* (the probe sample): the targets
    with that channel at 0."""
    register = next((r for r, (parameter, _key, _modes) in bound.items() if parameter == HIGHLIGHT), None)
    if register is None:
        return {"status": None, "reason": f"the base pass binds no {HIGHLIGHT}"}
    out = {"status": "read", "map": bound[register][1], "address": bound[register][2], "channels": {}}
    targets = run.run(points, fill)
    for channel in range(4):
        textures = dict(run.textures)
        image = np.array(textures[register])
        image[..., channel] = 0.0
        textures[register] = image
        without = run.run(points, fill, textures=textures)
        row = {}
        for target, label in ((EMISSION, "emission"), (BASE_COLOR, "base_color")):
            delta = np.abs(targets[target][:, :3] - without[target][:, :3]).max(axis=1)
            finite = np.isfinite(delta)
            row[label] = {"samples_changed": int((delta[finite] > 1 / 255).sum()), "samples": int(finite.sum()),
                          "max_change": round(float(delta[finite].max()), 5) if finite.any() else 0.0}
        out["channels"]["RGBA"[channel]] = row
    return out


def _fill(rows, columns, values: np.ndarray, shape) -> np.ndarray:
    """A map of *values* at their texels (the first sample of each), every other texel its nearest covered one."""
    from scipy import ndimage
    image = np.full(shape + (values.shape[1],), np.nan)
    image[rows[::-1], columns[::-1]] = values[::-1]                # the first sample over a texel wins
    empty = np.isnan(image[..., 0])
    _distance, (iy, ix) = ndimage.distance_transform_edt(empty, return_indices=True)
    return image[iy, ix]


def bake_case(case: Path, files, log=print, step=lambda text: None) -> None:
    """Every eye material of the scene in *case* drawn from what its base pass writes (see the module): the base
    map and an emission map replaced or added, ``eye`` stating it.  *files* is the shader bake's
    (``shader_bake.bake_case``).  Results are kept by the material and its geometry under ``files.cache``'s sibling
    ``eye``; a material the pass cannot be run for keeps its map and states why (``eye.status`` None)."""
    case = Path(case)
    path = case / "source_document.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    names = eye_materials(document)
    if not names:
        return
    arrays = dict(np.load(case / document["arrays"]))
    folder = Path(files.cache).parent / "eye"
    store, moved = None, False
    for name in names:
        record = document["materials"][name]
        if record.get("eye"):
            continue                                            # baked already
        geometry = hashlib.sha256()
        for index, part in enumerate(document["parts"]):
            if name in part["materials"]:
                for key in ("tri", "slot", "uv0", "uv1", "uv2", "uv3", "no", "tan", "col"):
                    if f"p{index}_{key}" in arrays:
                        geometry.update(np.ascontiguousarray(arrays[f"p{index}_{key}"]).tobytes())
        key = hashlib.sha256(f"{VERSION}|{record.get('source_json_sha256', '')}|{name}|{geometry.hexdigest()}"
                             .encode()).hexdigest()[:24]
        cached = folder / f"{key}.npz"
        step("以遊戲 shader 烘焙眼睛的高光")
        try:
            if cached.is_file():
                with np.load(cached) as stored:
                    baked = {"layers": [{"base": stored[f"base{k}"], "emission": stored[f"emission{k}"]}
                                        for k in range(int(stored["count"]))],
                             "layer": stored["layer"], "statement": json.loads(str(stored["statement"]))}
            else:
                if store is None:
                    store = A.read(files.archive())
                root, collections = files.root_of(record["source_package"])
                log(f"{name}：以遊戲的 base pass 烘焙眼睛（第一次要一兩分鐘）…")
                baked = bake_material(name, record, document, arrays, case, store,
                                      files.package(record["source_package"]), root, collections, files.address)
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cached, count=len(baked["layers"]), layer=baked["layer"],
                                    statement=json.dumps(baked["statement"]),
                                    **{f"{kind}{k}": layer[kind].astype(np.float32)
                                       for k, layer in enumerate(baked["layers"]) for kind in ("base", "emission")})
        except (EyeError, vm.ShaderError, preshader.PreshaderError, OSError, KeyError, ValueError,
                IndexError) as failure:
            log(f"{name}：遊戲的 base pass 無法在這裡執行（{failure}），眼睛照底色圖輸出、沒有高光")
            record["eye"] = {"game": "wuwa", "status": None,
                             "reason": f"the game's base pass could not be run for this eye: {failure}"}
            continue
        original = json.loads(json.dumps(record))
        for number, layer in enumerate(baked["layers"]):
            # triangles over the same texels drawn another way: a material (and a part) of their own
            target = record if number == 0 else document["materials"].setdefault(
                f"{name}_Overlap{number}", json.loads(json.dumps(original)))
            target["eye"] = apply(f"{name}_Overlap{number}" if number else name, target, layer,
                                  baked["statement"], case)
            log(f"{name}：眼睛的 base pass 烘成底色與發光（{number + 1}／{len(baked['layers'])}，"
                f"發光強度 {target['eye']['emission']['strength']:.3g}）")
        if len(baked["layers"]) > 1:
            from .shader_bake import _move_layers
            _move_layers(document, arrays, name, np.asarray(baked["layer"]))
            moved = True
    if moved:
        np.savez_compressed(case / document["arrays"], **arrays)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")


#: the maps a baked eye adds (the source's base map stays under its own name)
BASE_KEY, EMISSION_KEY = "EyeBasePass_BaseColor", "EyeBasePass_Emission"


def apply(name: str, record: dict, baked: dict, evidence: dict, case: Path) -> dict:
    """*record* drawn from *baked* (one map's ``base`` and ``emission``): its base map and emission replaced, and
    the ``eye`` statement it carries."""
    folder = case / "eye"
    folder.mkdir(exist_ok=True)
    source = record["base"]
    base_file, emission_file = folder / f"{name}_base.png", folder / f"{name}_emission.png"
    rgba = np.concatenate([_srgb(baked["base"][..., :3]), np.clip(baked["base"][..., 3:4], 0.0, 1.0)], axis=2)
    Image.fromarray(np.round(rgba * 255).astype(np.uint8), "RGBA").save(base_file)
    emission = np.asarray(baked["emission"], np.float64)
    strength = max(1.0, float(emission.max()))
    Image.fromarray(np.round(_srgb(emission / strength) * 255).astype(np.uint8), "RGB").save(emission_file)
    template = {k: v for k, v in record["maps"][source].items() if k not in ("file", "sha256")}
    record["maps"][BASE_KEY] = dict(template, file=base_file.relative_to(case).as_posix(), srgb=True)
    record["maps"][EMISSION_KEY] = dict(template, file=emission_file.relative_to(case).as_posix(), srgb=True,
                                        normal_map=False)
    record["base"] = BASE_KEY
    record["emission"] = {"map": EMISSION_KEY, "color": [1.0, 1.0, 1.0], "strength": strength}
    return {"game": "wuwa", "status": "baked",
            "program": "the material instance's own base pass (its inline shader map), run per texel",
            "source_base": source, "maps": {"emission": EMISSION_KEY},
            "base_color": {"map": BASE_KEY, "lit": True,
                           "basis": "SV_Target3 (GBufferC base colour), lit by the game's deferred toon lighting"},
            "emission": {"map": EMISSION_KEY, "strength": strength, "lit": False,
                         "basis": "SV_Target0 (the scene colour the base pass adds: Unreal's emissive), unlit"},
            "view": "head-on (the view along each texel's normal), game time 0, pre-exposure 1",
            "evidence": evidence}
