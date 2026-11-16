"""Where a Wuthering Waves ``MSM_ToonCommon`` material draws skin, as the game's own compiled shader decides it.

One material draws a character's skin and clothes (``MSM_ToonCommon`` with the static switch ``UseSkinMask``).
Its base pass multiplies the base colour by ``lerp(1, SkinColor, f)``; the texels where the skin factor *f* is 1
are the skin.  What *f* is read from — a mask map's channel or the vertex colour's, against what band — is
compiled into each instance's shader map, so it is not written here as a rule: it is measured on the game's
code (``shaders``), for every base-pass pixel shader of the instance's own shader map:

1. the Material uniform buffer from the instance's parameters (``shader_bake._frozen_values``); the float4 slots
   that move when ``SkinColor`` moves are SkinColor's;
2. in the shader, the lerp ``mad r, f, (SkinColor − 1), 1`` — an ``add`` of a SkinColor slot and −1, then a
   ``mad`` of that by a factor plus 1 — names the factor's register;
3. the shader is run (``shaders.vm``) up to that ``mad`` over lanes that sweep one input at a time over [0, 1]
   — each channel of the ``TypeMask`` map, then each channel of the vertex colour (``COLOR0``), the other one held
   at 0 and at 1 — with every other map a constant (run twice, two constants); the factor's value per lane is
   what the game multiplies by;
4. the input the factor follows is the band's source, the sweep's step is its threshold (the sweep has 0.9 and
   the next float above it, so ``> 0.9`` and ``≥ 0.9`` are told apart), and the factor on either side of it is
   what the game uses there.  A factor that follows no input is the material's constant (``UseSkinMask_DYN`` off:
   ``DefaultSkinFactor``).

Anything this cannot run or read is ``None`` with the reason: the reader is never handed a band made up.
"""
from __future__ import annotations

import re

import numpy as np

from .shaders import archive as A
from .shaders import dxbc, inline, preshader, vm

#: bump when what :func:`analyse` returns changes: the case cache key carries it
VERSION = 6
#: the sweep: an even grid and every 8-bit value (:data:`_ROUND`), and the floats either side of the literals the
#: game's band compares against, so ``> 0.9`` and ``≥ 0.9`` are told apart
_ROUND = np.unique(np.concatenate([np.linspace(0.0, 1.0, 1001), np.arange(256) / 255.0,
                                   [0.05, 0.3, 0.5, 0.9]]).astype(np.float32))
_GRID = np.unique(np.concatenate([_ROUND, [np.nextafter(np.float32(v), np.float32(d)) for v in (0.05, 0.3, 0.5, 0.9)
                                           for d in (0.0, 1.0)]]).astype(np.float32))
#: the material's scalars a reader of the band needs, with what the base pass does with each
PARAMETERS = ("UseSkinMask_DYN", "DefaultSkinFactor", "UseFaceAni_SkinMask")
SWITCHES = ("UseSkinMask", "UseTypeMask")
#: the two constants every map other than the swept one is given; a factor that changes between them is
#: not the band alone
FILLS = (0.25, 0.75)
#: the sweep of every map other than TypeMask together (:func:`_run`)
OTHER = "other maps"
#: how a variant whose factor follows a map other than the band, and cannot be read as a table, is refused
#: (:func:`_variant`)
FOLLOWS_OTHER = "the skin factor follows another of the material's maps"
#: what a lerp's factor can be read from: a register, a constant buffer's scalar, a shader input
FACTOR_KINDS = ("r", "cb", "v")
#: a map's texel as the base pass samples it from the exported 8-bit map: one of an 8-bit channel's 256 values
TEXELS = np.arange(256, dtype=np.float32) / 255.0
#: the most inputs a table is read over (every 8-bit value of each against every one of the other's)
TABLE_INPUTS = 2
#: the random texels a table is found and checked on (:func:`_table`): lanes, the backgrounds a table is read on,
#: and the seed (fixed, so a reading is reproduced)
CHECK_LANES, TABLE_BACKGROUNDS, SEED = 8192, 3, 1502
#: the values each read of the immediate constant buffer is given in turn
ICB = (0.0, 0.777778)
CHANNELS = "RGBA"


class BandError(ValueError):
    pass


class _Captured(Exception):
    pass


class _Probe(vm.Machine):
    """A machine that stops at one instruction and keeps the value of one operand there; a texture given as an
    (n × 4) array is that value per lane wherever it is sampled."""

    def __init__(self, program, n, inputs, buffers, textures, at: int, operand: vm.Operand):
        super().__init__(program, n, inputs, buffers, textures)
        self.at, self.operand, self.value, self.reached = program[at], operand, None, None

    def sample(self, register, uv):
        image = self.textures.get(register)
        if isinstance(image, np.ndarray) and image.ndim == 2:
            return image
        return super().sample(register, uv)

    def _execute(self, ins, active):
        if ins is self.at:
            self.lanes = None
            # a factor read from a constant buffer or a literal is one row: the same on every lane
            self.value = np.broadcast_to(self.f(self.operand)[:, 0], (self.n,)).copy()
            self.reached = active.copy()
            raise _Captured
        super()._execute(ins, active)


def _slots(record: dict, uniforms: inline.Uniforms, size, colour: str) -> set[int]:
    """The Material buffer's float4 slots that move with the vector parameter *colour*."""
    import copy
    from .shader_bake import _frozen_values
    base = _frozen_values(record, uniforms, size)
    moved = copy.deepcopy(record)
    colours = moved.setdefault("native_material", {}).setdefault("Colors", {})
    default = dict(uniforms.vectors).get(colour) or (1.0, 1.0, 1.0, 1.0)
    current = dict(colours.get(colour) or dict(zip(CHANNELS, default)))
    for channel in "RGB":
        current[channel] = float(current[channel]) + 0.123
    colours[colour] = current
    return {int(slot) for slot, _c in np.argwhere(np.abs(_frozen_values(moved, uniforms, size) - base) > 1e-9)}


def _lerp(program: list[vm.Instruction], material: int, slots: set[int]) -> tuple[int, vm.Operand] | None:
    """``(pc, factor)`` of the lerp ``1 + factor·(SkinColor − 1)``: a ``mad r, factor, d, 1`` after
    ``add d, SkinColor, −1`` (either order of the first two), or — where the compiler may not fuse them
    (``[precise]``) — a ``mul r, factor, d`` and then ``add r', r, 1``.  The pc is the instruction that reads the
    factor."""
    def ones(operand, mask, value):
        return operand.kind == "l" and np.allclose(operand.literal.view(np.float32)[list(mask)], value)

    def picked(operand, mask):
        # the components an operand gives under a destination mask (DXBC lines the source's swizzle up with the
        # mask; a short swizzle repeats its last component, as ``vm`` reads it)
        picks = list(operand.swizzle) + [operand.swizzle[-1]] * (4 - len(operand.swizzle))
        return {picks[c] for c in mask}

    def reads_difference(operand, mask, register, written):
        # the difference is the operand of its register whose components are the ones the ``add`` wrote: the
        # factor can sit in the same register (PlayerFemale 10011's ``Up``: ``add r9.yzw, SkinColor, −1`` then
        # ``mad r9.yzw, r9.xxxx, r9.yyzw, l(0, 1, 1, 1)``, IslandLife §7.1502)
        return operand.kind == "r" and operand.index == register and picked(operand, mask) <= set(written)

    def other(ins, register, written):
        a, b = ins.operands[1:3]
        mask = ins.operands[0].mask
        for difference, factor in ((a, b), (b, a)):
            # the factor is a register, or a scalar of a constant buffer read straight into the lerp (Shouanren
            # 10011's Up02: ``mad r5.xyz, cb4[85].zzzz, r5.xyzx, l(1, 1, 1, 0)``, the material's constant factor,
            # IslandLife §7.1502)
            if (reads_difference(difference, mask, register, written) and factor.kind in FACTOR_KINDS
                    and not reads_difference(factor, mask, register, written)):
                return factor
        return None

    for pc, ins in enumerate(program):
        if ins.op != "add" or ins.operands[0].kind != "r":
            continue
        sources = ins.operands[1:]
        if not any(s.kind == "cb" and s.index == material and s.slot in slots for s in sources):
            continue
        # a literal's components line up with the destination's mask (``add r6.xzw, …, l(-1, 0, -1, -1)``)
        if not any(ones(s, ins.operands[0].mask, -1.0) for s in sources):
            continue
        difference, written = ins.operands[0].index, ins.operands[0].mask
        for later in range(pc + 1, min(pc + 12, len(program))):
            nxt = program[later]
            if nxt.op == "mad" and ones(nxt.operands[3], nxt.operands[0].mask, 1.0):
                factor = other(nxt, difference, written)
                if factor is not None:
                    return later, factor
            if nxt.op == "mul":
                factor = other(nxt, difference, written)
                product = nxt.operands[0]
                for after in range(later + 1, min(later + 4, len(program))):
                    then = program[after]
                    if (factor is not None and then.op == "add"
                            and any(o.kind == "r" and o.index == product.index for o in then.operands[1:])
                            and any(ones(o, then.operands[0].mask, 1.0) for o in then.operands[1:])):
                        return later, factor
            if (nxt.operands and nxt.operands[0].kind == "r" and nxt.operands[0].index == difference
                    and set(nxt.operands[0].mask) & set(written)):
                break                                       # the difference is overwritten before any lerp
    return None


def _step(values: np.ndarray, factor: np.ndarray) -> dict:
    """The factor along a sweep as a band: where it steps, what it is either side."""
    levels = np.unique(np.round(factor, 6))
    if len(levels) == 1:
        return {"follows": False, "factor": float(levels[0])}
    order = np.argsort(values)
    values, factor = values[order], factor[order]
    changes = np.flatnonzero(np.abs(np.diff(factor)) > 1e-6)
    out = {"follows": True, "levels": [float(v) for v in levels]}
    if len(changes) == 1:
        k = int(changes[0])
        # the float32 the shader compares, written as its shortest decimal (0.9, not 0.8999999761581421)
        below, above = float(str(np.float32(values[k]))), float(str(np.float32(values[k + 1])))
        out.update(factor_below=float(factor[k]), factor_above=float(factor[k + 1]))
        # two neighbouring floats: the step is "above the lower" or "at or above the upper", whichever of them is
        # the round value (0.9, not the float just under it)
        neighbours = np.float32(above) == np.nextafter(np.float32(below), np.float32(2.0))
        if neighbours and np.float32(below) in _ROUND:
            out.update(comparison=">", threshold=below)
        elif neighbours and np.float32(above) in _ROUND:
            out.update(comparison=">=", threshold=above)
        else:
            out.update(comparison="between", threshold=[below, above])
    else:
        out["steps"] = [[float(values[k]), float(values[k + 1]), float(factor[k]), float(factor[k + 1])]
                        for k in changes[:16]]
    return out


def _run(shader: dxbc.Shader, program, at: int, factor: vm.Operand, buffers: dict, typemask: int | None,
         fill: float) -> tuple[dict, np.ndarray]:
    """One run over every sweep; ``({(source, channel, background): factor along the grid}, reached)``."""
    grid = _GRID
    n_grid = len(grid)
    sweeps = [(source, channel, background) for source in ("TypeMask", "COLOR0") for channel in range(4)
              for background in (0.0, 1.0) if source == "COLOR0" or typemask is not None]
    # every other map at once, each channel along the grid, with the band's own inputs held at 0 and at 1: a factor
    # that moves here follows a map that is not the band (an ID map decoded to integers and compared with
    # SkinMin / SkinMax, IslandLife §7.1492), which the two constant fills cannot show
    sweeps += [(OTHER, channel, background) for channel in range(4) for background in (0.0, 1.0)]
    n = n_grid * len(sweeps)
    lanes_mask = np.zeros((n, 4), np.float32)
    lanes_colour = np.zeros((n, 4), np.float32)
    lanes_other = np.full((n, 4), fill, np.float32)
    for k, (source, channel, background) in enumerate(sweeps):
        rows = slice(k * n_grid, (k + 1) * n_grid)
        if source == OTHER:
            lanes_mask[rows] = background
            lanes_colour[rows] = background
            lanes_other[rows, channel] = grid
            continue
        held, swept = (lanes_colour, lanes_mask) if source == "TypeMask" else (lanes_mask, lanes_colour)
        held[rows] = background
        swept[rows] = background
        swept[rows, channel] = grid
    inputs = {}
    colour_register = None
    for register, semantics in shader.inputs.items():
        inputs[register] = np.zeros((n, 4), np.float32)
        if any(semantic == "COLOR" and index == 0 for semantic, index, _mask in semantics):
            colour_register = register
            inputs[register] = lanes_colour
    textures = {}
    for register in shader.textures:
        textures[register] = lanes_other
    if typemask is not None:
        textures[typemask] = lanes_mask
    machine = _Probe(program, n, inputs, buffers, textures, at, factor)
    try:
        machine.run()
    except _Captured:
        pass
    if machine.value is None:
        raise BandError("the run ended before the SkinColor lerp")
    results = {}
    for k, key in enumerate(sweeps):
        if key[0] == "COLOR0" and colour_register is None:
            continue
        results[key] = machine.value[k * n_grid:(k + 1) * n_grid]
    return results, machine.reached


def _variant(shader: dxbc.Shader, buffers: dict, material: int, slots: set[int], typemask: int | None,
             textures: list[str] = ()) -> dict:
    """What one base-pass pixel shader's skin factor follows; *textures*: the material's texture parameters in the
    order of its uniform buffer's resources."""
    # the immediate constant buffer (a dither table) is not run here: its declaration is dropped and each read of
    # it given a constant, twice (two constants); a factor that differs between the two is refused below
    asm = re.sub(r"dcl_immediateConstantBuffer \{.*?\}\s*\}", "", shader.asm, flags=re.S)
    # ``[precise(xyz)]`` only forbids the compiler's reassociation of the result; the value is the same
    asm = re.sub(r" ?\[precise(?:\([xyzw]+\))?\]", "", asm)
    # a conservative depth output is a depth output: not a value the band reads
    asm = re.sub(r"oDepth(?:LE|GE)", "oDepth", asm)
    programs = [vm.parse(re.sub(r"icb\[[^\]]*\](\.[xyzw]+)?", f"l({value})", asm)) for value in ICB]
    program = programs[0]
    if not any(ins.op != "dcl" and any(o.kind == "cb" and o.index == material and o.slot in slots
                                       for o in ins.operands) for ins in program):
        return {"source": None, "skin_color_read": False}
    found = _lerp(program, material, slots)
    if found is None:
        raise BandError("SkinColor is read but not as a lerp from 1 (an unread form)")
    at, factor = found
    runs = [_run(shader, program, at, factor, buffers, typemask, fill) for fill in FILLS]
    second = _lerp(programs[1], material, slots)
    if second is None or second[0] != at:
        raise BandError("the SkinColor lerp moves when the dither table is given another value")
    runs.append(_run(shader, programs[1], at, second[1], buffers, typemask, FILLS[0]))
    (first, reached), *others = runs
    if not reached.all():
        raise BandError("some lanes do not reach the SkinColor lerp (it is under a branch)")
    if (any(not np.array_equal(first[key], other[key]) for other, _r in others for key in first)
            or any(source == OTHER and _step(_GRID, values)["follows"] for (source, _c, _b), values in first.items())):
        # the factor follows a map other than the band, or changes with the constants the other maps are given: an
        # ID map decoded to integers and compared with SkinMin / SkinMax (PlayerFemale 10011's ``Up``: RegionIDTex
        # R, its nibble chosen by TypeMask A, IslandLife §7.1502).  Read it as a table over the maps it follows
        try:
            return _table(shader, programs, [(at, factor), second], buffers, material, textures)
        except BandError as failure:
            raise BandError(f"{FOLLOWS_OTHER}, not read as a table: {failure}") from None
    first = {key: values for key, values in first.items() if key[0] != OTHER}
    followed = {}
    for (source, channel, background), factor_values in first.items():
        step = _step(_GRID, factor_values)
        if step["follows"]:
            followed.setdefault((source, channel), []).append((background, step))
    if not followed:
        constants = {float(np.round(v[0], 6)) for v in first.values()}
        if len(constants) != 1:
            raise BandError("the skin factor is constant along each sweep but differs between them")
        return {"source": None, "constant_factor": constants.pop(), "lerp_instruction": at}
    if len(followed) != 1:
        raise BandError("the skin factor follows more than one input: "
                        + ", ".join(f"{s} {CHANNELS[c]}" for s, c in followed))
    (source, channel), steps = next(iter(followed.items()))
    shapes = {repr({k: v for k, v in step.items()}) for _b, step in steps}
    if len(steps) != 2 or len(shapes) != 1:
        raise BandError(f"the skin factor's band on {source} {CHANNELS[channel]} depends on the other input")
    step = steps[0][1]
    out = {"source": source, "channel": CHANNELS[channel], "lerp_instruction": at}
    out.update({k: v for k, v in step.items() if k != "follows"})
    if source == "TypeMask":
        out["map"] = "TypeMask"                        # the material's map of that name (the sidecar's maps)
        out.update(_uv_set(shader, program, typemask))
    else:
        out["in_fbx"] = "the part's vertex colours (the FBX's LayerElementColor)"
    return out


def _names(shader: dxbc.Shader, material: int, textures: list[str]) -> dict[int, str]:
    """Each texture register's name: the material's texture parameter it is (a uniform buffer's resources are its
    textures, two entries each), or what else it is."""
    out = {}
    for register, (buffer, resource) in shader.textures.items():
        if buffer == material and resource % 2 == 0 and resource // 2 < len(textures):
            out[register] = textures[resource // 2]
        else:
            out[register] = f"t{register} (buffer {buffer} resource {resource}), {NOT_A_MAP}"
    return out


#: how a texture register that is not one of the material's maps is named (:func:`_names`)
NOT_A_MAP = "not a map of the material"


def _texel_run(shader: dxbc.Shader, program, at: int, factor: vm.Operand, buffers: dict,
               candidates: list[tuple], values: np.ndarray) -> np.ndarray:
    """The factor per lane where each candidate input ``(texture register | "COLOR0", channel)`` takes its column of
    *values* (8-bit, lanes × candidates); every lane must reach the lerp."""
    n = len(values)
    textures = {register: np.zeros((n, 4), np.float32) for register in shader.textures}
    colour = np.zeros((n, 4), np.float32)
    for k, (source, channel) in enumerate(candidates):
        (colour if source == "COLOR0" else textures[source])[:, channel] = TEXELS[values[:, k]]
    inputs = {}
    for register, semantics in shader.inputs.items():
        is_colour = any(semantic == "COLOR" and index == 0 for semantic, index, _mask in semantics)
        inputs[register] = colour if is_colour else np.zeros((n, 4), np.float32)
    machine = _Probe(program, n, inputs, buffers, textures, at, factor)
    try:
        machine.run()
    except _Captured:
        pass
    if machine.value is None:
        raise BandError("the run ended before the SkinColor lerp")
    if not machine.reached.all():
        raise BandError("some lanes do not reach the SkinColor lerp (it is under a branch)")
    return np.round(machine.value.astype(np.float64), 6)


def _boxes(table: np.ndarray) -> tuple[float, list[dict]]:
    """A factor table over one or two 8-bit inputs as the value most of it takes (``otherwise``) and boxes — per
    input an inclusive range of 8-bit values — where it takes another; the fewer boxes of the two input orders."""
    levels, counts = np.unique(table, return_counts=True)
    otherwise = float(levels[np.argmax(counts)])

    def runs(rows):
        out, start = [], 0
        for k in range(1, len(rows) + 1):
            if k == len(rows) or not np.array_equal(rows[k], rows[start]):
                out.append((start, k - 1, rows[start]))
                start = k
        return out

    def flat(grid, swapped):
        boxes = []
        if grid.ndim == 1:
            for lo, hi, value in runs(grid):
                if float(value) != otherwise:
                    boxes.append({"factor": float(value), "ranges": [[int(lo), int(hi)]]})
            return boxes
        for lo, hi, row in runs(list(grid)):
            for lo2, hi2, value in runs(row):
                if float(value) != otherwise:
                    ranges = [[int(lo), int(hi)], [int(lo2), int(hi2)]]
                    boxes.append({"factor": float(value), "ranges": ranges[::-1] if swapped else ranges})
        return boxes

    readings = [flat(table, False)] + ([flat(table.T, True)] if table.ndim == 2 else [])
    return otherwise, min(readings, key=len)


def _table(shader: dxbc.Shader, programs, lerps, buffers: dict, material: int, textures: list[str]) -> dict:
    """The skin factor as a table over the maps it follows: ``{"source": "maps", "inputs": […], "otherwise", "boxes"}``.

    Every texture register's every channel and the vertex colour's are inputs, given random 8-bit texels
    (:data:`CHECK_LANES` lanes, :data:`SEED`); an input the factor follows is one whose new random values change it
    on some lane.  The factor is then run over every 8-bit value of each followed input against every one of the
    other's (at most :data:`TABLE_INPUTS`), on :data:`TABLE_BACKGROUNDS` random values of the rest and with both
    dither-table constants; the table must be the same on all of them, and must give the factor of every random lane.
    A followed input that is not a map of the material (the vertex colour, a texture of another buffer) is refused:
    a reader has no map to look it up in."""
    names = _names(shader, material, textures)
    (at, factor), second = lerps
    if second is None:
        raise BandError("the SkinColor lerp moves when the dither table is given another value")
    colour = any(semantic == "COLOR" and index == 0
                 for semantics in shader.inputs.values() for semantic, index, _mask in semantics)
    candidates = [(register, channel) for register in sorted(shader.textures) for channel in range(4)]
    candidates += [("COLOR0", channel) for channel in range(4)] if colour else []
    rng = np.random.default_rng(SEED)
    base = rng.integers(0, 256, (CHECK_LANES, len(candidates)))
    blocks = [base]
    for k in range(len(candidates)):
        moved = base.copy()
        moved[:, k] = (base[:, k] + rng.integers(1, 256, CHECK_LANES)) % 256    # another value on every lane
        blocks.append(moved)
    factors = _texel_run(shader, programs[0], at, factor, buffers, candidates, np.concatenate(blocks))
    reference = factors[:CHECK_LANES]
    followed = [k for k in range(len(candidates))
                if not np.array_equal(factors[(k + 1) * CHECK_LANES:(k + 2) * CHECK_LANES], reference)]

    def named(k):
        source, channel = candidates[k]
        return ("the vertex colour COLOR0" if source == "COLOR0" else names[source]) + " " + CHANNELS[channel]

    if not followed:
        raise BandError("the skin factor changes with the constants the maps are given but follows no texel")
    if len(followed) > TABLE_INPUTS:
        raise BandError(f"the skin factor follows {len(followed)} inputs: " + ", ".join(named(k) for k in followed))
    foreign = [named(k) for k in followed if candidates[k][0] == "COLOR0" or NOT_A_MAP in names[candidates[k][0]]]
    if foreign:
        raise BandError("the skin factor follows an input that is not a map of the material: " + ", ".join(foreign))
    grid = np.stack(np.meshgrid(*[np.arange(256)] * len(followed), indexing="ij"), -1).reshape(-1, len(followed))
    tables = []
    for background in rng.integers(0, 256, (TABLE_BACKGROUNDS, len(candidates))):
        lanes = np.repeat(background[None, :], len(grid), axis=0)
        lanes[:, followed] = grid
        for program, (pc, operand) in ((programs[0], (at, factor)), (programs[1], second)):
            tables.append(_texel_run(shader, program, pc, operand, buffers, candidates, lanes))
    if any(not np.array_equal(table, tables[0]) for table in tables[1:]):
        raise BandError("the skin factor's table over " + ", ".join(named(k) for k in followed)
                        + " changes with the other inputs or the dither table")
    table = tables[0].reshape([256] * len(followed))
    if not np.isfinite(table).all():
        raise BandError("the skin factor's table has a value that is not finite")
    if not np.array_equal(table[tuple(base[:, followed].T)], reference):
        raise BandError("the table over " + ", ".join(named(k) for k in followed)
                        + " does not give the factor of the random texels")
    otherwise, boxes = _boxes(table)
    inputs = [{"map": names[candidates[k][0]], "channel": CHANNELS[candidates[k][1]],
               **_uv_set(shader, programs[0], candidates[k][0])} for k in followed]
    return {"source": "maps", "inputs": inputs, "values": "each input's texel as an 8-bit value (0-255)",
            "levels": sorted({float(v) for v in np.unique(table)}), "otherwise": otherwise, "boxes": boxes,
            "checked_lanes": CHECK_LANES, "lerp_instruction": at}


def _uv_set(shader: dxbc.Shader, program, register: int) -> dict:
    """Which of the mesh's UV sets the map at texture *register* is sampled at, when it is one directly."""
    coordinates = {(ins.operands[1].kind, ins.operands[1].index, ins.operands[1].swizzle[0])
                   for ins in program if ins.op.startswith("sample") and ins.operands[2].index == register}
    if len(coordinates) == 1:
        kind, index, component = next(iter(coordinates))
        for semantic, number, mask in shader.inputs.get(index, []) if kind == "v" else []:
            if semantic == "TEXCOORD" and "xyzw"[component] in mask:
                # Unreal packs two UV sets in one interpolant (xy, zw) when TEXCOORD0 is a float4
                packed = any(s == "TEXCOORD" and i == 0 and m == "xyzw"
                             for semantics in shader.inputs.values() for s, i, m in semantics)
                return {"uv_set": 2 * number + (component >= 2) if packed else number}
    return {"uv_set": None, "uv_set_reason": "the map is sampled at a coordinate the shader works out"}


def base_passes(store: A.Archive, map_index: int) -> list[tuple[int, dxbc.Shader]]:
    """The map's base-pass pixel shaders: they read the Material buffer and write the seven GBuffer targets."""
    out = []
    for index in store.shaders(map_index):
        if store.frequency(index) != A.PIXEL:
            continue
        shader = dxbc.read(store.code(index))
        if "Material" in shader.buffers and len(re.findall(r"dcl_output o\d", shader.asm)) >= 7:
            out.append((index, shader))
    return out


def shading_passes(store: A.Archive, map_index: int) -> list[tuple[int, dxbc.Shader]]:
    """Every pixel shader of the map that reads the Material buffer (a translucent map's colour passes among them)."""
    out = []
    for index in store.shaders(map_index):
        if store.frequency(index) == A.PIXEL:
            shader = dxbc.read(store.code(index))
            if "Material" in shader.buffers:
                out.append((index, shader))
    return out


def parameters(record: dict, uniforms: inline.Uniforms | None) -> dict:
    """The switches and scalars that steer the band: each value and where it comes from."""
    native = record.get("native_material") or {}
    switches, scalars = native.get("Switches") or {}, native.get("Scalars") or {}
    defaults = dict(uniforms.scalars) if uniforms is not None else {}
    out = {}
    for name in SWITCHES:
        # a switch the instance does not set is its parent's, compiled into the shader map; the band read off that
        # map is what it does (IslandLife §7.1492: Kanteleila 10011's Up sets no UseSkinMask and reads the band)
        out[name] = ({"value": bool(switches[name]), "from": "the instance's static switch"} if name in switches else
                     {"value": None, "from": "not set on the instance: its parent's value, compiled into the shader "
                                             "map (the band read off it is what it does)"})
    for name in PARAMETERS:
        if name in scalars:
            out[name] = {"value": float(scalars[name]), "from": "the instance's scalar"}
        elif name in defaults:
            out[name] = {"value": float(defaults[name]), "from": "the shader map's default (the instance does not set it)"}
        else:
            out[name] = {"value": None, "from": "not a parameter of this shader map"}
    return out


def analyse(record: dict, store: A.Archive, uexp: bytes, size, collections: list[dict]) -> dict:
    """The skin band of the material *record* (its instance's ``.uexp`` is *uexp*): see the module."""
    from .shader_bake import FRAME, _collection, _frozen_values, _sizes, collections_in_map
    resources = inline.read(uexp)
    if not resources:
        switches = (record.get("native_material") or {}).get("Switches") or {}
        if not switches:
            # an instance with no static switch of its own has no static permutation: it draws with its parent's
            # shader map, which is not in its package (Calbrena 10011's MI_Eye_OL over M_Empty, §7.1502)
            raise BandError("the instance sets no static switch, so it draws with its parent material's shader map, "
                            "which is not in its package and is not read here")
        # Xuanling 10011's Down sets 25 switches and carries a 436 KB package, but no frozen resource is found in it
        raise BandError("the instance sets static switches, but no inline shader map is read out of its package")
    uniforms = resources[0]
    digests = [uexp[i:i + 20] for i in range(len(uexp) - 19) if store.map_index(uexp[i:i + 20]) is not None]
    if not digests:
        raise BandError("the instance's shader map is not in the SM5 archive")
    map_index = store.map_index(digests[0])
    if "SkinColor" not in dict(uniforms.vectors):
        # no code of the map can lerp toward a colour it has no parameter for: it draws no skin (a translucent
        # ``Glass`` of Zhezhi 10011: six pixel shaders, none a seven-target base pass, no SkinColor; §7.1492)
        return {"status": "no_skin_color", "source": None, "skin_color_read": False,
                "reason": "SkinColor is not a parameter of the instance's shader map",
                "parameters": parameters(record, uniforms), "evidence": {"shader_map_sha1": digests[0].hex()}}
    slots = _slots(record, uniforms, size, "SkinColor")
    if not slots:
        raise BandError("no slot of the Material buffer follows SkinColor")
    material_values = _frozen_values(record, uniforms, size)
    typemask_index = next((i for i, (name, _ref) in enumerate(uniforms.textures) if name == "TypeMask"), None)
    variants, failures = [], []
    passes = base_passes(store, map_index)
    # a translucent material has no GBuffer pass: its colour is shaded in passes of fewer targets, among the
    # map's other Material-reading pixel shaders (depth, velocity …); the band is what every one of them that
    # tints toward SkinColor does, and a map none of whose passes does draws no skin (§7.1492: Linnaea 10011's
    # ``Alpha``, Mone 10011's ``Down``)
    translucent = not passes
    if translucent:
        passes = shading_passes(store, map_index)
    for index, shader in passes:
        material = shader.buffers["Material"]
        sizes = _sizes(shader.asm)
        buffers = {register: np.zeros((sizes.get(register, 1), 4)) for register in sizes}
        for buffer_name, values in FRAME.items():
            register = shader.buffers.get(buffer_name)
            if register is not None:
                for slot, value in values.items():
                    if slot < len(buffers[register]):
                        buffers[register][slot] = value
        buffers[material][:min(len(material_values), len(buffers[material]))] = material_values[:len(buffers[material])]
        for number, collection in enumerate(collections_in_map(uexp, collections)):
            target = shader.buffers.get(f"MaterialCollection{number}")
            if target is not None:
                values = _collection(collection)
                buffers[target][:min(len(values), len(buffers[target]))] = values[:len(buffers[target])]
        typemask = next((register for register, (buffer, resource) in shader.textures.items()
                         if buffer == material and typemask_index is not None and resource == 2 * typemask_index),
                        None)
        try:
            variants.append((index, _variant(shader, buffers, material, slots, typemask,
                                             [name for name, _ref in uniforms.textures])))
        except (BandError, vm.ShaderError, preshader.PreshaderError) as failure:
            failures.append((index, str(failure)))
    if not variants:
        reasons = sorted({reason for _i, reason in failures}) or ["the shader map has no base-pass pixel shader"]
        raise BandError("; ".join(reasons))
    # a variant whose factor follows another map and cannot be read as a table (:func:`_table`) is not unread: it
    # draws its skin by that map, so what the other variants read is not the material's band (§7.1492)
    elsewhere = sorted({reason for _i, reason in failures if reason.startswith(FOLLOWS_OTHER)})
    if elsewhere:
        raise BandError("; ".join(elsewhere))
    if translucent:
        tinting = [(i, r) for i, r in variants if r.get("skin_color_read") is not False]
        if not tinting and not failures:
            return {"status": "no_skin_color", "source": None, "skin_color_read": False,
                    "reason": "no pass of the translucent shader map tints toward SkinColor",
                    "parameters": parameters(record, uniforms),
                    "evidence": {"shader_map_sha1": digests[0].hex(), "translucent_passes": [i for i, _r in variants],
                                 "skin_color_slots": sorted(slots)}}
        if not tinting:
            raise BandError("; ".join(sorted({reason for _i, reason in failures})))
        variants = tinting
    keyed = {}
    for index, result in variants:
        key = repr({k: v for k, v in result.items() if k != "lerp_instruction"})
        keyed.setdefault(key, []).append(index)
    evidence = {"shader_map_sha1": digests[0].hex(),
                ("translucent_tinting_pixel_shaders" if translucent else "base_pass_pixel_shaders"):
                    [i for i, _r in variants],
                "skin_color_slots": sorted(slots),
                "unread_pixel_shaders": [{"shader": i, "reason": r} for i, r in failures]}
    if len(keyed) != 1:
        return {"status": "variants_differ", "variants": [dict(r, pixel_shader=i) for i, r in variants],
                "parameters": parameters(record, uniforms), "evidence": evidence}
    result = dict(variants[0][1])
    result.pop("lerp_instruction", None)
    status = ("table" if result.get("source") == "maps" else "read" if result.get("source")
              else "no_skin_color" if result.get("skin_color_read") is False else "constant")
    return {"status": status, **result, "parameters": parameters(record, uniforms), "evidence": evidence}


#: The shading models whose base pass reads a skin factor: the clothing-and-skin materials, and the face's and
#: the hair's too (Chun 10011's ``Face`` reads it off its vertex colour, Zanni 20011's and Katixiya 10011's off
#: ``TypeMask``, their ``Hair`` as the constant ``DefaultSkinFactor``).  The same factor lerps the GBufferB colour
#: the toon pass's SSS term reads (``SubsurfaceColor`` toward ``SkinSubsurfaceColor``, IslandLife §7.1525).
TOON_MODELS = ("MSM_ToonCommon", "MSM_ToonFace", "MSM_ToonHair")


def is_target(record: dict, slot_role) -> bool:
    """Whether the band is read for *record*: a material of :data:`TOON_MODELS` that is not the tacet mark.

    Not only the ``skin_and_clothing`` ones (the instance sets ``UseSkinMask``): a static switch the instance does
    not set is its parent's, compiled into its shader map, so what one without it draws is read off that map too
    (``interchange.sidecar.slot_role`` takes an ``MSM_ToonCommon`` role from the result; IslandLife §7.1492).
    A face's or a hair's band says where its SSS colour is the skin's (IslandLife §7.1525); a shader map with no
    ``SkinColor`` says so (``no_skin_color``)."""
    native = record.get("native_material") or {}
    model = ((native.get("Properties") or {}).get("BasePropertyOverrides") or {}).get("ShadingModel")
    return model in TOON_MODELS and slot_role(record)[0] != "effect"


def annotate_case(case, files, log=print) -> None:
    """Every toon material of the scene in *case* (:func:`is_target`) given its ``skin_band``
    (:func:`analyse`), or
    ``{"status": None, "reason": …}`` when the game's shader cannot be read for it; *files* is the shader bake's
    (``shader_bake.bake_case``).  Results are kept by the material's JSON hash in ``files.cache``'s sibling
    ``skin_band``.  The scene document is rewritten."""
    import hashlib
    import json
    from pathlib import Path

    from PIL import Image

    from ..interchange.sidecar import slot_role
    case = Path(case)
    path = case / "source_document.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    targets = [name for name, record in document["materials"].items()
               if not record.get("unused") and is_target(record, slot_role)]
    if not targets:
        return
    folder = Path(files.cache).parent / "skin_band"
    store = None
    for name in targets:
        record = document["materials"][name]
        key = hashlib.sha256(f"{VERSION}|{record.get('source_json_sha256', '')}|{record.get('source_package')}"
                             .encode()).hexdigest()[:24]
        cached = folder / f"{key}.json"
        if cached.is_file():
            record["skin_band"] = json.loads(cached.read_text(encoding="utf-8"))
            continue
        try:
            if store is None:
                store = A.read(files.archive())
            root, collections = files.root_of(record["source_package"])
            referenced = [Path(t.get("ObjectPath", "")).stem.split(".")[0] if isinstance(t, dict) else None
                          for t in root.get("ReferencedTextures", [])]

            def size(reference: int, maps=record["maps"], referenced=referenced):
                named = referenced[reference] if 0 <= reference < len(referenced) else None
                if named not in maps:
                    raise BandError(f"the material's maps have no texture {reference} ({named})")
                with Image.open(case / maps[named]["file"]) as opened:
                    return opened.size

            result = analyse(record, store, files.package(record["source_package"]), size, collections)
        except (BandError, vm.ShaderError, preshader.PreshaderError, OSError, KeyError, ValueError,
                IndexError) as failure:
            result = {"status": None, "reason": f"the game's shader could not be read for the skin band: {failure}"}
        log(f"{name}：皮膚帶 {result.get('status')}"
            + (f"（{result.get('source')} {result.get('channel')} {result.get('comparison')} {result.get('threshold')}）"
               if result.get("source") else ""))
        folder.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        record["skin_band"] = result
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
