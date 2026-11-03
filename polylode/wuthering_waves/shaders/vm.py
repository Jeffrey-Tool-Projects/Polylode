"""Run a pixel shader's assembly (``ps_5_0``, as ``d3dcompiler_47`` writes it) over many points at once.

Each point is a lane; a register is four 32-bit words per lane, kept as raw bits so the shader's integer and
bit operations (a comparison's 0xFFFFFFFF ``and`` 1.0's bits) mean what they mean on a GPU.  Branches and
loops run every lane together under an execution mask; ``discard`` marks a lane and takes it out.  What a
point cannot answer — screen-space derivatives, a mip level — is taken at the full-size image: a sample reads
level 0, bilinearly, wrapping, and a derivative is 0.

The instructions the character materials' passes use are handled; any other stops the run
(``ShaderError``), because a shader run with an instruction skipped is a coverage made up.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

COMPONENTS = "xyzw"


class ShaderError(ValueError):
    pass


@dataclass
class Operand:
    kind: str            # r v o cb l t s null oDepth
    index: int = 0
    slot: int = 0        # a constant buffer's float4
    swizzle: tuple = (0, 1, 2, 3)
    mask: tuple = (0, 1, 2, 3)
    negate: bool = False
    absolute: bool = False
    literal: np.ndarray | None = None


_OPERAND = re.compile(r"^(-)?(\|)?(r|v|o|t|s|cb|icb|x)(\d*)(?:\[(\d+)\])?(?:\.([xyzw]+))?\|?$")


def _literal(text: str) -> np.ndarray:
    words = []
    for value in [v.strip() for v in text.split(",")]:
        if value.lower().startswith("0x"):
            words.append(int(value, 16) & 0xFFFFFFFF)
        elif re.fullmatch(r"-?\d+", value):
            words.append(int(value) & 0xFFFFFFFF)
        else:
            words.append(int(np.float32(float(value)).view(np.uint32)))
    while len(words) < 4:
        words.append(words[-1])
    return np.array(words, dtype=np.uint32)


def _operand(text: str) -> Operand:
    text = text.strip()
    if text in ("null", "oDepth"):
        return Operand(text)
    negate = text.startswith("-")
    if text.startswith("-l(") or text.startswith("l("):
        return Operand("l", literal=_literal(text[text.index("(") + 1:text.rindex(")")]), negate=negate)
    match = _OPERAND.match(text)
    if match is None:
        raise ShaderError(f"an operand not read here: {text!r}")
    neg, absolute, kind, index, slot, components = match.groups()
    picks = tuple(COMPONENTS.index(c) for c in components) if components else (0, 1, 2, 3)
    return Operand(kind, int(index or 0), int(slot or 0), swizzle=picks, mask=picks, negate=bool(neg),
                   absolute=bool(absolute))


def _split(arguments: str) -> list[str]:
    parts, depth, current = [], 0, ""
    for c in arguments:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if c == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += c
    if current.strip():
        parts.append(current)
    return [p.strip() for p in parts]


@dataclass
class Instruction:
    op: str
    saturate: bool
    operands: list[Operand]
    resource_swizzle: tuple = (0, 1, 2, 3)


def parse(asm: str) -> list[Instruction]:
    body = asm.split("ps_5_0", 1)[1] if "ps_5_0" in asm else asm
    program = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("dcl_"):
            continue
        head, _, rest = line.partition(" ")
        head = re.sub(r"\(.*?\)", "", head)               # sample_indexable(texture2d)(float…) → sample_indexable
        rest = re.sub(r"^\([^)]*\)\s*", "", rest.strip()) if rest.strip().startswith("(") else rest
        saturate = head.endswith("_sat")
        op = head[:-4] if saturate else head
        operands, swizzle = [], (0, 1, 2, 3)
        for part in _split(rest):
            operand = _operand(part)
            if operand.kind == "t":
                swizzle = operand.swizzle
            operands.append(operand)
        program.append(Instruction(op, saturate, operands, swizzle))
    return program


class Machine:
    """One shader's program over *n* lanes: inputs, constant buffers and textures given, ``run`` → discarded."""

    def __init__(self, program: list[Instruction], n: int, inputs: dict[int, np.ndarray],
                 buffers: dict[int, np.ndarray], textures: dict[int, np.ndarray]):
        self.program, self.n = program, n
        self.inputs = {k: np.ascontiguousarray(v, dtype=np.float32).view(np.uint32) for k, v in inputs.items()}
        self.buffers = {k: np.ascontiguousarray(v, dtype=np.float32).view(np.uint32) for k, v in buffers.items()}
        self.textures = textures
        self.registers: dict[int, np.ndarray] = {}
        self.discarded = np.zeros(n, bool)

    # -- operands -----------------------------------------------------------------------------------------
    def _raw(self, operand: Operand) -> np.ndarray:
        """The operand's four words per lane (n × 4, uint32), swizzled."""
        # a literal or a buffer's float4 stays one row (1 × 4): numpy broadcasts it against the lanes
        if operand.kind == "l":
            words = operand.literal[None, :]
        elif operand.kind == "r":
            words = self.registers.setdefault(operand.index, np.zeros((self.n, 4), np.uint32))
        elif operand.kind == "v":
            words = self.inputs.get(operand.index)
            if words is None:
                raise ShaderError(f"input v{operand.index} not given")
        elif operand.kind == "cb":
            buffer = self.buffers.get(operand.index)
            if buffer is None or operand.slot >= buffer.shape[-2]:
                raise ShaderError(f"cb{operand.index}[{operand.slot}] not given")
            # a buffer is (slots × 4), or (lanes × slots × 4) where each point has values of its own
            words = buffer[:, operand.slot] if buffer.ndim == 3 else buffer[operand.slot][None, :]
        else:
            raise ShaderError(f"a {operand.kind} operand read as a value")
        picks = list(operand.swizzle) + [operand.swizzle[-1]] * (4 - len(operand.swizzle))
        return words[:, picks]

    def f(self, operand: Operand) -> np.ndarray:
        value = self._raw(operand).view(np.float32)
        if operand.absolute:
            value = np.abs(value)
        if operand.negate:
            value = -value
        return value

    def i(self, operand: Operand) -> np.ndarray:
        value = self._raw(operand).view(np.int32).astype(np.int64)
        if operand.negate:
            value = -value
        return value

    def u(self, operand: Operand) -> np.ndarray:
        return self._raw(operand).astype(np.uint64)

    def write(self, operand: Operand, value: np.ndarray, active: np.ndarray, saturate: bool = False,
              kind: str = "f") -> None:
        if operand.kind in ("null", "oDepth", "o"):
            return
        if operand.kind != "r":
            raise ShaderError(f"a write to {operand.kind}")
        if kind == "f":
            value = np.asarray(value, np.float32)
            if saturate:
                value = np.clip(np.nan_to_num(value, nan=0.0), 0.0, 1.0)
            words = value.view(np.uint32)
        else:
            words = (np.asarray(value).astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
        target = self.registers.setdefault(operand.index, np.zeros((self.n, 4), np.uint32))
        words = np.broadcast_to(words, (self.n, 4))
        everyone = active.all()
        for c in operand.mask:
            if everyone:
                target[:, c] = words[:, c]
            else:
                np.copyto(target[:, c], words[:, c], where=active)

    # -- sampling -----------------------------------------------------------------------------------------
    def sample(self, register: int, uv: np.ndarray) -> np.ndarray:
        image = self.textures.get(register)
        if image is None:
            raise ShaderError(f"texture t{register} not given")
        h, w = image.shape[:2]
        x, y = uv[:, 0] * w - 0.5, uv[:, 1] * h - 0.5
        x, y = np.nan_to_num(x), np.nan_to_num(y)
        x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        pick = lambda xx, yy: image[np.mod(yy, h), np.mod(xx, w)]
        return ((pick(x0, y0) * (1 - fx) + pick(x0 + 1, y0) * fx) * (1 - fy)
                + (pick(x0, y0 + 1) * (1 - fx) + pick(x0 + 1, y0 + 1) * fx) * fy).astype(np.float32)

    # -- the run --------------------------------------------------------------------------------------------
    def run(self, limit: int = 1024) -> np.ndarray:
        """The lanes the shader discards."""
        program, n = self.program, self.n
        ends = _structure(program)
        active = np.ones(n, bool)
        stack: list[dict] = []                         # if: saved, cond; loop: saved, start, broken, count
        pc = 0

        def alive(mask):
            mask = mask & ~self.discarded
            for frame in stack:
                if frame["kind"] == "loop":
                    mask = mask & ~frame["broken"]
            return mask

        while pc < len(program):
            ins = program[pc]
            op, args = ins.op, ins.operands
            if op in ("if_nz", "if_z"):
                cond = self._raw(args[0])[:, 0] != 0
                if op == "if_z":
                    cond = ~cond
                stack.append({"kind": "if", "saved": active.copy(), "cond": cond})
                active = active & cond
                if not active.any():
                    pc = ends[pc]                      # to its else or endif
                    continue
            elif op == "else":
                frame = stack[-1]
                active = alive(frame["saved"] & ~frame["cond"])
                if not active.any():
                    pc = ends[pc]
                    continue
            elif op == "endif":
                frame = stack.pop()
                active = alive(frame["saved"])
            elif op == "loop":
                stack.append({"kind": "loop", "saved": active.copy(), "start": pc, "broken": np.zeros(n, bool),
                              "count": 0})
            elif op == "endloop":
                frame = next(f for f in reversed(stack) if f["kind"] == "loop")
                frame["count"] += 1
                if frame["count"] > limit:
                    raise ShaderError("a loop ran past its limit")
                still = frame["saved"] & ~frame["broken"] & ~self.discarded
                if still.any():
                    active = still
                    pc = frame["start"] + 1
                    continue
                stack.pop()
                active = frame["saved"] & ~self.discarded
            elif op in ("break", "breakc_nz", "breakc_z"):
                lanes = active
                if op != "break":
                    cond = self._raw(args[0])[:, 0] != 0
                    lanes = active & (cond if op == "breakc_nz" else ~cond)
                frame = next(f for f in reversed(stack) if f["kind"] == "loop")
                frame["broken"] |= lanes
                active = active & ~lanes
            elif op in ("discard_nz", "discard_z"):
                cond = self._raw(args[0])[:, 0] != 0
                lanes = active & (cond if op == "discard_nz" else ~cond)
                self.discarded |= lanes
                active = active & ~lanes
            elif op == "ret":
                if not stack:
                    break
                active = np.zeros(n, bool)
            else:
                if active.any():
                    self._execute(ins, active)
            pc += 1
        return self.discarded

    def _execute(self, ins: Instruction, active: np.ndarray) -> None:
        op, a, sat = ins.op, ins.operands, ins.saturate
        f, i, u, w = self.f, self.i, self.u, self.write
        binary_float = {"add": np.add, "mul": np.multiply, "min": np.minimum, "max": np.maximum}
        if op == "mov":
            w(a[0], self._raw(a[1]).copy(), active, kind="u") if not (sat or a[1].negate or a[1].absolute) \
                else w(a[0], f(a[1]), active, sat)
        elif op in binary_float:
            w(a[0], binary_float[op](f(a[1]), f(a[2])), active, sat)
        elif op == "mad":
            w(a[0], f(a[1]) * f(a[2]) + f(a[3]), active, sat)
        elif op == "div":
            with np.errstate(divide="ignore", invalid="ignore"):
                w(a[0], f(a[1]) / f(a[2]), active, sat)
        elif op in ("dp2", "dp3", "dp4"):
            k = int(op[2])
            value = (f(a[1])[:, :k] * f(a[2])[:, :k]).sum(axis=1, keepdims=True)
            w(a[0], np.repeat(value, 4, axis=1), active, sat)
        elif op in ("rsq", "sqrt", "rcp", "exp", "log", "frc", "round_ni", "round_pi", "round_z", "round_ne"):
            x = f(a[1])
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                value = {"rsq": lambda: 1.0 / np.sqrt(x), "sqrt": lambda: np.sqrt(x), "rcp": lambda: 1.0 / x,
                         "exp": lambda: np.exp2(x), "log": lambda: np.log2(x), "frc": lambda: x - np.floor(x),
                         "round_ni": lambda: np.floor(x), "round_pi": lambda: np.ceil(x),
                         "round_z": lambda: np.trunc(x), "round_ne": lambda: np.rint(x)}[op]()
            w(a[0], value, active, sat)
        elif op == "sincos":
            x = f(a[2])
            w(a[0], np.sin(x), active, sat)
            w(a[1], np.cos(x), active, sat)
        elif op in ("ge", "lt", "eq", "ne"):
            x, y = f(a[1]), f(a[2])
            value = {"ge": x >= y, "lt": x < y, "eq": x == y, "ne": x != y}[op]
            w(a[0], np.where(value, 0xFFFFFFFF, 0), active, kind="u")
        elif op in ("ige", "ilt", "ieq", "ine", "uge", "ult"):
            x, y = (i(a[1]), i(a[2])) if op[0] == "i" else (u(a[1]), u(a[2]))
            value = {"ge": x >= y, "lt": x < y, "eq": x == y, "ne": x != y}[op[1:]]
            w(a[0], np.where(value, 0xFFFFFFFF, 0), active, kind="u")
        elif op in ("and", "or", "xor"):
            x, y = u(a[1]), u(a[2])
            w(a[0], {"and": x & y, "or": x | y, "xor": x ^ y}[op], active, kind="u")
        elif op == "not":
            w(a[0], ~u(a[1]) & 0xFFFFFFFF, active, kind="u")
        elif op == "movc":
            cond = self._raw(a[1]) != 0
            value = np.where(cond, self._raw(a[2]), self._raw(a[3]))
            w(a[0], value.view(np.float32), active, sat) if sat else w(a[0], value, active, kind="u")
        elif op in ("iadd", "imul", "ishl", "ishr", "ushr", "imax", "imin", "umax", "umin"):
            if op in ("ushr", "umax", "umin"):
                x, y = u(a[1]), u(a[2])
            else:
                x, y = i(a[1]), i(a[2])
            value = {"iadd": lambda: x + y, "imul": lambda: x * y, "ishl": lambda: x << (y & 31),
                     "ishr": lambda: x >> (y & 31), "ushr": lambda: x >> (y & 31), "imax": lambda: np.maximum(x, y),
                     "imin": lambda: np.minimum(x, y), "umax": lambda: np.maximum(x, y),
                     "umin": lambda: np.minimum(x, y)}[op]()
            w(a[0], value, active, kind="u")
        elif op == "ineg":
            w(a[0], -i(a[1]), active, kind="u")
        elif op == "udiv":
            x, y = u(a[2]), u(a[3])
            safe = np.where(y == 0, 1, y)
            w(a[0], np.where(y == 0, 0xFFFFFFFF, x // safe), active, kind="u")
            w(a[1], np.where(y == 0, 0xFFFFFFFF, x % safe), active, kind="u")
        elif op == "itof":
            w(a[0], i(a[1]).astype(np.float32), active, sat)
        elif op == "utof":
            w(a[0], u(a[1]).astype(np.float32), active, sat)
        elif op in ("ftoi", "ftou"):
            x = np.nan_to_num(f(a[1]))
            w(a[0], np.trunc(np.clip(x, -2 ** 31, 2 ** 32 - 1)).astype(np.int64), active, kind="u")
        elif op.startswith("deriv_"):
            w(a[0], np.zeros((self.n, 4), np.float32), active, sat)
        elif op.startswith("sample"):
            register = a[2].index
            coordinates = f(a[1])
            value = self.sample(register, coordinates[:, :2])[:, list(ins.resource_swizzle)]
            w(a[0], value, active, sat)
        elif op == "nop":
            pass
        else:
            raise ShaderError(f"instruction {op!r} is not run here")


def _structure(program: list[Instruction]) -> dict[int, int]:
    """For each if and else, where execution goes when no lane takes it: its else (past it) or its endif."""
    ends, stack = {}, []
    for pc, ins in enumerate(program):
        if ins.op in ("if_nz", "if_z"):
            stack.append(pc)
        elif ins.op == "else":
            ends[stack[-1]] = pc                       # the if's lanes all skip to the else (which runs)
            stack[-1] = pc
        elif ins.op == "endif":
            ends[stack.pop()] = pc                     # to the endif (which pops)
    return ends
