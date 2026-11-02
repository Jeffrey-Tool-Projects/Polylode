"""Run the source game's joint correction on a skeleton.

A pose that is not one of the source game's clips has no curves for the helper
joints its skin hangs on.  The game's studio fills them in with
``ExpressionBone.LateUpdate``; ``tools/bake_joint_correction.py`` converts that
method (and the 15 small ones it calls) from the user's install into
``assets_baked/joint_correction.json`` — instructions as they are, operands by
name — and this module runs them.

It is a small stack machine over just the vocabulary those methods use, with
the engine's side supplied here: transforms (:class:`Rig`, in the engine's own
left-handed space, converted from and to a :class:`Skeleton` at the edges),
vectors, quaternions, a TRS matrix and a dozen ``Mathf`` functions.  An
instruction or a call it has not been taught raises :class:`Unsupported` by
name — it never skips one.

Without the converted file there is no correction (:func:`corrector` answers
None) and the helper joints simply follow their bones, as before.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..core import math3d as m3
from . import bake as _bake

VERSION = 1
FILE = "joint_correction.json"
#: the nodes ``ExpressionBone.Initialize`` makes: name → (parent, local position), its own space
GOALS = {
    "n_goal_forearm_L": ("cf_j_shoulder_L", (0.0, 0.3, 0.0)),
    "n_goal_forearm_R": ("cf_j_shoulder_R", (0.0, 0.3, 0.0)),
    "n_leg_goal_L": ("cf_j_waist02", (-0.083, 0.4, -0.008)),
    "n_leg_goal_R": ("cf_j_waist02", (0.083, 0.4, -0.008)),
}
#: the scratch transform it parents to the character's root
CALC = "objCalculation"


class Unsupported(RuntimeError):
    """The converted program uses something this machine has not been taught."""


def program_path() -> Path:
    return _bake.bake_dir() / FILE


_program: list = [None, False]


def program() -> dict | None:
    """The converted joint correction, or None when it was never converted."""
    if not _program[1]:
        _program[1] = True
        path = program_path()
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if int(data.get("version", 0)) == VERSION:
                _program[0] = data
    return _program[0]


def forget() -> None:
    """Read the converted program again on next use (it was just converted)."""
    _program[0], _program[1] = None, False


# ---------------------------------------------------------------------------
# the engine's side: quaternions (x, y, z, w), vectors, a rig of transforms
# ---------------------------------------------------------------------------
def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def q_rotate(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)]


def q_inv(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w or 1.0
    return [-x / n, -y / n, -z / n, w / n]


def q_angle_axis(degrees, axis):
    n = math.sqrt(sum(c * c for c in axis)) or 1.0
    half = math.radians(degrees) * 0.5
    s = math.sin(half) / n
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)]


def q_euler(x, y, z):
    """``Quaternion.Euler``: z, then x, then y."""
    return q_mul(q_mul(q_angle_axis(y, (0, 1, 0)), q_angle_axis(x, (1, 0, 0))), q_angle_axis(z, (0, 0, 1)))


def q_matrix(q):
    x, y, z, w = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def q_euler_angles(q):
    """``Quaternion.eulerAngles``: the angles ``Quaternion.Euler`` would take, each in 0‥360."""
    m = q_matrix(q)
    sx = max(-1.0, min(1.0, -m[1][2]))
    x = math.asin(sx)
    if abs(sx) < 0.9999999:
        y, z = math.atan2(m[0][2], m[2][2]), math.atan2(m[1][0], m[1][1])
    else:
        y, z = math.atan2(-m[2][0], m[0][0]), 0.0
    return [math.degrees(a) % 360.0 for a in (x, y, z)]


class Node:
    """A transform of the rig, by index."""
    __slots__ = ("rig", "index")

    def __init__(self, rig, index):
        self.rig, self.index = rig, index


class Rig:
    """Transforms in the engine's space: local position, rotation and scale under a parent."""

    def __init__(self, names, parents, position, rotation, scale):
        self.names = list(names)
        self.index = {name: i for i, name in enumerate(self.names)}
        self.parent = list(parents)
        self.position = [list(map(float, p)) for p in position]
        self.rotation = [list(map(float, q)) for q in rotation]
        self.scale = [list(map(float, s)) for s in scale]
        self._world: dict = {}

    def node(self, name):
        i = self.index.get(name)
        return None if i is None else Node(self, i)

    def touched(self) -> None:
        self._world.clear()

    def world(self, i) -> np.ndarray:
        """The local-to-world matrix of node *i*."""
        got = self._world.get(i)
        if got is None:
            local = np.eye(4)
            local[:3, :3] = np.asarray(q_matrix(self.rotation[i])) * np.asarray(self.scale[i])[None, :]
            local[:3, 3] = self.position[i]
            got = local if self.parent[i] < 0 else self.world(self.parent[i]) @ local
            self._world[i] = got
        return got

    def world_rotation(self, i):
        q = self.rotation[i]
        up = self.parent[i]
        return q if up < 0 else q_mul(self.world_rotation(up), q)


def from_skeleton(skeleton) -> Rig:
    """*skeleton* at rest as a rig in the engine's space, with the correction's own nodes.

    The engine's space is this project's with X mirrored: a position's x and a
    quaternion's y and z change sign.
    """
    names = [bone.name for bone in skeleton.bones]
    parents = [bone.parent for bone in skeleton.bones]
    position, rotation, scale = [], [], []
    for bone in skeleton.bones:
        p, q = bone.local_pos, bone.local_rot
        position.append((-p[0], p[1], p[2]))
        rotation.append((q[0], -q[1], -q[2], q[3]))
        scale.append(getattr(bone, "local_scale", (1.0, 1.0, 1.0)))
    rig = Rig(names, parents, position, rotation, scale)
    for name, (parent, at) in GOALS.items():
        if parent in rig.index:
            _add(rig, name, rig.index[parent], at)
    _add(rig, CALC, -1, (0.0, 0.0, 0.0))
    return rig


def posed(rest: Rig, skeleton, pose: dict) -> Rig:
    """A copy of *rest* under *pose* (entries as ``Skeleton.evaluate`` takes them)."""
    rig = Rig.__new__(Rig)
    rig.names, rig.index, rig.parent = rest.names, rest.index, rest.parent
    rig.position, rig.rotation, rig.scale = list(rest.position), list(rest.rotation), list(rest.scale)
    rig._world = {}
    for name, entry in pose.items():
        i = skeleton.index.get(name)
        if i is None or not entry:
            continue
        bone = skeleton.bones[i]
        if entry.get("pos") is not None:
            p = np.asarray(bone.local_pos, dtype=np.float64) + np.asarray(entry["pos"], dtype=np.float64)
            rig.position[i] = [-float(p[0]), float(p[1]), float(p[2])]
        if entry.get("rot") is not None:
            q = m3.quat_mul(np.asarray(bone.local_rot, dtype=np.float64), np.asarray(entry["rot"], dtype=np.float64))
            rig.rotation[i] = [float(q[0]), -float(q[1]), -float(q[2]), float(q[3])]
        if entry.get("scale") is not None:
            rig.scale[i] = [float(c) for c in np.asarray(rest.scale[i]) * np.asarray(entry["scale"], dtype=np.float64)]
    return rig


def _add(rig: Rig, name: str, parent: int, at) -> None:
    rig.index[name] = len(rig.names)
    rig.names.append(name)
    rig.parent.append(parent)
    rig.position.append(list(map(float, at)))
    rig.rotation.append([0.0, 0.0, 0.0, 1.0])
    rig.scale.append([1.0, 1.0, 1.0])


def to_pose(rig: Rig, skeleton, before: tuple, pose: dict) -> dict:
    """*pose* with what the run wrote on *rig*, as entries ``Skeleton.evaluate`` takes.

    *before* is the rig's ``(position, rotation, scale)`` lists as they were:
    a write replaces the bone's entry, so the written ones are the entries
    that are no longer the same object.
    """
    out = dict(pose)
    was_p, was_q, was_s = before
    for i, bone in enumerate(skeleton.bones):
        if rig.position[i] is was_p[i] and rig.rotation[i] is was_q[i] and rig.scale[i] is was_s[i]:
            continue
        p, q, s = rig.position[i], rig.rotation[i], rig.scale[i]
        ours = np.array([q[0], -q[1], -q[2], q[3]])
        rest_scale = np.asarray(getattr(bone, "local_scale", (1.0, 1.0, 1.0)), dtype=np.float64)
        out[bone.name] = {
            "pos": np.array([-p[0], p[1], p[2]]) - np.asarray(bone.local_pos, dtype=np.float64),
            "rot": m3.quat_normalize(m3.quat_mul(m3.quat_conj(np.asarray(bone.local_rot, dtype=np.float64)), ours)),
            "scale": np.asarray(s) / np.maximum(rest_scale, 1e-8)}
    return out


# ---------------------------------------------------------------------------
# the machine
# ---------------------------------------------------------------------------
class Slot:
    """The address of a local, an argument or a field of one: what ``ldloca`` and ``ldflda`` push."""
    __slots__ = ("holder", "key")

    def __init__(self, holder, key):
        self.holder, self.key = holder, key

    def get(self):
        return self.holder[self.key]

    def put(self, value) -> None:
        self.holder[self.key] = value


FIELD = {"Vector3::x": 0, "Vector3::y": 1, "Vector3::z": 2}
COMPARE = {"beq": lambda a, b: a == b, "bge": lambda a, b: a >= b, "bgt": lambda a, b: a > b,
           "ble": lambda a, b: a <= b, "blt": lambda a, b: a < b, "bne.un": lambda a, b: a != b,
           "bge.un": lambda a, b: not a < b, "bgt.un": lambda a, b: not a <= b,
           "ble.un": lambda a, b: not a > b, "blt.un": lambda a, b: not a >= b}
BINARY = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
          "div": lambda a, b: a / b}


def _vec(a, b, op):
    if isinstance(a, list) and isinstance(b, list):
        return [op(x, y) for x, y in zip(a, b)]
    if isinstance(a, list):
        return [op(x, b) for x in a]
    return [op(a, y) for y in b]


def _trs(position, rotation, scale):
    rot = q_matrix(rotation)
    return [[rot[r][c] * scale[c] for c in range(3)] + [position[r]] for r in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def _local(kind):
    def get(node):
        return list(getattr(node.rig, kind)[node.index])

    def put(node, value):
        getattr(node.rig, kind)[node.index] = [float(c) for c in value]
        node.rig.touched()
    return get, put


_get_position, _set_position = _local("position")
_get_rotation, _set_rotation = _local("rotation")
_get_scale, _set_scale = _local("scale")


def _world_position(node):
    return [float(c) for c in node.rig.world(node.index)[:3, 3]]


def _set_world_position(node, at):
    up = node.rig.parent[node.index]
    local = at if up < 0 else (np.linalg.inv(node.rig.world(up)) @ np.array([*at, 1.0]))[:3]
    _set_position(node, local)


def _set_world_rotation(node, q):
    up = node.rig.parent[node.index]
    _set_rotation(node, q if up < 0 else q_mul(q_inv(node.rig.world_rotation(up)), q))


def _inverse_point(node, at):
    return [float(c) for c in (np.linalg.inv(node.rig.world(node.index)) @ np.array([*at, 1.0]))[:3]]


def _translate(node, x, y, z, space):
    """``Transform.Translate(x, y, z, relativeTo)``: along its own axes (``Space.Self``) or the world's."""
    here = _world_position(node)
    moved = q_rotate(node.rig.world_rotation(node.index), [x, y, z]) if space == 1 else [x, y, z]
    _set_world_position(node, [here[k] + moved[k] for k in range(3)])


def _angle(a, b):
    la, lb = math.sqrt(sum(c * c for c in a)), math.sqrt(sum(c * c for c in b))
    if la * lb < 1e-15:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (la * lb)))))


def _normalized(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v] if n > 1e-5 else [0.0, 0.0, 0.0]


#: ``Owner::Name`` → (how many values it takes, what it does).  ``None`` results push nothing.
NATIVE = {
    "Vector3::get_one": (0, lambda: [1.0, 1.0, 1.0]),
    "Vector3::get_zero": (0, lambda: [0.0, 0.0, 0.0]),
    "Vector3::get_up": (0, lambda: [0.0, 1.0, 0.0]),
    "Vector3::get_right": (0, lambda: [1.0, 0.0, 0.0]),
    "Vector3::get_forward": (0, lambda: [0.0, 0.0, 1.0]),
    "Vector3::get_normalized": (1, lambda v: _normalized(v.get() if isinstance(v, Slot) else v)),
    "Vector3::Angle": (2, _angle),
    "Vector3::op_Multiply": (2, lambda a, b: _vec(a, b, lambda x, y: x * y)),
    "Vector3::op_Division": (2, lambda a, b: _vec(a, b, lambda x, y: x / y)),
    "Vector3::op_Addition": (2, lambda a, b: _vec(a, b, lambda x, y: x + y)),
    "Vector3::op_Subtraction": (2, lambda a, b: _vec(a, b, lambda x, y: x - y)),
    "Quaternion::get_identity": (0, lambda: [0.0, 0.0, 0.0, 1.0]),
    "Quaternion::AngleAxis": (2, q_angle_axis),
    "Quaternion::op_Multiply": (2, q_mul),
    "Quaternion::get_eulerAngles": (1, lambda q: q_euler_angles(q.get() if isinstance(q, Slot) else q)),
    "Quaternion::Euler": (1, lambda v: q_euler(*v)),
    "Matrix4x4::TRS": (3, _trs),
    "Mathf::Abs": (1, abs), "Mathf::Min": (2, min), "Mathf::Max": (2, max),
    "Mathf::Sin": (1, math.sin), "Mathf::Cos": (1, math.cos), "Mathf::Tan": (1, math.tan),
    "Mathf::Asin": (1, lambda x: math.asin(max(-1.0, min(1.0, x)))),
    "Mathf::Acos": (1, lambda x: math.acos(max(-1.0, min(1.0, x)))),
    "Mathf::Atan2": (2, math.atan2), "Mathf::Sqrt": (1, math.sqrt),
    "Mathf::Clamp": (3, lambda value, low, high: max(low, min(high, value))),
    "Mathf::Clamp01": (1, lambda value: max(0.0, min(1.0, value))),
    "Mathf::Sign": (1, lambda x: 1.0 if x >= 0.0 else -1.0),
    "Single::IsNaN": (1, lambda x: x != x),
    "Object::op_Implicit": (1, lambda x: x is not None),
    "Object::op_Inequality": (2, lambda a, b: (a is None) != (b is None) or (
        a is not None and b is not None and a.index != b.index)),
    "Transform::get_localPosition": (1, _get_position),
    "Transform::set_localPosition": (2, _set_position),
    "Transform::get_localRotation": (1, _get_rotation),
    "Transform::set_localRotation": (2, _set_rotation),
    "Transform::get_localScale": (1, _get_scale),
    "Transform::set_localScale": (2, _set_scale),
    "Transform::get_position": (1, _world_position),
    "Transform::set_position": (2, _set_world_position),
    "Transform::get_rotation": (1, lambda node: list(node.rig.world_rotation(node.index))),
    "Transform::set_rotation": (2, _set_world_rotation),
    "Transform::InverseTransformPoint": (2, _inverse_point),
    "Transform::Translate": (5, _translate),
}
RETURNS_NOTHING = frozenset(name for name in NATIVE if "::set_" in name) | {"Transform::Translate"}


(ARG, ARG_ADDRESS, SET_ARG, LOCAL, LOCAL_ADDRESS, SET_LOCAL, CONSTANT, DUP, POP, NEG, RET, NOP, BINARY_OP,
 TO_FLOAT, TO_INT, INDIRECT, JUMP, JUMP_IF, JUMP_COMPARE, SWITCH, COMPONENT, COMPONENT_ADDRESS, CELL, OWN_FIELD,
 SET_COMPONENT, CALL, SET_INDIRECT) = range(27)


class Machine:
    """Runs the converted methods on a :class:`Rig`."""

    def __init__(self, data: dict):
        self.methods = data["methods"]
        self.entry = data["entry"]
        self.bones = {int(value): name for value, name in data["bones"].items()}
        self._compiled: dict = {}
        self.rig: Rig | None = None

    def run(self, rig: Rig) -> None:
        """Run the entry method — one frame of the correction — on *rig*."""
        self.rig = rig
        self.call(self.entry, ["this"])

    # the component's own state: every part of the correction on, its bones by enum value
    def _field(self, name: str):
        if name == "ExpressionBone.calcTranceform":
            return self.rig.node(CALC)
        if name.startswith("ExpressionBone.enable"):
            return True
        if name.endswith("k__BackingField"):
            return "bones"
        raise Unsupported(f"field {name}")

    def _compile(self, name: str) -> list:
        """A method's instructions as ``(kind, a, b)``: decided once, so a run only dispatches on a number."""
        jump = {ins[0]: at for at, ins in enumerate(self.methods[name]["code"])}
        out = []
        for _off, op, arg, target in self.methods[name]["code"]:
            short = op.removesuffix(".s")
            if op == "ldarga.s":
                out.append((ARG_ADDRESS, arg, None))
            elif op == "starg.s":
                out.append((SET_ARG, arg, None))
            elif op.startswith("ldarg"):
                out.append((ARG, arg if op.endswith(".s") else int(op[-1]), None))
            elif op.startswith("ldloca"):
                out.append((LOCAL_ADDRESS, arg, None))
            elif op.startswith("ldloc"):
                out.append((LOCAL, arg if op.endswith(".s") else int(op[-1]), None))
            elif op.startswith("stloc"):
                out.append((SET_LOCAL, arg if op.endswith(".s") else int(op[-1]), None))
            elif op.startswith("ldc."):
                out.append((CONSTANT, float(arg) if ".r" in op else int(arg), None))
            elif op == "ldnull":
                out.append((CONSTANT, None, None))
            elif op in ("dup", "pop", "neg", "ret", "nop"):
                out.append(({"dup": DUP, "pop": POP, "neg": NEG, "ret": RET, "nop": NOP}[op], None, None))
            elif op in BINARY:
                out.append((BINARY_OP, BINARY[op], None))
            elif op in ("op6b", "op6c"):                      # conv.r4 / conv.r8
                out.append((TO_FLOAT, None, None))
            elif op == "conv.i4":
                out.append((TO_INT, None, None))
            elif op == "op4e":                                # ldind.r4
                out.append((INDIRECT, None, None))
            elif op == "op56":                                # stind.r4
                out.append((SET_INDIRECT, None, None))
            elif short == "br":
                out.append((JUMP, jump[target], None))
            elif short in ("brfalse", "brtrue"):
                out.append((JUMP_IF, jump[target], short == "brtrue"))
            elif short in COMPARE:
                out.append((JUMP_COMPARE, jump[target], COMPARE[short]))
            elif op == "switch":
                out.append((SWITCH, [jump[t] for t in target], None))
            elif op in ("ldfld", "ldflda"):
                if arg in FIELD:
                    out.append((COMPONENT_ADDRESS if op == "ldflda" else COMPONENT, FIELD[arg], None))
                elif arg.startswith("Matrix4x4::m"):
                    out.append((CELL, int(arg[-2]), int(arg[-1])))
                else:
                    out.append((OWN_FIELD, arg, None))
            elif op == "stfld":
                if arg not in FIELD:
                    raise Unsupported(f"stfld {arg}")
                out.append((SET_COMPONENT, FIELD[arg], None))
            elif op in ("call", "callvirt", "newobj"):
                out.append((CALL, op, arg))
            else:
                raise Unsupported(f"instruction {op} in {name}")
        return out

    def call(self, name: str, args: list):
        code = self._compiled.get(name)
        if code is None:
            code = self._compiled[name] = self._compile(name)
        local: dict = {}
        stack: list = []
        push, pop = stack.append, stack.pop
        at = 0
        while True:
            kind, a, b = code[at]
            at += 1
            if kind == LOCAL:
                value = local[a]
                push(value[:] if type(value) is list else value)
            elif kind == CONSTANT:
                push(a)
            elif kind == LOCAL_ADDRESS:
                if a not in local:
                    local[a] = [0.0, 0.0, 0.0]
                push(Slot(local, a))
            elif kind == SET_LOCAL:
                local[a] = pop()
            elif kind == CALL:
                result = self._invoke(a, b, stack)
                if result is not _NOTHING:
                    push(result)
            elif kind == BINARY_OP:
                right = pop()
                push(a(pop(), right))
            elif kind == COMPONENT:
                owner = pop()
                push((owner.get() if type(owner) is Slot else owner)[a])
            elif kind == ARG:
                value = args[a]
                push(value[:] if type(value) is list else value)
            elif kind == JUMP_IF:
                if bool(pop()) == b:
                    at = a
            elif kind == OWN_FIELD:
                pop()
                push(self._field(a))
            elif kind == SET_COMPONENT:
                value, owner = pop(), pop()
                (owner.get() if type(owner) is Slot else owner)[a] = float(value)
            elif kind == ARG_ADDRESS:
                push(Slot(args, a))
            elif kind == JUMP:
                at = a
            elif kind == JUMP_COMPARE:
                right = pop()
                if b(pop(), right):
                    at = a
            elif kind == RET:
                return pop() if stack else None
            elif kind == CELL:
                owner = pop()
                push((owner.get() if type(owner) is Slot else owner)[a][b])
            elif kind == COMPONENT_ADDRESS:
                owner = pop()
                push(Slot(owner.get() if type(owner) is Slot else owner, a))
            elif kind == SWITCH:
                value = pop()
                if 0 <= value < len(a):
                    at = a[value]
            elif kind == NEG:
                push(-pop())
            elif kind == DUP:
                value = stack[-1]
                push(value[:] if type(value) is list else value)
            elif kind == POP:
                pop()
            elif kind == TO_FLOAT:
                push(float(pop()))
            elif kind == TO_INT:
                push(int(pop()))
            elif kind == INDIRECT:
                push(float(pop().get()))
            elif kind == SET_ARG:
                args[a] = pop()
            elif kind == SET_INDIRECT:
                value = pop()
                pop().put(float(value))

    def _invoke(self, op: str, name: str, stack: list):
        if name in self.methods:
            method = self.methods[name]
            count = method["params"] + (1 if method["this"] else 0)
            args = [stack.pop() for _ in range(count)][::-1]
            result = self.call(name, args)
            return result if method["returns"] else _NOTHING
        if name == "Vector3::.ctor":
            z, y, x = stack.pop(), stack.pop(), stack.pop()
            made = [float(x), float(y), float(z)]
            if op == "newobj":
                return made
            stack.pop().put(made)
            return _NOTHING
        if name.endswith("::get_Item"):
            key, _owner = stack.pop(), stack.pop()
            return self.rig.node(self.bones.get(int(key), ""))
        if name.endswith("::get_Count"):
            stack.pop()
            return len(self.bones)
        if name not in NATIVE:
            raise Unsupported(f"call {name}")
        count, function = NATIVE[name]
        args = [stack.pop() for _ in range(count)][::-1]
        result = function(*args)
        return _NOTHING if name in RETURNS_NOTHING else result


_NOTHING = object()


class Corrector:
    """The joint correction for one skeleton: ``apply(pose)`` gives the pose with its helper joints set."""

    def __init__(self, skeleton, data: dict):
        self.skeleton = skeleton
        self.machine = Machine(data)
        self.rest = from_skeleton(skeleton)

    def apply(self, pose: dict) -> dict:
        rig = posed(self.rest, self.skeleton, pose)
        before = (list(rig.position), list(rig.rotation), list(rig.scale))
        self.machine.run(rig)
        return to_pose(rig, self.skeleton, before, pose)


def corrector(skeleton) -> Corrector | None:
    """A :class:`Corrector` for *skeleton*, or None when the correction was never converted."""
    data = program()
    return None if data is None else Corrector(skeleton, data)
