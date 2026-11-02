"""The body armature's ``FullBodyBipedIK`` component, converted to our JSON (Animation_Progress D14).

``chara/oo_base.unity3d`` carries one ``FullBodyBipedIK`` on ``p_cf_body_bone``
(``MotionIK..ctor`` reads it with ``animBody.GetComponent<FullBodyBipedIK>()``).
Its serialized fields are what ``anim.fbbik`` solves with; the bundle ships no
type tree for the script, so :func:`parse_component` reads the bytes in the
field order of ``RootMotion.FinalIK.FullBodyBipedIK`` (as
``tools/origin_mono_dump.py chara/oo_base.unity3d --script FullBodyBipedIK``
prints it from ``Assembly-CSharp-firstpass.dll``) and refuses bytes that do not
end where the fields do.

The object references are ``Transform`` path ids; the bake resolves every one
to its GameObject's name, so the runtime (``origin.motion_ik``) reads names and
never opens a bundle.  Also kept: ``p_cf_body_bone``'s own local rotation,
which turns ``IKBeforeProcess``'s offsets (``objBodyBone.TransformDirection``)
and is the rotation ``PullBody`` reads its up axis from.
"""
from __future__ import annotations

import struct
from pathlib import Path

from ..core.paths import write_json_atomic
from .install import Install

FBBIK_VERSION = 1
BODY_BUNDLE = "chara/oo_base.unity3d"
#: ``ChaControl.objBodyBone``: the GameObject the component sits on
BODY_BONE = "p_cf_body_bone"
SCRIPT = "FullBodyBipedIK"


class _Reader:
    """Unity's serialized layout: little-endian, 4-byte aligned after bools, strings and arrays."""

    def __init__(self, raw: bytes):
        self.raw, self.at = raw, 0

    def _unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        if self.at + size > len(self.raw):
            raise ValueError(f"read past the end at {self.at} (+{size} of {len(self.raw)})")
        value = struct.unpack_from(fmt, self.raw, self.at)
        self.at += size
        return value

    def _align(self) -> None:
        self.at = (self.at + 3) & ~3

    def i32(self) -> int:
        """``int`` (and every enum)."""
        return self._unpack("<i")[0]

    def f32(self) -> float:
        """``float``."""
        return self._unpack("<f")[0]

    def flag(self) -> int:
        """``bool`` / ``UInt8``: one byte, then aligned."""
        value = self._unpack("<B")[0]
        self._align()
        return value

    def string(self) -> str:
        """``string``: a length, the UTF-8 bytes, aligned."""
        n = self.i32()
        if n < 0 or self.at + n > len(self.raw):
            raise ValueError(f"string of {n} bytes at {self.at}")
        out = self.raw[self.at:self.at + n].decode("utf-8")
        self.at += n
        self._align()
        return out

    def pptr(self) -> dict:
        """``PPtr<T>``: file id, then the 64-bit path id."""
        fid = self.i32()
        return {"m_FileID": fid, "m_PathID": self._unpack("<q")[0]}

    def v3(self) -> dict:
        """``Vector3``."""
        return dict(zip("xyz", self._unpack("<3f")))

    def q4(self) -> dict:
        """``Quaternion``."""
        return dict(zip("xyzw", self._unpack("<4f")))

    def array(self, item) -> list:
        """``T[]`` / ``List<T>``: a count, the items."""
        n = self.i32()
        if n < 0 or n > len(self.raw):
            raise ValueError(f"array of {n} at {self.at}")
        return [item() for _ in range(n)]


def _node(r: _Reader) -> dict:
    """``IKSolver.Node``."""
    return {"transform": r.pptr(), "weight": r.f32(), "solverPosition": r.v3(), "solverRotation": r.q4(),
            "defaultLocalPosition": r.v3(), "defaultLocalRotation": r.q4(), "length": r.f32(),
            "effectorPositionWeight": r.f32(), "effectorRotationWeight": r.f32(), "offset": r.v3()}


def _child_constraint(r: _Reader) -> dict:
    """``FBIKChain.ChildConstraint``."""
    return {"pushElasticity": r.f32(), "pullElasticity": r.f32(), "bone1": r.pptr(), "bone2": r.pptr()}


def _bend(r: _Reader) -> dict:
    """``IKConstraintBend``."""
    return {"bone1": r.pptr(), "bone2": r.pptr(), "bone3": r.pptr(), "bendGoal": r.pptr(),
            "direction": r.v3(), "rotationOffset": r.q4(), "weight": r.f32(),
            "defaultLocalDirection": r.v3(), "defaultChildDirection": r.v3()}


def _chain(r: _Reader) -> dict:
    """``FBIKChain``."""
    return {"pin": r.f32(), "pull": r.f32(), "push": r.f32(), "pushParent": r.f32(), "reach": r.f32(),
            "reachSmoothing": r.i32(), "pushSmoothing": r.i32(), "nodes": r.array(lambda: _node(r)),
            "children": r.array(r.i32), "childConstraints": r.array(lambda: _child_constraint(r)),
            "bendConstraint": _bend(r)}


def _effector(r: _Reader) -> dict:
    """``IKEffector``."""
    return {"bone": r.pptr(), "target": r.pptr(), "positionWeight": r.f32(), "rotationWeight": r.f32(),
            "position": r.v3(), "rotation": r.q4(), "positionOffset": r.v3(), "effectChildNodes": r.flag(),
            "maintainRelativePositionWeight": r.f32(), "childBones": r.array(r.pptr),
            "planeBone1": r.pptr(), "planeBone2": r.pptr(), "planeBone3": r.pptr(),
            "planeRotationOffset": r.q4()}


def _spine_mapping(r: _Reader) -> dict:
    """``IKMappingSpine``."""
    return {"spineBones": r.array(r.pptr), "leftUpperArmBone": r.pptr(), "rightUpperArmBone": r.pptr(),
            "leftThighBone": r.pptr(), "rightThighBone": r.pptr(), "iterations": r.i32(),
            "twistWeight": r.f32()}


def _limb_mapping(r: _Reader) -> dict:
    """``IKMappingLimb``."""
    return {"parentBone": r.pptr(), "bone1": r.pptr(), "bone2": r.pptr(), "bone3": r.pptr(),
            "maintainRotationWeight": r.f32(), "weight": r.f32()}


def _references(r: _Reader) -> dict:
    """``BipedReferences``."""
    out = {key: r.pptr() for key in ("root", "pelvis", "leftThigh", "leftCalf", "leftFoot", "rightThigh",
                                     "rightCalf", "rightFoot", "leftUpperArm", "leftForearm", "leftHand",
                                     "rightUpperArm", "rightForearm", "rightHand", "head")}
    out["spine"] = r.array(r.pptr)
    out["eyes"] = r.array(r.pptr)
    return out


def _solver(r: _Reader) -> dict:
    """``IKSolverFullBodyBiped``, with the fields ``IKSolver`` and ``IKSolverFullBody`` serialize first."""
    return {"IKPosition": r.v3(), "IKPositionWeight": r.f32(), "root": r.pptr(), "iterations": r.i32(),
            "chain": r.array(lambda: _chain(r)), "effectors": r.array(lambda: _effector(r)),
            "spineMapping": _spine_mapping(r), "boneMappings": r.array(lambda: {
                "bone": r.pptr(), "maintainRotationWeight": r.f32()}),
            "limbMappings": r.array(lambda: _limb_mapping(r)), "rootNode": r.pptr(),
            "spineStiffness": r.f32(), "pullBodyVertical": r.f32(), "pullBodyHorizontal": r.f32()}


def parse_component(raw: bytes) -> dict:
    """``FullBodyBipedIK``'s serialized fields as a dict (the type tree's names); ``ValueError`` when the bytes do not fit."""
    r = _Reader(raw)
    out = {"m_GameObject": r.pptr(), "m_Enabled": r.flag(), "m_Script": r.pptr(), "m_Name": r.string(),
           "timeStep": r.f32(), "fixTransforms": r.flag(), "references": _references(r), "solver": _solver(r)}
    if r.at != len(raw):
        raise ValueError(f"{len(raw) - r.at} bytes left after the fields")
    return out


def _pptrs(value) -> list:
    """Every ``PPtr`` dict inside *value*."""
    if isinstance(value, dict):
        if set(value) == {"m_FileID", "m_PathID"}:
            return [value]
        return [p for v in value.values() for p in _pptrs(v)]
    if isinstance(value, list):
        return [p for v in value for p in _pptrs(v)]
    return []


def transform_names(solver: dict, name_of) -> dict:
    """``{str(path id): name}`` for every reference in *solver*; ``KeyError`` for one in another file or unresolved."""
    out = {}
    for p in _pptrs(solver):
        pid = int(p["m_PathID"])
        if pid == 0:
            continue
        if int(p["m_FileID"]) != 0:
            raise KeyError(f"reference {pid} points into another file ({p['m_FileID']})")
        name = name_of(pid)
        if not name:
            raise KeyError(f"reference {pid} is not a transform of this bundle")
        out[str(pid)] = name
    return out


class _Bundle:
    """The body bundle's objects by path id, with the two lookups the bake needs."""

    def __init__(self, path: Path):
        import UnityPy
        self.env = UnityPy.load(str(path))
        self.by_id = {o.path_id: o for o in self.env.objects}

    def tree(self, pid: int) -> dict:
        """The base fields UnityPy reads for object *pid* (a script's own fields are not among them)."""
        return self.by_id[pid].read_typetree()

    def name(self, go_pid: int) -> str:
        """A GameObject's name."""
        return str(self.tree(go_pid).get("m_Name") or "")

    def transform_name(self, pid: int) -> str | None:
        """The name of the GameObject the Transform *pid* belongs to, or None when *pid* is not a Transform."""
        obj = self.by_id.get(pid)
        if obj is None or obj.type.name != "Transform":
            return None
        return self.name(int(obj.read_typetree()["m_GameObject"]["m_PathID"]))

    def script_class(self, obj) -> str:
        """The class name of a MonoBehaviour's script."""
        script = self.by_id.get(int(obj.read_typetree()["m_Script"]["m_PathID"]))
        return "" if script is None else str(script.read_typetree().get("m_ClassName") or "")

    def body_bone(self) -> tuple:
        """``(GameObject path id, Transform tree)`` of the one ``p_cf_body_bone``; ``LookupError`` otherwise."""
        found = []
        for o in self.env.objects:
            if o.type.name == "GameObject" and self.name(o.path_id) == BODY_BONE:
                found.append(o.path_id)
        if len(found) != 1:
            raise LookupError(f"{len(found)} GameObjects named {BODY_BONE}")
        for o in self.env.objects:
            if o.type.name == "Transform":
                tt = o.read_typetree()
                if int(tt["m_GameObject"]["m_PathID"]) == found[0]:
                    return found[0], tt
        raise LookupError(f"{BODY_BONE} has no Transform")

    def component(self, go_pid: int):
        """The ``FullBodyBipedIK`` MonoBehaviour on GameObject *go_pid*; ``LookupError`` unless exactly one."""
        hits = [o for o in self.env.objects if o.type.name == "MonoBehaviour"
                and int(o.read_typetree()["m_GameObject"]["m_PathID"]) == go_pid
                and self.script_class(o) == SCRIPT]
        if len(hits) != 1:
            raise LookupError(f"{len(hits)} {SCRIPT} components on {BODY_BONE}")
        return hits[0]


def read_bundle(path: Path) -> dict:
    """The payload :func:`bake_fbbik` writes, read off the body bundle at *path*."""
    bundle = _Bundle(path)
    go_pid, transform = bundle.body_bone()
    component = parse_component(bytes(bundle.component(go_pid).get_raw_data()))
    solver = component["solver"]
    rot = transform["m_LocalRotation"]
    return {"version": FBBIK_VERSION, "bundle": BODY_BUNDLE, "body_bone": BODY_BONE,
            "body_bone_rotation": [float(rot[k]) for k in "xyzw"],
            "enabled": int(component["m_Enabled"]), "time_step": component["timeStep"],
            "fix_transforms": int(component["fixTransforms"]), "solver": solver,
            "names": transform_names(solver, bundle.transform_name)}


def bake_fbbik(dest: Path, *, inst: Install | None = None) -> dict:
    """Write ``fbbik.json``: the body armature's ``FullBodyBipedIK``, references as names."""
    from .hlist import _install
    inst = inst or _install()
    if inst is None:
        return {"ok": False, "error": "no_origin_install"}
    path = inst.bundle(BODY_BUNDLE)
    if not path.is_file():
        return {"ok": False, "error": "no_body_bundle"}
    try:
        payload = read_bundle(path)
    except (LookupError, ValueError) as exc:          # KeyError is a LookupError
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    write_json_atomic(dest, payload, separators=(",", ":"))
    return {"ok": True, "effectors": len(payload["solver"]["effectors"]),
            "names": len(payload["names"]), "file": str(dest)}
