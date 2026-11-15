"""Wuthering Waves' swing chains for the FBX sidecar: the ``AnimNode_KawaiiPhysicsGroup`` nodes of a model's animation blueprint.

The game moves hair, skirts, ribbons and the chest with KawaiiPhysics (the
open-source plugin, version 1.6.1, modified by the developer), as nodes of
each model's own animation blueprint ``ABP_<model>_PC`` — the one the model
table's animation-blueprint column names (IslandLife §7.1217, §7.1511).  The
blueprint's class default object holds every node's properties; the reader's
``json`` mode writes them.  What is written here (``docs/sidecar_physics.md``):

* each node's roots, excluded bones and the chain under them, as 1.6.1's
  ``AddModifyBone`` walks it (every descendant of a root; an excluded bone
  ends its branch);
* the node's own properties, unchanged (centimetres, Unreal's axes), and in
  ``resolved`` what a reader needs in the FBX's units: the particle radius,
  the dummy end bone's length and the bone's forward axis in the FBX bone's
  own frame, gravity in the FBX's world, the teleport limits, the alpha
  (``None`` when an animation curve drives it: our animations do not carry
  the game's curves), and the keys of the ``CurveFloat`` assets the four
  parameter curves name;
* each limit (sphere, capsule, plane) placed at the reference pose the way
  1.6.1's ``UpdateSphericalLimits`` / ``UpdateCapsuleLimits`` /
  ``UpdatePlanerLimits`` place it, said in its driving bone's FBX frame.

Fields of the developer's version that 1.6.1 does not have are listed in
``unread`` with what was seen; nothing is made of them.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .source_scene import C

SYSTEM = "kawaii_physics"
NODE = "AnimGraphNode_KawaiiPhysicsGroup"
CURVES = ("DampingCurve", "WorldDampingLocationCurve", "WorldDampingRotationCurve", "StiffnessCurve",
          "RadiusCurve", "LimitAngleCurve", "HalfWidthCurve")
SPACE = ("geometry: the FBX bone's rest frame (as the FBX binds it, no scale), metres; world directions: the FBX's "
         "world, Y up, metres; params: the game's own units and conventions")
#: Unreal's centimetres to the FBX's metres
UNIT = 0.01
#: the FBX's world from the scene's (Z up): ``(x, y, z) → (x, z, −y)`` (``interchange.glb.TO_Y_UP``)
TO_Y_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
AXES = {"X_Positive": (0, 1.0), "X_Negative": (0, -1.0), "Y_Positive": (1, 1.0), "Y_Negative": (1, -1.0),
        "Z_Positive": (2, 1.0), "Z_Negative": (2, -1.0)}
#: the developer's fields (not in 1.6.1) and their values when they are at rest
UNREAD_DEFAULTS = {"MultFixParamsArray": [], "bUseInMultFix": False, "BindInfos": [], "bBindLockRotation": False,
                   "IgnoreMechanicsLevel": 0, "bNewIgnoreMechanics": False, "WindCurve": None,
                   "LODThreshold": -1, "HalfWidthCurve": None}


class KawaiiError(ValueError):
    """The blueprint names a bone the model does not have, or a field reads in a way this reader does not."""


def blueprint_package(row: dict, column: str) -> str | None:
    """``Client/Content/…/ABP_….uasset`` of a model-table row's animation-blueprint column."""
    value = row.get(column)
    path = (value or {}).get("AssetPathName") if isinstance(value, dict) else value
    if not path or path == "None":
        return None
    path = path.split(".")[0]
    return "Client/Content/" + path.removeprefix("/Game/") + ".uasset"


#: the model table's animation-blueprint column (its struct's field name)
BLUEPRINT_COLUMN = "动画蓝图_171_0392CD7E4090222C990267BFB07ACCFA"


def nodes_of(package: list) -> list[tuple[str, dict]]:
    """``[(node name, properties)]`` of every KawaiiPhysics group node on the blueprint's class default object."""
    defaults = [entry for entry in package if str(entry.get("Name", "")).startswith("Default__")]
    if len(defaults) != 1:
        raise KawaiiError(f"{len(defaults)} class default objects in the blueprint")
    properties = defaults[0].get("Properties") or {}
    return [(name, value) for name, value in properties.items()
            if name.startswith(NODE) and isinstance(value, dict)]


def curve_packages(nodes: list[tuple[str, dict]]) -> dict[str, str]:
    """``{object path: package}`` of every ``CurveFloat`` the nodes' parameter curves name."""
    out = {}
    for _name, node in nodes:
        for key in CURVES:
            ref = node.get(key)
            if isinstance(ref, dict) and ref.get("ObjectPath"):
                out[ref["ObjectPath"]] = ref["ObjectPath"].rsplit(".", 1)[0] + ".uasset"
    return out


def curve_keys(package: list) -> list[dict]:
    """The keys of a ``CurveFloat`` package (``FloatCurve.Keys``), each as stored."""
    curves = [entry for entry in package if entry.get("Type") == "CurveFloat"]
    if len(curves) != 1:
        raise KawaiiError(f"{len(curves)} CurveFloat objects in the package")
    keys = ((curves[0].get("Properties") or {}).get("FloatCurve") or {}).get("Keys")
    if keys is None:
        raise KawaiiError("the CurveFloat holds no FloatCurve keys")
    return [{key: entry.get(key) for key in ("Time", "Value", "InterpMode", "ArriveTangent", "LeaveTangent",
                                             "TangentWeightMode")} for entry in keys]


def rotator_quat(rotator: dict) -> np.ndarray:
    """Unreal's ``FRotator::Quaternion`` (degrees; x, y, z, w), as the engine computes it."""
    half = math.pi / 360.0
    sp, cp = math.sin(rotator["Pitch"] * half), math.cos(rotator["Pitch"] * half)
    sy, cy = math.sin(rotator["Yaw"] * half), math.cos(rotator["Yaw"] * half)
    sr, cr = math.sin(rotator["Roll"] * half), math.cos(rotator["Roll"] * half)
    return np.array([cr * sp * sy - sr * cp * cy, -cr * sp * cy - sr * cp * sy,
                     cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy])


def component_space(source_bones: list[dict]) -> np.ndarray:
    """Each bone's reference-pose transform in the mesh's component space (Unreal, centimetres, scale kept)."""
    out = []
    for index, bone in enumerate(source_bones):
        local = np.eye(4)
        local[:3, :3] = Rotation.from_quat(bone["rotation_xyzw"]).as_matrix() @ np.diag(bone["scale"])
        local[:3, 3] = bone["translation"]
        parent = bone["parent"]
        if not -1 <= parent < index:
            raise KawaiiError(f"bone {bone['name']} comes before its parent")
        out.append(out[parent] @ local if parent >= 0 else local)
    return np.asarray(out)


def _rigid(matrix: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(matrix[:3, :3])
    return u @ vt


def place_limit(limit: dict, cs: np.ndarray, index: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(location, rotation)`` of a limit at the reference pose in component space, as 1.6.1 places it.

    ``UpdateCapsuleLimits`` (and the sphere's and plane's) turns the driving
    bone's component-space transform into ``BCS_BoneSpace`` — relative to that
    bone's own component-space transform, so the identity — pre-multiplies its
    rotation by ``OffsetRotation``, adds ``OffsetLocation`` to its translation
    and turns it back: the limit rides the driving bone, offset in its frame
    (``FAnimationRuntime::ConvertCSTransformToBoneSpace``).  A limit without a
    driving bone stands at its offset.
    """
    offset = Rotation.from_quat(rotator_quat(limit["OffsetRotation"])).as_matrix()
    move = np.array([limit["OffsetLocation"][k] for k in "XYZ"], dtype=np.float64)
    name = (limit.get("DrivingBone") or {}).get("BoneName")
    if not name or name == "None":
        return move, offset
    if name not in index:
        raise KawaiiError(f"limit driven by {name!r}, which the model does not have")
    local = np.eye(4)
    local[:3, :3] = offset
    local[:3, 3] = move
    placed = cs[index[name]] @ local
    return placed[:3, 3], _rigid(placed)


def _scene(point_cm: np.ndarray) -> np.ndarray:
    """A component-space point (Unreal, cm) in the scene's space (right-handed, Z up, metres)."""
    return C @ point_cm * UNIT


def _local(rest: np.ndarray, point: np.ndarray) -> list:
    return [round(float(v), 7) for v in np.linalg.solve(rest, np.append(point, 1.0))[:3]]


def limits(node_name: str, node: dict, cs, parent, index, rest, names, humanoid, simulated) -> list:
    """The node's spheres, capsules and planes as sidecar collider rows."""
    rows = []
    groups = (("SphericalLimits", "sphere"), ("SphericalLimitsData", "sphere"), ("CapsuleLimits", "capsule"),
              ("CapsuleLimitsData", "capsule"), ("PlanarLimits", "plane"), ("PlanarLimitsData", "plane"))
    human_of = {bone: human for human, bone in (humanoid or {}).items()}
    for field, kind in groups:
        for number, limit in enumerate(node.get(field) or ()):
            driving = (limit.get("DrivingBone") or {}).get("BoneName")
            if not driving or driving == "None" or driving not in index:
                raise KawaiiError(f"{node_name}.{field}[{number}]: driving bone {driving!r} is not in the model")
            location, turn = place_limit(limit, cs, index)
            axis = turn[:, 2]
            bone = index[driving]
            frame = rest[bone]
            radius = float(limit.get("Radius", 0.0)) * UNIT
            if kind == "capsule":
                half = axis * float(limit["Length"]) * 0.5
                start, end = _local(frame, _scene(location + half)), _local(frame, _scene(location - half))
            elif kind == "sphere":
                start = end = _local(frame, _scene(location))
            else:
                start = _local(frame, _scene(location))
                normal = np.linalg.solve(frame[:3, :3], C @ axis)
                end = [round(float(v), 7) for v in normal / np.linalg.norm(normal)]
                radius = 0.0
            host, walk = None, bone
            while walk >= 0 and host is None:
                host = human_of.get(names[walk])
                walk = parent[walk]
            rows.append({"id": f"{node_name}/{field}/{number}", "kind": kind, "bone": driving,
                         "start": start, "end": end, "radius": [radius, radius],
                         "inside": kind == "sphere" and limit.get("LimitType") == "ESphericalLimitType::Inner",
                         "host": {"humanoid": host, "on_chain": driving in simulated},
                         "source": {"class": field, "node": driving,
                                    "fields": {k: v for k, v in limit.items() if k not in ("Location", "Rotation")}}})
    return rows


def _chain(roots: list[str], exclude: set[str], index: dict, children: list[list[int]], names) -> list[str]:
    """1.6.1's ``AddModifyBone`` from each root: depth first, an excluded bone ending its branch."""
    out = []

    def walk(bone):
        if names[bone] in exclude:
            return
        out.append(names[bone])
        for child in children[bone]:
            walk(child)

    for root in roots:
        if root not in index:
            raise KawaiiError(f"root bone {root!r} is not in the model")
        walk(index[root])
    return out


def _alpha(node: dict):
    """The node's alpha when it is a constant (``EAnimAlphaInputType::Float``); None when a curve or a bool drives it."""
    if node.get("AlphaInputType") != "EAnimAlphaInputType::Float":
        return None
    clamp = node.get("AlphaScaleBiasClamp") or {}
    if any(clamp.get(key) for key in ("bMapRange", "bClampResult", "bInterpResult")):
        return None
    bias = node.get("AlphaScaleBias") or {"Scale": 1.0, "Bias": 0.0}
    return float(node["Alpha"]) * float(bias.get("Scale", 1.0)) + float(bias.get("Bias", 0.0))


def physics(document: dict, rest: np.ndarray, blueprint: str, nodes: list[tuple[str, dict]],
            curves: dict[str, list | None]) -> dict:
    """The sidecar's ``physics`` section (``docs/sidecar_physics.md``) of an assembled model.

    *document* is the case's ``source_document.json`` (its ``source_bones`` the primary's, in Unreal's units;
    its ``bones`` the FBX's, primary first), *rest* its rigid rest frames (``bone_rest``), *curves* each named
    ``CurveFloat`` object's keys (None when it could not be read).
    """
    source = document["source_bones"]
    names = [bone["name"] for bone in source]
    if [bone["name"] for bone in document["bones"][:len(names)]] != names:
        raise KawaiiError("the assembled skeleton does not start with the primary's bones")
    index = {name: i for i, name in enumerate(names)}
    parent = [bone["parent"] for bone in source]
    children = [[] for _ in names]
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)
    cs = component_space(source)
    rest = np.asarray(rest, dtype=np.float64)
    chains, simulated = [], set()
    for name, node in nodes:
        roots = [r["BoneName"] for r in node.get("RootBones") or () if r.get("BoneName") not in (None, "None")]
        exclude = {r["BoneName"] for r in node.get("ExcludeBones") or ()}
        bones = _chain(roots, exclude, index, children, names)
        simulated.update(b for b in bones if b not in roots)
        chains.append((name, node, roots, exclude, bones))
    rows, colliders, unread = [], [], {}
    for name, node, roots, exclude, bones in chains:
        found = limits(name, node, cs, parent, index, rest, names, document.get("humanoid") or {}, simulated)
        colliders += found
        axis, sign = AXES[node.get("BoneForwardAxis", "EBoneForwardAxis::X_Positive").split("::")[-1]]
        forward = np.zeros(3)
        forward[axis] = sign
        gravity = np.array([node["Gravity"][k] for k in "XYZ"], dtype=np.float64)
        named = {key: node.get(key) for key in CURVES if isinstance(node.get(key), dict)}
        settings = node.get("PhysicsSettings") or {}
        odd = {key: node.get(key) for key, rest_value in UNREAD_DEFAULTS.items()
               if key in node and node.get(key) != rest_value}
        if settings.get("Shape") not in (None, "EPhyCollisionShape::Spherical"):
            odd["PhysicsSettings.Shape"] = settings.get("Shape")
            odd["PhysicsSettings.HalfWidth"] = settings.get("HalfWidth")
        odd["KuroLODMask"] = node.get("KuroLODMask")
        if node.get("AlphaInputType") != "EAnimAlphaInputType::Float":
            odd["AlphaCurveName"] = node.get("AlphaCurveName")
        if node.get("bEnableWind"):
            odd["bEnableWind"] = True
        unread[name] = odd
        params = {key: copy.deepcopy(value) for key, value in node.items()
                  if key not in ("ModifyBones", "PreSkelCompTransform", "ComponentPose", "SphericalLimits",
                                 "CapsuleLimits", "PlanarLimits", "SphericalLimitsData", "CapsuleLimitsData",
                                 "PlanarLimitsData", "RootBones", "ExcludeBones")}
        rows.append({
            "id": name, "system": SYSTEM, "enabled": bool(node.get("bEnableSimulate", True)),
            "frame": {"bone": None, "node": "component", "local": np.eye(4).tolist()},
            "roots": roots, "exclude": sorted(exclude), "bones": bones, "states": None,
            "colliders": [row["id"] for row in found], "empty_slots": 0, "params": params,
            "resolved": {
                "radius_m": float(settings.get("Radius", 0.0)) * UNIT,
                "dummy": {"length_m": float(node.get("DummyBoneLength", 0.0)) * UNIT,
                          # the bone's axis in the scene's frame: the rest frames are Unreal's turned by C (C·R·C)
                          "forward": [float(v) for v in C @ forward]},
                "gravity_world": [round(float(v), 7) for v in TO_Y_UP @ _scene(gravity)],
                "teleport_distance_m": float(node.get("TeleportDistanceThreshold", 300.0)) * UNIT,
                "teleport_rotation_deg": float(node.get("TeleportRotationThreshold", 10.0)),
                "target_framerate": int(node.get("TargetFramerate", 60)),
                "alpha": _alpha(node),
                "curves": {key: {"object": ref.get("ObjectPath"), "keys": curves.get(ref.get("ObjectPath"))}
                           for key, ref in named.items()}}})
    return {"format": 1, "status": "read", "system": SYSTEM, "space": SPACE,
            "basis": f"{blueprint}: the class default object's {NODE} nodes (KawaiiPhysics 1.6.1, the developer's "
                     "version)",
            "chains": rows, "colliders": colliders, "unread": {k: v for k, v in unread.items() if v}}
