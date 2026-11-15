"""Wuthering Waves' pose-space correctives for the FBX sidecar: the skeleton's ``RBFDriver`` objects and the
animation blueprint's ``AnimNode_PoseDriver`` nodes (IslandLife §7.1511, §7.1514; ``docs/sidecar_rbf.md``).

**The skeleton's drivers** (``<model>_Skeleton``, property ``RBFDrivers``) are the developer's own class
``RBFDriver``: each names a driver bone and the bone it hangs on (``BaseBoneName``), the attributes it reads
(``TranslationX/Y/Z``), the morph targets it drives, its targets (``DriveSettings``: an input value per attribute
and a weight per morph) and a ``WeightModifier``.  The class holds no kernel, radius or normalisation.  The
skeleton's ``Precomputations`` names a ``RBFPrecomputations`` asset (one per body type, ``<type>/RBFConfig``)
holding, per driver name, the solved weights on a grid of directions: ``LookupAngleXY`` × ``LookupAngleZ`` every
``Interval`` degrees up to ``RotationLimitY`` / ``RotationLimitZ`` (``RotationLimitX`` is 0: no twist axis),
``PrecomputeRadius`` (the rest input's length) and ``PreWeights`` indexed by ``WeightLookupTable``.

What the data shows, measured on 12 drivers of three body types (FemaleM, FemaleMS, FemaleXL; §7.1514):

* the input is the driver bone's translation in its base bone's rest frame as the base bone turns: the first
  target is the driver's reference-pose local translation to the float (``DriveSettings[0]`` = 8.503053 cm =
  ``Bip001LUpperArm``'s local X in ``Bip001LClavicle``), and every target lies on the sphere of that radius;
* the grid's node ``(i, j)`` is that translation pointing at azimuth ``atan2(y, x)`` = ``-RotationLimitY + i·Interval``
  and elevation ``asin(z / r)`` = ``-RotationLimitZ + j·Interval`` (the stored lookup angles are those values
  measured back, to 0.03°);
* the table is what :func:`solve` gives at every node to 6e-7: an interpolative RBF over the targets, each
  attribute divided by its column's L2 norm over the targets (a column whose norm is under :data:`ZERO_COLUMN` is
  divided by 1), the kernel ``exp(-d / w²)`` of the Euclidean distance with ``w`` the mean of the full N×N
  distance matrix, the outputs clipped to [0, 1] (unclipped the error is 2.6e-2; a Gaussian, inverse multiquadric
  or normalised additive solver misses by 0.1–0.9).  One table (FemaleMS ``weightClavicle_R``) does not match
  Katixiya's own targets (error 0.79: its peak sits at azimuth −8°) — the sidecar says so per driver.

What the class does with ``WeightModifier`` is in the game's code, not its data: the record is written as it is.

**The blueprint's pose drivers** are Unreal's ``AnimNode_PoseDriver`` (``RBFParams`` of ``FRBFParams``,
``PoseTargets`` with one rotation per source bone, a per-target ``TargetScale``, ``DrivePoses`` of a
``PoseAsset``), with the developer's offset type ``KuroCachedBone``.  Every target's rotation is the pose asset's
additive rotation of the source bone for that pose seen in the bone's own reference frame (``ref⁻¹·A·ref``, to
1e-5° on all six of Chun's moved targets; the additive itself differs by 3.3° where the turn is not about the
reference's own axis): the input is the source bone's turn from its reference pose in its own frame.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .kawaii import rotator_quat
from .source_scene import C

SYSTEM_DRIVER = "kuro_rbf_driver"
SYSTEM_POSE = "unreal_pose_driver"
#: Unreal's centimetres to the FBX's metres
UNIT = 0.01
#: a target column with an L2 norm under this (cm) is divided by 1: the FemaleM clavicle drivers' Y column
#: (9.7e-6 and 4.7e-6 cm, float noise of a zero) only reproduces the game's table divided by 1
ZERO_COLUMN = 1e-3
#: how far (cm) the first target may be from the driver bone's reference-pose translation in the scene
REST_CM = 1e-3
#: how far (degrees) a base bone's scene rest frame may turn from the game's mirrored one
FRAME_DEG = 1e-3
#: the attributes a driver may read, and their axis
ATTRIBUTES = {"ERBFDriverAttribute::TranslationX": 0, "ERBFDriverAttribute::TranslationY": 1,
              "ERBFDriverAttribute::TranslationZ": 2}
NODE = "AnimGraphNode_PoseDriver"
SPACE = ("vectors: the base bone's FBX rest frame (as the FBX binds it, no scale), metres; rotations: quaternions "
         "x, y, z, w in the FBX bone's own rest frame; game_*: Unreal's own values and axes (left-handed, cm), as "
         "read. FBX = diag(1, -1, 1) · game / 100 for a vector, (-x, y, -z, w) for a quaternion "
         "(source_scene.native_world mirrors Y)")


class RbfError(ValueError):
    """A driver reads in a way this reader does not, or names a bone the model does not have."""


# -- the skeleton's drivers -----------------------------------------------------------------------------------

def _skeleton(package: list) -> dict:
    found = [entry for entry in package if entry.get("Type") == "Skeleton"]
    if len(found) != 1:
        raise RbfError(f"{len(found)} Skeleton objects in the package")
    return found[0]


def precomputation_path(package: list) -> str | None:
    """``Client/Content/…/<type>_Skeleton_RBFPrecomputation.uasset`` the skeleton names, or None."""
    ref = (_skeleton(package).get("Properties") or {}).get("Precomputations") or {}
    path = ref.get("ObjectPath") if isinstance(ref, dict) else None
    return path.rsplit(".", 1)[0] + ".uasset" if path else None


def skeleton_drivers(package: list) -> list[dict]:
    """The skeleton's ``RBFDriver`` objects in its ``RBFDrivers`` order, each as plain fields."""
    objects = {entry.get("Name"): entry for entry in package if entry.get("Type") == "RBFDriver"}
    order = [str(ref.get("ObjectName", "")).rsplit(":", 1)[-1].rstrip("'")
             for ref in (_skeleton(package).get("Properties") or {}).get("RBFDrivers") or ()]
    if sorted(order) != sorted(objects):
        raise RbfError(f"the skeleton lists drivers {order}, the package holds {sorted(objects)}")
    out = []
    for name in order:
        p = objects[name].get("Properties") or {}
        attributes = list(p.get("DriverAttributes") or ())
        unknown = [a for a in attributes if a not in ATTRIBUTES]
        if unknown:
            raise RbfError(f"{p.get('DriverName')}: attributes {unknown} are not read here")
        morphs = list(p.get("DrivenMorphTargets") or ())
        targets = []
        for number, setting in enumerate(p.get("DriveSettings") or ()):
            values = {pose["Attribute"]: float(pose["Value"]) for pose in setting.get("DriverPoses") or ()}
            if sorted(values) != sorted(attributes):
                raise RbfError(f"{p.get('DriverName')} target {number}: attributes {sorted(values)}")
            weights = {w["Name"]: float(w["Value"]) for w in setting.get("DrivenWeights") or ()}
            strangers = sorted(set(weights) - set(morphs))
            if strangers:
                raise RbfError(f"{p.get('DriverName')} target {number}: weights for {strangers}, not driven")
            targets.append({"input": [values[a] for a in attributes],
                            "weights": {m: weights.get(m, 0.0) for m in morphs}})
        out.append({"object": name, "name": p.get("DriverName"), "driver_bone": p.get("DriverBoneName"),
                    "base_bone": p.get("BaseBoneName"), "attributes": attributes, "morphs": morphs,
                    "targets": targets, "weight_modifier": p.get("WeightModifier")})
    return out


def tables(package: list) -> dict:
    """``{driver name: table}`` of a ``RBFPrecomputations`` package, the weights as ``[xy][z][morph]``."""
    found = [entry for entry in package if entry.get("Type") == "RBFPrecomputations"]
    if len(found) != 1:
        raise RbfError(f"{len(found)} RBFPrecomputations objects in the package")
    out = {}
    for pair in (found[0].get("Properties") or {}).get("Precomputations") or ():
        value = pair["Value"]
        xy, z = list(value["LookupAngleXY"]), list(value["LookupAngleZ"])
        interval = float(value["Interval"])
        for axis, limit, angles in (("XY", "RotationLimitY", xy), ("Z", "RotationLimitZ", z)):
            expected = -float(value[limit]) + interval * np.arange(len(angles))
            if len(angles) != round(2 * float(value[limit]) / interval) + 1 \
                    or np.abs(np.asarray(angles) - expected).max() > 0.05:
                raise RbfError(f"{pair['Key']}: LookupAngle{axis} is not -{limit}..{limit} every {interval}")
        if float(value.get("RotationLimitX", 0.0)) != 0.0:
            raise RbfError(f"{pair['Key']}: a twist range (RotationLimitX) is not read here")
        for entry in value["WeightLookupTable"]:
            if entry["Value"] != entry["Key"]["IndexXY"] * len(z) + entry["Key"]["IndexZ"]:
                raise RbfError(f"{pair['Key']}: WeightLookupTable is not row-major over (XY, Z)")
        if len(value["WeightLookupTable"]) != len(xy) * len(z) or len(value["PreWeights"]) != len(xy) * len(z):
            raise RbfError(f"{pair['Key']}: {len(value['PreWeights'])} weights for a {len(xy)}×{len(z)} grid")
        weights = np.asarray([row["Data"] for row in value["PreWeights"]], dtype=np.float64)
        out[pair["Key"]] = {"interval": interval, "limit_xy": float(value["RotationLimitY"]),
                            "limit_z": float(value["RotationLimitZ"]), "radius": float(value["PrecomputeRadius"]),
                            "lookup_xy": xy, "lookup_z": z,
                            "weights": weights.reshape(len(xy), len(z), -1)}
    return out


def directions(table: dict) -> np.ndarray:
    """``[xy][z][3]``: the input each grid node stands for (game axes, cm)."""
    xy = np.radians(-table["limit_xy"] + table["interval"] * np.arange(len(table["lookup_xy"])))
    z = np.radians(-table["limit_z"] + table["interval"] * np.arange(len(table["lookup_z"])))
    a, e = np.meshgrid(xy, z, indexing="ij")
    return table["radius"] * np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)


def solve(inputs, weights, points) -> np.ndarray:
    """The weights the measured solver gives at *points* (``[..., attributes]``) from targets *inputs* × *weights*.

    Each attribute is divided by its column's L2 norm over the targets (1 when under :data:`ZERO_COLUMN`); the
    kernel is ``exp(-d / w²)`` of the Euclidean distance, ``w`` the mean of the full N×N target distance matrix;
    the coefficients interpolate the targets exactly; the result is clipped to [0, 1]."""
    inputs, weights = np.asarray(inputs, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    norm = np.linalg.norm(inputs, axis=0)
    norm = np.where(norm < ZERO_COLUMN, 1.0, norm)
    unit = inputs / norm
    distance = np.linalg.norm(unit[:, None] - unit[None], axis=-1)
    width = float(distance.mean()) ** 2
    coefficients = np.linalg.solve(np.exp(-distance / width), weights)
    there = np.asarray(points, dtype=np.float64) / norm
    return np.clip(np.exp(-np.linalg.norm(there[..., None, :] - unit, axis=-1) / width) @ coefficients, 0.0, 1.0)


def solver_statement(driver: dict) -> dict:
    inputs = np.asarray([t["input"] for t in driver["targets"]], dtype=np.float64)
    norm = np.linalg.norm(inputs, axis=0)
    divide = np.where(norm < ZERO_COLUMN, 1.0, norm)
    unit = inputs / divide
    distance = np.linalg.norm(unit[:, None] - unit[None], axis=-1)
    return {"kind": "interpolative", "kernel": "exp(-d / width^2)",
            "distance": "Euclidean over the attributes, each divided by its column's L2 norm over the targets",
            "column_norm_game_cm": [round(float(v), 7) for v in divide],
            "zero_column_below_cm": ZERO_COLUMN, "width": round(float(distance.mean()), 9),
            "width_rule": "mean of the full N×N target distance matrix (diagonal included)",
            "clip": [0.0, 1.0], "basis": "measured: reproduces the game's precomputed table (table.max_error)"}


# -- the blueprint's pose drivers ----------------------------------------------------------------------------

def pose_driver_nodes(package: list) -> list[tuple[str, dict]]:
    """``[(node name, properties)]`` of every ``AnimNode_PoseDriver`` on the blueprint's class default object."""
    defaults = [entry for entry in package if str(entry.get("Name", "")).startswith("Default__")]
    if len(defaults) != 1:
        raise RbfError(f"{len(defaults)} class default objects in the blueprint")
    properties = defaults[0].get("Properties") or {}
    return [(name, value) for name, value in properties.items() if name.startswith(NODE) and isinstance(value, dict)]


def pose_asset(package: list, bones: list[str] | None) -> dict:
    """``{"additive", "poses": {pose: {bone: {"rotation": game xyzw, "translation": game cm}}}}`` of a
    ``PoseAsset`` package, for *bones* (every track when None); a bone a pose does not key is at rest."""
    found = [entry for entry in package if entry.get("Type") == "PoseAsset"]
    if len(found) != 1:
        raise RbfError(f"{len(found)} PoseAsset objects in the package")
    properties = found[0].get("Properties") or {}
    container = properties.get("PoseContainer") or {}
    tracks = list(container.get("Tracks") or ())
    names = [row["DisplayName"] for row in container.get("PoseNames") or ()]
    poses = {}
    for name, pose in zip(names, container.get("Poses") or (), strict=True):
        buffer = {int(row["Key"]): int(row["Value"]) for row in pose.get("TrackToBufferIndex") or ()}
        held = {}
        for track, slot in buffer.items():
            bone = tracks[track]
            if bones is not None and bone not in bones:
                continue
            entry = pose["LocalSpacePose"][slot]
            held[bone] = {"rotation": [float(entry["Rotation"][k]) for k in "XYZW"],
                          "translation": [float(entry["Translation"][k]) for k in "XYZ"],
                          "scale": [float(entry["Scale3D"][k]) for k in "XYZ"]}
        poses[name] = held
    return {"additive": str(properties.get("bAdditivePose", "false")).lower() == "true", "poses": poses}


# -- frames ---------------------------------------------------------------------------------------------------

def fbx_vector(game) -> list:
    return [round(float(v), 9) for v in C @ np.asarray(game, dtype=np.float64) * UNIT]


def fbx_quat(game) -> list:
    """A rotation in a bone's own frame, from Unreal's axes to the FBX's (the frames are mirrored: ``C·R·C``)."""
    x, y, z, w = (float(v) for v in game)
    return [round(-x, 9), round(y, 9), round(-z, 9), round(w, 9)]


def _scene_local(rest: np.ndarray, parent: list[int], index: int) -> np.ndarray:
    return np.linalg.solve(rest[parent[index]], rest[index]) if parent[index] >= 0 else rest[index]


def _frame_check(document: dict, rest: np.ndarray, bone: str) -> float:
    """Degrees between a bone's scene rest frame (relative to its parent) and the game's, mirrored."""
    names = [b["name"] for b in document["bones"]]
    parent = [b["parent"] for b in document["bones"]]
    source = {b["name"]: b for b in document["source_bones"]}
    i = names.index(bone)
    if parent[i] < 0:
        return 0.0
    game = Rotation.from_quat(source[bone]["rotation_xyzw"]).as_matrix()
    scene = _scene_local(rest, parent, i)[:3, :3]
    scene = scene / np.cbrt(np.linalg.det(scene))
    return float(np.degrees(Rotation.from_matrix(scene.T @ (C @ game @ C)).magnitude()))


def _parts_with(document: dict, shape: str) -> list[str]:
    return [part["name"] for part in document["parts"] if shape in (part.get("shapes") or ())]


def driver_row(driver: dict, table: dict | None, table_package: str | None, document: dict,
               rest: np.ndarray, number: dict) -> dict:
    """One skeleton driver as a sidecar row (``docs/sidecar_rbf.md``)."""
    names = [b["name"] for b in document["bones"]]
    parent = [b["parent"] for b in document["bones"]]
    for role in ("driver_bone", "base_bone"):
        if driver[role] not in names:
            raise RbfError(f"{driver['name']}: {role} {driver[role]!r} is not in the model")
    d, b = names.index(driver["driver_bone"]), names.index(driver["base_bone"])
    if parent[d] != b:
        raise RbfError(f"{driver['name']}: {driver['driver_bone']} does not hang on {driver['base_bone']}")
    axes = [ATTRIBUTES[a] for a in driver["attributes"]]
    if axes != [0, 1, 2]:
        raise RbfError(f"{driver['name']}: attributes {driver['attributes']} are not X, Y, Z in order")
    scene_rest = _scene_local(rest, parent, d)[:3, 3]
    first = np.asarray(driver["targets"][0]["input"], dtype=np.float64)
    rest_gap = float(np.linalg.norm(C @ (scene_rest / UNIT) - first))
    frame_gap = _frame_check(document, rest, driver["base_bone"])
    inputs = np.asarray([t["input"] for t in driver["targets"]], dtype=np.float64)
    weights = np.asarray([[t["weights"][m] for m in driver["morphs"]] for t in driver["targets"]])
    radius = float(np.linalg.norm(first))
    row = {
        "id": driver["name"], "system": SYSTEM_DRIVER, "object": driver["object"],
        "driver_bone": driver["driver_bone"], "base_bone": driver["base_bone"],
        "measure": {"quantity": "translation", "attributes": driver["attributes"],
                    "rule": ("the driver bone's translation in the base bone's rest frame turned by the base bone's "
                             "turn from rest: v = turn(base) · rest translation of the driver bone; only its "
                             "direction changes, its length is the rest translation's")},
        "rest_translation": {"game_cm": [float(v) for v in first], "fbx_m": fbx_vector(first),
                             "scene_fbx_m": [round(float(v), 9) for v in scene_rest],
                             "gap_cm": round(rest_gap, 7), "base_frame_gap_deg": round(frame_gap, 7)},
        "outputs": {"kind": "shape_keys", "names": driver["morphs"],
                    "parts": {m: _parts_with(document, m) for m in driver["morphs"]}},
        "targets": [{"game_cm": [float(v) for v in t["input"]], "fbx_m": fbx_vector(t["input"]),
                     "direction_fbx": [round(float(v), 9) for v in C @ np.asarray(t["input"]) / radius],
                     "weights": t["weights"]} for t in driver["targets"]],
        "weight_modifier": _modifier(driver.get("weight_modifier"), number),
        "solver": solver_statement(driver),
    }
    problems = []
    if rest_gap > REST_CM:
        problems.append(f"the first target is {rest_gap:.4f} cm from {driver['driver_bone']}'s rest translation")
    if frame_gap > FRAME_DEG:
        problems.append(f"{driver['base_bone']}'s scene rest frame is {frame_gap:.4f}° from the game's")
    if table is None:
        row["table"] = {"status": None, "reason": (f"{table_package or 'the skeleton names no precomputation'} "
                                                   f"holds no table for {driver['name']}")}
    else:
        error = float(np.abs(solve(inputs, weights, directions(table)) - table["weights"]).max())
        row["table"] = {
            "status": "read", "package": table_package,
            "axes": ("[xy][z][morph]: node (i, j) is the input of length radius_cm at azimuth atan2(y, x) = "
                     "-limit_xy + i·interval and elevation asin(z / r) = -limit_z + j·interval, in the game's "
                     "base bone frame (y_game = -y_fbx)"),
            "interval_deg": table["interval"], "limit_xy_deg": table["limit_xy"], "limit_z_deg": table["limit_z"],
            "radius_cm": table["radius"], "morphs": driver["morphs"],
            "weights": np.round(table["weights"], 7).tolist(),
            "max_error": round(error, 9),
            "matches_targets": error < 1e-5}
    if problems:
        row["problems"] = problems
    return row


def _modifier(record: dict | None, number: dict) -> dict | None:
    """The ``WeightModifier`` as stored, its driver index also named; what the class does with it is not data."""
    if not record:
        return None
    keys = (((record.get("FactorCurve") or {}).get("EditorCurveData") or {}).get("Keys") or ())
    index = record.get("DstDriverIndex")
    return {"source_weight": record.get("SrcWeight"), "target_driver_index": index,
            "target_driver": number.get(index), "target_weight": record.get("DstWeight"),
            "factor_curve": [{"time": float(k["Time"]), "value": float(k["Value"]),
                              "interp": k.get("InterpMode")} for k in keys],
            "semantics": None,
            "reason": ("how RBFDriver applies the modifier is in the game's code, which is not read: the record "
                       "is written as stored")}


def pose_driver_row(name: str, node: dict, asset: dict | None, asset_package: str | None, document: dict) -> dict:
    """One ``AnimNode_PoseDriver`` as a sidecar row (``docs/sidecar_rbf.md``)."""
    names = {b["name"] for b in document["bones"]}
    sources = [s["BoneName"] for s in node.get("SourceBones") or ()]
    driven = [s["BoneName"] for s in node.get("OnlyDriveBones") or ()]
    for bone in [*sources, *driven]:
        if bone not in names:
            raise RbfError(f"{name}: bone {bone!r} is not in the model")
    if node.get("DriveSource") != "EPoseDriverSource::Rotation":
        raise RbfError(f"{name}: drive source {node.get('DriveSource')} is not read here")
    if node.get("DriveOutput") != "EPoseDriverOutput::DrivePoses":
        raise RbfError(f"{name}: output {node.get('DriveOutput')} is not read here")
    targets = []
    for target in node.get("PoseTargets") or ():
        transforms = target.get("BoneTransforms") or ()
        if len(transforms) != len(sources):
            raise RbfError(f"{name} {target.get('DrivenName')}: {len(transforms)} transforms for {len(sources)} bones")
        quats = [rotator_quat(t["TargetRotation"]) for t in transforms]
        targets.append({"pose": target.get("DrivenName"), "scale": float(target.get("TargetScale", 1.0)),
                        "rotation_game": [t["TargetRotation"] for t in transforms],
                        "quat_game": [[round(float(v), 9) for v in q] for q in quats],
                        "quat_fbx": [fbx_quat(q) for q in quats],
                        "distance_method": target.get("DistanceMethod"), "function": target.get("FunctionType"),
                        "custom_curve": (target.get("CustomCurve") or {}).get("Keys") if target.get("bApplyCustomCurve")
                        else None,
                        "hidden": bool(target.get("bIsHidden", False))})
    row = {"id": name, "system": SYSTEM_POSE, "source_bones": sources,
           "input": ("each source bone's turn from its reference pose in its own frame (ref⁻¹ · posed): the "
                     "offset type is the developer's (" + str(node.get("PoseDriverOffsetType")) + "), and every "
                     "target equals ref⁻¹ · A · ref of the pose asset's additive rotation A of that bone"),
           "eval_space_bone": (node.get("EvalSpaceBone") or {}).get("BoneName"),
           "offset_type": node.get("PoseDriverOffsetType"),
           "rbf_params": node.get("RBFParams"), "targets": targets,
           "only_drive_selected": bool(node.get("bOnlyDriveSelectedBones")), "driven_bones": driven,
           "output": node.get("DriveOutput"), "lod_threshold": node.get("LODThreshold")}
    if asset is None:
        row["pose_asset"] = {"status": None, "package": asset_package, "reason": "the pose asset could not be read"}
        return row
    poses = {}
    for pose, bones in asset["poses"].items():
        poses[pose] = {bone: {"quat_game": entry["rotation"], "quat_fbx": fbx_quat(entry["rotation"]),
                              "translation_game_cm": entry["translation"],
                              "translation_fbx_m": fbx_vector(entry["translation"]), "scale_game": entry["scale"]}
                       for bone, entry in bones.items() if not driven or bone in driven or bone in sources}
    row["pose_asset"] = {"status": "read", "package": asset_package, "additive": asset["additive"],
                         "space": ("additive local transforms: posed = A · reference (parent side, Unreal's "
                                   "FTransform accumulate); quat_fbx is A in the FBX's mirrored axes"),
                         "poses": poses}
    return row


def section(document: dict, rest, skeleton_package: str, drivers: list[dict], table_package: str | None,
            found_tables: dict | None, blueprint: str | None, nodes: list[tuple[str, dict]],
            assets: dict[str, dict | None]) -> dict:
    """The sidecar's ``rbf`` section (``docs/sidecar_rbf.md``) of an assembled model."""
    rest = np.asarray(rest, dtype=np.float64)
    number = {i: d["name"] for i, d in enumerate(drivers)}
    rows = [driver_row(d, (found_tables or {}).get(d["name"]), table_package, document, rest, number)
            for d in drivers]
    pose_rows = []
    for name, node in nodes:
        path = ((node.get("PoseAsset") or {}).get("ObjectPath") or "").rsplit(".", 1)[0]
        package = path + ".uasset" if path else None
        pose_rows.append(pose_driver_row(name, node, assets.get(package) if package else None, package, document))
    basis = [f"{skeleton_package}: the Skeleton's RBFDrivers"]
    if table_package:
        basis.append(f"{table_package}: RBFPrecomputations")
    if blueprint:
        basis.append(f"{blueprint}: the class default object's {NODE} nodes")
    return {"format": 1, "status": "read", "space": SPACE, "basis": basis, "drivers": rows,
            "pose_drivers": pose_rows}
