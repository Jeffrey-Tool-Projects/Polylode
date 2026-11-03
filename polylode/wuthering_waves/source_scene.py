"""A display model's original clothed scene, from the reader's mesh dumps, in the Blender interchange schema (§7.1121).

The game-file reader's ``mesh`` and ``metadata`` modes leave, per mesh, a
``<ident>.source.json.gz`` (bones, LOD 0, morphs and material slots, in
Unreal's left-handed Z-up centimetres), the textures as PNG with each
texture's colour space (``texture_metadata.json``) and each material as JSON.
:func:`prepare` turns one mesh into the arrays and scene document
``tools/ef_fbx_blender.py`` builds a Blender scene from (right-handed, Z up,
metres, every stream kept); :func:`assemble` adds a display model's other
parts — the UI child meshes on the primary's own skeleton, the decorations on
their sockets with their own bones.

Rest frames: Unreal keeps a scale in some rest transforms, which an armature
cannot.  :func:`bake_uniform` keeps every joint's world position and takes the
nearest rotation, and :func:`validate_skin_bound` proves that skinning every
vertex with the rigid frames lands within 0.05 mm of the native ones under a
turn of every bone; a model that does not stays unconverted.

Moved from the dev scripts of the 1121 extraction (``wuwa_prepare``,
``wuwa_rest_basis``, ``wuwa_assemble``) without changing what they compute.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

#: Unreal (left-handed, Z up) → right-handed Z up: mirror Y.
C = np.diag([1., -1., 1.])
#: millimetres a rigid rest frame may move a skinned vertex under the test turn
SKIN_BOUND_MM = .05


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- rest frames ----------------------------------------------------------------------------------------------

def bake_uniform(world, bones):
    """Rigid rest frames at the native joint positions; the proof, and both skin matrices under a test turn."""
    scales = np.cbrt(np.linalg.det(world[:, :3, :3]))
    if not np.all(np.abs(scales) > 1e-15):
        raise ValueError("a singular native rest scale needs its own adapter")
    rigid = world.copy()
    rigid[:, :3, :3] /= scales[:, None, None]
    orthogonality = np.einsum("nji,njk->nik", rigid[:, :3, :3], rigid[:, :3, :3])
    uniform = np.allclose(orthogonality, np.eye(3), atol=1e-10, rtol=0)
    u, _, vt = np.linalg.svd(rigid[:, :3, :3])
    rigid[:, :3, :3] = u @ vt
    if not np.all(np.linalg.det(rigid[:, :3, :3]) > 0):
        raise ValueError("a native rest frame is mirrored")
    native_pose, baked_pose = [], []
    for index, bone in enumerate(bones):
        parent = bone["parent"]
        axis = np.array([1. + index % 3, 1. + (index + 1) % 3, 1. + (index + 2) % 3])
        axis /= np.linalg.norm(axis)
        turn = np.eye(4)
        turn[:3, :3] = Rotation.from_rotvec(axis * np.deg2rad(25.)).as_matrix()
        native_local = np.linalg.solve(world[parent], world[index]) if parent >= 0 else world[index]
        baked_local = np.linalg.solve(rigid[parent], rigid[index]) if parent >= 0 else rigid[index]
        native_pose.append((native_pose[parent] if parent >= 0 else np.eye(4)) @ native_local @ turn)
        baked_pose.append((baked_pose[parent] if parent >= 0 else np.eye(4)) @ baked_local @ turn)
    native_skin = np.stack(native_pose) @ np.linalg.inv(world)
    baked_skin = np.stack(baked_pose) @ np.linalg.inv(rigid)
    error = float(np.abs(native_skin - baked_skin).max())
    if uniform and error >= 1e-9:
        raise ValueError(f"uniform rest frames changed the skin by {error}")
    proof = dict(method="signed rest scale retained in native metadata; rigid canonical axes and exact native world "
                        "joint positions; source-mesh rotation skinning bound required",
                 world_scales=scales.tolist(), all_bone_rotation_skin_matrix_error=error,
                 uniform_rest_scale_exact=bool(uniform), native_animation_translation_adapter_included=False)
    return rigid, proof, native_skin, baked_skin


def validate_skin_bound(proof, native_skin, baked_skin, points, joints, weights):
    """Raise unless every vertex skinned with both frame sets lands within :data:`SKIN_BOUND_MM`."""
    homogeneous = np.c_[points, np.ones(len(points))]
    share = weights / weights.sum(1)[:, None]
    result = []
    for matrices in (native_skin, baked_skin):
        posed = np.zeros((len(points), 3))
        for column in range(joints.shape[1]):
            posed += share[:, column, None] * np.einsum("nij,nj->ni", matrices[joints[:, column], :3], homogeneous)
        result.append(posed)
    error = float(np.linalg.norm(result[0] - result[1], axis=1).max() * 1000)
    if error > SKIN_BOUND_MM:
        raise ValueError(f"rigid rest frames move a vertex {error:.4f} mm (limit {SKIN_BOUND_MM})")
    proof.update(all_native_vertices_rotation_skin_error_mm=error, limit_mm=SKIN_BOUND_MM,
                 source_vertices_unchanged=True, joint_positions_unchanged=True)


def display_lengths(lengths, world, points, faces):
    """Bone display lengths no shorter than the mesh's edge spacing and the float32 endpoint uncertainty allows."""
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    distances = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    spacing = float(np.median(distances[distances > 0]))
    quantization = 2 * np.sqrt(3) * np.max(np.spacing(np.abs(world[:, :3, 3]).astype(np.float32)), axis=1)
    angular_budget_deg = .001
    minimum = np.maximum(spacing, quantization / np.sin(np.deg2rad(angular_budget_deg)))
    return np.maximum(lengths, minimum), dict(
        method="display tail length bounded by source spacing and float32 endpoint uncertainty; joint heads and "
               "rest rotations unchanged", source_median_edge_m=spacing, angular_budget_deg=angular_budget_deg,
        minimum_display_length_m=minimum.tolist())


# -- one mesh -------------------------------------------------------------------------------------------------

def native_world(bones: list[dict]) -> np.ndarray:
    result = []
    for index, bone in enumerate(bones):
        local = np.eye(4)
        local[:3, :3] = Rotation.from_quat(bone["rotation_xyzw"]).as_matrix() @ np.diag(bone["scale"])
        local[:3, 3] = bone["translation"]
        parent = bone["parent"]
        if not -1 <= parent < index:
            raise ValueError(f"bone {bone['name']} comes before its parent")
        result.append(result[parent] @ local if parent >= 0 else local)
    world = np.asarray(result)
    world[:, :3, :3] = np.einsum("ij,njk,kl->nil", C, world[:, :3, :3], C.T)
    world[:, :3, 3] = world[:, :3, 3] @ C.T * .01
    return world


def humanoid(names: set[str], bones: list[dict]) -> dict[str, str]:
    """The biped's human bones, each checked to be the child of the one before it."""
    result = {"Hips": "Bip001Pelvis", "Spine": "Bip001Spine", "Chest": "Bip001Spine1",
              "UpperChest": "Bip001Spine2", "Neck": "Bip001Neck", "Head": "Bip001Head"}
    for side, prefix in (("Left", "Bip001L"), ("Right", "Bip001R")):
        for role, suffix in (("Shoulder", "Clavicle"), ("UpperArm", "UpperArm"), ("LowerArm", "Forearm"),
                             ("Hand", "Hand"), ("UpperLeg", "Thigh"), ("LowerLeg", "Calf"), ("Foot", "Foot"),
                             ("Toes", "Toe0")):
            result[side + role] = prefix + suffix
    missing = set(result.values()) - names
    if missing:
        raise ValueError(f"not the native biped: {sorted(missing)} missing")
    lookup = {b["name"]: b for b in bones}
    for side in ("Left", "Right"):
        for start, end in (("UpperArm", "LowerArm"), ("LowerArm", "Hand"), ("UpperLeg", "LowerLeg"),
                           ("LowerLeg", "Foot"), ("Foot", "Toes")):
            parent = bones[lookup[result[side + end]]["parent"]]["name"]
            if parent != result[side + start]:
                raise ValueError(f"{result[side + end]} is under {parent}, not {result[side + start]}")
    return result


def material_record(native: dict, directory: Path, case: Path, used: bool) -> dict:
    """One material slot: its maps (PNG, colour space, SHA), which one is the base colour, and its native JSON."""
    pointer = native["Material"]
    if not pointer:
        if used:
            raise ValueError(f"a used slot has no material: {native}")
        return {"maps": {}, "base": None, "unused": True}
    package = pointer["ObjectPath"].rsplit(".", 1)[0]
    source = directory / (package + ".json")
    data = read(source)
    textures = {item["source"].rsplit(".", 1)[0]: item for item in read(directory / "texture_metadata.json")}
    maps = {}
    for key, texture in data["Textures"].items():
        if not texture:
            continue
        target = textures.get(texture["ObjectPath"].rsplit(".", 1)[0])
        if target is None:
            if texture["ObjectName"].startswith(("CurveLinearColorAtlas'", "TextureCube'")):
                continue
            raise ValueError(f"texture {key} was not delivered: {texture}")
        files = target["files"]
        if len(files) == 1 and files[0].lower().endswith(".png"):
            path = Path(files[0])
            # relative to the case where it can be; on another drive the absolute path (``case / file`` is it)
            same_drive = path.drive.lower() == Path(case).resolve().drive.lower()
            maps[key] = {"file": os.path.relpath(path, case) if same_drive else str(path), "srgb": target["SRGB"],
                         "normal_map": target["normal_map"], "sha256": sha(path)}
    base = "MainTex" if "MainTex" in maps else None
    profile = "main_texture"
    if base is None and "D" in maps and {"XingHenControl", "XHPower"} <= data["Scalars"].keys():
        base, profile = "D", "tacet_mark_texture"
    if base is None and "BaseCol" in maps and "FinalCol_Tint_Layer1" in data["Colors"]:
        base, profile = "BaseCol", "layered_effect_texture_preview"
    constant = None
    if base is None and data["BlendMode"] == 2 and "BaseColor" in data["Colors"]:
        constant = [data["Colors"]["BaseColor"][key] for key in "RGBA"]
        profile = "authored_glass_color_preview"
    missing_surface = (base is None and not data["Textures"] and not data["Colors"]
                       and data["Properties"].get("bUsedWithEmpty") is True)
    if missing_surface:
        profile = "unmapped_native_empty_pass"
    if used and base is None and constant is None and not missing_surface:
        raise ValueError(f"a used material has no verified base colour binding: {source}")
    record = {"maps": maps, "base": base, "profile": profile, "constant_color": constant,
              "native_surface_unmapped": bool(missing_surface), "native_material": data,
              "source_json_sha256": sha(source), "source_package": package, "unused": not used}
    # How a translucent surface covers is wired in its parent material's graph, which the cooked game does
    # not carry (no Opacity input on any of them).  Owner 2026-11-02 (§7.1455): the base map's alpha is the
    # opacity — every such map measured carries graded coverage (a glass at 102, hair from 0 to 255) — said
    # as the owner's choice, not a rule read from the source.  An additive pass with neither map nor colour
    # has nothing to add: kept, drawn clear.
    if data["BlendMode"] == 2 and base is not None and constant is None:
        record.update(blend="alpha", alpha_from="a",
                      coverage="owner-chosen: the base map's alpha as opacity (parent material graph not in the game)")
    elif data["BlendMode"] == 3 and base is None and constant is None:
        record.update(blend="add", coverage="additive pass with no map or colour: kept, drawn clear")
    return record


def prepare(directory: Path, case: Path, ident: str, auxiliary: bool = False) -> None:
    """``case/<ident>.npz`` and ``case/source_document.json`` from the reader's dump of *ident* in *directory*.

    An *auxiliary* part (not a display model's primary) need not carry the biped.
    """
    case.mkdir(parents=True, exist_ok=True)
    source = directory / f"{ident}.source.json.gz"
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        native = json.load(stream)
    if "morphs" not in native:
        raise ValueError(f"{source}: the reader's metadata pass has not written the morphs")
    bones = [{"name": b["name"], "parent": b["parent"]} for b in native["bones"]]
    world = native_world(native["bones"])
    native_rest = world.copy()
    world, scale_proof, native_skin, baked_skin = bake_uniform(world, bones)
    hum = {} if auxiliary else humanoid({b["name"] for b in bones}, bones)
    children = [[j for j, b in enumerate(bones) if b["parent"] == index] for index in range(len(bones))]
    lengths = []
    for index, bone in enumerate(bones):
        spans = [np.linalg.norm(world[child, :3, 3] - world[index, :3, 3]) for child in children[index]]
        spans = [length for length in spans if length > 1e-8]
        parent = bone["parent"]
        if not spans and parent >= 0:
            spans = [np.linalg.norm(world[parent, :3, 3] - world[index, :3, 3])]
        lengths.append(min(spans) if spans and min(spans) > 1e-8 else .005)
    arrays = {"bone_rest": world, "bone_native_rest": native_rest}
    lod, = native["lods"]
    vertices = lod["vertices"]
    co = np.asarray([v["co"] for v in vertices]) @ C.T * .01
    no = np.asarray([v["no"][:3] for v in vertices]) @ C.T
    tan = np.asarray([v["tangent"] for v in vertices])
    tan[:, :3] = tan[:, :3] @ C.T
    faces = np.asarray(lod["indices"], dtype=np.int32).reshape(-1, 3)
    arrays["bone_length"], length_proof = display_lengths(np.array(lengths), world, co, faces)
    columns = max(len(v["influences"]) for v in vertices)
    joints = np.zeros((len(vertices), columns), dtype=np.int32)
    weights = np.zeros_like(joints, dtype=np.float32)
    for index, vertex in enumerate(vertices):
        for column, influence in enumerate(vertex["influences"]):
            joints[index, column] = influence["Bone"]
            weights[index, column] = influence["Weight"]
    if not np.all(np.abs(weights.sum(1) - 1) < 2e-5):
        raise ValueError(f"{ident}: weights that do not sum to one")
    validate_skin_bound(scale_proof, native_skin, baked_skin, co, joints, weights)
    slots = np.full(len(faces), -1, dtype=np.int32)
    for section in lod["sections"]:
        first = section["FirstIndex"] // 3
        last = first + section["NumFaces"]
        if not np.all(slots[first:last] == -1):
            raise ValueError(f"{ident}: two sections share faces")
        slots[first:last] = section["MaterialIndex"]
    if not np.all(slots >= 0):
        raise ValueError(f"{ident}: faces in no section")
    arrays.update(p0_co=co.astype(np.float32), p0_no=no.astype(np.float32), p0_tan=tan.astype(np.float32),
                  p0_tri=faces, p0_slot=slots, p0_jnt=joints, p0_wgt=weights)
    uvs = [np.asarray([v["uv"] for v in vertices])]
    uvs.extend(np.asarray([[value["U"], value["V"]] for value in channel]) for channel in lod["extra_uvs"])
    for channel, values in enumerate(uvs):
        values[:, 1] = 1 - values[:, 1]
        arrays[f"p0_uv{channel}"] = values.astype(np.float32)
    if lod["colors"]:
        arrays["p0_col"] = np.asarray([[v[key] for key in "RGBA"] for v in lod["colors"][0]["Colors"]],
                                      dtype=np.float32) / 255
    shape_names = []
    for morph in native["morphs"]:
        deltas = morph["lods"][lod["source_lod"]]
        if not deltas:
            continue
        index = len(shape_names)
        shape_names.append(morph["name"])
        arrays[f"p0_s{index}_idx"] = np.asarray([value["index"] for value in deltas], dtype=np.int32)
        arrays[f"p0_s{index}_delta"] = (np.asarray([value["delta"] for value in deltas]) @ C.T * .01).astype(np.float32)
    materials = {item["SlotName"]: material_record(item, directory, case, index in slots)
                 for index, item in enumerate(native["materials"])}
    part = {"name": ident, "materials": [item["SlotName"] for item in native["materials"]],
            "uvs": list(range(len(uvs))), "colors": bool(lod["colors"]), "tangents": True,
            "shapes": shape_names, "hidden": False}
    document = {"name": ident + "_Rig", "bones": bones, "humanoid": hum, "parts": [part], "materials": materials,
                "arrays": ident + ".npz", "source": str(source), "source_sha256": sha(source),
                "coordinate_transform": C.tolist(), "unit_scale": .01, "source_bones": native["bones"],
                "uniform_rest_scale_bake": scale_proof, "bone_display_precision": length_proof}
    np.savez_compressed(case / (ident + ".npz"), **arrays)
    (case / "source_document.json").write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")


# -- a display model ------------------------------------------------------------------------------------------

def local_transform(record):
    """A decoration's socket-relative transform, in the converted space."""
    transform = np.eye(4)
    rotation, scale, translation = record["Rotation"], record["Scale3D"], record["Translation"]
    transform[:3, :3] = (C @ Rotation.from_quat([rotation[key] for key in "XYZW"]).as_matrix()
                         @ np.diag([scale[key] for key in "XYZ"]) @ C.T)
    transform[:3, 3] = np.array([translation[key] for key in "XYZ"]) @ C.T * .01
    return transform


def merge_materials(document, source_document, part):
    """Add a part's materials; a same-named one with other maps gets a name of its own."""
    renamed = {}
    for name, record in source_document["materials"].items():
        existing = document["materials"].get(name)
        same = existing is not None and existing.get("native_material") == record.get("native_material")
        if same:
            same = ({key: (value["sha256"], value["srgb"]) for key, value in existing["maps"].items()}
                    == {key: (value["sha256"], value["srgb"]) for key, value in record["maps"].items()})
        alias = name
        if existing is not None and not same:
            alias = name[:54] + "_" + hashlib.sha256(record["source_package"].encode()).hexdigest()[:8]
            if alias in document["materials"]:
                raise ValueError(f"two materials named {alias}")
        renamed[name] = alias
        if not same:
            document["materials"][alias] = record
    part["materials"] = [renamed[name] for name in part["materials"]]


def place_streams(source_arrays, placement):
    """A decoration's streams moved by its socket placement (normals by the inverse transpose)."""
    values = {key: value.copy() for key, value in source_arrays.items() if key.startswith("p0_")}
    linear = placement[:3, :3]
    values["p0_co"] = (values["p0_co"] @ linear.T + placement[:3, 3]).astype(np.float32)
    for key in ("p0_no", "p0_tan"):
        old = values[key][:, :3].astype(np.float64)
        moved = old @ (np.linalg.inv(linear) if key == "p0_no" else linear.T)
        length = np.linalg.norm(moved, axis=1)
        moved *= np.divide(np.linalg.norm(old, axis=1), length, out=np.ones(len(old)), where=length > 0)[:, None]
        values[key][:, :3] = moved
    for key in values:
        if key.endswith("_delta"):
            values[key] = (values[key] @ linear.T).astype(np.float32)
    if np.linalg.det(linear) < 0:
        values["p0_tri"] = values["p0_tri"][:, [0, 2, 1]]
        values["p0_tan"][:, 3] *= -1
    return values


def assemble(row: dict, sources: Path, cases: Path) -> Path:
    """The display model *row* (``assembly.display_models``) as one scene in ``cases/<ident>``; that folder.

    *sources* holds the reader's dump of each part in ``<part ident>/``.
    """
    idents = [part["package"].rsplit("/", 1)[-1] for part in row["parts"]]
    for index, part_ident in enumerate(idents):
        prepare(sources / part_ident, cases / part_ident, part_ident, auxiliary=index > 0)
    ident = row["ident"]
    case = cases / ident
    document = read(case / "source_document.json")
    arrays = dict(np.load(case / document["arrays"]))
    primary_bones = copy.deepcopy(document["bones"])
    primary_native = arrays["bone_native_rest"].copy()
    world = list(primary_native)
    lengths = list(arrays["bone_length"])
    primary_lookup = {bone["name"]: index for index, bone in enumerate(primary_bones)}
    composition, shared_reference_bounds = [], []
    for index, part_ident in enumerate(idents):
        folder = cases / part_ident
        source_document = read(folder / "source_document.json")
        source_arrays = dict(np.load(folder / source_document["arrays"]))
        record = row["parts"][index]
        composition.append(dict(ident=part_ident, native_source=source_document["source"],
                                native_source_sha256=source_document["source_sha256"], component=record))
        if index == 0:
            continue
        part = copy.deepcopy(source_document["parts"][0])
        part["hidden"] = record["hidden"]
        if record.get("attachment_socket"):
            anchor = primary_lookup[record["attachment_socket"]]
            placement = primary_native[anchor] @ local_transform(record["attachment_transform"])
            offset = len(document["bones"])
            prefix = "A" + hashlib.sha256(part_ident.encode()).hexdigest()[:8] + "_"
            for bone_index, bone in enumerate(source_document["bones"]):
                parent = offset + bone["parent"] if bone["parent"] >= 0 else anchor
                document["bones"].append(dict(name=prefix + bone["name"], parent=parent, native_name=bone["name"]))
                world.append(placement @ source_arrays["bone_native_rest"][bone_index])
                lengths.append(source_arrays["bone_length"][bone_index])
            values = place_streams(source_arrays, placement)
            values["p0_jnt"] += offset
        else:
            if source_document["bones"] != primary_bones:
                raise ValueError(f"{part_ident} is not on the primary's skeleton")
            own_rigid, own_proof, own_skin, _ = bake_uniform(source_arrays["bone_native_rest"], source_document["bones"])
            _, _, _, shared_skin = bake_uniform(primary_native, primary_bones)
            joint_error = float(np.linalg.norm(source_arrays["bone_native_rest"][:, :3, 3] - primary_native[:, :3, 3],
                                               axis=1).max() * 1000)
            if joint_error > .01:
                raise ValueError(f"{part_ident}: its joints are {joint_error:.4f} mm off the primary's")
            validate_skin_bound(own_proof, own_skin, shared_skin, source_arrays["p0_co"], source_arrays["p0_jnt"],
                                source_arrays["p0_wgt"])
            shared_reference_bounds.append(dict(
                ident=part_ident, joint_offset_mm=joint_error, source_vertices_unchanged=True,
                all_vertices_rotation_error_mm=own_proof["all_native_vertices_rotation_skin_error_mm"]))
            values = {key: value for key, value in source_arrays.items() if key.startswith("p0_")}
        merge_materials(document, source_document, part)
        document["parts"].append(part)
        for key, value in values.items():
            arrays[f"p{index}_" + key[3:]] = value
    arrays["bone_native_rest"] = np.array(world)
    arrays["bone_rest"], proof, native_skin, baked_skin = bake_uniform(arrays["bone_native_rest"], document["bones"])
    points, faces, offset, part_bounds = [], [], 0, []
    for index in range(len(document["parts"])):
        co, tri = arrays[f"p{index}_co"], arrays[f"p{index}_tri"]
        part_proof = proof.copy()
        validate_skin_bound(part_proof, native_skin, baked_skin, co, arrays[f"p{index}_jnt"], arrays[f"p{index}_wgt"])
        part_bounds.append(dict(part=document["parts"][index]["name"],
                                error_mm=part_proof["all_native_vertices_rotation_skin_error_mm"]))
        points.append(co)
        faces.append(tri + offset)
        offset += len(co)
    arrays["bone_length"], length_proof = display_lengths(np.array(lengths), arrays["bone_rest"],
                                                          np.concatenate(points), np.concatenate(faces))
    document.update(source_composition=composition, assembly_model_ids=row["model_ids"],
                    default_display_model_id=row["default_model_id"], uniform_rest_scale_bake=proof,
                    bone_display_precision=length_proof, native_part_rotation_skin_bounds=part_bounds,
                    shared_reference_bounds=shared_reference_bounds)
    # the game's character is one skeletal mesh, a section per material: one part per section, as a viewer,
    # Blender or Unity lists separate objects (after the checks above, which are the native mesh's)
    from ..interchange.split import by_material
    document, arrays = by_material(document, arrays)
    np.savez_compressed(case / document["arrays"], **arrays)
    (case / "source_document.json").write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return case
