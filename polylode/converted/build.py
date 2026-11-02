"""Build source-qualified geometry warehouses from individually verified covered deliveries.

The output is a conversion candidate, not a material/appearance acceptance.
Every source keeps its own identity evidence, arrays, morphs and material maps.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.paths import write_json_atomic
from ..endfield import body as fitting
from ..endfield.axes import TO_Z_UP
from ..endfield.hrole import H_BONES, stand_in_skeleton
from ..mesh.skin import split_archive_skin
from ..origin import bake, converted_rig as CR, hanim, motion_ik
from . import geometry, lineage, material_ownership, names, store
from .scope import EXCLUDED_WUWA

#: The sources whose characters arrive as covered deliveries with a lent, fitted body (``inputs``);
#: own-body Blender deliveries are ``blender_source``'s.
DELIVERY_SOURCES = ("gf2", "wuthering_waves")
#: How the lent body's limbs are sized across (``endfield.body.CROSS_SECTIONS``): by each segment's own
#: length until the character's skin can be measured apart from her clothes (PROGRESS #177)
CROSS_SECTION = "length"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _component(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value)).strip(" .")
    if not value or value in (".", ".."):
        raise store.WarehouseError("empty character folder name")
    return value


@dataclass(frozen=True)
class Input:
    """An individually identified model/outfit and its own delivery evidence."""
    source: str
    ident: str
    character: str
    case: Path
    native_folder: Path
    arrays: Path
    identity: dict


def inputs(project: Path, source: str) -> list[Input]:
    """Read only the verified female covered plans and the current adult model delivery scope."""
    if source not in DELIVERY_SOURCES:
        raise store.WarehouseError(f"unsupported conversion source: {source}")
    game = store.SOURCES[source]
    base = project / game / ("covered_1096" if source == "gf2" else "covered_1121")
    rows = _read(base / "plan.json")["rows"]
    result = []
    for row in rows:
        ident = row["ident"]
        if ident != _component(ident):
            raise store.WarehouseError(f"invalid source identifier: {ident}")
        native = project / f"{game}_Converted" / ident
        if source == "gf2":
            owners = row["characters"]
            if not owners or any(owner.get("gender") != "female" or not owner.get("gender_evidence")
                                 for owner in owners):
                raise store.WarehouseError(f"unverified GF2 identity: {ident}")
            character = "__".join(sorted({f'{owner["id"]}_{owner["name"]}' for owner in owners}))
            arrays = native / f"{ident}.npz"
        else:
            if ident in EXCLUDED_WUWA:
                continue
            if not (native / f"{ident}_covered.blend").is_file():
                raise store.WarehouseError(f"verified character delivery is missing: {ident}")
            owners = row["owners"]
            if not owners or any(not owner.get("body_type", "").startswith("Female")
                                 or not owner.get("role_row_sha256") or not owner.get("preload_row_sha256")
                                 for owner in owners):
                raise store.WarehouseError(f"unverified Wuthering Waves identity: {ident}")
            character, localized = names.wuthering_name(project, [owner["role_name_key"] for owner in owners])
            row = {**row, "localized_identity": localized}
            arrays = base / ident / f"{ident}.npz"
        result.append(Input(source, ident, _component(character), base / ident, native, arrays, row))
    if len({item.ident for item in result}) != len(result):
        raise store.WarehouseError("duplicate character identifiers in source plan")
    return result


def relocate(item: Input, output: Path) -> Path:
    """Move an existing candidate to its verified character folder without duplicating it."""
    game = output / store.SOURCES[item.source]
    target = game / item.character / item.ident
    if not game.is_dir():
        return target
    matches = [parent / item.ident for parent in game.iterdir()
               if parent.is_dir() and (parent / item.ident).is_dir()]
    if len(matches) > 1:
        raise store.WarehouseError(f"duplicate existing conversion folders: {item.ident}")
    if matches and matches[0].resolve() != target.resolve():
        original = matches[0]
        if not all(path.resolve().is_relative_to(game.resolve()) for path in (original, target)):
            raise store.WarehouseError("character folder move leaves the source warehouse")
        target.parent.mkdir(parents=True, exist_ok=True)
        original.rename(target)
        if not any(original.parent.iterdir()):
            original.parent.rmdir()
    return target


def _link(source: Path, destination: Path) -> None:
    """Copy a source member, using a same-volume hard link when available; never overwrite an inode."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if store.digest(destination) == store.digest(source):
            return
        raise store.WarehouseError(f"existing warehouse member differs: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _materials(document: dict, source: Path, folder: Path) -> None:
    for material in document["materials"].values():
        for binding in material.get("maps", {}).values():
            path = (source / binding["file"]).resolve()
            sha = store.digest(path)
            if binding.get("sha256") is not None and sha != binding["sha256"]:
                raise store.WarehouseError(f"source material texture changed: {path}")
            relative = f"textures/{sha}{path.suffix.lower()}"
            _link(path, folder / relative)
            binding.update(file=relative, sha256=sha)


def prepare_materials(folder: Path, catalog: material_ownership.Gf2Catalog) -> None:
    """Keep exact per-slot native shader owners beside a GF2 package's deduplicated export materials."""
    metadata = store.read(folder)
    if metadata["source"] != "gf2":
        raise store.WarehouseError("GF2 material owners cannot be attached to another source")
    document = _read(folder / "native.json")
    owners = catalog.prepare(metadata["ident"], document)
    relative = "native_material_owners.json"
    path = folder / relative
    if path.is_file() and _read(path) == owners and metadata["files"].get(relative) == store.digest(path):
        return
    write_json_atomic(path, owners, indent=1)
    metadata["files"][relative] = store.digest(path)
    metadata["ready_for_display"] = False
    metadata["full_appearance_acceptance_claimed"] = False
    write_json_atomic(folder / "warehouse.json", metadata, indent=1)


def _mapping(piece, target, human: dict, evidence: dict) -> dict:
    """Reuse recorded finger/fit mappings and the established body fitter's semantic limb roles."""
    mapping = dict(evidence["bone_mapping"])
    for name in piece.bones:
        if name not in mapping:
            # A recorded finger correspondence is mandatory, never an Endfield-name guess.
            if any(token in name for token in ("thumb", "index", "middle", "ring", "little")):
                raise store.WarehouseError(f"body source finger correspondence missing: {name}")
            mapping[name] = fitting._bone_target(name, human, set(target.index))
    absent = set(mapping.values()) - set(target.index)
    if absent:
        raise store.WarehouseError(f"body fit targets absent nodes: {sorted(absent)}")
    return mapping


def goal_joints() -> dict:
    """The MotionIK goal → reference joint table every converted rig carries (``converted_rig.remap_goals``)."""
    settings = motion_ik.load_settings(_read(hanim.FBBIK_FILE))
    return CR.goal_joints_from_ik(settings, _read(hanim.MOTION_IK_FILE).get("files") or {})


def _opaque_bone_names(item: Input, document: dict, authored: np.ndarray) -> np.ndarray:
    """Restore a local-frame suit's bone names using its own recorded source-index aliases."""
    body_document = item.case / "body_document.json"
    if not body_document.is_file():
        return authored
    body_doc = _read(body_document)
    frame = body_doc.get("covered_frame")
    if frame is None:
        return authored
    if frame.get("origin") != [0.0, 0.0, 0.0]:
        raise store.WarehouseError("opaque frame origin needs a verified placement adapter")
    aliases = {}
    for index, alias in frame["aliases"].items():
        source = int(index)
        if (not 0 <= source < len(document["bones"]) or body_doc["bones"][source]["name"] != alias
                or alias in aliases):
            raise store.WarehouseError("opaque frame alias does not identify its native source bone")
        aliases[alias] = document["bones"][source]["name"]
    if not set(authored.tolist()) <= set(aliases):
        raise store.WarehouseError("opaque frame is missing an authored bone alias")
    mapped = [aliases[name] for name in authored.tolist()]
    if len(mapped) != len(set(mapped)):
        raise store.WarehouseError("opaque frame merges distinct source influences")
    return np.array(mapped)


def _body(item: Input, document: dict, native: dict, source_body, donor) -> tuple:
    target = geometry.skeleton(document, native)
    fit_evidence = _read(item.case / "fit.json")
    mapping = _mapping(donor, target, document["humanoid"], fit_evidence)
    # this source's own skin is not told from its clothes yet (PROGRESS #177): its limbs keep the
    # rule they were fitted by before girth was measured, each segment sized by its own length
    fitted = fitting.fit_body(donor, target, document["humanoid"], bone_mapping=mapping,
                             cross_section=CROSS_SECTION)
    helpers = fitting.helper_nodes(donor, target, document["humanoid"], bone_mapping=mapping,
                                   cross_section=CROSS_SECTION)
    for name, (parent, local) in helpers.items():
        target.add_matrix(name, parent, local)
    target.finalize()
    reference = stand_in_skeleton(source_body.implant)
    fingers = {name: node for name, node in fit_evidence["bone_mapping"].items()
               if any(token in name for token in ("thumb", "index", "middle", "ring", "little"))}
    landmarks = {name.removeprefix(fitting.GRAFT): name for name in helpers
                 if reference.has(name.removeprefix(fitting.GRAFT))}
    rig = CR.build(reference, target, document["humanoid"], H_BONES,
                   fingers=fingers, landmarks=landmarks, goal_joints=goal_joints())
    with np.load(item.case / "export_pose_reference.npz", allow_pickle=False) as original:
        positions, faces = original["rest"], original["faces"]
    with np.load(item.case / "weight_solved.npz", allow_pickle=False) as solved:
        weights, authored = solved["weights"], solved["bone_names"]
    bone_names = _opaque_bone_names(item, document, authored)
    if len(weights) != len(positions) or not set(bone_names.tolist()) <= set(target.index):
        raise store.WarehouseError("opaque body topology, full weights or bone names disagree")
    donor_weights, donor_indices = split_archive_skin(donor.skin, bone_count=len(donor.bones))
    arrays = {"opaque_positions": positions @ TO_Z_UP[:3, :3], "opaque_faces": faces,
              "opaque_weights": weights, "opaque_bone_names": bone_names, "opaque_authored_bone_names": authored,
              "donor_positions": fitted.mesh.positions, "donor_normals": fitted.mesh.normals,
              "donor_faces": fitted.mesh.indices.reshape(-1, 3), "donor_uv": donor.mesh.uvs,
              "donor_weights": donor_weights, "donor_indices": donor_indices,
              "donor_bone_names": np.array([fitting.GRAFT + name for name in donor.bones])}
    return target, rig, helpers, arrays, mapping


def build(item: Input, output: Path) -> Path:
    """Fit a real source body, retain the opaque delivery, and write a verified rig candidate.

    Native model streams and material textures are unchanged; only their file
    locations change. No screenshot, render, face inspection or thumbnail is made.
    """
    delivery = _read(item.case / "delivery.json")
    if not all(delivery.get(key) is True for key in ("independent_formal_reopen", "complete_opaque_body_built",
                                                    "original_clothes_accessories_retained", "originals_unchanged")):
        raise store.WarehouseError(f"covered delivery is not verified: {item.ident}")
    folder = relocate(item, output)
    folder.mkdir(parents=True, exist_ok=True)
    document = copy.deepcopy(_read(item.case / "source_document.json"))
    with np.load(item.arrays, allow_pickle=False) as archive:
        native = {name: archive[name] for name in archive.files}
    original_folder = item.native_folder if item.source == "gf2" else item.case
    _materials(document, original_folder, folder)
    source = bake.read_item("base", "body")
    if source is None:
        raise store.WarehouseError("the source game's converted female body is missing")
    donor = next(piece for piece in source.pieces if piece.mesh.name == "o_body_a")
    target, rig, helpers, arrays, mapping = _body(item, document, native, source, donor)
    _link(item.arrays, folder / "native.npz")
    write_json_atomic(folder / "native.json", document, indent=1)
    temporary = folder / "body.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, folder / "body.npz")
    # A real independently reopened model remains available beside the new warehouse candidate.
    references = {}
    for kind in ("blend", "fbx"):
        source_path, proof = lineage.covered_file(item.case.parents[2], item.source, item.ident, delivery, kind)
        _link(source_path, folder / f"{item.ident}_covered.{kind}")
        references[f"{item.ident}_covered.{kind}"] = proof
    files = {path.relative_to(folder).as_posix(): store.digest(path) for path in folder.rglob("*")
             if path.is_file() and path.name not in ("warehouse.json", "validation.json", "card.png")
             and path.suffix.lower() not in (".blend", ".fbx")}
    meta = {"version": store.VERSION, "source": item.source, "ident": item.ident,
            "character": item.character, "identity": item.identity, "files": files, "reference_files": references,
            "rig": CR.to_payload(rig), "helper_nodes": helpers, "body_mapping": mapping,
            "scope": "complete source body plus opaque inner clothing and original costume/accessories",
            "status": "geometry_candidate", "ready_for_display": False,
            "pending": ["donor fit/join/deformation acceptance", "clean donor skin material binding",
                        "native shader renderer dispatch", "source-qualified actor cards and UI load"],
            "donor": {"source": "base/body:o_body_a", "vertices": donor.mesh.vertex_count,
                      "source_skin_texture_used": False, "tattoo_texture_imported": False,
                      "fit_accepted": False},
            "opaque": {"complete_weights_retained": True, "maximum_influences": int((arrays["opaque_weights"] > 0).sum(1).max())},
            "provenance": {"source_arrays_sha256": store.digest(item.arrays),
                           "fit_receipt_sha256": store.digest(item.case / "fit.json"),
                           "covered_delivery_sha256": store.digest(item.case / "delivery.json"),
                           "source_body_archive_sha256": store.digest(bake.bake_dir() / "base/body.npz"),
                           "reference_rig_sha256": store.digest(__import__("polylode.origin.rig", fromlist=["CACHE"]).CACHE)},
            "screenshots_created": 0, "renders_created": 0, "full_appearance_acceptance_claimed": False}
    write_json_atomic(folder / "warehouse.json", meta, indent=1)
    return folder


def index(output: Path, source: str, folders: list[Path]) -> dict:
    """Publish a source-qualified index after validating every referenced package."""
    game = output / store.SOURCES[source]
    rows = []
    for folder in folders:
        meta = store.read(folder)
        if meta["source"] != source:
            raise store.WarehouseError("cross-source item in warehouse index")
        path = folder / "warehouse.json"
        rows.append({"ident": meta["ident"], "character": meta["character"], "status": meta["status"],
                     "file": path.relative_to(game).as_posix(), "sha256": store.digest(path)})
    data = {"version": store.VERSION, "source": source, "items": sorted(rows, key=lambda row: row["ident"])}
    write_json_atomic(game / "index.json", data, indent=1)
    return data
