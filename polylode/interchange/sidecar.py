"""What an exported FBX cannot hold, written beside it as ``<model>.islandlife.json``.

The FBX (``interchange.fbx``) carries the skeleton, the skinned meshes, the
shape keys and each material's base map (and emission map); the game's own
statements that a reader needs to rebuild the character go here, read off the
same interchange scene the FBX was written from:

``identity``    the game, the model's ids, the install's version and the source
                files' hashes (the export's source says them, :func:`write`'s
                *identity*);
``humanoid``    the human-bone table (Unity's ``HumanBodyBones`` name → bone of
                the file) and where it comes from — a table the game has, a
                convention checked against the hierarchy, or *missing* with what
                was looked at;
``materials``   every material a part names: its native values as the game's
                material holds them (shader, keywords, floats, colours; Wuthering
                Waves' material instance), the character-shading parameters
                (``extractor.character_styles.native_style``) and every map —
                the ones the FBX embeds named by the FBX slot they are in, every
                other one copied once into ``textures/<sha256>.<ext>`` with its
                hash and colour space;
``parts``       each mesh, why it is hidden (the source switched it off, a coarser
                level of detail, a dormitory or alternate version), and each
                material slot's role (body skin, face, eye, hair, clothing,
                overlay, outline, effect, weapon … or ``unknown``) with what
                decided it; whether the FBX carries its vertex colours; for a
                part the source switched off, which node or renderer did it and
                (GF2's cut-scene prefabs) the timeline tracks on its way;
``expressions`` the shape keys the FBX's BlendShapes already hold, and the
                expressions the game drives with bones (Endfield's face
                controls and presets), as local bone transforms in the FBX's own
                space — see :func:`expressions` for why they are not baked into
                shape keys.

Wuthering Waves' ``skin_and_clothing`` materials also carry ``skin_band``: where the
game's own compiled shader draws skin (``wuthering_waves.skin_band``); GF2's
materials a weapon prefab draws with carry ``weapon`` (``gf2.scene_facts``).
These, ``vertex_colors``, ``source_off``, ``timeline`` and ``skin_materials`` were
added to version 1 without moving or reinterpreting a field: a reader of the first
version reads the same values, and ``weapon`` is a role such a reader treats as it
treats any role it has no use for (as ``unknown``).

Nothing is filled in by guess: a value the source does not say is written as
``null`` with the reason, a role the data does not decide is ``unknown``.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

#: what a reader checks before it reads anything else; a change that moves or reinterprets a field raises VERSION
SCHEMA = "polylode.islandlife-sidecar"
VERSION = 1
SUFFIX = ".islandlife.json"
#: the folder beside the model the maps the FBX does not embed are copied into, named by their content
TEXTURES = "textures"

#: Unity's required human bones (``HumanTrait.RequiredBone``); the rest of :data:`HUMAN_BONES` are optional
REQUIRED_HUMAN = ("Hips", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg", "LeftFoot",
                  "RightFoot", "Spine", "Head", "LeftUpperArm", "RightUpperArm", "LeftLowerArm",
                  "RightLowerArm", "LeftHand", "RightHand")

#: how far a bone's local transform in an expression may be from its rest and count as unmoved
STILL = 1e-6


class SidecarError(ValueError):
    """The scene says something the sidecar cannot state truthfully (a human bone that is not a bone of it)."""


def path_for(model: Path) -> Path:
    """``<folder>/<stem>.islandlife.json`` beside *model* (``<stem>.fbx``)."""
    model = Path(model)
    return model.with_name(model.stem + SUFFIX)


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# -- roles ----------------------------------------------------------------------------------------------------

#: Endfield (``HGRP/…``): the character shaders the game draws each kind of surface with (the same split the
#: game's own runtime keeps: ``endfield.part_roles.classify_faces``)
ENDFIELD_ROLES = {"HGRP/CharacterNPR_Hair": "hair", "HGRP/CharacterNPR_Eye": "eye",
                  "HGRP/CharacterNPR_OverlayShadow": "overlay", "HGRP/CharacterNPR_VFX": "effect",
                  "HGRP/CharacterNPR": "clothing"}
#: Wuthering Waves: the material's shading model (``BasePropertyOverrides.ShadingModel``)
WUWA_ROLES = {"MSM_ToonHair": "hair", "MSM_ToonOutline": "outline"}
#: GF2: only its eye shaders say what they draw; the ``uber`` shader draws skin, cloth, hair, face and weapons alike
GF2_ROLES = {"gf_shader/pbr/character/eye": "eye", "gf_shader/pbr/character/eyeblend_add": "eye_overlay",
             "gf_shader/pbr/character/eyeblend_multiply": "eye_overlay"}
#: how the FBX carries a part's vertex colours (``interchange.fbx``), as the sidecar says it once
VERTEX_COLORS = ("LayerElementColor 'Col' of the part's geometry, ByPolygonVertex / IndexToDirect, RGBA linear 0..1: "
                 "the source's own values (Wuthering Waves: the mesh's 8-bit colour / 255, what its shaders read as "
                 "COLOR0; Endfield and GF2: Unity's mesh colours)")
#: the GF2 material float some shaders name after skin; :func:`gf2_skin_materials` reports what it holds
GF2_SKIN_FLOAT = "_IsSkin"


def slot_role(record: dict) -> tuple[str, str]:
    """``(role, basis)`` of a material record: what the game's own data says the surface is, ``unknown`` otherwise.

    Roles: ``body_skin``, ``face``, ``eye``, ``eye_overlay``, ``hair``, ``clothing``, ``skin_and_clothing`` (one
    material drawing both, told apart by a mask map), ``overlay``, ``outline``, ``effect``, ``unknown``."""
    shader = str(record.get("shader") or "")
    keywords = record.get("keywords") or []
    if shader.startswith("HGRP/"):
        if shader == "HGRP/CharacterNPR_Skin":
            if "_SDFLIGHTMAP" in keywords:
                return "face", "shader HGRP/CharacterNPR_Skin with the face lighting keyword _SDFLIGHTMAP"
            return "body_skin", "shader HGRP/CharacterNPR_Skin without _SDFLIGHTMAP"
        if shader in ENDFIELD_ROLES:
            return ENDFIELD_ROLES[shader], f"shader {shader}"
        if shader.startswith(("HGRP/Effect/", "HGRP/VFX")):
            return "effect", f"effect shader {shader}"
        return "unknown", f"shader {shader} draws no one kind of surface"
    if shader.startswith("gf_shader/"):
        if shader in GF2_ROLES:
            return GF2_ROLES[shader], f"shader {shader}"
        if record.get("weapon"):
            named = ", ".join(f"{row['weapon']} ({row['prefab']})" for row in record["weapon"])
            return "weapon", ("the material is one a weapon prefab of GunWeaponData draws with (the same material "
                              f"asset): {named}")
        return "unknown", (f"shader {shader} draws skin, clothing, hair, face and weapons alike; the prefab has no "
                           "component naming its renderers' kinds")
    native = record.get("native_material")
    if native:
        overrides = (native.get("Properties") or {}).get("BasePropertyOverrides") or {}
        model = overrides.get("ShadingModel")
        switches = native.get("Switches") or {}
        if model in WUWA_ROLES:
            return WUWA_ROLES[model], f"shading model {model}"
        if model == "MSM_ToonFace":
            if switches.get("UseEyeSDF"):
                return "eye", "shading model MSM_ToonFace with the static switch UseEyeSDF"
            return "face", "shading model MSM_ToonFace without UseEyeSDF"
        if record.get("profile") == "tacet_mark_texture":
            return "effect", "the tacet mark's parameters XingHenControl and XHPower (profile tacet_mark_texture)"
        if model == "MSM_ToonCommon" and switches.get("UseSkinMask"):
            return "skin_and_clothing", "shading model MSM_ToonCommon with the static switch UseSkinMask"
        return "unknown", f"shading model {model} draws no one kind of surface"
    return "unknown", "the material has no native shader record"


# -- maps -----------------------------------------------------------------------------------------------------

def _maps(record: dict, folder: Path, out_folder: Path, carried: dict, used: bool, copied: dict) -> dict:
    """Each map of *record*: where it is (the FBX slot it is embedded in, or its copy under ``textures/``), its hash
    and colour space."""
    in_fbx = {}
    for slot in ("DiffuseColor", "TransparentColor", "EmissiveColor"):
        if carried.get(slot):
            in_fbx.setdefault(carried[slot], []).append(slot)
    out = {}
    for prop, entry in (record.get("maps") or {}).items():
        source = Path(entry["file"])
        source = source if source.is_absolute() else folder / source
        row = {"source_name": source.name, "color_space": "srgb" if entry.get("srgb") else "linear",
               "scale": [float(v) for v in entry.get("scale") or (1.0, 1.0)],
               "offset": [float(v) for v in entry.get("offset") or (0.0, 0.0)]}
        if "normal_map" in entry:
            row["normal_map"] = bool(entry["normal_map"])
        if not source.is_file():
            row.update(file=None, sha256=None, missing=f"the converted map is not there: {source}")
            out[prop] = row
            continue
        digest = copied.get(source) or sha256(source)
        row["sha256"] = digest
        if prop in in_fbx:
            row.update(file=None, in_fbx=in_fbx[prop])
            if carried.get("alpha_from") and prop == record.get("base"):
                row["fbx_alpha_from"] = carried["alpha_from"]
        elif not used:
            row.update(file=None, skipped="no triangle of the model is drawn with this material")
        else:
            name = f"{digest}{source.suffix.lower() or '.png'}"
            target = out_folder / TEXTURES / name
            if source not in copied:
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.is_file():
                    temporary = target.with_name(target.name + ".tmp")
                    shutil.copyfile(source, temporary)
                    temporary.replace(target)
                copied[source] = digest
            row["file"] = f"{TEXTURES}/{name}"
        out[prop] = row
    return out


def _material(name: str, record: dict, folder: Path, out_folder: Path, carried: dict, triangles: int,
              copied: dict) -> dict:
    from ..extractor.character_styles import native_style
    native = {key: value for key, value in record.items() if key not in ("maps", "skin_band", "weapon")}
    style = native_style(record)
    role, basis = slot_role(record)
    extra = {}
    if role == "skin_and_clothing":
        extra["skin_band"] = record.get("skin_band") or {
            "status": None, "reason": "the export did not read the game's shader for this material's skin band"}
    if record.get("weapon"):
        extra["weapon"] = record["weapon"]
    return {"triangles": int(triangles), "role": role, "role_basis": basis, **extra, "native": native,
            "style": style, "fbx": {key: carried.get(key) for key in ("DiffuseColor", "TransparentColor",
                                                                     "EmissiveColor", "Opacity", "alpha_from")},
            "maps": _maps(record, folder, out_folder, carried, triangles > 0, copied)}


# -- the human bones ------------------------------------------------------------------------------------------

def humanoid(document: dict, statement: dict) -> dict:
    """The scene's human-bone table with where it comes from (*statement*: ``basis``, ``source``, and ``reason``
    when there is none) and how complete it is."""
    from ..endfield.components import HUMAN_BONES
    bones = {bone["name"] for bone in document["bones"]}
    table = dict(document.get("humanoid") or {})
    strangers = sorted(f"{human} → {bone}" for human, bone in table.items() if bone not in bones)
    if strangers:
        raise SidecarError(f"human bones that are not bones of the scene: {', '.join(strangers)}")
    missing_required = [name for name in REQUIRED_HUMAN if name not in table]
    status = "missing" if not table else "partial" if missing_required else "complete"
    out = {"status": status, "bones": table, "basis": statement.get("basis") if table else None,
           "source": statement.get("source"),
           "required_missing": missing_required,
           "optional_missing": [name for name in HUMAN_BONES if name not in table and name not in REQUIRED_HUMAN]}
    if not table:
        out["reason"] = statement.get("reason") or "the source gives no human-bone table"
    for key in ("fingers", "checked"):
        if key in statement:
            out[key] = statement[key]
    return out


# -- expressions ----------------------------------------------------------------------------------------------

def _locals(frames: np.ndarray, parents: list[int]) -> np.ndarray:
    """Each bone's transform relative to its parent in the FBX's space (Y up: ``glb.TO_Y_UP``), as the FBX
    writer computes ``Lcl``: the turn cancels between a child and its parent, and stays on a root."""
    from .glb import TO_Y_UP
    world = np.einsum("ij,njk->nik", TO_Y_UP, np.asarray(frames, dtype=np.float64))
    out = np.empty_like(world)
    for index, parent in enumerate(parents):
        out[index] = world[index] if parent < 0 else np.linalg.solve(world[parent], world[index])
    return out


def expressions(document: dict, arrays) -> dict:
    """The model's expressions as a reader applies them to the FBX.

    Shape-key expressions (GF2, Wuthering Waves: one source shape each) are the FBX's BlendShape channels already —
    listed by name with the parts that carry them, nothing else is needed.  Expressions the game drives with bones
    (Endfield's face controls, emotions and poses on its face rig: its models have no shape keys) are written as the
    local transforms of the bones they move, in the FBX's own space and units, plus any shape weights and the
    material values they set.  They are not baked into shape keys: a shape key is a fixed offset of the rest mesh, so
    a face pose baked into one is only right while the head is at rest — under a body animation or with two
    expressions mixed the baked offsets and the bones would both move the face.  Kept as bone poses they compose
    with the skeleton the way the game composes them; and a material value (the face's emotion map index) has no
    shape-key form at all.
    """
    parts = document["parts"]
    shapes: dict[str, list[str]] = {}
    for part in parts:
        for name in part.get("shapes") or []:
            shapes.setdefault(name, []).append(part["name"])
    bones = document["bones"]
    parents = [bone["parent"] for bone in bones]
    rest = _locals(arrays["bone_rest"], parents) if "bone_rest" in arrays else None
    poses = []
    for index, animation in enumerate(document.get("animations") or []):
        if animation.get("kind") != "expression" or f"anim{index}" not in arrays or rest is None:
            continue
        local = _locals(np.asarray(arrays[f"anim{index}"])[-1], parents)
        moved = {bones[i]["name"]: np.round(local[i, :3], 7).tolist()
                 for i in range(len(bones)) if np.abs(local[i] - rest[i]).max() > STILL}
        weights = {}
        for p, part in enumerate(parts):
            key = f"p{p}_anim{index}_weights"
            if key in arrays:
                last = np.asarray(arrays[key])[-1]
                held = {part["shapes"][s]: round(float(w), 6) for s, w in enumerate(last) if abs(float(w)) > STILL}
                if held:
                    weights[part["name"]] = held
        single = {name for held in weights.values() for name in held}
        if not moved and len(single) == 1 and all(set(held.values()) == {1.0} for held in weights.values()):
            continue                  # one source shape at full weight: that shape key itself, in the FBX already
        poses.append({key: animation[key] for key in ("name", "clip", "controls", "additive", "unbound_controls",
                                                      "material_values") if key in animation}
                     | {"bones": moved, "shape_weights": weights})
    out = {"shape_keys": {"in_fbx": "BlendShape channel per shape on each part named", "shapes": shapes},
           "bone_poses": {"space": "the FBX's bone local transform (Lcl), Y up, metres; rows of the 3×4 matrix",
                          "poses": poses}}
    if document.get("expression_controls"):
        out["controls"] = document["expression_controls"]
    if document.get("expression_presets"):
        out["presets"] = document["expression_presets"]
    if document.get("expression_failure"):
        out["reason"] = f"bone-driven expressions could not be built: {document['expression_failure']}"
    elif not poses and not shapes:
        out["reason"] = ("no expressions: the model has no shape keys, and no bone-driven expressions were "
                         "worked out for this export")
    return out


def gf2_skin_materials(materials: dict) -> dict | None:
    """Whether GF2's own data names the character's skin materials: None for another game's model.

    It does not: the prefab's root carries a Transform and an Animator and no component naming its renderers'
    kinds, no table lists renderers or materials by kind, and the one float named after skin, ``_IsSkin``, is 0 or
    absent on every material (skin, face and clothes alike: of the 2 019 materials of the 161 character models
    converted from the install on 2026-11-13, 315 hold 0 and 1 704 do not have it).  What this model's materials
    hold is written, so a reader sees it."""
    gf2 = {name: record for name, record in materials.items() if str(record.get("shader", "")).startswith("gf_shader/")}
    if not gf2:
        return None
    values = {name: (record.get("floats") or {}).get(GF2_SKIN_FLOAT) for name, record in gf2.items()}
    if any(value not in (None, 0, 0.0) for value in values.values()):
        return {"status": "flag_set", GF2_SKIN_FLOAT: values,
                "reason": f"a material sets {GF2_SKIN_FLOAT}: what the shader does with it is not read here"}
    return {"status": None, GF2_SKIN_FLOAT: values,
            "reason": ("GF2's data names no skin material: the prefab has no component naming its renderers' kinds, "
                       f"no table lists them, and {GF2_SKIN_FLOAT} is 0 or absent on every material, skin and clothes "
                       "alike")}


# -- the file -------------------------------------------------------------------------------------------------

def write(document_path: Path, model: Path, fbx_report: dict, *, identity: dict, humanoid_statement: dict,
          target: Path | None = None) -> Path:
    """``<model stem>.islandlife.json`` beside *model* (the FBX written from the scene at *document_path*, whose
    writer gave *fbx_report*), with the maps the FBX does not embed copied into ``textures/``."""
    from .. import __version__
    from ..core.paths import write_json_atomic
    document_path, model = Path(document_path), Path(model)
    target = Path(target) if target is not None else path_for(model)
    folder = document_path.parent
    document = json.loads(document_path.read_text(encoding="utf-8"))
    arrays = np.load(folder / document["arrays"])
    carried = (fbx_report or {}).get("materials") or {}
    counts: dict[str, int] = {}
    parts = []
    for index, part in enumerate(document["parts"]):
        slots = np.asarray(arrays[f"p{index}_slot"]) if f"p{index}_slot" in arrays else np.zeros(0, int)
        per_slot = np.bincount(slots, minlength=len(part["materials"])) if len(slots) else \
            np.zeros(len(part["materials"]), int)
        rows = []
        for slot, name in enumerate(part["materials"]):
            counts[name] = counts.get(name, 0) + int(per_slot[slot])
            record = document["materials"].get(name) if name else None
            role, basis = slot_role(record) if record is not None else ("unknown", "the slot names no material")
            rows.append({"slot": slot, "material": name, "triangles": int(per_slot[slot]), "role": role,
                         "role_basis": basis})
        hidden_by = list(part.get("hidden_by") or (["source"] if part.get("hidden") else []))
        row = {"name": part["name"], "path": part.get("path", ""), "hidden": bool(part.get("hidden")),
               "hidden_by": hidden_by, "vertices": part.get("vertices"),
               "triangles": int(len(slots)), "shapes": list(part.get("shapes") or []),
               "vertex_colors": bool(part.get("colors")) and f"p{index}_col" in arrays, "slots": rows}
        for key in ("source_off", "timeline"):
            if key in part:
                row[key] = part[key]
        parts.append(row)
    copied: dict = {}
    materials = {name: _material(name, document["materials"][name], folder, target.parent, carried.get(name) or {},
                                 counts.get(name, 0), copied)
                 for name in dict.fromkeys(m for part in document["parts"] for m in part["materials"] if m)}
    sidecar = {"schema": SCHEMA, "version": VERSION, "generator": f"Polylode {__version__}",
               "model": {"file": model.name, "sha256": sha256(model), "bytes": model.stat().st_size,
                         "rig": document["name"], "space": "Y up, metres, as the FBX (GlobalSettings)"},
               "identity": identity,
               "humanoid": humanoid(document, humanoid_statement),
               "materials": materials,
               "parts": parts,
               "vertex_colors": VERTEX_COLORS,
               "expressions": expressions(document, arrays)}
    if document.get("timeline"):
        sidecar["timeline"] = document["timeline"]
    if document.get("facts_failure"):
        sidecar["facts_failure"] = document["facts_failure"]
    skin = gf2_skin_materials(document["materials"])
    if skin is not None:
        sidecar["skin_materials"] = skin
    # compact: three quarters of an Endfield sidecar is its face poses, and indented it was 8.5 MB rather than 3.6
    write_json_atomic(target, sidecar, separators=(",", ":"))
    return target
