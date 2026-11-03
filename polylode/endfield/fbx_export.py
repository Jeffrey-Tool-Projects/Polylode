"""Write converted Endfield characters as rigged FBX (and ``.blend``) files and read each one back.

The package side of ``tools/ef_export_fbx.py`` (its CLI) and of the model
extractor: :mod:`.rigscene` decides what goes in a file, ``tools/ef_fbx_blender.py``
(run by Blender as a ``.py`` file) writes it and reads it back, and this module
checks the read-back against the scene (:func:`problems`).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import fbxfile
from . import model as M
from . import rigscene as RS
from . import store

WRITER = Path(__file__).resolve().parents[2] / "tools" / "ef_fbx_blender.py"
CATEGORY = "chara"
#: tolerances of the read-back, in the report's own units
BONE_MM, BONE_DEG, REST_MM, POSED_MM = 0.01, 0.01, 0.01, 0.05


@dataclass
class Target:
    folder: str                # ``<body type>/<name>``
    key: tuple                 # what orders the targets of one folder
    source: str                # ``chara/<ident>`` or a model container
    ident: str = ""            # the stored item, when it is one
    container: str = ""        # the model container, when it is one
    stem: str = ""             # file name without ``.fbx``


@dataclass
class Outcome:
    rows: list = field(default_factory=list)       # manifest
    reports: list = field(default_factory=list)    # validation
    failed: dict = field(default_factory=dict)     # source → reason


def targets(pipeline, only: str) -> list[Target]:
    found: list[Target] = []
    for ident in store.idents(CATEGORY):
        extra = store.read_extra(CATEGORY, ident) or {}
        container = str(extra.get("model") or "")
        folder = M.folder_of(container) or f"other/{ident}"
        found.append(Target(folder, (container, ident), f"{CATEGORY}/{ident}",
                            ident=ident))
    taken = {t.folder for t in found}
    if pipeline is not None:
        for container in pipeline.model_containers():
            folder = M.folder_of(container)
            if folder and folder not in taken:
                found.append(Target(folder, (container, ""), container,
                                    container=container))
    by_folder: dict[str, list[Target]] = {}
    for target in found:
        if only.lower() in target.folder.lower():
            by_folder.setdefault(target.folder, []).append(target)
    out = []
    for folder, group in sorted(by_folder.items()):
        group.sort(key=lambda t: t.key)
        for index, target in enumerate(group):
            target.stem = (folder.rpartition("/")[2] if index == 0
                           else target.ident or Path(target.container).stem)
            out.append(target)
    return out


def load(target: Target, pipeline) -> tuple[list, dict, dict]:
    if target.ident:
        item = store.read_item(CATEGORY, target.ident)
        if item is None or not item.nodes:
            raise LookupError(f"{target.source} is not in the store")
        return item.pieces, item.nodes, store.read_extra(CATEGORY, target.ident) or {}
    return pipeline.read_model(target.container)


def copy_textures(scene: RS.Scene, folder: Path) -> None:
    for file, (cab, name) in scene.textures.items():
        source = store.texture_path(cab, name)
        if source is None or not source.is_file():
            raise LookupError(f"texture {name} ({cab}) is not in the store")
        destination = folder / RS.TEXTURE_DIR / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def problems(scene: RS.Scene, report: dict) -> list[str]:
    """What the read-back says is wrong with the file (empty when nothing)."""
    out = []
    if report["armatures"] != 1:
        return [f"{report['armatures']} armatures"]
    if (report["bones"] != len(scene.bones) or report["bones_missing"]
            or report["bones_in_file"] != len(scene.bones)):
        out.append(f"{report['bones']} bones ({report['bones_in_file']} in the "
                   f"file) for {len(scene.bones)}")
    if report["parents_wrong"]:
        out.append(f"{report['parents_wrong']} bones under another parent")
    if report["bone_offset_mm"] > BONE_MM or report["bone_turn_deg"] > BONE_DEG:
        out.append(f"bones off by {report['bone_offset_mm']:.4f} mm, "
                   f"{report['bone_turn_deg']:.4f} deg ({report['bone_turn_worst']})")
    if report["bind_offset_mm"] > BONE_MM or report["bind_turn_deg"] > BONE_DEG:
        out.append(f"bind matrices off by {report['bind_offset_mm']:.4f} mm, "
                   f"{report['bind_turn_deg']:.4f} deg ({report['bind_worst']})")
    if not report["binds"] and any(len(part.positions) for part in scene.parts):
        out.append("no skin cluster in the file")
    for part, row in zip(scene.parts, report["parts"]):
        name = part.name
        if not row["found"]:
            out.append(f"{name}: not in the file")
            continue
        # what an import keeps: Blender drops a face on another's three vertices
        kept = (len(part.triangles) - report["degenerate"].get(name, 0)
                - report["repeated"].get(name, 0))
        if row["vertices"] != len(part.positions) or row["triangles"] != kept:
            out.append(f"{name}: {row['vertices']} vertices, {row['triangles']} "
                       f"triangles for {len(part.positions)}, {kept}")
            continue
        if not row["armature"] or row["weighted"] != row["weighted_expected"]:
            out.append(f"{name}: {row['weighted']} of {row['weighted_expected']} "
                       f"vertices weighted")
        if row["rest_mm"] > REST_MM or row["posed_mm"] > POSED_MM:
            out.append(f"{name}: rest off {row['rest_mm']:.4f} mm, posed off "
                       f"{row['posed_mm']:.4f} mm")
        if row["weighted"] and row["moved_mm"] <= POSED_MM:
            out.append(f"{name}: the pose did not move it")
        if row["shape_keys"] != len(part.shapes):
            out.append(f"{name}: {row['shape_keys']} blend shapes for "
                       f"{len(part.shapes)}")
    return out


def blender_run(blender: Path, mode: str, document: Path, fbx: Path) -> dict:
    """Run the Blender script in *mode*; returns the report it wrote."""
    report_path = fbx.with_suffix(f".{mode}.json")
    report_path.unlink(missing_ok=True)
    done = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
         "--python", str(WRITER), "--", mode, str(document), str(fbx), str(report_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if done.returncode != 0 or not report_path.is_file():
        raise RuntimeError(f"Blender ({mode}) exit {done.returncode}: "
                           f"{(done.stdout + done.stderr)[-1500:]}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.unlink()
    return report


def scene_document(target: Target, out_dir: Path, pipeline, shaders=None, log=print, poses=None) -> tuple:
    """*target*'s scene, its maps copied to ``<out_dir>/<folder>/textures``, and its interchange document there,
    every material drawable: an effect run as the game compiled it, an emission baked (``vfx_bake``; *shaders*
    fetches a shader the cache lacks).  With *poses* (what opens the pipeline when the clips are not decoded yet),
    the character's display poses go in as animations (``poses``)."""
    from . import poses as P
    from . import vfx_bake
    pieces, nodes, extra = load(target, pipeline)
    scene = RS.scene_of(pieces, nodes, target.stem)
    folder = out_dir / target.folder
    copy_textures(scene, folder)
    document = RS.write(scene, folder, target.stem, extra, target.source)
    vfx_bake.complete(document, shaders, log)
    if poses is not None:
        P.add(document, scene, nodes, extra, poses, log)
    return scene, document


def write_one(target: Target, out_dir: Path, blender: Path, pipeline,
              keep_scene: bool = False, blend: bool = False, shaders=None, log=print) -> tuple[dict, dict]:
    """Write *target* as ``<out_dir>/<folder>/<stem>.fbx`` and read it back; with *blend*, also the ``.blend``.

    The ``.blend`` is the scene Blender built for the FBX, saved with its
    images packed and opened again: its meshes and armature must be the
    scene's, and every image inside it.
    """
    scene, document = scene_document(target, out_dir, pipeline, shaders, log)
    folder = out_dir / target.folder
    if blend:
        data = json.loads(document.read_text(encoding="utf-8"))
        data["write_blend"] = True
        document.write_text(json.dumps(data, indent=1), encoding="utf-8")
    fbx = folder / f"{target.stem}.fbx"
    bones = [(bone.name, bone.parent, bone.rest) for bone in scene.bones]
    try:
        counts = blender_run(blender, "write", document, fbx)
        refined = fbxfile.refine(fbx, bones)
        report = blender_run(blender, "check", document, fbx)
        if blend:
            report["blend"] = blender_run(blender, "blend", document, fbx)
    finally:
        if not keep_scene:
            (folder / f"{target.stem}.npz").unlink(missing_ok=True)
    in_file = fbxfile.bone_errors(fbx, bones)
    report.update(counts)
    report.update(bones_in_file=in_file["bones"], bone_offset_mm=in_file["offset_mm"],
                  bone_turn_deg=in_file["turn_deg"], bone_turn_worst=in_file["worst"],
                  binds=in_file["binds"], bind_offset_mm=in_file["bind_offset_mm"],
                  bind_turn_deg=in_file["bind_turn_deg"], bind_worst=in_file["bind_worst"],
                  refined_deg=refined[0], refined_mm=refined[1] * 1000.0)
    found = problems(scene, report)
    if blend:
        found += blend_problems(scene, report["blend"])
    row = {"actor": target.folder, "fbx": str(fbx), "source": target.source,
           "bones": len(scene.bones), "meshes": len(scene.parts),
           "vertices": int(sum(len(p.positions) for p in scene.parts)),
           "triangles": int(sum(len(p.triangles) for p in scene.parts)),
           "materials": len(scene.materials), "textures": len(scene.textures),
           "blend_shapes": int(sum(len(p.shapes) for p in scene.parts)),
           "triangles_left_out": int(sum(report["degenerate"].values())),
           "triangles_repeated": int(sum(report["repeated"].values())),
           "hidden": [p.name for p in scene.parts if p.hidden],
           "no_base_colour": sorted(name for name, record in scene.materials.items()
                                    if record["base"] is None),
           "success": not found, "problems": found}
    report.update(actor=target.folder, fbx=str(fbx), ok=not found)
    return row, report


def blend_problems(scene: RS.Scene, report: dict) -> list[str]:
    """What opening the ``.blend`` again says is wrong with it (empty when nothing)."""
    out = []
    if report["armatures"] != 1:
        out.append(f".blend: {report['armatures']} armatures")
    for part in scene.parts:
        row = report["meshes"].get(part.name)
        if row is None:
            out.append(f".blend: {part.name} is not in it")
        elif row["vertices"] != len(part.positions) or not row["armature"]:
            out.append(f".blend: {part.name} has {row['vertices']} vertices for {len(part.positions)}"
                       + ("" if row["armature"] else ", no armature"))
    if report["not_packed"]:
        out.append(f".blend: images not inside it: {', '.join(report['not_packed'])}")
    return out
