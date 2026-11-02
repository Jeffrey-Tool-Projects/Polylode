"""Build and account for every local FBX, including separate source versions."""
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from ..origin import bake
from ..origin.prefab import Piece
from . import store
from .actor import _rest_mesh, build_skeleton
from .body import bake_body, native_skin_shader
from .body_scope import require_body_scope
from .part_roles import (
    REGIONS,
    classify_piece,
    connected_faces,
    face_reference,
    face_regions,
    facial_context,
    facial_evidence,
    facial_reference_islands,
    has_sdf_face,
    painted_garment_faces,
)
from .source_inventory import build_inventory


def jobs_from_inventory(inventory: dict, output_root: Path) -> list[dict]:
    """A job per file; no folder or character id can collapse versions."""
    jobs = []
    for record in inventory["records"]:
        relative = Path(record["fbx"])
        jobs.append({"source": str(Path(inventory["source_root"]) / relative),
                     "relative_fbx": record["fbx"], "prefab": record["prefab"],
                     "output": str((output_root / relative.with_suffix("")).resolve()),
                     "body_arrays": None, "source_gaps": record["gaps"],
                     "classification": {}, "body_status": "original clothed source only"})
    return jobs


def reconcile_results(inventory: dict, output_root: Path) -> dict:
    """Rebuild the canonical account from actual input paths and saved results."""
    rows = []
    for job in jobs_from_inventory(inventory, output_root):
        path = Path(job['output']) / 'result.json'
        row = {'relative_fbx': job['relative_fbx'], 'status': 'unprocessed',
               'reason': 'No saved result for this actual source file'}
        if path.is_file():
            try:
                saved = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(saved, dict) or saved.get('relative_fbx') != job['relative_fbx']:
                    raise ValueError('Result identity does not match this source version')
                if saved.get('status') not in {'built', 'needs_review', 'failed'}:
                    raise ValueError('Unknown saved build status')
                row = saved
            except (OSError, ValueError) as exc:
                row.update(status='invalid_report', reason=str(exc))
        digest = {key: row[key] for key in (
            'relative_fbx', 'status', 'reason', 'body_status', 'body_error', 'source_gaps',
            'original_reopen_verified', 'original_rig_reopen_verified',
            'source_raw_reopen_verified', 'blender_file', 'missing_texture_files') if key in row}
        digest.update(_result_currentness(job, row))
        digest['result_file'] = str(path)
        blend = Path(job['output']) / (Path(job['source']).stem + '_complete.blend')
        digest['editable_blend_present'] = blend.is_file()
        digest['preview_count'] = len(row.get('previews', []))
        seams = row.get('seams') or {}
        digest['seam_count'] = len(seams.get('seams', []))
        digest['head_seam_count'] = len(seams.get('head_seams', []))
        digest['seams_accepted'] = seams.get('accepted')
        digest['seam_revision'] = seams.get('revision')
        digest['unresolved_seams'] = seams.get('unresolved', [])
        rows.append(digest)
    counts = dict(Counter(row['status'] for row in rows))
    processed = sum(counts.get(status, 0) for status in ('built', 'needs_review', 'failed'))
    summary = {'version': 1, 'actual_fbx_count': inventory['fbx_count'],
               'processed_count': processed, 'filtered': '', 'status_counts': counts,
               'editable_blend_count': sum(row['editable_blend_present'] for row in rows),
               'current_editable_blend_count': sum(row['artifact_current'] for row in rows),
               'current_body_count': sum(row['body_acceptance_current'] for row in rows),
               'all_inputs_processed': processed == inventory['fbx_count'], 'results': rows}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def _result_currentness(job: dict, row: dict) -> dict:
    from .sample_support import artifacts_current
    output, source = Path(job['output']), Path(job['source'])
    blend = output / (source.stem + '_complete.blend')
    current, reason = artifacts_current(row, source, blend)
    seams = row.get('seams') or {}
    from .seams import COMPOSITION_REVISION
    accepted = (current and row.get('body_render_accepted') is True
                and seams.get('accepted') is True and seams.get('revision') == COMPOSITION_REVISION)
    if accepted:
        try:
            from .certification import require_current
            ident = job['prefab'].split('/', 1)[1]
            composite = (store.read_extra('body', ident) or {}).get('composite') or {}
            require_current(ident, composite)
            accepted = composite['source_certification'] == row.get('source_certification')
        except (LookupError, OSError, ValueError) as exc:
            accepted, reason = False, str(exc)
    previews = row.get('previews') or []
    present = sum(isinstance(name, str) and (output / name).is_file() for name in previews)
    return {'artifact_current': current, 'artifact_issue': reason, 'body_acceptance_current': bool(accepted),
            'present_preview_count': present, 'body_render_accepted': row.get('body_render_accepted')}


def _body_piece(mesh, skeleton) -> Piece:
    names = [bone.name for bone in skeleton.bones]
    worlds = np.stack([bone.bind_world for bone in skeleton.bones]).astype(np.float32)
    skin = np.c_[mesh.bone_wt, mesh.bone_idx].astype(np.float32)
    return Piece(mesh, names, skin, np.linalg.inv(worlds).astype(np.float32), bone_world=list(worlds))


def _save_body_arrays(job: dict, render, source_mesh, skeleton, body_extra: dict) -> None:
    """Keep fitted source editable even when a seam cannot be accepted."""
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    arrays = output / "body_arrays.npz"
    np.savez_compressed(arrays, positions=render.positions, normals=render.normals,
                        uvs=render.uvs, indices=render.indices,
                        bone_idx=render.bone_idx, bone_wt=render.bone_wt, colors=render.colors,
                        source_positions=source_mesh.positions, source_indices=source_mesh.indices,
                        source_uvs=source_mesh.uvs, source_bone_idx=source_mesh.bone_idx,
                        source_bone_wt=source_mesh.bone_wt)
    ident = job["prefab"].split("/", 1)[1]
    from .body_archive import export_original_archive
    job.update(export_original_archive(ident, output))
    job.update(body_arrays=str(arrays), body_bones=[b.name for b in skeleton.bones],
               body_texture=str(store.home(ident) / "body" / ident / "extra_albedo.png"),
               body_source_vertices=source_mesh.vertex_count,
               body_graft=body_extra.get("body_graft", []),
               source_piece_ranges=body_extra["source_piece_ranges"])
    body_item = store.read_item("body", ident)
    if body_item is not None and "normal" in body_item.maps:
        native_item = store.read_item("chara", ident)
        shader = native_skin_shader(native_item.pieces) if native_item is not None else None
        if shader is None:
            raise ValueError(f"body/{ident}: source normal has no native skin shader")
        job.update(body_normal_texture=str(store.home(ident) / "body" / ident / "extra_normal.png"),
                   body_normal_shader=shader)
    else:
        job.pop("body_normal_texture", None)
        job.pop("body_normal_shader", None)


def _painted_as_outfit(piece, roles: list, face_skin: np.ndarray | None = None) -> list:
    """*roles* with a garment painted into a native skin part's texture given to the outfit.

    (:func:`part_roles.painted_garment_faces`.)  Under such faces there is no
    skin to show, so their island is not an opening the lent body is fitted
    to: the body runs on beneath it, and it comes off like any other garment.
    *face_skin* (:func:`facial_skin`) is the skin colour the rule falls back
    on when most of the part is painted.
    """
    descriptor = ((piece.shader or {}).get("maps") or {}).get("_BaseMap")
    skin = [row for row in roles if row["role"] == "native_skin"]
    if not skin or descriptor is None:
        return roles
    texture = store.read_texture(descriptor["cab"], descriptor["name"])
    if texture is None:
        raise FileNotFoundError(descriptor)
    from .skin_color import uv_colors
    triangles = piece.mesh.indices.reshape(-1, 3)
    faces = sorted(face for row in skin for face in row["faces"])
    painted = set(painted_garment_faces(piece.mesh.positions, triangles, piece.mesh.uvs, faces,
                                        lambda uvs: uv_colors(texture, uvs), face_skin).tolist())
    if not painted:
        return roles
    # the island that carries the garment goes with it, bare skin and all: a garment painted on a
    # leg is cut by the texture, not along the mesh's edges, and the skin left around it would be a
    # ragged cuff standing off the body that is shown once the garment is gone
    return [{**row, "role": "outfit",
             "classification_evidence": "skin island carrying a garment painted into the skin texture: texels "
                                        "off the part's skin colour, in a region that wraps the body"}
            if row["role"] == "native_skin" and painted & set(row["faces"]) else row for row in roles]


def facial_skin(pieces, facial_records: list) -> np.ndarray | None:
    """The character's facial skin colour (linear RGB): the median over her facial faces' centres.

    Each face is sampled in its own material's base map times its
    ``_BaseColor`` (``skin_color._material_colors``).  ``None`` without a
    facial surface or when no facial piece has a base map.
    """
    from .skin_color import _material_colors
    colours = []
    for record in facial_records:
        piece = pieces[record["piece"]]
        if "_BaseMap" not in ((piece.shader or {}).get("maps") or {}):
            continue
        corners = piece.mesh.indices.reshape(-1, 3)[np.asarray(record["faces"], dtype=np.int64)]
        colours.append(_material_colors(piece.shader, piece.mesh.uvs[corners].mean(axis=1)))
    return np.median(np.concatenate(colours), axis=0) if colours else None


#: A joint this many bones or more below the nearest one the humanoid table names is soft tissue's
#: (a breast's chain runs 5 to 12 deep; a limb's helpers 1 or 2, a finger's last joint 3).
SOFT_DEPTH = 5
#: Lent skin this near an island's own surface lies under it, and is not what its rim is joined to.
SOFT_UNDER = 0.005
#: A run of native skin smaller than this many faces, left when the part over soft tissue is taken
#: out, is a crumb and goes with it.
SOFT_CRUMB = 16
#: Rings of faces taken with the part over soft tissue, so the new rim stands clear of it: the lent
#: body is fitted to the rim, and a rim at the tissue's very edge has the fit pull on it.  Measured on
#: Zhuangfy (§7.967): the composer's first fit leaves 8 crossings with none, 8 with one, nothing with
#: two; three reach her neck's rim and leave the head nothing to attach to.
SOFT_MARGIN = 2
#: How near a rim the lent skin is looked at, and the share of it on soft tissue's joints from which
#: the rim cannot stand there.  Measured over the four bodies' seventeen seams (§7.964): fifteen
#: are at 0%, Avywen's collar at 25.6% (it crosses the top of the chest and looks right), and the
#: rim of Zhuangfy's upper-chest skin, which runs across both breasts, at 78.8%.
SOFT_NEAR = 0.03
SOFT_SHARE = 0.5


def lent_softness(character: str, source_mesh) -> tuple | None:
    """``(positions, soft)`` of the lent body for :func:`seam_inputs`: which of its vertices are soft tissue.

    A vertex is soft when the joint it hangs on most, on the rig the body
    came from, is :data:`SOFT_DEPTH` bones or more below the nearest bone
    the humanoid table names — the far end of a chain that is no limb.
    *source_mesh* is the fitted body (its vertices are the source body's, in
    order); None when the body it was fitted from is not in the store.
    """
    from ..mesh.skin import split_archive_skin
    from .hrole import H_BONES, SOURCE_BODY, stand_in_skeleton
    stored = store.read_item("body_source", character)
    piece = next((p for p in (stored.pieces if stored is not None else []) if p.mesh.name == SOURCE_BODY), None)
    if piece is None:
        return None
    rig = stand_in_skeleton(getattr(stored, "implant", None))
    named = {rig.index[name] for name in H_BONES.values() if rig.has(name)}
    depth = np.zeros(len(rig.bones), dtype=np.int64)
    for index, bone in enumerate(rig.bones):                 # parents come first
        depth[index] = 0 if index in named or bone.parent < 0 else depth[bone.parent] + 1
    count = len(piece.binds)
    weights, indices = split_archive_skin(np.asarray(piece.skin, dtype=np.float64), bone_count=count)
    heaviest = indices[np.arange(len(indices)), np.argmax(weights, axis=1)]
    of_bone = np.array([depth[rig.index[name]] if rig.has(name) else 0 for name in piece.bones[:count]])
    soft = of_bone[heaviest] >= SOFT_DEPTH
    return source_mesh.positions[:len(soft)].astype(np.float64), soft


def _soft_rimmed_as_outfit(mesh, roles: list, lent: tuple | None) -> list:
    """*roles* with a native skin island whose rim runs across the lent body's soft tissue given to the outfit.

    The lent body is cut to an island's rim and bridged to it.  A rim round a
    neck, an arm or a leg meets the same limb on the lent body.  A rim across
    a breast does not: the character's own skin there is a sheet shaped for
    what it wore (Zhuangfy's upper chest is flat down to where her top
    began), and the lent breast is cut in half by it — a shelf across the
    chest when she is undressed.  The part of such an island that lies over
    the soft tissue is not skin the body can be joined to: it goes to the
    outfit, comes off with it, and the body shows through; the rest of the
    island (her shoulders and neck) stays skin, with its rim above the breast.

    *mesh* is the piece's rest mesh, *lent* :func:`lent_softness`.
    """
    from scipy.spatial import cKDTree

    from .mesh_cut import tidy
    if lent is None or not any(row["role"] == "native_skin" for row in roles):
        return roles
    positions, soft = lent
    tree = cKDTree(positions)
    triangles = mesh.indices.reshape(-1, 3).astype(np.int64)
    _, welded = np.unique(np.round(mesh.positions.astype(np.float64) / 1e-6).astype(np.int64), axis=0,
                          return_inverse=True)
    welded = welded.reshape(-1)
    place = np.zeros((int(welded.max()) + 1, 3))
    place[welded] = mesh.positions.astype(np.float64)
    # the rims of the skin as a whole: an island is a run of faces by their own indices, and a body
    # split down its middle for its UVs is two of them with a "rim" along the split that is none
    skin = [row for row in roles if row["role"] == "native_skin"]
    corners = welded[triangles[np.concatenate([np.asarray(row["faces"], dtype=np.int64) for row in skin])]]
    own = cKDTree(place[np.unique(corners)])
    edges = np.sort(np.concatenate([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    rim = unique[count == 1]
    crossed: set[int] = set()
    for loop in (connected_faces(np.column_stack([rim, rim[:, :1]])) if len(rim) else []):
        places = np.unique(rim[loop])
        near = tree.query_ball_point(place[places], SOFT_NEAR)
        beside = np.unique(np.concatenate([np.asarray(n, dtype=np.int64) for n in near]))
        if len(beside):
            beside = beside[own.query(positions[beside])[0] > SOFT_UNDER]         # past the rim, not under it
        if len(beside) >= 8 and soft[beside].mean() > SOFT_SHARE:
            crossed.update(places.tolist())
    if not crossed:
        return roles
    # Not the whole island: the same skin runs on up to the neck, which the head is attached to.
    # What goes to the outfit is the part that lies over the soft tissue (the lent vertex nearest a
    # face's middle is soft), with the crumbs that leaves; the rim that remains is above it.
    out = []
    for row in roles:
        faces = np.asarray(row["faces"], dtype=np.int64)
        on_crossed = row["role"] == "native_skin" and bool(crossed & set(welded[triangles[faces]].ravel().tolist()))
        if not on_crossed and not (row["role"] == "native_skin" and _joined(row, roles, crossed, welded, triangles)):
            out.append(row)
            continue
        middle = place[welded[triangles[faces]]].mean(axis=1)
        # along the mesh's own quads, teeth taken off: a seam has to follow the rim this leaves
        over = tidy(mesh.positions, triangles, faces, soft[tree.query(middle)[1]], SOFT_MARGIN)
        keep = faces[~over]
        kept = []
        for part in (connected_faces(welded[triangles[keep]]) if len(keep) else []):
            (kept if len(part) >= SOFT_CRUMB else []).append(keep[part])
        kept = np.sort(np.concatenate(kept)) if kept else np.zeros(0, dtype=np.int64)
        gone = np.setdiff1d(faces, kept)
        if len(kept):
            out.append({**row, "faces": kept.tolist(), "vertices": np.unique(triangles[kept]).tolist()})
        if len(gone):
            out.append({**row, "role": "outfit", "faces": gone.tolist(),
                        "vertices": np.unique(triangles[gone]).tolist(),
                        "classification_evidence": "native skin lying over the lent body's soft tissue, on an "
                                                   "island whose rim ran across it"})
    return out


def _joined(row: dict, roles: list, crossed: set, welded: np.ndarray, triangles: np.ndarray) -> bool:
    """Whether *row* shares a place with a native skin row that touches a *crossed* rim (the other half of it)."""
    mine = set(welded[triangles[np.asarray(row["faces"], dtype=np.int64)]].ravel().tolist())
    for other in roles:
        if other is row or other["role"] != "native_skin":
            continue
        theirs = set(welded[triangles[np.asarray(other["faces"], dtype=np.int64)]].ravel().tolist())
        if theirs & crossed and theirs & mine:
            return True
    return False


def neck_ring_faces(triangles, skin_faces, head_faces) -> np.ndarray:
    """Head-labelled skin faces that meet the native skin, as one neck ring.

    A face bound only to the head is labelled head. When a connected patch of
    those faces shares a vertex with this piece's native skin, it is the neck
    of that skin. The seam is cut with the ring still on the skin: taking it
    off leaves the body's neck opening short of the collar. A head patch that
    does not meet the skin stays off the seam.
    """
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    skin_faces = np.asarray(list(skin_faces), dtype=np.int64)
    head_faces = np.asarray(list(head_faces), dtype=np.int64)
    if not len(skin_faces) or not len(head_faces):
        return np.zeros(0, dtype=np.int64)
    skin_vertices = set(np.unique(triangles[skin_faces]).tolist())
    parent = {int(face): int(face) for face in head_faces}

    def find(face):
        while parent[face] != face:
            parent[face] = parent[parent[face]]
            face = parent[face]
        return face

    owner = {}
    for face in head_faces:
        for vertex in triangles[int(face)]:
            other = owner.setdefault(int(vertex), int(face))
            a, b = find(int(face)), find(other)
            parent[max(a, b)] = min(a, b)
    groups = {}
    for face in head_faces:
        groups.setdefault(find(int(face)), []).append(int(face))
    kept = []
    for faces in groups.values():
        if set(np.unique(triangles[faces]).tolist()) & skin_vertices:
            kept.extend(faces)
    return np.asarray(sorted(kept), dtype=np.int64)


def _keep_neck_ring(piece, roles: list) -> list:
    """Move a neck ring from the head label back onto the native skin."""
    triangles = piece.mesh.indices.reshape(-1, 3)
    skin = [face for row in roles if row["role"] == "native_skin" for face in row["faces"]]
    head = [face for row in roles
            if row["role"] == "head" and row.get("shader") == "HGRP/CharacterNPR_Skin"
            for face in row["faces"]]
    ring = set(neck_ring_faces(triangles, skin, head).tolist())
    if not ring:
        return roles
    kept = []
    for row in roles:
        if row["role"] != "head" or row.get("shader") != "HGRP/CharacterNPR_Skin":
            kept.append(row)
            continue
        stay = [face for face in row["faces"] if face not in ring]
        if stay:
            kept.append({**row, "faces": stay,
                         "vertices": np.unique(triangles[stay]).tolist()})
    faces = np.asarray(sorted(ring), dtype=np.int64)
    kept.append({"role": "native_skin", "faces": faces.tolist(),
                 "vertices": np.unique(triangles[faces]).tolist(), "shader": "HGRP/CharacterNPR_Skin",
                 "classification_evidence": "head-weighted skin sharing the native skin boundary, "
                                            "kept as its neck ring"})
    return kept


#: The body regions a native skin island stands on for it to be skin shown between garments
#: (:func:`_framed_as_outfit`): limbs and the torso.  Hands, feet, the head and the neck are the
#: character's own shape, and the lent body is joined to them there.  The shoulders are not: their
#: skin runs to the collar, where the lent body meets the head through it (taken, Zhuangfy's body
#: could not be composed: "added Head collar lacks certified authored attachment", §7.1107).
FRAMED_REGIONS = frozenset({"UpperChest", "Chest", "Spine", "Hips",
                            "LeftUpperArm", "LeftLowerArm", "RightUpperArm", "RightLowerArm",
                            "LeftUpperLeg", "LeftLowerLeg", "RightUpperLeg", "RightLowerLeg"})
#: How near the head skin an island's rim lies when the island joins it (m).  Over the five composed
#: bodies (§7.1107) rims that join the head lie 9.7-16.3 mm from it (median over the rim) and the
#: nearest that does not lies 37 mm off: Avywen's shoulders carry her neck to the head and stay.
JOINS_HEAD = 0.025


def _rims(positions: np.ndarray, triangles: np.ndarray) -> list:
    """The boundary loops of *triangles* (``(F, 3)``, UV seams welded) as arrays of positions."""
    _, welded = np.unique(np.round(np.asarray(positions, dtype=np.float64) / 1e-6).astype(np.int64),
                          axis=0, return_inverse=True)
    corners = welded[triangles]
    edges = np.sort(np.concatenate([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    rim = unique[count == 1]
    if not len(rim):
        return []
    first = {int(w): int(i) for i, w in enumerate(welded)}
    return [np.asarray(positions, dtype=np.float64)[[first[int(w)] for w in np.unique(rim[component])]]
            for component in connected_faces(np.column_stack([rim, rim[:, :1]]))]


def _framed_as_outfit(mesh, roles: list, regions: np.ndarray, head, lent=frozenset()) -> list:
    """*roles* with native skin shown only between garments given to the outfit.

    A band of the character's own skin on a limb or the torso, between a
    sleeve and a glove or shorts and stockings, is what the outfit left
    bare: its texture is painted with the garments' shadows and it stands
    off the lent body (Typhoea's thighs 8-16 mm, §7.1085).  Kept as skin it
    showed as a darker, thicker cuff with a seam on each side once the
    clothes came off (owner, 2026-10-16).  Given to the outfit it is drawn
    while dressed and comes off with the clothes, the lent body running on
    beneath it.  An island stays skin when it stands on a hand, a foot, the
    head or the neck (*regions*: per face, a :data:`part_roles.REGIONS`
    index), or when a rim of it joins the head skin (*head*: a tree of the
    head skin's positions; :data:`JOINS_HEAD`) — the lent body is joined to
    the head through it.  *lent* (the item table's ``lent_regions``, a
    converted request's declaration) names hands or feet whose own skin goes
    to the outfit as well: cut to sit in her shoes, it is no foot to show,
    and the lent body's is shown there instead.
    """
    triangles = mesh.indices.reshape(-1, 3).astype(np.int64)
    out = []
    for row in roles:
        if row["role"] != "native_skin":
            out.append(row)
            continue
        faces = np.asarray(row["faces"], dtype=np.int64)
        named = [REGIONS[r] for r in regions[faces] if r >= 0]
        on = max(set(named), key=named.count) if named else ""
        joins = head is not None and any(float(np.median(head.query(rim)[0])) < JOINS_HEAD
                                         for rim in _rims(mesh.positions, triangles[faces]))
        if on in lent and not joins:
            out.append({**row, "role": "outfit",
                        "classification_evidence": f"native skin on the {on}, a region the item's table lends "
                                                   "(lent_regions): the lent body is shown there instead"})
        elif on in FRAMED_REGIONS and not joins:
            out.append({**row, "role": "outfit",
                        "classification_evidence": f"native skin on the {on} shown between garments, joined to no "
                                                   "head skin: the lent body runs on beneath it"})
        else:
            out.append(row)
    return out


#: what native skin a rebuild gives away, and the evidence each row is recorded with
GIVEN_AWAY = {
    "head": "head-bound native skin the lent body cannot reach, kept with the head",
    "outfit": "native skin the lent body cannot be joined to, taken off with the outfit",
}


def _with_role(piece, roles: list, faces, role: str) -> list:
    """*roles* with native skin *faces* given to *role* (:func:`unjoinable_skin`)."""
    faces = {int(face) for face in faces}
    if not faces:
        return roles
    triangles = piece.mesh.indices.reshape(-1, 3)
    kept, moved = [], []
    for row in roles:
        if row["role"] != "native_skin" or not faces & set(row["faces"]):
            kept.append(row)
            continue
        stay = [face for face in row["faces"] if face not in faces]
        # only faces that are native skin here move: one another rule already gave away
        # (the neck ring, a painted garment) would otherwise be listed twice
        moved.extend(face for face in row["faces"] if face in faces)
        if stay:
            kept.append({**row, "faces": stay, "vertices": np.unique(triangles[stay]).tolist()})
    if not moved:
        return kept
    moved = sorted(moved)
    kept.append({"role": role, "faces": moved, "vertices": np.unique(triangles[moved]).tolist(),
                 "shader": "HGRP/CharacterNPR_Skin", "classification_evidence": GIVEN_AWAY[role]})
    return kept


#: composer refusals that say an island of native skin cannot be joined to the lent body as it stands
UNJOINABLE = (
    "native surface lacks nearby added correspondence",
    "projected opening boundary crosses",
    "opening cap has no clear triangulation in its charts",
    "native footprint creates unmatched added openings",
    "missing or ambiguous added cut opening",
)


def unjoinable_skin(inputs: dict, seams: dict, item) -> dict:
    """Native skin islands the composer refused to join: ``{"head" | "outfit": {piece index: faces}}``.

    An island refused for one of :data:`UNJOINABLE` is not the lent body's to
    meet.  Its faces bound mostly to the head (mean head mass of their
    corners at least one half) are the head's skin and go with the head
    (Wulfa: a 10-face band 96% on the head at the nape, 63 mm from the lent
    body; with the §7.1051 ring, the back of her head too).  The rest is
    given to the outfit, as skin lying over
    the lent body's soft tissue already is (§7.967): it is drawn while the
    clothes are on and the lent skin under it shows when they come off
    (Wulfa's palms: an opening zigzagging between the fingers that no flat
    chart shows simply).  Only a piece's own native skin, not the head's
    retained neck regions.
    """
    from .seams import _authored_facial_surface, _islands
    head = inputs["head_bones"]
    facial = [entry for entry in inputs.get("head_surfaces", []) if _authored_facial_surface(entry)]
    face_points = (np.concatenate([np.asarray(entry["mesh"].positions, dtype=np.float64)[
        np.unique(entry["mesh"].indices.reshape(-1, 3)[entry["faces"]])] for entry in facial])
        if facial else np.zeros((0, 3)))
    out: dict[str, dict[int, set]] = {"head": {}, "outfit": {}}
    for issue in seams.get("unresolved") or []:
        if issue.get("reason") not in UNJOINABLE or "island_index" not in issue:
            continue
        row = inputs["native"][issue["piece_index"]]
        piece = row.get("piece")
        if piece is None or not any(piece is p for p in item.pieces):
            continue
        mesh = row["mesh"]
        island = np.asarray(_islands(mesh, row["faces"])[issue["island_index"]])
        mass = np.sum(mesh.bone_wt * np.isin(mesh.bone_idx, list(head)), axis=1)
        # face by face: a neck island refused whole still has its upper part on the head
        # (Typhoea: neck skin 27% on the head over all, the part behind the head wholly on it)
        on_head = mass[mesh.indices.reshape(-1, 3)[island]].mean(axis=1) >= 0.5
        if _touches(mesh, island, face_points):
            # an island the face's own edge runs into continues the face: all of it is the head's,
            # or the head's opening runs up the back of the head (Typhoea: the face is open under
            # the hair and her neck carries on from it; split by weight, the part on the head lay
            # 57 mm from the face and the collar was pulled up to 1.42 m, §7.1065)
            on_head[:] = True
        index = next(i for i, p in enumerate(item.pieces) if p is piece)
        for role, faces in (("head", island[on_head]), ("outfit", island[~on_head])):
            if len(faces):
                out[role].setdefault(index, set()).update(faces.tolist())
    return out


def _touches(mesh, faces, points: np.ndarray) -> bool:
    """Whether *faces* of *mesh* share at least two of *points* (the composer's weld, :data:`SAME_POINT`)."""
    if not len(points) or not len(faces):
        return False
    corners = np.asarray(mesh.positions, dtype=np.float64)[np.unique(mesh.indices.reshape(-1, 3)[faces])]
    keys = {tuple(key) for key in np.round(points / SAME_POINT).astype(np.int64).tolist()}
    return sum(tuple(key) in keys for key in np.round(corners / SAME_POINT).astype(np.int64).tolist()) >= 2


def _sub_mesh(mesh, faces):
    """The part of *mesh* its *faces* use, as a mesh of its own (every per-vertex stream kept)."""
    from ..mesh.mesh import Mesh
    part = mesh.copy()
    part.indices = part.indices.reshape(-1, 3)[np.asarray(faces, dtype=np.int64)].ravel().copy()
    vertices, inverse = np.unique(part.indices, return_inverse=True)
    count = part.vertex_count
    for field in ('positions', 'normals', 'uvs', 'colors', 'bone_idx', 'bone_wt',
                  *[name for name, _ in Mesh.OPTIONAL_STREAMS]):
        stream = getattr(part, field)
        if len(stream) == count:
            setattr(part, field, stream[vertices].copy())
    part.indices = inverse.astype(np.uint32)
    return part


#: how close two pieces' points are to be the same point of one surface (the composer's own weld)
SAME_POINT = 1e-6


def _join_head_skin(head_surfaces: list, face):
    """The face and the head skin that continues it, as one surface: ``(head surfaces, face reference)``.

    A head can be two meshes: the face, and skin bound to the head in the
    body's mesh that carries on from the face's edge (Wulfa: the back of the
    head down to the nape, 130 faces).  The opening a lent collar is fitted
    to is the lower edge of the two together; the face's edge alone runs up
    the back of the head, and a collar drawn there is pulled 17 cm (§7.1060).
    Head skin that shares at least two points with the face (the composer's
    own weld) joins the facial surface it touches and the face reference.
    Nothing joins when the head skin touches more than one facial surface.
    The joined entry keeps the face's ``shader``; ``vertex_materials``
    (``[(first vertex, end, shader)]``) says which material draws each of
    its vertices, so a colour sampled there uses the texture its UVs address.
    """
    from ..mesh.mesh import Mesh
    from .part_roles import DECLARED_ROLE, HEAD_SKIN_CONTINUATION
    from .seams import _authored_facial_surface
    facial = [i for i, entry in enumerate(head_surfaces) if _authored_facial_surface(entry)]
    if not facial:
        return head_surfaces, face
    # the face the head skin carries on from: of the facial surfaces (a face, and brows drawn
    # as skin — Typhoea has the two), the one it shares points with (§7.1065)
    faces_of = {i: _sub_mesh(head_surfaces[i]["mesh"], head_surfaces[i]["faces"]) for i in facial}
    joined, parts, onto = [], [], []
    for i, entry in enumerate(head_surfaces):
        shader = entry.get("shader") or {}
        declared_skin = (shader.get(DECLARED_ROLE) == "head"
                         and shader.get(HEAD_SKIN_CONTINUATION) is True)
        if i in faces_of or (shader.get("_shader") != "HGRP/CharacterNPR_Skin" and not declared_skin):
            continue
        part = _sub_mesh(entry["mesh"], entry["faces"])
        for k, face_part in faces_of.items():
            gap = np.sqrt(((part.positions[:, None, :].astype(np.float64)
                            - face_part.positions[None, :, :].astype(np.float64)) ** 2).sum(2)).min(1)
            if int((gap <= SAME_POINT).sum()) >= 2:
                joined.append(i)
                parts.append(part)
                onto.append(k)
                break
    if not joined or len(set(onto)) != 1:
        return head_surfaces, face
    facial = [onto[0]]
    target = head_surfaces[facial[0]]
    face_part = faces_of[facial[0]]
    merged = Mesh("head_skin")
    merged.append(face_part)
    reference = Mesh("native_face_reference")
    reference.append(face)
    # each joined part keeps its own material: its UVs address its own texture, not the face's
    materials = [(0, face_part.vertex_count, target["shader"])]
    for i, part in zip(joined, parts):
        materials.append((merged.vertex_count, merged.vertex_count + part.vertex_count, head_surfaces[i]["shader"]))
        merged.append(part)
        reference.append(part)
    entry = {**target, "mesh": merged, "faces": list(range(merged.tri_count)),
             "joined_head_skin": [head_surfaces[i]["piece_index"] for i in joined],
             "vertex_materials": materials}
    kept = [e for i, e in enumerate(head_surfaces) if i not in joined and i != facial[0]]
    return kept + [entry], reference


def seam_inputs(item, skeleton, extra: dict, lent: tuple | None = None, given: dict | None = None) -> dict:
    """What ``seams.compose_skin`` is given for one character, from its stored original.

    The native skin islands (with the head's retained neck regions after
    them), the head surfaces, the face reference and the head's bones, and
    the classification they came from.  One owner: the build and anything
    that composes a stored body again read the same inputs.  *lent*
    (:func:`lent_softness`) is the body the islands will be joined to; with
    it an island it cannot be joined to goes to the outfit.  *given*
    (``{"head" | "outfit": {piece index: faces}}``, :func:`unjoinable_skin`)
    is native skin a rebuild found the lent body cannot be joined to.  Head
    skin continuing the face joins it (:func:`_join_head_skin`).
    """
    from .seams import native_head_neck_regions
    face_evidence = None if has_sdf_face(item.pieces) else facial_evidence(extra)
    facial_records = facial_reference_islands(item.pieces, skeleton, extra, face_evidence)
    facial_faces = {}
    for record in facial_records:
        if record['evidence'] != 'authored facial SDF':
            facial_faces.setdefault(record['piece'], []).extend(record['faces'])
    native, head_surfaces, groups, evidence = [], [], {}, {}
    face_skin = facial_skin(item.pieces, facial_records)
    classified = []
    for piece_index, piece in enumerate(item.pieces):
        roles = classify_piece(piece, skeleton, extra["humanoid"],
                               facial_faces=facial_faces.get(piece_index, ()))
        roles = _painted_as_outfit(piece, roles, face_skin)
        roles = _soft_rimmed_as_outfit(_rest_mesh(piece, skeleton), roles, lent)
        classified.append(_keep_neck_ring(piece, roles))
    from scipy.spatial import cKDTree
    head_points = [_rest_mesh(piece, skeleton).positions[np.unique(piece.mesh.indices.reshape(-1, 3)[row["faces"]])]
                   for piece, roles in zip(item.pieces, classified) for row in roles if row["role"] == "head"]
    head = cKDTree(np.concatenate(head_points).astype(np.float64)) if head_points else None
    for piece_index, (piece, roles) in enumerate(zip(item.pieces, classified)):
        name = f"{piece.shader['_rendName']}#{piece.mat_slot}"
        mesh = _rest_mesh(piece, skeleton)
        roles = _framed_as_outfit(mesh, roles, face_regions(mesh, skeleton, extra["humanoid"]), head,
                                  frozenset(extra.get("lent_regions") or ()))
        for role in GIVEN_AWAY:
            roles = _with_role(piece, roles, ((given or {}).get(role) or {}).get(piece_index, ()), role)
        grouped = {}
        for island in roles:
            grouped.setdefault(island["role"], []).extend(island["faces"])
        groups[name] = grouped
        evidence[name] = roles
        if grouped.get('head'):
            head_surfaces.append({'mesh': _rest_mesh(piece, skeleton), 'faces': grouped['head'],
                                 'piece_index': piece_index, 'piece': piece, 'shader': piece.shader,
                                 'facial_evidence': next((record for record in facial_records
                                                          if record['piece'] == piece_index), None)})
        if grouped.get("native_skin"):
            native.append({"mesh": _rest_mesh(piece, skeleton), "piece": piece,
                           "faces": grouped["native_skin"]})
    face = face_reference(item.pieces, skeleton, extra, face_evidence)
    head = extra['humanoid'].get('Head')
    head_bones = set(skeleton.descendants(head)) if skeleton.has(head) else set()
    retained_necks = native_head_neck_regions(head_surfaces, skeleton, extra['humanoid'])
    native.extend(retained_necks)
    head_surfaces, face = _join_head_skin(head_surfaces, face)
    return {
        'native': native, 'head_surfaces': head_surfaces, 'groups': groups, 'evidence': evidence,
        'face': face, 'head_bones': head_bones,
        'facial_context': facial_context(skeleton, extra, face_evidence) if face_evidence else None,
        'facial_evidence_files': (face_evidence or {}).get('source_evidence_files', []),
        'native_head_neck_evidence': [
            {key: value for key, value in region.items() if key not in ('mesh', 'piece', 'faces')}
            for region in retained_necks]}


def prepare_body(job: dict, card: Path) -> None:
    """Make source and seam-fitted render bodies, retaining both in the store."""
    from .seams import compose_skin
    ident = job["prefab"].split("/", 1)[1]
    extra = store.read_extra("chara", ident) or {}
    require_body_scope(extra, ident)
    bake_body(ident, card)
    item = store.read_item("chara", ident)
    body_item = store.read_item("body", ident)
    skeleton = build_skeleton(body_item.nodes, ident)
    raw = body_item.pieces[0]
    source_mesh = _rest_mesh(raw, skeleton)
    inputs = seam_inputs(item, skeleton, extra, lent_softness(ident, source_mesh))
    job['facial_context'] = inputs['facial_context']
    job['facial_evidence_files'] = inputs['facial_evidence_files']
    native, head_surfaces, groups, evidence = (inputs['native'], inputs['head_surfaces'],
                                               inputs['groups'], inputs['evidence'])
    face, head_bones = inputs['face'], inputs['head_bones']
    job['native_head_neck_evidence'] = inputs['native_head_neck_evidence']
    render, seams = compose_skin(source_mesh, native, skeleton, face_mesh=face,
                                 head_bones=head_bones, head_surfaces=head_surfaces)
    job["seams"] = seams
    job["classification"] = groups
    job["classification_evidence"] = evidence
    if seams.get("unresolved"):
        job["body_status"] = "unresolved native seams"
        _save_body_arrays(job, source_mesh, source_mesh, skeleton, store.read_extra("body", ident))
        job["body_render_accepted"] = False
        return
    from .skin_color import calibrate_skin
    job["skin_color"] = calibrate_skin(render, native, seams, body_item.maps["albedo"],
                                        head_surfaces=head_surfaces)
    manifest = {"version": 1, "body_triangles": render.tri_count,
                "native_excluded_faces": [], "covered_faces": [], "parts": groups,
              "classification_evidence": evidence,
              "facial_context": job['facial_context'],
              "facial_evidence_files": job['facial_evidence_files'],
                "native_head_neck_evidence": job['native_head_neck_evidence'],
                "render_tints": render.colors.tolist(),
                "source_native_excluded_faces": seams["source_excluded_faces"],
                "seams": seams, "coverage_status": "awaiting garment ray audit"}
    body_extra = store.read_extra("body", ident) or {}
    body_extra["composite"] = manifest
    body_extra["composite_status"] = "seams prepared; coverage awaiting Blender"
    with store.opened(ident, write=True):
        bake.write_item("body", ident, [_body_piece(render, skeleton), raw],
                        source=store.SOURCE, nodes=body_item.nodes, label=ident,
                        extra_maps=body_item.maps)
        bake.write_named_table(store.SOURCE, "body", ident, body_extra)
    # the dressed body's mask, by the rule the run time masks it by.  Measured here and handed to
    # the worker: Blender's Python has no scipy, and its own ray audit is the first rule alone
    from . import pieces
    from .remask import outfit_cover
    job["covered_by_rule"] = outfit_cover(ident, render, skeleton)
    job["coverage_rule"] = pieces.RULE_EXTREMITIES
    _save_body_arrays(job, render, source_mesh, skeleton, body_extra)
    from .certification import capture
    job['body_input_fingerprint'] = capture(ident, manifest)
    job.update(body_status="prepared", body_render_accepted=True)
    job["source_gaps"] = [gap for gap in job["source_gaps"] if gap != "fitted body item missing"]


def apply_coverage(job: dict, report: dict) -> None:
    """Persist the same reversible face mask inspected by the Blender worker."""
    if not job.get("body_arrays") or not job.get("body_render_accepted") or report.get("status") == "failed":
        return
    from .composite import validate_manifest
    ident = job["prefab"].split("/", 1)[1]
    extra = store.read_extra("body", ident)
    manifest = extra["composite"]
    manifest["covered_faces"] = report["covered_faces"]
    from . import pieces
    # the rule the worker's list was cut by: the one the job handed it, or its own ray audit
    manifest["coverage_status"] = report.get("coverage_rule") or pieces.RULE_HALF
    validate_manifest(manifest, manifest["body_triangles"])
    manifest['coverage_verified'] = True
    from .certification import capture
    certificate = capture(ident, manifest)
    prepared = job.get('body_input_fingerprint') or {}
    if any(prepared.get(key) != certificate[key] for key in ('files', 'missing_maps')):
        raise ValueError('body fitting inputs changed during Blender conversion; rebuild this body')
    manifest['source_certification'] = certificate
    report['source_certification'] = manifest['source_certification']
    extra["composite_status"] = "native seams and reversible garment mask prepared"
    with store.opened(ident, write=True):
        bake.write_named_table(store.SOURCE, "body", ident, extra)
    # the rest of the way (``recompose.settle``): the seams fitted again, the source rig's joints
    # kept, and the list measured again on the refitted surface (the refit moves the skin it is
    # measured on, so a few faces along the seams change sides)
    from .recompose import settle
    report["settled"] = settle(ident)
    report["remask"] = report["settled"]["remask"]


def _run_job(job: dict, blender: Path, render_size: int) -> dict:
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    job["render_size"] = render_size
    payload = output / "job.json"
    payload.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    script = Path(__file__).with_name("sample_blender.py")
    process = subprocess.run([str(blender), "-b", "--factory-startup", "--python-exit-code", "1",
                              "--python", str(script), "--", str(payload)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", check=False)
    (output / "blender.log").write_text(process.stdout, encoding="utf-8")
    report_path = output / "validation.json"
    if process.returncode or not report_path.is_file():
        return {"relative_fbx": job["relative_fbx"], "status": "failed",
                "reason": f"Blender exit {process.returncode}; see {output / 'blender.log'}"}
    return json.loads(report_path.read_text(encoding="utf-8"))


def build_samples(source_root: Path, output_root: Path, blender: Path,
                  body_cards: dict | None = None, only: str = "", render_size: int = 320,
                  preview_engine: str = 'auto') -> dict:
    """Continue after each failed input and preserve its concrete error report."""
    inventory = build_inventory(source_root)
    from .sample_support import preview_render_settings
    preview_render_settings(preview_engine, True)
    jobs = jobs_from_inventory(inventory, output_root)
    cards = body_cards or {}
    rows = []
    for job in jobs:
        if only and only.lower() not in job["relative_fbx"].lower():
            continue
        print(f"BUILD {job['relative_fbx']}", flush=True)
        job['preview_engine'] = preview_engine
        try:
            body_allowed = True
            if job["prefab"]:
                ident = job["prefab"].split("/", 1)[1]
                extra = store.read_extra("chara", ident) or {}
                from .body_scope import JUVENILE_TYPES
                from .model import folder_of
                family = folder_of(extra.get("model", "")).split("/", 1)[0]
                if family in JUVENILE_TYPES:
                    try:
                        require_body_scope(extra, ident)
                    except ValueError:
                        body_allowed = False
                        job["body_status"] = "juvenile: original clothed source only"
                        job["source_gaps"] = [g for g in job["source_gaps"] if g != "fitted body item missing"]
            if body_allowed and job["prefab"] and job["prefab"] in cards:
                try:
                    prepare_body(job, Path(cards[job["prefab"]]))
                except (ValueError, LookupError, OSError, KeyError) as exc:
                    job["body_error"] = f"{type(exc).__name__}: {exc}"
                    job["body_status"] = "unresolved body preparation; original clothed source retained"
                    job["body_arrays"] = None
            row = _run_job(job, blender, render_size)
            if job.get("body_error"):
                row["body_error"] = job["body_error"]
                if row["status"] != "failed":
                    row["status"] = "needs_review"
            apply_coverage(job, row)
            if body_allowed and job["prefab"] and job["prefab"] in cards:
                # its body is part of what its origin rig is made from (``origin_rig.refresh``, §7.1232)
                from . import origin_rig
                stands, row["origin_rig"] = origin_rig.refresh(job["prefab"].split("/", 1)[1])
                if not stands and row["status"] != "failed":
                    row["status"] = "needs_review"
        except (ValueError, LookupError, OSError, KeyError) as exc:
            row = {"relative_fbx": job["relative_fbx"], "status": "failed",
                   "reason": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        out = Path(job["output"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"RESULT {row['status']}: {job['relative_fbx']}", flush=True)
    summary = {"version": 1, "actual_fbx_count": inventory["fbx_count"],
               "processed_count": len(rows), "filtered": only, "results": rows}
    output_root.mkdir(parents=True, exist_ok=True)
    name = "summary.json" if not only else "filtered_summary.json"
    (output_root / name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    reconcile_results(inventory, output_root)
    return summary
