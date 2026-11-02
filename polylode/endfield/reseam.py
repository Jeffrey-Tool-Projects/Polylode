"""Fit a stored completed body's seams again without the folds the first fit left.

``seams.compose_skin`` cuts the lent body open at each native skin opening and
carries the cut onto the opening's rim; the displacement is spread over a band
of the body.  It takes the ``nearest`` spreading whenever that passes its
checks, and none of those checks looks at whether the band folded: on a body
whose cut has centimetres to travel (41.5 mm on EndminF's neck, 54.8 mm on
EndminM's) the faces around the seam end up tens of degrees apart and some
doubled over, which is the ring of creases at the base of the neck (§7.925).

This composes the same body with the continuous (``harmonic``) spreading and
then fairs the moved band — each moved point halfway toward the mean of its
neighbours, the rim and everything unmoved staying put — a few steps at a
time.  **Every amount is put through the composer's own checks**, the one for
refitted skin crossing itself included (which the first fit was never asked);
of the amounts that pass, the one whose worst fold is smallest is kept.  When
none passes, a candidate replaces the stored body only if the stored body
fails those checks itself and the candidate fails them on fewer faces
(:func:`least_bad`); what it still fails is recorded with it.  Otherwise
nothing is written.

The body's topology does not change, so its bridges, its face provenance and
its stored outfit mask stay valid; its tints are calibrated again on the new
surface and its certification renewed.
"""
from __future__ import annotations

import copy

import numpy as np

from ..origin import bake
from . import seams as S
from . import store
from .actor import ActorError, build_skeleton

#: How many fairing steps are tried, at most.
STEPS = 16
#: How far the stored body may be from this composer's own first fit and still be its (float32).
SAME_BODY = 1e-4
#: What a refit of this kind is recorded as.
METHOD = "harmonic+fairing"
#: A faired point belongs to a limb seam's tube within this many times the ring's own radius of the axis.
TUBE_REACH = 2.0


def fold_angles(mesh, moved: np.ndarray) -> tuple[float, float, float]:
    """Mean, 95th percentile and largest angle (degrees) between faces sharing an edge.

    Over the faces that touch a *moved* vertex; UV duplicates count as one point.
    """
    tri = mesh.indices.reshape(-1, 3)
    position = mesh.positions.astype(np.float64)
    normal = np.cross(position[tri[:, 1]] - position[tri[:, 0]], position[tri[:, 2]] - position[tri[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    _, welded = np.unique(np.round(position / 1e-6).astype(np.int64), axis=0, return_inverse=True)
    corner = welded[tri]
    faces = np.flatnonzero(moved[tri].any(axis=1))
    edges = np.sort(np.stack([corner[faces][:, [0, 1]], corner[faces][:, [1, 2]],
                              corner[faces][:, [2, 0]]], axis=1), axis=2).reshape(-1, 2)
    owner = np.repeat(faces, 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    pair = np.flatnonzero(np.all(edges[1:] == edges[:-1], axis=1))
    if not len(pair):
        return 0.0, 0.0, 0.0
    cosine = np.einsum("ij,ij->i", normal[owner[pair]], normal[owner[pair + 1]])
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(angle.mean()), float(np.percentile(angle, 95)), float(angle.max())


def stored_folds(stored, first, moved: np.ndarray, apart: float) -> tuple[float, float, float]:
    """:func:`fold_angles` of the *stored* body, asked of the composer's *first* fit when the stored body is it.

    *apart* is how far the two are (:data:`SAME_BODY`).  The stored copy has
    been through float32 once more than the candidates it is weighed against,
    and a crease the fairing never reaches then reads a few ten-thousandths of
    a degree apart: Wulfa's 154.4 degree fold kept every refit out because
    the worst fold is compared first (§7.1289).
    """
    return fold_angles(first if apart <= SAME_BODY else stored, moved)


def _issues(mesh, baseline: dict, source, inputs: dict, kept: int) -> list:
    """What the composer's checks say of *mesh*: ``[(reason, count)]``, empty when it passes."""
    report = copy.deepcopy(baseline)
    report["unresolved"] = []
    S._validate_bridges(mesh, report, kept)
    S._validate_surfaces(mesh, inputs["native"], inputs["face"], report)
    S._head_fit_surfaces(source, source, mesh, inputs["native"], inputs["face"], report, scope="skin")
    reference = S._head_reference(inputs["face"], inputs["head_surfaces"], inputs["head_bones"])
    S._validate_head_attachments(source, mesh, reference, inputs["head_bones"], report)
    return [(issue["reason"], issue.get("intersection_count", issue.get("triangles", 1)))
            for issue in report["unresolved"]]


def _moved(mesh, source) -> np.ndarray:
    moved = np.zeros(mesh.vertex_count, dtype=bool)
    count = source.vertex_count
    moved[:count] = np.any(mesh.positions[:count] != source.positions, axis=1)
    return moved


def _cut_twins(welded: np.ndarray, baseline: dict) -> np.ndarray:
    """Original source vertices sharing an explicit cut constraint, including UV twins."""
    cuts = [vertex for seam in [*baseline.get('seams', []), *baseline.get('head_seams', [])]
            for vertex in seam.get('source_cut_vertices', [])]
    return np.isin(welded, welded[np.asarray(cuts, dtype=np.int64)])


def candidates(source, first, smooth, baseline: dict, inputs: dict):
    """``(steps, mesh, issues, folds)`` for the smooth fit faired 0‥:data:`STEPS` steps."""
    count = source.vertex_count
    kept = int(np.sum(np.asarray(baseline["render_face_source"]) >= 0))
    welded, adjacency = S._surface_graph(source)
    moved = _moved(smooth, source)
    on_bridge = np.zeros(smooth.vertex_count, dtype=bool)
    on_bridge[np.unique(smooth.indices.reshape(-1, 3)[kept:])] = True
    cut_twins = _cut_twins(welded, baseline)
    on_bridge[:count] |= cut_twins
    free_vertex = moved[:count] & ~on_bridge[:count]
    free = np.zeros(len(adjacency), dtype=bool)
    free[welded[np.flatnonzero(free_vertex)]] = True
    free[welded[np.flatnonzero(~free_vertex)]] = False       # a fixed twin fixes the point
    point = np.zeros((len(adjacency), 3))
    point[welded] = smooth.positions[:count].astype(np.float64)
    ids = np.flatnonzero(free)
    around = [np.asarray([other for other, _length in adjacency[i]], dtype=np.int64) for i in ids]
    put = np.flatnonzero(free[welded])
    # the fit's own shading normals at the rim and on the native rings are kept
    same = (on_bridge & np.all(first.positions == smooth.positions, axis=1)
            & np.all(first.normals == smooth.normals, axis=1))
    for steps in range(STEPS + 1):
        if steps and len(ids):
            mean = np.stack([point[n].mean(axis=0) if len(n) else point[i] for i, n in zip(ids, around)])
            point[ids] = 0.5 * point[ids] + 0.5 * mean
        mesh = smooth.copy()
        mesh.positions[put] = point[welded[put]].astype(np.float32)
        S.refresh_skin_normals(mesh, source)
        mesh.normals[same] = smooth.normals[same]
        mesh.normals[:count][cut_twins] = smooth.normals[:count][cut_twins]
        yield steps, mesh, _issues(mesh, baseline, source, inputs, kept), fold_angles(mesh, moved)


def limb_axes(skeleton, humanoid: dict) -> list:
    """``[(start, unit axis, length)]``: each arm and leg segment between its human joints, at rest."""
    from .body import HINGES
    out = []
    for upper, lower in HINGES:
        if upper in humanoid and lower in humanoid and humanoid[upper] in skeleton.index                 and humanoid[lower] in skeleton.index:
            start, end = skeleton.world_position(humanoid[upper]), skeleton.world_position(humanoid[lower])
            out.append((start, (end - start) / np.linalg.norm(end - start), float(np.linalg.norm(end - start))))
    return out


def _round(points: np.ndarray, axis: tuple) -> tuple:
    """``(station 0..1 along the segment, distance from its axis)`` of each point."""
    start, unit, length = axis
    rel = np.asarray(points, dtype=np.float64) - start
    along = rel @ unit
    return along / length, np.linalg.norm(rel - np.outer(along, unit), axis=1)


def restore_tubes(mesh, smooth, source, baseline: dict, axes: list) -> int:
    """Give the faired skin round a limb seam back the distance from the limb's axis it had before fairing.

    Fairing moves each point halfway to its neighbours' mean, which is a curvature flow: on a limb it
    narrows the tube.  Three steps took EndminF's lent forearm from 109 mm round, 20 mm above her own
    wrist, to 100 mm (98 at 15 mm, under her own 103 at the seam) — a neck the fit never had.  Each faired
    point keeps where the fairing put it along and round the axis; only its distance from the axis goes
    back.  Only round the tube a seam closes on (its ring goes round an arm or leg segment: its centre on
    the segment, within half the ring's radius of the axis): the
    nearest seam's ring is the one that moved the point, and points within :data:`TUBE_REACH` of that
    ring's own radius of the axis.  A neck, a palm or a head seam closes round no limb, and its band is
    left as faired.  In place; returns how many vertices moved.
    """
    from scipy.spatial import cKDTree
    count = source.vertex_count
    faired = np.flatnonzero(np.any(mesh.positions[:count] != smooth.positions[:count], axis=1))
    rings = []
    for seam in [*baseline.get("seams", []), *baseline.get("head_seams", [])]:
        cut = np.asarray(seam.get("source_cut_vertices", []), dtype=np.int64)
        if not len(cut):
            continue
        points = smooth.positions[cut].astype(np.float64)
        tube, reach = None, None
        for k, axis in enumerate(axes):
            station, off = _round(points.mean(axis=0)[None], axis)
            radius = float(np.mean(_round(points, axis)[1]))
            # the ring goes round this segment: its centre on the segment, near its axis
            if 0.0 <= station[0] <= 1.0 and off[0] < 0.5 * radius and (tube is None or off[0] < tube[1]):
                tube, reach = (k, float(off[0])), TUBE_REACH * radius
        rings.append((cKDTree(points), None if tube is None else tube[0], reach))
    if not len(faired) or not any(k is not None for _tree, k, _reach in rings):
        return 0
    before = smooth.positions[faired].astype(np.float64)
    nearest = np.argmin(np.stack([tree.query(before)[0] for tree, _k, _reach in rings]), axis=0)
    positions = mesh.positions.astype(np.float64)
    moved = 0
    for ring, (_tree, k, reach) in enumerate(rings):
        if k is None:
            continue
        rows = faired[nearest == ring]
        station, was = _round(smooth.positions[rows], axes[k])
        now_station, now = _round(positions[rows], axes[k])
        keep = (station >= 0.0) & (station <= 1.0) & (was < reach) & (now > 1e-9)
        start, unit, length = axes[k]
        rows, was, now, now_station = rows[keep], was[keep], now[keep], now_station[keep]
        on_axis = start + np.outer(now_station * length, unit)
        positions[rows] = on_axis + (positions[rows] - on_axis) * (was / now)[:, None]
        moved += len(rows)
    mesh.positions = positions.astype(mesh.positions.dtype)
    return moved


#: A crossing of a bridge with the kept skin that :func:`lowered` is there for.
UNDER_BRIDGE = "bridge intersects retained added surface"
#: How far it lowers the skin a step, and how far at most.
LOWER_STEP = 0.0005
LOWER_MOST = 0.008


def lowered(mesh, baseline: dict, source, inputs: dict, moved: np.ndarray) -> tuple | None:
    """``(mesh, folds)`` with the kept skin that pokes through a bridge taken under it; None if that fails.

    A bridge is the strip of faces between the lent skin's cut and the rim of
    the character's own.  Where the lent body swells just inside its cut (a
    collar bone under the edge of a neck's skin) the strip passes through it,
    and no amount of fairing moves that swell: it is not part of what the fit
    moved.  The faces the composer's check names are moved inward along their
    own normals, :data:`LOWER_STEP` at a time and :data:`LOWER_MOST` at most,
    until every check passes.
    """
    kept = int(np.sum(np.asarray(baseline["render_face_source"]) >= 0))
    provenance = np.asarray(baseline["render_face_source"])
    render_of = {int(face): row for row, face in enumerate(provenance) if face >= 0}
    faces = mesh.indices.reshape(-1, 3).astype(np.int64)
    on_bridge = np.zeros(mesh.vertex_count, dtype=bool)
    on_bridge[np.unique(faces[provenance < 0])] = True
    source_welded, _ = S._surface_graph(source)
    on_bridge[:source.vertex_count] |= _cut_twins(source_welded, baseline)
    work = mesh.copy()
    inward = -mesh.normals.astype(np.float64)
    _, welded = np.unique(np.round(mesh.positions.astype(np.float64) / 1e-6).astype(np.int64), axis=0,
                          return_inverse=True)
    welded = welded.reshape(-1)
    for _step in range(round(LOWER_MOST / LOWER_STEP)):
        report = copy.deepcopy(baseline)
        report["unresolved"] = []
        S._validate_surfaces(work, inputs["native"], inputs["face"], report)
        crossed = sorted({int(pair[1]) for issue in report["unresolved"] if issue["reason"] == UNDER_BRIDGE
                          for pair in issue.get("pairs", [])})
        if not crossed:
            S.refresh_skin_normals(work, source)
            work.normals[on_bridge] = mesh.normals[on_bridge]          # the seam's own shading normals are kept
            issues = _issues(work, baseline, source, inputs, kept)
            return None if issues else (work, fold_angles(work, moved | _moved(work, source)))
        rows = np.unique(faces[[render_of[face] for face in crossed if face in render_of]])
        rows = rows[~on_bridge[rows]]                   # the strip's own corners stay where the seam has them
        if not len(rows):
            return None
        # the faces round them go half as far, and theirs a quarter: a dent with a slope, not a pit
        # whose walls cross the skin beside it
        share = np.zeros(work.vertex_count)
        share[rows] = 1.0
        for ring in (0.5, 0.25):
            beside = np.unique(faces[np.isin(faces, np.flatnonzero(share > 0)).any(axis=1) & (provenance >= 0)])
            beside = beside[(share[beside] == 0) & ~on_bridge[beside]]
            share[beside] = ring
        # corners at one place move as one
        at_place = np.zeros(int(welded.max()) + 1)
        np.maximum.at(at_place, welded, share)
        share = np.where(on_bridge, 0.0, at_place[welded])
        positions = work.positions.astype(np.float64)
        positions += inward * (share * LOWER_STEP)[:, None]
        work.positions = positions.astype(work.positions.dtype)
    return None


def chosen(tried: list, lower) -> tuple:
    """``(kept, lowered)``: which of the candidates *tried* is kept, as ``(steps, mesh, folds)``, or None.

    Of those that pass every check, the one with the smallest worst fold
    (then the smaller 95th percentile, then the less fairing).  When none
    passes, a candidate on which only bridges still cross the kept skin is
    handed to *lower* (:func:`lowered` for its mesh), fewest crossings and
    least fairing first, and the first that then passes is kept.

    The range of fairing tried is :data:`STEPS` and no more on purpose:
    fairing smooths everything the fit moved, which reaches well past the
    seam.  With 48 steps to choose from, the smallest worst fold was at 43
    for Avywen and had flattened her breasts (§7.964).
    """
    passing = [(steps, mesh, folds) for steps, mesh, issues, folds in tried if not issues]
    if passing:
        return min(passing, key=lambda row: (row[2][2], row[2][1], row[0])), False
    only_bridges = sorted((row for row in tried if all(reason == UNDER_BRIDGE for reason, _n in row[2])),
                          key=lambda row: (sum(n for _reason, n in row[2]), row[0]))
    for steps, mesh, _issues_left, _folds in only_bridges:
        under = lower(mesh)
        if under is not None:
            return (steps, under[0], under[1]), True
    return None, False


def least_bad(tried, stored_issues: list) -> tuple | None:
    """``(steps, mesh, folds, issues)`` when nothing passes but something beats what is stored.

    A stored body that fails the checks itself (the first fit is never asked
    whether its refitted skin crosses itself) may be replaced by a candidate
    that fails them on **fewer** faces; the fewest wins, then the smaller
    worst fold.  None when the stored body passes, or no candidate fails less.
    """
    count = lambda issues: sum(int(n) for _reason, n in issues)
    limit = count(stored_issues)
    failing = [(steps, mesh, folds, issues) for steps, mesh, issues, folds in tried
               if issues and count(issues) < limit]
    return min(failing, key=lambda row: (count(row[3]), row[2][2], row[2][1], row[0])) if failing else None


def refit(character: str, *, write: bool = False, again: bool = False) -> dict:
    """Fit ``body/<character>``'s seams again; returns what was measured and whether it was written.

    A body already refitted at the current composition revision is left alone unless *again*: then the refit is
    made over (a longer fairing range, say) and kept only when it leaves
    fewer issues than the stored refit has on record, or none and smaller folds.
    """
    from .body import GRAFT, append_body_nodes
    from .certification import capture
    from .samples import _body_piece, _rest_mesh, seam_inputs
    from .skin_color import calibrate_skin
    extra = store.read_extra("chara", character) or {}
    body_extra = store.read_extra("body", character) or {}
    manifest = body_extra.get("composite")
    item = store.read_item("chara", character)
    body_item = store.read_item("body", character)
    if not manifest or item is None or body_item is None or len(body_item.pieces) < 2:
        raise ActorError(f"body/{character}: no composed body with its fitted source in the store")
    skeleton = build_skeleton(item.nodes, character)
    append_body_nodes(skeleton, body_item.nodes)
    source = _rest_mesh(body_item.pieces[1], skeleton)
    stored = _rest_mesh(body_item.pieces[0], skeleton)
    if any(name.startswith(GRAFT) for name in body_item.nodes):
        # the composer knows the source by the character's bones: its kept helper joints folded
        # back for the fit (they are appended after every other bone, so the rest keep their rows)
        from .recompose import without_helpers
        plain = build_skeleton(item.nodes, character)
        append_body_nodes(plain, {name: node for name, node in body_item.nodes.items()
                                  if not name.startswith(GRAFT)})
        without_helpers(source, skeleton, plain)
    from .recompose import anatomy_held, stored_given_skin
    from .samples import lent_softness
    inputs = seam_inputs(item, skeleton, extra, lent_softness(character, source), stored_given_skin(character))
    arguments = {"face_mesh": inputs["face"], "head_bones": inputs["head_bones"],
                 "head_surfaces": inputs["head_surfaces"],
                 "held": anatomy_held(character, source.vertex_count)}
    if manifest["seams"].get("projection_repair"):
        # the stored fit is the composer's projected-contour one; fit that one again, not the plain one
        arguments["project_contours"] = True
    current_revision = manifest["seams"].get("revision") == S.COMPOSITION_REVISION
    redo = (again or not current_revision) and manifest["seams"].get("refit_method") == METHOD
    if manifest["seams"].get("refit_method") == METHOD and not again and current_revision:
        folds = manifest["seams"]["fold_degrees"]
        return {"character": character, "written": False, "stored_method": METHOD,
                "moved_vertices": len(manifest["seams"]["refitted_source_vertices"]),
                "stored_folds": (folds["mean"], folds["p95"], folds["max"]),
                "reason": "already refitted"}
    first, baseline = S._compose_pass(source, inputs["native"], skeleton, **arguments)
    if stored.vertex_count != first.vertex_count or not np.array_equal(stored.indices, first.indices):
        raise ActorError(f"body/{character}: the stored body is not this composer's independent fit "
                         f"(another topology); build the body again instead")
    moved = _moved(first, source)
    apart = float(np.abs(stored.positions - first.positions).max())
    report = {"character": character, "written": False, "moved_vertices": int(moved.sum()),
              "stored_method": manifest["seams"].get("refit_method"),
              "stored_folds": stored_folds(stored, first, moved, apart), "stored_differs_from_first_fit": apart}
    kept = int(np.sum(np.asarray(baseline["render_face_source"]) >= 0))
    # asked of the first fit itself when the stored body is it: the stored copy has been through
    # float32 once more, and a vertex that differs in its last bit would count as refitted
    report["stored_issues"] = (_issues(first, baseline, source, inputs, kept) if apart <= SAME_BODY
                               else [("the stored body is another fit than this composer's first", 1)])
    if redo:
        # the stored body is the earlier refit: what it left is on its record
        report["stored_issues"] = [tuple(issue) for issue in manifest["seams"].get("refit_issues", [])]
    smooth, evidence = S._compose_pass(source, inputs["native"], skeleton, refit_method="harmonic", **arguments)
    if smooth.vertex_count != first.vertex_count or not np.array_equal(smooth.indices, first.indices):
        raise ActorError(f"body/{character}: the continuous fit has another topology than the stored one")
    tried = list(candidates(source, first, smooth, evidence, inputs))
    best, under = chosen(tried, lambda mesh: lowered(mesh, evidence, source, inputs, _moved(mesh, source)))
    left: list = []
    if under:
        report["lowered_under_bridges"] = True
    report["tried"] = [{"steps": steps, "folds": folds, "issues": issues} for steps, _mesh, issues, folds in tried]
    if best is None:
        worse = least_bad(tried, report["stored_issues"])
        if worse is None:
            report["reason"] = "no amount of fairing passes the composer's checks"
            return report
        best, left = worse[:3], worse[3]
    steps, mesh, folds = best
    if not under and not left:
        # the fairing chosen as it is, then the limb tubes it narrowed given back their girth
        # (kept only when the composer's checks still pass)
        tubes = mesh.copy()
        moved_back = restore_tubes(tubes, smooth, source, evidence, limb_axes(skeleton, extra.get("humanoid") or {}))
        if moved_back:
            S.refresh_skin_normals(tubes, source)
            unchanged = np.all(tubes.positions == mesh.positions, axis=1)
            tubes.normals[unchanged] = mesh.normals[unchanged]
            if not _issues(tubes, evidence, source, inputs, kept):
                mesh, folds = tubes, fold_angles(tubes, _moved(tubes, source))
                report["tube_radius_restored"] = moved_back
    report.update(steps=steps, folds=folds, issues_left=left)
    if (current_revision and (folds[2], folds[1]) >= (report["stored_folds"][2], report["stored_folds"][1])
            and not report["stored_issues"]):
        report["reason"] = "the stored body folds no more than the best refit"
        return report
    mesh.bone_idx, mesh.bone_wt, mesh.uvs = stored.bone_idx, stored.bone_wt, stored.uvs
    seams = copy.deepcopy(manifest["seams"])
    seams.update(revision=S.COMPOSITION_REVISION, refit_method=METHOD, fairing_steps=steps,
                 fold_degrees={"mean": folds[0], "p95": folds[1], "max": folds[2]},
                 refit_issues=[list(issue) for issue in left],
                 first_fit_issues=(manifest["seams"].get("first_fit_issues", []) if redo
                                   else [list(issue) for issue in report["stored_issues"]]),
                 refitted_source_vertices=np.flatnonzero(_moved(mesh, source)[:source.vertex_count]).tolist(),
                 tube_radius_restored=int(report.get("tube_radius_restored", 0)),
                 # the first fit's open issues are answered by this fit: what it leaves is ``refit_issues``
                 unresolved=[])
    from .recompose import _refresh_refit_acceptance
    _refresh_refit_acceptance(seams)
    if not seams["accepted"] and manifest["seams"].get("accepted") is True:
        # a body the composer accepted is not replaced by a refit that answers no more of what its
        # first fit left: Wulfa's groin kept its 10 crossings either way, and writing the refit
        # took a loadable body out of the scene (§7.1065)
        report["reason"] = "no refit answers more than the accepted body"
        return report
    for row in seams["seams"]:
        row["refit_method"] = METHOD
    report["skin_color"] = calibrate_skin(mesh, inputs["native"], seams, body_item.maps["albedo"],
                                          head_surfaces=inputs["head_surfaces"])
    if not write:
        return report
    manifest["seams"] = seams
    manifest["render_tints"] = mesh.colors.tolist()
    with store.opened(character, write=True):
        bake.write_item("body", character, [_body_piece(mesh, skeleton), body_item.pieces[1]],
                        source=store.SOURCE, nodes=body_item.nodes, label=character,
                        extra_maps=body_item.maps)
    manifest["source_certification"] = capture(character, manifest)
    with store.opened(character, write=True):
        bake.write_named_table(store.SOURCE, "body", character, body_extra)
    report["written"] = True
    if manifest.get("coverage_verified") is True:
        # its surface moved; a mask of the rule's own follows it.  A body composed just now is
        # not loadable yet: ``recompose.finish`` measures its mask once it accepts the refit
        from .recompose import first_mask
        first_mask(character)
    return report
