"""Compose a stored completed body again, when what counts as its native skin has changed.

A completed body is the lent body cut to the openings of the character's own
skin and bridged to them (``seams.compose_skin``).  Which faces are that skin
is a classification (``samples.seam_inputs``); when the rule behind it changes
for a character — a garment painted into her skin texture is clothing, not
skin (``part_roles.painted_garment_faces``) — the openings are elsewhere and
the cut has to be made again.

This does it from what the store holds, with the composer the build uses: the
fitted source body (its helper joints folded back onto the bones they hang
under, which is how the composer knows it), the same inputs, the same checks.
What follows a build follows this too, in this order::

    python tools/refit_endfield_body_seams.py --only <character> --write
    python tools/keep_endfield_body_joints.py --only <character> --write
    python tools/remask_endfield_body.py --only <character> --write

(the first is run here already when the first fit's bridges cross something:
a body does not load with a seam issue open).
"""
from __future__ import annotations

import numpy as np

from ..origin import bake
from . import pieces, store
from . import seams as S
from .actor import ActorError, _rest_mesh, build_skeleton
from .body import GRAFT, append_body_nodes

#: ``composite["mask_owner"]`` of a body whose mask this module measured
MASK_OWNER = "runtime rule"
#: an issue the seam refit is there to resolve (``reseam.refit``); anything else is a seam not made
REFITTABLE = "intersect"


def without_helpers(mesh, skeleton, plain) -> None:
    """Rebind *mesh* (on *skeleton*) to *plain*, the same skeleton without the kept helper joints.

    Each kept joint's weight goes to the bone it hangs under and a vertex's
    weights on one bone are summed — the skin as it was before
    ``body.keep_helpers``.
    """
    to_plain = np.zeros(len(skeleton.bones), dtype=np.int64)
    for index, bone in enumerate(skeleton.bones):
        at = index
        while skeleton.bones[at].name.startswith(GRAFT):
            at = skeleton.bones[at].parent
        to_plain[index] = plain.index[skeleton.bones[at].name]
    count = mesh.vertex_count
    dense = np.zeros((count, len(plain.bones)))
    np.add.at(dense, (np.arange(count)[:, None], to_plain[mesh.bone_idx.astype(np.int64)]),
              mesh.bone_wt.astype(np.float64))
    width = max(int((dense > 0).sum(axis=1).max()), 1)
    order = np.argsort(-dense, axis=1, kind="stable")[:, :width]
    # summing a vertex's shares of one bone can land a float32 step over one
    weight = np.minimum(np.take_along_axis(dense, order, axis=1), 1.0)
    mesh.bone_idx = np.where(weight > 0, order, 0).astype(np.int32)
    mesh.bone_wt = weight.astype(np.float32)


def recompose(character: str, *, write: bool = False, force: bool = False) -> dict:
    """Compose ``body/<character>`` again under the current classification of its native skin.

    Returns what changed.  Nothing is written when classification and composition
    revision are current — unless *force*, for a body whose stored fit cannot be refitted
    any more (its source was encoded again since, and the composer orders its
    triangles otherwise).  It is refused when the composer leaves a seam unmade; an
    accepted first fit whose bridges still cross something is written with
    that recorded, for the seam refit to resolve.
    """
    from .samples import _body_piece, lent_softness, seam_inputs
    extra = store.read_extra("chara", character) or {}
    body_extra = store.read_extra("body", character) or {}
    manifest = body_extra.get("composite")
    item = store.read_item("chara", character)
    body_item = store.read_item("body", character)
    if not manifest or item is None or body_item is None or len(body_item.pieces) < 2:
        raise ActorError(f"body/{character}: no composed body with its fitted source in the store")
    skeleton = build_skeleton(item.nodes, character)
    append_body_nodes(skeleton, body_item.nodes)
    nodes = {name: node for name, node in body_item.nodes.items() if not name.startswith(GRAFT)}
    plain = build_skeleton(item.nodes, character)
    append_body_nodes(plain, nodes)
    rebound = _rest_mesh(body_item.pieces[1], skeleton)
    without_helpers(rebound, skeleton, plain)
    # composed from the source as the store will hand it back: the piece's own encoding moves a
    # position by a float32 step, and the composer's triangle order is not indifferent to that
    kept_source = _body_piece(rebound, plain)
    source = _rest_mesh(kept_source, plain)
    inputs = seam_inputs(item, plain, extra, lent_softness(character, source), stored_given_skin(character))
    stored = {name: {role: sorted(faces) for role, faces in roles.items()}
              for name, roles in manifest["parts"].items()}
    now = {name: {role: sorted(faces) for role, faces in roles.items()} for name, roles in inputs["groups"].items()}
    changed = sorted(name for name in now if now[name] != stored.get(name))
    report = {"character": character, "written": False, "reclassified": {
        name: {role: len(faces) for role, faces in now[name].items()} for name in changed}}
    if not changed and not force and manifest["seams"].get("revision") == S.COMPOSITION_REVISION:
        if write and (manifest.get("coverage_verified") is False or manifest["seams"].get("accepted") is not True):
            # composed again by a run that stopped before its mask was measured
            report.update(finish(character), written=True, resumed=True)
            return report
        report["reason"] = "its native skin is classified as it was when it was composed"
        return report
    return _compose(character, source, plain, nodes, kept_source, body_item, body_extra, inputs,
                    report, write=write)


def anatomy_held(character: str, vertex_count: int) -> np.ndarray:
    """Per vertex of ``body/<character>``'s lent body, how much of it is the lent anatomy (``compose_skin``'s *held*).

    The body piece's share of weight on the anatomy's joints (``body.ANATOMY``),
    from the stored source body; a separate anatomy piece (the man's shaft)
    is held whole.  Read off the source because the fit merges those joints
    into the hips.
    """
    from ..mesh.skin import split_archive_skin
    from .body import ANATOMY
    extra = store.read_extra("body", character) or {}
    source = store.read_item("body_source", character)
    held = np.zeros(vertex_count)
    if source is None:
        return held
    pieces_ = {piece.mesh.name: piece for piece in source.pieces}
    for row in extra.get("source_piece_ranges") or []:
        span = slice(row["vertex_start"], row["vertex_start"] + row["vertices"])
        piece = pieces_.get(row["source_mesh"])
        if row["source_mesh"] != "o_body_a":
            held[span] = 1.0
        elif piece is not None:
            weights, indices = split_archive_skin(np.asarray(piece.skin, dtype=np.float64),
                                                  bone_count=len(piece.bones))
            on = np.asarray([any(token in name for token in ANATOMY) for name in piece.bones], dtype=bool)
            held[span] = np.sum(weights * on[indices], axis=1)[: row["vertices"]]
    return held


def stored_given_skin(character: str) -> dict:
    """The native skin a rebuild gave away (``samples.unjoinable_skin``), as the body's table keeps it."""
    raw = (store.read_extra("body", character) or {}).get("given_skin_faces") or {}
    return {role: {int(piece): set(faces) for piece, faces in pieces_.items()} for role, pieces_ in raw.items()}


def collar_offset(source, inputs: dict, seams: dict, skeleton):
    """How far the lent collar stands from the head opening it is fitted to, horizontally; ``None`` when it need not.

    Only when the composer had to fit the lent body's own collar to the head
    and could not (the character's own neck skin did not take its place).
    The offset is between the two rings' centres; ``body.fit_body`` moves the
    neck by it.
    """
    # an attempt the composer certified is no trouble: moving a collar that is already joined to
    # the head took Avywen's shoulders off her own skin (§7.1064)
    trouble = any(str(issue.get("reason", "")).startswith("added Head collar")
                  for issue in seams.get("unresolved") or [])
    if not trouble:
        return None
    head = inputs["head_bones"]
    collars = S._head_collar_loops(source, source, head)
    candidates = S._head_outline_candidates(inputs["head_surfaces"], head)
    if not collars or not candidates:
        return None
    collar = collars[0][1]
    try:
        opening = S._choose_head_outline(source, collar, candidates, skeleton)
    except S._SeamIssue:
        return None
    shift = (opening["mesh"].positions[opening["vertices"]].astype(np.float64).mean(0)
             - source.positions[collar].astype(np.float64).mean(0))
    shift[1] = 0.0
    return shift


def settled_inputs(character: str, source, plain, given: dict | None = None) -> tuple:
    """Use the build's exact classification/composition loop, without publishing.

    The last report records each round so authoring diagnostics can explain
    a refused build using the same inputs as the production composer.
    """
    from .samples import lent_softness, seam_inputs, unjoinable_skin

    given = {} if given is None else given
    item = store.read_item("chara", character)
    extra = store.read_extra("chara", character) or {}
    held = anatomy_held(character, source.vertex_count)
    lent = lent_softness(character, source)
    inputs = seam_inputs(item, plain, extra, lent, given)
    history = []
    for _ in range(4):
        _render, report = S.compose_skin(
            source, inputs["native"], plain, face_mesh=inputs["face"],
            head_bones=inputs["head_bones"], head_surfaces=inputs["head_surfaces"], held=held,
        )
        more = unjoinable_skin(inputs, report, item)
        history.append({"accepted": report.get("accepted"),
                        "unresolved": report.get("unresolved", []),
                        "given": {role: {piece: len(faces) for piece, faces in pieces_.items()}
                                  for role, pieces_ in more.items()}})
        fresh = False
        for role, pieces_ in more.items():
            for piece, faces in pieces_.items():
                known = given.setdefault(role, {}).setdefault(piece, set())
                fresh |= bool(set(faces) - known)
                known.update(faces)
        if not fresh:
            break
        inputs = seam_inputs(item, plain, extra, lent, given)
    report["classification_settling"] = history
    return inputs, report, given


def rebuild(character: str, card, cross_section: str | None = None, thumb_carry: str | None = None,
            *, arm_root: str | None = None, unmeasured_limbs=None, arm_residual: str | None = None) -> dict:
    """Build ``body/<character>`` from *card* again: a fresh fit, then composed and settled as :func:`recompose` does.

    For a body whose stored fit is not to be trusted, or that has none.
    ``body.bake_body`` fits the card's body to the character (its scope
    check and its skin, without the card's marks); the composer then cuts it
    to the character's own skin from that fit as the store hands it back.
    Before the cut is kept: native skin the composer refuses to join is given
    to the head or the outfit (``samples.unjoinable_skin``, stored as
    ``given_skin_faces``), and when the lent collar has to be fitted to the
    head it is first stood under that opening (:func:`collar_offset`, a
    second fit).  Always writes: the fit is stored before it can be composed.
    *cross_section*: the fit's rule (``body.fit_rule``; default the body's own); *thumb_carry*:
    ``body.thumb_rule``'s, the same way. *arm_root*: ``body.arm_root_rule``'s;
    the same choice is used in both the initial fit and the collar refit, as is
    *unmeasured_limbs* (``body.unmeasured_rule``), *arm_residual* (``body.arm_residual_rule``).
    """
    from .body import bake_body

    def fitted(shift=None):
        bake_body(character, card, collar_shift=shift, cross_section=cross_section, thumb_carry=thumb_carry,
                  arm_root=arm_root, unmeasured_limbs=unmeasured_limbs, arm_residual=arm_residual)
        body_item = store.read_item("body", character)
        plain = build_skeleton(body_item.nodes, character)
        return body_item, plain, _rest_mesh(body_item.pieces[0], plain)

    body_item, plain, source = fitted()
    inputs, seams, given = settled_inputs(character, source, plain)
    shift = collar_offset(source, inputs, seams, plain)
    if shift is not None:
        body_item, plain, source = fitted(shift)
        inputs, seams, given = settled_inputs(character, source, plain, given)
    body_extra = store.read_extra("body", character) or {}
    body_extra["given_skin_faces"] = {
        role: {str(piece): sorted(faces) for piece, faces in pieces_.items() if faces}
        for role, pieces_ in given.items() if any(pieces_.values())} or None
    report = {"character": character, "written": False, "card": str(card),
              "given_skin_faces": {role: {piece: len(faces) for piece, faces in pieces_.items()}
                                   for role, pieces_ in given.items()},
              "collar_shift": None if shift is None else [round(float(v), 4) for v in shift],
              "reclassified": {name: {role: len(faces) for role, faces in roles.items()}
                               for name, roles in inputs["groups"].items()}}
    return _compose(character, source, plain, body_item.nodes, body_item.pieces[0], body_item, body_extra,
                    inputs, report, write=True, was_seams=0)


def _compose(character: str, source, plain, nodes: dict, kept_source, body_item, body_extra: dict,
             inputs: dict, report: dict, *, write: bool, was_seams: int | None = None) -> dict:
    """Cut the lent *source* to the native skin in *inputs*, and with *write* store it and :func:`settle` it.

    The step :func:`recompose` and :func:`rebuild` share.  Refused when the
    composer leaves a seam unmade; a first fit whose bridges still cross
    something is written with that on record, for the seam refit.
    """
    from .samples import _body_piece
    from .skin_color import calibrate_skin
    manifest = body_extra.get("composite") or {}
    render, seams = S.compose_skin(source, inputs["native"], plain, face_mesh=inputs["face"],
                                   head_bones=inputs["head_bones"], head_surfaces=inputs["head_surfaces"],
                                   held=anatomy_held(character, source.vertex_count))
    left = seams.get("unresolved") or []
    report.update(seams=len(seams.get("seams", [])),
                  was_seams=len(manifest["seams"].get("seams", [])) if was_seams is None else was_seams,
                  source_faces_kept=int(np.sum(np.asarray(seams["render_face_source"]) >= 0)),
                  source_faces=source.tri_count,
                  first_fit_issues=[(issue["reason"], issue.get("intersection_count", 1)) for issue in left])
    # what made the attachment fail rides along: a bare "lacks certified attachment" says nothing
    measured = ("attachment_max_gap_m", "attachment_unmatched_arc_max_m", "attachment_matched_arc_gap_max_m",
                "attachment_skin_weight_l1_max")
    unmade = [issue["reason"] + (f" ({issue['attachment_reason']}; "
                                 + ", ".join(f"{k} {issue[k]:.3g}" for k in measured if k in issue) + ")"
                                 if issue.get("attachment_reason") else "")
              for issue in left if REFITTABLE not in issue["reason"]]
    if unmade:
        tried = (seams.get("head_attachment_attempt") or {}).get("reason")
        raise ActorError(f"body/{character}: the composer leaves a seam unmade: {'; '.join(sorted(set(unmade)))}"
                         + (f"; fitting the collar to the head: {tried}" if tried else ""))
    report["skin_color"] = calibrate_skin(render, inputs["native"], seams, body_item.maps["albedo"],
                                          head_surfaces=inputs["head_surfaces"])
    if not write:
        return report
    made = {"version": 1, "body_triangles": render.tri_count, "native_excluded_faces": [], "covered_faces": [],
            "parts": inputs["groups"], "classification_evidence": inputs["evidence"],
            "facial_context": inputs["facial_context"], "facial_evidence_files": inputs["facial_evidence_files"],
            "native_head_neck_evidence": inputs["native_head_neck_evidence"],
            "render_tints": render.colors.tolist(),
            "source_native_excluded_faces": seams["source_excluded_faces"],
            "seams": seams, "coverage_status": pieces.RULE_HALF, "coverage_verified": False,
            # no audit named faces beyond the rays: the list is the rule's alone and is measured again
            # whenever the body's surface moves (:func:`first_mask`)
            "mask_owner": MASK_OWNER}
    body_extra["composite"] = made
    body_extra["composite_status"] = "seams composed again; mask measured by the runtime rule"
    body_extra.pop("helper_joints", None)
    with store.opened(character, write=True):
        bake.write_item("body", character, [_body_piece(render, plain), kept_source],
                        source=store.SOURCE, nodes=nodes, label=character,
                        extra_maps=body_item.maps)
        bake.write_named_table(store.SOURCE, "body", character, body_extra)
    report["written"] = True
    if left:
        from .reseam import refit
        report["refit"] = refit(character, write=True)
    report.update(finish(character))
    report["settled"] = settle(character)
    return report


def settle(character: str) -> dict:
    """Take a composed body the rest of the way: seams fitted again, source joints kept, mask by the current rule.

    The three steps every body needs after it is composed, in the one order
    that works (the refit moves the surface the mask is measured on; the
    kept joints are bound to the refitted source), and each is a no-op on a
    body that already had it.  One owner: the build (``samples.apply_coverage``)
    and :func:`recompose` both end here.  A refit the body refuses is put on
    record under ``refit`` and the other two steps are still made — the body
    loads as the composer left it.
    """
    from .actor import ActorError
    from .body import keep_helpers
    from .remask import remask
    from .reseam import refit
    out = {}
    try:
        out["refit"] = refit(character, write=True)
    except ActorError as refusal:
        out["refit"] = {"refused": str(refusal)}
    out["joints"] = keep_helpers(character, write=True)
    out["remask"] = remask(character, write=True)
    return out


def _refresh_refit_acceptance(seams: dict) -> None:
    """Replace inherited acceptance flags using this refit's per-kind improvement policy."""
    first = {reason: count for reason, count in seams.get("first_fit_issues", [])}
    left = {reason: count for reason, count in seams.get("refit_issues", [])}
    seams.pop("accepted_with_issues", None)
    seams["accepted"] = not seams.get("unresolved") and all(
        count < first.get(reason, 0) for reason, count in left.items())
    if seams["accepted"] and left:
        seams["accepted_with_issues"] = sorted(left)


def finish(character: str) -> dict:
    """Accept a body composed again and measure its mask; what :func:`recompose` ends with.

    The first fit of such a body may have been written with issues open for
    the seam refit.  It is accepted when the refit answered them: nothing
    left, or — as for a body the build accepted — only fewer of the same
    kinds than the first fit had, which stay on record as ``refit_issues``.
    """
    from .certification import capture
    body_extra = store.read_extra("body", character)
    made = body_extra["composite"]
    seams = made["seams"]
    if seams.get("unresolved"):
        raise ActorError(f"body/{character}: composed again, but the seam refit did not answer its first fit's "
                         f"issues; the stored body does not load until they are resolved")
    if seams.get("accepted") is not True:
        _refresh_refit_acceptance(seams)
        if not seams["accepted"]:
            first = dict(seams.get("first_fit_issues", []))
            left = dict(seams.get("refit_issues", []))
            raise ActorError(f"body/{character}: its refit is no better than its first fit ({left} against {first})")
    # the mask is measured on the loaded body, and a body loads only with a verified mask: an empty one first
    made["coverage_verified"] = True
    made["source_certification"] = capture(character, made)
    with store.opened(character, write=True):
        bake.write_named_table(store.SOURCE, "body", character, body_extra)
    with store.opened(character, write=True):
        bake.write_named_table(store.SOURCE, "body", character, body_extra)
    return {"covered": first_mask(character), "issues_left": seams.get("refit_issues", [])}


def first_mask(character: str) -> int:
    """Measure the mask of a body this module composed by the first rule again; how many faces it names.

    For after its surface moved (the seam refit): ``remask`` takes a list to
    the current rule only from the first rule's own list for the surface as
    it is.  A body whose mask came from elsewhere is left alone (-1).
    """
    from .certification import capture
    body_extra = store.read_extra("body", character)
    made = body_extra["composite"]
    if made.get("mask_owner") != MASK_OWNER:
        return -1
    made["covered_faces"] = []
    made["coverage_status"] = pieces.RULE_HALF
    made["source_certification"] = capture(character, made)
    with store.opened(character, write=True):
        bake.write_named_table(store.SOURCE, "body", character, body_extra)
    made["covered_faces"] = covered_by_outfit(character)
    made["source_certification"] = capture(character, made)
    with store.opened(character, write=True):
        bake.write_named_table(store.SOURCE, "body", character, body_extra)
    return len(made["covered_faces"])


def covered_by_outfit(character: str) -> list:
    """The stored body's faces its whole outfit covers, by the first coverage rule (``pieces.RULE_HALF``).

    The list a freshly composed body starts from; ``remask`` takes it to the
    current rule.
    """
    from .body import load_complete_actor
    actor = load_complete_actor(character)
    visibility = actor.composite_visibility
    drawn = visibility.drawable()
    worn = {index: part for index, part in enumerate(actor.model.parts)
            if drawn.get(id(part)) and part is not visibility.body}
    corners, owners = pieces.Outfit.of(worn).corners()
    source = visibility.source
    half, _whole = pieces.coverage_by_rule(source.positions, source.indices, corners, owners[:, 0])
    return [face for face, owned in enumerate(half) if owned]
