"""Measure a completed body's whole-outfit mask again, under the current coverage rule.

``body/<character>`` keeps one list of the body faces its outfit covers
(``composite``'s ``covered_faces``), made when the body was built.  The rule
it was made by (``pieces.RULE_HALF``) leaves a thin limb's faces drawn where
the body is fuller than what is worn: at an ankle the hose sits deeper inside
the body than half the ankle's thickness, and the body stands out through it.
``pieces.RULE_NEAR_WALL`` adds exactly that case, ``pieces.RULE_CLOSED_IN``
then leaves out what is left in patches the outfit closes in: skin inside a
boot, a sliver seen through a slit in a stocking; and ``pieces.RULE_EXTREMITIES``
the rest of a hand or foot mostly left out or closed in: toes through a sole.

:func:`remask` measures both rules from the stored item.  That the old rule
gives the stored list back is what shows the measurement here is the one the
list was made with (to within :data:`STRAY` — a ray grazing an edge falls on
one side here and the other there).  Only then is anything written, and what
is written is the stored list plus the faces the new term alone adds: nothing
the list names is taken out, whether a ray found it or the body build hid it
by name.
"""
from __future__ import annotations

import numpy as np

from ..origin import bake
from . import pieces, store
from .actor import ActorError

#: the share of a body's faces the old rule may find here that the stored list
#: lacks, and still be the same measurement
STRAY = 1e-3


def whole_outfit(positions, indices, corners, owners, never=None, regions=None, lent=()) -> list:
    """The body faces a whole outfit covers under the current rule, measured afresh.

    ``pieces.RULE_NEAR_WALL``'s faces and what is left in patches the outfit
    closes in (``pieces.RULE_CLOSED_IN``) less the faces *never* drawn anyway,
    with each hand and foot mostly left out or closed in left out whole
    (``pieces.extremities``, given the faces' *regions*; not those the item *lent*).  *corners* ``(M, 3, 3)`` and
    *owners* ``(M,)`` are the outfit's triangles (``pieces.Outfit.corners``).
    This is what the build hands the Blender worker, so the scene's dressed
    body is cut by the rule the run time masks it by (:func:`outfit_cover`).
    """
    _half, whole = pieces.coverage_by_rule(positions, indices, corners, owners)
    covered = np.array([bool(owned) for owned in whole], dtype=bool)
    shut = pieces.closed_patches(positions, indices, corners, covered, never)
    left_out = covered | shut
    if regions is not None:
        left_out = pieces.extremities(left_out, regions, positions, indices, corners, lent=lent)
    return np.flatnonzero(left_out).tolist()


def outfit_cover(character: str, body_mesh, skeleton) -> list:
    """:func:`whole_outfit` for *body_mesh* (skinned on *skeleton*) under ``chara/<character>`` as the item draws it.

    The outfit is what the run time measures against (:func:`remask`): every
    part the item draws.  The body is not stored as a completed one yet when
    the build asks, so the dressed item is loaded on its own.
    """
    from .actor import load_actor
    from .part_roles import face_regions
    actor = load_actor("chara", character)
    worn = {index: part for index, part in enumerate(actor.model.parts) if part.visible}
    corners, owners = pieces.Outfit.of(worn).corners()
    regions = face_regions(body_mesh, skeleton, actor.extra.get("humanoid") or {})
    return whole_outfit(body_mesh.positions, body_mesh.indices, corners, owners[:, 0], regions=regions,
                        lent=frozenset(actor.extra.get("lent_regions") or ()))


def remask(character: str, *, write: bool = False) -> dict:
    """Measure ``body/<character>``'s mask under both rules; with *write*, store the new one.

    Returns the counts.  ``reproduced`` is whether the old rule finds the
    stored list's faces and at most :data:`STRAY` of the body's besides;
    nothing is written unless it does.  Writing renews the item's source
    certification, which binds the manifest.
    """
    from .body import load_complete_actor
    from .certification import capture
    from .composite import validate_manifest
    actor = load_complete_actor(character)
    visibility = actor.composite_visibility
    drawn = visibility.drawable()
    worn = {index: part for index, part in enumerate(actor.model.parts)
            if drawn.get(id(part)) and part is not visibility.body}
    corners, owners = pieces.Outfit.of(worn).corners()
    source = visibility.source
    from .part_roles import face_regions
    half, whole = pieces.coverage_by_rule(source.positions, source.indices, corners, owners[:, 0])
    old = {face for face, owned in enumerate(half) if owned}
    new = {face for face, owned in enumerate(whole) if owned}
    extra = store.read_extra("body", character)
    manifest = extra["composite"]
    stored = set(manifest["covered_faces"])
    beyond = stored - old
    report = {"character": character, "faces": source.tri_count, "stored": len(stored),
              "old_rule": len(old), "new_rule": len(new), "added": len(new - old - stored),
              "kept_beyond_rays": len(beyond), "missing_from_stored": len(old - stored),
              "stored_rule": manifest.get("coverage_status"),
              "reproduced": len(old - stored) <= STRAY * source.tri_count, "written": False}
    if not write or not report["reproduced"]:
        return report
    known = (pieces.RULE_HALF, pieces.RULE_FAR_WALL, pieces.RULE_NEAR_WALL, pieces.RULE_CLOSED_IN,
             pieces.RULE_EXTREMITIES)
    if manifest.get("coverage_status") not in known:
        raise ActorError(f"body/{character}: its mask was made by a rule this does not know")
    covered = np.zeros(source.tri_count, dtype=bool)
    covered[sorted(stored | (new - old))] = True
    shut = pieces.closed_patches(source.positions, source.indices, corners, covered,
                                 ~visibility._permanent)
    left_out = pieces.extremities(covered | shut,
                                  face_regions(source, actor.skeleton, actor.extra.get("humanoid") or {}),
                                  source.positions, source.indices, corners,
                                  lent=frozenset(actor.extra.get("lent_regions") or ()))
    manifest["covered_faces"] = np.flatnonzero(left_out).tolist()
    manifest["coverage_status"] = pieces.RULE_EXTREMITIES
    report["closed_in"] = int((shut & ~covered).sum())
    report["extremities"] = int((left_out & ~(covered | shut)).sum())
    validate_manifest(manifest, manifest["body_triangles"])
    manifest["source_certification"] = capture(character, manifest)
    with store.opened(character, write=True):
        bake.write_named_table(store.SOURCE, "body", character, extra)
    report.update(written=True, covered=len(manifest["covered_faces"]))
    return report
