"""The card's shape sliders edited on a dressed character, without dressing it again (#12).

``ChaControl.SetShapeBodyValue`` / ``SetShapeFaceValue`` write one slot of
``shapeValueBody`` / ``shapeValueFace`` and run the shape pass again; the meshes
are skinned to the bones, so they follow.  Ours the same way :mod:`pushup_live`
reshapes the chest: the pass runs on a copy of the bare graft (the skeleton
before any shaping, kept by the dress), and each bone whose shaped rest moved
gets **the change** composed onto the live rig — not the new rest outright,
because the live rig carries more than the shape (the garments' settled
chains, ``_apply_sway``; the Pushup swap).  The meshes keep the bind the dress
gave them (``Skeleton.finalize`` is not called), which is exactly what moving a
bone does to a skinned mesh in the game.

The height slot is the one this cannot do: the graft writes the height into
every bone position (``rig_scale``), so a new height is a new dress.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np

from ...core import math3d as m3

#: ``ChaFileDefine.BodyShapeIdx.Height`` — the slot the graft carries.
HEIGHT = 0


def snapshot(skel) -> list:
    """Every bone's local ``(position, rotation, scale)``."""
    return [(np.array(b.local_pos, dtype=np.float64),
             np.array(b.local_rot, dtype=np.float64),
             np.array(getattr(b, "local_scale", np.ones(3)), dtype=np.float64))
            for b in skel.bones]


def begin(built) -> None:
    """Before the dress's shape pass: keep the bare graft."""
    built.bare_graft = copy.deepcopy(built.skeleton)


def settle(built) -> None:
    """After the dress's shape pass: remember the rest it wrote."""
    built.shaped_rest = snapshot(built.skeleton)


def needs_dress(face: bool, index: int) -> bool:
    """Whether a slot's edit takes a new dress rather than a reshape."""
    return not face and int(index) == HEIGHT


def reshape(built, card, rig_scale: float, *, outfit: int = 0) -> int:
    """Shape the bare graft with *card* as it is now; move the live rig by the change.

    Returns how many bones moved.  A character dressed before the bare graft
    was kept (or without a card) has nothing to reshape from and returns 0.
    """
    bare = getattr(built, "bare_graft", None)
    old = getattr(built, "shaped_rest", None)
    if bare is None or not old:
        return 0
    from . import dress
    from . import pushup_live as PL
    live = getattr(built, "pushup", None)
    view = PL.card_view(live.inp, live.pair) if live is not None else card
    sk = copy.deepcopy(bare)
    shaped = SimpleNamespace(skeleton=sk)
    dress.apply_card_shape(shaped, view, rig_scale, coordinate=int(outfit))
    # the breast and buttock chains hang from the new shape (the per-frame
    # solve starts from them, ``bust_live``)
    built.bust_chains = list(getattr(shaped, "bust_chains", []) or [])
    built.__dict__.pop("bust_mask_shapes", None)      # shaped for the card before this edit
    moved = _move_to(built, snapshot(sk), old)
    if live is not None:
        # its cached shapes were made against the card before this edit
        live.diffs.clear()
        live.shaped = snapshot(built.skeleton)
        live.dress_pair = live.pair
        live.touched.clear()
    return moved


def _move_to(built, new: list, old: list) -> int:
    """Move each bone of the live rig by how far its rest moved from *old* to *new*; returns bones moved."""
    skel = built.skeleton
    moved = 0
    for i, ((np_, nr, ns), (op, orot, os_)) in enumerate(zip(new, old)):
        if np.allclose(np_, op, atol=1e-9) and np.allclose(nr, orot, atol=1e-9) \
                and np.allclose(ns, os_, atol=1e-9):
            continue
        b = skel.bones[i]
        pos = np.asarray(b.local_pos, dtype=np.float64) + (np_ - op)
        rot = m3.quat_normalize(m3.quat_mul(nr, m3.quat_mul(m3.quat_conj(orot), b.local_rot)))
        scl = np.asarray(getattr(b, "local_scale", np.ones(3)), dtype=np.float64) \
            * np.where(np.abs(os_) > 1e-12, ns / np.where(np.abs(os_) > 1e-12, os_, 1.0), 1.0)
        skel.set_rest_local(i, pos, rot, scl)
        moved += 1
    built.shaped_rest = new
    return moved


#: ``ChaFileDefine.cf_BustShapeMaskID``: the body sliders the nine breast shapes are
BUST_MASK_SLIDERS = (5, 6, 7, 8, 9, 10, 11, 12, 13)
#: what ``UpdateShapeBody`` drives a disabled one with (the array literal it builds; the nipple's
#: puff is the 0); the last, ``cf_ShapeMaskNipStand``, takes 0.5 and skips the stand rate
BUST_MASK_DEFAULTS = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5)
NIP_STAND = 8


class _Masked:
    """A card view with some breast sliders driven as ``UpdateShapeBody`` drives the disabled ones."""

    def __init__(self, card, on: tuple):
        self._card = card
        body = dict(card.body() or {})
        values = list(body.get("shapeValueBody") or [])
        for i, (slider, keep) in enumerate(zip(BUST_MASK_SLIDERS, on)):
            if not keep and slider < len(values):
                values[slider] = BUST_MASK_DEFAULTS[i]
        body["shapeValueBody"] = values
        self._body = body
        sections = dict(getattr(card, "sections", {}) or {})
        if not on[NIP_STAND]:
            sections["Status"] = {**(sections.get("Status") or {}), "nipStandRate": 0.0}
        self.sections = sections

    def body(self) -> dict:
        return self._body

    def __getattr__(self, name):
        return getattr(self._card, name)


def mask_bust(built, card, rig_scale: float, left: tuple, right: tuple, *, outfit: int = 0) -> int:
    """``DisableShapeBodyID`` for both breasts: shape each side with its disabled sliders, move the rig.

    *left* / *right* are the nine breast shapes a side, true for on
    (``YureCtrl``'s ``BreastShapeInfo``).  ``UpdateShapeBody`` drives a side's
    disabled sliders with fixed values (``BUST_MASK_DEFAULTS``) and updates
    that side's items alone (``updateMask``), so each side is shaped on its
    own and takes the bones on its side — a centre bone the last pass's,
    the right's, as the game runs right after left.  The breast chains are
    set up again from the new shape (``reSetupDynamicBoneBust``).  Returns
    how many bones moved; 0 with nothing to reshape from.
    """
    bare = getattr(built, "bare_graft", None)
    old = getattr(built, "shaped_rest", None)
    if bare is None or not old:
        return 0
    from . import dress
    from . import pushup_live as PL
    live = getattr(built, "pushup", None)
    view = PL.card_view(live.inp, live.pair) if live is not None else card

    # a row goes back and forth between a few states, so each combination is shaped once
    cache = built.__dict__.setdefault("bust_mask_shapes", {})
    pair = getattr(live, "pair", None)

    def shaped(on: tuple):
        key = (tuple(on), int(outfit), pair)
        if key not in cache:
            sk = copy.deepcopy(bare)
            ns = SimpleNamespace(skeleton=sk)
            dress.apply_card_shape(ns, _Masked(view, tuple(on)), rig_scale, coordinate=int(outfit))
            cache[key] = (snapshot(sk), list(getattr(ns, "bust_chains", []) or []))
        return cache[key]

    by_left, chains_left = shaped(left)
    by_right, chains_right = shaped(right)
    new = [by_left[i] if bone.name.endswith("_L") else by_right[i] for i, bone in enumerate(bare.bones)]
    built.bust_chains = ([c for c in chains_left if c.index == 0]
                         + [c for c in chains_right if c.index != 0])
    return _move_to(built, new, old)
