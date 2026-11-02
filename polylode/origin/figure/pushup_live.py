"""Pushup while the garments come off: the chest the ones still on hold.

:mod:`pushup` computes the push once, from the card's stored states.  The
plugin keeps doing it: ``ClothesStateChangeEvent`` / ``SetClothesStatePostfix``
call ``RecalculateBody`` whenever a garment's state changes, so taking off the
bra puts the base chest back (or the top's push, if the top is still on).

How that lands on our rig without re-dressing: the meshes were retargeted onto
the rig *after* the shape pass, and ``Skeleton.finalize`` wrote their bind
(``inv_bind``) from that shaped rest.  Changing a bone's **local rest** while
keeping that bind is exactly what moving a bone does to a skinned mesh in the
game — the body, and whatever garments are still on, follow.  So each
combination of "which of bra / top still counts" is shaped once on a copy of
the bare graft (the same :func:`dress.apply_card_shape` the dress ran), and the
bones whose local rest differs from the dress-time shape are swapped into the
live rig.  Nothing may call ``finalize`` after that, or the swap is baked into
the bind and the mesh snaps back; nothing on the run-time path does.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from . import pushup as PU


class BodyOverride:
    """The card, with ``body()`` answering a rewritten body block.

    Everything that shapes the character reads the body through
    ``card.body()`` (``shape_apply``, ``bust``, ``body_head``), which is where
    the game's own ``fileBody`` would be read after a plugin wrote into it.
    """

    def __init__(self, card, body: dict):
        self._card = card
        self._body = body

    def body(self) -> dict:
        return self._body

    def __getattr__(self, name):
        return getattr(self._card, name)


@dataclass
class Inputs:
    """What ``CalculatePush`` reads, resolved once per character."""
    card: object
    outfit: int
    body: dict
    base: dict
    bra: dict
    top: dict
    bra_worn: bool
    top_worn: bool


def inputs(card, outfit: int, report=None, worn=None) -> Inputs | None:
    """The card's Pushup data for *outfit*, or ``None`` when there is nothing to push.

    *worn(slot_name)* says whether the slot holds a garment
    (``IsClothesStateKind``); the dress passes its resolver.
    """
    bra, top, saved, reason = PU.read(card)
    if reason and report is not None:
        report.missing.append(f"Pushup：{reason}")
    if not bra and not top:
        return None
    try:
        body = card.body() or {}
    except Exception as exc:                         # noqa: BLE001
        if report is not None:
            report.missing.append(f"Pushup：讀不到卡片（{exc}）")
        return None
    base = PU.base_from_body(body)
    if base is None:
        if report is not None:
            report.missing.append("Pushup：卡片沒有胸部滑桿")
        return None
    if saved:
        # the main game's ``BaseData`` is ``Pushup_BodyData``
        base = {k: float(saved.get(k, v)) for k, v in base.items()}
    try:
        bra_worn, top_worn = bool(worn("bra")), bool(worn("top"))
    except Exception as exc:                         # noqa: BLE001
        if report is not None:
            report.missing.append(f"Pushup：讀不到卡片（{exc}）")
        return None
    return Inputs(card, int(outfit), body, base, bra, top, bra_worn, top_worn)


def counting(inp: Inputs, states) -> tuple[bool, bool]:
    """``(bra counts, top counts)`` for garment *states* (``CurrentlyWearing``)."""
    st = list(states or [])

    def state(i: int) -> int:
        return int(st[i]) if i < len(st) else 0
    return (PU.worn_data(inp.bra, inp.outfit, inp.bra_worn, state(PU.BRA_KIND)) is not None,
            PU.worn_data(inp.top, inp.outfit, inp.top_worn, state(PU.TOP_KIND)) is not None)


def card_view(inp: Inputs, pair: tuple[bool, bool]):
    """The card as the shape pass sees it with *pair* counting."""
    values = PU.pushed(
        inp.base,
        inp.bra.get(inp.outfit) if pair[0] else None,
        inp.top.get(inp.outfit) if pair[1] else None)
    if values is None:
        return inp.card
    return BodyOverride(inp.card, PU.body_with(inp.body, values))


def _snapshot(skel) -> list:
    return [(np.array(b.local_pos, dtype=np.float64),
             np.array(b.local_rot, dtype=np.float64),
             np.array(getattr(b, "local_scale", np.ones(3)), dtype=np.float64))
            for b in skel.bones]


@dataclass
class Live:
    """A character's Pushup state on its live rig."""
    inp: Inputs
    rig_scale: float
    bare: object                  # the graft before shaping (a deep copy)
    shaped: list                  # dress-time local (pos, rot, scale) per bone
    pair: tuple                   # the combination the rig is shaped for now
    dress_pair: tuple
    diffs: dict = field(default_factory=dict)   # pair -> {bone: (pos, rot, scale)}
    touched: set = field(default_factory=set)


def begin(skel, inp: Inputs, pair: tuple, rig_scale: float) -> Live:
    """Before the dress's shape pass: keep the bare graft."""
    return Live(inp, float(rig_scale), copy.deepcopy(skel), [], pair, pair)


def settle(live: Live, skel) -> None:
    """After the dress's shape pass: remember what it wrote."""
    live.shaped = _snapshot(skel)


def _diff(live: Live, pair: tuple, shape) -> dict:
    hit = live.diffs.get(pair)
    if hit is not None:
        return hit
    from types import SimpleNamespace
    sk = copy.deepcopy(live.bare)
    shape(SimpleNamespace(skeleton=sk), card_view(live.inp, pair),
          live.rig_scale, coordinate=live.inp.outfit)
    out = {}
    for i, (now, was) in enumerate(zip(_snapshot(sk), live.shaped)):
        if any(not np.allclose(a, b, atol=1e-7) for a, b in zip(now, was)):
            out[i] = now
    live.diffs[pair] = out
    return out


def follow(built, states) -> bool:
    """Reshape *built*'s chest for garment *states*; True when the rig changed."""
    live = getattr(built, "pushup", None)
    if live is None or not live.shaped:
        return False
    pair = counting(live.inp, states)
    if pair == live.pair:
        return False
    from . import dress
    diff = {} if pair == live.dress_pair else _diff(live, pair, dress.apply_card_shape)
    skel = built.skeleton
    for i in live.touched | set(diff):
        skel.set_rest_local(i, *diff.get(i, live.shaped[i]))
    live.touched |= set(diff)
    live.pair = pair
    return True
