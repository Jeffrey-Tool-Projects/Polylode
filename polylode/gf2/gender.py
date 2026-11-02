"""Female-only export policy backed by a frozen, reviewed playable roster.

IOP Wiki's Doll revision176479 identifies GF2 Dolls as female. The reviewed
local GunCharacterData/GunData joins bind that statement to these identities;
neither a human skeleton nor a character-name prefix establishes gender.
New or changed table identities require a new evidence review.

The model extractor exports the game's own model of every playable
character, women and men, with no review (owner 2026-11-02, §7.1455): it
runs its scans and exports inside :func:`all_playable`, in a GF2 store of
its own, and a selection made there carries :data:`POLICY_ALL`, which only
that mode accepts.  The game's own processes never enter it.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

from .female_roster import IDENTITIES

POLICY = 'female_only'
EVIDENCE = 'gf2_female_playable_roster_2026_10_13'
#: every playable character of the install's own table: the model extractor's scope
POLICY_ALL = 'all_playable'
_mode = threading.local()


@contextmanager
def all_playable():
    """Within: every playable identity of the install's table passes, not only the reviewed women."""
    previous = getattr(_mode, 'everyone', False)
    _mode.everyone = True
    try:
        yield
    finally:
        _mode.everyone = previous


def everyone() -> bool:
    return getattr(_mode, 'everyone', False)


def policy() -> str:
    """The scope a selection made now is recorded under."""
    return POLICY_ALL if everyone() else POLICY
SOURCES = (
    'https://www.iopwiki.com/index.php?title=Doll&oldid=176479',
    'https://www.iopwiki.com/index.php?title=GFL2_Doll_Index&oldid=175254',
)


def verified(character: dict) -> bool:
    """Reject unknown identities and explicit non-female classifications (in :func:`all_playable`: any
    identity the install's table joins, ``playable.roster``)."""
    identity = tuple(character.get(key) for key in ('id', 'gun_id', 'name', 'model'))
    if everyone():
        return all(value is not None for value in identity)
    return character.get('gender', 'female') == 'female' and identity in IDENTITIES


def classify(character: dict) -> dict:
    """Annotate a table identity; unknown entries remain available for diagnosis."""
    identity = tuple(character.get(key) for key in ('id', 'gun_id', 'name', 'model'))
    approved = character.get('gender', 'female') == 'female' and identity in IDENTITIES
    return dict(character, gender='female' if approved else character.get('gender', 'unknown'),
                gender_evidence=EVIDENCE if approved else None)


def selection_verified(selection: dict) -> bool:
    """Require a current policy record and a nonempty verified character set; a :data:`POLICY_ALL`
    selection only inside :func:`all_playable`."""
    characters = selection.get('characters', [])
    if selection.get('gender_scope') == POLICY_ALL:
        return everyone() and selection.get('eligible') is True and bool(characters) and all(map(verified, characters))
    return (selection.get('eligible') is True and selection.get('gender_scope') == POLICY
            and bool(characters) and all(verified(c) and c.get('gender') == 'female'
            and c.get('gender_evidence') == EVIDENCE for c in characters))
