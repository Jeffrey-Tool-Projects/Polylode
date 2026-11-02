"""Fresh Koikatsu character + asset loader (project-local data only).

Public entry::

    from polylode.origin.figure import dress_character
    report = dress_character(built, card)

Reads cards via ``chara.cardimport``, resolves ids from ``assets_cache`` lists
and zipmods, and attaches **every** mesh piece (body, full head, hair, all
nine clothes slots, accessories).  No drive-letter game path at runtime.
"""
from __future__ import annotations

from .dress import apply_card_shape, dress_character
from .report import Attached, DressReport

__all__ = ["dress_character", "apply_card_shape", "DressReport", "Attached"]
