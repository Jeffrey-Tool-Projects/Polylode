"""The games whose converted characters this runtime runs, and where each one's model folders are.

The runtime (:mod:`.actor`, :mod:`.body`, :mod:`.controller`, the H path) was
written for Endfield's items.  A character converted from another game goes
through the same path (CLAUDE.md 四之二, §7.1209／§7.1232): its conversion
writes the same item format into its own model folder
``Charas/<game>/<character>/<model>/``, with a descriptor naming its game.

What differs between games is read from each item's data, never from a
character list: Endfield's surfaces by their shader names, another game's by
the role table its conversion declared (``part_roles.DECLARED_ROLE``), its
scope by the owner's recorded adult basis (``body_scope.declared_scope``).
"""
from __future__ import annotations

#: ``source`` of a game's descriptors → its folder under ``Charas`` (Endfield first: it is the one
#: a new Endfield model folder is made in, ``store.ensure_home``).
GAMES = {"endfield": "Endfield", "arknights": "Arknights"}
