"""One-line shim so asset loading can say which stage it is in.

In IslandLife this pumped the game window's loading screen (``ui.loading``); Polylode has no such
window, and every call failed into ``swallowed``.  Its jobs report through ``core.progress`` instead,
so a stage named here is a step on the extractor's progress bar.  A tick without a name only told that
loading screen "still working": here it says nothing.

Lives here rather than in either caller because ``bake`` and ``figure`` both need it and ``figure``
imports ``bake``.
"""
from __future__ import annotations

from ..core import progress


def tick(msg: str | None = None, *, force: bool = False) -> None:
    """The stage *msg* begins (``core.progress.step``); ``None`` keeps the current one."""
    if msg is not None:
        progress.step(msg)
