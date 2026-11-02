"""One-line shim so asset loading can say which stage it is in.

Dressing a character runs as a load job on a worker thread
(``ui.loading.run_cooperative``, §7.526): a tick there only records the stage
label, and the GUI thread shows it.  Where the same code runs on the GUI
thread — a tool, or a load not yet moved to a job — a tick pumps the event
loop instead, rate-limited on a clock (§7.108).

Lives here rather than in either caller because ``bake`` and ``figure`` both
need it and ``figure`` imports ``bake`` — and because the import has to stay
soft: these modules run headless in every tool in ``tools/``, where there is no
Qt at all.  ``ui.loading.report_progress`` rate-limits itself on a clock, so
this is cheap enough to sit inside a decode loop.
"""
from __future__ import annotations

from ..core import swallowed


def tick(msg: str | None = None, *, force: bool = False) -> None:
    """Pump the UI if one is attached.  *msg* ``None`` keeps the current label."""
    try:
        from ..ui.loading import report_progress
        report_progress(msg, force=force)
    except Exception as exc: # noqa: BLE001
        swallowed.note("origin.progress.tick: report_progress", exc)
