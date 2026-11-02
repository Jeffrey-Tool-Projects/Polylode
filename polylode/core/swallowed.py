"""Exceptions a caller chose to survive — counted, and the first of each kind printed.

Keeping going after an error is often right (one unreadable object in a
third-party bundle should not stop a character loading, a progress label that
cannot be updated should not stop the load it reports on), but **keeping going
without saying so** turns a failure into a result.  Two sections paid for that:
the eviction guard of §7.148 sat behind an ``except Exception: pass`` that ate
the ``ImportError`` meant to report it, for as long as it existed; and §7.181's
texture generators raised on every call while four sections cited the run as
evidence that they had been exercised.

So a handler that survives calls :func:`note`.  The tally is **countable** — a
tool prints :func:`tally` beside its results, the way ``origin_audit`` prints
``meshimport.load_failures`` — and the first occurrence of each
``(where, exception type)`` goes to stderr with its message, so the reason is
on screen the first time and the log is not flooded by the thousandth.

The specific tallies that already exist (``bake.PROVIDER_ERRORS``,
``meshimport.LOAD_FAILURES``, ``TextureCache.evicted``) stay: each of those is a
statement about *what a run means* and has its own reader.  This one is for the
handlers that had no reader at all.
"""
from __future__ import annotations

import sys

#: ``"where: ExceptionType"`` → how many times it was swallowed this process.
SWALLOWED: dict[str, int] = {}


def note(where: str, exc: BaseException) -> None:
    """Record that *where* caught *exc* and carried on.

    *where* names the handler (``"module.function: what failed"``), not the
    exception, so two different failures at one site count separately and one
    failure at two sites is not merged.  The first time a key is seen, one line
    goes to stderr; after that it is only counted.  Never raises — it runs
    inside ``except`` blocks, and a reporter that can fail is a second place
    for the original error to disappear.
    """
    key = f"{where}: {type(exc).__name__}"
    n = SWALLOWED.get(key, 0) + 1
    SWALLOWED[key] = n
    if n != 1:
        return
    try:
        msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        print(f"[swallowed] {key}" + (f": {msg[:200]}" if msg else ""),
              file=sys.stderr)
    except Exception:                                # noqa: BLE001
        # stderr can be None or closed in a frozen GUI build; the count above
        # is already recorded, which is the part a tool reads.
        return


def tally() -> dict[str, int]:
    """A copy of the counts — empty means nothing was swallowed this process."""
    return dict(SWALLOWED)


def clear() -> None:
    """Forget the counts (a tool that runs several passes in one process)."""
    SWALLOWED.clear()


def summary(limit: int = 12) -> list[str]:
    """The tally as printable lines, most frequent first (at most *limit*)."""
    rows = sorted(SWALLOWED.items(), key=lambda kv: (-kv[1], kv[0]))
    return [f"{n:6d}  {key}" for key, n in rows[:limit]]
