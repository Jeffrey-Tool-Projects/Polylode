"""How far a long step has got, for whoever shows it: the extractor's window, through its job's ``PROGRESS`` lines.

A converter calls :func:`report` inside its long loops (``done`` of ``total``) and :func:`step` where it starts
a step it cannot count.  Nothing listens unless a caller installed a sink with :func:`reporting`, so the
converters run the same with no window and pay one empty loop per call.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

#: ``sink(done, total, text)``; *total* 0 for a step that cannot be counted, *text* "" to keep the last one
Sink = Callable[[int, int, str], None]

_sinks: list[Sink] = []


def report(done: int, total: int, text: str = "") -> None:
    for sink in _sinks:
        sink(int(done), int(total), str(text))


def step(text: str) -> None:
    """A step that starts now and has no count: shown as moving, with *text*."""
    report(0, 0, text)


@contextmanager
def reporting(sink: Sink):
    """*sink* hears every report made inside the block."""
    _sinks.append(sink)
    try:
        yield sink
    finally:
        _sinks.remove(sink)
