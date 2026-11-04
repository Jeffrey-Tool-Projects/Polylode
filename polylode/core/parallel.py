"""Splitting work over worker processes: how many, and in which runs."""
from __future__ import annotations

import os

#: the most worker processes one step starts; fewer on a computer of fewer cores (half of them)
WORKERS_MOST = 6


def worker_count(items: int, most: int = WORKERS_MOST) -> int:
    """How many workers *items* pieces of work are split over: half the cores, at most *most*, at most one each."""
    return max(1, min(items, max(1, (os.cpu_count() or 2) // 2), most))


def slices(items: list, count: int) -> list[list]:
    """*items* in *count* runs of consecutive items, as even as they go: put back together in order, they are *items*."""
    size, extra = divmod(len(items), count)
    out, start = [], 0
    for k in range(count):
        stop = start + size + (1 if k < extra else 0)
        out.append(items[start:stop])
        start = stop
    return [run for run in out if run]
