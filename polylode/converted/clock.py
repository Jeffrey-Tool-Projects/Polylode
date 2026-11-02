"""Scene simulation time, stepped by the update owner and sampled by render passes."""
from __future__ import annotations

import math
from dataclasses import dataclass


def _seconds(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('scene clock requires nonnegative finite seconds')
    return result


@dataclass(frozen=True)
class FrameTime:
    """Immutable host simulation sample, before any source-specific float32 conversion."""
    elapsed: float = 0.
    delta: float = 0.
    smooth_delta: float = 0.
    index: int = 0

    def __post_init__(self) -> None:
        for name in ('elapsed', 'delta', 'smooth_delta'):
            object.__setattr__(self, name, _seconds(getattr(self, name)))
        if type(self.index) is not int or self.index < 0:
            raise ValueError('scene frame index must be a nonnegative integer')


class FrameClock:
    """One scene's simulation clock; rendering/capture never advances it.

    The host update owner explicitly supplies both deltas. Elapsed time stays
    double precision until each source packs its own shader ABI. A rejected
    update leaves the entire previous snapshot intact.
    """

    def __init__(self) -> None:
        self._frame = FrameTime()

    def snapshot(self) -> FrameTime:
        """Return the current immutable sample, shared by every pass and capture of this frame."""
        return self._frame

    def advance(self, delta: float, *, smooth_delta: float) -> FrameTime:
        """Advance exactly once for an actual host simulation update, without reading wall time."""
        delta, smooth_delta = _seconds(delta), _seconds(smooth_delta)
        candidate = FrameTime(self._frame.elapsed + delta, delta, smooth_delta, self._frame.index + 1)
        self._frame = candidate
        return candidate
