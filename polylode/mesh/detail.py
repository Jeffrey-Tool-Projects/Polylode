"""Global mesh detail level.

Every generator reads its tessellation counts through here, so one setting
trades triangle count against silhouette smoothness.  The character is a
cel-shaded figure viewed at roughly full-body or bust framing, where the
difference between "medium" and "high" is invisible but the triangle count —
and therefore the outline pass — nearly halves.
"""
from __future__ import annotations

LEVELS = {
    "low":    0.58,
    "medium": 0.78,
    "high":   1.00,
    "ultra":  1.25,
}

_level = "medium"


def set_level(name: str) -> str:
    global _level
    if name in LEVELS:
        _level = name
    return _level


def level() -> str:
    return _level


def scale() -> float:
    return LEVELS[_level]


def n(base: int, minimum: int = 4) -> int:
    """Scaled segment count, never below ``minimum``."""
    return max(minimum, int(round(base * LEVELS[_level])))


def even(base: int, minimum: int = 4) -> int:
    """Scaled count forced even — needed where a ring is mirrored."""
    v = n(base, minimum)
    return v if v % 2 == 0 else v + 1


def count(base: int, minimum: int = 1) -> int:
    """Scaled instance count (hair strands, pleats, accessory segments)."""
    return max(minimum, int(round(base * LEVELS[_level])))
