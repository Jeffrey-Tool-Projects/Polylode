"""Which converted characters are never converted, read by the warehouse build and the model extractor alike."""
from __future__ import annotations

#: Wuthering Waves models the owner excluded (CLAUDE.md, 2026-10-17, §7.1213): Youhu, Encore and Verina.
EXCLUDED_WUWA = frozenset(("R2T1YouHuMd10011", "R2T1AnkeMd10011", "R2T1JueyuanMd10011"))
