"""The one axis convention shared with scene files, free of renderer imports.

A Blender worker imports this without the renderer's GL, which
:mod:`.rigscene` (through :mod:`.shading`) needs.
"""
from __future__ import annotations

import numpy as np

#: this project's space (Y up) → Z up: ``(x, y, z) → (x, −z, y)``
TO_Z_UP = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
