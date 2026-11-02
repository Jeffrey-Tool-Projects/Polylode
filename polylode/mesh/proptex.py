"""Tileable material maps for the procedural scenery (#125).

Generated, not downloaded: every map is a function of a name and a seed, so the
release carries no image files and nothing of anyone else's.  The maps are
**modulation** — the shader multiplies them into the palette colour
(``albedo.rgb *= t.rgb``) — so each keeps its mean near 1 and draws only the
variation: grain, seams, grout, weave.  Every map wraps at its edges (the
noise lattice is periodic), because a prop's UVs run in metres and repeat.
"""
from __future__ import annotations

import numpy as np

SIZE = 256


def _lattice_noise(size: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Value noise on a ``cells × cells`` lattice that wraps, smoothstep-interpolated."""
    grid = rng.random((cells, cells))
    t = np.arange(size) * cells / size
    i0 = np.floor(t).astype(int) % cells
    i1 = (i0 + 1) % cells
    f = t - np.floor(t)
    f = f * f * (3.0 - 2.0 * f)
    rows0 = grid[i0][:, i0] * (1 - f)[None, :] + grid[i0][:, i1] * f[None, :]
    rows1 = grid[i1][:, i0] * (1 - f)[None, :] + grid[i1][:, i1] * f[None, :]
    return rows0 * (1 - f)[:, None] + rows1 * f[:, None]


def _fbm(size: int, cells: int, octaves: int, rng: np.random.Generator,
         gain: float = 0.5) -> np.ndarray:
    """Octaves of :func:`_lattice_noise`, normalised to 0..1."""
    out = np.zeros((size, size))
    amp, total = 1.0, 0.0
    for k in range(octaves):
        c = cells * (2 ** k)
        if c > size:
            break
        out += amp * _lattice_noise(size, c, rng)
        total += amp
        amp *= gain
    out /= max(total, 1e-9)
    lo, hi = out.min(), out.max()
    return (out - lo) / max(hi - lo, 1e-9)


def _grey(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + (hi - lo) * np.clip(v, 0.0, 1.0)


def _uv(size: int):
    t = np.arange(size) / size
    return np.meshgrid(t, t)          # (u across, v down)


def plaster(size: int, rng) -> np.ndarray:
    return _grey(_fbm(size, 4, 5, rng, 0.55), 0.90, 1.0)


def concrete(size: int, rng) -> np.ndarray:
    base = _grey(_fbm(size, 4, 6, rng, 0.6), 0.80, 1.0)
    pits = rng.random((size, size)) > 0.996
    base[pits] *= 0.75
    return base


def wood(size: int, rng, planks: int = 4) -> np.ndarray:
    """Planks along v: grain lines bent by noise, a darker seam between planks."""
    u, v = _uv(size)
    warp = _fbm(size, 4, 3, rng)
    plank = np.floor(u * planks)
    offset = rng.random(planks)[plank.astype(int) % planks]
    grain = 0.5 + 0.5 * np.sin((u * planks * 9.0 + warp * 3.0 + offset * 7.0) * np.pi * 2.0)
    tone = 0.84 + 0.10 * offset + 0.08 * grain
    seam = np.abs((u * planks) % 1.0 - 0.5) > 0.48
    tone[seam] *= 0.72
    return np.clip(tone, 0.0, 1.0)


def floorboards(size: int, rng, rows: int = 4, cols: int = 2) -> np.ndarray:
    """Boards across u, staggered ends along v."""
    u, v = _uv(size)
    row = np.floor(v * rows).astype(int) % rows
    stagger = rng.random(rows)[row]
    col = np.floor(u * cols + stagger)
    shade = rng.random((rows, cols * 2))[row, col.astype(int) % (cols * 2)]
    warp = _fbm(size, 4, 3, rng)
    grain = 0.5 + 0.5 * np.sin((v * rows * 14.0 + warp * 2.5) * np.pi * 2.0)
    tone = 0.84 + 0.10 * shade + 0.06 * grain
    seam = (np.abs((v * rows) % 1.0 - 0.5) > 0.485) | \
        (np.abs((u * cols + stagger) % 1.0 - 0.5) > 0.493)
    tone[seam] *= 0.70
    return np.clip(tone, 0.0, 1.0)


def tiles(size: int, rng, n: int = 4) -> np.ndarray:
    """A square grid with grout lines, each tile a touch different."""
    u, v = _uv(size)
    iu, iv = np.floor(u * n).astype(int), np.floor(v * n).astype(int)
    shade = rng.random((n, n))[iv % n, iu % n]
    tone = 0.92 + 0.07 * shade + 0.03 * _fbm(size, 8, 3, rng)
    grout = (np.abs((u * n) % 1.0 - 0.5) > 0.47) | (np.abs((v * n) % 1.0 - 0.5) > 0.47)
    tone[grout] = 0.66
    return np.clip(tone, 0.0, 1.0)


def fabric(size: int, rng, threads: int = 64) -> np.ndarray:
    u, v = _uv(size)
    weave = 0.5 + 0.25 * np.sin(u * threads * np.pi * 2.0) * np.sin(v * threads * np.pi * 2.0)
    return np.clip(0.86 + 0.10 * weave + 0.06 * _fbm(size, 4, 3, rng), 0.0, 1.0)


def stone(size: int, rng) -> np.ndarray:
    blotch = _fbm(size, 3, 6, rng, 0.62)
    return _grey(blotch, 0.74, 1.0)


def sand(size: int, rng) -> np.ndarray:
    fine = rng.random((size, size))
    return np.clip(0.90 + 0.06 * _fbm(size, 6, 4, rng) + 0.05 * fine, 0.0, 1.0)


def leaves(size: int, rng) -> np.ndarray:
    return _grey(_fbm(size, 8, 4, rng, 0.65), 0.72, 1.0)


def bark(size: int, rng) -> np.ndarray:
    u, v = _uv(size)
    warp = _fbm(size, 4, 3, rng)
    ridges = 0.5 + 0.5 * np.sin((u * 12.0 + warp * 2.0) * np.pi * 2.0)
    return np.clip(0.78 + 0.18 * ridges * (0.6 + 0.4 * _fbm(size, 8, 2, rng)), 0.0, 1.0)


def grass(size: int, rng) -> np.ndarray:
    """Short turf: soft patches and a fine speckle of blades."""
    blades = rng.random((size, size))
    return np.clip(0.80 + 0.12 * _fbm(size, 6, 4, rng) + 0.10 * blades, 0.0, 1.0)


def litter(size: int, rng) -> np.ndarray:
    """A forest floor: blotches of leaf litter and darker soil between."""
    blotch = _fbm(size, 5, 5, rng, 0.6)
    speck = rng.random((size, size)) > 0.985
    out = 0.78 + 0.20 * blotch
    out[speck] *= 0.8
    return np.clip(out, 0.0, 1.0)


def brushed(size: int, rng) -> np.ndarray:
    streak = _lattice_noise(size, 64, rng)
    streak = np.repeat(streak.mean(axis=1, keepdims=True), size, axis=1)
    return np.clip(0.88 + 0.10 * streak + 0.02 * rng.random((size, size)), 0.0, 1.0)


#: name → generator; a palette key opts in through :data:`FOR_PALETTE`.
GENERATORS = {
    "plaster": plaster, "concrete": concrete, "wood": wood, "floorboards": floorboards,
    "tiles": tiles, "fabric": fabric, "stone": stone, "sand": sand, "leaves": leaves,
    "bark": bark, "brushed": brushed, "grass": grass, "litter": litter,
}

#: ``props.PALETTE`` key → ``(generator, metres one repeat covers)``.  A key not
#: listed stays a flat colour (glass, water, paper, chalk …).
FOR_PALETTE = {
    "wood": ("wood", 1.0), "wood_dark": ("wood", 1.0),
    "metal": ("brushed", 1.0), "metal_dark": ("brushed", 1.0),
    "concrete": ("concrete", 2.0), "wall": ("plaster", 2.0), "wall_warm": ("plaster", 2.0),
    "floor": ("floorboards", 1.6), "foliage": ("leaves", 1.5), "foliage2": ("leaves", 1.5),
    "palm": ("leaves", 1.5), "trunk": ("bark", 1.0), "fabric": ("fabric", 0.5),
    "fabric2": ("fabric", 0.5), "sand": ("sand", 2.0), "stone": ("stone", 1.5),
    "stone_dark": ("stone", 1.5), "red": ("plaster", 2.0), "white": ("plaster", 2.0),
    "green": ("plaster", 2.0), "tile": ("tiles", 1.2), "grass": ("grass", 1.5),
    "forest_floor": ("litter", 2.5), "trim": ("plaster", 2.0),
}


def generate(name: str, seed: int = 0, size: int = SIZE) -> np.ndarray:
    """One map as ``(size, size, 4)`` ``float32`` RGBA in 0..1 (grey, opaque).

    The format of every generator ``gl.textures.TEXTURE_FACTORY`` holds.
    """
    fn = GENERATORS[name]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), *name.encode("utf-8")]))
    grey = np.clip(fn(int(size), rng), 0.0, 1.0).astype(np.float32)
    out = np.ones((int(size), int(size), 4), dtype=np.float32)
    out[..., :3] = grey[..., None]
    return out
