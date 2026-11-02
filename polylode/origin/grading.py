"""Illusion's colour grading LUT, read from the local install.

The card thumbnails and everything the player sees in game go through
``AmplifyColorEffect``, and §7.40 measured what it is actually configured with:
one 32-cube LUT named ``Lut_Amount``, at full strength, with no tone mapper and
no blend --

    Tonemapper 0   Exposure 1.0   BlendAmount 0.0   UseDepthMask False
    LutTexture -> sharedassets2.assets: Texture2D 'Lut_Amount' (1024x32, RGB24)

Both the chara maker (``level2``) and life mode (``level16``) name the same
texture, so there is one grade to copy rather than one per screen.

It lives beside the executable rather than in ``abdata`` -- image effects are
built-in resources, not character assets -- which is why this does not go
through ``install.bundle()``.  Nothing is bundled with the project: a machine
without Koikatsu installed gets ``None`` here and the grade stays off, the same
contract as every other Illusion asset (CLAUDE.md 四).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import install
from ..core import swallowed

LUT_NAME = "Lut_Amount"
# where the shipped scenes' cameras point; both apps keep it in the same file
LUT_ASSETS = ("Koikatu_Data/sharedassets2.assets",
              "CharaStudio_Data/sharedassets2.assets")
_cache: list = [None, False]            # [array, tried]


def cached_path() -> Path:
    return install.cache_dir() / "grading" / f"{LUT_NAME}.npy"


def extract_from(root) -> dict:
    """Pull the LUT out of an install into the project cache.

    Called by ``sync_from_install``; separate so it can be re-run on its own
    after a game update without re-mirroring 30 GB of bundles.
    """
    root = Path(root)
    out = {"ok": False, "reason": "", "dest": str(cached_path())}
    if not root.is_dir():
        out["reason"] = f"找不到安裝目錄 {root}"
        return out
    try:
        import UnityPy
    except Exception as exc:                         # noqa: BLE001
        out["reason"] = f"UnityPy 不可用: {exc}"
        return out
    tried = []
    for rel in LUT_ASSETS:
        p = root / rel
        tried.append(str(p))
        if not p.is_file():
            continue
        try:
            env = UnityPy.load(str(p))
        except Exception as exc:                     # noqa: BLE001
            out["reason"] = f"{p.name} 讀取失敗: {exc}"
            continue
        for obj in env.objects:
            if obj.type.name != "Texture2D":
                continue
            try:
                tex = obj.read()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.grading.extract_from: obj.read", exc)
                continue
            if str(getattr(tex, "m_Name", "")) != LUT_NAME:
                continue
            arr = np.asarray(tex.image.convert("RGB"), dtype=np.uint8)
            dst = cached_path()
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, arr)
            _cache[0], _cache[1] = arr, True
            out.update(ok=True, shape=list(arr.shape), source=str(p))
            return out
    if not out["reason"]:
        out["reason"] = f"{LUT_NAME} 不在 {tried}"
    return out


def lut() -> np.ndarray | None:
    """``(32, 1024, 3)`` uint8 strip, or None when it was never synced.

    Runtime reads the project cache only -- ``sync_from_install`` is what
    reaches into the install (§four) -- so a machine that has never synced
    gets None here and the grade stays off rather than the renderer going
    looking for Koikatsu while a screen is opening.  Rows are kept in the
    file's own order: the shader's v is ``g * 31/32 + 0.5/32`` with no flip,
    and ``TextureCache.upload`` is what applies the GL origin flip.
    """
    if _cache[1]:
        return _cache[0]
    _cache[1] = True
    p = cached_path()
    if p.is_file():
        try:
            _cache[0] = np.load(p)
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.grading.lut: _cache[0] = np.load(p)", exc)
            _cache[0] = None
    return _cache[0]


def available() -> bool:
    return lut() is not None


#: The strip's layout: a 32-cube, blue slices side by side, green down.
SIZE = 32
#: :func:`inverse` starts each target from the nearest of this many graded samples per axis …
INVERSE_SAMPLES = 64
#: … and refines it by this many damped Newton steps.
INVERSE_STEPS = 12
_inverse: list = [None, False]


def graded(colour: np.ndarray, strip: np.ndarray) -> np.ndarray:
    """*colour* (``(..., 3)``, gamma, 0‥1) through *strip* the way the post pass samples it.

    ``POST_FS.applyLut``: two neighbouring blue slices, each read bilinearly at
    texel centres, mixed by the blue fraction.  *strip* is ``(32, 1024, 3)``
    in 0‥1, rows in the file's order: the upload flips them for GL's origin,
    so green 0 is the **last** row.
    """
    cube = np.asarray(strip, dtype=np.float64)[::-1].reshape(SIZE, SIZE, SIZE, 3)   # [green, blue, red]
    c = np.clip(np.asarray(colour, dtype=np.float64), 0.0, 1.0) * (SIZE - 1)
    low = np.minimum(np.floor(c).astype(np.int64), SIZE - 2)
    f = c - low
    r0, g0, b0 = low[..., 0], low[..., 1], low[..., 2]
    out = np.zeros(c.shape)
    for db in (0, 1):
        for dg in (0, 1):
            for dr in (0, 1):
                weight = ((f[..., 0] if dr else 1 - f[..., 0]) * (f[..., 1] if dg else 1 - f[..., 1])
                          * (f[..., 2] if db else 1 - f[..., 2]))
                out += weight[..., None] * cube[g0 + dg, b0 + db, r0 + dr]
    return out


def inverse() -> np.ndarray | None:
    """The strip that undoes :func:`lut`, in the same layout, float 0‥1; None without one.

    ``inverse[graded colour] = colour``.  Something drawn by another game's
    programs is not that game's picture once Koikatsu's grade has been through
    it, and the grade is one pass over the whole frame; those programs read
    this on their way out so that the pass lands them where they started
    (``gl/shaders_endfield.py``).

    The grade is steep and couples the channels (mid grey comes out darker
    and tinted), so stepping ``x += target − graded(x)`` does not settle.
    Each grid point instead starts from the nearest of a dense set of graded
    samples and is refined by damped Newton steps on the table's own trilinear
    form.  The grade is not one-to-one everywhere — it sends every dark colour
    to black — and a target it cannot produce is left at the colour whose
    grade is nearest.
    """
    if _inverse[1]:
        return _inverse[0]
    _inverse[1] = True
    strip = lut()
    if strip is None:
        return None
    from scipy.spatial import cKDTree
    forward = np.asarray(strip, dtype=np.float64) / 255.0
    axis = np.linspace(0.0, 1.0, SIZE)
    green, blue, red = np.meshgrid(axis, axis, axis, indexing="ij")
    target = np.stack([red, green, blue], axis=-1).reshape(-1, 3)
    fine = np.linspace(0.0, 1.0, INVERSE_SAMPLES)
    samples = np.stack(np.meshgrid(fine, fine, fine, indexing="ij"), axis=-1).reshape(-1, 3)
    x = samples[cKDTree(graded(samples, forward)).query(target)[1]]
    step = 0.5 / (SIZE - 1)
    for _ in range(INVERSE_STEPS):
        here = graded(x, forward)
        jacobian = np.stack([(graded(np.clip(x + step * np.eye(3)[k], 0.0, 1.0), forward)
                              - graded(np.clip(x - step * np.eye(3)[k], 0.0, 1.0), forward)) / (2 * step)
                             for k in range(3)], axis=-1)
        residual = target - here
        # damped least squares: a flat spot of the table (a crushed black) moves nothing
        normal = np.einsum("nki,nkj->nij", jacobian, jacobian) + 1e-3 * np.eye(3)
        move = np.linalg.solve(normal, np.einsum("nki,nk->ni", jacobian, residual)[..., None])[..., 0]
        trial = np.clip(x + move, 0.0, 1.0)
        better = np.linalg.norm(target - graded(trial, forward), axis=1) < np.linalg.norm(residual, axis=1)
        x[better] = trial[better]
    x = x.reshape(SIZE, SIZE, SIZE, 3)
    _inverse[0] = np.ascontiguousarray(x.reshape(SIZE, SIZE * SIZE, 3)[::-1]).astype(np.float32)
    return _inverse[0]
