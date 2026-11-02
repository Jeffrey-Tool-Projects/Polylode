"""Illusion's ``AnimationKeyInfo`` blobs — the curves behind every shape slider.

``ChaControl.InitShapeBody`` / ``InitShapeFace`` do not morph vertices: Origin has no
Unity blendshapes on the body (``m_Shapes`` is empty).  A slider instead drives a
**bone's local TRS** along a pre-baked curve, and the mesh follows through normal
skinning.  Those curves live in two ``TextAsset`` blobs::

    chara/oo_base.unity3d      → cf_anmShapeBody     (119 entries)
    chara/bo_head_00.unity3d   → cf_anmShapeHead_00  ( 89 entries)

Wire format (little-endian, written by .NET ``BinaryWriter``)::

    int32                       number of entries
    per entry:
        7-bit length-prefixed UTF-8 string   entry name
        int32                                number of keys
        keys × (int32 + 9 × float32)         one key, see below

A key is ``AnmKeyInfo``: ``int no`` then ``Vector3 pos``, ``Vector3 rot``,
``Vector3 scl``.  Rotation is **Euler degrees**, not a quaternion — that is why
the category table can flag ``rx``/``ry``/``rz`` independently (a quaternion has
no separable axes).  ``no`` is the key's own index, 0…24 in all 208 shipped
entries; it is read and discarded rather than folded into the stride, so a build
that starts using it shows up as a parse error instead of silently shifting
every field by one.  (Reading it as a float instead, as this parser first did,
makes every key look like ``0.0`` — 24 as float32 bits is 3.4e-44.)

Every shipped entry carries exactly 25 keys spanning slider rate 0…1.

Rotation must not be lerped like the other two
----------------------------------------------
The curves store Euler angles in **0…360**, so a track that passes through zero
reads ``… 0, 0, 359.5, 359, 358.5 …`` and the segment that crosses carries a
360° jump.  ``GetInfo`` interpolates position and scale with ``Vector3.Lerp``
but each rotation component with ``Mathf.LerpAngle``, which takes the short way
round; interpolating that segment linearly instead sweeps almost a full turn —
inside it the error reaches **172.8°**.

That segment is not in some corner of the range.  The tracks are authored
symmetrically about key 12, which is rate 0.5 and holds the neutral value 0, so
the crossing sits **at the middle of the slider** — where most cards are.  It
hit 16 body curves (bust size and rotation, belly, hip, skirt, thigh) and 13
face curves (both ears, eye tilt, all six eyebrow rows, ``cf_J_megane_rx_ear``);
the test card was carrying 81.3° of nonsense on ``cf_a_bust02/03_size``, worth
72.7 mm at the end of the nipple chain.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..prefab import text_asset_bytes

FLOATS_PER_KEY = 10          # ``int no`` + three Vector3, all four bytes wide


def lerp_angle(a, b, t):
    """``UnityEngine.Mathf.LerpAngle``, componentwise.

    ``d = Repeat(b - a, 360); if (d > 180) d -= 360; return a + d * Clamp01(t)``
    with ``Repeat(x, n) = x - Floor(x / n) * n`` — both read out of
    ``UnityEngine.dll``'s IL, not from memory.  ``np.remainder`` is floor-based
    and so is exactly ``Repeat`` for a positive length.  *t* is already in
    ``[0, 1)`` by construction here, so the clamp is left implicit.
    """
    d = np.remainder(np.asarray(b) - np.asarray(a), 360.0)
    return a + np.where(d > 180.0, d - 360.0, d) * t


def _read_7bit(buf: bytes, off: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        if off >= len(buf):
            raise ValueError("字串長度前綴超出檔尾")
        c = buf[off]
        off += 1
        n |= (c & 0x7F) << shift
        if not c & 0x80:
            return n, off
        shift += 7
        if shift > 28:
            raise ValueError("字串長度前綴不合法")


# Curves whose rotation SliderUnlocker refuses to extrapolate, keeping the
# clamped value instead (``SliderMath.SafeCalculateRotation``).  Transcribed
# rather than reasoned about: the list is breasts, mouth, lower legs, brow tips
# and the bust size curves, i.e. the places where a straight line through the
# two end keys sends a joint somewhere the author never drew.
_ROT_LOCKED = ("cf_s_Mune", "cf_s_Mouth", "cf_s_LegLow", "cf_s_MayuTip")


def unlocked_sample(keys, name: str, rate: float, clamped):
    """A slider outside 0…1 — ``SliderUnlocker``'s arithmetic, not the game's.

    **86.6% of this install's 1,386 cards store at least one shape slider
    outside 0…1** (1,149 on the face, 759 on the body), so this is the normal
    case, not a curiosity.  They can, because BetterRepack ships
    ``SliderUnlocker``, and a card authored with it is only meaningful with it:
    stock Origin would index off the end of the key list, which is why the plugin
    clamps the rate on the way in and then *replaces* the answer on the way out
    (``Hooks.GetInfoPreHook`` / ``GetInfoPostHook``).

    What it replaces it with ignores the intermediate keys entirely — a straight
    line through the **first and last** key, evaluated unclamped
    (``SliderMath.CalculatePosition`` / ``CalculateScale``)::

        first + (last - first) * rate

    Rotation is the same line, but its delta is unwrapped first: if the total
    ``last - first`` disagrees in sign with the curve's own first step, ±360 is
    added so the extrapolation continues the way the curve actually turns
    (``CalculateRotation``).  Six curve families are excluded and keep the
    clamped rotation (:data:`_ROT_LOCKED`).

    Transcribing a plugin is the rule here, not an exception: the reference
    install is the authority (CLAUDE.md 一之二), and Sideloader's id rewriting
    (§7.72) and ABMX's bone scaling (§7.98) are already in the pipeline for the
    same reason.  Without it every one of those 1,200 cards is shaped at the
    clamp — measured against a live character, the beret card's
    ``cf_J_FaceUp_ty`` came out 2.59% too large, which put its hat 4.3 mm high
    (§7.102).

    *clamped* is the vanilla result, already computed, exactly as the plugin's
    post-hook receives it.
    """
    n = len(keys)
    first, last = keys[0], keys[n - 1]
    pos = first[1:4] + (last[1:4] - first[1:4]) * rate
    scl = first[7:10] + (last[7:10] - first[7:10]) * rate
    rot = clamped[1]
    # ``CalculateRotation`` reads ``keys[1]`` for the direction of the first
    # step, so a one-key curve would throw there.  It cannot matter: with one
    # key ``first`` and ``last`` are the same row, the delta is zero, and the
    # line is the clamped value — which is what is kept.
    if n > 1 and not (any(s in name for s in _ROT_LOCKED)
                      or ("thigh" in name and "01" in name)
                      or (name.startswith("cf_a_bust") and name.endswith("_size"))):
        delta = last[4:7] - first[4:7]
        step_up = (keys[1][4:7] - first[4:7]) >= 0.0
        delta = np.where((delta > 0.0) & ~step_up, delta - 360.0, delta)
        delta = np.where((delta < 0.0) & step_up, delta + 360.0, delta)
        rot = first[4:7] + delta * rate
    return pos, rot, scl


@dataclass
class ShapeCurves:
    """entry name → (keys, 10) float array."""

    entries: dict[str, np.ndarray]

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def sample(self, name: str, rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate one entry at slider *rate* → (pos, rot_deg, scale).

        ``AnimationKeyInfo.GetInfo(name, rate, ref Vector3[3], bool[3])``, the
        overload ``ShapeInfoBase.ChangeValue`` calls: the two ends are taken
        whole (which is also how the game avoids indexing one past the last
        key), everything between is ``Vector3.Lerp`` for position and scale and
        ``Mathf.LerpAngle`` per component for rotation.

        Outside 0…1 it is :func:`unlocked_sample` instead — see there for why a
        slider gets outside 0…1 at all, and why that is not an edge case.

        The ``bool[3]`` says which of the three vectors to fill; the ones left
        out keep the caller's zeroed slot and are then gated again by the
        category row's per-component ``use`` flags, so filling all three and
        letting :func:`shape_run.sources` apply ``use`` is the same program.
        """
        keys = self.entries[name]
        n = len(keys)
        rate = float(rate)
        clamped = min(max(rate, 0.0), 1.0)
        if n == 1 or clamped == 0.0:
            k = keys[0]
            out = k[1:4].copy(), k[4:7].copy(), k[7:10].copy()
        elif clamped == 1.0:
            k = keys[n - 1]
            out = k[1:4].copy(), k[4:7].copy(), k[7:10].copy()
        else:
            t = clamped * (n - 1)
            i = int(np.floor(t))
            f = t - i
            a, b = keys[i], keys[i + 1]
            out = (a[1:4] + (b[1:4] - a[1:4]) * f,
                   lerp_angle(a[4:7], b[4:7], f),
                   a[7:10] + (b[7:10] - a[7:10]) * f)
        if clamped != rate:
            out = unlocked_sample(keys, name, rate, out)
        return out

    def key_count(self, name: str) -> int:
        return int(len(self.entries[name]))


def parse(blob: bytes) -> ShapeCurves:
    if len(blob) < 4:
        raise ValueError("AnimationKeyInfo 為空")
    (count,) = struct.unpack_from("<I", blob, 0)
    off = 4
    out: dict[str, np.ndarray] = {}
    for _ in range(count):
        length, off = _read_7bit(blob, off)
        name = blob[off:off + length].decode("utf-8", "replace")
        off += length
        (nkeys,) = struct.unpack_from("<I", blob, off)
        off += 4
        span = nkeys * FLOATS_PER_KEY
        need = span * 4
        if off + need > len(blob):
            raise ValueError(f"{name}: 關鍵幀資料長度不足")
        vals = np.frombuffer(blob, dtype="<f4", count=span, offset=off)
        off += need
        out[name] = vals.reshape(nkeys, FLOATS_PER_KEY).astype(np.float64)
    if off != len(blob):
        raise ValueError(f"AnimationKeyInfo 尾端剩下 {len(blob) - off} 位元組未解析")
    return ShapeCurves(out)


def to_payload(value: ShapeCurves | None) -> dict | None:
    """The bake's JSON of a parsed table (``bake.write_named_table``); exact —
    every float32 of the blob is a float64 whose ``repr`` reads back bit for bit."""
    if value is None:
        return None
    return {"curves": {n: a.tolist() for n, a in value.entries.items()}}


def from_payload(payload: dict | None) -> ShapeCurves | None:
    """:func:`to_payload` back."""
    if payload is None:
        return None
    return ShapeCurves({n: np.asarray(v, dtype=np.float64)
                        for n, v in payload["curves"].items()})


def source_of(mod=None) -> str:
    """Bake source a curve table is filed under: the install, or the mod row's GUID.

    A mod row's request follows Sideloader's rule (registrants of the bundle
    path in load order, then the install — §7.481), a stock row's reads the
    install; the two can answer differently for one ``(bundle, asset)``.
    """
    from .. import bake
    if mod is None:
        return bake.INSTALL
    return str(getattr(mod, "guid", "") or "")


def _read_blob(bundle_rel: str, asset: str, mod=None) -> bytes | None:
    """The named TextAsset's bytes, out of the install or through a mod row.

    A head mod ships its **own** copy of the curve table -- 178 of the local
    mod rows in CategoryNo 100 carry a ``cf_anmShapeHead_00`` TextAsset inside
    their own bundle -- and that is the one the row names, so reading the base
    game's instead would play the card's sliders against a different head.
    For a mod row the bundle is the one Sideloader would hand out for the
    table's own ``(bundle, asset)`` (§7.481).
    """
    if mod is not None:
        from .. import sideloader as SL
        env = SL.load_mod_bundle(mod, bundle=bundle_rel, asset=asset)
        return None if env is None else text_asset_bytes(env, asset)
    from .load_mesh import _open_bundle
    # a stock table by name is Sideloader's to hand out too (§7.489)
    loader = _open_bundle(bundle_rel, asset)
    if loader is None:
        return None
    try:
        return text_asset_bytes(loader.env, asset)
    finally:
        loader.env = None
        loader.by_id = {}


_CACHE: dict[tuple[str, str, str], ShapeCurves | None] = {}


def curves(bundle_rel: str, asset: str, mod=None, *,
           from_bundle: bool = False) -> ShapeCurves | None:
    """Parsed curves for one asset — from the bake, else out of the bundle.

    Cached per process.  ``from_bundle`` is the converter's read: it skips the
    bake (it is what fills it) and the cache.
    """
    source = source_of(mod)
    key = (source, bundle_rel, asset)
    if not from_bundle:
        if key in _CACHE:
            return _CACHE[key]
        if source:
            from .. import bake
            recorded, payload = bake.named_table(source, bundle_rel, asset)
            if recorded:
                _CACHE[key] = from_payload(payload)
                return _CACHE[key]
    result: ShapeCurves | None = None
    try:
        blob = _read_blob(bundle_rel, asset, mod)
        if blob:
            result = parse(blob)
    except (ValueError, struct.error, OSError):
        result = None
    if not from_bundle:
        _CACHE[key] = result
    return result


BODY_CURVES = ("chara/oo_base.unity3d", "cf_anmShapeBody")


def body_curves(*, from_bundle: bool = False) -> ShapeCurves | None:
    return curves(*BODY_CURVES, from_bundle=from_bundle)


def head_request(head) -> tuple:
    """``(bundle, asset, mod)`` the face sliders of *head* (a ``HeadAssets``) read.

    The row's ``ShapeAnime`` column.  A mod head asks through its row only when
    the table sits in the head's own bundle; one that names a stock bundle is
    read like a stock head.  One owner for the shaping pass and the converter,
    so the bake is filed under the key the run time looks up.
    """
    mod = head.mod if head.source == "mod" and head.shape_bundle == head.bundle else None
    return head.shape_bundle, head.shape_asset, mod


def head_curves(bundle: str = "chara/bo_head_00.unity3d",
                asset: str = "cf_anmShapeHead_00",
                mod=None, *, from_bundle: bool = False) -> ShapeCurves | None:
    """The face sliders' key-frame table — **per head**, not a constant.

    The ``bo_head`` row's ``ShapeAnime`` column names it (``cf_anmShapeHead_00``
    for head 0, ``_02`` for head 200 …), so a card that changed its head was
    having its face sliders played against another head's curves (§7.41).
    The defaults keep the stock head working for callers with no card.
    """
    return curves(bundle, asset, mod, from_bundle=from_bundle)
