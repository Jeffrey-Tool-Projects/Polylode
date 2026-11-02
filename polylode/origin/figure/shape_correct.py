"""``list/shapecorrect`` — the per-bone offsets the bone-type body adds.

A card carries ``ChaFileBody.typeBone``.  With it set, ``Update`` adds a fixed
offset to a couple of dozen bones on top of whatever the sliders asked for::

    b.trfBone.SetLocalScale(curve.x * ... + correctValue[<key>].vctScl.x, ...)

and the head's own scale compensation is scaled by
``1 / (1 + correctValue[cf_s_head].vctScl.y)``.  With it clear, every one of
those terms is the literal 0 the other arm of the ternary supplies, so this
asset is only read when some card actually wants it.

``ShapeBodyInfoFemale.LoadCorrectInfo`` reads the ``shapecorrect`` TextAsset as
a raw ``BinaryReader`` stream — an ``int32`` count followed by nine floats per
entry (position, rotation, scale) — indexed by ``CorrectKeyName``, whose member
order is :data:`shape_update.CORRECT_KEYS`.  There is no name in the file; the
enum *is* the row order, which is why it is transcribed alongside the program
rather than guessed from the bone list.
"""
from __future__ import annotations

import struct

import numpy as np

from ..prefab import text_asset_bytes
from .shape_update import CORRECT_KEYS

BUNDLE = "list/shapecorrect/shapecorrect.unity3d"
ASSET = "shapecorrect"
FLOATS_PER_ENTRY = 9


def parse(blob: bytes) -> dict[str, np.ndarray]:
    """bone key → (3, 3) array of position / rotation / scale offsets."""
    if len(blob) < 4:
        raise ValueError("shapecorrect 為空")
    (count,) = struct.unpack_from("<i", blob, 0)
    need = 4 + count * FLOATS_PER_ENTRY * 4
    if count < 0 or need > len(blob):
        raise ValueError(f"shapecorrect: {count} 筆超出檔案長度")
    vals = np.frombuffer(blob, dtype="<f4", count=count * FLOATS_PER_ENTRY,
                         offset=4).reshape(count, 3, 3).astype(np.float64)
    if count != len(CORRECT_KEYS):
        # The row order is the enum's; a different length means the install's
        # asset and the transcribed enum have drifted apart, and silently
        # pairing them up would put one bone's offset on another.
        raise ValueError(f"shapecorrect: {count} 筆，CorrectKeyName 有 "
                         f"{len(CORRECT_KEYS)} 個")
    return {name: vals[i] for i, name in enumerate(CORRECT_KEYS)}


def _text_asset() -> bytes | None:
    from .load_mesh import _open_bundle
    # read by name, so from the source Sideloader would hand it out of (§7.489)
    loader = _open_bundle(BUNDLE, ASSET)
    if loader is None:
        return None
    try:
        return text_asset_bytes(loader.env, ASSET)
    finally:
        loader.env = None
        loader.by_id = {}


def to_payload(value: dict[str, np.ndarray] | None) -> dict | None:
    """The bake's JSON of the parsed offsets (``bake.write_named_table``)."""
    if value is None:
        return None
    return {"offsets": {n: a.tolist() for n, a in value.items()}}


def from_payload(payload: dict | None) -> dict[str, np.ndarray] | None:
    """:func:`to_payload` back."""
    if payload is None:
        return None
    return {n: np.asarray(v, dtype=np.float64) for n, v in payload["offsets"].items()}


_CACHE: dict[int, dict[str, np.ndarray] | None] = {}


def table(*, from_bundle: bool = False) -> dict[str, np.ndarray] | None:
    """Parsed offsets — from the bake, else out of the bundle; ``None`` when absent.

    Cached per process.  ``from_bundle`` is the converter's read: past the bake
    and the cache.
    """
    if not from_bundle:
        if 0 in _CACHE:
            return _CACHE[0]
        from .. import bake
        recorded, payload = bake.named_table(bake.INSTALL, BUNDLE, ASSET)
        if recorded:
            _CACHE[0] = from_payload(payload)
            return _CACHE[0]
    try:
        blob = _text_asset()
        result = parse(blob) if blob else None
    except (ValueError, struct.error, OSError):
        result = None
    if not from_bundle:
        _CACHE[0] = result
    return result


def offsets() -> dict[str, np.ndarray]:
    """Same, but never ``None`` — an absent asset means no correction."""
    return table() or {}


def head_correct_value(correct: dict[str, np.ndarray] | None) -> float:
    """``ChaControl.UpdateShapeFace``: ``1 / (1 + correctValue[cf_s_head].scl.y)``."""
    row = (correct or {}).get("cf_s_head")
    return 1.0 / (1.0 + (float(row[2][1]) if row is not None else 0.0))
