"""A JSON form for the structured values the loaders read out of a bundle once.

``bake.write_named_table`` stores a table as JSON (§7.482).  The shape tables
are plain enough to write by hand; the armature's colliders, the bust's
``DynamicBone_Ver02`` specs and the gaze parameters are dataclasses holding
numpy arrays and tuples, and a hand-written form per type would be three more
copies of the same walk.  This is that walk, once:

* ``dict``  → ``{"$m": [[key, value], …]}`` — keys keep their type (the bust
  table is keyed by int);
* ``tuple`` → ``{"$t": […]}``, ``list`` → ``[…]``;
* ``numpy.ndarray`` → ``{"$nd": nested lists, "dt": dtype, "sh": shape}`` and a
  numpy scalar → ``{"$np": value, "dt": dtype}`` — both come back with their
  dtype, so a float32 stays float32 and every value reads back bit for bit
  (``repr`` of a float round-trips);
* a dataclass → ``{"$dc": class name, "f": {field: value}}``.

Decoding a dataclass only instantiates a class the **caller names** (*types*);
an unknown name is an error, never an import.
"""
from __future__ import annotations

import dataclasses

import numpy as np


def encode(value):
    """*value* as JSON-ready data (see the module docstring for the forms)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"$dc": type(value).__name__,
                "f": {f.name: encode(getattr(value, f.name))
                      for f in dataclasses.fields(value)}}
    if isinstance(value, np.ndarray):
        return {"$nd": value.tolist(), "dt": value.dtype.str, "sh": list(value.shape)}
    if isinstance(value, np.generic):
        return {"$np": value.item(), "dt": value.dtype.str}
    if isinstance(value, dict):
        return {"$m": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"$t": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"tablecodec: {type(value).__name__} 沒有 JSON 形式")


def decode(data, types=()):
    """:func:`encode`'s value back; *types* are the dataclasses it may build."""
    by_name = {t.__name__: t for t in types}
    return _decode(data, by_name)


def _decode(data, by_name: dict):
    if isinstance(data, list):
        return [_decode(v, by_name) for v in data]
    if not isinstance(data, dict):
        return data
    if "$nd" in data:
        return np.asarray(data["$nd"], dtype=np.dtype(data["dt"])).reshape(data["sh"])
    if "$np" in data:
        return np.dtype(data["dt"]).type(data["$np"])
    if "$t" in data:
        return tuple(_decode(v, by_name) for v in data["$t"])
    if "$m" in data:
        return {_decode(k, by_name): _decode(v, by_name) for k, v in data["$m"]}
    if "$dc" in data:
        cls = by_name.get(data["$dc"])
        if cls is None:
            raise ValueError(f"tablecodec: 不認得的型別 {data['$dc']!r}")
        return cls(**{k: _decode(v, by_name) for k, v in data["f"].items()})
    raise ValueError(f"tablecodec: 不認得的形式 {sorted(data)[:3]}")
