"""A part of several materials as one part per material: what a viewer, Blender or Unity lists as separate objects.

A source whose character is one mesh with a section per material (Wuthering Waves: one skeletal mesh of hair,
face, upper and lower body, shoes…) exports as one object.  :func:`by_material` gives each material section its
own part, on the same skeleton: the vertices its triangles use (with every stream that runs per vertex), its
triangles renumbered, and the shape keys that move any of them.
"""
from __future__ import annotations

import copy
import re

import numpy as np

#: the streams that run one value per vertex (``p<i>_<name>``); ``uv<n>`` are too
PER_VERTEX = ("co", "no", "tan", "col", "jnt", "wgt")
#: what else a part's arrays hold: its triangles, their material slots, and its shape keys
PER_PART = ("tri", "slot")
SHAPE = re.compile(r"s(\d+)_(idx|delta)")


def section_name(part: str, material: str) -> str:
    """The name of *part*'s section of *material*: the material's own name less ``MI_`` and less what it shares
    with the start of the part's name, when that is most of it (``MI_R2T1YounuoMd10011Hair`` of
    ``R2T1YounuoMd10011`` → ``R2T1YounuoMd10011_Hair``; the eyes' ``MI_R2T1ChunMd1001Eye``, a digit shorter,
    → ``R2T1ChunMd10011_Eye``); a material of its own name (``MI_5XingStar``) keeps it."""
    own = material[3:] if material.startswith("MI_") else material
    shared = 0
    while shared < min(len(own), len(part)) and own[shared] == part[shared]:
        shared += 1
    if shared * 4 >= len(part) * 3 and shared < len(own):
        own = own[shared:].lstrip("_")
    return f"{part}_{own}"


def by_material(document: dict, arrays: dict) -> tuple[dict, dict]:
    """*document* and *arrays* with each part of more than one material slot split into one part per slot used,
    in slot order; a part of one slot stays as it is.  Every other array is carried over unchanged.

    ``ValueError`` for a part array this does not know how to split: a stream left behind would be a part that
    quietly lost something.
    """
    document = copy.deepcopy(document)
    parts, out = [], {key: value for key, value in arrays.items() if not re.match(r"p\d+_", key)}
    for index, part in enumerate(document["parts"]):
        own = {key[len(f"p{index}_"):]: value for key, value in arrays.items() if key.startswith(f"p{index}_")}
        unknown = [key for key in own if key not in PER_VERTEX + PER_PART
                   and not key.startswith("uv") and not SHAPE.fullmatch(key)]
        if unknown:
            raise ValueError(f"{part['name']}: no rule to split {', '.join(sorted(unknown))}")
        slots = np.asarray(own["slot"])
        used = [int(s) for s in np.unique(slots)]
        if len(used) <= 1:
            pieces = [(part, own)]
        else:
            pieces = []
            for slot in used:
                triangles = np.asarray(own["tri"])[slots == slot]
                keep = np.unique(triangles)                       # the old numbers of the vertices it uses
                renumber = np.full(len(own["co"]), -1, dtype=np.int64)
                renumber[keep] = np.arange(len(keep))
                piece = {key: np.asarray(value)[keep] for key, value in own.items()
                         if key in PER_VERTEX or key.startswith("uv")}
                piece["tri"] = renumber[triangles]
                piece["slot"] = np.zeros(len(triangles), dtype=slots.dtype)
                shapes = []
                for number, name in enumerate(part.get("shapes", [])):
                    moved = np.asarray(own[f"s{number}_idx"])
                    inside = renumber[moved] >= 0
                    if inside.any():
                        piece[f"s{len(shapes)}_idx"] = renumber[moved[inside]]
                        piece[f"s{len(shapes)}_delta"] = np.asarray(own[f"s{number}_delta"])[inside]
                        shapes.append(name)
                material = part["materials"][slot]
                record = dict(part, name=section_name(part["name"], material), materials=[material], shapes=shapes)
                pieces.append((record, piece))
        for record, piece in pieces:
            for key, value in piece.items():
                out[f"p{len(parts)}_{key}"] = value
            parts.append(record)
    document["parts"] = parts
    return document, out
