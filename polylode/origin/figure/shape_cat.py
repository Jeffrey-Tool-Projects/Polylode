"""``list/customshape.unity3d`` — which curve each shape slider drives.

Two tab-separated tables pair a slider index with the curve entries it plays and
the channels it is allowed to write::

    categoryId <tab> curveName <tab> px py pz rx ry rz sx sy sz

The nine trailing columns are 0/1 flags.  They matter: a curve carries a full
TRS in every key, but only the flagged channels are copied onto the bone.
``cf_a_arm02`` for instance bakes a position track that stays at the origin and
a scale track that does the work — writing the untouched position anyway would
be harmless there and destructive on a bone whose rest offset is not zero.

One slider drives several curves (a bust size row plays five), and one curve can
serve several sliders, so this is a many-to-many list rather than a lookup.

``cf_customhead`` names real face bones (``cf_J_ChinLow``).  ``cf_custombody``
names ``cf_a_*`` curves that are *not* transforms in any shipped bundle — see
``shape_apply`` for how those are resolved and checked.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..prefab import text_asset_bytes

BUNDLE = "list/customshape.unity3d"
BODY_TABLE = "cf_custombody"
HEAD_TABLE = "cf_customhead"

CHANNELS = ("px", "py", "pz", "rx", "ry", "rz", "sx", "sy", "sz")


@dataclass(frozen=True)
class ShapeRow:
    category: int
    curve: str
    flags: tuple[bool, ...]
    face: bool = False          # which enum ``category`` indexes

    @property
    def writes_position(self) -> bool:
        return any(self.flags[0:3])

    @property
    def writes_rotation(self) -> bool:
        return any(self.flags[3:6])

    @property
    def writes_scale(self) -> bool:
        return any(self.flags[6:9])

    @property
    def slider(self) -> str:
        """What this row's slider is called in ``ChaFileDefine``."""
        return slider_name(self.category, self.face)

    def describe(self) -> str:
        on = [c for c, f in zip(CHANNELS, self.flags) if f]
        return f"{self.slider}:{self.curve}[{','.join(on) or '-'}]"


def _table_text(name: str) -> str | None:
    from .load_mesh import _open_bundle
    # read by name, so from the source Sideloader would hand it out of (§7.489)
    loader = _open_bundle(BUNDLE, name)
    if loader is None:
        return None
    try:
        blob = text_asset_bytes(loader.env, name)
    finally:
        loader.env = None
        loader.by_id = {}
    return None if blob is None else blob.decode("utf-8", "replace")


def parse(text: str, face: bool = False) -> list[ShapeRow]:
    rows: list[ShapeRow] = []
    for line in text.splitlines():
        cols = line.split("\t")
        if len(cols) < 2 + len(CHANNELS):
            continue
        try:
            category = int(cols[0])
            flags = tuple(bool(int(c)) for c in cols[2:2 + len(CHANNELS)])
        except ValueError:
            continue
        curve = cols[1].strip()
        if curve:
            rows.append(ShapeRow(category, curve, flags, face))
    return rows


def to_payload(rows: list[ShapeRow] | None) -> dict | None:
    """The bake's JSON of a parsed table (``bake.write_named_table``)."""
    if rows is None:
        return None
    return {"rows": [[r.category, r.curve, [int(f) for f in r.flags]] for r in rows]}


def from_payload(payload: dict | None, face: bool = False) -> list[ShapeRow] | None:
    """:func:`to_payload` back; *face* is which enum the categories index."""
    if payload is None:
        return None
    return [ShapeRow(int(c), str(curve), tuple(bool(f) for f in flags), face)
            for c, curve, flags in payload["rows"]]


_CACHE: dict[str, list[ShapeRow]] = {}


def table(name: str, *, from_bundle: bool = False) -> list[ShapeRow]:
    """The rows of one of the two tables — from the bake, else out of the bundle.

    ``from_bundle`` is the converter's read: past the bake and the cache.
    """
    face = name == HEAD_TABLE
    if not from_bundle:
        cached = _CACHE.get(name)
        if cached is not None:
            return cached
        from .. import bake
        recorded, payload = bake.named_table(bake.INSTALL, BUNDLE, name)
        if recorded:
            rows = from_payload(payload, face) or []
            _CACHE[name] = rows
            return rows
    text = _table_text(name)
    rows = parse(text, face=face) if text else []
    if not from_bundle:
        _CACHE[name] = rows
    return rows


def body_rows() -> list[ShapeRow]:
    return table(BODY_TABLE)


def head_rows() -> list[ShapeRow]:
    return table(HEAD_TABLE)


# ---------------------------------------------------------------------------
# what each slider index means
# ---------------------------------------------------------------------------
# ``ChaFileDefine.BodyShapeIdx`` / ``FaceShapeIdx``, in value order, straight
# out of ``Assembly-CSharp``'s metadata.  44 and 52 members — exactly the
# lengths of ``shapeValueBody`` / ``shapeValueFace`` — and the ``categoryId``
# column of ``cf_custombody`` / ``cf_customhead`` indexes them directly (44 and
# 52 distinct ids, 0..43 and 0..51, with no gaps).  Checked against the tables:
# id 4 ``BustSize`` lists every ``cf_a_bust*_size`` curve, id 21 ``Belly``
# lists ``cf_a_berry`` (berry = 腹), id 26 ``Hip`` lists ``cf_a_siri`` (尻).
#
# Until now the loader was table-driven without knowing what any slot *was*,
# which is fine for applying a card but makes a card↔slider editor impossible
# and leaves every diagnostic reading "category 43" instead of "ArmLow".
BODY_SHAPE_NAMES = (
    "Height", "HeadSize", "NeckW", "NeckZ",
    "BustSize", "BustY", "BustRotX", "BustX", "BustRotY", "BustSharp",
    "BustForm", "AreolaBulge", "NipWeight", "NipStand",
    "BodyShoulderW", "BodyShoulderZ", "BodyUpW", "BodyUpZ",
    "BodyLowW", "BodyLowZ", "WaistY", "Belly",
    "WaistUpW", "WaistUpZ", "WaistLowW", "WaistLowZ",
    "Hip", "HipRotX",
    "ThighUpW", "ThighUpZ", "ThighLowW", "ThighLowZ",
    "KneeLowW", "KneeLowZ", "Calf", "AnkleW", "AnkleZ",
    "ShoulderW", "ShoulderZ", "ArmUpW", "ArmUpZ",
    "ElbowW", "ElbowZ", "ArmLow",
)

FACE_SHAPE_NAMES = (
    "FaceBaseW", "FaceUpZ", "FaceUpY", "FaceUpSize", "FaceLowZ", "FaceLowW",
    "ChinLowY", "ChinLowZ", "ChinY", "ChinW", "ChinZ",
    "ChinTipY", "ChinTipZ", "ChinTipW",
    "CheekBoneW", "CheekBoneZ", "CheekW", "CheekZ", "CheekY",
    "EyebrowY", "EyebrowX", "EyebrowRotZ", "EyebrowInForm", "EyebrowOutForm",
    "EyelidsUpForm1", "EyelidsUpForm2", "EyelidsUpForm3",
    "EyelidsLowForm1", "EyelidsLowForm2", "EyelidsLowForm3",
    "EyeY", "EyeX", "EyeZ", "EyeTilt", "EyeH", "EyeW", "EyeInX", "EyeOutY",
    "NoseTipH", "NoseY", "NoseBridgeH",
    "MouthY", "MouthW", "MouthZ",
    "MouthUpForm", "MouthLowForm", "MouthCornerForm",
    "EarSize", "EarRotY", "EarRotZ", "EarUpForm", "EarLowForm",
)

# ``shapeValueBody[0]`` is the height slider, which the loader deliberately
# skips: it is the uniform ``cf_n_height`` scale and is already expressed by
# ``rig_scale`` (§7.9).  Applying both squares it.
HEIGHT_SLIDER = BODY_SHAPE_NAMES.index("Height")


def slider_name(category: int, face: bool = False) -> str:
    """``43`` → ``"ArmLow"``; the raw index when it is out of range."""
    names = FACE_SHAPE_NAMES if face else BODY_SHAPE_NAMES
    return names[category] if 0 <= category < len(names) else str(category)
