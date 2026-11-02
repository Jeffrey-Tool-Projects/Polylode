"""ABMX (``KKABMPlugin.ABMData``) — the card's own per-bone TRS overrides.

The sliders in ``shape_update`` are Illusion's; this is the *other* half of what
decides where a card's bones end up.  ABMX (KKABMX, "Advanced Bone Modifier")
is a BepInEx plugin that stores a scale / length / position / rotation override
per bone in the card's extended data, and the maker applies it on top of the
shaped rig every frame.  It is not an exotic case: on this install **973 of
1,386 cards (70.2%) carry the section and 809 (58.4%) carry a non-empty
``boneData``** — including the height bone on 178 of them, so a renderer that
skips it draws most cards at the wrong size (§7.98).

Everything here is transcribed from ``BepInEx/plugins/KKABMX.dll`` rather than
inferred from what the numbers look like (CLAUDE.md 一之二).  The three parts
that matter:

``BoneModifier.Apply``
    ::

        if (mod.HasScale())     bone.localScale = _sclBaseline ⊙ mod.ScaleModifier
        if (mod.HasRotation())  bone.localRotation = _rotBaseline * Euler(mod.RotationModifier)
        if (mod.HasLength())    bone.localPosition = _posBaseline * mod.LengthModifier
                                                    + mod.PositionModifier
        else if (mod.HasPosition())
                                bone.localPosition = _posBaseline + mod.PositionModifier

    ``HasX`` is "not the identity" per channel — scale ≠ (1,1,1), length ≠ 1,
    position ≠ (0,0,0), rotation ≠ (0,0,0) — so a modifier that stores the
    default for a channel leaves that channel alone.  That is why a stored
    scale of ``(0,0,0)`` is a real instruction (collapse the bone) and not a
    missing value: the plugin's own "unset" is ``(1,1,1)``.

``<CollectBaselineCo>d__63``
    The baseline is **not** the prefab rest pose.  The coroutine sets
    ``updateShapeFace``/``updateShapeBody``, calls ``ChaControl.LateUpdateForce``
    and only then reads each bone's ``localScale`` / ``localPosition`` /
    ``localRotation``.  So ABMX multiplies onto the *shaped* rig, which is why
    this runs at the end of :func:`figure.dress.apply_card_shape` and not
    before it.

``BoneModifier.GetModifier`` / ``IsCoordinateSpecific``
    ``CoordinateModifiers.Length > 1`` means the card stores one entry per
    outfit; otherwise entry 0 applies to all of them.  1,302 of this install's
    18,510 rows are the seven-entry form.

Two conversions apply to everything read here, the same two ``shape_run``
already does, because our world is Illusion's mirrored through the X plane
(``prefab._trs`` negates the local X and nothing else):

* a **position** offset has its X negated and is multiplied by ``unit``
  (the graft's ``rig_scale``: :func:`rig.graft_full_skeleton` bakes that factor
  into every local translation, where the game instead carries it as
  ``cf_n_height``'s scale, so an offset authored in Illusion units has to be
  brought into ours);
* a **rotation** is mirrored by negating the quaternion's y and z.

Scale needs neither — it is dimensionless, and a per-axis scale is unchanged by
a reflection through an axis plane.

**Known difference from the game.** KKABMX re-applies in every ``LateUpdate``,
so a bone carrying a rotation or position modifier is *pinned* to
``baseline ∘ modifier`` and an animation cannot move it.  Here the override is
folded into the skeleton's rest TRS and the animation composes on top, which is
identical for a still frame or the bind pose and diverges only for an animated
bone that also carries a rotation/position modifier.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from ...core import math3d as m3
from ...core import swallowed

#: ``PluginData`` key, and the two save versions ``BoneController.ReadModifiers``
#: accepts.  Version 1 is the legacy comma-separated text
#: (``OldDataConverter.MigrateOldStringData``), 2 the MessagePack blob; anything
#: else the plugin refuses with ``NotSupportedException``, and so do we.
EXT_KEY = "KKABMPlugin.ABMData"

#: MessagePack-CSharp writes an LZ4-compressed payload as ext type 99.
_LZ4_EXT = 99

ONE = (1.0, 1.0, 1.0)
ZERO = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoneModifierData:
    """``KKABMX.Core.BoneModifierData`` — four ``[Key]`` fields, in order.

    Rows shorter than four are normal, not corrupt: 9,134 of this install's
    26,350 stored modifiers have only two entries, written by a plugin version
    whose ``BoneModifierData`` was scale + length.  A missing field is the
    field's default, which is what MessagePack-CSharp does with a short array.
    """
    scale: tuple = ONE
    length: float = 1.0
    position: tuple = ZERO
    rotation: tuple = ZERO

    def has_scale(self) -> bool:
        return tuple(self.scale) != ONE

    def has_length(self) -> bool:
        return float(self.length) != 1.0

    def has_position(self) -> bool:
        return tuple(self.position) != ZERO

    def has_rotation(self) -> bool:
        return tuple(self.rotation) != ZERO

    def is_empty(self) -> bool:
        return not (self.has_scale() or self.has_length()
                    or self.has_position() or self.has_rotation())


@dataclass
class ModReport:
    """What the pass did, in the shape ``RunReport`` uses.

    ``missing`` and ``uncarried`` are deliberately two lists.  A name Illusion's
    own rig does not have at all is a failure to resolve and belongs in
    ``DressReport.missing``; a name the rig *has* but this project does not put
    on the character armature is a known exclusion, and calling it missing
    would bury the first kind under the second.  On this install the second
    kind is 3 rows across 3 cards (``cf_ik_hips``, ``p_cf_body_bone``).  The
    ``cf_hit_*`` sway colliders, which most cards' rows name (243 cards), are
    on the armature since §7.540, so a row scaling one reaches the collider
    the live solve places on it (§7.545).
    """
    version: int = 0
    bones: int = 0              # bones actually written
    rows: int = 0               # rows the card stored
    empty: int = 0              # rows whose modifier is the identity
    channels: int = 0           # scale/length/position/rotation writes
    missing: list = field(default_factory=list)    # not in Illusion's rig either
    uncarried: list = field(default_factory=list)  # in the rig, not on ours
    reason: str = ""

    def describe(self) -> str:
        if self.reason:
            return f"ABMX: {self.reason}"
        out = (f"ABMX v{self.version}: {self.rows} 列、{self.channels} 個通道到 "
               f"{self.bones} 根骨頭")
        if self.empty:
            out += f"（{self.empty} 列是預設值）"
        if self.uncarried:
            out += f"，{len(self.uncarried)} 列指到骨架不帶的節點"
        if self.missing:
            out += f"，{len(self.missing)} 列的骨頭原作也沒有"
        return out


def combine(mods) -> BoneModifierData:
    """``BoneModifier.CombineModifiers``: scale and length multiply, position
    and rotation add — how KKABMX folds a plugin's ``BoneEffect`` onto the
    card's own modifier for the same bone."""
    scale, length = np.ones(3), 1.0
    pos, rot = np.zeros(3), np.zeros(3)
    for m in mods:
        scale = scale * np.asarray(m.scale, dtype=np.float64)
        length *= float(m.length)
        pos = pos + np.asarray(m.position, dtype=np.float64)
        rot = rot + np.asarray(m.rotation, dtype=np.float64)
    return BoneModifierData(scale=tuple(scale), length=length,
                            position=tuple(pos), rotation=tuple(rot))


def _vec3(v, default) -> tuple:
    try:
        return (float(v[0]), float(v[1]), float(v[2]))
    except (TypeError, IndexError, ValueError):
        return default


def _modifier(row) -> BoneModifierData | None:
    if row is None:
        return None
    if not isinstance(row, (list, tuple)):
        return None
    return BoneModifierData(
        scale=_vec3(row[0], ONE) if len(row) > 0 else ONE,
        length=float(row[1]) if len(row) > 1 and row[1] is not None else 1.0,
        position=_vec3(row[2], ZERO) if len(row) > 2 else ZERO,
        rotation=_vec3(row[3], ZERO) if len(row) > 3 else ZERO)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _msgpack_lz4(blob: bytes):
    """Unwrap MessagePack-CSharp's ``LZ4Block`` envelope, then unpack.

    The envelope is a msgpack ext of type 99 whose body is ``<msgpack int
    uncompressed length><lz4 block>``.  A blob that is not that envelope is
    plain MessagePack and is unpacked as it stands.
    """
    from ...chara.cardimport import _unpack
    b = bytes(blob)
    if b[:1] == b"\xc9":                                   # ext32
        n = struct.unpack_from(">I", b, 1)[0]
        kind, body = b[5], b[6:6 + n]
    elif b[:1] == b"\xc8":                                 # ext16
        n = struct.unpack_from(">H", b, 1)[0]
        kind, body = b[3], b[4:4 + n]
    elif b[:1] == b"\xc7":                                 # ext8
        n = b[1]
        kind, body = b[2], b[3:3 + n]
    else:
        return _unpack(b)
    if kind != _LZ4_EXT:
        raise ValueError(f"未知的 MessagePack ext 型別 {kind}")
    head = body[0]
    if head < 0x80:
        size, off = head, 1
    elif head == 0xcc:
        size, off = body[1], 2
    elif head == 0xcd:
        size, off = struct.unpack_from(">H", body, 1)[0], 3
    elif head == 0xce:
        size, off = struct.unpack_from(">I", body, 1)[0], 5
    elif head == 0xd2:
        size, off = struct.unpack_from(">i", body, 1)[0], 5
    else:
        raise ValueError(f"未知的長度前綴 {head:#x}")
    import lz4.block
    return _unpack(lz4.block.decompress(body[off:], uncompressed_size=size))


def _trimmed_name(name: str) -> str:
    """``<DeserializeToModifiers>g__GetTrimmedName`` — drop a ``__N`` suffix.

    The legacy format writes one line per coordinate and distinguishes them by
    appending ``__0`` … ``__6`` to the bone name, so the group key is the name
    without it.  The plugin's test is on the *second and third* characters from
    the end both being ``_``, which is why a real bone name ending in a single
    underscore is left alone.
    """
    if len(name) >= 3 and name[-2] == "_" and name[-3] == "_":
        return name[:-3]
    return name


def _read_legacy(text: str) -> dict:
    """``OldDataConverter.DeserializeToModifiers``.

    One line per bone per coordinate, comma separated, at least six fields:
    ``[?, name(__N), ?, scale.x, scale.y, scale.z, length?]``.  Only scale and
    length exist in this format — position and rotation came later — so those
    two stay at their defaults, which is exactly what the plugin does.
    """
    groups: dict[str, list] = {}
    for line in str(text).split("\n"):
        line = line.strip()
        if not line:
            continue
        f = line.split(",")
        if len(f) < 6:
            continue
        try:
            scale = (float(f[3]), float(f[4]), float(f[5]))
            length = float(f[6]) if len(f) > 6 else 1.0
        except ValueError:
            continue
        groups.setdefault(_trimmed_name(f[1]), []).append((f[1], scale, length))
    out = {}
    for name, rows in groups.items():
        rows.sort(key=lambda r: r[0])       # OrderBy(x => x.splitValues[1])
        out[name] = [BoneModifierData(scale=s, length=ln) for _n, s, ln in rows]
    return out


def read_modifiers(card) -> tuple[dict, int, str]:
    """``BoneController.ReadModifiers`` — ``{bone: [modifier per coordinate]}``.

    Returns ``(modifiers, version, reason)``; *reason* is non-empty when the
    card carries the section but it could not be read, and the caller reports
    that rather than silently continuing with an unshaped rig (§一之一).
    """
    try:
        ext = card.sections.get("KKEx") or {}
    except AttributeError:
        return {}, 0, ""
    data = ext.get(EXT_KEY)
    if data is None:
        return {}, 0, ""
    if not isinstance(data, (list, tuple)) or len(data) < 2:
        return {}, 0, "卡片的 ABMData 不是 (version, data) 這個形狀"
    version = int(data[0]) if isinstance(data[0], int) else 0
    blob = (data[1] or {}).get("boneData")
    if not blob:
        return {}, version, ""
    if version == 1:
        if isinstance(blob, (bytes, bytearray)):
            blob = bytes(blob).decode("utf-8", "replace")
        return _read_legacy(blob), version, ""
    if version != 2:
        # ``ReadModifiers`` throws NotSupportedException here.  Say so rather
        # than guessing at a layout we have not read.
        return {}, version, f"不支援的 ABMData 版本 {version}"
    try:
        rows = _msgpack_lz4(blob)
    except Exception as exc:                             # noqa: BLE001
        return {}, version, f"ABMData 解不開（{exc}）"
    out: dict = {}
    for row in rows or []:
        # ``SerializedBoneModifier`` is (BoneName, CoordinateModifiers,
        # BoneLocation).  Every one of this install's 18,510 rows is the
        # two-field form, i.e. BoneLocation 0 = Unknown, which the plugin
        # resolves to BodyTop — the character's own armature, which is the only
        # place this project has bones by name anyway.  A row that does name a
        # location is skipped rather than applied to the body.
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        if len(row) > 2 and int(row[2] or 0) not in (0, 1):
            continue
        mods = [_modifier(m) for m in (row[1] or [])]
        if mods:
            out[str(row[0])] = mods
    return out, version, ""


# ---------------------------------------------------------------------------
# applying
# ---------------------------------------------------------------------------
def _baseline(skeleton) -> dict:
    """``BoneModifier._sclBaseline`` / ``_posBaseline`` / ``_rotBaseline``.

    Cached on the skeleton because the plugin caches it too (``_hasBaseline``):
    a caller that re-runs the shape pass on the same graft must not read an
    already-modified bone back as its own baseline, which would compound the
    modifier every time.  A fresh graft builds new ``Bone`` objects, so the
    cache dies with the rig it was collected from.
    """
    got = getattr(skeleton, "_abmx_baseline", None)
    if got is None:
        got = {}
        setattr(skeleton, "_abmx_baseline", got)
    return got


def clear_baseline(skeleton) -> None:
    """Forget the collected baseline — for a caller that re-shapes the rig."""
    if hasattr(skeleton, "_abmx_baseline"):
        delattr(skeleton, "_abmx_baseline")


def apply_bone_modifiers(skeleton, card, unit: float = 1.0, *,
                         coordinate: int = 0, report=None,
                         effects: dict | None = None) -> ModReport:
    """Apply a card's ABMX overrides to *skeleton*, after the shape pass.

    *effects* is ``{bone: [BoneModifierData, …]}`` from other plugins'
    ``BoneEffect``s (``BoneController.ApplyEffects``): each is combined onto
    that bone's card modifier (:func:`combine`), and a bone the card leaves
    alone gets one anyway (``GetOrAddModifier``).
    """
    mods, version, reason = read_modifiers(card)
    out = ModReport(version=version, reason=reason)
    effects = {k: list(v) for k, v in (effects or {}).items() if v}
    if reason:
        # The card carries the section and we could not read it: that is a
        # character shaped differently from the one the card describes, so it
        # is reported rather than passed over (§一之一).
        if report is not None:
            report.missing.append(f"ABMX: {reason}")
        mods = {}
        if not effects:
            return out
    if not mods and not effects:
        return out
    mods = dict(mods)
    for name in effects:
        mods.setdefault(name, [])
    base = _baseline(skeleton)
    try:
        from ..rig import rig as _rig
        in_reference = set(_rig().matrices)
    except Exception as exc:                             # noqa: BLE001
        swallowed.note("origin.figure.bonemod.apply_bone_modifiers: from ..rig import rig as _rig", exc)
        in_reference = set()
    missing: list = []
    uncarried: list = []
    for name, coords in mods.items():
        out.rows += bool(coords)
        # ``GetModifier``: one entry applies to every coordinate; more than one
        # is per-outfit, and an index past the end has nothing to say.
        if len(coords) == 1:
            mod = coords[0]
        elif coordinate < len(coords):
            mod = coords[coordinate]
        else:
            mod = None
        if name in effects:
            mod = combine([mod or BoneModifierData()] + effects[name])
        if mod is None or mod.is_empty():
            out.empty += bool(coords)       # a card row, not an idle effect
            continue
        if not skeleton.has(name):
            (uncarried if name in in_reference else missing).append(name)
            continue
        bone = skeleton.bones[skeleton.id(name)]
        if name not in base:
            base[name] = (bone.local_pos.copy(), bone.local_rot.copy(),
                          bone.local_scale.copy())
        b_pos, b_rot, b_scl = base[name]
        wrote = 0
        if mod.has_scale():
            bone.local_scale = b_scl * np.asarray(mod.scale, dtype=np.float64)
            wrote += 1
        if mod.has_rotation():
            q = m3.quat_from_euler(*np.radians(mod.rotation), order="YXZ")
            bone.local_rot = m3.quat_normalize(
                m3.quat_mul(b_rot, np.array([q[0], -q[1], -q[2], q[3]])))
            wrote += 1
        offset = np.array([-mod.position[0], mod.position[1],
                           mod.position[2]]) * float(unit)
        if mod.has_length():
            bone.local_pos = b_pos * float(mod.length) + offset
            wrote += 1
        elif mod.has_position():
            bone.local_pos = b_pos + offset
            wrote += 1
        if wrote:
            out.bones += 1
            out.channels += wrote
    out.missing = sorted(set(missing))
    out.uncarried = sorted(set(uncarried))
    if out.channels:
        skeleton.finalize()
    if report is not None and out.missing:
        report.missing.append(
            f"ABMX: 卡片指名了 {len(out.missing)} 根原作骨架也沒有的骨頭"
            f"（{', '.join(out.missing[:4])}"
            f"{'…' if len(out.missing) > 4 else ''}）")
    return out
