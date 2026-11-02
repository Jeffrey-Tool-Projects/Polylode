"""Koikatsu H AnimationClip → IslandLife ``AnimationClip``.

Origin packs under ``abdata/h/anim/{female,male}`` store motion as **non-legacy**
clips whose payload is a ``ClipMuscleConstant`` (StreamedClip + DenseClip +
ConstantClip).  Bindings are **Transform paths** (CRC32 of
``cf_j_root/cf_n_height/...``), not pure humanoid muscle indices — 329/329
paths on the reference armature resolve that way.

Conversion (import / bake time only):

1. Resolve path CRC → bone name via ``oo_base`` / ``p_cf_body_bone`` hierarchy.
2. Expand each binding to float channels (pos 3 / rot 4 / scale 3).
3. Sample Streamed → Dense → Constant in that curve-index order (AssetStudio).
4. Emit absolute local TRS (Illusion LH, then mirror into our RH convention).
5. Serialise to ``assets_baked/h_anim/`` as our JSON clip format.

Runtime only reads the baked JSON (or converts on the fly if bake is empty).
Nothing from Illusion is redistributed.
"""
from __future__ import annotations

import json
import re
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..core.i18n import tr
from ..core.paths import write_json_atomic
from .install import Install, configured

# ---------------------------------------------------------------------------
# Free H position → Origin pack id + motion tokens
# ---------------------------------------------------------------------------
# Pack filenames under abdata/h/anim/{female,male}/ are ``AA_BB_CC.unity3d``.
# Clip bake names are ``origin_{f|m}_{AA_BB_CC}_{stem}``.
#
# First byte is the *mode family* (from clip contents, not English names):
#   00_*  aibu  — Idle / caress only, no WLoop
#   01_*  houshi — service: Oral_Idle, IN/OUT, *no* Insert  → oral / paizuri
#   02_*  sonyu  — has Insert + WLoop  → actual penetration poses
#   04/07/09_*   more sonyu families
# Mapping 正常位→01 was wrong: Free H "missionary" played houshi (乳交).
#
# Within sonyu, packs differ by posture (female L_WLoop hips @ t=0):
#   02_06 / 02_12 / 02_13 — pitch ≈ −77..−81°, hips low  → on-back
#   02_20 / 02_00         — pitch ≈ 0 / odd Y-rot, hips high → upright
#   04_00 / 07_20         — bent torso, both sexes have packs → rear variants
# Prefer packs present for *both* female and male bakes.
_POS_PACKS = {
    "missionary": ("02_06_00", "02_12_00", "02_13_00", "04_00_00", "09_54_00"),
    "cowgirl": ("02_20_00", "02_00_00", "07_20_00"),
    "doggy": ("04_00_00", "07_20_00", "02_00_00"),
}
# Houshi / caress — never pick these for the three Free H sex positions.
_HOUSHI_OR_AIBU = frozenset({
    "00_00_00", "01_00_00", "01_06_00", "01_13_00", "01_20_00",
    "06_20_00", "08_54_00",
})
# Motion-strength / loop tokens preferred inside a pack (not gender).
_POS_MOTION = {
    "missionary": ("WLoop", "SLoop", "OLoop", "Insert", "IN_Loop"),
    "cowgirl": ("WLoop", "SLoop", "OLoop", "Insert"),
    "doggy": ("WLoop", "SLoop", "OLoop", "Insert"),
}

# Bake root (our format only — not Illusion binaries).  Taken from
# ``bake.bake_dir()`` rather than spelled out again, so the h clips follow the
# same rule as every other baked item: they live where the *user* baked them,
# never inside the release (§7.145).  Resolved once at import — the env
# override is read per process (``bake_verify`` re-launches), not per call.
from . import bake as _bake
from ..core import swallowed

BAKE_H_DIR = _bake.bake_dir() / _bake.H_ANIM_DIR   # one owner for the name
BAKE_MANIFEST = BAKE_H_DIR / "manifest.json"
#: What a baked clip's name starts with, and what an earlier version named it with (read as this one,
#: :func:`migrate_baked_names`)
CLIP_PREFIX, LEGACY_CLIP_PREFIX = "origin_", "kk_"
#: An H menu row's key, as :func:`list_menu_entries` makes it, and as an earlier version did
ROW_PREFIX, LEGACY_ROW_PREFIX = "originact_", "kkact_"
_MIGRATED: set = set()


def current_name(name):
    """*name* (a clip's or a row's) as this version names it."""
    if isinstance(name, str):
        if name.startswith(LEGACY_ROW_PREFIX):
            return ROW_PREFIX + name[len(LEGACY_ROW_PREFIX):]
        if name.startswith(LEGACY_CLIP_PREFIX):
            return CLIP_PREFIX + name[len(LEGACY_CLIP_PREFIX):]
    return name


def migrate_baked_names(root=None) -> int:
    """Rename what an earlier version baked under :data:`LEGACY_CLIP_PREFIX` in place; how many files moved.

    The clips' files and their names in ``manifest.json`` (and the studio
    clips' ``studio/manifest.json``) take :data:`CLIP_PREFIX`.  Done once per
    folder and process; a manifest with no old name is left as it is.
    """
    import os
    root = Path(root) if root is not None else BAKE_H_DIR
    if root in _MIGRATED:
        return 0
    moved = 0
    for manifest in (root / "manifest.json", root / "studio" / "manifest.json"):
        if not manifest.is_file():
            continue
        text = manifest.read_text(encoding="utf-8")
        if '"' + LEGACY_CLIP_PREFIX not in text and "/" + LEGACY_CLIP_PREFIX not in text:
            continue
        base = manifest.parent

        def renamed(value):
            nonlocal moved
            if isinstance(value, dict):
                out = {}
                for key, item in value.items():
                    if key == "file" and isinstance(item, str):
                        head, _sep, leaf = item.rpartition("/")
                        new_leaf = current_name(leaf)
                        if new_leaf != leaf:
                            new = f"{head}/{new_leaf}" if head else new_leaf
                            if (base / item).is_file() and not (base / new).exists():
                                os.replace(base / item, base / new)
                                moved += 1
                            item = new
                        out[key] = item
                    else:
                        out[current_name(key)] = renamed(item)
                return out
            if isinstance(value, list):
                return [renamed(item) for item in value]
            return current_name(value)

        write_json_atomic(manifest, renamed(json.loads(text)), indent=2)
    _MIGRATED.add(root)
    return moved

# 1 → clip names are ``origin_{sex}_{pack}_{stem}`` and nothing else is recorded.
# 2 → every entry carries ``pack`` / ``controller`` / ``stage``.  Those are not
#     recoverable from the name: a controller name (``khs_f_n18``) and a state
#     name (``L_M_OUT_A``) both contain ``_``, so splitting the name cannot say
#     where one ends and the other begins.  Read the fields, do not re-parse.
MANIFEST_VERSION = 2

# Bones we keep even if the clip has hundreds of helper nodes (fingers stay).
# ``cm_J_dan``: the man's shaft and scrotum.  The clips animate them, and on a row without an aim
# list (the handjob rows: ``Lookat_dan.LoadList`` releases and returns) the clip is the only thing
# that poses the shaft; left out, it hung at rest, 30–130 mm from where the game draws it (§7.1182).
_KEEP_PREFIXES = (
    "cf_j_", "cf_s_", "cf_d_", "cf_n_", "cf_J_", "cf_pv_", "a_n_", "cm_J_dan",
)


def _kept_bone(bone: str, sex: str = "") -> bool:
    """Whether a clip curve on *bone* is converted, in a clip of a body of *sex* (``"f"`` / ``"m"``, ``""`` either).

    Beyond the prefixes, the ``k_f_*`` nulls something in the H scene reads,
    each by the name its list gives: the ones the ``dan_*`` lists aim at
    (``lookat_dan.null_names``), the ``MotionIK`` targets and chains
    (:func:`motion_ik_targets`), and the parents of the row's items and of the
    caress hands on the body they are looked up under (:func:`attach_nulls`).
    The clips animate them and each reader takes where the Animator put them,
    so a clip without those curves leaves the null at the rest pose — 13.4 mm
    off for the median clip of female ``02_00_00`` (§7.511), 74 mm for the toy
    in her hand on 1,49.  The other ``k_f_*`` nulls have no reader in the scene
    (no list names them, no string in ``Assembly-CSharp`` but ``ChaReference``'s,
    whose keys only Studio's ``Data.GetParent`` asks for, and no component on
    the body holds one) and are left out.
    """
    from .rig import game_name, scope_of
    # a bone under an attached prefab (``p_tang2:cf_j_tang_01``) by its game name
    own = game_name(bone) if scope_of(bone) else bone
    if any(own.startswith(p) for p in _KEEP_PREFIXES):
        return True
    from .lookat_dan import null_names
    return bone in null_names() or bone in motion_ik_targets() or bone in attach_nulls(sex)


#: ``hlist.ITEM_BODY_ROLES``' roles by the sex of the body they name
_ROLE_SEX = {"male": "m", "female": "f", "female1": "f"}


def attach_nulls(sex: str = "") -> frozenset:
    """The body nulls the H scene parents something to on a body of *sex* (``"f"`` / ``"m"``; ``""`` both).

    From ``attach.json`` (``hlist.bake_attach``): the item lists the rows name
    (``ItemObject.LoadItem``, under the body its ``numToWhomParent`` names:
    ``k_f_handR_00`` of the woman holds the pen of いたずら愛撫 and the toy of
    立ちオナホ手コキ) and the caress areas' hand parents (``HandCtrl.Init``,
    under the woman: ``k_f_mune03L_02`` …).  Empty before it is baked.
    """
    global _ATTACH_NULLS
    if _ATTACH_NULLS is None:
        from .hlist import load_attach
        d = load_attach(ATTACH_FILE)
        out: dict[str, set] = {"f": set(), "m": set()}
        for pairs in (d.get("item_parents") or {}).values():
            for role, name in pairs:
                if role in _ROLE_SEX:
                    out[_ROLE_SEX[role]].add(name)
        for names in (d.get("aibu_parents") or {}).values():
            out["f"].update(names)
        _ATTACH_NULLS = {k: frozenset(v) for k, v in out.items()}
    if sex in _ATTACH_NULLS:
        return _ATTACH_NULLS[sex]
    return _ATTACH_NULLS["f"] | _ATTACH_NULLS["m"]


_ATTACH_NULLS: dict | None = None


def motion_ik_targets() -> frozenset:
    """Every bone a baked ``MotionIK`` part names as its target or chain (D14).

    The IK pulls toward where the animation put that bone, so the clip bake
    keeps its curves — most are ``k_f_*`` nulls on the partner, which the
    clips animate.  Empty before ``motion_ik.json`` is baked.
    """
    global _MOTION_IK_TARGETS
    if _MOTION_IK_TARGETS is None:
        out: set = set()
        try:
            files = json.loads(MOTION_IK_FILE.read_text(encoding="utf-8")).get("files") or {}
        except (OSError, ValueError):
            files = {}
        for states in files.values():
            for st in states:
                for part in (st.get("parts") or {}).values():
                    out.update(n for n in (part.get("target"), part.get("chain")) if n)
        _MOTION_IK_TARGETS = frozenset(out)
    return _MOTION_IK_TARGETS


_MOTION_IK_TARGETS: frozenset | None = None


def _install() -> Install | None:
    inst = configured()
    if inst is not None and inst.valid:
        return inst
    return None


# ===========================================================================
# Path CRC map (Animator-relative paths start at cf_j_root)
# ===========================================================================
def _unity_path_crc(path: str) -> int:
    return zlib.crc32(path.encode("utf-8")) & 0xFFFFFFFF


@lru_cache(maxsize=4)
def bone_path_crc_map(root: str = "cf_j_root") -> dict[int, str]:
    """``{crc32(path): bone_name}`` for the body armature under *root*.

    A clip's binding paths are relative to the ``Animator`` that plays it: the
    body's sits on ``cf_j_root``; Studio's hand animations have their own
    ``Animator`` further down (``HandAnimeCtrl``, E1d), so their paths start there.
    """
    inst = _install()
    if inst is None:
        return {}
    try:
        from .rig import ReferenceRig, game_name, scope_of
        rig = ReferenceRig().build(inst)
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] rig build failed: {exc}")
        return {}
    if not rig.built:
        return {}
    parents = rig.parents
    out: dict[int, str] = {}

    def chain_to_root(name: str) -> list[str] | None:
        parts: list[str] = []
        cur = name
        seen: set[str] = set()
        while cur and cur not in seen:
            if cur == root and root != "cf_j_root":
                return list(reversed(parts)) or None
            parts.append(cur)
            if cur == "cf_j_root":
                return list(reversed(parts)) if root == "cf_j_root" else None
            seen.add(cur)
            cur = parents.get(cur) or ""
        return None

    for name in parents:
        ch = chain_to_root(name)
        if not ch:
            continue
        # The path is the game's names; the bone it resolves to is our key.  The
        # ``p_tang2`` tongue's bones share the body's names, so its paths
        # (``…/cf_s_head/p_tang2/cf_j_tang_01``) only reach its own bones keyed
        # under that root (``rig.ATTACHED_ROOTS``, PROGRESS #30).
        path = "/".join(game_name(part) if scope_of(part) else part for part in ch)
        out[_unity_path_crc(path)] = name
    # empty path / root alone
    out[_unity_path_crc(root)] = root
    out[0] = root
    return out


# ===========================================================================
# Streamed / Dense / Constant sampling (AssetStudio layout)
# ===========================================================================
def _read_streamed_frames(data_u32) -> list[tuple[float, list[tuple[int, tuple[float, float, float, float]]]]]:
    """Parse StreamedClip uint buffer → ``[(time, [(curve_index, (a, b, c, d)), ...])]``.

    Each key starts a cubic segment of its curve that runs to the curve's next
    key: ``v(t) = ((a·dt + b)·dt + c)·dt + d`` with ``dt = t − time``
    (:func:`_streamed_segments`).  Leading frame is often ``time = -inf``
    (template); still returned.
    """
    raw = np.asarray(data_u32, dtype=np.uint32)
    if raw.size == 0:
        return []
    buf = raw.tobytes()
    frames: list[tuple[float, list[tuple[int, float]]]] = []
    off = 0
    n = len(buf)
    while off + 8 <= n:
        time = struct.unpack_from("<f", buf, off)[0]
        off += 4
        num_keys = struct.unpack_from("<i", buf, off)[0]
        off += 4
        if num_keys < 0 or num_keys > 100_000:
            break
        keys: list[tuple[int, tuple[float, float, float, float]]] = []
        for _ in range(num_keys):
            if off + 20 > n:
                return frames
            index = struct.unpack_from("<i", buf, off)[0]
            coeff = struct.unpack_from("<4f", buf, off + 4)
            off += 20
            keys.append((int(index), tuple(float(c) for c in coeff)))
        frames.append((float(time), keys))
    return frames


def _binding_dim(attribute: int, type_id: int) -> int:
    # typeID 4 = Transform (Unity ClassID)
    if int(type_id) == 4:
        if attribute in (1, 3, 4):  # pos / scale / euler
            return 3
        if attribute == 2:  # quaternion
            return 4
    return 1


def _streamed_segments(frames, n_curves: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per curve, ``(key times, (K, 4) cubic coefficients)`` of a streamed clip.

    A streamed key is not a sample: it is the start of a cubic segment,
    ``((a·dt + b)·dt + c)·dt + d`` from its time to the next key's.  Keyframe
    reduction leaves few keys on a smooth curve, so drawing straight lines
    between the ``d`` values cuts the curve; on the male ``khs_m_n06`` loops
    that tilted his pelvis 0.9° against the game (§7.1207).  A curve the first
    timed frame does not key starts from its template (``-inf``) key's value.
    """
    times: list[list[float]] = [[] for _ in range(n_curves)]
    coeffs: list[list[tuple]] = [[] for _ in range(n_curves)]
    template: dict[int, tuple] = {}
    first = True
    for t, keys in frames:
        if not np.isfinite(t):
            if t < 0:
                template.update((i, c) for i, c in keys if 0 <= i < n_curves)
            continue
        if first:
            first = False
            keyed = {i for i, _c in keys}
            for i, c in template.items():
                if i not in keyed:
                    times[i].append(t)
                    coeffs[i].append((0.0, 0.0, 0.0, c[3]))
        for i, c in keys:
            if 0 <= i < n_curves:
                times[i].append(t)
                coeffs[i].append(c)
    return [(np.asarray(times[i], np.float64), np.asarray(coeffs[i], np.float64).reshape(-1, 4))
            for i in range(n_curves)]


def _eval_segments(key_times: np.ndarray, coeffs: np.ndarray, times: np.ndarray) -> np.ndarray:
    """One streamed curve at *times*: the segment that started last, as its cubic.

    Before the first key the curve holds that key's start value.
    """
    seg = np.clip(np.searchsorted(key_times, times, side="right") - 1, 0, len(key_times) - 1)
    dt = np.maximum(times - key_times[seg], 0.0)
    a, b, c, d = coeffs[seg].T
    return ((a * dt + b) * dt + c) * dt + d


def _sample_float_curves(dense, streamed, const_data, times: np.ndarray) -> np.ndarray:
    """Return ``(T, n_curves)`` float samples for the muscle clip payload."""
    n_streamed = int(getattr(streamed, "curveCount", 0) or 0)
    n_dense = int(getattr(dense, "m_CurveCount", 0) or 0)
    const = np.asarray(getattr(const_data, "data", None) or [], dtype=np.float64)
    n_const = int(const.size)
    n_curves = n_streamed + n_dense + n_const
    T = len(times)
    out = np.zeros((T, n_curves), dtype=np.float64)

    # --- streamed: each key's cubic, up to the curve's next key ----------
    frames = _read_streamed_frames(getattr(streamed, "data", None) or [])
    if n_streamed > 0 and frames:
        for ci, (kt, coeffs) in enumerate(_streamed_segments(frames, n_streamed)):
            if kt.size:
                out[:, ci] = _eval_segments(kt, coeffs, times)

    # --- dense ------------------------------------------------------------
    if n_dense > 0:
        rate = float(getattr(dense, "m_SampleRate", 30) or 30)
        begin = float(getattr(dense, "m_BeginTime", 0) or 0)
        n_frames = int(getattr(dense, "m_FrameCount", 0) or 0)
        arr = np.asarray(getattr(dense, "m_SampleArray", []) or [], dtype=np.float64)
        if n_frames >= 1 and arr.size >= n_frames * n_dense:
            samples = arr[: n_frames * n_dense].reshape(n_frames, n_dense)
            # frame i at time begin + i/rate
            frame_times = begin + np.arange(n_frames, dtype=np.float64) / max(rate, 1e-6)
            for j in range(n_dense):
                out[:, n_streamed + j] = np.interp(
                    times, frame_times, samples[:, j],
                    left=samples[0, j], right=samples[-1, j])

    # --- constant ---------------------------------------------------------
    if n_const > 0:
        base = n_streamed + n_dense
        n = min(n_const, n_curves - base)
        out[:, base:base + n] = const[:n]

    return out


def _mirror_pos(p: np.ndarray) -> np.ndarray:
    """Illusion LH → our RH (same as ``rig._trs``)."""
    return np.array([-float(p[0]), float(p[1]), float(p[2])], dtype=np.float64)


def _mirror_quat(q: np.ndarray) -> np.ndarray:
    # (x, -y, -z, w) then normalise
    out = np.array([float(q[0]), -float(q[1]), -float(q[2]), float(q[3])],
                   dtype=np.float64)
    n = float(np.linalg.norm(out))
    return out / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def _transform_buckets(gens, floats: np.ndarray, path_map: dict, sex: str = "", *,
                       filtered: bool = True) -> dict[str, dict]:
    """``{bone: {"pos": (T,3), "rot": (T,4), "scale": (T,3)}}`` of a muscle clip's Transform bindings.

    *gens* are the clip's ``genericBindings`` in curve order, *floats* the
    ``(T, n_curves)`` samples (:func:`_sample_float_curves`), *path_map* the
    binding path hashes of the playing ``Animator`` (:func:`bone_path_crc_map`).
    Positions and rotations come out in this project's mirrored space; a bone
    the map does not reach, or that :func:`_kept_bone` drops for a body of
    *sex*, is skipped.  *filtered* False keeps every node the map reaches (an
    item's own Animator, whose nodes are not body bones).
    """
    from ..core import math3d as m3
    curve_i = 0
    buckets: dict[str, dict] = {}
    T = len(floats)
    n_curves = floats.shape[1]
    for b in gens:
        attr = int(getattr(b, "attribute", 0) or 0)
        type_id = int(getattr(b, "typeID", 0) or 0)
        path_h = int(getattr(b, "path", 0) or 0)
        dim = _binding_dim(attr, type_id)
        if curve_i + dim > n_curves:
            break
        bone = path_map.get(path_h)
        if bone is None or (filtered and not _kept_bone(bone, sex)):
            curve_i += dim
            continue
        block = floats[:, curve_i:curve_i + dim]
        curve_i += dim
        slot = buckets.setdefault(bone, {})
        if type_id == 4 and attr == 1 and dim == 3:  # position
            slot["pos"] = np.array([_mirror_pos(block[i]) for i in range(T)], dtype=np.float64)
        elif type_id == 4 and attr == 2 and dim == 4:  # quaternion
            rot = np.array([_mirror_quat(block[i]) for i in range(T)], dtype=np.float64)
            # fix double-cover
            for i in range(1, T):
                if float(np.dot(rot[i], rot[i - 1])) < 0:
                    rot[i] = -rot[i]
            slot["rot"] = rot
        elif type_id == 4 and attr == 3 and dim == 3:  # scale
            slot["scale"] = block.copy()
        elif type_id == 4 and attr == 4 and dim == 3:  # euler degrees → quat
            # ``localEulerAnglesRaw`` is degrees, as the legacy
            # ``m_EulerCurves`` below are; "under 4, so radians" read a
            # 3° curve as 172° (§7.515).  ``YXZ`` is ``qy·qx·qz``,
            # ``Quaternion.Euler``'s order.
            slot["rot"] = np.array([_mirror_quat(m3.quat_from_euler(*np.deg2rad(block[i]), order="YXZ"))
                                    for i in range(T)], dtype=np.float64)
    return buckets


def sample_transforms(clip_obj, time: float, path_map: dict) -> dict[str, dict]:
    """A Unity muscle clip's Transform channels at exactly *time* (seconds), as ``{bone: {"pos"|"rot"|"scale": …}}``.

    For an ``Animator`` that is not the body's (*path_map* hashes the paths
    under its own transform): no resampling grid, the curves evaluated at
    *time* itself.  Empty when the clip has no muscle payload.
    """
    muscle = getattr(clip_obj, "m_MuscleClip", None)
    data = getattr(getattr(muscle, "m_Clip", None), "data", None) if muscle is not None else None
    if data is None:
        return {}
    dense = getattr(data, "m_DenseClip", None)
    streamed = getattr(data, "m_StreamedClip", None)
    if dense is None or streamed is None:
        return {}
    floats = _sample_float_curves(dense, streamed, getattr(data, "m_ConstantClip", None),
                                  np.array([float(time)], dtype=np.float64))
    cbc = getattr(clip_obj, "m_ClipBindingConstant", None)
    gens = list(getattr(cbc, "genericBindings", None) or []) if cbc else []
    return {bone: {k: v[0] for k, v in slot.items()}
            for bone, slot in _transform_buckets(gens, floats, path_map).items()}


def _unity_clip_to_ours(clip_obj, name: str | None = None, *,
                       sample_fps: float = 30.0, root: str = "cf_j_root", sex: str = "",
                       path_map: dict | None = None):
    """Convert one Unity AnimationClip object to our absolute-local clip.

    *root* is the transform the playing ``Animator`` sits on — what the
    clip's binding paths are relative to (:func:`bone_path_crc_map`).  An
    Animator that is not on a body (an H item's, ``ItemObject.LoadAnimation``)
    passes its own *path_map* (``{crc32(path): node}``): every node it reaches
    is kept, and one is enough.
    """
    from ..anim.clip import AnimationClip

    name = name or str(getattr(clip_obj, "m_Name", None)
                       or getattr(clip_obj, "name", None) or "origin_h")
    own_map = path_map is not None
    path_map = path_map if own_map else bone_path_crc_map(root)

    muscle = getattr(clip_obj, "m_MuscleClip", None)
    if muscle is None:
        return _legacy_clip_to_ours(clip_obj, name, sex)

    stop = float(getattr(muscle, "m_StopTime", 0) or 0)
    start = float(getattr(muscle, "m_StartTime", 0) or 0)
    op = getattr(muscle, "m_Clip", None)
    data = getattr(op, "data", None) if op is not None else None
    if data is None:
        return None
    dense = getattr(data, "m_DenseClip", None)
    streamed = getattr(data, "m_StreamedClip", None)
    const = getattr(data, "m_ConstantClip", None)
    if dense is None or streamed is None:
        return None

    duration = max(stop - start, 0.05)
    # sample grid
    n_frames = max(2, int(round(duration * sample_fps)) + 1)
    times = np.linspace(start, start + duration, n_frames)

    try:
        floats = _sample_float_curves(dense, streamed, const, times)
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] sample fail {name}: {exc}")
        return None

    cbc = getattr(clip_obj, "m_ClipBindingConstant", None)
    gens = list(getattr(cbc, "genericBindings", None) or []) if cbc else []
    if not gens:
        return None
    buckets = _transform_buckets(gens, floats, path_map, sex, filtered=not own_map)

    if len(buckets) < (1 if own_map else 3):
        return None

    t_rel = times - times[0]
    clip = AnimationClip(name, duration=float(t_rel[-1]), loop=True)
    clip.tags = {"intimate", "origin", "absolute", "dense"}
    for bone, slot in buckets.items():
        clip.set_curve(
            bone, t_rel,
            rots=slot.get("rot"),
            poss=slot.get("pos"),
            scales=slot.get("scale"),
        )
    return clip


def _legacy_clip_to_ours(clip_obj, name: str, sex: str = ""):
    from ..anim.clip import AnimationClip
    from ..core import math3d as m3

    ours = AnimationClip(name, duration=1.0, loop=True)
    ours.tags = {"intimate", "origin", "legacy", "absolute"}
    for curve_list_name, kind in (
        ("m_RotationCurves", "rot"),
        ("m_PositionCurves", "pos"),
        ("m_EulerCurves", "euler"),
    ):
        for c in getattr(clip_obj, curve_list_name, None) or []:
            path = str(getattr(c, "path", "") or "")
            leaf = path.replace("\\", "/").split("/")[-1]
            if not _kept_bone(leaf, sex):
                continue
            curve = getattr(c, "curve", None) or getattr(c, "m_Curve", None)
            keys = getattr(curve, "m_Curve", None) if curve is not None else None
            if not keys:
                continue
            times, vals = [], []
            for k in keys:
                times.append(float(getattr(k, "time", 0) or getattr(k, "m_Time", 0) or 0))
                v = getattr(k, "value", None)
                if v is None:
                    continue
                if kind == "rot":
                    vals.append(_mirror_quat(
                        np.array([float(v.x), float(v.y), float(v.z), float(v.w)])))
                elif kind == "pos":
                    vals.append(_mirror_pos(
                        np.array([float(v.x), float(v.y), float(v.z)])))
                else:
                    vals.append(_mirror_quat(m3.quat_from_euler(
                        *np.deg2rad([float(v.x), float(v.y), float(v.z)]),
                        order="YXZ")))
            if times and vals:
                ta = np.asarray(times, dtype=np.float64)
                if kind == "pos":
                    ours.set_curve(leaf, ta, poss=np.stack(vals))
                else:
                    ours.set_curve(leaf, ta, rots=np.stack(vals))
                ours.duration = max(ours.duration, float(ta[-1]))
    return ours if ours.curves else None


# ===========================================================================
# Absolute local → our additive pose (needs live skeleton bind)
# ===========================================================================
#: The bones above ``cf_n_height``, whose translations the height scale does not reach.
UNSCALED = frozenset({"BodyTop", "p_cf_body_bone", "cf_j_root", "cf_n_height"})


def height_scaled(skeleton, index: int) -> bool:
    """Whether bone *index*'s translation takes the height scale: whether it hangs below ``cf_n_height``.

    Read from the skeleton's own parents, so a node hung on it later counts as where it hangs -- an H
    item under ``cf_j_root`` (7,0's ``p_item_hand``) is above the scale, as in the game, where an item
    under a body bone is below it.  Kept on the skeleton by its bone count; a skeleton with no
    ``cf_n_height`` falls back to :data:`UNSCALED` by name.
    """
    bones = skeleton.bones
    cached = getattr(skeleton, "_height_scaled", None)
    if cached is None or cached[0] != len(bones):
        top = skeleton.index.get("cf_n_height")
        below = None
        if top is not None:
            below = set()
            for i in range(len(bones)):
                j = bones[i].parent
                while j is not None and j >= 0 and j != top:
                    j = bones[j].parent
                if j == top:
                    below.add(i)
        cached = (len(bones), below)
        try:
            skeleton._height_scaled = cached
        except AttributeError:
            pass
    below = cached[1]
    return bones[index].name not in UNSCALED if below is None else index in below


def absolute_pose_to_additive(skeleton, abs_pose: dict) -> dict:
    """Unity absolute local TRS → ``Skeleton.evaluate`` additive pose entries.

    ``local_rot * pose_rot = abs_rot`` ⇒ ``pose_rot = conj(local) * abs``,
    ``pos`` less the local position, ``scale`` over the local scale.

    A translation below ``cf_n_height`` is scaled by the skeleton's ``rig_scale`` first: in the
    game that node carries the height slider's scale and every clip translation under it is scaled
    with it, while the graft bakes the factor into the bone positions instead.  Without it a body
    sat higher by ``(1 − scale) ×`` the clip's hip height — measured in the game's own H scene,
    her hips 306.7 mm against our 344.4 mm, his 575.6 against 614.9, the ratios her 0.879 and his
    0.936 height scale (§7.1177).

    A converted character on the source game's rig (``origin.converted_rig``)
    keeps its own lengths: ``skeleton.clip_offsets[bone]`` is added after the
    scale, ``pos·k + offset − local_pos`` (the ABMX length/position form,
    KKABMX ``BoneModifier.Apply``; on EndminF's three clips every chain keeps its
    length to 0.0 mm this way, against 73.9 mm off when the clip's lengths win,
    PROGRESS #169).  KKABMX pins a modified channel to ``baseline∘modifier``,
    which would also freeze an animated ``cf_j_hips``; adding the offset to the
    animated value instead is this project's rule, as ``figure/bonemod.py``
    already does for its position modifiers.  A card has no offsets and its
    translations are computed exactly as without them.  Each is
    done for every bone at once — the rotation is ``math3d.quat_mul`` /
    ``quat_normalize`` term for term, per column (§7.422: this runs every frame
    for every role, and one bone at a time it was 2.5 ms a call).
    """
    from ..core import math3d as m3

    if skeleton is None:
        return dict(abs_pose)
    out: dict = {}
    rot_bones: list = []
    locs: list = []
    abss: list = []
    pos_bones: list = []
    pos_pairs: list = []
    scale_bones: list = []
    scale_pairs: list = []
    one = np.ones(3)
    k = float(getattr(skeleton, "rig_scale", 1.0) or 1.0)
    always = getattr(skeleton, "always_pos", None) or {}
    offsets = getattr(skeleton, "clip_offsets", None) or {}
    shape_reference = getattr(skeleton, "clip_reference", None) or {}
    for bone, e in abs_pose.items():
        i = skeleton.index.get(bone, -1)
        if i < 0:
            continue
        b = skeleton.bones[i]
        if e.get("rot") is not None:
            rot_bones.append(bone)
            locs.append(shape_reference[bone][0] if bone in shape_reference else b.local_rot)
            abss.append(e["rot"])
        if e.get("pos") is not None:
            pos_bones.append(bone)
            pos = np.asarray(e["pos"], dtype=np.float64)
            fixed = always.get(bone)
            if fixed:
                # ``UpdateAlways`` writes these after the animation (``shape_apply.BODY_ALWAYS``, §7.1179)
                pos = pos.copy()
                for axis, value in fixed.items():
                    pos[axis] = value
            pos = pos * k if k != 1.0 and height_scaled(skeleton, i) else pos
            offset = offsets.get(bone) if offsets else None
            if offset is not None:
                pos = pos + offset
            pos_pairs.append((pos, b.local_pos))
        if e.get("scale") is not None:
            scale_bones.append(bone)
            scale_pairs.append((e["scale"], shape_reference[bone][1] if bone in shape_reference
                                else getattr(b, "local_scale", one)))
        out[bone] = {}
    if pos_bones:
        pp = np.asarray(pos_pairs, dtype=np.float64)
        for bone, row in zip(pos_bones, pp[:, 0] - pp[:, 1]):
            out[bone]["pos"] = row
    if scale_bones:
        sp = np.asarray(scale_pairs, dtype=np.float64)
        for bone, row in zip(scale_bones, sp[:, 0] / np.maximum(sp[:, 1], 1e-8)):
            out[bone]["scale"] = row
    if rot_bones:
        la = np.asarray(locs, dtype=np.float64)
        bb = np.asarray(abss, dtype=np.float64)
        ax, ay, az, aw = -la[:, 0], -la[:, 1], -la[:, 2], la[:, 3]
        bx, by, bz, bw = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        q = np.stack([aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx,
                      aw * bz + ax * by - ay * bx + az * bw,
                      aw * bw - ax * bx - ay * by - az * bz], axis=1)
        n = np.sqrt(np.einsum("ij,ij->i", q, q))
        ok = n > m3.EPS
        q[ok] /= n[ok, None]
        q[~ok] = m3.quat_identity()
        for bone, row in zip(rot_bones, q):
            out[bone]["rot"] = row
    return {bone: entry for bone, entry in out.items() if entry}


# ===========================================================================
# Bundle scan + bake
# ===========================================================================
def list_h_bundle_candidates(inst: Install | None = None, *, limit: int = 40) -> list[Path]:
    """Interleave female/male packs so a small ``limit`` still covers both sexes."""
    inst = inst or _install()
    if inst is None or not inst.valid:
        return []
    ab = inst.abdata
    by_sex: dict[str, list[Path]] = {}
    for sex in ("female", "male"):
        d = ab / "h" / "anim" / sex
        if not d.is_dir():
            by_sex[sex] = []
            continue
        packs = sorted(
            d.glob("*.unity3d"),
            key=lambda p: (0 if p.stem.endswith("00_00") else 1, p.name),
        )
        by_sex[sex] = packs
    found: list[Path] = []
    # round-robin so --bundles 4 → f0, m0, f1, m1
    i = 0
    while len(found) < limit:
        added = False
        for sex in ("female", "male"):
            packs = by_sex.get(sex) or []
            if i < len(packs):
                found.append(packs[i])
                added = True
                if len(found) >= limit:
                    break
        if not added:
            break
        i += 1
    return found


def _stage_score(stem: str) -> int:
    """Rank a controller state so a small ``max_clips`` still gets the useful ones."""
    nl = stem.lower()
    score = 0
    if "wloop" in nl:
        score += 5
    if "sloop" in nl:
        score += 4
    if "oloop" in nl or "mloop" in nl:
        score += 3
    if "insert" in nl:
        score += 2
    if "idle" in nl and "stop" not in nl:
        score += 1
    return score


#: What :func:`extract_clips_by_controller` hands back in place of a clip when
#: the caller already has that clip on disk (``have``).  Converting a clip is
#: the slow part of a bake, and re-converting one only to learn it exists made
#: a non-``--force`` run as slow as a forced one (§7.371).
ON_DISK = object()

#: How a clip of the two-person size grid (§7.374) names the other woman's
#: size: ``…_M_WLoop@S``.  The **stage** stays ``M_WLoop`` — everything that
#: reads the phase off it keeps working — and the other size is its own
#: manifest field, ``other_size``.
OTHER_SIZE_SEP = "@"


def other_size_suffix(other: str) -> str:
    """The name suffix for the other woman's size prefix (``"S_"`` → ``"@S"``)."""
    return f"{OTHER_SIZE_SEP}{other[:1]}" if other else ""


def animator_root(clip_objs, candidates) -> str | None:
    """The one candidate transform every Transform binding of *clip_objs* resolves under.

    A binding path is relative to the ``Animator`` that plays the clip, so the
    right root is the one whose :func:`bone_path_crc_map` holds every path
    hash; a clip set that more than one candidate explains (or none) answers
    ``None`` rather than picking.  Studio's hand animations are the case
    (``HandAnimeCtrl.animator``, E1d).
    """
    paths: set[int] = set()
    for obj in clip_objs:
        if obj is None:
            continue
        try:
            data = obj.read()
        except (OSError, ValueError, KeyError, AttributeError) as exc:
            swallowed.note("origin.hanim.animator_root: obj.read", exc)
            continue
        cbc = getattr(data, "m_ClipBindingConstant", None)
        for b in (list(getattr(cbc, "genericBindings", None) or []) if cbc else []):
            if int(getattr(b, "typeID", 0) or 0) == 4 and int(getattr(b, "path", 0) or 0):
                paths.add(int(b.path))
    if not paths:
        return None
    hits = [c for c in candidates if paths <= set(bone_path_crc_map(c))]
    return hits[0] if len(hits) == 1 else None


def extract_clips_by_controller(path: Path, *, max_clips: int = 24,
                                controllers: set | None = None,
                                stage_filter: re.Pattern | None = None,
                                all_clips: bool = False,
                                have=None,
                                layer_jobs: dict | None = None,
                                keep: set | None = None,
                                root: "str | tuple" = "cf_j_root") -> tuple[list, dict]:
    """``([(clip, meta), …], report)`` — one entry per **(controller, state)**.

    This is the identifying bake.  The extractor it replaces keyed on the
    clip's *name*, and a name does not identify a clip inside a pack:
    ``02_00_00`` holds 2,665 clips under 481 distinct names across 23
    controllers, so de-duplicating on the name kept 1 of every 23 and dropped
    which position it belonged to.  Across the 32 packs that is 10,591 of
    16,648 clips (63.6%) that no name could reach (§7.112).

    What identifies a clip is the controller that plays it: ``m_AnimationClips``
    indexed by the state's ``m_ClipID``.  ``(controller, state)`` collides **0
    times** in all 32 packs, so ``origin_{sex}_{pack}_{controller}_{state}`` is a
    key — and the state name is also what Origin itself plays (``SetPlay`` →
    ``Animator.Play(name)``, §7.111), so the runtime asks for the same thing.

    Two kinds of state deliberately produce nothing rather than a plausible
    clip (``CLAUDE.md`` 一之一): a **blend-tree** state is N clips blended by a
    parameter, not one (``Te_kokan_D_kosuru`` is the 3×3 grid of §7.111), and a
    state whose ``m_ClipID`` resolves to nothing is reported, not guessed.
    Both are counted in the returned report.

    *layer_jobs* is :func:`layer_jobs` — the synced layers ``AnimatorLayerCtrl``
    turns on (``al_*``, D1d).  Their leaves bake like any clip, cut to the
    layer's skeleton mask, with ``layer`` in the meta; see
    :func:`_layer_wanted`.

    *keep* is ``{(controller, state)}`` baked whatever *stage_filter* and the
    stage ranking say — the states Studio's ``HAnime_*`` rows play
    (:func:`studio_anime.studio_h_states`).  *root* is where the playing
    ``Animator`` sits (:func:`bone_path_crc_map`); a tuple of candidates
    picks, per controller, the one that resolves every Transform binding of
    its clips (:func:`animator_root`) -- a controller none of them explains is
    counted ``unrooted`` and skipped.
    """
    from .meshimport import _load_bundle, clear_cache as clear_bundle_cache
    from .hlist import (controller_states, controller_size_trees,
                        controller_size_grids, controller_motion_trees)

    report = {"controllers": 0, "states": 0, "blend_tree": 0, "unresolved": 0,
              "converted": 0, "shared": 0, "failed": 0, "skipped_ctrl": 0,
              "truncated": 0}
    out: list = []
    ctrl_root: dict[str, str] = {}
    try:
        env = _load_bundle(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] bundle fail {path.name}: {exc}")
        try:
            clear_bundle_cache()
        except Exception as exc: # noqa: BLE001
            swallowed.note("origin.hanim.extract_clips_by_controller: clear_bundle_cache", exc)
        report["failed"] = 1
        return out, report

    sex = "f" if "female" in str(path).lower() else (
        "m" if "male" in str(path).lower() else "x")
    clip_objs: dict[int, object] = {}
    ctrl_objs = []
    for obj in env.objects:
        if obj.type.name == "AnimationClip":
            clip_objs[int(obj.path_id)] = obj
        elif obj.type.name == "AnimatorController":
            ctrl_objs.append(obj)

    # (score, controller, state, clip_path_id) — ranked so a capped bake keeps
    # the loops the player actually spends time in.
    wanted: list[tuple[int, str, str, int, str, dict]] = []
    for obj in ctrl_objs:
        try:
            tree = obj.read_typetree()
        except Exception as exc: # noqa: BLE001
            swallowed.note("origin.hanim.extract_clips_by_controller: obj.read_typetree", exc)
            continue
        cname = str(tree.get("m_Name") or "")
        if controllers is not None and cname not in controllers:
            report["skipped_ctrl"] += 1
            continue
        report["controllers"] += 1
        ptrs = tree.get("m_AnimationClips") or []
        size_trees = controller_size_trees(tree)
        size_grids = controller_size_grids(tree)
        motion_trees = controller_motion_trees(tree)

        def pid_of(index: int, ptrs=ptrs) -> int:
            ptr = ptrs[index]
            return int(ptr.get("m_PathID", 0)) if isinstance(ptr, dict) \
                else int(getattr(ptr, "path_id", 0))

        if isinstance(root, (tuple, list)):
            found = animator_root([clip_objs.get(pid_of(i)) for i in range(len(ptrs))], root)
            if found is None:
                report["unrooted"] = report.get("unrooted", 0) + 1
                continue
            ctrl_root[cname] = found
        for stem, ids in controller_states(tree):
            report["states"] += 1
            # Studio's rows name a state without its size (``Drink``; the controller has
            # ``S_Drink``…), and ``played_stage`` goes in by the ``S_`` one: kept by either (§7.1133)
            kept = keep is not None and ((cname, stem) in keep or (
                stem[:2] in SIZE_PREFIXES and (cname, stem[2:]) in keep))
            if stage_filter is not None and not stage_filter.search(stem) and not kept:
                continue
            if not ids:
                report["unresolved"] += 1
                continue
            if len(ids) > 1:
                # A body-size tree is the same three clips other packs author
                # as ``S_X``/``M_X``/``L_X`` states (§7.362), so it is baked as
                # those three and the runtime's size blend plays it (§7.371).
                # Any other tree is N clips mixed by a parameter we do not
                # drive, and stays counted rather than approximated.
                # The two-person grid (§7.374) is the same thing once per size
                # of the *other* woman, so it bakes as nine clips keyed by both.
                motion = motion_trees.get(stem)
                if motion is not None:
                    # ``MLoop`` (D1d): two motions per size cell, blended at run
                    # time by ``motion``.  Each leaf bakes under the name Origin
                    # gives it (``M_MLoop1`` / ``M_MLoop2``) — the same stems the
                    # masturbation action controllers override the base with.
                    pids_m = [(param, m, other, [pid_of(i) for i in row])
                              for param, m, other, row in motion]
                    if any(p not in clip_objs for *_, row in pids_m for p in row):
                        report["unresolved"] += 1
                        continue
                    report["motion_tree"] = report.get("motion_tree", 0) + 1
                    score = _stage_score(stem)
                    for param, m, other, row in pids_m:
                        for pref, leaf_pid in zip(SIZE_PREFIXES, row):
                            wanted.append((score, cname, f"{pref}{stem}{m + 1}", leaf_pid,
                                           other, {"motion_param": param, "motion": m}))
                    continue
                grid = size_grids.get(stem)
                if grid is not None:
                    rows = [(other, row) for other, row in zip(SIZE_PREFIXES, grid)]
                elif stem in size_trees:
                    rows = [("", size_trees[stem])]
                else:
                    report["blend_tree"] += 1
                    continue
                report["size_tree"] = report.get("size_tree", 0) + 1
                pids = [(other, [pid_of(i) for i in row]) for other, row in rows]
                if any(p not in clip_objs for _, row in pids for p in row):
                    report["unresolved"] += 1
                    continue
                score = _stage_score(stem)
                if stage_filter is None and not all_clips and score == 0 and not kept:
                    continue
                for other, row in pids:
                    for pref, leaf_pid in zip(SIZE_PREFIXES, row):
                        wanted.append((score, cname, pref + stem, leaf_pid, other, {}))
                continue
            pid = pid_of(ids[0])
            if pid not in clip_objs:
                report["unresolved"] += 1
                continue
            score = _stage_score(stem)
            # ``stage_filter`` is the gate when given; otherwise fall back to
            # the stage ranking so an unfiltered call still skips the noise.
            if stage_filter is None and not all_clips and score == 0 and not kept:
                continue
            wanted.append((score, cname, stem, pid, "", {}))

        jobs = (layer_jobs or {}).get((path.stem, cname))
        if jobs:
            wanted.extend(_layer_wanted(tree, cname, jobs, pid_of, clip_objs,
                                        stage_filter, report))

    wanted.sort(key=lambda x: (-x[0], x[1], x[2], x[4]))
    # Convert each distinct clip once: two states can play the same clip
    # (``Te_siri_sawari_L`` and ``_R`` both do), and conversion is the slow part.
    converted: dict[int, object] = {}
    for i, (score, cname, stem, pid, other, extra) in enumerate(wanted):
        if len(out) >= max_clips:
            # Say what the cap dropped.  ``max_clips`` defaults to a small
            # number for quick tests, and a bake that silently stops at it
            # looks exactly like a pack that only had that many states.
            report["truncated"] = len(wanted) - i
            break
        name = f"{CLIP_PREFIX}{sex}_{path.stem}_{cname}_{stem}"[:96] + other_size_suffix(other)
        if extra.get("defaults") is not None:
            from ..anim.clip import AnimationClip
            out.append((AnimationClip(stem, 1.0, True), name,
                        {"sex": sex, "pack": path.stem, "controller": cname, "stage": stem, "clip_id": 0,
                         **{k: v for k, v in extra.items() if k != "mask"}}))
            continue
        if pid in converted:
            clip = converted[pid]
            reused = True
        elif have is not None and have(f"{sex}/{name}.json"):
            clip = converted[pid] = ON_DISK
            reused = False
        else:
            reused = False
            try:
                data = clip_objs[pid].read()
                clip = _unity_clip_to_ours(data, name=stem,
                                           root=ctrl_root.get(cname, "cf_j_root")
                                           if isinstance(root, (tuple, list)) else root, sex=sex)
            except Exception as exc:  # noqa: BLE001
                print(f"[hanim] convert fail {path.stem}/{cname}/{stem}: {exc}")
                clip = None
            converted[pid] = clip
            if clip is not None:
                report["converted"] += 1
        if clip is not None and clip is not ON_DISK and "mask" in extra:
            clip = _masked_clip(clip, extra["mask"])
        if clip is None or (clip is not ON_DISK and len(clip.curves) < 3
                            and "mask" not in extra):
            continue
        meta = {"sex": sex, "pack": path.stem, "controller": cname,
                "stage": stem, "clip_id": pid, "clip": _peek_name(clip_objs[pid])}
        if cname in ctrl_root:
            meta["root"] = ctrl_root[cname]
        if other:
            meta["other_size"] = other
        meta.update({k: v for k, v in extra.items() if k != "mask"})
        out.append((clip, name, meta))
        if reused:
            report["shared"] += 1
    try:
        clear_bundle_cache()
    except Exception as exc: # noqa: BLE001
        swallowed.note("origin.hanim.extract_clips_by_controller: clear_bundle_cache", exc)
    return out, report


#: Manifest fields a synced-layer leaf carries on top of a clip's (D1d).
LAYER_FIELDS = ("layer", "layer_name", "layer_mode", "layer_state",
                "layer_param", "breast", "motion_param", "motion", "serves", "defaults")


def layer_jobs(actions) -> dict:
    """``{(pack, controller): {layer: {base state: [served controller, …]}}}``.

    Where each role's ``al_`` layers live: on the controller its animator
    **runs** — the base, since ``SetupAnimatorOverrideController`` builds the
    override from the base and only swaps clips (§7.393), or the role's own
    controller when the row names no base.  ``khh_f_43`` has one layer and
    lists 82; ``khh_f_base`` has 87.

    A layer below :data:`hlist.LAYER_CTRL_FIRST` is keyed ``"*"`` (every
    state): ``Proc`` sets any listed index under ``layerCount`` to 1 but
    ``ResetAllLayer`` only zeroes 73 and up, so once set it stays set — and
    ``ChangeAnimator`` sets the row's ``lstIdLayer[0]`` at load, before any
    state plays.  For 73 and up that load-time weight is undone at once by
    ``alCtrl.Load``'s own ``ResetAllLayer``, so only the ``al_`` rows count.
    """
    from .hlist import LAYER_CTRL_FIRST

    out: dict = {}
    for a in actions or ():
        for key in ("female", "male", "female1"):
            role = a.get(key) or {}
            host = ((role.get("base_pack"), role.get("base_controller"))
                    if role.get("base_controller")
                    else (role.get("pack"), role.get("controller")))
            if not all(host):
                continue
            pairs = [(state, int(li)) for state, ids in (role.get("layers") or {}).items()
                     for li in ids]
            if int(role.get("id_layer", -1)) >= 0:
                pairs.append(("*", int(role["id_layer"])))
            for state, li in pairs:
                state = "*" if li < LAYER_CTRL_FIRST else state
                served = out.setdefault(host, {}).setdefault(li, {}).setdefault(state, [])
                if role.get("controller") not in served:
                    served.append(role.get("controller"))
    return out


def _layer_wanted(tree: dict, cname: str, jobs: dict, pid_of, clip_objs: dict,
                  stage_filter, report: dict) -> list:
    """``wanted`` rows for one controller's synced layers (D1d).

    Each leaf of a listed state's tree becomes one clip named
    ``L{layer}_{state}_m{motion}_b{bust child}``.  The clip is the base's own
    — **not** the action's: the action is laid over the base by name
    (``AnimatorOverrideController[clip.name] = clip``), and a layer leaf shares
    its name with a Base Layer leaf (``S_WLoop1`` is thirteen clips in
    ``khh_f_base``).  Only one reading makes the layer mean anything: the
    Base Layer's copy, which comes first, is the one overridden.  Were every
    same-named original swapped, a layer's ``S_``/``M_``/``L_`` children would
    play the action's whole-body clip and its ``SM_``/``ML_`` children the
    base's.  Measured in the game (§7.423): Studio playing ``khh_f_43``'s
    ``S_WLoop1`` frozen at 0.25 with ``Breast`` 0 and 1 puts every masked
    ``cf_j_*`` arm joint on this bake's ``b0`` / ``b4`` leaf to 0.00°, where
    the action's clip is 12–25° off.  The mask is resolved to bone names here and applied
    to the curves, so the runtime overrides exactly what the layer writes.
    """
    from .hlist import controller_layer_trees, BREAST_PARAMS, MOTION_PARAMS

    crc = bone_path_crc_map()
    out = []
    trees = controller_layer_trees(tree, {li: set(st) for li, st in jobs.items()})
    report["layers"] = report.get("layers", 0) + len(trees)
    for li, info in trees.items():
        # Path hash 0 is the empty path — the Animator's own GameObject, which
        # every skeleton mask carries at weight 1.  ``bone_path_crc_map`` aliases
        # it to ``cf_j_root`` for clip bindings (no H clip binds it: 0 of 8,600
        # in three packs), so taken through that map it reads as "the layer
        # writes cf_j_root" — while the mask's own ``cf_j_root`` entry is 0.
        # ``khr_f_02_00``'s hand layer authors its root 180° / 305 mm off the
        # Base Layer's, so the alias turned the woman round.
        hashes = [h for h in info["mask"] if h != 0]
        mask = frozenset(crc[h] for h in hashes if h in crc)
        lost = sum(1 for h in hashes if h not in crc)
        if lost:
            report["layer_mask_unresolved"] = report.get("layer_mask_unresolved", 0) + lost
        if info["skipped"]:
            report.setdefault("layer_skipped", []).extend(
                f"{info['name']}:{st}" for st in info["skipped"])
        for state in info.get("defaults") or ():
            # an empty state writing defaults: no clip, only the bones it puts back (D11)
            if stage_filter is not None and not stage_filter.search(state):
                continue
            out.append((_stage_score(state), cname, f"L{li}_{state}_defaults", None, "", {
                "layer": li, "layer_name": info["name"], "layer_mode": info["mode"],
                "layer_state": state, "layer_param": "", "breast": 0, "motion_param": "", "motion": 0,
                "defaults": sorted(mask),
                "serves": sorted(set(jobs[li].get(state, [])) | set(jobs[li].get("*", []))), "mask": mask}))
        for state, leaves in info["states"].items():
            if stage_filter is not None and not stage_filter.search(state):
                continue
            # Ranked like the state it rides on: a capped bake should not keep a
            # layer and drop the clip underneath it.
            score = _stage_score(state)
            param = next(p for _, c in leaves for p in c if p in BREAST_PARAMS)
            mparam = next((p for _, c in leaves for p in c if p in MOTION_PARAMS), "")
            for cid, coords in leaves:
                pid = pid_of(cid)
                if pid not in clip_objs:
                    report["unresolved"] += 1
                    continue
                m = int(coords.get(mparam, 0)) if mparam else 0
                k = int(coords[param])
                out.append((score, cname, f"L{li}_{state}_m{m}_b{k}", pid, "", {
                    "layer": li, "layer_name": info["name"],
                    "layer_mode": info["mode"], "layer_state": state,
                    "layer_param": param, "breast": k,
                    "motion_param": mparam, "motion": m,
                    "serves": sorted(set(jobs[li].get(state, []))
                                     | set(jobs[li].get("*", []))), "mask": mask}))
    return out


def _masked_clip(clip, mask):
    """A copy of *clip* keeping only the bones in *mask* (a layer's skeleton mask)."""
    from ..anim.clip import AnimationClip

    out = AnimationClip(clip.name, clip.duration, clip.loop)
    out.curves = {b: c for b, c in clip.curves.items() if b in mask}
    out.tags = set(clip.tags)
    return out


def _clip_to_json(clip) -> dict:
    d = clip.to_dict()
    d["space"] = "absolute" if "absolute" in clip.tags else "additive"
    return d


def _clip_from_json(d: dict):
    from ..anim.clip import AnimationClip
    clip = AnimationClip.from_dict(d)
    clip.name = current_name(clip.name)
    if d.get("space") == "absolute":
        clip.tags.add("absolute")
    return clip


#: Bundles a ``resume`` bake has finished, one path per line.  Only a run that
#: asks for ``resume`` reads it, and a run that reaches its end deletes it, so
#: it never outlives the retries it is for (§7.371).
RESUME_FILE = BAKE_H_DIR / ".resume"


class ManifestUnreadable(RuntimeError):
    """An existing clip manifest that cannot be read as an index."""


def starting_index(manifest: Path, *, force: bool, packs: set | None) -> dict:
    """The clip index a bake starts from, before it converts anything.

    ``force`` rewrites what this run converts; it does not forget what it did
    not touch.  Starting from an empty index whenever ``force`` was set meant
    ``--force --pack X`` left a manifest of pack X alone — 75 of ~9,000
    entries, with every other file still on disk (§7.371).  A pack-scoped
    force drops only that pack's entries; an unscoped one starts clean and
    never reads the file.

    A manifest that exists but does not parse raises :class:`ManifestUnreadable`
    instead of standing in for an empty one: an empty start under a pack-scoped
    force is §7.371 again, and under no force it rebuilds only the bundles this
    run reaches (``limit_bundles``), so either way the rewritten manifest would
    silently lose entries whose files are still on disk (§7.496).
    """
    index: dict = {"version": MANIFEST_VERSION, "clips": {}}
    if not manifest.is_file() or (force and not packs):
        return index
    try:
        migrate_baked_names()
        index = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestUnreadable(f"{manifest}: {exc}") from exc
    if not isinstance(index, dict):
        raise ManifestUnreadable(f"{manifest}: top level is {type(index).__name__}, not an object")
    if force and packs:
        for key in ("clips", "layers"):
            index[key] = {k: v for k, v in (index.get(key) or {}).items()
                          if str(v.get("pack") or "") not in packs}
    return index


def bake_h_clips(*, limit_bundles: int = 8, max_per_bundle: int = 12,
                 force: bool = False, progress=None,
                 all_clips: bool = False, packs: set | None = None,
                 stage_filter: "re.Pattern | None" = None,
                 resume: bool = False) -> dict:
    """Convert H packs into ``assets_baked/h_anim/**.json``.

    ``progress`` is an optional ``callable(done, total, message)`` for UI.
    Return ``{"cancelled": True}`` if the callback returns a truthy "cancel".
    ``all_clips``：不限 WLoop/SLoop…，bundle 內每條可轉的 AnimationClip 都 bake。
    ``packs``：只轉這些 pack id（``{"02_00_00"}``），其餘略過。
    ``stage_filter``：對 **controller 狀態名** 的正規式；不給就用預設的階段組。
    """
    def _prog(done: int, total: int, msg: str) -> bool:
        if progress is None:
            return False
        try:
            return bool(progress(done, total, msg))
        except Exception as exc:  # noqa: BLE001
            swallowed.note("origin.hanim.bake_h_clips._prog: return bool(progress(done, total, msg))", exc)
            return False

    inst = _install()
    if inst is None:
        return {"ok": False, "error": "no_origin_install", "written": 0}
    # Bake can also read already-synced assets_cache without a full install.
    BAKE_H_DIR.mkdir(parents=True, exist_ok=True)
    if packs:
        bundles = [b for b in list_h_bundle_candidates(inst, limit=10_000)
                   if b.stem in packs]
    else:
        bundles = list_h_bundle_candidates(inst, limit=limit_bundles)
    if not bundles:
        return {"ok": False, "error": "no_bundles", "written": 0,
                "dir": str(BAKE_H_DIR)}
    written = 0
    skipped = 0
    failed: list[str] = []
    # Packs that produced nothing *on purpose* — kept out of ``failed`` so the
    # failure list stays short enough to trust line by line.
    empty: list[str] = []
    try:
        index = starting_index(BAKE_MANIFEST, force=force, packs=packs)
    except ManifestUnreadable as exc:
        return {"ok": False, "error": f"manifest_unreadable: {exc}", "written": 0,
                "dir": str(BAKE_H_DIR)}
    # v1 entries stay readable; they simply carry no controller, so the menu
    # keeps flagging their rows ambiguous until this bake replaces them.
    index["version"] = MANIFEST_VERSION

    # The action list has to exist *before* the clips, because it is what says
    # which controllers a pack's 106 actions actually name — without it the
    # bake would convert all 23 controllers of ``02_00_00`` including the ones
    # no action references.  It is cheap (16 list bundles) next to the clips.
    action_ctrls: dict = {}
    try:
        from .hlist import bake_actions
        act = bake_actions(ACTIONS_FILE)
        if not act.get("ok"):
            print(f"[hanim] action list skipped: {act.get('error')} "
                  f"{act.get('failed') or ''}（留著上一份 actions.json）")
        else:
            print(f"[hanim] action list: {act['actions']} 個動作 {act['by_mode']}")
        _actions.cache_clear()
        action_ctrls = _controllers_by_pack()
        jobs = layer_jobs(_actions())
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] action list failed: {exc}")
        jobs = {}
    # Studio's ``HAnime_*`` rows play states no phase of ours does (``Pull``,
    # ``Drink``…) and sometimes a base controller no action names (E1a).
    try:
        from .studio_anime import studio_h_states
        studio_keep = studio_h_states()
    except (OSError, ValueError, KeyError) as exc:
        swallowed.note("origin.hanim.bake_h_clips: studio_h_states", exc)
        studio_keep = {}

    # The H expression table and the head's pattern sets ride along: both come
    # off the same install, both are tiny, and the clips are useless for a face
    # without them (H clips carry no blend-shape channel at all — §7.130).
    try:
        from .hlist import bake_hface
        hf = bake_hface(HFACE_FILE)
        if not hf.get("ok"):
            print(f"[hanim] HFaceList skipped: {hf.get('error')}")
        else:
            print(f"[hanim] HFaceList: {hf['rows']} 列／{hf['personalities']} 個性"
                  f"（吐舌 {hf['tongue_rows']}；{len(hf['missing_files'])} 組無檔）")
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] HFaceList failed: {exc}")
    try:
        from .hlist import bake_breath
        br = bake_breath(BREATH_FILE)
        if not br.get("ok"):
            print(f"[hanim] breath skipped: {br.get('error')}")
        else:
            print(f"[hanim] breath: {br['states']} 個 state／{br['infos']} 筆條件、"
                  f"{br['voices']} 個語音／{br['levels']} 個等級")
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] breath failed: {exc}")
    try:
        from .hlist import bake_voice
        vo = bake_voice(BAKE_H_DIR / "voice.json")     # ``h_voice.VOICE_FILE`` (it imports this module)
        if not vo.get("ok"):
            print(f"[hanim] voice skipped: {vo.get('error')}")
        else:
            print(f"[hanim] voice: {vo['personalities']} 個個性、{vo['patterns']} 組模式、"
                  f"{vo['lines']} 句（{vo['n_missing']} 句缺長度）")
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] voice failed: {exc}")
    try:
        from .hlist import bake_eyeneck
        en = bake_eyeneck(EYENECK_FILE)
        if not en.get("ok"):
            print(f"[hanim] eyeneck skipped: {en.get('error')}")
        else:
            print(f"[hanim] eyeneck: {en['ids']} 個 id／{en['files']} 份清單、"
                  f"{en['states']} 個 state／{en['actions']} 個動作"
                  f"（{len(en['missing_files'])} 份缺檔）")
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] eyeneck failed: {exc}")
    try:
        from .hlist import bake_motion_ik
        mi = bake_motion_ik(MOTION_IK_FILE)
        global _MOTION_IK_TARGETS
        _MOTION_IK_TARGETS = None         # the clips below keep the bones it names
        if not mi.get("ok"):
            print(f"[hanim] motion_ik skipped: {mi.get('error')}")
        else:
            print(f"[hanim] motion_ik: {mi['files']} 份／{mi['states']} 個 state"
                  f"（{len(mi['missing_files'])} 份缺檔）")
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] motion_ik failed: {exc}")
    try:
        from .fbbik_bake import bake_fbbik
        fb = bake_fbbik(FBBIK_FILE)
        if not fb.get("ok"):
            print(f"[hanim] fbbik skipped: {fb.get('error')}")
        else:
            print(f"[hanim] fbbik: {fb['effectors']} 個 effector／{fb['names']} 個參照")
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] fbbik failed: {exc}")
    try:
        from .hlist import bake_lookat
        from .lookat_dan import clear_lookat_cache
        la = bake_lookat(LOOKAT_FILE)
        # the clips below keep the curves of the nulls these lists name
        clear_lookat_cache()
        if not la.get("ok"):
            print(f"[hanim] lookat skipped: {la.get('error')}")
        else:
            print(f"[hanim] lookat: {la['actions']} 個動作／{la['lists']} 份清單、"
                  f"{la['rows']} 列（{len(la['missing_files'])} 份缺檔）")
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] lookat failed: {exc}")
    try:
        from .hlist import bake_attach
        at = bake_attach(ATTACH_FILE)
        global _ATTACH_NULLS
        _ATTACH_NULLS = None              # the clips below keep the nulls it names
        if not at.get("ok"):
            print(f"[hanim] attach skipped: {at.get('error')}")
        else:
            print(f"[hanim] attach: {at['items']} 個動作／{at['item_lists']} 份道具清單、"
                  f"{at['aibu_layers']} 份愛撫清單；{len(at['nulls'])} 個 null"
                  f"（{len(at['missing_files'])} 份缺檔）")
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] attach failed: {exc}")
    try:
        from .hlist import bake_items
        it = bake_items(ITEMS_FILE)
        global _ROW_ITEMS
        _ROW_ITEMS = None
        if not it.get("ok"):
            print(f"[hanim] items skipped: {it.get('error')}")
        else:
            print(f"[hanim] items: {it['lists']} 份道具清單、{it['items']} 個道具（{len(it['missing'])} 個缺）；"
                  f"道具自己的動作 {it.get('animated')}")
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] items failed: {exc}")
    try:
        from .hlist import bake_flow
        fl = bake_flow(FLOW_FILE)
        _flow_data.cache_clear()
        if not fl.get("ok"):
            print(f"[hanim] flow skipped: {fl.get('error')}")
        else:
            print(f"[hanim] flow: {fl['rows']} 列、{fl['clips']} 個 clip（{fl['one_pass']} 個只播一次；"
                  f"{len(fl['failed'])} 份讀不到）")
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] flow failed: {exc}")
    try:
        from .hlist import bake_yure
        yr = bake_yure(YURE_FILE)
        if not yr.get("ok"):
            print(f"[hanim] yure skipped: {yr.get('error')}")
        else:
            print(f"[hanim] yure: {yr['actions']} 個動作／{yr['tables']} 份表、{yr['states']} 個狀態"
                  f"（{len(yr['missing_files'])} 份缺檔）")
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] yure failed: {exc}")
    try:
        from .figure.face_morph import bake_face_patterns
        fp = bake_face_patterns()
        if not fp.get("ok"):
            print(f"[hanim] face patterns skipped: {fp.get('error')}")
        else:
            print(f"[hanim] face patterns: {fp['slots']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] face patterns failed: {exc}")

    # total steps ≈ bundles (load) + clips; estimate upper bound for the bar
    total_steps = max(1, len(bundles) * max(1, max_per_bundle))
    step = 0
    if _prog(0, total_steps, f"準備 {len(bundles)} 個 bundle…"):
        return {"ok": False, "cancelled": True, "written": 0}

    # ``^([SML]_)?Loop$`` is the peeping actions' main loop (``khn_f_0N``'s
    # size tree, §7.371); a bare ``Loop`` would also take the hand layers'
    # ``A_Loop`` / ``M_Loop`` / ``S_Loop`` that no phase of ours plays.  The
    # caress kiss loop and climax loop are taken by name — the area and the
    # climax of a caress row play them (``CARESS_AREAS``, §7.397).
    filt = stage_filter if stage_filter is not None else (
        None if all_clips else re.compile(
            r"(WLoop|SLoop|OLoop|MLoop|Insert|Idle|OUT|IN_|^(?:[SML]_)?Loop$"
            r"|(?:^|_)K_Loop$|(?:^|_)Orgasm_Loop$)", re.I))
    tally = {"controllers": 0, "states": 0, "blend_tree": 0, "unresolved": 0,
             "shared": 0, "skipped_ctrl": 0, "truncated": 0}
    done_before: set[str] = set()
    if resume and RESUME_FILE.is_file():
        done_before = {ln.strip() for ln in
                       RESUME_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()}
    finished = None       # the previous bundle, if it really got through
    for bi, b in enumerate(bundles):
        # Save what the previous bundles added before opening the next one.  A
        # run killed from outside (the guard's memory floor, §7.371) otherwise
        # loses every entry it made, and a retry has to redo all of them; with
        # this each retry starts where the last one was stopped, because a
        # non-force run keeps the manifest it finds.
        if bi:
            write_json_atomic(BAKE_MANIFEST, index, indent=2)
            # Only a bundle that loaded counts as done.  The first version
            # marked whatever came before, and a bundle whose load failed
            # under the memory cap (``report["failed"]``, no exception) was
            # then skipped by every retry — female 02_00_00, the largest
            # pack, ended up with no entries at all (§7.371).
            if resume and finished is not None:
                with RESUME_FILE.open("a", encoding="utf-8") as fh:
                    fh.write(finished + "\n")
        finished = None
        if str(b) in done_before:
            continue
        sex = "female" if "female" in str(b).lower() else (
            "male" if "male" in str(b).lower() else "x")
        sex_c = sex[0] if sex in ("female", "male") else "x"
        out_dir = BAKE_H_DIR / sex
        out_dir.mkdir(parents=True, exist_ok=True)
        if _prog(step, total_steps,
                 f"[{bi + 1}/{len(bundles)}] 讀取 {b.parent.name}/{b.name}…"):
            return {"ok": False, "cancelled": True, "written": written,
                    "total": len(index.get("clips") or {}), "failed": failed}
        # Restrict to the controllers Origin's own action list names for this pack.
        # ``None`` (no list) means "every controller" — still identifying, just
        # not scoped to what a player can pick.
        ctrls = action_ctrls.get((sex_c, b.stem))
        keep = studio_keep.get((sex_c, b.stem))
        if ctrls is not None and keep:
            ctrls = set(ctrls) | {c for c, _ in keep}
        try:
            clips, rep = extract_clips_by_controller(
                b, max_clips=max_per_bundle, controllers=ctrls,
                stage_filter=filt, all_clips=all_clips,
                have=None if force else (lambda rel: (BAKE_H_DIR / rel).is_file()),
                layer_jobs=jobs, keep=keep)
        except Exception as exc:  # noqa: BLE001
            print(f"[hanim] bundle fail {b.name}: {exc}")
            failed.append(f"{sex}/{b.stem}: {exc}")
            step = min(total_steps, step + max_per_bundle)
            continue
        if rep.get("failed") and not rep.get("controllers"):
            failed.append(f"{sex}/{b.stem}: bundle 載入失敗")
            step = min(total_steps, step + max_per_bundle)
            continue
        finished = str(b)
        for k in tally:
            tally[k] += int(rep.get(k, 0) or 0)
        if rep.get("truncated"):
            print(f"[hanim] {sex}/{b.stem}: --per {max_per_bundle} 丟掉 "
                  f"{rep['truncated']} 個 state")
        if rep.get("layer_skipped"):
            # A state the ``al_`` list turns a layer on in, whose tree for that
            # layer is missing or of another shape: Origin plays it, we do not.
            failed.append(f"{sex}/{b.stem}: layer 沒轉 {rep['layer_skipped']}")
        if rep.get("layer_mask_unresolved"):
            failed.append(f"{sex}/{b.stem}: layer 遮罩 {rep['layer_mask_unresolved']} "
                          f"個路徑對不到骨頭")
        if rep.get("unresolved"):
            # Report, never substitute — ``m/08_54_00`` has 60 such states.
            failed.append(f"{sex}/{b.stem}: {rep['unresolved']} state 解不到 clip")
        if not clips:
            # 0 clips is not a failure by itself.  A pack whose only matching
            # states are blend trees has nothing *to* bake — those are N clips
            # blended by a parameter, not one (§7.111 的 3×3 網格), and we
            # deliberately do not invent one.  Filing that under ``failed``
            # sends the next round hunting a parse bug: ``female/04_00_00``
            # (後背位 pack 的鄰居) reads as broken when its 148 states hit the
            # stage filter twice and both hits are blend trees.  Split it the
            # way §7.98 split ``missing``: 查不到 vs 刻意不帶.
            if rep.get("blend_tree") and not rep.get("unresolved"):
                empty.append(
                    f"{sex}/{b.stem}: 0 clips（{rep['blend_tree']} 個命中的 "
                    f"state 全是 blend tree，刻意不轉；共 {rep.get('states', 0)}"
                    f" state）")
            else:
                failed.append(f"{sex}/{b.stem}: 0 clips")
        # Two states can play the same clip; write the JSON once and let both
        # manifest entries point at it.
        file_for_clip: dict[tuple, str] = {}
        for clip, name, meta in clips:
            pid = int(meta.get("clip_id", 0))
            # A layer leaf is written cut to its mask, so it never shares a
            # file with the whole clip even when the two are one object.
            fkey = (pid, meta.get("layer"))
            bucket = "layers" if "layer" in meta else "clips"
            rel = file_for_clip.get(fkey) or f"{sex}/{name}.json"
            dest = BAKE_H_DIR / rel
            step = min(total_steps, step + 1)
            if clip is ON_DISK:
                # Not converted: the file is the clip.  Only its two numbers
                # are needed for the manifest, and reading them back is what a
                # reindex costs instead of a conversion.
                try:
                    d = json.loads(dest.read_text(encoding="utf-8"))
                    bones, dur = len(d.get("curves") or ()), float(d.get("duration") or 0.0)
                except (OSError, ValueError) as exc:
                    failed.append(f"{sex}/{name}: 既有的 {rel} 讀不回來（{exc}）")
                    continue
            else:
                bones, dur = len(clip.curves), clip.duration
            entry = {
                "file": rel, "bones": bones,
                "duration": dur, "sex": sex,
                "pack": meta["pack"], "controller": meta["controller"],
                "stage": meta["stage"],
            }
            if meta.get("clip"):
                # the clip's own name: the override (and ``base_clip_states``)
                # goes by it, and it is not always the stage (§7.507)
                entry["clip"] = meta["clip"]
            if meta.get("other_size"):
                entry["other_size"] = meta["other_size"]
            if meta.get("motion_param"):
                entry["motion_param"] = meta["motion_param"]
                entry["motion"] = int(meta["motion"])
            if "layer" in meta:
                entry.update({k: meta[k] for k in LAYER_FIELDS if k in meta})
            if dest.is_file() and not force:
                index.setdefault(bucket, {})[name] = entry
                file_for_clip.setdefault(fkey, rel)
                skipped += 1
                if _prog(step, total_steps, f"略過已有 {name}"):
                    write_json_atomic(BAKE_MANIFEST, index, indent=2)
                    return {"ok": False, "cancelled": True, "written": written,
                            "skipped": skipped, "failed": failed,
                            "total": len(index.get("clips") or {})}
                continue
            if fkey not in file_for_clip:
                write_json_atomic(dest, _clip_to_json(clip))
                written += 1
                print(f"[hanim] baked {rel} bones={len(clip.curves)} "
                      f"dur={clip.duration:.2f}s")
            file_for_clip.setdefault(fkey, rel)
            index.setdefault(bucket, {})[name] = entry
            if _prog(step, total_steps,
                     f"已 bake {written}：{name}（{len(clip.curves)} 骨）"):
                write_json_atomic(BAKE_MANIFEST, index, indent=2)
                return {"ok": False, "cancelled": True, "written": written,
                        "skipped": skipped, "failed": failed,
                        "total": len(index.get("clips") or {})}
        # if fewer clips than max_per_bundle, still advance the bar
        step = min(total_steps, (bi + 1) * max_per_bundle)
        # free UnityPy env before the next (possibly huge) pack
        try:
            from .meshimport import clear_cache
            clear_cache()
        except Exception as exc: # noqa: BLE001
            swallowed.note("origin.hanim.bake_h_clips: clear_cache", exc)

    write_json_atomic(BAKE_MANIFEST, index, indent=2)
    if resume and finished is not None:
        with RESUME_FILE.open("a", encoding="utf-8") as fh:
            fh.write(finished + "\n")
    if resume and not failed:
        RESUME_FILE.unlink(missing_ok=True)
    _prog(total_steps, total_steps,
          f"完成：新寫 {written}，略過 {skipped}，共 {len(index.get('clips') or {})}")
    return {
        "ok": True, "written": written, "skipped": skipped,
        "total": len(index.get("clips") or {}),
        "bundles": [str(b) for b in bundles], "dir": str(BAKE_H_DIR),
        "failed": failed, "empty": empty, "states": tally,
    }


def _load_baked_catalog() -> dict:
    if not BAKE_MANIFEST.is_file():
        return {}
    try:
        migrate_baked_names()
        man = json.loads(BAKE_MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:
        swallowed.note("origin.hanim._load_baked_catalog: man = json.loads(BAKE_MANIFEST.read_text(encodin", exc)
        return {}
    clips = {}
    for name, meta in (man.get("clips") or {}).items():
        path = BAKE_H_DIR / meta["file"]
        if not path.is_file():
            continue
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            clip = _clip_from_json(d)
            clip.name = name          # shared file → report the asked-for name
            clips[name] = clip
        except Exception as exc:  # noqa: BLE001
            print(f"[hanim] bake read fail {path.name}: {exc}")
    return clips


@lru_cache(maxsize=1)
def catalog_h_clips(*, limit_bundles: int = 8, prefer_bake: bool = True) -> dict:
    """``{clip_name: AnimationClip}`` — baked first, else live convert."""
    inst = _install()
    if inst is None:
        return {"clips": {}, "bundles": [], "error": "no_origin_install", "source": None}

    if prefer_bake:
        baked = _load_baked_catalog()
        if baked:
            return {
                "clips": baked,
                "bundles": [],
                "error": None,
                "source": "bake",
                "install": str(inst.root),
            }

    bundles = list_h_bundle_candidates(inst, limit=limit_bundles)
    clips = {}
    filt = re.compile(r"(WLoop|SLoop|OLoop|Insert|Idle)", re.I)
    ctrls = _controllers_by_pack()
    for b in bundles[:limit_bundles]:
        sex_c = "f" if "female" in str(b).lower() else "m"
        # Same identification as the bake — a no-bake machine must not get the
        # arbitrary-controller clip the name-keyed path used to hand back.
        got, _rep = extract_clips_by_controller(
            b, max_clips=8, controllers=ctrls.get((sex_c, b.stem)),
            stage_filter=filt)
        for c, name, _meta in got:
            c.name = name
            clips[name] = c
        if len(clips) >= 40:
            break
    return {
        "clips": clips,
        "bundles": [str(b) for b in bundles],
        "error": None if clips else ("no_clips" if bundles else "no_bundles"),
        "source": "live",
        "install": str(inst.root),
    }


def clear_hanim_cache() -> None:
    catalog_h_clips.cache_clear()
    bone_path_crc_map.cache_clear()
    list_menu_entries.cache_clear()
    _manifest_clip_index.cache_clear()
    _manifest_layer_index.cache_clear()
    _layer_index.cache_clear()
    _actions.cache_clear()
    _pack_controller_count.cache_clear()
    _controllers_by_pack.cache_clear()
    _stage_index.cache_clear()
    _controller_clips.cache_clear()
    _flow_data.cache_clear()
    controller_aware_bake.cache_clear()
    _clip_singleton.clear()
    try:
        from .lookat_dan import clear_lookat_cache
        clear_lookat_cache()
    except Exception as exc: # noqa: BLE001
        swallowed.note("origin.hanim.clear_hanim_cache: clear_lookat_cache", exc)


def _clip_pack_id(name: str) -> str | None:
    """``origin_f_01_00_00_M_WLoop2`` → ``01_00_00``."""
    parts = name.split("_")
    # origin, sex, aa, bb, cc, …
    if len(parts) < 5 or parts[0] != "origin":
        return None
    return f"{parts[2]}_{parts[3]}_{parts[4]}"


def _peek_name(obj) -> str:
    """An ``AnimationClip``'s ``m_Name`` without decoding the clip, ``""`` if unreadable."""
    try:
        return str(obj.peek_name() or "")
    except (OSError, ValueError, AttributeError) as exc:
        swallowed.note("origin.hanim._peek_name", exc)
        return ""


def clip_source_name(name: str) -> str:
    """The baked clip *name*'s own ``m_Name`` in its bundle, ``""`` for a manifest without it."""
    return str((_manifest_clip_index().get(name) or {}).get("clip") or "")


def _clip_stem(name: str) -> str:
    """The controller state this clip is baked for.

    Read from the manifest when it is there.  Splitting the name only works for
    v1 entries — from v2 the controller sits between the pack and the stage and
    both halves contain ``_``.
    """
    meta = _manifest_clip_index().get(name) or {}
    if meta.get("controller"):
        return str(meta.get("stage") or name)
    parts = name.split("_")
    if len(parts) < 6 or parts[0] != "origin":
        return name
    return "_".join(parts[5:])


# First-byte family for menu grouping (from clip content audit §7.106).
_FAMILY_BY_PREFIX = {
    "00": ("aibu", "愛撫"),
    "01": ("houshi", "奉仕"),
    "02": ("sonyu", "挿入"),
    "03": ("sonyu", "挿入"),
    "04": ("sonyu", "挿入"),
    "05": ("sonyu", "挿入"),
    "06": ("houshi", "奉仕"),
    "07": ("sonyu", "挿入"),
    "08": ("houshi", "奉仕"),
    "09": ("sonyu", "挿入"),
}


# Family labels keyed by ``EMode`` name — the list says which mode a row is in,
# so this replaces guessing the family from the pack's first byte.
_FAMILY_LABEL = {
    "aibu": "愛撫", "houshi": "奉仕", "sonyu": "挿入",
    "masturbation": "自慰", "peeping": "偷窺", "lesbian": "百合",
    "houshi3P": "奉仕3P", "sonyu3P": "挿入3P",
    "houshi3PMMF": "奉仕3P男男女", "sonyu3PMMF": "挿入3P男男女",
    "none": "其他", "other": "其他",
}


def family_for_pack(pack: str) -> tuple[str, str]:
    """``(family_key, zh_label)`` for pack id ``AA_BB_CC``."""
    if not pack:
        return ("other", "其他")
    return _FAMILY_BY_PREFIX.get(pack[:2], ("other", "其他"))


def layout_hint_for_pack(pack: str) -> str:
    """``body_stage`` layout key used only for world spacing / camera.

    Full body orientation is already in the clip; this only picks dual spacing.
    """
    if not pack:
        return "missionary"
    # Upright / straddle sonyu packs
    if pack in ("02_20_00", "02_00_00") or pack.startswith("07_"):
        return "cowgirl"
    # Bent / rear-ish
    if pack.startswith(("03_", "04_")):
        return "doggy"
    # Houshi lying service → flat spacing; sonyu 02_06/12/13 on-back
    return "missionary"


def _stem_label(stem: str) -> str:
    """Short Chinese-ish label for a clip stem (strength + loop kind)."""
    s = stem or ""
    strength = ""
    if s.startswith(("SS_", "LL_", "MM_", "ML_", "SM_", "LS_", "LM_")):
        strength = s[:2]
        rest = s[3:] if s[2:3] == "_" else s[2:]
    elif s.startswith(("S_", "M_", "L_")):
        strength = s[0]
        rest = s[2:]
    else:
        rest = s
    kind = rest
    for tok, lab in (
            ("WLoop", "弱循環"), ("SLoop", "強循環"), ("OLoop", "高潮循環"),
            ("InsertIdle", "插入待機"), ("Insert", "插入"),
            ("IN_Start", "內射開始"), ("IN_Loop", "內射循環"),
            ("OUT_Start", "拔出開始"), ("OUT_Loop", "拔出循環"), ("OUT_A", "拔出"),
            ("Oral_Idle", "口交待機"), ("Oral", "口交"),
            ("Idle", "待機"), ("Stop_Idle", "停止待機"),
    ):
        if tok in rest:
            kind = lab
            break
    # S / M / L is the woman's body size (:data:`SIZE_PREFIXES`); the two-letter
    # ones are not a size and keep the labels they had.
    str_lab = {"S": "小", "M": "中", "L": "大", "SS": "極弱", "LL": "極強",
               "MM": "中中", "ML": "中強", "SM": "弱中", "LS": "強弱",
               "LM": "強中"}.get(strength, strength)
    if str_lab and kind:
        return f"{str_lab}·{kind}"
    return stem or "?"


@lru_cache(maxsize=1)
def _manifest_clip_index() -> dict:
    """``{name: {file, bones, duration, sex}}`` from bake manifest only."""
    if not BAKE_MANIFEST.is_file():
        return {}
    try:
        migrate_baked_names()
        man = json.loads(BAKE_MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        swallowed.note("origin.hanim._manifest_clip_index: man = json.loads(BAKE_MANIFEST.read_text(encodin", exc)
        return {}
    return dict(man.get("clips") or {})


# Per-clip cache so Free H can switch menu rows without re-reading JSON.
_clip_singleton: dict = {}


def load_clip(name: str):
    """Load one baked clip by name (single JSON — never the full catalog)."""
    if not name:
        return None
    hit = _clip_singleton.get(name)
    if hit is not None:
        return hit
    # Reuse a warmed full catalog only if it was already built this session.
    if catalog_h_clips.cache_info().currsize > 0:
        try:
            clip = (catalog_h_clips().get("clips") or {}).get(name)
            if clip is not None:
                _clip_singleton[name] = clip
                return clip
        except Exception as exc: # noqa: BLE001
            swallowed.note("origin.hanim.load_clip: catalog_h_clips", exc)
    meta = _manifest_clip_index().get(name) or _manifest_layer_index().get(name)
    if not meta:
        return None
    path = BAKE_H_DIR / meta["file"]
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        clip = _clip_from_json(d)
        # Two states can share one JSON (both ``Te_siri_sawari_L`` and ``_R``
        # play the same clip), so the file carries whichever name wrote it.
        # Report the name that was asked for — it is what the notes display.
        clip.name = name
        _clip_singleton[name] = clip
        return clip
    except Exception as exc:  # noqa: BLE001
        print(f"[hanim] load_clip {name}: {exc}")
        return None


@dataclass(frozen=True)
class HMenuEntry:
    """One Free H menu row.

    A row is **one H action** — a row of Origin's own ``AnimationInfo_*`` list
    (``正常位``, ``立ち愛撫``), which names an ``AnimatorController`` whose states
    are the clip stems.  106 rows, not the 4,680 pack×stem pairs the menu used
    to list: those were the *stages* of an action (``Idle`` / ``Insert`` /
    ``WLoop1`` / ``OLoop`` / ``OUT_*``), one per row.
    """
    key: str                 # stable id (= female name if any else male)
    label: str
    family: str              # aibu | houshi | sonyu | masturbation | …  (EMode)
    family_label: str
    pack: str
    stem: str
    name_f: str | None
    name_m: str | None
    layout: str              # body_stage key for spacing/camera
    # Which female null the shaft is aimed at for this **act** — vaginal /
    # oral / breast / none.  Separate from ``layout`` because one posture
    # serves several acts, exactly as Origin keys its ``dan_*`` rows by motion
    # rather than by pack (§7.142 十二).
    contact: str = "vaginal"
    action_id: int = -1
    action_name: str = ""    # Japanese name from the list ("正常位")
    controller_f: str = ""
    controller_m: str = ""
    stages: tuple = ()       # every state of this action's controller
    ambiguous: bool = False   # see ``_pack_controller_count``
    # The action names a **third** role (``pathFemale1``), i.e. it is a 3P
    # routine — 12 of the 90 rows.  Staging only two bodies is what made those
    # rows look wrong (§7.134); Free H now loads a third (§7.136).
    needs_third: bool = False
    name_t: str | None = None      # third role's baked clip
    third_sex: str = ""            # "f" | "m" — from the controller, not the
                                   # field name, which lies on MMF rows
    controller_t: str = ""
    # each role's base controller (``path*Base``: the woman, the man, the third), whose tree a
    # state of the action controller is played through — which ``motion`` parameter it reads
    bases: tuple = ("", "", "")
    # each role's base states' ``(state, speed parameter, m_Speed)`` (the woman, the man, the third):
    # a role whose state is bound to another parameter, or none, runs at its own rate (``HRuntime.role_phase``)
    role_speeds: tuple = ((), (), ())
    # The H point categories the row can be played on (the list's ``categorys``: 0/1 floor and bed,
    # 2-5 chair, 6/7 desk, 8 wall, 9 standing, 3000+ 3P …) — what picks the place in a room (``game.hroom``)
    categories: tuple = ()
    # ``(category, (pos.x, pos.y, pos.z, rot.x, rot.y, rot.z))`` in the row's category order, for the
    # categories whose ``fileMove`` names a list (Unity's axes, degrees): how far a point of that
    # category moves both bodies (``HSceneProc.SetLocalPosition``, ``game.hroom.move_for``)
    moves: tuple = ()
    # ``AnimationInfo_{mode}``'s file index — the EMode minus one, and the
    # number that keys ``breath_{mode}_00`` and ``HFaceList_c*_00_{mode}``
    # (§7.131).  Kept as a field rather than parsed back out of ``key``.
    mode: int = -1
    # The **base** controller's clip stems, a size-tree state as its three
    # ``S_``/``M_``/``L_`` leaves (``hlist.controller_stems``); with
    # ``base_speeds``' state names they reduce an action stem (``M_WLoop1``)
    # to the base state ``breath_*`` is keyed by — see :func:`base_state_for`.
    base_states: tuple = ()
    # ``(state, speed parameter, m_Speed)`` for each base state
    # (``hlist.controller_speeds``, §7.394) — a tuple so the row stays
    # hashable.  Empty for an ``actions.json`` written before it was read.
    base_speeds: tuple = ()
    # ``(clip name, base state)`` pairs (``hlist.controller_clip_states``):
    # which base state an action clip plays under, read off the trees (§7.405)
    base_clip_states: tuple = ()
    # Origin itself points more than one action at this controller.  Those actions
    # differ by *location*, not motion: ``椅子オナニー`` and ``トイレオナニー`` are
    # both ``kht_f_01`` and differ only in ``lstCategory`` (1011 vs 2003) and
    # ``useChair``.  We have no H points, so the two play the same thing — say
    # so on the row, because an unlabelled pair reads as "the menu is broken".
    shared_motion: bool = False


ACTIONS_FILE = BAKE_H_DIR / "actions.json"
# The H expression table (``HFaceList_*`` → our format).  Written next to the
# clips because it is keyed by the same EMode the actions are (§7.130 二).
HFACE_FILE = BAKE_H_DIR / "hface.json"
# When each of those rows plays: breath patterns (by base-layer state) plus the
# per-personality voice→face table (§7.131).
BREATH_FILE = BAKE_H_DIR / "breath.json"
#: ``HMotionEyeNeck``'s lists: where each role looks in each state (C1)
EYENECK_FILE = BAKE_H_DIR / "eyeneck.json"
#: ``Lookat_dan``'s lists: which female null the shaft aims at in each state (§7.501)
LOOKAT_FILE = BAKE_H_DIR / "lookat.json"
#: ``MotionIK``'s data: each action controller's IK targets and bone corrections (Animation_Progress D14)
MOTION_IK_FILE = BAKE_H_DIR / "motion_ik.json"
#: The body armature's ``FullBodyBipedIK`` settings, references as bone names (``fbbik_bake``, D14)
FBBIK_FILE = BAKE_H_DIR / "fbbik.json"
#: ``YureCtrl``'s tables: which breast / hip chains and breast shapes each state switches (§7.1125)
YURE_FILE = BAKE_H_DIR / "yure.json"
#: The body nulls the row's items and the caress hands hang on (``hlist.bake_attach``), whose curves the bake keeps
ATTACH_FILE = BAKE_H_DIR / "attach.json"
#: The row's items' prefab trees (``hlist.bake_items``): what ``ItemObject.LoadItem`` hangs on the bodies
ITEMS_FILE = BAKE_H_DIR / "items.json"
_ROW_ITEMS: dict | None = None
#: What the action procs' flow reads off the lists and the clips (``hlist.bake_flow``, D13): each row's
#: ``paramFemale.isAnal`` and each clip's ``m_LoopTime``
FLOW_FILE = BAKE_H_DIR / "flow.json"


@lru_cache(maxsize=1)
def _flow_data() -> dict:
    from .hlist import load_flow
    return load_flow(FLOW_FILE)


def row_is_anal(mode: int, action_id: int) -> bool | None:
    """The row's ``paramFemale.isAnal`` (whether it offers the anal inserts), or None when ``flow.json`` lacks it."""
    row = (_flow_data().get("rows") or {}).get(f"{int(mode)}/{int(action_id)}")
    return None if row is None else bool(row.get("is_anal"))


def row_siru_paste(mode: int, action_id: int, woman: int = 0) -> dict | None:
    """The paste list ``ChangeAnimator`` loads for the row's woman *woman* (``fileSiruPaste``): ``{}`` when the row
    names none, None when ``flow.json`` lacks the row or the list it names."""
    data = _flow_data()
    row = (data.get("rows") or {}).get(f"{int(mode)}/{int(action_id)}")
    if row is None:
        return None
    files = row.get("siru") or ["", ""]
    name = files[woman] if woman < len(files) else ""
    if not name:
        return {}
    return (data.get("siru") or {}).get(name)


def row_metaball(mode: int, action_id: int) -> list | None:
    """The shot's emitters ``ChangeAnimator`` loads for the row (``fileMetaballCtrl``): ``[]`` when it names none,
    None when ``flow.json`` lacks the row or its list."""
    data = _flow_data()
    row = (data.get("rows") or {}).get(f"{int(mode)}/{int(action_id)}")
    if row is None:
        return None
    name = row.get("metaball") or ""
    if not name:
        return []
    return (data.get("metaball") or {}).get(name)


def fluid_assets() -> tuple:
    """``(bullets, nulls)`` of ``flow.json``: the bullet prefabs and the H scene's emitter / stop nulls."""
    data = _flow_data()
    return dict(data.get("bullets") or {}), dict(data.get("nulls") or {})


#: where ``flow.json``'s particle systems keep their maps (``particle_bake``)
PARTICLE_DIR = BAKE_H_DIR / "particles"


def fluid_particles() -> dict | None:
    """``HParticleCtrl``'s systems by id (``flow.json``'s ``particles``), or None when the bake has none."""
    data = _flow_data()
    parts = data.get("particles")
    return None if parts is None else dict(parts)


def clip_loop_time(name: str) -> bool | None:
    """The baked clip *name*'s ``m_LoopTime`` (off: it plays once and holds), or None when ``flow.json`` lacks it."""
    meta = _manifest_clip_index().get(name) or {}
    key = f"{str(meta.get('sex') or '')[:1]}/{meta.get('pack') or ''}/{meta.get('clip') or ''}"
    got = (_flow_data().get("loop_time") or {}).get(key)
    return None if got is None else bool(got)


def row_items(mode: int, action_id: int) -> list | None:
    """The item rows ``ChangeAnimator`` loads for a row (``paramFemale.fileItem``), ``[]`` for none.

    None when ``items.json`` is not baked though the row names a list (convert the H animations again).
    """
    global _ROW_ITEMS
    if _ROW_ITEMS is None:
        from .hlist import load_attach, load_items
        _ROW_ITEMS = {"names": dict(load_attach(ATTACH_FILE).get("items") or {}), "lists": load_items(ITEMS_FILE)}
    name = _ROW_ITEMS["names"].get(f"{int(mode)}/{int(action_id)}")
    if not name:
        return []
    rows = _ROW_ITEMS["lists"].get(name)
    return None if rows is None else list(rows)

# Stage preference inside one action: the loop the player spends most time in,
# then progressively duller fallbacks.  These are *state names* of the action's
# controller, so the list is data-checked (a stage that is not in ``states`` is
# simply skipped) rather than a guess about what a pack contains.
_STAGE_PREF = (
    "M_WLoop1", "WLoop1", "M_WLoop", "WLoop", "M_SLoop1", "SLoop1",
    "SLoop", "M_OLoop", "OLoop", "M_InsertIdle", "InsertIdle",
    "M_Idle", "Idle", "M_Insert", "Insert",
    # the peeping actions' In / Loop / Out (size trees, §7.371)
    "M_Loop", "Loop",
)

# Our ``body_stage`` spacing/camera key, chosen from the **action name** in Origin's
# list.  Origin has no such concept (world placement is baked into its clips), so
# this table is ours; it replaces ``layout_hint_for_pack``, which read the pack
# id — and 23 of the 28 penetration actions share pack ``02_00_00``, so that
# could never separate 正常位 from 立位.
_LAYOUT_BY_NAME = (
    ("騎乗位", "cowgirl"),
    ("駅弁", "standing"),
    ("立位", "standing"),
    ("壁", "standing"),
    ("フェンス", "standing"),
    ("側位", "spoon"),
    ("バック", "doggy"),
    ("後背位", "doggy"),
    ("プレス", "mating_press"),
    ("椅子", "lap"),
    ("正常位", "missionary"),
)

# 奉仕 and 愛撫 need their **own** tables, not more rows in the one above: those
# names contain the same furniture tokens (椅子バック愛撫, 壁バック愛撫,
# 椅子両手フェラ) and a single ordered list has to answer 壁 before it can
# answer 愛撫, so whichever family loses gets the other one's posture.  Both
# families used to fall through the sonyu table entirely and land on
# ``missionary`` — a woman on her back to give a フェラ, 45 rows of it.
#
# Inside a family the split is the **posture axis** the names already carry
# (§7.141 一): the act (フェラ / 手コキ / パイズリ) changes the hands and head,
# the prefix changes where the two bodies are, and only the latter is a layout.
_LAYOUT_HOUSHI = (
    ("立ち", "service_stand"),
    ("プール", "service_stand"),
    ("椅子", "service_chair"),
    ("ベンチ", "service_chair"),
)
_LAYOUT_AIBU = (
    ("仰向け", "caress_lie"),
    ("四つん這い", "caress_bend"),
    ("バック", "caress_bend"),
    ("座り", "caress_sit"),
    ("しゃがみ", "caress_sit"),
)

# Which female null the shaft is aimed at, by **act**.  One posture serves
# several acts, so this is a separate table from the layout one — and Origin does
# the same, keying its ``dan_*`` rows by motion rather than by pack.
#
# Every one of the 34 奉仕 rows was matched to its ``dan_*`` list (§7.142 十二).
# The pattern has no exceptions: フェラ / 舐め → ``k_f_head_00``, パイズリ →
# ``k_f_spine03_03``, 挿入 → ``k_f_kokan_00``, and **every 亀頭いじり / 手コキ
# row has no list at all** — 9 of them (ids 0, 1, 2, 16, 17, 32, 33, 36, 52),
# so Origin genuinely does not aim the shaft there.  ``none`` is therefore what Origin
# does, not a gap in our data.
#
# ``コキ`` and ``いじり`` come **before** ``舐め`` because 立ち玉舐め手コキ
# contains both and Origin gives it no list: the hand is the act, the licking is
# decoration.  ``立ちオナホ手コキ`` is the one row that does aim, at
# ``dan_kotei`` — a **prop** null (the onahole), not a body one — and we have no
# such prop, so it stays ``none`` too.
_CONTACT_BY_NAME = (
    ("パイズリ", "breast"),
    ("コキ", "none"),
    ("いじり", "none"),
    ("フェラ", "oral"),
    ("咥え", "oral"),
    ("舐め", "oral"),
    ("イラマチオ", "oral"),
    ("愛撫", "none"),
    ("オナニー", "none"),
)


def layout_for_action(name: str, family: str = "") -> str:
    """``body_stage`` layout key for one menu row.

    *family* picks the table; without it the 挿入 one is used, which is what
    every caller had before the 奉仕 / 愛撫 tables existed.
    """
    name = name or ""
    # ``startswith``: the 3P families are ``houshi3P`` / ``sonyu3PMMF`` and want
    # the same table as their two-body parents — matching on equality left the
    # ten 3P 奉仕 rows (``Wフェラ``, ``イラマチオ``…) back on ``missionary``.
    family = family or ""
    if family.startswith("houshi"):
        for token, layout in _LAYOUT_HOUSHI:
            if token in name:
                return layout
        return "service_floor"       # the un-prefixed rows: she kneels, he sits
    if family.startswith("aibu"):
        for token, layout in _LAYOUT_AIBU:
            if token in name:
                return layout
        return "caress_stand"
    if family == "masturbation":
        return "caress_sit" if ("椅子" in name or "座り" in name) else "caress_stand"
    for token, layout in _LAYOUT_BY_NAME:
        if token in name:
            return layout
    return "missionary"


def contact_for_action(name: str) -> str:
    """Which ``body_stage`` contact an action makes: oral / breast / none / …

    Defaults to ``vaginal``: the 挿入 family is the one whose names carry no
    act token at all (正常位, 後背位 …), so falling through means penetration.
    """
    for token, contact in _CONTACT_BY_NAME:
        if token in (name or ""):
            return contact
    return "vaginal"


@lru_cache(maxsize=1)
def _actions() -> tuple:
    from .hlist import load_actions
    return tuple(load_actions(ACTIONS_FILE))


@lru_cache(maxsize=1)
def _pack_controller_count() -> dict:
    """``{(sex, pack): n_controllers}`` — how many actions share one pack.

    A v1 bake names clips ``origin_{sex}_{pack}_{stem}``, and a stem is a controller
    *state name*, which is **not unique inside a pack**: ``02_00_00`` holds
    2,665 clips under 481 distinct names (156 collide) across 23 controllers.
    The v1 extractor de-duplicated on the name, so for those packs the baked
    clip was whichever controller UnityPy happened to yield first.
    Rows that can only be answered by such an entry are flagged ``ambiguous``
    instead of pretending the clip belongs to the action that named it.
    """
    counts: dict = {}
    for a in _actions():
        for sex, role in (("f", a.get("female")), ("m", a.get("male"))):
            if not role or not role.get("pack"):
                continue
            counts.setdefault((sex, role["pack"]), set()).add(
                role.get("controller") or "")
    return {k: len(v) for k, v in counts.items()}


def role_sex(controller: str, field: str = "") -> str:
    """``"f"`` / ``"m"`` for a role, **from the controller name**.

    The list's third role is called ``pathFemale1`` in every 3P row, and on the
    MMF families that name is a lie: ``houshi3PMMF`` / ``sonyu3PMMF`` put the
    **second man** there (``khh3_m_01_01``, ``khs3_m_02_01``), while the FFM
    families put the second woman (``khh3_f_00_01``).  Trusting the field name
    sent the MMF third role to the *female* bundle, where its controller does
    not exist — so those six clips were **never baked** and the row could only
    ever stage two bodies (§7.136).

    ``CLAUDE.md`` 七: 不要照方法名／欄位名推功能.  The controller name is what
    the data says, and ``_f_`` / ``_m_`` is unambiguous across all 106 rows.
    """
    c = str(controller or "")
    if "_m_" in c or c.endswith("_m"):
        return "m"
    if "_f_" in c or c.endswith("_f"):
        return "f"
    # No marker in the name: fall back to the field, which is right for the
    # two roles that are not overloaded.
    return "m" if field == "male" else "f"


@lru_cache(maxsize=1)
def _controllers_by_pack() -> dict:
    """``{(sex, pack): {controller, …}}`` — every controller an action names.

    Both roles *and* their base controllers: ``pathFemaleBase`` holds the state
    machine, so a bake that skipped it would drop the states the runtime asks
    for on the base layer.
    """
    out: dict = {}
    for a in _actions():
        for key in ("female", "male", "female1"):
            role = a.get(key)
            if not role:
                continue
            sex = role_sex(role.get("controller"), key)
            for pk, ct in ((role.get("pack"), role.get("controller")),
                           (role.get("base_pack"), role.get("base_controller"))):
                if pk and ct:
                    out.setdefault((sex, pk), set()).add(ct)
    return out


@lru_cache(maxsize=1)
def _stage_index() -> dict:
    """``{(sex, pack, controller, stage): clip_name}`` from the manifest.

    Built from the recorded fields, never by splitting the name: a controller
    name and a state name both contain ``_`` (``khs_f_n18`` / ``L_M_OUT_A``),
    so ``origin_f_02_00_00_khs_f_n18_L_M_OUT_A`` has no parseable boundary.
    """
    out: dict = {}
    for name, meta in _manifest_clip_index().items():
        ctrl = meta.get("controller")
        if not ctrl:
            continue                      # v1 entry — no identity to key on
        sex = str(meta.get("sex") or "")
        out[(sex[0] if sex else "", meta.get("pack") or "", ctrl,
             str(meta.get("stage") or "")
             + other_size_suffix(str(meta.get("other_size") or "")))] = name
    return out


@lru_cache(maxsize=1)
def _controller_clips() -> dict:
    """``{(sex, pack, controller): (clip name, …)}`` -- :func:`_stage_index` grouped, names in stage order."""
    out: dict = {}
    for (sx, pk, ct, _st), nm in sorted(_stage_index().items()):
        out.setdefault((sx, pk, ct), []).append(nm)
    return {k: tuple(v) for k, v in out.items()}


def played_stage(sex: str, pack: str, controller: str, state: str) -> str | None:
    """The baked clip a controller's *state* starts from when ``Animator.Play`` gets it by name.

    Studio's ``HAnime_*`` rows name a **state** of the pack's controller
    (``OCIChar.LoadAnime`` → ``CharAnimeCtrl.Play``, E1a); the bake keeps its
    leaves.  A size tree is picked by its ``S_`` leaf (:meth:`HRuntime._sized_clip`
    blends the three) and a motion split by its ``1`` leaf, the ``motion`` 0 one
    (:func:`motion_pair` finds the other).  A state on the two-person grid
    (§7.374) has one leaf per other-woman size after it; any of them is a start
    — the runtime picks the cell.  ``None`` when the pack is not baked.
    """
    stages = {st: name for (s, p, c, st), name in _stage_index().items()
              if s == str(sex or "")[:1] and p == pack and c == controller}
    for cand in (f"S_{state}1", f"S_{state}", f"{state}1", state):
        if cand in stages:
            return stages[cand]
    # a grid leaf is keyed with the other woman's size after :data:`OTHER_SIZE_SEP` (``S_MLoop1@S``,
    # :func:`_stage_index`); looked for after ``_`` it was never found (12 Studio rows, §7.1133)
    grid = sorted(st for st in stages
                  if st.startswith((f"S_{state}1{OTHER_SIZE_SEP}", f"S_{state}{OTHER_SIZE_SEP}")))
    return stages[grid[0]] if grid else None


@lru_cache(maxsize=1)
def _manifest_layer_index() -> dict:
    """``{name: entry}`` of the synced-layer leaves (``manifest["layers"]``, D1d).

    Kept out of ``clips`` so nothing that lists, sizes or pairs clips — the
    menu, :func:`_stage_index`, :func:`size_variants` — ever sees a layer leaf
    as a state of its own.
    """
    if not BAKE_MANIFEST.is_file():
        return {}
    try:
        migrate_baked_names()
        man = json.loads(BAKE_MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        swallowed.note("origin.hanim._manifest_layer_index: man = json.loads(BAKE_MANIFEST.read_text(encodin", exc)
        return {}
    return dict(man.get("layers") or {})


@lru_cache(maxsize=1)
def _layer_index() -> dict:
    """``{(sex, served controller, base state): {layer: {...}}}`` from the bake.

    Each layer carries ``param`` (``Breast`` / ``Breast1``), ``motion_param``
    (``""`` when the tree has no motion split), ``mode`` and ``cells``
    ``{(motion, bust child): clip name}``.
    """
    out: dict = {}
    for name, e in _manifest_layer_index().items():
        sex = str(e.get("sex") or "")[:1]
        for ctrl in e.get("serves") or ():
            lay = out.setdefault((sex, ctrl, str(e.get("layer_state") or "")), {})
            info = lay.setdefault(int(e["layer"]), {
                "name": e.get("layer_name") or "", "mode": int(e.get("layer_mode") or 0),
                "param": e.get("layer_param") or "",
                "motion_param": e.get("motion_param") or "", "cells": {}})
            if e.get("defaults") is not None:
                info["defaults"] = frozenset(e["defaults"])     # an empty state writing defaults (D11)
                continue
            info["cells"][(int(e.get("motion") or 0), int(e.get("breast") or 0))] = name
    return out


def layers_for(name: str) -> tuple[str, dict] | None:
    """``(base state, {layer: info})`` the ``al_`` list turns on while *name* plays.

    *name* is an action (or base) clip; the list is keyed by **base** state
    (``AnimatorLayerCtrl.Proc`` gets what ``SetPlay`` was given), so the clip's
    stage is reduced much as :func:`base_state_for` does — as is, without its
    size letter, without the trailing motion digit — and last to the one base
    state that is the same name behind a size letter of its own (the base's
    ``M_OUT_Start`` holds the ``S_``/``M_``/``L_OUT_Start`` leaves).  ``None``
    when the controller has no list or this state turns nothing on.
    """
    meta = _manifest_clip_index().get(name) or {}
    ctrl = str(meta.get("controller") or "")
    stage = str(meta.get("stage") or "")
    if not ctrl or not stage:
        return None
    sex = str(meta.get("sex") or "")[:1]
    idx = _layer_index()
    have = {st for (sx, c, st) in idx if sx == sex and c == ctrl}
    if not have:
        return None
    _, rest = _strength_of(stage)
    bare = re.sub(r"\d+$", "", rest)
    for cand in (stage, rest, bare):
        if cand in have:
            return cand, idx[(sex, ctrl, cand)]
    alias = [st for st in have if _strength_of(st)[0] and _strength_of(st)[1] in (rest, bare)]
    if len(alias) == 1:
        return alias[0], idx[(sex, ctrl, alias[0])]
    return None


def breast_pair(value: float) -> tuple[int, int, float]:
    """``(lower child, upper child, weight of upper)`` of a bust tree at *value*.

    Unity's 1D blend over :data:`hlist.BREAST_THRESHOLDS`, clamped at the ends
    like :func:`size_pair` (a SliderUnlocker value past 0‥1 plays the end
    child).
    """
    from .hlist import BREAST_THRESHOLDS as t

    v = float(value)
    if v <= t[0]:
        return 0, 0, 0.0
    if v >= t[-1]:
        return len(t) - 1, len(t) - 1, 0.0
    for i in range(len(t) - 1):
        if t[i] <= v <= t[i + 1]:
            return i, i + 1, (v - t[i]) / (t[i + 1] - t[i])
    return 0, 0, 0.0


@lru_cache(maxsize=1)
def controller_aware_bake() -> bool:
    """True when baked entries carry the controller, not just pack+stem.

    Until they do, every action sharing a pack resolves to the *same* clip, so
    the shortcuts must keep their old pack-based pick: it is wrong about which
    position it plays, but at least it plays three different ones.  A menu row
    is still worth grouping by action — that part is right either way.
    """
    return bool(_stage_index())


def _pick_stage(states: list[str], index: dict, sex: str, pack: str,
                controller: str = "") -> tuple[str | None, str | None]:
    """``(stage, baked_clip_name)`` for one role — first preference that baked."""
    have = set(states or ())
    stages = _stage_index()

    def candidates(stage: str):
        # Controller-qualified entry first — that is the only one that actually
        # identifies this action's clip inside a shared pack.
        if controller:
            hit = stages.get((sex, pack, controller, stage))
            if hit:
                yield hit
            # a two-person grid's row plays at the other woman's middle size
            hit = stages.get((sex, pack, controller,
                              stage + other_size_suffix(SIZE_PREFIXES[1])))
            if hit:
                yield hit
        yield f"{CLIP_PREFIX}{sex}_{pack}_{stage}"

    for stage in _STAGE_PREF:
        if stage not in have:
            continue
        for name in candidates(stage):
            if name in index:
                return stage, name
    # No preferred stage baked: take any state that did bake, so a partially
    # baked install still plays something rather than silently showing nothing.
    # The middle size first: of a size tree's three leaves (§7.371) it is the
    # one the blend plays at the default height, where alphabetical order
    # handed the row its tall-woman clip.
    for stage in sorted(have, key=lambda s: (not s.startswith("M_"), s)):
        for name in candidates(stage):
            if name in index:
                return stage, name
    return None, None


def moves_tongue(clip) -> bool:
    """Whether *clip* turns the tongue chain (``cf_j_tang*``): the one part of a man's head an H row uses.

    Read off the clip itself, not the row's name: on this install the man's clip of the two 3P
    ``騎乗位クンニ`` rows turns it 34°, and no other man's clip turns it at all (§7.1168).
    """
    if clip is None:
        return False
    for bone, curve in getattr(clip, "curves", {}).items():
        if "tang" not in bone.lower() or curve.rots is None or len(curve.rots) < 2:
            continue
        q = np.asarray(curve.rots, dtype=np.float64)
        if float(np.abs(q @ q[0]).min()) < 1.0 - 1e-6:
            return True
    return False


@lru_cache(maxsize=1)
def _move_table() -> dict:
    from .hlist import load_moves
    return load_moves(ACTIONS_FILE)


def _row_moves(action: dict) -> tuple:
    """The row's ``(category, move)`` pairs (:attr:`HMenuEntry.moves`); a list the table lacks is reported."""
    table = _move_table()
    out = []
    for category, name in zip(action.get("categories") or (), action.get("moves") or ()):
        if not name:
            continue
        if name not in table:
            print(f"[hanim] {action.get('name')}: move list {name} is not in actions.json, the row stands unmoved")
            continue
        out.append((int(category), tuple(float(v) for v in table[name])))
    return tuple(out)


@lru_cache(maxsize=1)
def list_menu_entries() -> tuple:
    """One row per **H action** from Origin's list; falls back to pack×stem rows.

    The action list (``assets_baked/h_anim/actions.json``, written by
    ``hlist.bake_actions``) is our format — the runtime never reads ``h/list``
    bundles.  Without it we still list the baked clips the old way so a cache
    baked before this existed keeps working.
    """
    index = _manifest_clip_index()
    if not index:
        return tuple()
    actions = _actions()
    if actions:
        shared = _pack_controller_count()
        # How many actions name the same controller — see ``shared_motion``.
        ctrl_uses: dict = {}
        for a in actions:
            for sx, key in (("f", "female"), ("m", "male")):
                c = (a.get(key) or {}).get("controller")
                if c:
                    ctrl_uses[(sx, c)] = ctrl_uses.get((sx, c), 0) + 1
        entries: list[HMenuEntry] = []
        for a in actions:
            f = a.get("female") or {}
            m = a.get("male") or {}
            pack_f, pack_m = f.get("pack", ""), m.get("pack", "")
            stage_f, name_f = _pick_stage(f.get("states"), index, "f", pack_f,
                                          f.get("controller") or "")
            stage_m, name_m = _pick_stage(m.get("states"), index, "m", pack_m,
                                          m.get("controller") or "")
            if not name_f and not name_m:
                continue          # nothing baked for this action yet
            fam = str(a.get("mode_name") or "other")
            fam_lab = _FAMILY_LABEL.get(fam, fam)
            action_name = str(a.get("name") or "")
            dual = "雙" if (name_f and name_m) else ("女" if name_f else "男")
            # Ambiguous only when the row had to fall back to a v1 entry (no
            # controller recorded) *and* more than one action shares that pack —
            # that is exactly the case where the clip cannot be attributed.
            amb = False
            for sx, pk, nm in (("f", pack_f, name_f), ("m", pack_m, name_m)):
                if not nm:
                    continue
                if index.get(nm, {}).get("controller"):
                    continue
                if shared.get((sx, pk), 0) > 1:
                    amb = True
            ctrl_f = str(f.get("controller") or "")
            ctrl_m = str(m.get("controller") or "")
            t = a.get("female1") or {}
            ctrl_t = str(t.get("controller") or "")
            t_sex = role_sex(ctrl_t, "female1") if ctrl_t else ""
            stage_t, name_t = (_pick_stage(t.get("states"), index, t_sex,
                                           t.get("pack", ""), ctrl_t)
                               if ctrl_t else (None, None))
            shared_motion = (ctrl_uses.get(("f", ctrl_f), 0) > 1
                             or ctrl_uses.get(("m", ctrl_m), 0) > 1)
            needs_third = bool(ctrl_t)
            entries.append(HMenuEntry(
                key=f"{ROW_PREFIX}{a.get('mode')}_{a.get('id')}",
                label=(f"{fam_lab} · {action_name}（{dual}"
                       + ("·共用包" if amb else "")
                       + ("·同動作" if shared_motion else "")
                       + ("·需三人" if needs_third else "") + "）"),
                family=fam,
                family_label=fam_lab,
                pack=pack_f or pack_m,
                stem=stage_f or stage_m or "",
                name_f=name_f,
                name_m=name_m,
                layout=layout_for_action(action_name, fam),
                contact=contact_for_action(action_name),
                action_id=int(a.get("id", -1)),
                action_name=action_name,
                controller_f=ctrl_f,
                controller_m=ctrl_m,
                stages=tuple(sorted(set(f.get("states") or m.get("states") or ()))),
                ambiguous=amb,
                shared_motion=shared_motion,
                mode=int(a.get("mode", -1)),
                categories=tuple(int(c) for c in a.get("categories") or ()),
                moves=_row_moves(a),
                needs_third=needs_third,
                name_t=name_t,
                third_sex=t_sex if name_t else "",
                controller_t=ctrl_t,
                bases=tuple(str(r.get("base_controller") or "") for r in (f, m, t)),
                role_speeds=tuple(tuple((str(s), str(v[0]), float(v[1]))
                                        for s, v in sorted((r.get("base_speeds") or {}).items()))
                                  for r in (f, m, t)),
                base_states=tuple(sorted(set(
                    f.get("base_states") or m.get("base_states") or ()))),
                base_speeds=tuple(
                    (str(s), str(v[0]), float(v[1]))
                    for s, v in sorted((f.get("base_speeds")
                                        or m.get("base_speeds") or {}).items())),
                base_clip_states=tuple(sorted(
                    (str(c), str(st)) for c, st in (f.get("base_clip_states")
                                                  or m.get("base_clip_states")
                                                  or {}).items())),
            ))
        if entries:
            return tuple(entries)
    return _legacy_menu_entries(index)


def _legacy_menu_entries(index: dict) -> tuple:
    """Pre-action-list rows: one per baked ``pack/stem``.

    Kept only for caches baked before ``actions.json`` existed — these rows are
    *stages*, so the same action appears dozens of times.
    """
    by_ps: dict[tuple[str, str], dict[str, str]] = {}
    for name in index:
        parts = name.split("_")
        if len(parts) < 6 or parts[0] != "origin":
            continue
        sex = parts[1]
        if sex not in ("f", "m"):
            continue
        pack = f"{parts[2]}_{parts[3]}_{parts[4]}"
        stem = "_".join(parts[5:])
        by_ps.setdefault((pack, stem), {})[sex] = name

    entries: list[HMenuEntry] = []
    for (pack, stem), sexes in sorted(by_ps.items(), key=lambda x: (x[0][0], x[0][1])):
        name_f = sexes.get("f")
        name_m = sexes.get("m")
        # Prefer listing under female; if male-only (rare), still list.
        key = name_f or name_m
        if not key:
            continue
        fam, fam_lab = family_for_pack(pack)
        dual = "雙" if (name_f and name_m) else ("女" if name_f else "男")
        label = f"{fam_lab} · {pack} · {_stem_label(stem)}（{dual}）"
        entries.append(HMenuEntry(
            key=key,
            label=label,
            family=fam,
            family_label=fam_lab,
            pack=pack,
            stem=stem,
            name_f=name_f,
            name_m=name_m,
            layout=layout_hint_for_pack(pack),
        ))
    return tuple(entries)


def menu_entry_by_key(key: str) -> "HMenuEntry | None":
    """The menu row whose key or clip name is *key* (an earlier version's names read as this one's)."""
    if not key:
        return None
    key = current_name(key)
    for e in list_menu_entries():
        if e.key == key or e.name_f == key or e.name_m == key:
            return e
    return None


def base_state_for(stem: str, base_states, speed_states,
                   clip_states=None, clip: str = "") -> str:
    """The base **state** a clip stem plays under, or ``""``.

    *clip* is the clip's own ``m_Name`` (the manifest's ``clip``), which is
    what the override matches and what *clip_states* is keyed by.  The stem
    is the action controller's state and is usually the same name, but not
    always: 正常位's state ``M_M_OUT_A`` plays the clip ``M_OUT_A`` (§7.507).

    Origin runs two controllers: the **base** one holds the phase machine and the
    action one only overrides its clips by name, so what is playing is always
    a base state — the one ``breath_*`` / ``HFaceList`` and the ``al_`` lists
    are keyed by (``AnimatorStateInfo.IsName``) and whose ``m_Speed`` binding
    sets the rate.  One answer for all of them.

    *speed_states* are the base controller's state names
    (``HMenuEntry.base_speeds``); *base_states* are its clip stems, where a
    size-tree state is listed as its three ``S_`` / ``M_`` / ``L_`` leaves
    (``hlist.controller_stems``), and an action controller's stems carry the
    size prefix of the clip they override (``M_WLoop1`` plays ``WLoop``).  In
    order: the stem is a leaf of a size-tree state behind its prefix; else it
    is a state itself; else it is the state behind its prefix.  The first test
    is what separates the caress base's two ``M_Idle``s: the stem ``M_Idle``
    sits beside ``S_Idle`` / ``L_Idle`` and is ``Idle`` at size M (unbound,
    ×0.7), while ``L_M_Idle`` is the state ``M_Idle`` (bound to
    ``speedBody``).  The trailing strength digit is tried stripped too.

    Matching the stem against the **leaves** before the states is the point:
    since §7.371 made *base_states* the leaves, a rule that tried the stem as
    is first answered ``L_Idle`` for a clip of ``Idle``, and the face and
    breath lookups missed on 2/3 of the stages (§7.405).
    """
    states = set(speed_states or ())
    leaves = set(base_states or ())
    if not stem or not states:
        return ""
    # *clip_states* (``HMenuEntry.base_clip_states``) is the answer itself:
    # the base's trees say which state each clip name plays under, which is
    # how ``L_OUT_Loop`` reaches ``M_OUT_Loop`` (§7.405).  The spelling rules
    # below are for an ``actions.json`` written before it was read.
    got = dict(clip_states or ()).get(clip or stem)
    if got in states:
        return got
    for cand in dict.fromkeys((stem, re.sub(r"\d+$", "", stem))):
        pref, rest = _strength_of(cand)
        if pref and rest in states and all(p + rest in leaves for p in SIZE_PREFIXES):
            return rest
        if cand in states:
            return cand
        if pref and rest in states:
            return rest
    return ""


#: Caress area → the base state ``HAibu.Proc`` plays while that area is worked
#: (§7.397).  It picks by ``flags.click`` (``HFlag.ClickKind``): ``mouth`` (15)
#: → ``K_Touch`` / ``K_Loop``, ``muneL`` / ``muneR`` (16/17) → ``M_``,
#: ``kokan`` (18) → ``A_``, ``anal`` / ``siriL`` / ``siriR`` (19‥21) → ``S_``;
#: the ``*_Touch`` state leads into the loop named here, and those four are the
#: caress states the base binds to ``speedBody``.  With no area it is ``Idle``.
CARESS_AREAS = {"mouth": "K_Loop", "mune": "M_Idle", "kokan": "A_Idle",
                "siri": "S_Idle"}
#: The caress climax loop (``Orgasm_Start`` → ``Orgasm_Loop`` → ``Orgasm_A``).
CARESS_CLIMAX = "Orgasm_Loop"


def same_controller_clip(name: str, state: str) -> str | None:
    """The clip of *state* in *name*'s own controller, at *name*'s size.

    ``None`` when that controller has no such state — the caller keeps what it
    had rather than playing a neighbour's.
    """
    meta = _manifest_clip_index().get(name) or {}
    ctrl = meta.get("controller")
    if not ctrl:
        return None
    sex = str(meta.get("sex") or "")[:1]
    size, _ = _strength_of(str(meta.get("stage") or ""))
    other = other_size_suffix(str(meta.get("other_size") or ""))
    idx = _stage_index()
    for st in dict.fromkeys((size + state + other, size + state, state)):
        hit = idx.get((sex, meta.get("pack") or "", ctrl, st))
        if hit:
            return hit
    return None


def state_clip_name(name: str, state: str, base_states, speed_states,
                    clip_states=None) -> str | None:
    """The clip *name*'s own controller plays for the **base** state *state*, or ``None``.

    The action procs ``SetPlay`` base states (``M_IN_Start``, ``OUT_A``...) and
    the action controller overrides them by clip name, so the state a stage
    plays under is :func:`base_state_for` of it -- 正常位's stage ``M_M_OUT_A``
    is the clip ``M_OUT_A``, which plays as ``OUT_A`` (§7.507).  Of the stages
    of the same sex, pack and controller (and the other woman's size, on a
    two-person grid) that reduce to *state*, the one at *name*'s own size wins,
    then the first by name (a motion-0 leaf, ``*1``, before its ``*2``).
    ``None`` when the controller overrides no clip for that state.
    """
    index = _manifest_clip_index()
    meta = index.get(name) or {}
    ctrl = meta.get("controller")
    if not ctrl or not state:
        return None
    sex = str(meta.get("sex") or "")[:1]
    pack = str(meta.get("pack") or "")
    size, _ = _strength_of(str(meta.get("stage") or ""))
    other = other_size_suffix(str(meta.get("other_size") or ""))
    best = None
    for nm in _controller_clips().get((sex, pack, ctrl), ()):
        m2 = index.get(nm) or {}
        if other_size_suffix(str(m2.get("other_size") or "")) != other:
            continue
        stage = str(m2.get("stage") or "")
        if base_state_for(stage, base_states, speed_states, clip_states,
                          str(m2.get("clip") or "")) != state:
            continue
        rank = (0 if size and stage.startswith(size) else 1, stage)
        if best is None or rank < best[0]:
            best = (rank, nm)
    return best[1] if best else None


#: The loop states whose base trees split on ``motion``: motion 0 is ``*1``, motion 1 is ``*2``.
#: ``MLoop`` (lesbian, masturbation, D1d) and the sonyu / houshi ``WLoop`` / ``SLoop`` and their
#: ``A_`` versions — read off ``khs_f_base``, ``khs_m_base`` and ``khh_f_base`` with
#: ``hlist.controller_motion_trees`` (§7.1178).
MOTION_LOOPS = ("MLoop1", "WLoop1", "SLoop1")


def motion_pair(name: str, base: str = "") -> tuple[str, str] | None:
    """``(motion-1 clip, motion parameter)`` when *name* is the motion-0 leaf of a loop that splits on motion.

    Such a loop is the same cell twice, ``*Loop1`` at ``motion`` 0 and ``*Loop2`` at 1
    (``hlist.controller_motion_trees``; :data:`MOTION_LOOPS`).  The parameter comes from the bake.
    An action state that overrides its base's clips by name (``L_MLoop1`` over ``kht_f_base``,
    ``S_WLoop1`` over ``khs_f_base``) carries none: it plays through the tree of the role's base
    controller *base* (``path*Base``), and the same state there names the parameter — the 3P rows'
    two women run ``khs3_f_base1`` on ``motion`` and ``khs3_f_base2`` on ``motion1``
    (:func:`base_motion_param`).  Without a base that says, it is ``motion``.  ``None`` for anything
    else, or when the pair is not baked.
    """
    meta = _manifest_clip_index().get(name) or {}
    stage = str(meta.get("stage") or "")
    ctrl = meta.get("controller")
    if not ctrl or not stage.endswith(MOTION_LOOPS):
        return None
    sex = str(meta.get("sex") or "")[:1]
    other = other_size_suffix(str(meta.get("other_size") or ""))
    key = (sex, meta.get("pack") or "", ctrl, stage[:-1] + "2" + other)
    hit = _stage_index().get(key)
    if not hit:
        return None
    param = meta.get("motion_param") or (base_motion_param(sex, base, stage + other) if base else None)
    return hit, str(param or "motion")


@lru_cache(maxsize=1)
def _base_motion_params() -> dict:
    """``{(sex, controller, stage + other size): motion parameter}`` of every baked clip that records one."""
    out: dict = {}
    for meta in _manifest_clip_index().values():
        param, ctrl = meta.get("motion_param"), meta.get("controller")
        if param and ctrl:
            out[(str(meta.get("sex") or "")[:1], ctrl,
                 str(meta.get("stage") or "") + other_size_suffix(str(meta.get("other_size") or "")))] = str(param)
    return out


def base_motion_param(sex: str, base: str, stage: str) -> str | None:
    """The parameter base controller *base*'s state *stage* (with its other-size suffix) splits on, or None."""
    return _base_motion_params().get((str(sex)[:1], base, stage))


def _stem_motion_kind(stem: str) -> str:
    """Classify a clip stem into a play-phase bucket."""
    s = (stem or "").lower()
    if "insertidle" in s:
        return "InsertIdle"
    if "insert" in s:
        return "Insert"
    if "oloop" in s:
        return "OLoop"
    if "sloop" in s:
        return "SLoop"
    if "wloop" in s:
        return "WLoop"
    if "out_" in s or s.startswith("out") or "_out" in s:
        return "OUT"
    if "in_loop" in s or "in_start" in s or s.startswith("in_"):
        return "IN"
    if _strength_of(stem or "")[1].lower() == "loop":
        # the peeping rows' loop (``HPeeping.Proc``: ``In`` → ``Loop`` → ``Out``); a caress
        # area's ``K_Loop`` is not one
        return "Loop"
    if "idle" in s:
        return "Idle"
    return "Other"


# **The one-letter prefix is the body size, not a strength** (§7.362).  The base
# controller's phase states are 1D blend trees over the animator float
# ``height`` with thresholds 0 / 0.5 / 1 and three children -- ``S_WLoop``,
# ``M_WLoop``, ``L_WLoop`` (``kht_f_base``, ``khs_f_base`` …) -- and the action
# controller supplies those three clips.  ``HActionBase.SetDefaultAnimatorFloat``
# writes ``height`` = ``female.GetShapeBodyValue(0)``, the woman's height
# slider, to the woman, the man, the second man and the item; the second woman
# gets her own.  So a short woman plays mostly ``S_`` and a tall one ``L_``, and
# the man follows *her* size.  Measured on 正常位 ``WLoop1``: the S and L
# versions differ at the hands -- the man's hand IK targets by 6 cm, his
# forearms by 13° -- and hardly at all at the pelvis.
from .hlist import (SIZE_TREE_PREFIXES as SIZE_PREFIXES,  # noqa: E402
                    SIZE_TREE_THRESHOLDS as SIZE_THRESHOLDS)


def size_pair(height: float) -> tuple[str, str, float]:
    """``(lower, upper, weight of upper)`` of Unity's 1D blend at *height*.

    Linear between the two neighbouring thresholds, clamped at the ends -- a
    SliderUnlocker value past 0..1 plays the end clip, as the blend tree does.
    """
    h = float(height)
    t = SIZE_THRESHOLDS
    if h <= t[0]:
        return SIZE_PREFIXES[0], SIZE_PREFIXES[0], 0.0
    if h >= t[-1]:
        return SIZE_PREFIXES[-1], SIZE_PREFIXES[-1], 0.0
    for i in range(len(t) - 1):
        if t[i] <= h <= t[i + 1]:
            return (SIZE_PREFIXES[i], SIZE_PREFIXES[i + 1],
                    (h - t[i]) / (t[i + 1] - t[i]))
    return SIZE_PREFIXES[1], SIZE_PREFIXES[1], 0.0


def size_variants(name: str) -> dict[str, str]:
    """``{"S_": clip, "M_": clip, "L_": clip}`` for the state *name* plays.

    Same sex, pack, controller and stage, with the size letter swapped -- read
    off the manifest's recorded fields (never by splitting the name, see
    :func:`_stage_index`).  Empty when the entry has no controller (a v1 bake)
    or its stage carries no size letter.
    """
    meta = _manifest_clip_index().get(name) or {}
    ctrl = str(meta.get("controller") or "")
    stage = str(meta.get("stage") or "")
    if not ctrl or stage[:2] not in SIZE_PREFIXES:
        return {}
    sex = str(meta.get("sex") or "")[:1]
    pack = str(meta.get("pack") or "")
    other = other_size_suffix(str(meta.get("other_size") or ""))
    index = _stage_index()
    out = {}
    for pref in SIZE_PREFIXES:
        hit = index.get((sex, pack, ctrl, pref + stage[2:] + other))
        if hit:
            out[pref] = hit
    return out


def other_size_variants(name: str) -> dict[str, str]:
    """``{"S_": clip, "M_": clip, "L_": clip}`` across **the other woman's** size.

    The second axis of the two-person grid (§7.374): same state and own size,
    the ``other_size`` swapped.  Empty for every clip that is not on a grid.
    """
    meta = _manifest_clip_index().get(name) or {}
    ctrl = str(meta.get("controller") or "")
    if not ctrl or not meta.get("other_size"):
        return {}
    key = (str(meta.get("sex") or "")[:1], str(meta.get("pack") or ""), ctrl,
           str(meta.get("stage") or ""))
    index = _stage_index()
    out = {}
    for pref in SIZE_PREFIXES:
        hit = index.get(key[:3] + (key[3] + other_size_suffix(pref),))
        if hit:
            out[pref] = hit
    return out


def _blend_keys(bone: str, a, b, w: float):
    """Two curves on the **same** time grid, blended key by key.

    Rotations by slerp per key — the two versions differ by up to 13° on the
    sampled action, so it stays a slerp rather than an nlerp (§7.362) — done
    over all the keys at once by ``clip._slerp_rows``, the row form of
    ``math3d.quat_slerp`` (same branches, agrees to rounding; §7.421).
    """
    from ..anim.clip import BoneCurve, _slerp_rows
    rots = None
    if a.rots is not None and b.rots is not None:
        rots = _slerp_rows(np.asarray(a.rots, dtype=np.float64),
                           np.asarray(b.rots, dtype=np.float64), w)
    elif a.rots is not None or b.rots is not None:
        rots = (a.rots if w < 0.5 else b.rots)
    def mix(x, y):
        if x is None and y is None:
            return None
        if x is None or y is None:
            return x if x is not None else y
        return x * (1.0 - w) + y * w
    return BoneCurve(bone, a.times.copy(), rots,
                     mix(a.poss, b.poss), mix(a.scales, b.scales))


def blend_size_clip(lo, hi, w: float, *, name: str | None = None):
    """The two body-size versions of one state, blended once into one clip.

    Unity blends them every frame, but the parameter it blends by (``height``)
    is **constant for a character**, so blending per frame is 5x the work for
    the same pose: sampling two 161-bone clips and mixing came to 13.3 ms a
    role against 2.6 ms for one (§7.362).  Blending the *curves* instead gives
    a clip that samples like any other.

    Keys land on the union of the two time grids, which is exact for the
    positions and scales (linear between keys) and is the two clips' own grid
    in practice — the S/M/L versions of a state are authored on the same one.
    The name stays *lo*'s or the caller's, because the stage it encodes is what
    ``_clip_stem`` reads downstream.
    """
    from ..anim.clip import AnimationClip
    from ..core import math3d as m3
    w = float(min(max(w, 0.0), 1.0))
    if hi is None or w <= 0.0:
        return lo
    if lo is None or w >= 1.0:
        return hi
    out = AnimationClip(name or getattr(lo, "name", ""),
                        duration=float(lo.duration) * (1.0 - w)
                        + float(hi.duration) * w,
                        loop=bool(getattr(lo, "loop", True)))
    out.tags = set(getattr(lo, "tags", set()))
    out.events = list(getattr(lo, "events", []))
    for bone in set(lo.curves) | set(hi.curves):
        a, b = lo.curves.get(bone), hi.curves.get(bone)
        if a is None or b is None:
            src = a or b
            out.curves[bone] = src
            continue
        if (len(a.times) == len(b.times)
                and np.array_equal(a.times, b.times)):
            # The common case and the cheap one: the two versions of a state are
            # authored on one grid (161 of 161 bones, 41 keys each, on the
            # sampled action), so the keys blend as arrays.  Going through
            # ``sample`` for every key instead cost **751 ms** a clip, which is
            # a visible hitch at a phase switch; this is 24 ms.
            out.curves[bone] = _blend_keys(bone, a, b, w)
            continue
        times = np.unique(np.concatenate([a.times, b.times]))
        rots = poss = scales = None
        if a.rots is not None or b.rots is not None:
            rots = np.array([m3.quat_slerp(a.sample(float(t)).get(
                "rot", m3.quat_identity()),
                b.sample(float(t)).get("rot", m3.quat_identity()), w)
                for t in times], dtype=np.float64)
        if a.poss is not None or b.poss is not None:
            poss = np.array([a.sample(float(t)).get("pos", np.zeros(3))
                             * (1.0 - w)
                             + b.sample(float(t)).get("pos", np.zeros(3)) * w
                             for t in times], dtype=np.float64)
        if a.scales is not None or b.scales is not None:
            scales = np.array([a.sample(float(t)).get("scale", np.ones(3))
                               * (1.0 - w)
                               + b.sample(float(t)).get("scale", np.ones(3)) * w
                               for t in times], dtype=np.float64)
        out.set_curve(bone, times, rots, poss, scales)
    out.duration = float(lo.duration) * (1.0 - w) + float(hi.duration) * w
    return out


def _strength_of(stem: str) -> tuple[str, str]:
    """``\"L_WLoop1\"`` → ``(\"L_\", \"WLoop1\")``; no prefix → ``(\"\", stem)``.

    Named for what it was once taken to be; the one-letter prefix is the body
    size (see :data:`SIZE_PREFIXES`).
    """
    for pref in ("SS_", "LL_", "MM_", "ML_", "SM_", "LS_", "LM_",
                 "S_", "M_", "L_"):
        if stem.startswith(pref):
            return pref, stem[len(pref):]
    return "", stem


def sibling_clip_names(name: str) -> dict[str, str]:
    """Phase → clip name for the **same controller**, keyed by motion kind.

    ``{\"WLoop\": \"…L_WLoop1\", \"Insert\": \"…L_Insert\", …}`` so Free H can
    play Insert on entry and WLoop while connected.

    The siblings of a stage are the other states of *its own controller*, which
    is what the manifest now records.  §7.110 derived them from the pack, and
    §7.111 showed why that could not hold: 23 of the 28 penetration actions
    live in ``02_00_00``, so "same pack" pulled in the stages of 22 other
    positions and which one you got was arbitrary.  A v1 entry has no
    controller, so those keep the old pack-wide behaviour below.
    """
    if not name or not name.startswith(CLIP_PREFIX):
        return {}
    meta = _manifest_clip_index().get(name) or {}
    if meta.get("controller"):
        sex = str(meta.get("sex") or "")[:1]
        pack = str(meta.get("pack") or "")
        ctrl = str(meta.get("controller") or "")
        stem = str(meta.get("stage") or "")
        strength, rest = _strength_of(stem)
        # A penetration controller holds the anal states beside the vaginal ones
        # (``A_InsertIdle``, ``A_SLoop`` …) and the game plays the ``A_`` one only
        # while ``flags.isAnalPlay`` (``HSonyu.LoopProc`` IL_0176–0195, the same in
        # ``H3PSonyu`` / ``H3PDarkSonyu``): the siblings stay on the side the row's
        # own stage is on.  Not the idles: a caress controller's ``A_Idle`` is an area.
        anal = rest.startswith("A_")
        # (kind → (rank, name)); rank 0 = same strength letter as the stage the
        # player is on, so a "強" loop does not answer with the "弱" Insert.
        best: dict[str, tuple[int, str, str]] = {}
        for (sx, pk, ct, st), nm in sorted(_stage_index().items()):
            if (sx, pk, ct) != (sex, pack, ctrl):
                continue
            kind = _stem_motion_kind(st)
            if kind != "Idle" and _strength_of(st)[1].startswith("A_") != anal:
                continue
            rank = 0 if (strength and st.startswith(strength)) else 1
            cur = best.get(kind)
            if cur is None or rank < cur[0]:
                best[kind] = (rank, st, nm)
        out = {k: v[2] for k, v in best.items()}
        out[_stem_motion_kind(stem)] = name
        return out
    parts = name.split("_")
    if len(parts) < 6:
        return {}
    sex, pack = parts[1], f"{parts[2]}_{parts[3]}_{parts[4]}"
    stem = "_".join(parts[5:])
    # Strength token: leading S_/M_/L_/ML_/… before the motion word
    strength = ""
    for pref in ("SS_", "LL_", "MM_", "ML_", "SM_", "LS_", "LM_",
                 "S_", "M_", "L_"):
        if stem.startswith(pref):
            strength = pref
            break
    index = _manifest_clip_index()
    out: dict[str, str] = {}
    prefix = f"{CLIP_PREFIX}{sex}_{pack}_{strength}"
    for n in index:
        if not n.startswith(prefix):
            # also allow same pack without strength lock when stem has none
            if strength:
                continue
            if not n.startswith(f"{CLIP_PREFIX}{sex}_{pack}_"):
                continue
        st = _clip_stem(n)
        kind = _stem_motion_kind(st)
        # Prefer exact strength match; first win keeps WLoop1 over WLoop2 etc.
        if kind not in out:
            out[kind] = n
        elif strength and st.startswith(strength) and not _clip_stem(out[kind]).startswith(strength):
            out[kind] = n
    # Always include the source name under its kind
    out[_stem_motion_kind(stem)] = name
    return out


def phase_clip_name(siblings: dict[str, str], phase: str,
                    pace: float = 0.5) -> str | None:
    """Pick which sibling to play for an encounter *phase*."""
    if not siblings:
        return None
    pace = float(pace)

    def _first(*keys: str) -> str | None:
        for k in keys:
            if k in siblings:
                return siblings[k]
        return None

    ph = (phase or "connected").lower()
    if ph in ("setup", "foreplay"):
        return _first("Idle", "InsertIdle", "WLoop", "Insert")
    if ph == "entry":
        return _first("Insert", "InsertIdle", "WLoop", "Idle")
    if ph == "climax":
        return _first("OLoop", "SLoop", "WLoop", "Insert")
    if ph == "afterglow":
        return _first("OUT", "Idle", "WLoop", "InsertIdle")
    # connected — pace chooses weak vs strong loop; a peeping row loops its ``Loop``
    if pace >= 0.62:
        return _first("SLoop", "WLoop", "OLoop", "Loop", "Insert")
    return _first("WLoop", "SLoop", "Loop", "Insert", "Idle")


def menu_families() -> list[tuple[str, str]]:
    """``[(family_key, zh_label), ...]`` present in the bake."""
    seen = {}
    for e in list_menu_entries():
        seen[e.family] = e.family_label
    order = ["sonyu", "houshi", "aibu", "masturbation", "peeping", "lesbian",
             "sonyu3P", "houshi3P", "sonyu3PMMF", "houshi3PMMF", "other"]
    out = [(k, seen[k]) for k in order if k in seen]
    for k, lab in sorted(seen.items()):
        if k not in order:
            out.append((k, lab))
    return out


def menu_entry_for_layout(position_key: str) -> "HMenuEntry | None":
    """The first insertion row whose list name stages as *position_key* (``正常位`` for ``missionary``), or None.

    Origin's own list names the position, so this is the row a layout key stands for wherever no
    menu row was picked (life mode).
    """
    key = (position_key or "missionary").lower()
    for e in (list_menu_entries() if controller_aware_bake() else ()):
        if e.family == "sonyu" and layout_for_action(e.action_name) == key:
            return e
    return None


def match_clip_for_position(position_key: str, *, sex: str = "f"):
    """Pick best Origin clip for Free H position (``sex`` = ``f`` / ``m``).

    Sex is taken only from the bake name prefix ``origin_f_`` / ``origin_m_``
    (path sex).  Clip stems like ``M_WLoop`` mean *medium body size*, not male
    (:data:`SIZE_PREFIXES`).

    Only **sonyu** packs (penetration — have Insert) are used for Free H sex
    positions.  Houshi packs (``01_*``, Oral_Idle) are service / paizuri and
    must not win missionary.
    """
    # Score on **names**, from the manifest — every rule below reads the name,
    # and the only thing that ever came from the clip body was a tiebreak the
    # manifest carries as ``bones``.  Going through ``catalog_h_clips`` parsed
    # all 6,057 baked clips (2.67 GB) to choose one: ~150 s frozen at Free H
    # boot, on top of the same parse from ``status_report`` (§7.108).
    index = _manifest_clip_index()
    if not index:
        return None
    key = (position_key or "missionary").lower()
    sex_c = "f" if sex in (0, "f", "female", "0") else "m"
    # Origin's own list names the position (``正常位`` / ``騎乗位`` / ``後背位``), so
    # ask it before falling back to guessing from pack ids — the pack cannot
    # answer this: 23 of the 28 penetration actions live in ``02_00_00``, and
    # the old ``_POS_PACKS`` mapped "missionary" onto ``02_06_00``, which is
    # 種付けプレス (mating press), while ``02_00_00`` (正常位) was listed under
    # cowgirl.
    e = menu_entry_for_layout(key)
    name = (e.name_f if sex_c == "f" else e.name_m) if e is not None else None
    clip = load_clip(name) if name else None
    if clip is not None:
        return clip
    packs = _POS_PACKS.get(key) or _POS_PACKS["missionary"]
    motion = _POS_MOTION.get(key) or ("WLoop", "SLoop")
    sex = "f" if sex in (0, "f", "female", "0") else "m"
    prefix = f"{CLIP_PREFIX}{sex}_"
    pack_set = set(packs)
    preferred = packs[0]

    scored = []
    for name, meta in index.items():
        if not name.startswith(prefix):
            continue
        pack = _clip_pack_id(name)
        if pack is None or pack not in pack_set:
            continue
        if pack in _HOUSHI_OR_AIBU:
            continue
        nl = name.lower()
        # Service / fondling stems even if they slipped into a sonyu pack.
        if any(bad in nl for bad in (
                "oral", "mune", "paizuri", "hose", "houshi", "aibu",
                "kokan_d", "denma", "sita_", "te_kokan")):
            continue
        # Prefer vaginal WLoop over A_WLoop (anal variant inside same pack).
        if "_a_wloop" in nl or "_a_sloop" in nl or "_a_oloop" in nl or "_a_insert" in nl:
            score = 5
        else:
            score = 20
        if pack == preferred:
            score += 40
        else:
            score += max(0, 24 - 4 * packs.index(pack))
        for i, tok in enumerate(motion):
            if tok.lower() in nl:
                score += 12 - i
                break
        if "wloop" in nl:
            score += 10
        elif "sloop" in nl:
            score += 6
        elif "oloop" in nl:
            score += 3
        elif "insert" in nl:
            score += 4
        if "idle" in nl and "wloop" not in nl and "sloop" not in nl:
            score -= 20
        scored.append((score, int(meta.get("bones") or 0), name))

    if not scored:
        # Partial bake: any *sonyu* WLoop for this sex — still never houshi.
        for name, meta in index.items():
            if not name.startswith(prefix):
                continue
            pack = _clip_pack_id(name)
            if pack is None or pack in _HOUSHI_OR_AIBU:
                continue
            nl = name.lower()
            if "oral" in nl:
                continue
            score = 1
            if "wloop" in nl:
                score += 5
            if "insert" in nl:
                score += 3
            if "idle" in nl and "wloop" not in nl:
                score -= 2
            scored.append((score, int(meta.get("bones") or 0), name))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return load_clip(scored[0][2])          # one JSON, not the catalogue


def status_report() -> str:
    # Count from the manifest, which already names every baked clip.  Asking
    # ``catalog_h_clips`` instead meant ``_load_baked_catalog`` parsed all 6,058
    # clip JSONs — 2.67 GB of keyframes — to print one number: over 50 s frozen
    # on the GUI thread every time Free H was built (§7.108).  The clips
    # themselves are loaded per row by ``_clip_singleton`` when one is played.
    index = _manifest_clip_index()
    if index:
        inst = _install()
        # Show the split rather than one total: a v1 entry plays *a* clip from
        # its pack but cannot say which action's, so "6,057 clip" read as full
        # coverage when 63.6% of the packs' clips had no name at all (§7.112).
        n2 = sum(1 for m in index.values() if m.get("controller"))
        return (tr("H 動畫：{clips} clip（來源 bake；可歸屬 {named}／未歸屬 {unnamed}）").format(
            clips=len(index), named=n2, unnamed=len(index) - n2) + (f" · {inst.root}" if inst is not None else ""))
    cat = catalog_h_clips()
    if cat.get("error") == "no_origin_install":
        return tr("H 動畫：未設定素材來源（使用自有的 clip）")
    n = len(cat.get("clips") or {})
    src = cat.get("source") or "?"
    b = len(cat.get("bundles") or [])
    return (tr("H 動畫：{clips} clip（來源 {source}{bundles}）").format(
        clips=n, source=src, bundles=f" / {b} bundle" if b else "") + f" · {cat.get('install', '')}")
