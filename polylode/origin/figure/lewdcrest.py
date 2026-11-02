"""LewdCrestX (``KKEx["LewdCrestX"]``) — the crest drawn on the lower belly.

Read from the local ``BepInEx/plugins/KK_LewdCrestX.dll``:

* **Card data.**  ``LewdCrestXController.ReadData``: ``CurrentCrest`` (a
  ``CrestType``) and ``HideCrestGraphic``.
* **The picture.**  ``LewdCrestXPlugin.LoadAssets`` loads the DLL's embedded
  resource ``KK_LewdCrestX.crests`` with ``AssetBundle.LoadFromMemory``;
  ``CrestInfo.GetTexture`` is ``LoadAsset<Texture2D>(Id.ToString())`` — the
  asset is named after the enum member (2048², sRGB on this install).
* **Where.**  ``ApplyCrestTexture`` removes its own entries from
  ``KoiSkinOverlayController.AdditionalTextures``, and unless the crest is
  ``None`` or ``HideCrestGraphic`` adds ``AdditionalTexture(tex, BodyOver,
  this, 1010)``, then ``UpdateTexture(BodyOver)``.  Overlay Mods'
  ``GetOverlayTextures(type)`` yields the card's own stored overlay first and
  then the additional ones of that type ordered by ``ApplyOrder`` ⇒ the crest
  is composited onto the body map **after** the card's ``BodyOver``
  (:mod:`skin_overlay`).

The DLL is the plugin author's and its pictures are theirs: the converter
reads them on the user's machine into the user's own bake
(``bake.write_texture("lewdcrest", id, …)``) and the release carries none
(``CLAUDE.md`` 四之一).

* **Bones** (``LewdCrestXBoneModifier``, added to the ABMX ``BoneController``
  in ``LewdCrestXController.Start`` for every character).  Three crests answer
  ``GetAffectedBones``: ``reduction`` (8), ``lactation`` (11) and ``vibrancy``
  (30); every other crest answers none.  ``GetEffect`` hands back the table
  row — scale only, length 1 — and for ``vibrancy`` lerps the scale from one
  by ``lewdness / 120 + HExperience · 0.1`` (eased by ``MoveTowards``) **when
  there is a heroine**; without one it is the full row.  This build has no
  save, so there is no heroine (as in :mod:`pregnancy`).  The effect does not
  read ``HideCrestGraphic``.
"""
from __future__ import annotations

from pathlib import Path

GUID = "LewdCrestX"
DLL_REL = "BepInEx/plugins/KK_LewdCrestX.dll"
RESOURCE = "KK_LewdCrestX.crests"
BAKE_CATEGORY = "lewdcrest"
#: ``AdditionalTexture``'s ``ApplyOrder`` — after the card's own overlay
APPLY_ORDER = 1010

#: ``CrestType`` (``origin_il_dump --dll KK_LewdCrestX.dll --enum CrestType``)
CREST_TYPES = {
    0: "None", 1: "animalistic", 2: "breedgasm", 3: "command",
    4: "cumdiction", 5: "deepfocus", 6: "destruction", 7: "forlorn",
    8: "reduction", 9: "glow", 10: "gullible", 11: "lactation",
    12: "liberated", 13: "libido", 14: "mantraction", 15: "messaging",
    16: "mindmelt", 17: "pheromone", 18: "receptacle", 19: "regrowth",
    20: "reprogram", 21: "restore", 22: "sensitivity", 23: "siphoning",
    24: "slave", 25: "soulchained", 26: "statistics", 27: "suffer",
    28: "task", 29: "triggered", 30: "vibrancy", 31: "violove",
    32: "broodmother",
}


#: ``LewdCrestXBoneModifier..cctor``: crest id → ``{bone: scale}``
BONE_SCALE = {
    30: {"cf_d_bust01_L": 1.25, "cf_d_bust01_R": 1.25,
         "cf_d_bnip01_L": 1.15, "cf_d_bnip01_R": 1.15,
         "cf_s_bnip02_L": 1.25, "cf_s_bnip02_R": 1.25,
         "cf_s_siri_L": 1.25, "cf_s_siri_R": 1.25,
         "cf_s_waist01": 0.9, "cf_s_waist02": 1.15,
         "cf_s_thigh01_L": 1.15, "cf_s_thigh01_R": 1.15,
         "cf_s_thigh02_L": 1.15, "cf_s_thigh02_R": 1.15,
         "cf_s_thigh03_L": 1.15, "cf_s_thigh03_R": 1.15,
         "cf_n_height": 1.08},
    11: {"cf_d_bust01_L": 1.25, "cf_d_bust01_R": 1.25,
         "cf_d_bnip01_L": 1.2, "cf_d_bnip01_R": 1.2},
    8: {"cf_d_bust01_L": 0.8, "cf_d_bust01_R": 0.8,
        "cf_d_bust02_L": 0.7, "cf_d_bust02_R": 0.7,
        "cf_s_siri_L": 0.8, "cf_s_siri_R": 0.8,
        "cf_n_height": 0.91,
        "cf_s_waist01": 0.9, "cf_s_waist02": 0.9,
        "cf_s_thigh01_L": 0.9, "cf_s_thigh01_R": 0.9,
        "cf_s_thigh02_L": 0.9, "cf_s_thigh02_R": 0.9,
        "cf_s_thigh03_L": 0.9, "cf_s_thigh03_R": 0.9},
}


def read(card) -> tuple[int, bool, str]:
    """``(crest id, hidden, reason)`` off the card; ``(0, False, "")`` without one."""
    try:
        sec = ((card.sections.get("KKEx") or {}) if card is not None else {}).get(GUID)
    except AttributeError:
        return 0, False, ""
    if sec is None:
        return 0, False, ""
    data = sec[1] if isinstance(sec, (list, tuple)) and len(sec) >= 2 else sec
    if not isinstance(data, dict):
        return 0, False, "LewdCrestX 區段不是 (version, data) 這個形狀"
    raw = data.get("CurrentCrest", 0)
    try:
        crest = int(raw)
    except (TypeError, ValueError):
        crest = -1
    if crest not in CREST_TYPES:
        # ``ApplyCrestTexture`` logs "Unknown crest type" and resets to None
        return 0, False, f"LewdCrestX 的淫紋 {raw!r} 不認得（外掛會重設成無）"
    return crest, bool(data.get("HideCrestGraphic", False)), ""


def overlay(card, report=None):
    """The crest map to composite over ``BodyOver``, or ``None``.

    A card that names a crest the bake does not hold is reported — the
    converter has not been run since the plugin was installed, or the plugin
    is not in the source install (一之一).
    """
    crest, hidden, reason = read(card)
    if reason and report is not None:
        report.missing.append(reason)
    if crest == 0 or hidden:
        return None
    from .. import bake
    tex = bake.read_texture(BAKE_CATEGORY, crest)
    if tex is None:
        if report is not None:
            report.missing.append(
                f"淫紋 {CREST_TYPES[crest]} 沒有轉出（轉檔時讀 {DLL_REL}）")
        return None
    if report is not None:
        report.replaced.append(f"淫紋（LewdCrestX）：{CREST_TYPES[crest]}")
    return tex


def effects(card) -> dict:
    """``{bone: [BoneModifierData]}`` for :func:`bonemod.apply_bone_modifiers`
    (no heroine ⇒ the full row, see the module docstring)."""
    from .bonemod import BoneModifierData
    crest, _hidden, _reason = read(card)
    return {bone: [BoneModifierData(scale=(s, s, s))]
            for bone, s in BONE_SCALE.get(crest, {}).items()}


def bake_crests(source_root: Path | None, force: bool = False, log=None) -> dict:
    """Converter step: every crest picture out of the plugin DLL into the bake."""
    from .. import bake
    from ..netres import manifest_resources
    from ..prefab import PrefabLoader

    say = log or (lambda msg: None)
    if source_root is None:
        return {"ok": 0, "reason": "沒有來源安裝"}
    dll = Path(source_root) / DLL_REL
    if not dll.is_file():
        say(f"  lewdcrest: 來源安裝沒有 {DLL_REL}，略過")
        return {"ok": 0, "reason": "沒有這個外掛"}
    if not force and all(bake.read_texture(BAKE_CATEGORY, i) is not None
                         for i in CREST_TYPES if i):
        say("  lewdcrest: 已轉過，略過")
        return {"ok": 0, "skipped": len(CREST_TYPES) - 1}
    try:
        from ..meshimport import memory_bundle
        blob = manifest_resources(dll).get(RESOURCE)
        if blob is None:
            raise ValueError(f"DLL 裡沒有 {RESOURCE}")
        loader = PrefabLoader(dll, env=memory_bundle(blob))
    except Exception as exc:                                  # noqa: BLE001
        say(f"  ! lewdcrest: {exc}")
        return {"ok": 0, "reason": str(exc)}
    ok, missing = 0, []
    for ident, name in CREST_TYPES.items():
        if not ident:
            continue
        tex = loader.texture_by_name(name)
        if tex is None:
            missing.append(name)
            continue
        bake.write_texture(BAKE_CATEGORY, ident, tex)
        ok += 1
    say(f"  lewdcrest: 轉出 {ok}，缺 {len(missing)} {missing or ''}")
    return {"ok": ok, "missing": missing}
