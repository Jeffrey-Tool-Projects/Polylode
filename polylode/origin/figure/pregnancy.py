"""KK_Pregnancy's belly — a ``BoneEffect`` on the ABMX bone controller.

Read from the local ``BepInEx/plugins/KK_Pregnancy.dll`` (§7.453, §7.454):

* **Data.**  In the main game ``PregnancyCharaController.ReadData`` takes the
  heroine's save data when there is a heroine, else the card's own extended
  data (``PregnancyData.Load``: one key per serialised field, ``Week`` among
  them).  This build has no save, so it is the card.
* **When.**  ``PregnancyBoneEffect.GetAffectedBones`` answers only when the
  character is pregnant (``IsPregnant`` is ``Week > 0``) — or inside the
  maker / Studio / an H scene — and then sets the percent::

      GetPregnancyEffectPercent = 0 if Week > LeaveSchoolWeek
                                  else Clamp01((Week - 1) / (LeaveSchoolWeek - 1))

  with ``LeaveSchoolWeek = 41`` (``PregnancyData..cctor``).
* **What** (``GetEffect``): a pregnant character's bones in
  :data:`PREGNANCY_FULL` get ``LerpModifier(full, percent)``; the bones in
  :data:`BELLY_FULL` (pregnant, or inflated) get it at
  ``max(percent, inflation + percent / 2)``.  Inflation is a runtime amount
  (``InflationAmount`` from H), 0 on load.  ``LerpModifier`` lerps scale and
  length from 1 and position and rotation from 0.

The two tables are ``PregnancyBoneEffect..cctor`` as written, the
``BoneModifierData(scale, length[, position, rotation])`` rows in order
(``dev`` script over the ``origin_il_dump`` listing).
"""
from __future__ import annotations

from .bonemod import BoneModifierData

GUID = "KK_Pregnancy"
LEAVE_SCHOOL_WEEK = 41

#: ``_bellyFullValues`` — (scale, length, position, rotation)
BELLY_FULL = {
    'cf_s_spine01': ((1.62, 1.5, 1.9), 1.0, (0.0, 0.0, 0.05), (10.0, 0.0, 0.0)),
    'cf_s_spine02': ((1.0, 1.0, 1.0), 1.0, (0.0, 0.0, 0.01), (-5.0, 0.0, 0.0)),
    'cf_s_waist01': ((1.25, 1.55, 1.7), 1.0, (0.0, 0.02, 0.12), (15.0, 0.0, 0.0)),
    'cf_s_waist02': ((1.17, 1.0, 1.0), 1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    'cf_d_sk_00_00': ((1.0, 1.0, 1.0), 1.0, (0.0, -0.02, 0.14), (-10.0, 0.0, 0.0)),
    'cf_d_sk_07_00': ((1.0, 1.0, 1.0), 1.0, (-0.03, -0.02, 0.12), (-10.0, 0.0, 0.0)),
    'cf_d_sk_01_00': ((1.0, 1.0, 1.0), 1.0, (0.03, -0.02, 0.12), (-10.0, 0.0, 0.0)),
    'cf_d_sk_06_00': ((1.0, 1.0, 1.0), 1.0, (-0.03, 0.0, 0.05), (0.0, 0.0, 0.0)),
    'cf_d_sk_02_00': ((1.0, 1.0, 1.0), 1.0, (0.03, 0.0, 0.05), (0.0, 0.0, 0.0)),
}

#: ``_pregnancyFullValues``
PREGNANCY_FULL = {
    'cf_d_bust01_L': ((1.2, 1.2, 1.2), 1.0), 'cf_d_bust01_R': ((1.2, 1.2, 1.2), 1.0),
    'cf_d_bnip01_L': ((1.2, 1.2, 1.2), 1.0), 'cf_d_bnip01_R': ((1.2, 1.2, 1.2), 1.0),
    'cf_s_siri_L': ((1.1, 1.0, 1.2), 1.0), 'cf_s_siri_R': ((1.1, 1.0, 1.2), 1.0),
    'cf_s_thigh00_L': ((1.04, 1.0, 1.04), 1.0), 'cf_s_thigh00_R': ((1.04, 1.0, 1.04), 1.0),
    'cf_s_thigh01_L': ((1.04, 1.0, 1.04), 1.0), 'cf_s_thigh01_R': ((1.04, 1.0, 1.04), 1.0),
    'cf_s_thigh02_L': ((1.04, 1.0, 1.04), 1.0), 'cf_s_thigh02_R': ((1.04, 1.0, 1.04), 1.0),
    'cf_s_thigh03_L': ((1.04, 1.0, 1.04), 1.0), 'cf_s_thigh03_R': ((1.04, 1.0, 1.04), 1.0),
    'cf_s_leg01_L': ((1.04, 1.0, 1.04), 1.0), 'cf_s_leg01_R': ((1.04, 1.0, 1.04), 1.0),
    'cf_s_leg02_L': ((1.04, 1.0, 1.04), 1.0), 'cf_s_leg02_R': ((1.04, 1.0, 1.04), 1.0),
}


def week(card) -> int:
    """``PregnancyData.Week`` off the card (0 without the section)."""
    try:
        sec = ((card.sections.get("KKEx") or {}) if card is not None else {}).get(GUID)
    except AttributeError:
        return 0
    data = sec[1] if isinstance(sec, (list, tuple)) and len(sec) >= 2 else sec
    if not isinstance(data, dict):
        return 0
    try:
        return int(data.get("Week") or 0)
    except (TypeError, ValueError):
        return 0


def effect_percent(w: int) -> float:
    """``GetPregnancyEffectPercent``."""
    if w > LEAVE_SCHOOL_WEEK:
        return 0.0
    return min(1.0, max(0.0, (w - 1.0) / (LEAVE_SCHOOL_WEEK - 1.0)))


def _lerp(full, t: float) -> BoneModifierData:
    scale, length = full[0], full[1]
    pos = full[2] if len(full) > 2 else (0.0, 0.0, 0.0)
    rot = full[3] if len(full) > 3 else (0.0, 0.0, 0.0)
    return BoneModifierData(
        scale=tuple(1.0 + (s - 1.0) * t for s in scale),
        length=1.0 + (length - 1.0) * t,
        position=tuple(p * t for p in pos),
        rotation=tuple(r * t for r in rot))


def effects(card, inflation: float = 0.0) -> dict:
    """``{bone: [BoneModifierData]}`` for :func:`bonemod.apply_bone_modifiers`.

    Empty for a character that is not pregnant (the effect's
    ``GetAffectedBones`` answers nothing outside the maker / Studio / H then).
    """
    w = week(card)
    if w <= 0:
        return {}
    p = effect_percent(w)
    out = {name: [_lerp(full, p)] for name, full in PREGNANCY_FULL.items()}
    belly = max(p, float(inflation) + p / 2.0)
    for name, full in BELLY_FULL.items():
        out.setdefault(name, []).append(_lerp(full, belly))
    return out
