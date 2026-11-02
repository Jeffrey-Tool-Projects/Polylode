"""Declarative character parameter schema.

Every maker control is described once here; the UI, the card serialiser and the
randomiser all read the same table.  Tabs and sections mirror Koikatsu's maker
layout.

Hair and clothing are **the card's**, so the only keys left for them are the
ones the Illusion path actually reads: which of the seven coordinates to wear,
and the hair colour dressing falls back to where a part names none.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.i18n import N_


SLIDER = "slider"
CHOICE = "choice"
COLOR = "color"
TEXT = "text"
INT = "int"
BOOL = "bool"


@dataclass
class ParamDef:
    key: str
    label: str
    kind: str = SLIDER
    default: Any = 0.5
    options: list | None = None
    tab: str = "body"
    section: str = N_("全體")
    tip: str = ""
    lo: float = 0.0
    hi: float = 1.0

    def clone(self, **kw) -> "ParamDef":
        d = dict(self.__dict__)
        d.update(kw)
        return ParamDef(**d)


_DEFS: list[ParamDef] = []


def _s(key, label, default=0.5, tab="body", section=N_("全體"), tip="",
       lo=0.0, hi=1.0):
    _DEFS.append(ParamDef(key, label, SLIDER, default, None, tab, section, tip,
                          lo, hi))


def _c(key, label, options, default=0, tab="body", section=N_("全體"), tip=""):
    _DEFS.append(ParamDef(key, label, CHOICE, default, list(options), tab, section, tip))


def _col(key, label, default, tab="body", section=N_("全體"), tip=""):
    _DEFS.append(ParamDef(key, label, COLOR, list(default), None, tab, section, tip))


def _t(key, label, default="", tab="param", section=N_("基本"), tip=""):
    _DEFS.append(ParamDef(key, label, TEXT, default, None, tab, section, tip))


def _i(key, label, default, lo, hi, tab="param", section=N_("基本"), tip=""):
    _DEFS.append(ParamDef(key, label, INT, default, None, tab, section, tip, lo, hi))


def _b(key, label, default, tab="body", section=N_("全體"), tip=""):
    _DEFS.append(ParamDef(key, label, BOOL, default, None, tab, section, tip))


# ==========================================================================
# BODY
# ==========================================================================
_s("body.height", N_("身高"), 0.42, "body", N_("全體"), "1.38 m – 1.80 m")
_s("body.head_size", N_("頭部大小"), 0.5, "body", N_("全體"))
_s("body.build", N_("體型厚度"), 0.5, "body", N_("全體"))
_s("body.torso_length", N_("軀幹長度"), 0.5, "body", N_("全體"))
_s("body.posture", N_("姿勢前後"), 0.5, "body", N_("全體"))
_s("body.hip_tilt", N_("骨盆傾斜"), 0.5, "body", N_("全體"))
_s("body.neck_length", N_("頸長"), 0.5, "body", N_("全體"))
_s("body.neck_width", N_("頸粗"), 0.5, "body", N_("全體"))
_s("body.shoulder_width", N_("肩寬"), 0.5, "body", N_("全體"))
_s("body.shoulder_height", N_("肩高"), 0.5, "body", N_("全體"))
_s("body.shoulder_slope", N_("肩斜"), 0.5, "body", N_("全體"))
# ``body.deltoid`` was here and is gone (§7.237): swept end to end it moved
# **0 vertices**, because Origin's deltoid is the flesh **between** our trunk and
# our arm — 132 of its 176 vertices are inside neither — and the deformer it
# drove pushed on the trunk.  It comes back with the junction (N25 c), measured;
# a slider that cannot reach any geometry is worse than no slider (七).
_s("body.collarbone", N_("鎖骨"), 0.35, "body", N_("全體"))

_s("body.bust_size", N_("胸部大小"), 0.45, "body", N_("胸部"))
_s("body.bust_vertical", N_("胸部上下"), 0.5, "body", N_("胸部"))
_s("body.bust_spacing", N_("胸部間距"), 0.5, "body", N_("胸部"))
_s("body.bust_depth", N_("胸部前後"), 0.5, "body", N_("胸部"))
_s("body.bust_angle_v", N_("胸部上下角"), 0.5, "body", N_("胸部"))
_s("body.bust_angle_h", N_("胸部左右角"), 0.5, "body", N_("胸部"))
_s("body.bust_softness", N_("胸部柔軟"), 0.5, "body", N_("胸部"))
_s("body.bust_round", N_("胸部圓潤"), 0.5, "body", N_("胸部"))
_s("body.bust_crease", N_("胸下摺痕"), 0.0, "body", N_("胸部"),
   N_("衣服會沿用這道凹陷，穿著時建議保持 0"))

_s("body.chest_width", N_("胸廓寬"), 0.5, "body", N_("上半身"))
_s("body.chest_depth", N_("胸廓厚"), 0.5, "body", N_("上半身"))
_s("body.waist_width", N_("腰寬"), 0.5, "body", N_("上半身"))
_s("body.waist_depth", N_("腰厚"), 0.5, "body", N_("上半身"))
_s("body.belly", N_("腹部"), 0.5, "body", N_("上半身"))

_s("body.hip_width", N_("骨盆寬"), 0.5, "body", N_("下半身"))
_s("body.hip_width2", N_("臀寬"), 0.5, "body", N_("下半身"))
_s("body.hip_depth", N_("臀厚"), 0.5, "body", N_("下半身"))
_s("body.butt_size", N_("臀部大小"), 0.5, "body", N_("下半身"))
_s("body.butt_height", N_("臀部高度"), 0.5, "body", N_("下半身"))
_s("body.butt_soft", N_("臀部柔軟"), 0.5, "body", N_("下半身"))

_b("body.genitals", N_("性器官網格"), True, "body", N_("性器官"),
   N_("程序化生成性器官與結合處 proxy；脫衣後顯示"))
_s("body.genital_size", N_("性器官大小"), 0.5, "body", N_("性器官"))
_s("body.clitoris_size", N_("陰蒂大小"), 0.5, "body", N_("性器官"))
_s("body.penis_length", N_("陰莖長度"), 0.5, "body", N_("性器官"))
_s("body.penis_girth", N_("陰莖粗細"), 0.5, "body", N_("性器官"))

_s("body.arm_length", N_("手臂長度"), 0.5, "body", N_("手腳"))
_s("body.forearm_length", N_("前臂長度"), 0.5, "body", N_("手腳"))
_s("body.arm_thick", N_("上臂粗細"), 0.5, "body", N_("手腳"))
_s("body.forearm_thick", N_("前臂粗細"), 0.5, "body", N_("手腳"))
_s("body.wrist_thick", N_("手腕粗細"), 0.5, "body", N_("手腳"))
_s("body.hand_size", N_("手掌大小"), 0.5, "body", N_("手腳"))
_s("body.leg_length", N_("腿長"), 0.5, "body", N_("手腳"))
_s("body.thigh_length", N_("大腿長度"), 0.5, "body", N_("手腳"))
_s("body.calf_length", N_("小腿長度"), 0.5, "body", N_("手腳"))
_s("body.thigh_thick", N_("大腿粗細"), 0.5, "body", N_("手腳"))
_s("body.knee_thick", N_("膝蓋粗細"), 0.5, "body", N_("手腳"))
_s("body.calf_thick", N_("小腿粗細"), 0.5, "body", N_("手腳"))
_s("body.ankle_thick", N_("腳踝粗細"), 0.5, "body", N_("手腳"))
_s("body.foot_size", N_("腳掌大小"), 0.5, "body", N_("手腳"))

_col("body.skin_color", N_("膚色"), (0.988, 0.855, 0.792), "body", N_("膚質"))
_col("body.skin_shadow", N_("陰影色"), (0.86, 0.72, 0.76), "body", N_("膚質"))
_s("body.tan", N_("曬痕"), 0.0, "body", N_("膚質"))
_s("body.skin_gloss", N_("皮膚光澤"), 0.16, "body", N_("膚質"))
_col("body.nail_color", N_("指甲色"), (0.98, 0.86, 0.86), "body", N_("膚質"))
_s("body.skin_rim", N_("邊緣光"), 0.35, "body", N_("膚質"))

_s("pose.arm_angle", N_("手臂張開"), 0.30, "body", N_("基準姿勢"))
_s("pose.leg_spread", N_("雙腳張開"), 0.35, "body", N_("基準姿勢"))
_s("render.outline_width", N_("描邊粗細"), 0.5, "body", N_("描繪"))
_s("render.shade_steps", N_("陰影階數"), 0.34, "body", N_("描繪"))
_s("render.shade_feather", N_("陰影柔邊"), 0.35, "body", N_("描繪"))

# ==========================================================================
# FACE
# ==========================================================================
_col("face.iris_color", N_("瞳色"), (0.36, 0.62, 0.86), "face", N_("眼睛"))
_col("face.iris_grad_color", N_("瞳孔漸層色"), (0.50, 0.50, 0.60), "face", N_("眼睛"))
# ``create_eye``'s ``_Blend``: the game hands the card's ``gradBlend`` to
# ``SetFloat`` unclamped and 96 local cards sit outside [0, 1] (§7.345), so the
# stored value may not be clamped either or the eye it drives changes (§7.411)
_s("face.iris_gradient", N_("瞳孔漸層強度"), 0.0, "face", N_("眼睛"),
   lo=float("-inf"), hi=float("inf"))

# The blush is ``create_head``'s ``_tex4uv`` decal: the card names a row of
# ``mt_cheek`` (``baseMakeup.cheekId``) and these two are the colour and the
# strength it is composited with (``cheekColor``.rgb and .a).  Both are read by
# ``figure.body_head.CHEEK_OVERRIDE`` — until §7.157 neither was, so the two
# sliders moved and the face did not (measured: 0.00% of the render, 0 levels).
#
# 「腮紅樣式」and「腮紅位置」used to sit here and are **gone**, because neither
# has anything behind it: the style *is* the ``mt_cheek`` row the card names
# (nine of them locally, not the three invented labels), and there is no
# per-card position at all — the placement is ``_tex4uv`` on
# ``cf_m_face_create``, one constant (0.5, 0.64, −0.33, −4.0) shared by every
# character.  Drawing our own blush somewhere else to give the slider something
# to do is exactly what §一之四 forbids, and a control that looks like it works
# is worse than an absent one (§一之一).
_s("face.blush", N_("腮紅濃度"), 0.0, "face", N_("化妝"),
   N_("卡片的 cheekColor 透明度；沒有腮紅圖層（cheekId = 0）的卡片沒有東西可以上色"))
_col("face.blush_color", N_("腮紅色"), (0.98, 0.55, 0.58), "face", N_("化妝"),
     N_("卡片的 cheekColor"))

# ==========================================================================
# HAIR — the card supplies the strands and their colours
# ==========================================================================
# Only where a card's hair part carries no colour of its own does dressing
# fall back to this (``dress._part_colors``) — every local card's parts do, so
# with a card the row is the card's (``CARD_REDRESS_KEYS`` leaves it out, §7.550)
_col("hair.color", N_("髮色"), (0.28, 0.20, 0.24), "hair", N_("顏色"))

# ==========================================================================
# CLOTHES — a card carries seven coordinates; this picks which one is worn
# ==========================================================================
# In ``ChaFileDefine.CoordinateType`` order (School01, School02, Gym, Swim,
# Club, Plain, Pajamas — the maker's 学生服（校内）…お泊り tabs).  The list
# used to name 1 / 4 / 5 / 6 after other outfits (私服 / 睡衣 / 浴衣 / 外出)
# and offer an eighth (打工) that no card without MoreOutfits has.
_c("clothes.coordinate", N_("服裝組合"),
   [N_("制服（校內）"), N_("制服（放學）"), N_("體操服"), N_("泳裝"), N_("社團"), N_("私服"), N_("過夜")], 0,
   "clothes", N_("組合"))

# ==========================================================================
# PARAMETERS / PROFILE
# ==========================================================================
PERSONALITIES = [
    N_("元氣"), N_("溫柔"), N_("冷靜"), N_("傲嬌"), N_("天然"), N_("認真"), N_("大姐姐"), N_("妹系"), N_("文靜"),
    N_("活潑"), N_("腹黑"), N_("無口"), N_("熱血"), N_("膽小"), N_("強氣"), N_("撒嬌"), N_("神祕"), N_("毒舌"),
    N_("母性"), N_("中二"), N_("運動"), N_("書呆"), N_("貴族"), N_("野性"), N_("夢幻"), N_("現實"), N_("懶散"),
    N_("完美主義"), N_("好奇"), N_("沉穩"),
]
CLUBS = [
    N_("無社團"), N_("戀活社"), N_("游泳社"), N_("偶像社"), N_("啦啦隊"), N_("漫畫社"), N_("園藝社"),
    N_("天文社"), N_("料理社"), N_("美術社"), N_("音樂社"), N_("文藝社"), N_("茶道社"), N_("田徑社"),
    N_("網球社"), N_("科學社"), N_("攝影社"), N_("電腦社"), N_("手工藝社"), N_("占卜社"),
]
TRAITS = [
    N_("愛乾淨"), N_("怕生"), N_("貪吃"), N_("愛睡覺"), N_("運動白痴"), N_("成績優秀"), N_("音痴"),
    N_("路痴"), N_("愛哭"), N_("笑點低"), N_("怕黑"), N_("動物之友"), N_("手巧"), N_("節儉"), N_("浪漫"),
    N_("現實主義"), N_("夜貓子"), N_("早起"), N_("潔癖"), N_("遲鈍"),
]
BLOOD = ["A", "B", "O", "AB"]

_t("param.last_name", N_("姓"), "島野", "param", N_("基本"))
_t("param.first_name", N_("名"), "美咲", "param", N_("基本"))
_t("param.nickname", N_("暱稱"), "美咲", "param", N_("基本"))
_i("param.age", N_("年齡"), 18, 18, 60, "param", N_("基本"), N_("本作僅支援 18 歲以上角色"))
_c("param.sex", N_("性別"), [N_("女性"), N_("男性")], 0, "param", N_("基本"))
_i("param.birth_month", N_("生日（月）"), 4, 1, 12, "param", N_("基本"))
_i("param.birth_day", N_("生日（日）"), 1, 1, 31, "param", N_("基本"))
_c("param.blood", N_("血型"), BLOOD, 0, "param", N_("基本"))
_i("param.height_cm", N_("身高標示 (cm)"), 158, 130, 200, "param", N_("基本"))
_c("param.personality", N_("個性"), PERSONALITIES, 0, "param", N_("性格"))
_c("param.club", N_("社團"), CLUBS, 1, "param", N_("性格"))
_c("param.trait1", N_("特質 1"), [N_("無")] + TRAITS, 0, "param", N_("性格"))
_c("param.trait2", N_("特質 2"), [N_("無")] + TRAITS, 0, "param", N_("性格"))
_c("param.trait3", N_("特質 3"), [N_("無")] + TRAITS, 0, "param", N_("性格"))
_t("param.description", N_("角色說明"), "", "param", N_("性格"))
_t("param.style", N_("美術風格"), "anime", "param", N_("基本"),
   N_("本作僅支援動漫／卡通風格角色"))

# ==========================================================================
SCHEMA: list[ParamDef] = list(_DEFS)
BY_KEY: dict[str, ParamDef] = {d.key: d for d in SCHEMA}

TAB_ORDER = ["body", "face", "hair", "clothes", "param"]
TAB_LABELS = {"body": N_("體型"), "face": N_("臉部"), "hair": N_("頭髮"), "clothes": N_("服裝"),
              "param": N_("參數")}


def sections(tab: str) -> list[str]:
    out = []
    for d in SCHEMA:
        if d.tab == tab and d.section not in out:
            out.append(d.section)
    return out


def params_in(tab: str, section: str | None = None) -> list[ParamDef]:
    return [d for d in SCHEMA if d.tab == tab and (section is None or d.section == section)]


def defaults() -> dict:
    """A fresh parameter dict with every schema default applied."""
    out: dict = {}
    for d in SCHEMA:
        out[d.key] = list(d.default) if d.kind == COLOR else d.default
    return out


#: keys an earlier version saved a card with → this version's
LEGACY_KEYS = {"param.kk_personality": "param.origin_personality"}


def normalise(params: dict) -> dict:
    """Fill missing keys and coerce types (an earlier version's keys renamed, :data:`LEGACY_KEYS`); returns the same dict."""
    for old, new in LEGACY_KEYS.items():
        if old in params:
            value = params.pop(old)
            params.setdefault(new, value)
    for d in SCHEMA:
        v = params.get(d.key)
        if v is None:
            params[d.key] = list(d.default) if d.kind == COLOR else d.default
            continue
        try:
            if d.kind == SLIDER:
                params[d.key] = min(d.hi, max(d.lo, float(v)))
            elif d.kind in (CHOICE, INT):
                params[d.key] = int(v)
            elif d.kind == BOOL:
                params[d.key] = bool(v)
            elif d.kind == COLOR:
                c = [float(x) for x in list(v)[:4]]
                params[d.key] = c if len(c) >= 3 else list(d.default)
            elif d.kind == TEXT:
                params[d.key] = str(v)
        except (TypeError, ValueError):
            params[d.key] = list(d.default) if d.kind == COLOR else d.default
    return params


def display_name(params: dict) -> str:
    ln = str(params.get("param.last_name", "")).strip()
    fn = str(params.get("param.first_name", "")).strip()
    return (ln + " " + fn).strip() or "未命名"


def choice_label(key: str, params: dict) -> str:
    d = BY_KEY.get(key)
    if not d or not d.options:
        return ""
    i = int(params.get(key, d.default) or 0)
    return d.options[i % len(d.options)] if d.options else ""


# --------------------------------------------------------------------------
# randomiser
# --------------------------------------------------------------------------
# Colours that must not be uniform-random: skin has to stay skin, the sclera
# has to stay white, or the randomiser produces green people.
FIXED_COLORS = {
    "body.skin_shadow", "face.sclera_color", "face.highlight_color",
    "face.mouth_inner", "face.tongue_color",
}
SKIN_COLORS = {"body.skin_color"}


def randomize(params: dict, rng, *, tabs: tuple = ("body", "face", "hair", "clothes"),
              strength: float = 1.0) -> dict:
    """Randomise the given tabs in place. Slider spread is clamped by strength."""
    for d in SCHEMA:
        if d.tab not in tabs:
            continue
        if d.kind == SLIDER:
            base = float(d.default)
            lo = max(0.0, base - 0.5 * strength)
            hi = min(1.0, base + 0.5 * strength)
            params[d.key] = rng.uniform(lo, hi)
        elif d.kind == CHOICE and d.options:
            params[d.key] = int(rng.randrange(len(d.options)))
        elif d.kind == COLOR:
            if d.key in FIXED_COLORS:
                continue
            if d.key in SKIN_COLORS:
                # skin has to stay skin: jitter the schema default rather than
                # draw a colour, or the randomiser produces green people
                jitter = rng.uniform(-0.06, 0.02)
                params[d.key] = [min(1.0, max(0.0, c + jitter))
                                 for c in list(d.default)[:3]]
            else:
                params[d.key] = [rng.uniform(0.12, 1.0) for _ in range(3)]
        elif d.kind == BOOL:
            params[d.key] = bool(rng.random() > 0.5)
    # keep the result plausible: no giant heads
    params["body.head_size"] = min(params.get("body.head_size", 0.5), 0.72)
    # keep faces readable: eyes must not collapse and lashes must stay dark
    params["face.eye_width"] = max(params.get("face.eye_width", 0.5), 0.30)
    params["face.eye_height"] = max(params.get("face.eye_height", 0.5), 0.30)
    return params
