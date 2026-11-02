"""Illusion's facial blend shapes, registered as this project's morphs.

Origin does not pose the face with bones — the 75 ``cf_J_*`` head bones carry the
*shape* sliders, and the expression itself is blend shapes on the head meshes.
``cf_O_face`` ships 81 channels, ``cf_O_mayuge`` 34, the lash sheets 27 each,
teeth and tongue 39 and 36; every one of them is a single-frame channel over a
sparse ``(index, delta)`` list, and the indices address the mesh's own vertex
order (``cf_O_face``: 2695 vertices on both sides, largest shape index 2694).

Names are ``<group>.<meshcode><suffix>`` — ``eye_face.f00_def_cl``,
``eye_line_u.elu00_def_cl``, ``eye_siroL.sL00_def_cl`` — so the *suffix* is
what identifies an expression across the ten meshes that have to move together.
Registering each shape under its suffix is what makes
``MorphRig.set("def_cl", 1.0)`` close the lids, the lashes, the sclera and the
tear meshes in one call, which is what Illusion does with a shared weight.

The suffix vocabulary is read off the assets, not invented:

* ``def_cl`` (10 meshes) — the eye group's closed lids.  This is the blink.
* ``def_op`` (5 meshes: face, teeth, canine, nose line, tongue) — mouth open.
* ``egao`` smile, ``bisyou`` slight smile, ``uresi`` happy, ``oko`` angry,
  ``komaru``/``koma`` troubled, ``setunai`` pained, ``odoro`` surprised,
  ``tere`` bashful, ``sinken`` serious, ``gyu`` squeezed shut,
  ``winkl`` / ``winkr`` — each in a ``_cl`` (eyes) and/or ``_op`` (mouth) form.
* ``a_l_op`` / ``a_s_op`` … — the あいうえお visemes, large and small.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...mesh.blendshape import BlendShape, MorphRig

# Which logical morph this project already drives maps onto which Origin suffix.
# The left-hand names are the ones ``chara.expression`` has always used, so the
# existing expression table, blink and lip-sync work unchanged on a Origin head.
ALIAS = {
    "blink": "def_cl",
    "jaw_open": "a_l_op",
    "mouth_big": "o_l_op",
    "smile_pull_L": "egao_op",
    "smile_pull_R": "egao_op",
    "brow_up": "odoro_op",
    "brow_sad": "komaru_cl",
    "brow_furrow": "oko_cl",
    "brow_down": "sinken_cl",
    "cheek_puff_L": "tere_op",
    "cheek_puff_R": "tere_op",
}


def suffix_of(name: str) -> str:
    """``eye_face.f00_def_cl`` → ``def_cl``."""
    tail = name.split(".", 1)[-1]
    return tail.split("_", 1)[-1] if "_" in tail else tail


def register(rig: MorphRig, part, shapes) -> int:
    """Put one piece's Origin shapes on *part*.  Returns how many were added."""
    if part is None or not shapes:
        return 0
    n_vert = part.mesh.vertex_count if part.mesh is not None else 0
    s = rig.register(part)
    added = 0
    for name, idx, delta, *rest in shapes:
        nd = rest[0] if rest else None
        if len(idx) == 0 or n_vert == 0:
            continue
        # a stale bake could carry indices for a different mesh; dropping the
        # shape is better than writing past the end of the vertex array
        if int(idx.max()) >= n_vert:
            continue
        # The shape is stored under its **full** Origin name and the group suffix
        # is an *alias*.  Both are needed: the suffix lets a caller say "blink"
        # without Illusion's vocabulary, while a pattern addresses one mesh at
        # a time and has to be able to say *which* ``def_cl`` — ``cf_O_face``
        # carries both ``eye_face.*`` and ``kuti_face.*``, and six suffixes
        # (``egao_op``, ``bisyou_op``, ``ikari_op``, ``doya_op``, ``aseri_op``,
        # ``keno_op``) exist in both.  Storing it twice instead of aliasing
        # would make ``apply`` add the same delta once per name (§7.131).
        #
        # The suffix is **not** a way to drive a whole group: broadcasting
        # ``def_cl`` reaches ten meshes and the eye group is nine of them — the
        # tenth is ``cf_O_mayuge``, which belongs to ``FBSCtrlEyebrow`` and was
        # having its brows held 8% shut by every blink (§7.137).  Use
        # :func:`set_eye_close` / :func:`write_paths`.
        s.add(BlendShape(name, idx, delta, nd, name))
        s.alias(suffix_of(name), name)
        added += 1
    # the aliases are plain extra names for shapes this part already has, so a
    # caller can say "blink" without knowing Illusion's vocabulary
    for alias, target in ALIAS.items():
        tgt = s.resolve(target)
        if tgt:
            s.alias(alias, tgt)
    return added


# ---------------------------------------------------------------------------
# Gaze
# ---------------------------------------------------------------------------
# ``EyeLookMaterialControll`` is a MonoBehaviour on the head prefab, one per
# eye, and it does not rotate anything: every frame it writes its ``offset``
# Vector2 into the iris material's ``_MainTex`` / ``_overtex1`` / ``_overtex2``
# texture offset.  The eyeballs do not rotate either -- see ``eye_look`` -- so
# this offset is the whole of the gaze's horizontal and vertical travel.
#
# ``SetEyeTexOffsetX`` negates the value when ``eyeLR == 1``, and the reason is
# **not** that the meshes are mirrored: they are not.  Fitting each iris mesh's
# own UV against its world position gives ``dx/du = +0.0729`` on *both* eyes
# (the two use different windows of the same atlas, 0.091 apart in u, not a
# mirror), so one u shift slides both irises the same way.  What the negation
# mirrors is the card's ``pupilX``, which is a face-design slider and has to
# move the two pupils symmetrically about the nose.  The gaze rides on top of
# it unmirrored -- ``angleHRate`` is a common direction, not a per-eye one (see
# ``eye_look``) -- which is what keeps a look to one side from crossing them.
#
# The numbers below are the ones serialised on ``p_cf_head_00``, not guesses:
#
#     power        0.001
#     InsideLimit -100   OutsideLimit 100     -> x within +-0.1
#     UpLimit      -80   DownLimit     80     -> y within +-0.08
#     Limit (Rect) x 80, y 80, width 100, height 100
#
# The rest state -- ``offset`` from the card's ``pupilX`` / ``pupilY``, run
# through the component's waits -- is baked into our iris map by
# ``body_head.eye_texture`` (see ``eye_look_st``), so what travels through
# ``Material.uv_offset`` here is the *delta* the runtime gaze adds on top, and
# it is the real formula rather than a linear stand-in: ``body_head`` runs the
# whole of ``Update`` twice, once at the gaze and once at the zero the map was
# baked at, and the difference divided by the tiling is the shift.  Both the
# ``Lerp(1, 5, scale)`` travel multiplier and the clamps at the ends therefore
# apply, which a ``rate * limit`` approximation dropped.
IRIS_PARTS = ("origin_head_cf_Ohitomi_L02", "origin_head_cf_Ohitomi_R02")


def _lerp_accel(a: float, b: float, t: float) -> float:
    """``MathfEx.LerpAccel`` — ``Mathf.Lerp(a, b, Mathf.Sqrt(t))``."""
    t = max(0.0, min(1.0, float(t)))
    return a + (b - a) * (t ** 0.5)


def gaze_open_max(open_max: float, h: float = 0.0, v: float = 0.0,
                  look: dict | None = None) -> float:
    """``correctOpenMax`` for a gaze — ``FaceBlendShape.LateUpdate``.

    *open_max* is ``EyesCtrl.OpenMax`` (``body_head.eyes_open_max``), **not**
    the already-capped resting value: the correction is scaled by it as well as
    capped against it, and pre-capping cost the male head 2% of its travel.

    The lids close as the eyes look down or sideways.  Transcribed::

        hi = min(1 - EyeLookUpCorrect, OpenMax)
        c  = v > 0 ? LerpAccel(0, Up,   v)
                   : -LerpAccel(0, Down, -v)
        c -= LerpAccel(0, Side, |h|)
        c  = Clamp(c, -max(Down, Side), Up) * OpenMax
        correctOpenMax = hi + c

    With the gaze at rest this is just ``hi``, which is why it is also what
    ``blendshape.rest_blink`` calls rather than keeping a second copy of the
    line.  But "at rest" means ``h = v = 0``, and a character actually being
    looked through does not sit there: ``angleHRate`` is ±0.2203 with the eyes
    straight ahead (see :mod:`eye_look`), and the side term alone takes
    ``def_cl`` from 0.08 to 0.15.

    ``EyeLookDownCorrect`` is **0.33** on the stock female head, so a full
    downward look shuts the lids by a third of their travel on top of that —
    far more than the gaze does to anything else on the face, and the reason
    the card's own face shot cannot be read as a neutral pose (PROGRESS §7.47,
    §7.48).
    """
    look = look or {}
    up = _num(look.get("upCorrect"), 0.0)
    down = _num(look.get("downCorrect"), 0.0)
    side = _num(look.get("sideCorrect"), 0.0)
    open_max = float(open_max)
    hi = min(1.0 - up, open_max)
    v = max(-1.0, min(1.0, float(v)))
    h = max(-1.0, min(1.0, float(h)))
    c = _lerp_accel(0.0, up, v) if v > 0 else -_lerp_accel(0.0, down, -v)
    c -= _lerp_accel(0.0, side, abs(h))
    c = max(-max(down, side), min(up, c)) * open_max
    return max(0.0, min(1.0, hi + c))


def _num(value, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def look_at(built, h=0.0, v: float = 0.0, *, blend: bool = True) -> int:
    """Apply one frame of gaze: both irises and the lids.  Returns eyes moved.

    *h* is ``EyeLookCalc.GetAngleHRate`` **per eye** — ``angleHRate[0]`` and
    ``[1]``.  They are not each other's negation and neither is zero at rest:
    both fall as the character looks to their left, by different amounts,
    because each eye's bending range is measured nasally and temporally rather
    than left and right (see :mod:`eye_look`).  *v* is ``GetAngleVRate``,
    positive looking up.  A scalar *h* is used for both eyes.

    Origin does the two together in ``FaceBlendShape.LateUpdate``, and they have to
    stay together: the iris alone makes a downward look read as a stare,
    because ``EyeLookDownCorrect`` (0.33 on the stock female head) is what puts
    the lid over it.  The lid half is taken from ``angleHRate[0]``, which is
    the index ``LateUpdate`` reads.

    Everything else comes off *built*, so a caller only has to produce the
    three numbers: the card's pupil sliders and each eye's component for the
    ST, and the head's three gaze corrections for the lids.

    *blend* plays the new ``correctOpenMax`` through the face rig, the way
    ``LateUpdate`` closes with ``EyesCtrl.CalcBlend``.  Turn it off when
    something else replays the rig later in the same frame —
    ``ExpressionController`` does, and rebuilding ten meshes twice a frame is
    not free.  Leaving it on is what makes a still frame (an offline render, a
    paused screen) show the lids the gaze asked for.
    """
    if built is None:
        return 0
    model = getattr(built, "model", None)
    if model is None:
        return 0
    if isinstance(h, (int, float)):
        h = (float(h), float(h))
    h = [max(-1.0, min(1.0, float(x))) for x in h]
    v = max(-1.0, min(1.0, float(v)))

    from . import body_head as BH
    face = getattr(built, "eye_face", None)
    looks = getattr(built, "eye_look", None) or [None, None]
    rests = getattr(built, "eye_st", None) or [None, None]
    n = 0
    if isinstance(face, dict):
        for part in model.parts:
            # ``_m1`` etc. can follow when a renderer has a second material pass
            base = part.name.split("_m")[0] if "_m" in part.name[-3:] else part.name
            if base not in IRIS_PARTS:
                continue
            side = 1 if base.endswith("R02") else 0
            part.material.uv_offset = BH.eye_look_delta(
                face, side, looks[side], (h[side], v), rests[side])
            n += 1
        built.eye_gaze = ((h[0], h[1]), v)
        place_highlights(built)

    look = looks[0] or looks[1]
    if look:
        built.eyes_open_correct = gaze_open_max(
            float(getattr(built, "eyes_open_max", 1.0)), h[0], v, look)
        rig = getattr(built, "face_morphs", None)
        if blend and rig is not None:
            from ...mesh.blendshape import rest_blink
            # Same scoping as everywhere else the lids are written: the eye
            # meshes, not every set that happens to carry ``def_cl`` (§7.137).
            set_eye_close(built, rig, rest_blink(built))
            rig.apply()
    return n



def place_highlights(built) -> int:
    """Each iris's ``highlight_xf`` for the last gaze and this frame's shake.

    ``look_at`` calls it with the gaze it just wrote; the character's frame
    calls it again after ticking the shake, so a character whose gaze did not
    run this frame (no target) still shakes.  Returns the irises written.
    """
    from . import body_head as BH
    face = getattr(built, "eye_face", None)
    model = getattr(built, "model", None)
    if not isinstance(face, dict) or model is None:
        return 0
    looks = getattr(built, "eye_look", None) or [None, None]
    rests = getattr(built, "eye_st", None) or [None, None]
    shakes = getattr(built, "eye_shake", None) or [None, None]
    (h0, h1), v = getattr(built, "eye_gaze", ((0.0, 0.0), 0.0))
    n = 0
    for part in model.parts:
        base = part.name.split("_m")[0] if "_m" in part.name[-3:] else part.name
        if base not in IRIS_PARTS:
            continue
        side = 1 if base.endswith("R02") else 0
        sh = shakes[side]
        part.material.highlight_xf = BH.eye_highlight_xf(
            face, side, looks[side], ((h0, h1)[side], v), rests[side],
            sh.current if sh is not None and sh.on else None)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Expression patterns (``FBSCtrl*``) — Origin's eyebrow / eyes / mouth numbers
# ---------------------------------------------------------------------------
# A pattern is not one shape, it is a **Close/Open pair** blended by how open
# that feature currently is.  ``FBSBase.CalculateBlendShape``, line for line:
#
#     openMax  = correctOpenMax >= 0 ? correctOpenMax : OpenMax
#     rate     = Lerp(OpenMin, openMax, openRate)
#     if FixedRate >= 0: rate = FixedRate
#     openPct  = (int)Clamp(rate * 100, 0, 100);  closePct = 100 - openPct
#     weight[co.Close] += w * closePct;  weight[co.Open] += w * openPct
#
# ``>= 0``, not ``> 0`` — the IL is ``ldc.r4 0; ldfld FixedRate; bgt.un`` and
# the assignment is on the *fall-through*, so it is skipped only when
# ``0 > FixedRate``.  The distinction had no teeth while the only writers were
# the prefab (−0.1) and an H row (positive), and it grew some the moment the
# card's own mouth reached here: ``ChangeMouthFixed(false)`` writes **−1** and
# ``ChangeMouthOpenMax`` writes ``mouthOpenMax`` when ``mouthFixed``, which on
# this install is **1,385 of 1,388 cards**, and 1,211 of those ask for **0.0**
# (§7.137).  Read ``> 0`` those become "no fixed rate" and a talking character
# opens a mouth the card pinned shut.
#
# ``ChangePtn(ptn, open)``'s second argument is **not** "open the mouth": it
# reaches ``ChangeFace``, whose only use of it is
# ``open ? blendTimeCtrl.Start() : blendTimeCtrl.End()`` — cross-fade or snap.
# ``SetFace`` passes ``true`` for every slot, so patterns always cross-fade and
# the openness comes from the rate below (§7.131).
#
# ``openRate`` is the voice's mouth drive, so a silent character sits at
# ``OpenMin`` — which is exactly the knob ``HFaceList`` sets per row
# (``openMinMouth``).  Every 舐め/咥え row writes **1.0**, which pins the pair
# at 100% Open no matter what the voice does; that is how Origin holds a mouth open
# around a tongue.  ``FixedRate`` is −0.1 (off) on the prefab.
#
# ``Close``/``Open`` arrive here as *shape names* from
# ``prefab.face_blend_patterns`` and are reduced to the suffix this module
# registers shapes under, so one weight drives all five mouth meshes
# (``cf_O_face`` … ``o_tang``) the way Origin drives the five ``FBSTarget`` rows.
_PATTERN_SETS: dict = {}
_EYESET: dict | None = None
# 2: the ``cha_eyeset`` map rides along — the card's ``eyesPtn`` is a row id in
#    that list, not the number ``FBSCtrlEyes`` wants (§7.137).
PATTERN_VERSION = 2


def pattern_file():
    from ..bake import bake_dir, FACE_PTN_FILE
    return bake_dir() / FACE_PTN_FILE


def _read_eyeset() -> dict:
    """Import side — ``cha_eyeset`` reduced to what ``ChangeEyesPtn`` reads.

    ``{row id: {"ptn", "obj", "base", "hitomi", "tex", "name"}}``.  The row id
    is the card's ``eyesPtn``; ``ptn`` is the ``EyesPtn`` column, which is the
    number that actually reaches ``FBSCtrlEyes.ChangePtn``.  ``obj`` /
    ``base`` / ``hitomi`` are ``EyeObjNo`` / ``EyeBase`` / ``EyeHitomi`` — the
    renderer toggles for the gag-eye rows.
    """
    from .. import lists as L

    out: dict[int, dict] = {}
    for rid, row in (L.rows_by_category_no(2) or {}).items():
        def _i(key: str) -> int:
            try:
                return int(str(row.get(key, 0) or 0))
            except (TypeError, ValueError):
                return 0
        tex = str(row.get("MainTex") or "")
        out[int(rid)] = {
            "ptn": _i("EyesPtn"), "obj": _i("EyeObjNo"),
            "base": _i("EyeBase"), "hitomi": _i("EyeHitomi"),
            "tex": "" if tex in ("", "0", "-") else tex,
            "name": str(row.get("Name") or ""),
        }
    return out


def bake_face_patterns(dest=None) -> dict:
    """Import side — write the pattern table as **our** format.

    The table lives on ``bo_head_00.unity3d`` and the eye-set map in
    ``list/characustom``; a released build must not open either
    (``CLAUDE.md`` 四之一), so the runtime reads only the JSON this writes.
    Small enough to ship (three slots, 88 patterns, 39 eye-set rows).
    """

    from .. import prefab as pf

    dest = dest or pattern_file()
    table = pf.face_blend_patterns()
    if not table:
        return {"ok": False, "error": "no_head_bundle", "slots": 0}
    eyeset = _read_eyeset()
    from ...core.paths import write_json_atomic
    write_json_atomic(dest, {"version": PATTERN_VERSION, "slots": table,
                             "eyeset": {str(k): v
                                        for k, v in sorted(eyeset.items())}},
                      separators=(",", ":"))
    remapped = sorted(k for k, v in eyeset.items() if v["ptn"] != k)
    return {
        "ok": True,
        "slots": {k: len((v.get("targets") or [{}])[0].get("patterns") or ())
                  for k, v in table.items()},
        "targets": {k: [t.get("object") for t in (v.get("targets") or [])]
                    for k, v in table.items()},
        # The rows where the card's number is *not* the controller's number.
        # Zero of them would mean the table failed to load, not that the two
        # agree — locally it is 18 of 39.
        "eyeset": len(eyeset), "eyeset_remapped": remapped,
        "file": str(dest),
    }


def _raw_pattern_table() -> dict:
    """Baked JSON first; the bundle only as a dev-machine fallback."""
    import json

    p = pattern_file()
    if p.is_file():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if int(d.get("version") or 0) == PATTERN_VERSION:
                return dict(d.get("slots") or {})
        except Exception as exc:                     # noqa: BLE001
            print(f"[face_morph] face_ptn read fail: {exc}")
    from .. import prefab as pf

    return pf.face_blend_patterns() or {}


def eyeset_table() -> dict:
    """``{row id: row}`` for ``cha_eyeset`` — baked JSON first, bundle after.

    Empty when neither is readable, and the caller reports that rather than
    assuming the row id *is* the pattern (``CLAUDE.md`` 一之一).  It is not:
    of the 39 base rows, 18 name a different ``EyesPtn`` — 瞳に星 and
    瞳にハート are eye pattern **2**, 目に炎 is **10**, and the six
    ぎゅっと閉じ rows are **22‥27**.
    """
    global _EYESET
    if _EYESET is None:
        import json

        got: dict = {}
        p = pattern_file()
        if p.is_file():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if int(d.get("version") or 0) == PATTERN_VERSION:
                    got = {int(k): v for k, v in (d.get("eyeset") or {}).items()}
            except Exception as exc:                 # noqa: BLE001
                print(f"[face_morph] eyeset read fail: {exc}")
        if not got:
            try:
                got = _read_eyeset()
            except Exception as exc:                 # noqa: BLE001
                print(f"[face_morph] eyeset bundle read fail: {exc}")
                got = {}
        _EYESET = got
    return _EYESET


def eyes_pattern(eyes_ptn: int) -> tuple[int, dict]:
    """The card's ``eyesPtn`` → ``(FBS pattern, the eye-set row)``.

    ``ChaControl.ChangeEyesPtn`` looks the card's number up in ``cha_eyeset``
    (CategoryNo 2) and hands ``FBSCtrlEyes.ChangePtn`` the row's ``EyesPtn``
    column — the two are equal only for rows 0‥20.  A number with no row falls
    back to row 0 the way the IL does (``fileStatus.eyesPtn = 0`` then a second
    ``TryGetValue``); with no table at all the row comes back empty and the
    pattern is the card's own number, which the caller should report.
    """
    table = eyeset_table()
    row = table.get(int(eyes_ptn))
    if row is None and table:
        row = table.get(0)
    if not row:
        return int(eyes_ptn), {}
    return int(row.get("ptn", eyes_ptn)), row


def pattern_table(slot: str = "mouth") -> dict | None:
    """``{ptn: (close_suffix, open_suffix)}`` for ``eyebrow`` / ``eyes`` / ``mouth``.

    ``None`` when neither the baked table nor the head bundle is there — the
    caller reports that rather than substituting a table (``CLAUDE.md`` 一之一).
    """
    if slot not in _PATTERN_SETS:
        table = _raw_pattern_table()
        for key, data in table.items():
            targets = data.get("targets") or []
            n = max((len(t.get("patterns") or ()) for t in targets), default=0)
            rows: dict[int, tuple[str, str]] = {}
            per_target: dict[str, dict] = {}
            for i in range(n):
                # Every target agrees on the suffix — the five mouth meshes all
                # name pattern 21 ``*_name_op``.  Take the first that has one.
                close = open_ = ""
                for t in targets:
                    ps = t.get("patterns") or ()
                    if i >= len(ps):
                        continue
                    c, o = ps[i]
                    close = close or (suffix_of(c) if c else "")
                    open_ = open_ or (suffix_of(o) if o else "")
                    per_target.setdefault(str(t.get("object") or ""), {})[i] = (c, o)
                rows[i] = (close, open_)
            _PATTERN_SETS[key] = {
                "patterns": rows,
                # The meshes this slot is allowed to touch.  Not decoration:
                # ``def_cl`` is *both* the eye group's closed lids (10 meshes,
                # what ``rest_blink`` drives) and the mouth's closed lips, and
                # ``MorphRig.set`` with a bare suffix broadcasts to every set
                # that has it — so setting mouth pattern 21 through the bare
                # name would shut the character's eyes as a side effect.  Origin
                # has no such problem because ``FBSCtrlMouth.FBSTarget`` lists
                # exactly five meshes; :func:`apply_pattern` honours that list.
                "targets": [str(t.get("object") or "") for t in targets],
                # ``{mesh: {ptn: (close_name, open_name)}}`` — full shape names,
                # which is what :func:`apply_pattern` writes.
                "by_target": per_target,
                "open_min": float(data.get("open_min") or 0.0),
                "open_max": float(data.get("open_max") or 1.0),
                "fixed_rate": float(data.get("fixed_rate", -0.1)),
            }
        _PATTERN_SETS.setdefault(slot, None)
    return _PATTERN_SETS.get(slot)


def pattern_weights(ptn: int, *, slot: str = "mouth", open_rate: float = 0.0,
                    open_min: float | None = None,
                    open_max: float | None = None,
                    fixed_rate: float | None = None,
                    correct_open_max: float = -1.0) -> dict:
    """``{shape_suffix: weight 0..1}`` for one pattern — the formula above.

    *open_min* / *open_max* / *fixed_rate* default to the head prefab's own
    values; ``HFaceList`` overrides ``open_min`` per row (``openMinMouth``).
    ``fixed_rate`` is set by loading a card with ``mouthFixed``
    (``ChangeMouthOpenMax``); no H proc writes it (§7.396).  An empty dict
    means the pattern number is not in the table, which is a miss the caller
    should report.
    """
    table = pattern_table(slot)
    if not table:
        return {}
    pair = (table["patterns"] or {}).get(int(ptn))
    if pair is None:
        return {}
    omin = table["open_min"] if open_min is None else float(open_min)
    omax = table["open_max"] if open_max is None else float(open_max)
    fixed = table["fixed_rate"] if fixed_rate is None else float(fixed_rate)
    if correct_open_max >= 0.0:
        omax = float(correct_open_max)
    rate = omin + (omax - omin) * max(0.0, min(1.0, float(open_rate)))
    # ``>= 0``: the branch that skips this is ``0 > FixedRate``, so a fixed
    # rate of exactly **0** is a pinned-shut mouth, not "unset" (§7.137).
    if fixed >= 0.0:
        rate = fixed
    open_pct = int(max(0.0, min(100.0, rate * 100.0)))   # Origin truncates to int
    close_pct = 100 - open_pct
    close, open_ = pair
    out: dict[str, float] = {}
    if close:
        out[close] = out.get(close, 0.0) + close_pct / 100.0
    if open_:
        out[open_] = out.get(open_, 0.0) + open_pct / 100.0
    return out


def pattern_paths(ptn: int, *, slot: str = "mouth", **kw) -> dict:
    """``{"<mesh>/<full shape name>": weight}`` for one pattern.

    Mesh-qualified rather than suffix-keyed, for the reason in
    :func:`pattern_table`'s ``targets`` note — and the key is the *mesh* name,
    not our part name, because the caller may not have the rig yet.  Empty when
    the pattern is not in the table, which is a miss worth reporting.
    """
    weights = pattern_weights(int(ptn), slot=slot, **kw)
    table = pattern_table(slot) or {}
    pair = (table.get("patterns") or {}).get(int(ptn))
    if not weights or pair is None:
        return {}
    close_w, open_w = weights.get(pair[0], 0.0), weights.get(pair[1], 0.0)
    out: dict[str, float] = {}
    for mesh, ptns in (table.get("by_target") or {}).items():
        names = ptns.get(int(ptn)) or ptns.get(str(ptn))
        if not names:
            continue
        for shape, w in ((names[0], close_w), (names[1], open_w)):
            if shape:
                out[f"{mesh}/{shape}"] = float(w)
    return out


def tongue_out(ptn: int) -> bool:
    """``ChangeMouthPtn`` flips ``fileStatus.tongueState`` exactly on 21.

    22 (``kuwae``) keeps the tongue in but is grouped with 21 by the other two
    calls in that method: ``UseAdjustWidthScale(!(21 or 22))`` and
    ``DisableShapeMouth(21 <= ptn <= 22)``.
    """
    return int(ptn) == 21


def holds_in_mouth(ptn: int) -> bool:
    """Whether mouth pattern *ptn* is the one that holds something in the mouth (``kuwae``, 22).

    ``ChangeMouthPtn`` groups the two oral patterns by ``DisableShapeMouth(21 <= ptn <= 22)``;
    of the two, the one whose tongue stays in (:func:`tongue_out` is the other) is the mouth
    closed round what it holds.
    """
    return 21 <= int(ptn) <= 22 and not tongue_out(ptn)


# ---------------------------------------------------------------------------
# The face the card was saved wearing
# ---------------------------------------------------------------------------
# The last thing ``<ChangeHeadAsync>`` does, once ``fbsCtrl`` is on the head:
#
#     ChangeEyebrowPtn(status.eyebrowPtn, true)
#     ChangeEyebrowOpenMax(status.eyebrowOpenMax)
#     ChangeEyesPtn(status.eyesPtn, true)
#     ChangeEyesOpenMax(status.eyesOpenMax)
#     ChangeMouthPtn(status.mouthPtn, true)
#     ChangeMouthOpenMax(status.mouthOpenMax)
#     fbsCtrl.EyeLookController = eyeLookCtrl
#     ChangeEyesBlinkFlag(status.eyesBlink)
#
# so a character's saved expression is not a maker-only nicety — it is the face
# it wears the moment it loads, in every scene.  What each call leaves behind:
#
#   ChangeEyebrowOpenMax(v)  OpenMax  = Clamp01(v);  if !eyesBlink: openRate = 1
#   ChangeEyesOpenMax(v)     OpenMax  = Clamp(v, 0, 0.92)
#                            if !eyesBlink: openRate = Clamp01(v)
#   ChangeMouthOpenMax(v)    OpenMax  = Clamp01(v)
#                            if mouthFixed: FixedRate = v
#   ChangeEyesBlinkFlag(b)   BlinkCtrl.SetFixedFlags(!b)
#                            if !b: eyes.openRate = 1; eyebrow.openRate = 1
#
# **The ``openRate`` ``ChangeEyesOpenMax`` writes is dead.**  It only runs when
# ``!eyesBlink``, and ``ChangeEyesBlinkFlag`` runs after it in exactly that
# branch and puts 1 back.  So both slots rest at ``openRate = 1`` whether the
# character blinks or not, and the resting rate is simply ``OpenMax`` — which
# for the eyes is ``eyes_open_max`` (already wired) and for the eyebrow is
# ``eyebrowOpenMax`` (1.0 on all 1,388 local cards).
#
# The mouth is the odd one: its ``openRate`` is the *voice*, so a silent
# character sits at ``OpenMin`` — and ``mouthOpenMax`` would never reach the
# face at all if it were not for ``FixedRate``, which ``mouthFixed`` turns on
# for 1,385 of the 1,388 cards here.  ``mouthOpenMax`` is therefore the mouth's
# whole contribution, and it is 0 on 87.2% of them (§7.137).


@dataclass(frozen=True)
class CardFace:
    """``ChaFileStatus``'s expression block, resolved.

    The defaults are ``ChaFileStatus.MemberInit``'s, read rather than picked —
    and they are **not** what the cards on this machine hold.  The engine
    starts at ``mouthFixed = false`` / ``mouthOpenMax = 1``; the maker saves
    ``mouthFixed = true`` (1,385 of 1,388 local cards) with whatever rate the
    artist left, usually **0**.  A field missing from a card gets Illusion's
    number, not the common one (§7.137).
    """

    eyebrow_ptn: int = 0
    eyes_ptn: int = 0           # the card's own number: a ``cha_eyeset`` row id
    eyes_fbs_ptn: int = 0       # what that row hands ``FBSCtrlEyes``
    mouth_ptn: int = 0
    eyebrow_open_max: float = 1.0
    mouth_open_max: float = 1.0
    mouth_fixed: bool = False
    eyes_blink: bool = True
    tongue: bool = False
    gag_eye: str = ""           # the gag-eye map, when the row names one
    #: ``KK_EyeControl.DisableBlinking``: every ``ChangeEyesBlinkFlag(true)``
    #: arrives as false, the H rows' included
    blink_disabled: bool = False
    #: ``KK_EyeControl.EyeOpenMax``: every ``ChangeEyesOpenMax`` argument is
    #: capped here first — the H face's per-frame blend included (§7.450)
    eye_open_cap: float = 1.0

    @property
    def is_default(self) -> bool:
        return (self.eyebrow_ptn == 0 and self.eyes_fbs_ptn == 0
                and self.mouth_ptn == 0)


#: ``KK_EyeControl``'s card data (``KKEx``): ``EyeOpenMax`` / ``DisableBlinking``
EYE_CONTROL = "com.deathweasel.bepinex.eyecontrol"


def eye_control(card) -> tuple[float, bool]:
    """``(open cap, blinking disabled)`` from ``KK_EyeControl`` (§7.449).

    ``LoadExtendedSaveData`` starts from ``(1, false)`` and reads the two keys.
    Outside Studio its ``MainGameHooks`` are prefixes on ``ChaControl``::

        ChangeEyesOpenMax(ref v):   if v > EyeOpenMax: v = EyeOpenMax
        ChangeEyesBlinkFlag(ref b): if b: b = !DisableBlinking

    — so the cap bites before the game's own clamp, and "blink" can only ever
    be switched off.
    """
    cap, off = 1.0, False
    try:
        sec = ((card.sections.get("KKEx") or {}) if card is not None else {}
               ).get(EYE_CONTROL)
        data = sec[1] if isinstance(sec, (list, tuple)) and len(sec) > 1 else None
    except AttributeError:
        data = None
    if isinstance(data, dict):
        try:
            cap = float(data.get("EyeOpenMax", 1.0))
        except (TypeError, ValueError):
            cap = 1.0
        off = bool(data.get("DisableBlinking", False))
    return cap, off


def card_face(card) -> CardFace | None:
    """Read the card's saved expression.  ``None`` when it has no ``Status``.

    ``eyesOpenMax`` is deliberately absent: it is ``EyesCtrl.OpenMax``, which
    already arrives through ``body_head.eyes_open_max`` → ``built.eyes_open_max``
    and drives the lid rest and the gaze.  A second copy here would be a second
    owner of the same number.
    """
    status = getattr(card, "sections", None)
    status = (status or {}).get("Status") if isinstance(status, dict) else None
    if not status:
        return None

    def _f(key: str, default: float) -> float:
        try:
            return float(status.get(key, default))
        except (TypeError, ValueError):
            return default

    def _i(key: str) -> int:
        try:
            return int(status.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    eyes_row = _i("eyesPtn")
    fbs, row = eyes_pattern(eyes_row)
    mouth = _i("mouthPtn")
    return CardFace(
        eyebrow_ptn=_i("eyebrowPtn"), eyes_ptn=eyes_row, eyes_fbs_ptn=fbs,
        mouth_ptn=mouth,
        eyebrow_open_max=min(1.0, max(0.0, _f("eyebrowOpenMax", 1.0))),
        mouth_open_max=min(1.0, max(0.0, _f("mouthOpenMax", 1.0))),
        mouth_fixed=bool(status.get("mouthFixed", False)),
        # ``ChangeEyesBlinkFlag(status.eyesBlink)``, through KK_EyeControl
        eyes_blink=(bool(status.get("eyesBlink", True))
                    and not eye_control(card)[1]),
        blink_disabled=eye_control(card)[1],
        eye_open_cap=eye_control(card)[0],
        tongue=tongue_out(mouth),
        # ``EyeObjNo`` picks one of three gag-eye quads and ``EyeBase`` /
        # ``EyeHitomi`` hide the real eye behind it.  We have no gag-eye
        # object, so this is carried to be *reported*, not silently dropped.
        gag_eye=str(row.get("tex") or "") if int(row.get("obj") or 0) else "",
    )


def eye_close_paths(close: float, ptn: int = 0) -> dict:
    """``{"<mesh>/<shape>": weight}`` for the eye slot held *close* shut.

    Use this instead of ``rig.set("def_cl", …)``.  The bare suffix broadcasts
    to every set that carries it, and ``cf_O_mayuge`` carries one too
    (``mayuge.mayu00_def_cl``) — so the blink was closing the **eyebrows** 8%
    along with the lids, 0.371 mm on a stock head, on every character (§7.137).
    Origin cannot make that mistake: ``FBSCtrlEyes.FBSTarget`` lists the nine eye
    meshes and the brow belongs to ``FBSCtrlEyebrow``, which plays its **own**
    pattern at its **own** ``OpenMax`` (the card's ``eyebrowOpenMax``, 1.0 on
    every local card, so nothing at rest).  That is §7.130 二 again — scope by
    the controller's own target list.  The brow does share the blink's *rate*
    (``SyncBlink``, §7.351) — through :func:`rest_face_paths`'s
    ``eyebrow_open_rate``, never through this function.

    *close* is ``1 - rate``: ``blendshape.rest_blink`` at rest, driven to 1 by
    a blink.  ``ptn`` matters because the Close half is only ``def_cl`` for 16
    of the 28 eye patterns; patterns 2 and 4 close on ``egao_cl`` instead.
    """
    return pattern_paths(int(ptn), slot="eyes",
                         open_rate=1.0 - float(close),
                         open_min=0.0, open_max=1.0)


def set_eye_close(built, rig, close: float) -> int:
    """Hold this character's eye pair *close* shut, on the eye meshes only.

    The one place that knows both halves: the card's eye pattern (which shapes)
    and the caller's close weight (how far).  Falls back to the bare ``def_cl``
    broadcast when there is no pattern table at all — a character with resting
    lids and a slightly closed brow beats one staring wide open.
    """
    face = getattr(built, "card_face", None)
    ptn = int(getattr(face, "eyes_fbs_ptn", 0) or 0)
    n = write_paths(rig, eye_close_paths(float(close), ptn))
    if not n and rig is not None:
        rig.set("def_cl", float(close))
    return n


def write_paths(rig, paths: dict) -> int:
    """Write ``{"<mesh>/<shape>": w}`` onto *rig*, resolving our part prefix.

    Part names carry our own prefix (``origin_head_cf_O_face``), and a shape that
    the head does not have is skipped rather than invented.  Returns how many
    weights landed — 0 on a non-empty *paths* is a miss worth reporting.
    """
    if rig is None or not paths:
        return 0
    n = 0
    for path, w in paths.items():
        mesh, _, shape = path.partition("/")
        if not shape:
            continue
        for part_name, s in rig.sets.items():
            if part_name.endswith(mesh) and s.has(shape):
                rig.set(f"{part_name}/{shape}", float(w))
                n += 1
    return n


def rest_face_paths(face: CardFace | None, *,
                    eye_close: float | None = None,
                    eyebrow_open_rate: float = 1.0) -> dict:
    """``{"<mesh>/<shape>": weight}`` for the saved face standing still.

    *eye_close* is the eye pair's Close weight — ``1 - correctOpenMax``, i.e.
    ``blendshape.rest_blink``, blended with whatever is closing the lids this
    frame.  Passing ``None`` leaves the eye slot out entirely, which is what a
    caller that owns the blink itself wants.

    *eyebrow_open_rate* is the brow's ``openRate``: 1 at rest, and the blink's
    own rate while one is happening when the head's ``SyncBlink`` is set —
    ``FaceBlendShape.LateUpdate`` hands ``BlinkCtrl.GetOpenRate()`` to
    ``EyebrowCtrl.CalcBlend`` as well as to the eyes, and that stores it when
    ``SyncBlink`` (§7.351).  The mouth has no such driver: its rate is the
    card's ``FixedRate``, constant for this character.

    *face* may be ``None`` for a character with no ``Status`` block, and the
    answer is then the three slots at pattern 0 — Illusion's own defaults
    (``ChaFileStatus.MemberInit``), not "leave the face alone".  Pattern 0 is
    not a no-op: it is what holds the lids at their resting fraction.
    """
    face = face or CardFace()
    out: dict[str, float] = {}
    out.update(pattern_paths(face.eyebrow_ptn, slot="eyebrow",
                             open_rate=float(eyebrow_open_rate),
                             open_max=face.eyebrow_open_max))
    # ``openRate`` is the voice and nothing is speaking; ``FixedRate`` is the
    # card's ``mouthOpenMax`` when ``mouthFixed``, −1 (off) when it is not.
    out.update(pattern_paths(
        face.mouth_ptn, slot="mouth", open_rate=0.0,
        fixed_rate=face.mouth_open_max if face.mouth_fixed else -1.0))
    if eye_close is not None:
        out.update(eye_close_paths(float(eye_close), face.eyes_fbs_ptn))
    return out
