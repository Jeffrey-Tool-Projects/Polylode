"""KK_InvisibleBody (``KKEx["KK_InvisibleBody"]``) — a character drawn without her body.

Read from the local ``BepInEx/plugins/KK_Plugins/KK_InvisibleBody.dll``:

* **Card data.**  ``InvisibleBodyCharaController.OnReload``: ``Visible`` (a
  bool, default ``true``) out of the extended data, in every mode (Studio only
  skips reloads after the first).  ``Visible == true`` does nothing.
* **What** (``SetVisibleState`` → ``IterateVisible``): from
  ``BodyTop/p_cf_body_bone{,_low}/cf_j_root`` and the body's
  ``BodyTop/p_c{f,m}_body_00{,_low}/cf_o_root/``, every renderer gets
  ``enabled = Visible``; the walk does **not** go below a node named in
  ``InvisibleBody.AccessoryAttachPoints`` (the 54 ``a_n_*`` points) unless that
  node's parent is ``ct_hairB``.  So the body, the genitals, the head (it hangs
  in the bone tree) and the hair (at ``cf_J_FaceUp_ty``) go dark; the clothes
  (under ``BodyTop/ct_clothes*``, outside those roots) and the accessories
  (below an attach point) stay.

In our model that line is the part group: ``clothes`` and ``accessory`` keep
drawing, everything else the card dressed is hidden (hidden, not removed —
``Renderer.enabled = false`` leaves the object there, as
:func:`matedit.character_hidden_parts` does).  The parts are marked
``card_hidden``: Free H and life set ``visible = True`` on the body, face and
hair groups for every scene, and the card's choice has to outlive that.
"""
from __future__ import annotations

GUID = "KK_InvisibleBody"
#: groups that live outside the plugin's walk
KEEP_GROUPS = ("clothes", "accessory")


def visible(card) -> tuple[bool, str]:
    """``(Visible, reason)``; ``(True, "")`` without the section."""
    try:
        sec = ((card.sections.get("KKEx") or {}) if card is not None else {}).get(GUID)
    except AttributeError:
        return True, ""
    if sec is None:
        return True, ""
    data = sec[1] if isinstance(sec, (list, tuple)) and len(sec) >= 2 else sec
    if not isinstance(data, dict):
        return True, "隱形身體外掛的區段不是 (version, data) 這個形狀"
    v = data.get("Visible", True)
    if not isinstance(v, bool):
        return True, f"隱形身體外掛的 Visible 是 {v!r}，不是 bool"
    return v, ""


def apply(built, card, report=None) -> list[str]:
    """Hide what the plugin hides; returns the part names hidden."""
    on, reason = visible(card)
    if reason and report is not None:
        report.missing.append(reason)
    if on or built is None:
        return []
    done = []
    for part in list(getattr(built.model, "parts", []) or []):
        if getattr(part, "group", "") in KEEP_GROUPS:
            continue
        if not part.card_hidden:
            part.visible = False
            part.card_hidden = True
            done.append(str(part.name))
    if report is not None:
        report.replaced.append(
            f"隱形身體外掛：身體、頭與頭髮關掉 {len(done)} 片（衣服與配件留著）")
    return done
