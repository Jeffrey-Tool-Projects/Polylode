"""MaterialEditor (``com.deathweasel.bepinex.materialeditor``): which renderers
are drawn at all, and the copies, shader / queue, float / colour and texture
edits on their materials (the second half is at the end of this module).

The plugin lets a card author switch individual renderers off, and 992 of this
install's rows do exactly that (§7.120).  Nothing read them, so every one of
those renderers was drawn: the parts the author hid to make their own outfit
work came back, which is what "the ModPack characters look a bit wrong" is made
of — spare geometry and the poke-through it causes.

Everything here is transcribed from ``BepInEx/plugins/KK_Plugins/
KK_MaterialEditor.dll`` rather than inferred (``CLAUDE.md`` 一之二).

``MaterialEditorCharaController.<LoadData>d__30.MoveNext``
    ::

        for p in RendererPropertyList:
            if p.ObjectType == Clothing  and not clothes:     continue
            if p.ObjectType == Accessory and not accessories: continue
            if p.ObjectType == Hair      and not hair:        continue
            if p.ObjectType in (Clothing, Accessory) \\
               and p.CoordinateIndex != CurrentCoordinateIndex: continue
            if p.ObjectType == Character and not body:        continue
            MaterialAPI.SetRendererProperty(
                FindGameObject(p.ObjectType, p.Slot),
                p.RendererName, p.Property, p.Value)

    The coordinate filter is on **Clothing and Accessory only** — hair and the
    character itself are not per-outfit.

``MaterialEditorCharaController.FindGameObject``
    ::

        Clothing  -> chaCtrl.GetClothes(slot)
        Accessory -> AccessoriesApi.GetAccessoryObject(chaCtrl, slot)
        Hair      -> chaCtrl.GetHair(slot).gameObject
        Character -> chaCtrl.gameObject            <-- the whole character

``MaterialAPI.SetRendererProperty`` / ``SetRendererEnabled``
    ::

        if property == Enabled: SetRendererEnabled(go, name, value == 1)
        ...
        for r in go.GetComponentsInChildren<Renderer>(includeInactive: true):
            if r.NameFormatted() == name: r.enabled = enabled

    and ``GetRendererList(null)`` returns an **empty** list, so a row naming a
    slot that holds nothing is a no-op rather than an error.  So is a row whose
    ``RendererName`` matches nothing: ``SetRendererEnabled`` just returns false.
    Both matter, because rows outlive the item they were authored against —
    the plugin ships a ``PurgeOrphanedProperties`` hotkey for exactly that.
    Measured over 400 cards, of the 233 accessory rows that survive the
    coordinate filter, **208 (89.3%) name a renderer we resolve**, 14 name one
    that is in no prefab we load (orphans, and no-ops in the game too) and 11
    sit in ``moreAccessories`` slots we do not load at all (§7.138).

``Extensions.NameFormatted``
    ``name.Replace("(Instance)", "").Replace(" Instance", "").Trim()`` — Unity
    appends those to instantiated objects, and the stored name never has them.

Of the renderer rows only ``RendererProperties.Enabled`` is read.  The other
four (``ShadowCastingMode``, ``ReceiveShadows``, ``RecalculateNormals``,
``UpdateWhenOffscreen``) have no equivalent in this renderer.  Of the material
lists, keywords and material renames are not applied, nor are the texture,
copy and normal-map rows listed in ``PROGRESS.md`` F-ME.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

#: ``KKEx`` section key, from the plugin's ``GUID``.
EXT_KEY = "com.deathweasel.bepinex.materialeditor"

#: ``KK_Plugins.MaterialEditor.ObjectType``
UNKNOWN, CLOTHING, ACCESSORY, HAIR, CHARACTER = 0, 1, 2, 3, 4
#: ``KK_Plugins.MaterialEditor.RendererProperties``
ENABLED = 0


def name_formatted(name: str) -> str:
    """``Extensions.NameFormatted`` — how the plugin spells a renderer's name."""
    return str(name or "").replace("(Instance)", "").replace(" Instance", "").strip()


@dataclass(frozen=True)
class HiddenRenderers:
    """Which renderers this card switched off, keyed the way Origin looks them up.

    ``by_object`` is ``{(ObjectType, Slot): {name, …}}`` for the three
    per-object kinds; ``character`` is the ``ObjectType.Character`` rows, which
    the game applies to **every** renderer under the character and so are not
    slot-scoped.
    """

    by_object: dict = field(default_factory=dict)
    character: frozenset = frozenset()
    rows: int = 0                 # Enabled rows that survived the filters
    skipped_coordinate: int = 0   # rows belonging to another outfit
    reason: str = ""              # non-empty when the section could not be read
    #: ``{(ObjectType, Slot): {name, …}}`` of the rows that switch a renderer
    #: **on** (``value == 1``) — a no-op unless the prefab ships it disabled
    shown: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.by_object or self.character)

    def for_object(self, obj_type: int, slot: int) -> frozenset:
        return self.by_object.get((int(obj_type), int(slot)), frozenset())

    def shown_for(self, obj_type: int, slot: int) -> frozenset:
        return self.shown.get((int(obj_type), int(slot)), frozenset())

    def hides(self, obj_type: int, slot: int, name: str) -> bool:
        n = name_formatted(name)
        if not n:
            return False
        return n in self.for_object(obj_type, slot) or n in self.character


_EMPTY = HiddenRenderers()


def read(card, coordinate: int = 0) -> HiddenRenderers:
    """The card's switched-off renderers for the coordinate being worn.

    An unreadable section comes back as an empty result with a ``reason`` the
    caller reports; it does not come back as "this card hides nothing", because
    those two look identical on screen and only one of them is a bug
    (``CLAUDE.md`` 一之一).
    """
    from .bonemod import _msgpack_lz4

    try:
        ext = (card.sections.get("KKEx") or {}) if card is not None else {}
    except AttributeError:
        return _EMPTY
    data = ext.get(EXT_KEY)
    if data is None:
        return _EMPTY
    if not isinstance(data, (list, tuple)) or len(data) < 2:
        return HiddenRenderers(reason="MaterialEditor 區段不是 (version, data) 這個形狀")
    blob = (data[1] or {}).get("RendererPropertyList")
    if not blob:
        return _EMPTY
    try:
        rows = _msgpack_lz4(blob)
    except Exception as exc:                                 # noqa: BLE001
        return HiddenRenderers(reason=f"MaterialEditor 的 RendererPropertyList 解不開（{exc}）")
    if not isinstance(rows, list):
        return HiddenRenderers(reason="RendererPropertyList 不是列表")

    by_object: dict = {}
    shown: dict = {}
    character: set = set()
    kept = other_coord = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            if int(row.get("Property", -1)) != ENABLED:
                continue
            # ``SetRendererProperty`` does ``Int32.Parse(value)`` and then
            # ``SetRendererEnabled(go, name, value == 1)``: 1 switches the
            # renderer on, any other number off.
            on = int(str(row.get("Value", "")).strip()) == 1
        except (TypeError, ValueError):
            continue
        try:
            obj = int(row.get("ObjectType", UNKNOWN))
            slot = int(row.get("Slot", -1))
        except (TypeError, ValueError):
            continue
        if obj in (CLOTHING, ACCESSORY):
            try:
                if int(row.get("CoordinateIndex", 0)) != int(coordinate):
                    other_coord += 1
                    continue
            except (TypeError, ValueError):
                continue
        name = name_formatted(row.get("RendererName"))
        if not name:
            continue
        if on:
            if obj in (CLOTHING, ACCESSORY, HAIR):
                shown.setdefault((obj, slot), set()).add(name)
            continue
        kept += 1
        if obj == CHARACTER:
            character.add(name)
        elif obj in (CLOTHING, ACCESSORY, HAIR):
            by_object.setdefault((obj, slot), set()).add(name)
    return HiddenRenderers(
        {k: frozenset(v) for k, v in by_object.items()},
        frozenset(character), kept, other_coord,
        shown={k: frozenset(v) for k, v in shown.items()})


def renderer_name(piece) -> str:
    """A piece's renderer name, as :mod:`prefab` stashed it in the material dict."""
    return name_formatted((getattr(piece, "shader", None) or {}).get("_rendName"))


def unity_drawn(pieces, hidden: HiddenRenderers | None,
                obj_type: int, slot: int) -> tuple[list, int, int]:
    """``(kept, inactive, disabled)``: drop what Unity itself never draws (§7.509).

    Unity draws a renderer while its GameObject and every ancestor are active
    and the renderer is enabled.  :mod:`prefab` records the two ways a
    prefab ships one that fails that — ``_offNode``, an inactive node that
    none of Origin's run-time switches (clothes states, ``objOpt``,
    ``objSleeves``, an accessory's ``rendAlpha``) ever turns on, and
    ``_rendOff``, ``m_Enabled`` 0 — and nothing in ``ChaControl`` sets
    ``Renderer.enabled``, so both stay off in the game.  The one thing that
    turns a disabled renderer on is a MaterialEditor ``Enabled 1`` row
    (``SetRendererEnabled`` sets ``enabled``, never ``SetActive``), so that
    reaches ``_rendOff`` and not ``_offNode``.  An archive baked before the
    fields existed has neither and keeps every piece.
    """
    on = hidden.shown_for(obj_type, slot) if hidden is not None else frozenset()
    keep, inactive, disabled = [], 0, 0
    for p in pieces or ():
        sh = getattr(p, "shader", None) or {}
        if sh.get("_offNode"):
            inactive += 1
            continue
        if sh.get("_rendOff") and renderer_name(p) not in on:
            disabled += 1
            continue
        keep.append(p)
    return keep, inactive, disabled


def filter_pieces(pieces, hidden: HiddenRenderers | None,
                  obj_type: int, slot: int) -> tuple[list, int, list, bool]:
    """``(kept, dropped, unmatched, nameless)`` for one object's pieces.

    *unmatched* is the names the card asked to hide that no piece carries —
    the caller reports them rather than assuming they were handled, because a
    silently-ignored row and a correctly-applied one look the same on screen.
    A renderer with several materials is several pieces sharing one name, so
    dropping by name drops all of its passes, which is what disabling the
    renderer does in Unity.

    *nameless* is true when **no** piece of this item carries a renderer name
    at all, which does not mean the rows are orphans — it means the item came
    out of an archive baked before :mod:`prefab` started writing the name.
    The two produce the same empty match and want opposite responses (re-bake
    versus ignore), so they are reported apart.
    """
    if not hidden or not pieces:
        return list(pieces or []), 0, [], False
    # Slot-scoped only.  ``ObjectType.Character`` rows are handed the whole
    # character by ``FindGameObject`` and so belong to a pass over every part,
    # not to any one slot — see :func:`character_hidden_parts`.
    want = set(hidden.for_object(obj_type, slot))
    if not want:
        return list(pieces), 0, [], False
    keep, dropped, seen = [], 0, set()
    for p in pieces:
        n = renderer_name(p)
        if n:
            seen.add(n)
        if n and n in want:
            dropped += 1
            continue
        keep.append(p)
    return keep, dropped, sorted(want - seen), not seen


def character_hidden_parts(built, hidden: HiddenRenderers | None) -> tuple[list, list]:
    """``(parts hidden, names nothing matched)`` for the ``Character`` rows.

    ``FindGameObject(Character)`` is ``chaCtrl.gameObject``, so these rows are
    matched against **every** renderer on the character — body, head, clothes,
    accessories alike — and one name legitimately hides several parts.  Locally
    they are all head meshes plus one ``o_body_a``: cards that switch off their
    own eyebrows, eyelines, irises, teeth or, in one case, the whole body.

    A part is hidden rather than removed, because ``Renderer.enabled = false``
    is what the plugin does: the object is still there, and the rest of the
    pipeline (masks, morphs, the expression rig) still has something to hold.
    The part is marked ``card_hidden`` so no scene's "show the body" brings it
    back.
    """
    if not hidden or not hidden.character or built is None:
        return [], []
    want = set(hidden.character)
    done, seen = [], set()
    for part in list(getattr(built.model, "parts", []) or []):
        n = name_formatted(getattr(part, "origin_renderer", ""))
        if not n:
            continue
        seen.add(n)
        if n in want and not part.card_hidden:
            part.visible = False
            part.card_hidden = True
            done.append(str(part.name))
    return done, sorted(want - seen)


# ---------------------------------------------------------------------------
# Material rows: float / colour / render queue (``PROGRESS.md`` F-ME3, F-ME4)
#
# ``<LoadData>d__30.MoveNext`` applies the material lists with the renderer
# rows' ObjectType / CoordinateIndex filter, in this order:
# ``MaterialCopyList`` (first of all), ``MaterialShaderList`` (``SetShader`` then ``SetRenderQueue``), …,
# ``MaterialFloatPropertyList`` (skipped when ``CheckBlacklist``), keywords,
# ``MaterialColorPropertyList``, textures.  Every setter goes through
# ``MaterialAPI.GetObjectMaterials(go, materialName)`` — each material of each
# renderer under the object whose ``NameFormatted()`` equals the row's name —
# and ``SetFloat`` / ``SetColor`` write ``"_" + Property`` only
# ``if material.HasProperty(...)``.
#
# ``MaterialAPI.CopyMaterial(go, name, copyName)``: for every renderer under the
# object that has a material called *name* and no material called *copyName*
# yet, ``Instantiate`` that material, name it *copyName* (``name.MECopyN``) and
# **append** it to ``renderer.materials``.  A material past the last submesh
# draws the last submesh again, so a copy is a second pass over that geometry
# with the original's settings — until later rows edit it by its own name.
#
# ``MaterialEditorPlugin.CheckBlacklist(materialName, property)`` is true for
# ``alpha_a`` / ``alpha_b`` on every material, and for ``AlphaMask`` on the body
# material (``^c[mf]_m_body(\.MECopy\d+)?$``).
# ---------------------------------------------------------------------------

#: The ``Queue`` tag of each shader the plugin can swap *to* on its own —
#: read out of the ``KK_Shaders.unity3d`` it embeds (``KK_MaterialEditor.dll``
#: resource ``KK_Plugins.Resources.KK_Shaders.unity3d``, first SubShader),
#: 2000 (Geometry) where the shader carries no tag.  ``MaterialAPI.SetShader``
#: assigns ``material.shader``, which puts the material back on the new
#: shader's queue, and only then applies the row's own ``RenderQueue``.
SWAP_SHADER_QUEUE = {
    "main_alpha": 3040, "main_color": 2000, "main_emblem": 2450,
    "main_emblem_clothes": 2450, "main_hair": 2000, "main_hair_front": 3040,
    "main_item": 2450, "main_item_ditherd": 2000, "main_item_emission": 2450,
    "main_item_studio": 2450, "main_item_studio_add": 3000,
    "main_item_studio_alpha": 3000, "main_opaque": 2450, "main_opaque2": 2450,
    "main_skin": 2350, "main_StandardMDK_studio": 2450, "main_texture": 2450,
    "main_texture_studio": 3000, "mnpb": 2000, "toon_eye_lod0": 3000,
    "toon_eyew_lod0": 3000, "toon_glasses_lod0": 4010, "toon_nose_lod0": 3000,
    "toon_textureanimation": 4450,
}

#: ``MaterialEditorPlugin.EyeMaterials`` (its static constructor): a
#: character-level shader swap on these keeps the material's queue
#: (``SetShader(..., preserveRenderQueue: true)``).
EYE_MATERIALS = frozenset({
    "cf_m_eyeline_00_up", "cf_m_eyeline_down", "cf_m_eyeline_kage",
    "cf_m_hitomi_00", "cf_m_mayuge_00", "cf_m_sirome_00",
})

#: ``MaterialTexturePropertyList.Property`` → the :class:`prefab.Piece` field
#: holding that map.  ``SetTextureWithProperty`` hands the card's own image
#: (``TextureDictionary[TexID]``) to ``SetTexture(go, material, property)``
#: after the game has built the material, so it replaces whatever was bound —
#: including a ``create_*`` composite.  ``NormalMap`` is left out: the plugin
#: first runs it through its own ``normal_convert`` shader
#: (``ConvertNormalMap``), which is not transcribed; those rows are counted.
TEXTURE_FIELDS = {
    "MainTex": "texture", "ColorMask": "mask", "DetailMask": "detail",
    "AlphaMask": "alpha_mask", "AnotherRamp": "ramp", "LineMask": "line_mask",
}

#: The texture rows ``body_head`` applies on the character itself (§7.385,
#: §7.386): ``_MainTex`` per material, and the face's ``_AlphaMask``.
CHARACTER_TEXTURES = frozenset({"MainTex", "AlphaMask"})

_BLACKLIST_ALWAYS = frozenset({"alpha_a", "alpha_b"})
_BODY_MATERIAL = re.compile(r"^c[mf]_m_body(\.MECopy\d+)?$")


def blacklisted(material: str, prop: str) -> bool:
    """``MaterialEditorPlugin.CheckBlacklist``."""
    if prop in _BLACKLIST_ALWAYS:
        return True
    return bool(_BODY_MATERIAL.match(str(material))) and prop == "AlphaMask"


@dataclass(frozen=True)
class MaterialEdits:
    """The card's float / colour / queue rows for the coordinate being worn.

    ``rows`` maps ``(ObjectType, Slot)`` — ``(CHARACTER, 0)`` for the rows the
    game hands the whole character — to ``(material name, kind, property,
    value)`` in the plugin's application order: shader swaps and queues, then
    floats, then colours.  A swap to a shader outside :data:`SWAP_SHADER_QUEUE`
    (another plugin's — ``Goo``, ``IBL_Shader``, …) is not applied and is
    counted in ``foreign_shaders`` for the caller to report.
    """

    rows: dict = field(default_factory=dict)
    foreign_shaders: tuple = ()
    reason: str = ""
    # ``TextureDictionary``: TexID → the image file's bytes
    textures: dict = field(default_factory=dict)
    # texture rows not applied: ``{property: count}`` — properties outside
    # :data:`TEXTURE_FIELDS`, and on the character itself everything but
    # ``MainTex`` (``"character <property>"``): ``body_head`` builds the body
    # and face maps from the create props, not from the piece's fields
    unapplied_textures: dict = field(default_factory=dict)
    _decoded: dict = field(default_factory=dict, compare=False, repr=False)
    # ``MaterialCopyList`` rows on the character itself, which are not applied:
    # the body and head loaders key their special cases on mesh names, and a
    # second piece carrying ``o_body_a``'s name would replace the body part
    character_copies: int = 0

    def __bool__(self) -> bool:
        return bool(self.rows)

    def for_object(self, obj_type: int, slot: int) -> list:
        key = (int(obj_type), 0 if int(obj_type) == CHARACTER else int(slot))
        return self.rows.get(key, [])


def _color(value) -> tuple | None:
    """``Extensions.ToColor``; msgpack already hands a serialised ``Color`` as a list."""
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = str(value or "").split(",")
    if len(parts) < 3:
        return None
    vals = [float(v) for v in parts[:4]]
    if len(vals) == 3:
        vals.append(1.0)
    return tuple(vals)


def _row_scope(row: dict, coordinate: int) -> tuple | None:
    obj = int(row.get("ObjectType", UNKNOWN))
    if obj not in (CLOTHING, ACCESSORY, HAIR, CHARACTER):
        return None
    if obj in (CLOTHING, ACCESSORY) and int(row.get("CoordinateIndex", 0)) != int(coordinate):
        return None
    return (obj, 0 if obj == CHARACTER else int(row.get("Slot", -1)))


def read_material_edits(card, coordinate: int = 0) -> MaterialEdits:
    """The card's material rows that survive ``LoadData``'s filters."""
    from .bonemod import _msgpack_lz4

    try:
        ext = (card.sections.get("KKEx") or {}) if card is not None else {}
    except AttributeError:
        return MaterialEdits()
    data = ext.get(EXT_KEY)
    if not isinstance(data, (list, tuple)) or len(data) < 2:
        return MaterialEdits()
    lists = data[1] or {}
    rows: dict = {}
    foreign: list = []
    char_copies = 0
    unapplied: dict = {}

    def rows_of(key: str) -> list:
        blob = lists.get(key)
        got = _msgpack_lz4(blob) if blob else []
        return [r for r in got if isinstance(r, dict)] if isinstance(got, list) else []

    try:
        for row in rows_of("MaterialCopyList"):
            key = _row_scope(row, coordinate)
            copy = name_formatted(row.get("MaterialCopyName"))
            if key is None or not copy:
                continue
            if key[0] == CHARACTER:
                char_copies += 1
                continue
            rows.setdefault(key, []).append(
                (name_formatted(row.get("MaterialName")), "copy", "", copy))
        for row in rows_of("MaterialShaderList"):
            key = _row_scope(row, coordinate)
            if key is None:
                continue
            mat = name_formatted(row.get("MaterialName"))
            shader = str(row.get("ShaderName") or "")
            if shader:
                leaf = shader.split("/")[-1]
                if shader.startswith("Shader Forge/") and leaf in SWAP_SHADER_QUEUE:
                    keep = key[0] == CHARACTER and mat in EYE_MATERIALS
                    rows.setdefault(key, []).append(
                        (mat, "shader", "keep" if keep else "", shader))
                else:
                    foreign.append(shader)
            if row.get("RenderQueue") is not None:
                rows.setdefault(key, []).append(
                    (mat, "queue", "", int(row["RenderQueue"])))
        for row in rows_of("MaterialFloatPropertyList"):
            key = _row_scope(row, coordinate)
            mat = name_formatted(row.get("MaterialName"))
            prop = str(row.get("Property") or "")
            if key is None or not prop or blacklisted(mat, prop):
                continue
            rows.setdefault(key, []).append((mat, "float", prop, float(row.get("Value"))))
        for row in rows_of("MaterialColorPropertyList"):
            key = _row_scope(row, coordinate)
            prop = str(row.get("Property") or "")
            col = _color(row.get("Value"))
            if key is None or not prop or col is None:
                continue
            rows.setdefault(key, []).append(
                (name_formatted(row.get("MaterialName")), "color", prop, col))
        for row in rows_of("MaterialTexturePropertyList"):
            key = _row_scope(row, coordinate)
            mat = name_formatted(row.get("MaterialName"))
            prop = str(row.get("Property") or "")
            if key is None or not prop or blacklisted(mat, prop):
                continue
            # ``SetTextureWithProperty`` then ``SetTextureOffset`` then
            # ``SetTextureScale``; each one is a no-op on a null value
            if row.get("TexID") is not None:
                why = (f"character {prop}"
                       if key[0] == CHARACTER and prop not in CHARACTER_TEXTURES
                       else None if prop in TEXTURE_FIELDS else prop)
                if why:
                    unapplied[why] = unapplied.get(why, 0) + 1
                else:
                    rows.setdefault(key, []).append(
                        (mat, "tex", prop, str(row["TexID"])))
            for kind in ("Offset", "Scale"):
                v = row.get(kind)
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    rows.setdefault(key, []).append(
                        (mat, kind.lower(), prop, (float(v[0]), float(v[1]))))
        texd = lists.get("TextureDictionary")
        if isinstance(texd, (bytes, bytearray)):
            texd = _msgpack_lz4(texd)
        textures = ({str(k): bytes(v) for k, v in texd.items()
                     if isinstance(v, (bytes, bytearray))}
                    if isinstance(texd, dict) else {})
    except (TypeError, ValueError, KeyError) as exc:
        return MaterialEdits(reason=f"MaterialEditor 的材質列解不開（{exc}）")
    return MaterialEdits(rows, tuple(foreign), character_copies=char_copies,
                         textures=textures, unapplied_textures=unapplied)


def _is_color(v) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) == 4
            and all(isinstance(c, float) for c in v))


def _copy_materials(pieces: list, copies: list) -> tuple[list, int, int]:
    """``CopyMaterial`` for each ``(material, copy name)``: ``(pieces, made, skipped)``.

    The copy is material *k* of its renderer (one past the last slot we
    have), and material *k* draws submesh ``min(k, submeshes - 1)`` — the whole
    mesh when there is one submesh.  The new piece takes that submesh's
    geometry from the piece already drawing it and the original material's
    dict under the copy's name.  When no piece draws that submesh (a renderer
    with fewer materials than submeshes: the geometry was never loaded) the
    copy is skipped and counted rather than drawn over the wrong triangles.
    """
    out, made, skipped = list(pieces), 0, 0
    for mat, copy in copies:
        by_rend: dict = {}
        for p in out:
            sh = getattr(p, "shader", None) or {}
            by_rend.setdefault(str(sh.get("_rendName") or ""), []).append(p)
        for group in by_rend.values():
            names = {name_formatted((p.shader or {}).get("_matName")) for p in group}
            if mat not in names or copy in names:
                continue
            src = next(p for p in group
                       if name_formatted((p.shader or {}).get("_matName")) == mat)
            k = 1 + max(int(getattr(p, "mat_slot", 0) or 0) for p in group)
            subs = int((src.shader or {}).get("_submeshes", 1) or 1)
            want = min(k, subs - 1)
            geo = next((p for p in group if subs == 1
                        or int(getattr(p, "mat_slot", 0) or 0) == want), None)
            if geo is None:
                skipped += 1
                continue
            out.append(geo._replace(shader={**(src.shader or {}), "_matName": copy},
                                    mat_slot=k))
            made += 1
    return out, made, skipped


def _texture(edits: MaterialEdits, tex_id: str):
    """``TextureDictionary[tex_id]`` decoded (sRGB, like the plugin's own
    ``Texture2D(2, 2, fmt, mip)`` + ``LoadImage``), or ``None``."""
    from .load_mesh import decode_image_bytes
    if tex_id not in edits._decoded:
        edits._decoded[tex_id] = decode_image_bytes(edits.textures.get(tex_id))
    return edits._decoded[tex_id]


def edit_pieces(pieces, edits: MaterialEdits | None, obj_type: int,
                slot: int) -> tuple[list, int, list, dict]:
    """``(pieces, values written, names nothing carried, counts)``.

    *counts* is ``copies`` (material copies made), ``uncopied`` (copies whose
    submesh was never loaded) and ``bad_textures`` (texture rows whose
    ``TexID`` is not in the dictionary or does not decode).

    A matching piece comes back as a copy with its own material dict: a baked
    item is shared between characters, and the edit belongs to this card.
    ``HasProperty`` is read as "the material's saved properties carry that name
    with that type" — Unity serialises every property the shader declares into
    ``m_SavedProperties``, and the type test is what keeps a row named
    ``shader`` off this project's own ``_shader`` entry.  The queue goes into
    ``_queue`` (``m_CustomRenderQueue``'s slot), which
    :func:`attach.apply_shader` reads first.
    """
    rows = edits.for_object(obj_type, slot) if edits else []
    if not rows or not pieces:
        return list(pieces or []), 0, [], {"copies": 0, "uncopied": 0, "bad_textures": 0}
    copies = [(mat, value) for mat, kind, _prop, value in rows if kind == "copy"]
    pieces, made, skipped = _copy_materials(list(pieces), copies)
    by_mat: dict = {}
    for mat, kind, prop, value in rows:
        if kind != "copy":
            by_mat.setdefault(mat, []).append((kind, prop, value))
    out, written, bad = [], 0, 0
    present = {name_formatted((getattr(p, "shader", None) or {}).get("_matName"))
               for p in pieces}
    for p in pieces:
        sh = getattr(p, "shader", None) or {}
        mat = name_formatted(sh.get("_matName"))
        todo = by_mat.get(mat) if mat else None
        if not todo:
            out.append(p)
            continue
        new = dict(sh)
        maps, map_cs = {}, dict(getattr(p, "map_cs", None) or {})
        for kind, prop, value in todo:
            key = "_" + prop
            if kind == "tex":
                fld = TEXTURE_FIELDS.get(prop)
                img = _texture(edits, value) if fld else None
                if img is None:
                    bad += 1
                    continue
                maps[fld] = img.px
                map_cs[fld] = bool(img.srgb)
                # ``_me<Property>``: this map is the card's, bound after the game
                # built the material — for ``MainTex`` on a garment or hair that
                # replaces the create pass's composite (``attach``)
                new["_me" + prop] = 1
                written += 1
                continue
            if kind in ("offset", "scale"):
                st = new.get(key + "_ST")
                if not (isinstance(st, (list, tuple)) and len(st) == 4):
                    continue
                st = list(st)
                if kind == "scale":
                    st[0], st[1] = value
                else:
                    st[2], st[3] = value
                new[key + "_ST"] = tuple(st)
                written += 1
                continue
            if kind == "shader":
                new["_shader"] = str(value)
                new["_shaderQueue"] = SWAP_SHADER_QUEUE[str(value).split("/")[-1]]
                if prop != "keep":
                    new["_queue"] = -1
            elif kind == "queue":
                new["_queue"] = int(value)
            elif kind == "float" and isinstance(new.get(key), float):
                new[key] = float(value)
            elif kind == "color" and _is_color(new.get(key)):
                new[key] = [float(c) for c in value]
            else:
                continue
            written += 1
        out.append(p._replace(shader=new, map_cs=map_cs, **maps) if maps
                   else p._replace(shader=new))
    wanted = set(by_mat) | {m for m, _c in copies}
    return out, written, sorted(wanted - present), {
        "copies": made, "uncopied": skipped, "bad_textures": bad}
