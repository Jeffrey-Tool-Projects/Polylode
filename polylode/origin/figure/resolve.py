"""Resolve card item ids → base AssetRef or zipmod ModRef (project cache)."""
from __future__ import annotations

from dataclasses import dataclass

from ..install import configured
from .. import lists as L
from .. import sideloader as SL

# Card accessory ``type`` → list category.  The card's type *is*
# ``ChaListDefine.CategoryNo``: ao_hair 121 … ao_kokan 130, straight off the
# enum.  Two entries used to be shifted (127 read as body, 130 as leg), which
# sent every leg accessory to the wrong list before the sibling-category sweep
# quietly found it again somewhere else.
ACC_TYPE_TO_CAT = {
    121: "acc_hair", 122: "acc_head", 123: "acc_face", 124: "acc_neck",
    125: "acc_body", 126: "acc_waist", 127: "acc_leg",
    128: "acc_arm", 129: "acc_hand", 130: "acc_kokan",
}

# Clothes list category, from ``ChaListDefine.CategoryNo`` (co_top 105 …
# co_shoes 112).  ``shoes_inner`` / ``shoes_outer`` are our own split of one
# list; both are 112.
CLOTHES_CAT_NO = {
    "top": 105, "bottom": 106, "bra": 107, "panty": 108, "gloves": 109,
    "legwear": 110, "socks": 111, "shoes": 112,
    "shoes_inner": 112, "shoes_outer": 112,
}

HAIR_SLOTS = ("hair_back", "hair_front", "hair_side", "hair_option")
HAIR_UAR = (
    "ChaFileHair.HairBack", "ChaFileHair.HairFront",
    "ChaFileHair.HairSide", "ChaFileHair.HairOption",
)
# ``<ChangeHairAsync>c__Iterator5``'s ``defId``: ``new int[5]`` with ``[1] = 1``,
# indexed by the hair kind and handed to ``LoadCharaFbxDataAsync`` as its
# ``defaultId`` — so a back hair nothing answers for is ロングヘア (id 0) and a
# front hair is id 1 (§7.490).  ``<ChangeHeadAsync>`` passes a literal 0, which
# :func:`resolve_head` deliberately does not follow.
HAIR_DEFAULT_ID = (0, 1, 0, 0)

_ENUM_NAME = {no: name for name, no in SL.CATEGORY_ENUM.items()}


def _net_int32(text) -> bool:
    """``Int32.TryParse(text)`` (``NumberStyles.Integer``): blanks around, a sign, digits, in range."""
    import re
    s = str(text or "")
    if not re.fullmatch(r"\s*[+-]?\d+\s*", s):
        return False
    return -2**31 <= int(s) <= 2**31 - 1


def keeps_stock_row(cat_no: int, row, main_ab: bool = True) -> bool:
    """Whether a field naming an uninstalled mod keeps its id for the stock *row* of that number.

    ``ResolveStructure``'s not-installed branch (``Sideloader.dll``)::

        if InternalDataList[category].ContainsKey(id):
            mainAB = row.dictInfo[MainAB]?.Replace("chara/", "").Replace(".unity3d", "")
                         .Replace(category.ToString() + "_", "").Replace("/", "")
            if int.TryParse(mainAB): id = 99999999        // (1 for a maker accessory
            else: keep the id                            //  with Keep missing accessories)
        else: id = 99999999

    — for **every** structure, not only accessories (``isinst PartsInfo`` only
    picks 1 over 99999999, §7.488).  A row in a numbered game bundle
    (``chara/co_top_00.unity3d`` → ``00``) is therefore *not* the answer; one in
    any other bundle — or with no ``MainAB`` at all — is.  ``False`` with no
    row.  *row*'s ``bundle`` is its ``MainAB`` whenever its table has that
    column (it comes first in ``lists.AB_COLUMNS``); *main_ab* says whether it
    does — five texture lists do not, and there ``TryGetValue`` misses, the
    string is null and the id is always kept (:func:`keeps_list_row`).
    """
    if row is None:
        return False
    if not main_ab:
        return True
    name = _ENUM_NAME.get(int(cat_no), "")
    ab = str(getattr(row, "bundle", "") or "")
    stripped = (ab.replace("chara/", "").replace(".unity3d", "")
                .replace(f"{name}_", "").replace("/", ""))
    return not _net_int32(stripped)


def keeps_list_row(category: str, ident: int) -> bool:
    """:func:`keeps_stock_row` for a row of our list *category*, from the list index alone.

    For the texture-only lists, whose ``MainAB`` column some tables do not
    have (``mt_pattern``, ``mt_emblem``, ``mt_ramp``, ``mt_hairgloss``,
    ``mt_eye_gradation``, §7.493).  A 「なし」 row is not a ref here and reads
    as not kept — its ``MainAB`` is ``0``, which parses.
    """
    idx = L.index()
    try:
        row = idx.get(category, int(ident))
    except (TypeError, ValueError):
        return False
    return keeps_stock_row(SL.CATEGORY_NO_OF.get(category, 0), row,
                           main_ab=category in getattr(idx, "main_ab", set()))


def _fbx_default(base, category: str, item_id: int, default_id: int, reason: str) -> "Resolved":
    """``LoadCharaFbxDataAsync``'s lookup once the id itself answers nothing: *default_id*, then 0.

    What the game draws for a hair or head it cannot find — never nothing, and
    never a mod (they sit at LocalSlots far above these numbers).  The reason
    stays on the answer so the report can still say the card's own choice is
    missing (一之一).
    """
    for alt in (default_id, 0):
        row = base.get(category, int(alt))
        if row is not None:
            return Resolved("base", category, item_id, ref=row,
                            reason=f"{reason} —— 照遊戲退到預設 #{alt}")
    return Resolved("missing", category, item_id, reason=reason)


@dataclass
class Resolved:
    kind: str                   # base | mod | combo | missing
    category: str
    item_id: int
    ref: object = None          # AssetRef | ModRef | None
    sub_parts: list | None = None
    reason: str = ""


def _base_index():
    inst = configured()
    if not inst.valid:
        return None, None
    return inst, L.index(inst)


def uar_named(prop: str, item_id, res: dict | None):
    """``(guid, slot)`` if the card's own resolution table names this field.

    **This has to be asked before the base list, not after.**  Sideloader's
    ``UniversalAutoResolver.ResolveStructure`` walks the loaded card and, for
    every field its UAR table names, *rewrites the id in place* to the mod's
    LocalSlot before ``ChaControl`` reads anything::

        var intResolve = TryGetResolutionInfo(ext.Slot, key.ToString(),
                                              key.Category, ext.GUID);
        if (intResolve != null)
            kv.Value.SetMethod(structure, intResolve.LocalSlot);

    Sideloader hands out LocalSlots from 100000000 up, so a base row can never
    answer for a named field — and the ids cards actually store are the mod's
    *own* slot numbers, which authors number from 1.  Collisions are therefore
    the normal case, not an edge one: over 183 local cards, 174 accessory slots
    and 42 garments in the coordinate the card actually wears carried a mod id
    that the base list also has, and asking the base list first turned every
    one of them into an unrelated item — DeathWeasel hair extensions rendered
    as 水泳ゴーグル, a Short Overall as a セーラータイプ top, a cleavage bra as a
    スクール水着 (§7.72).  73 of those 183 cards wore at least one.

    ``None`` means the card says nothing about the field, which is
    ``CompatibilityResolve``'s case: there the id is left alone, so the base
    list answers first and a (category, slot) scan only runs when it has no
    such row.  That is the order the callers below already use.

    The lookup falls back to the property alone because that is what the IL
    matches on (``FirstOrDefault(x => x.Property == searchProperty.property)``);
    the id-keyed hit is tried first only so a stale entry cannot outvote the id
    the card currently holds.
    """
    if not res:
        return None
    try:
        keyed = res.get((prop, int(item_id)))
    except (TypeError, ValueError):
        keyed = None
    return keyed or res.get((prop, None))


def row_kind(ref) -> int:
    """``ChaListDefine.KeyType.Kind`` (26) off a list row, 0 when absent.

    On a ``co_top`` row this is what says the garment is assembled from
    sub-parts rather than from its own mesh: ``<ChangeClothesTopPartsAsync>``
    reads ``libTop.Kind`` and only **1** (``セーラータイプ``) and **2**
    (``ジャケットタイプ``) select a ``cpo_*`` category — everything else has no
    parts.  ``なし`` is id 0 and carries no ``Kind`` at all.

    Reading "the row's MainData is ``p_dummy``" instead is nearly the same
    thing and breaks on exactly one row: id 0 is a ``p_dummy`` too, so a card
    wearing no top got a sailor uniform built out of ``subPartsId`` — which on
    those cards is ``[0, 0, 0]`` and would have loaded three real sub-parts.
    """
    try:
        return int((getattr(ref, "extra", None) or {}).get("Kind", 0) or 0)
    except (TypeError, ValueError):
        return 0


def resolve_clothes(category: str, item_id: int, *,
                    field: str = "", outfit: int = 0,
                    sub_parts=None, res: dict | None = None) -> Resolved:
    inst, base = _base_index()
    if base is None:
        return Resolved("missing", category, item_id, reason="no assets_cache")
    list_cat = category if category != "shoes_inner" and category != "shoes_outer" else "shoes"
    # normalize shoes_*
    if category.startswith("shoes"):
        list_cat = "shoes"

    mods = SL.index()
    res = res or {}
    gs = None
    if field:
        prop = f"outfit{outfit}.ChaFileClothes.{field}"
        gs = uar_named(prop, item_id, res)
        if gs:
            # Pin to ``ChaListDefine.CategoryNo``.  A pack that ships a whole
            # outfit numbers its garments from one run, so ``(guid, slot)``
            # alone is ambiguous and ``find`` answers with whichever category
            # it walks into first: over 100 cards, **125 of 799** slots came
            # back as the wrong garment — a skirt for gloves, shorts for socks,
            # a bra for panties (§7.66 五).  Only one slot in that sample
            # resolves at all without the pin, and it is not worth 125.
            mref = mods.find(gs[0], gs[1], CLOTHES_CAT_NO.get(list_cat, 0))
            if mref is not None:
                return Resolved("mod", list_cat, item_id, ref=mref)

    # No UAR entry: the id keeps the value the card stored, so whichever row
    # owns that number answers.  A UAR entry naming a mod that is not
    # installed only keeps it when the stock row sits outside a numbered game
    # bundle (:func:`keeps_stock_row`, checked above); §7.72 read the
    # ``isinst PartsInfo`` in that branch as "only accessories are cleared",
    # and it only picks 1 over 99999999 (§7.488).
    ref = base.get(list_cat, int(item_id))
    if gs and not keeps_stock_row(CLOTHES_CAT_NO.get(list_cat, 0), ref):
        # ``ResolveStructure`` set the id to 99999999, and every garment's
        # ``defaultId`` is 0 — 「なし」 in all seven lists (§7.477) — so the
        # slot is bare in the game; saying so beats drawing the stock row.
        return Resolved("missing", list_cat, item_id,
                        reason=f"{list_cat}#{item_id}:"
                               f" 未安裝 {gs[0]}#{gs[1]}")
    if ref is not None and list_cat == "top" and row_kind(ref) in (1, 2):
        return Resolved("combo", "top", item_id, ref=ref, sub_parts=list(sub_parts or []))
    if ref is not None and ref.asset not in ("p_dummy", "", "0", "-"):
        return Resolved("base", list_cat, item_id, ref=ref)
    if gs:
        # Named, not installed, and no stock row of that number either: nothing
        # owns the id.  The family sweep below is ``CompatibilityResolve``'s
        # path and must not run here — it would answer with an unrelated
        # garment that merely shares the number (§7.66 三 2).
        return Resolved("missing", list_cat, item_id,
                        reason=f"{list_cat}#{item_id}:"
                               f" 未安裝 {gs[0]}#{gs[1]}")
    if ref is not None:
        # A "なし" row — every clothing category has exactly one and it is id 0.
        # It **is** an answer: ``CompatibilityResolve`` stops at the base list,
        # so the (category, slot) sweep below must not run for it.  Letting it
        # run picks whichever mod happens to number an item 0, and plenty do —
        # 43 gloves, 106 shoes, 32 tops, 21 bottoms on this install.  A card
        # asking for bare hands came back wearing ``Serval Gloves``.
        return Resolved("base", list_cat, item_id, ref=ref)
    # ``CompatibilityResolve``: the same CategoryNo, the first claimant in
    # Sideloader's load order (``SideloaderIndex.find_any``, §7.478).  This used
    # to widen the category into a "family" (``legwear`` also took ``socks``)
    # and fall back to a substring match, walking the mods in file-path order —
    # none of which the IL does.
    mref = mods.find_any(list_cat, int(item_id))
    if mref is not None:
        return Resolved("mod", list_cat, item_id, ref=mref)
    return Resolved("missing", list_cat, item_id,
                    reason=f"no base/mod for {list_cat}#{item_id}")


# ``ChaControl.<ChangeHeadAsync>`` picks CategoryNo 100, or 501 when the
# character is male *and* exType is 1 (``ex_bo_head``, one row in the base
# install).  exType is not modelled here, so 100 is what gets asked for.
HEAD_CAT_NO = 100
HEAD_UAR = "ChaFileFace.headId"
# ``ChaListDefine.CategoryNo``: bo_hair_b / _f / _s / _o, in HAIR_SLOTS order.
HAIR_CAT_NO = (101, 102, 103, 104)


def resolve_head(head_id: int, *, res: dict | None = None) -> Resolved:
    """The card's ``fileFace.headId`` → base list row or zipmod head.

    Modded heads are the common case rather than an edge one: locally a
    quarter of the female cards ask for id 1, which no base list has --
    it is ``com.DeathWeasel.HeadNoEars``, shipped with BetterRepack.  The
    card's own UAR names the mod, so that is tried before any scan; a scan
    by (category, slot) alone can only pick whichever mod registered the id
    last, and heads are not interchangeable.
    """
    try:
        head_id = int(head_id)
    except (TypeError, ValueError):
        return Resolved("missing", "head", 0, reason="headId 不是整數")
    inst, base = _base_index()
    if base is None:
        return Resolved("missing", "head", head_id, reason="no assets_cache")
    mods = SL.index()
    gs = uar_named(HEAD_UAR, head_id, res)
    if gs:
        mref = mods.find(gs[0], gs[1], HEAD_CAT_NO)
        if mref is not None:
            return Resolved("mod", "head", head_id, ref=mref)

    ref = base.get("head", head_id)
    if gs and not keeps_stock_row(HEAD_CAT_NO, ref):
        # The game would draw head 0 here (``<ChangeHeadAsync>`` hands
        # ``LoadCharaFbxDataAsync`` a ``defaultId`` of 0); a head is the face,
        # so this stays a reported miss rather than someone else's face
        # (``body_head.head_assets``, §7.490).
        return Resolved("missing", "head", head_id,
                        reason=f"headId {head_id}：未安裝 {gs[0]}#{gs[1]}")
    if ref is not None and ref.asset not in ("", "0", "-"):
        return Resolved("base", "head", head_id, ref=ref)
    if gs:
        return Resolved("missing", "head", head_id,
                        reason=f"headId {head_id}：未安裝 {gs[0]}#{gs[1]}")
    mref = mods.find_by_cat_slot(HEAD_CAT_NO, head_id)
    if mref is not None:
        return Resolved("mod", "head", head_id, ref=mref)
    return Resolved("missing", "head", head_id,
                    reason=f"headId {head_id} 在清單與 zipmod 都找不到")


def resolve_hair(slot_index: int, item_id: int, *,
                 res: dict | None = None) -> Resolved:
    if slot_index < 0 or slot_index >= len(HAIR_SLOTS):
        return Resolved("missing", "hair", item_id, reason="bad slot")
    cat = HAIR_SLOTS[slot_index]
    inst, base = _base_index()
    if base is None:
        return Resolved("missing", cat, item_id, reason="no assets_cache")
    # **id 0 is a list row like any other.**  Whether a slot is empty is the
    # row's business — "なし" ships as a real row pointing at ``p_dummy``, which
    # :func:`dress.is_empty_slot` reads — and back hair has no such row at all:
    # ``hair_back`` id 0 is ``ロングヘア`` (``p_cf_hair_b_01``).  Treating 0 as
    # "nothing" here took the back hair off every card that picked the first
    # style, 4% of them (8 of 200 sampled), leaving a bald skull under whatever
    # front hair they had — and reported **nothing**, because this returned an
    # "empty" the caller is supposed to skip silently.  ``ChangeHairAsync`` has
    # no such test: it hands the id straight to ``LoadCharaFbxDataAsync``.
    if int(item_id) < 0:
        return Resolved("missing", cat, item_id, reason="negative id")
    mods = SL.index()
    res = res or {}
    prop = HAIR_UAR[slot_index]
    # ``ChaListDefine.CategoryNo``: bo_hair_b 101 … bo_hair_o 104, the number
    # ``ChangeHairAsync`` hands ``GetCategoryInfo``.
    cat_no = HAIR_CAT_NO[slot_index]
    gs = uar_named(prop, item_id, res)
    # Pin the lookup to that number.  ``SideloaderIndex.find`` walks a guid's
    # categories in dictionary order when it is not given one, and a mod that
    # ships hair *and* clothes numbers them from the same run: on
    # ``vk.HairNClothes.part1`` slot 885 is a back hairstyle **and** a military
    # jacket, and the unpinned lookup put the jacket on a boy's head (§7.66 六).
    if gs:
        mref = mods.find(gs[0], gs[1], cat_no)
        if mref is not None:
            return Resolved("mod", cat, item_id, ref=mref)

    ref = base.get(cat, int(item_id))
    if gs and not keeps_stock_row(cat_no, ref):
        return _fbx_default(base, cat, item_id, HAIR_DEFAULT_ID[slot_index],
                            f"hair {cat}#{item_id}：未安裝 {gs[0]}#{gs[1]}")
    if ref is not None:
        return Resolved("base", cat, item_id, ref=ref)
    mref = mods.find_by_cat_slot(cat_no, int(item_id))
    if mref is not None:
        return Resolved("mod", cat, item_id, ref=mref)
    return _fbx_default(base, cat, item_id, HAIR_DEFAULT_ID[slot_index],
                        f"hair {cat}#{item_id} 在清單與 zipmod 都找不到")


def resolve_accessory(type_no: int, item_id: int, *,
                      res: dict | None = None, acc_index: int = 0,
                      coordinate: int = 0) -> Resolved:
    """One accessory slot → the asset it names.

    The card's ``type`` **is** ``ChaListDefine.CategoryNo``, and that is the
    number ``ChangeAccessoryAsync`` hands to ``GetCategoryInfo``, so the id is
    only ever looked up in that one category.  Searching the sibling accessory
    categories on a miss used to be the fallback here; the ids are per-category
    and a miss means the item is a mod, not that it moved next door.
    """
    cat = ACC_TYPE_TO_CAT.get(int(type_no), "acc_body")
    inst, base = _base_index()
    if base is None:
        return Resolved("missing", cat, item_id, reason="no assets_cache")

    mods = SL.index()
    res = res or {}
    prop = f"outfit{coordinate}.accessory{acc_index}.ChaFileAccessory.PartsInfo.id"
    gs = uar_named(prop, item_id, res)
    if gs:
        # The card named the mod.  Pin the lookup to the card's own type — one
        # mod can number a hat and a jacket alike, and an unpinned ``find``
        # answers with whichever category it walks into first (§7.66 五).  Over
        # 100 cards, 2,610 accessories resolve pinned and **none** needs the
        # unpinned retry, so there is nothing to trade off.
        #
        # If that (guid, slot) is not installed, ``ResolveStructure`` rewrites
        # the id to 99999999 unless the stock row of that number sits outside a
        # numbered game bundle (:func:`keeps_stock_row` — the same rule for
        # every structure, §7.488).  Falling through to "some other mod also
        # numbers an item 525" is how a nurse ended up under a pirate hat and a
        # detective hat (§7.66 三 2).
        mref = mods.find(gs[0], gs[1], int(type_no))
        if mref is not None:
            return Resolved("mod", cat, item_id, ref=mref)
        row = base.get(cat, int(item_id))
        if keeps_stock_row(int(type_no), row):
            return Resolved("base", cat, item_id, ref=row)
        return Resolved("missing", cat, item_id,
                        reason=f"acc type {type_no} id {item_id}:"
                               f" 未安裝 {gs[0]}#{gs[1]}")

    ref = base.get(cat, int(item_id))
    if ref is not None:
        return Resolved("base", cat, item_id, ref=ref)
    mref = mods.find_by_cat_slot(int(type_no), int(item_id))
    if mref is not None:
        return Resolved("mod", cat, item_id, ref=mref)
    return Resolved("missing", cat, item_id,
                    reason=f"acc type {type_no} id {item_id}")
