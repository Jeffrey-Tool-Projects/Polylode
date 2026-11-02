"""Koikatsu H action list → our format (import side only).

**A row of ``abdata/h/list/AnimationInfo_{mode:00}`` is one H action** — the
thing the player picks in Origin's own menu ("立ち愛撫", "密着手コキ").  It is *not*
one clip: the row names an ``AnimatorController`` inside an ``h/anim`` pack, and
that controller's **states are the clip stems** we already bake
(``L_WLoop1``, ``S_Insert``, ``M_Idle`` …).

So the menu row count is the number of list rows (**106**), not the number of
baked clips (6,057) nor pack×stem pairs (4,680).  Everything below is read from
the install; nothing Illusion-owned is redistributed — :func:`bake_actions`
writes our own JSON under ``assets_baked/h_anim/`` and the runtime reads only
that (``CLAUDE.md`` 四之一).

Column map is transcribed from ``HSceneProc.CreateAllAnimationList``.  That
method reads ``ExcelData.Get(row, col++)`` with a running counter, so the column
order *is* the read order — except ``id``, which is read with a literal ``1``
and then skipped by a bare increment (IL_00f8).  Do not renumber from the data.

The pack/controller columns were confirmed against the bundles themselves: for
all six (path, name) pairs every one of the 403 references resolves to an
``AnimatorController`` that exists in the named bundle (0 missing).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..core.paths import write_json_atomic
from .install import Install, configured
from ..core import swallowed

# Every bundle a list reader could not open during the current bake.  The bake
# functions below clear it first and **refuse to write** when it is not empty:
# a reader that skips a bundle it could not load returns a result that is
# merely shorter, and a shorter table overwrote a good one.  That happened —
# under memory pressure ``UnityPy.load`` fails on a few bundles (the error
# prints as "Error parsing file"), and one ``bake_hanim`` run rewrote
# ``actions.json`` with 171 roles whose states were empty and ``breath.json``
# with 60 of its 108 states (§7.357).  One unreadable bundle now costs the
# whole file, which keeps the previous one on disk and says why.
_READ_FAILURES: list[str] = []


def _read_failed(what: str, exc: BaseException) -> None:
    print(f"[hlist] {what}: {type(exc).__name__}: {exc}")
    _READ_FAILURES.append(f"{what}: {type(exc).__name__}")


def _refuse_partial(key: str) -> dict | None:
    """The bake's answer when a bundle failed to read, else ``None``."""
    if not _READ_FAILURES:
        return None
    return {"ok": False, "error": "bundle_read_failed", key: 0,
            "failed": list(_READ_FAILURES)}


# ---------------------------------------------------------------------------
# EMode — the authoritative family, from the ``EMode`` enum in Assembly-CSharp.
# The first byte of a pack filename correlates with it but is *not* it: the
# list row says which mode a row belongs to (file ``AnimationInfo_{mode:00}``).
# ---------------------------------------------------------------------------
EMODE = (
    "none", "aibu", "houshi", "sonyu", "masturbation",
    "peeping", "lesbian", "houshi3P", "sonyu3P", "houshi3PMMF", "sonyu3PMMF",
)
# ``AnimationInfo_{i}`` → ``EMode``: **the file index is one less than the enum
# value**, because ``EMode.none`` has no list file.  ``CreateAllAnimationList``
# stores the loop counter straight into ``AnimationListInfo.mode``, so that
# field is the *file* index, not the enum — ``LoadAnimationListInfo`` is where
# the two meet (its ``switch`` maps an ``EMode`` onto ``lstUseAnimInfo[N]``).
#
# Confirmed twice over on this install, because reading the field name as the
# enum silently labels every caress row "none":
#   file 0 → controllers ``kha_*``  and 立ち愛撫 / 仰向け愛撫      → aibu
#   file 1 → ``khh_*``  亀頭いじり / 片手コキ                      → houshi
#   file 2 → ``khs_*``  正常位 / 開脚正常位 / 後背位                → sonyu
#   file 3 → ``kht_*``  オナニー系                                 → masturbation
#   file 4 → ``khn_*``  トイレ覗き / シャワー覗き                   → peeping
#   file 5 → ``khr_*``  具合わせ / 椅子クンニ / 互い弄り             → lesbian
EMODE_FILE_OFFSET = 1


def emode_name(file_index: int) -> str:
    i = int(file_index) + EMODE_FILE_OFFSET
    return EMODE[i] if 0 <= i < len(EMODE) else f"mode{file_index}"

# Columns — transcribed one by one, not inferred from the data.
COL_NAME = 0          # nameAnimation — the *Japanese action name* ("立ち愛撫"),
                      # not a clip name.  Reading the field name as "the
                      # animation to play" is exactly the mistake §7.x warns of.
COL_ID = 1            # id (literal Get(row, 1) + TryParse)
COL_CATEGORY = 2      # lstCategory — '/'-separated ints
COL_POSTURE = 3       # posture
COL_KIND_HOUSHI = 4   # kindHoushi
COL_IS_EXPERIENCE = 5 # isExperience
COL_MOVES = 6         # each category's ``fileMove``, '/'-separated in category order (IL_01f6‥IL_026d:
                      # used only when it splits into as many as ``lstCategory``); pathMapObjectNull is the next
COL_USE_DESK = 9      # useDesk
COL_USE_CHAIR = 10    # useChair
COL_MALE_BASE = 11    # pathMaleBase.assetpath / .file  (base controller)
COL_MALE = 14         # pathMale.assetpath / .file      (action controller)
COL_FEMALE_BASE = 22  # pathFemaleBase
COL_FEMALE = 26       # pathFemale
COL_FEMALE1_BASE = 59 # pathFemaleBase1 (3P rows only — 80/82-column tables)
COL_FEMALE1 = 63      # pathFemale1
COL_MALE_SON = 52     # numMaleSon (``GetIntTryParse(…, 0)``; the column after numVoiceKindID, §7.1115)
COL_MALE1_SON = 78    # numFemaleUpperCloth1 (``GetIntTryParse(…, 0)``, after paramFemale1.isAnal 77)
COL_NUM_CTRL = 42     # numCtrl (``Int32.TryParse`` into the field, IL_0749–0755; after nameCamera 41)

_LIST_DIR = ("h", "list")


@dataclass
class HRole:
    """One character's animator for an action: base state machine + action."""
    pack: str = ""             # ``01_20_00`` — bundle stem, matches bake names
    controller: str = ""       # ``khh_f_52``
    base_pack: str = ""
    base_controller: str = ""  # ``khh_f_base`` — holds the state machine
    states: list[str] = field(default_factory=list)   # action clip stems
    base_states: list[str] = field(default_factory=list)
    # base-layer state → ``[speed parameter name or "", m_Speed]`` — the
    # animator's playback multiplier for that state (§7.392/§7.394)
    base_speeds: dict = field(default_factory=dict)
    # clip **name** → the base state that plays it (:func:`controller_clip_states`):
    # the action controller overrides the base's clips by name, so an action
    # stem is found here, not by spelling (§7.405)
    base_clip_states: dict = field(default_factory=dict)
    # base state → the synced layers ``AnimatorLayerCtrl`` turns on in it
    # (``al_<controller>``, :func:`parse_layer_lists`); the indices are into
    # the controller the animator runs, i.e. the base when there is one
    layers: dict = field(default_factory=dict)
    # ``param*.lstIdLayer[0]`` — the layer ``ChangeAnimator`` sets to weight 1
    # when the action loads (the column after ``path*Base``); -1 for none
    id_layer: int = -1


@dataclass
class HAction:
    """One row of ``AnimationInfo_*`` — one selectable H action."""
    mode: int          # list-file index (what Origin stores in ``mode``)
    mode_name: str     # EMode name — see ``EMODE_FILE_OFFSET``
    id: int
    name: str                  # Japanese action name from column 0
    categories: list[int] = field(default_factory=list)
    # ``lstCategory[i].fileMove``: the ``h/list`` move list a point of ``categories[i]`` shifts both
    # bodies by (``HSceneProc.SetLocalPosition``); "" where the row names none
    moves: list[str] = field(default_factory=list)
    posture: int = 0
    kind_houshi: int = 0
    use_desk: int = 0
    use_chair: int = 0
    female: HRole | None = None
    male: HRole | None = None
    female1: HRole | None = None
    # ``numMaleSon`` and ``numFemaleUpperCloth1`` as read (-1: the row ends before the column, the
    # ``AnimationListInfo`` constructor's value).  ``ChangeAnimator`` sets ``male.status.visibleSon``
    # to the first == 1 and ``male1``'s, the MMF scene's second man, to the second == 1.
    male_son: int = -1
    male1_son: int = -1
    # ``numCtrl``: which of ``flags.motion`` / ``motion1`` the 3P procs hand the women
    # (``H3PSonyu.Proc``, ``H3PHoushi.Proc``); a failed parse is 0, as ``TryParse`` leaves it
    num_ctrl: int = 0


def _row_int(cols: list[str], i: int) -> int:
    """``GetIntTryParse(Get(row, i), 0)`` on a row that reaches column *i*, else -1 (never read)."""
    return _int(cols, i, 0) if i < len(cols) else -1


def _install() -> Install | None:
    """Import-side source: the synced cache when it carries ``h/list``.

    ``h/list`` was not in the sync job list until this was written, so a cache
    synced earlier has ``h/anim`` but no list.  Fall back to the external
    install rather than reporting "no rows" — which reads like *the game has no
    list* instead of *we never copied it*.
    """
    from .install import source_install

    for inst in (configured(), source_install()):
        if inst is not None and inst.valid and \
                inst.abdata.joinpath(*_LIST_DIR).is_dir():
            return inst
    return None


def _cell(cols: list[str], i: int) -> str:
    return cols[i].strip() if 0 <= i < len(cols) else ""


def _int(cols: list[str], i: int, default: int = 0) -> int:
    v = _cell(cols, i)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Import side — reads the install (bundles).  Never called at runtime.
# ---------------------------------------------------------------------------
def _read_list_rows(inst: Install) -> dict[int, list[list[str]]]:
    """``{mode: [row_columns, ...]}`` from every ``AnimationInfo_*`` text asset.

    The assets are scattered across the 16 ``h/list/*.unity3d`` bundles, so the
    whole directory is scanned rather than guessing which bundle holds which.
    """
    import re

    out: dict[int, list[list[str]]] = {}
    for m, rows in _text_assets(inst, re.compile(r"AnimationInfo_(-?\d+)$"),
                                "list"):
        if rows:
            out.setdefault(int(m.group(1)), []).extend(rows)
    return out


def unwrap(x):
    """Peel UnityPy's ``{'data': …}`` wrappers off a typetree node."""
    while isinstance(x, dict) and len(x) == 1 and "data" in x:
        x = x["data"]
    return x


def controller_states(tree: dict) -> list[tuple[str, list[int]]]:
    """``[(state_name, [clip_index, …]), …]`` for one controller typetree.

    ``clip_index`` indexes the controller's **own** ``m_AnimationClips`` PPtr
    array — that array, not the clip's name, is what says which clip a state
    plays.  Verified on ``01_20_00`` (1 controller, 30 states, 30 clips): the
    mapping is a bijection and every clip's ``m_Name`` equals its state name;
    and across all 32 packs the controllers reach **16,648 of 16,648** clips,
    so nothing in a pack is unreachable this way (§7.112).

    A state usually yields one index.  Blend-tree states yield one per corner
    (``Te_kokan_D_kosuru`` → 9, §7.111) and are *not* a single clip; the caller
    decides what to do with those rather than silently taking the first.
    """
    tos = {int(k): v for k, v in (tree.get("m_TOS") or [])}
    n_clips = len(tree.get("m_AnimationClips") or [])
    ctrl = unwrap(tree.get("m_Controller") or {})
    out: list[tuple[str, list[int]]] = []
    for sm in unwrap(ctrl.get("m_StateMachineArray")) or []:
        sm = unwrap(sm)
        for st in unwrap(sm.get("m_StateConstantArray")) or []:
            st = unwrap(st)
            full = unwrap(st.get("m_FullPathID"))
            label = tos.get(int(full or 0), "")
            if not label:
                continue
            # ``Base Layer.L_WLoop1`` → ``L_WLoop1``; extra layers are
            # named ``09:Te_kokan_D_kosuru.Te_kokan_D_kosuru``.
            stem = label.split(".")[-1]
            ids: list[int] = []
            for bt in unwrap(st.get("m_BlendTreeConstantArray")) or []:
                bt = unwrap(bt)
                for node in unwrap(bt.get("m_NodeArray")) or []:
                    node = unwrap(node)
                    cid = node.get("m_ClipID")
                    if cid is None:
                        continue
                    cid = int(cid)
                    # Interior blend-tree nodes carry -1 (0xFFFFFFFF); only
                    # leaves index the clip array.
                    if 0 <= cid < n_clips:
                        ids.append(cid)
            out.append((stem, ids))
    return out


#: A body-size blend tree, as the ``kht``/``khr``/``khn`` controllers author
#: their phase states: 1D (``m_BlendType`` 0) over the animator float
#: ``height``, thresholds exactly 0 / 0.5 / 1, three **leaf** children.  The
#: action controllers of other packs spell the same thing as three plain
#: states ``S_X`` / ``M_X`` / ``L_X`` (§7.362), so this shape is the one a
#: bake can flatten without inventing anything (§7.371).
SIZE_TREE_PARAM = "height"
SIZE_TREE_THRESHOLDS = (0.0, 0.5, 1.0)
#: The three leaves' stage prefixes, in threshold order -- the names other
#: packs give the same clips as plain states.
SIZE_TREE_PREFIXES = ("S_", "M_", "L_")
_NO_CLIP = 0xFFFFFFFF


def _size_leaves(nodes: list, i: int, tos: dict, param: str,
                 n_clips: int) -> list[int] | None:
    """The three clip indices of node *i*, if it is a size node over *param*.

    A size node is 1D (``m_BlendType`` 0) over the animator float *param*,
    thresholds exactly :data:`SIZE_TREE_THRESHOLDS`, and three **leaf**
    children in threshold order.
    """
    if not 0 <= i < len(nodes):
        return None
    node = nodes[i]
    if int(node.get("m_BlendType", -1)) != 0:
        return None
    if tos.get(int(node.get("m_BlendEventID", _NO_CLIP)), "") != param:
        return None
    thr = unwrap(unwrap(node.get("m_Blend1dData") or {}).get("m_ChildThresholdArray")) or []
    if [round(float(x), 6) for x in thr] != list(SIZE_TREE_THRESHOLDS):
        return None
    kids = [int(k) for k in (unwrap(node.get("m_ChildIndices")) or [])]
    if len(kids) != 3:
        return None
    clips = [int(nodes[k].get("m_ClipID", _NO_CLIP)) if 0 <= k < len(nodes) else -1
             for k in kids]
    return clips if all(0 <= c < n_clips for c in clips) else None


def _state_trees(tree: dict):
    """``(state name, nodes, tos)`` — each state's blend tree **as its own layer plays it**.

    A state machine that synced layers share keeps one motion set per layer,
    so its states carry one tree each (``m_BlendTreeConstantArray``, indexed
    by the layer's ``m_StateMachineMotionSetIndex``).  ``khr_f_02_*`` is that:
    its Base Layer's phases are the size grid, and two synced hand layers
    (``khr_p_02_00_te_R`` / ``_te_LR``, motion sets 1 and 2) play ``Breast1``
    trees on the same states.  The tree taken is the motion set of the
    **first** layer on the state machine — the one that owns it; a state with
    no tree for that set is skipped.
    """
    tos = {int(k): v for k, v in (tree.get("m_TOS") or [])}
    ctrl = unwrap(tree.get("m_Controller") or {})
    owner_set: dict[int, int] = {}
    for layer in unwrap(ctrl.get("m_LayerArray")) or []:
        layer = unwrap(layer)
        smi = int(layer.get("m_StateMachineIndex", -1))
        owner_set.setdefault(smi, int(layer.get("m_StateMachineMotionSetIndex", 0) or 0))
    for smi, sm in enumerate(unwrap(ctrl.get("m_StateMachineArray")) or []):
        motion_set = owner_set.get(smi, 0)
        for st in unwrap(unwrap(sm).get("m_StateConstantArray")) or []:
            st = unwrap(st)
            label = tos.get(int(unwrap(st.get("m_FullPathID")) or 0), "")
            if not label:
                continue
            trees = unwrap(st.get("m_BlendTreeConstantArray")) or []
            index = unwrap(st.get("m_BlendTreeConstantIndexArray")) or list(range(len(trees)))
            if not 0 <= motion_set < len(index):
                continue
            k = int(index[motion_set])
            if not 0 <= k < len(trees):
                continue
            nodes = [unwrap(n) for n in unwrap(unwrap(trees[k]).get("m_NodeArray")) or []]
            yield label.split(".")[-1], nodes, tos


def controller_size_trees(tree: dict) -> dict[str, list[int]]:
    """``{state: [S clip, M clip, L clip]}`` for the states that are size trees.

    Only the exact shape :data:`SIZE_TREE_PARAM` / :data:`SIZE_TREE_THRESHOLDS`
    with three leaves qualifies; anything else — a 2D tree, another parameter
    (``motion``), a nested node — is left to the caller's "blend tree, not
    converted" count rather than approximated (``CLAUDE.md`` 一之一).  The
    indices are into the controller's own ``m_AnimationClips``, like
    :func:`controller_states`.  The two-person grid is
    :func:`controller_size_grids`.
    """
    n_clips = len(tree.get("m_AnimationClips") or [])
    out: dict[str, list[int]] = {}
    for label, nodes, tos in _state_trees(tree):
        if len(nodes) != 4:
            continue
        clips = _size_leaves(nodes, 0, tos, SIZE_TREE_PARAM, n_clips)
        if clips is not None:
            out[label] = clips
    return out


#: The outer axis of a two-person size grid: ``height1``, which
#: ``HLesbian.Proc`` sets on each woman's animator to **the other woman's**
#: height slider (0 when there is no second woman), while ``height`` stays her
#: own (``SetDefaultAnimatorFloat``).  ``互い弄り``'s ``khr_f_02_*`` phases are
#: a size node over one of the two whose three children are size nodes over
#: the other — nine leaves, one per pair of sizes (§7.374).  The nesting
#: differs by role: ``khr_f_02_00`` (the first woman) splits on ``height1``
#: first, ``khr_f_02_01`` (the second) on ``height``.
SIZE_GRID_PARAM = "height1"


def controller_size_grids(tree: dict) -> dict[str, list[list[int]]]:
    """``{state: [[own S, M, L] for the other woman's S, M, L]}`` — the 3×3 size grid.

    The exact shape only: 13 nodes, a size node over one of
    :data:`SIZE_TREE_PARAM` / :data:`SIZE_GRID_PARAM` at the root whose three
    children are size nodes over the other.  Either nesting is returned
    indexed the same way (other woman's size, then her own); the blend is a
    product of the two 1D weights, so the order does not change the pose.
    ``MLoop``, which nests a ``motion`` node under each of those, is 31 nodes
    and is not this.
    """
    n_clips = len(tree.get("m_AnimationClips") or [])
    out: dict[str, list[list[int]]] = {}
    for label, nodes, tos in _state_trees(tree):
        if len(nodes) != 13:
            continue
        root = nodes[0]
        if int(root.get("m_BlendType", -1)) != 0:
            continue
        outer = tos.get(int(root.get("m_BlendEventID", _NO_CLIP)), "")
        if outer not in (SIZE_GRID_PARAM, SIZE_TREE_PARAM):
            continue
        inner = SIZE_TREE_PARAM if outer == SIZE_GRID_PARAM else SIZE_GRID_PARAM
        thr = unwrap(unwrap(root.get("m_Blend1dData") or {}).get("m_ChildThresholdArray")) or []
        if [round(float(x), 6) for x in thr] != list(SIZE_TREE_THRESHOLDS):
            continue
        kids = [int(k) for k in (unwrap(root.get("m_ChildIndices")) or [])]
        if len(kids) != 3:
            continue
        rows = [_size_leaves(nodes, k, tos, inner, n_clips) for k in kids]
        if any(r is None for r in rows):
            continue
        if outer == SIZE_TREE_PARAM:          # rows[own][other] -> [other][own]
            rows = [list(col) for col in zip(*rows)]
        out[label] = rows
    return out


#: The two animator floats a ``MLoop`` tree blends its pair of motions by —
#: ``flags.motion`` / ``flags.motion1``, each driven by ``CalcFluctuation``
#: (§7.391).  Which one a controller reads is per controller: ``khr_f_01_00``
#: (the first woman) splits on ``motion1`` and ``khr_f_01_01`` on ``motion``.
MOTION_PARAMS = ("motion", "motion1")
MOTION_THRESHOLDS = (0.0, 1.0)


def _tree_leaves(nodes: list, i: int, tos: dict, n_clips: int,
                 coords: dict, out: list) -> bool:
    """Walk a blend tree whose every node is a 1D size, motion or bust split.

    Appends ``(clip index, {param: child index})`` per leaf.  ``False`` for any
    other shape (2D, an unknown parameter, other thresholds, a parameter split
    twice on one path) — the caller then leaves the state uncounted rather
    than approximating it (``CLAUDE.md`` 一之一).
    """
    if not 0 <= i < len(nodes):
        return False
    node = nodes[i]
    kids = [int(k) for k in (unwrap(node.get("m_ChildIndices")) or [])]
    if not kids:
        cid = int(node.get("m_ClipID", _NO_CLIP))
        if not 0 <= cid < n_clips:
            return False
        out.append((cid, dict(coords)))
        return True
    if int(node.get("m_BlendType", -1)) != 0:
        return False
    par = tos.get(int(node.get("m_BlendEventID", _NO_CLIP)), "")
    want = (SIZE_TREE_THRESHOLDS if par in (SIZE_TREE_PARAM, SIZE_GRID_PARAM)
            else MOTION_THRESHOLDS if par in MOTION_PARAMS
            else BREAST_THRESHOLDS if par in BREAST_PARAMS else None)
    thr = unwrap(unwrap(node.get("m_Blend1dData") or {}).get("m_ChildThresholdArray")) or []
    if want is None or par in coords or len(kids) != len(want) \
            or [round(float(x), 6) for x in thr] != list(want):
        return False
    for j, k in enumerate(kids):
        coords[par] = j
        if not _tree_leaves(nodes, k, tos, n_clips, coords, out):
            return False
    del coords[par]
    return True


def controller_motion_trees(tree: dict) -> dict[str, list[tuple]]:
    """``{state: [(motion param, motion index, other size prefix, [S, M, L clip])]}``.

    The ``MLoop`` states of the lesbian and masturbation bases (D1d): a
    ``motion`` (or ``motion1``) node with thresholds 0 / 1 over two body-size
    trees, or — ``khr_f_02_*`` — the 3×3 two-person size grid with such a node
    under each cell.  Motion 0 is the ``*_MLoop1`` clip and 1 the ``*_MLoop2``
    one in every tree on this install, which is also what the masturbation
    action controllers name the states they override the base with.  Only a
    state whose every leaf has a size and exactly one motion coordinate, and
    whose rows are complete, is returned.
    """
    n_clips = len(tree.get("m_AnimationClips") or [])
    out: dict[str, list[tuple]] = {}
    for label, nodes, tos in _state_trees(tree):
        leaves: list = []
        if len(nodes) < 2 or not _tree_leaves(nodes, 0, tos, n_clips, {}, leaves):
            continue
        params = {p for _, c in leaves for p in c if p in MOTION_PARAMS}
        if len(params) != 1:
            continue
        param = params.pop()
        cells: dict = {}
        for cid, c in leaves:
            if SIZE_TREE_PARAM not in c:
                break
            other = SIZE_TREE_PREFIXES[c[SIZE_GRID_PARAM]] if SIZE_GRID_PARAM in c else ""
            cells.setdefault((c[param], other), [None] * 3)[c[SIZE_TREE_PARAM]] = cid
        else:
            if all(None not in row for row in cells.values()):
                out[label] = [(param, m, other, row)
                              for (m, other), row in sorted(cells.items())]
    return out


#: ``AnimatorLayerCtrl.ResetAllLayer`` zeroes every layer from this index up
#: before ``Proc`` turns the listed ones on; the ones below it (the Base Layer
#: and the ``HandCtrl`` caress item layers, §7.111) are not its business.
LAYER_CTRL_FIRST = 73
#: The bust floats a synced layer's tree splits on: ``Breast`` is the
#: character's own ``GetShapeBodyValue(4)`` (``BustSize``) and ``Breast1`` the
#: other woman's (``HLesbian.Proc``); five children at quarter steps.
BREAST_PARAMS = ("Breast", "Breast1")
BREAST_THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_layer_lists(inst: Install | None = None) -> dict[str, dict[str, list[int]]]:
    """``{file: {base state: [layer index, …]}}`` from the ``al_<file>`` lists.

    ``AnimatorLayerCtrl.Load`` reads ``LoadAllListText(path, "al_" + file)``
    through ``GetListString`` (rows on newlines, cells on tabs); column 0 is
    the state, and **every later non-empty cell** is one ``Int32.Parse``d layer
    index.  A state seen again is ``Clear``ed and refilled, so the last row
    wins.  ``Proc(state)`` then zeroes layers 73‥ (:data:`LAYER_CTRL_FIRST`)
    and sets each listed index below ``layerCount`` to weight 1 — called from
    ``HActionBase.SetPlay`` with the base state being played.  A state listed
    with no index (``Orgasm``) turns every such layer off.
    """
    import re

    inst = inst or _install()
    out: dict[str, dict[str, list[int]]] = {}
    if inst is None:
        return out
    for m, rows in _text_assets(inst, re.compile(r"al_(.+)$"), "al"):
        table = out.setdefault(m.group(1), {})
        for cols in rows:
            state = cols[0].strip() if cols else ""
            if not state:
                continue
            try:
                table[state] = [int(c) for c in cols[1:] if c.strip()]
            except ValueError as exc:
                _read_failed(f"al_{m.group(1)} row {state}", exc)
    return out


def controller_layer_trees(tree: dict, wanted: dict[int, set]) -> dict[int, dict]:
    """What each synced layer in *wanted* plays: ``{layer: {…}}``.

    *wanted* is ``{layer index: {base state, …}}``; ``"*"`` among the states
    means every state with a tree in that layer's motion set (a layer that
    stays on, :data:`LAYER_CTRL_FIRST`), and a state reached only through it
    is not reported as skipped.  A synced layer shares the
    Base Layer's state machine and keeps its own motion set
    (``m_StateMachineMotionSetIndex``), so a state's tree for the layer is
    ``m_BlendTreeConstantArray[m_BlendTreeConstantIndexArray[set]]``.  Each
    entry carries the layer's name, blending mode (0 override, 1 additive),
    the path hashes its skeleton mask weights above 0, and per state the
    leaves as ``(clip index, {param: child index})`` over one of
    :data:`BREAST_PARAMS` (under a ``motion`` split where the base has one).
    A state with no tree in that motion set, or any other shape, is listed
    under ``"skipped"`` rather than approximated (``CLAUDE.md`` 一之一).
    """
    tos = {int(k): v for k, v in (tree.get("m_TOS") or [])}
    n_clips = len(tree.get("m_AnimationClips") or [])
    ctrl = unwrap(tree.get("m_Controller") or {})
    layers = [unwrap(x) for x in unwrap(ctrl.get("m_LayerArray")) or []]
    sms = [unwrap(x) for x in unwrap(ctrl.get("m_StateMachineArray")) or []]
    out: dict[int, dict] = {}
    for li, states in sorted(wanted.items()):
        if not 0 <= li < len(layers):
            continue
        layer = layers[li]
        smi = int(layer.get("m_StateMachineIndex", -1))
        if not 0 <= smi < len(sms):
            continue
        mset = int(layer.get("m_StateMachineMotionSetIndex", 0) or 0)
        mask = unwrap(unwrap(layer.get("m_SkeletonMask") or {}).get("m_Data")) or []
        entry = {
            "name": str(tos.get(int(layer.get("m_Binding", 0) or 0), "")),
            "mode": int(layer.get("(int&)m_LayerBlendingMode",
                                  layer.get("m_LayerBlendingMode", 0)) or 0),
            "mask": [int(unwrap(e).get("m_PathHash", 0)) for e in mask
                     if float(unwrap(e).get("m_Weight", 0) or 0) > 0],
            "states": {}, "skipped": [], "defaults": [],
        }
        for st in unwrap(sms[smi].get("m_StateConstantArray")) or []:
            st = unwrap(st)
            label = tos.get(int(unwrap(st.get("m_FullPathID")) or 0), "").split(".")[-1]
            if label not in states and "*" not in states:
                continue
            trees = unwrap(st.get("m_BlendTreeConstantArray")) or []
            index = unwrap(st.get("m_BlendTreeConstantIndexArray")) or []
            k = int(index[mset]) if 0 <= mset < len(index) else -1
            leaves: list = []
            if k < 0 and not unwrap(st.get("m_LeafInfoArray")) and bool(st.get("m_WriteDefaultValues")):
                # no motion in this layer's set at all, and the state writes defaults: with the layer
                # on, Unity puts every bone the layer's mask covers back to its default value (D11)
                if label in states:
                    entry["defaults"].append(label)
                continue
            if 0 <= k < len(trees):
                nodes = [unwrap(n) for n in unwrap(unwrap(trees[k]).get("m_NodeArray")) or []]
                if not _tree_leaves(nodes, 0, tos, n_clips, {}, leaves):
                    leaves = []
            params = {p for _, c in leaves for p in c if p in BREAST_PARAMS}
            extra = any(p not in BREAST_PARAMS and p not in MOTION_PARAMS
                        for _, c in leaves for p in c)
            if len(params) != 1 or extra:
                if label in states:
                    entry["skipped"].append(label)
                continue
            entry["states"][label] = leaves
        out[li] = entry
    return out


def _bundle_controllers(inst: Install, rel_path: str) -> dict[str, list[str]]:
    """``{controller name: [state, …]}`` for every ``AnimatorController`` in a bundle.

    **One load per bundle.**  The table was built one (bundle, controller) pair
    at a time, and 403 references name about thirty bundles — ``02_00_00`` was
    opened a dozen times, once per controller in it.  UnityPy environments hold
    reference cycles, so each copy stayed resident until a collection ran, and
    reading the action list alone climbed past **9 GB** (§7.357).
    """
    import gc
    import UnityPy

    p = inst.abdata / rel_path
    if not p.is_file():
        return {}
    try:
        env = UnityPy.load(str(p))
    except Exception as exc:  # noqa: BLE001
        _read_failed(f"controller bundle {rel_path}", exc)
        return {}
    out: dict[str, list[str]] = {}
    clip_names = None
    try:
        for obj in env.objects:
            if obj.type.name != "AnimatorController":
                continue
            try:
                tree = obj.read_typetree()
            except Exception as exc: # noqa: BLE001
                swallowed.note("origin.hlist._bundle_controllers: obj.read_typetree", exc)
                continue
            name = str(tree.get("m_Name") or "")
            if name not in out:
                out[name] = controller_stems(tree)
                _SPEEDS[(rel_path, name)] = controller_speeds(tree)
                if clip_names is None:
                    clip_names = {o.path_id: o.peek_name() for o in env.objects
                                  if o.type.name == "AnimationClip"}
                _CLIP_STATES[(rel_path, name)] = controller_clip_states(
                    tree, [clip_names.get(int(c.get("m_PathID") or 0), "")
                           for c in tree.get("m_AnimationClips") or []])
    finally:
        del env
        gc.collect()
    return out


#: ``(bundle, controller)`` → :func:`controller_speeds`, filled by the same
#: single load :func:`_bundle_controllers` makes of each bundle.
_SPEEDS: dict = {}
#: ``(bundle, controller)`` → :func:`controller_clip_states`, the same load.
_CLIP_STATES: dict = {}


def controller_clip_states(tree: dict, clip_names: list[str]) -> dict:
    """Clip name → the one state that plays it (every clip of every tree leaf).

    ``SetupAnimatorOverrideController`` swaps the base's clips for the action's
    **by name**, so this is what an action stem means: base state
    ``M_OUT_Loop`` is a size tree whose leaves are the clips ``S_OUT_Loop`` /
    ``M_OUT_Loop`` / ``L_OUT_Loop``, and the action's ``L_OUT_Loop`` plays
    there — no spelling rule reaches that, since the state's own ``M_`` is not
    a size letter.  A name two states share is left out (the caller falls back
    to the spelling rules and says so).  *clip_names* is ``m_AnimationClips``
    resolved to names, index for index.
    """
    seen: dict[str, set] = {}
    for state, idxs in controller_states(tree):
        for i in idxs:
            if 0 <= i < len(clip_names) and clip_names[i]:
                seen.setdefault(clip_names[i], set()).add(state)
    return {c: next(iter(st)) for c, st in seen.items() if len(st) == 1}


def controller_speeds(tree: dict) -> dict:
    """Base-layer state → ``[speed parameter name or "", m_Speed]``.

    ``StateConstant.m_SpeedParamID`` is the parameter Unity multiplies the
    state's own ``m_Speed`` by; ``0`` means none.  Only the first state machine
    (the Base Layer) is read — it is the one ``SetPlay`` drives.  The
    per-action controllers carry no parameters at all; what plays is the base's
    state machine with the action's clips laid over it
    (``SetupAnimatorOverrideController``, §7.393), so this is read off bases.
    """
    tos = {int(k): v for k, v in (tree.get("m_TOS") or [])}
    ctrl = tree.get("m_Controller") or {}
    sms = ctrl.get("m_StateMachineArray") or []
    if not sms:
        return {}
    out = {}
    for st in (sms[0].get("data") or {}).get("m_StateConstantArray") or []:
        s = st.get("data") or {}
        name = tos.get(int(s.get("m_NameID", 0) or 0))
        if not name:
            continue
        pid = int(s.get("m_SpeedParamID", 0) or 0)
        out[str(name)] = [str(tos.get(pid, "")) if pid else "",
                          float(s.get("m_Speed", 1.0) or 0.0)]
    return out


def controller_params(tree: dict) -> list[str]:
    """The names of an ``AnimatorController``'s parameters (``m_Values``, in order).

    What ``OCIChar.CheckAnimeParam`` asks when Studio decides whether its
    option sliders drive ``height1`` / ``Breast1`` or ``height`` / ``Breast`` (E1a).
    """
    tos = {int(k): v for k, v in (tree.get("m_TOS") or [])}
    ctrl = unwrap(tree.get("m_Controller") or {})
    values = unwrap(unwrap(ctrl.get("m_Values") or {}).get("m_ValueArray")) or []
    return [str(tos.get(int(unwrap(v).get("m_ID", 0) or 0), "")) for v in values]


def controller_stems(tree: dict) -> list[str]:
    """The clip stems this controller's states bake to, in state order.

    A size tree (:func:`controller_size_trees`) bakes as three clips named
    ``S_X`` / ``M_X`` / ``L_X`` (§7.371), so it lists as those three -- the
    spelling every other pack's controller already uses.  A two-person grid
    (:func:`controller_size_grids`) lists the same three: the other woman's
    size is a second key on the baked clip, not a different state.  Listing the bare
    state name instead left the six actions whose phases are size trees with no
    baked clip to pick, i.e. no menu row, after the clips themselves were baked.
    """
    trees = controller_size_trees(tree)
    trees.update(controller_size_grids(tree))
    out: list[str] = []
    for stem, _ in controller_states(tree):
        if stem in trees:
            out.extend(pref + stem for pref in SIZE_TREE_PREFIXES)
        else:
            out.append(stem)
    return out


def _controller_states(inst: Install, rel_path: str, name: str,
                       cache: dict | None = None) -> list[str]:
    """State names of one ``AnimatorController`` — these are our clip stems.

    Returns ``[]`` when the controller is not in that bundle; the caller reports
    it rather than substituting a plausible-looking set (``CLAUDE.md`` 一之一).
    *cache* holds :func:`_bundle_controllers` per bundle for one parse.  The
    first controller of a name wins, as the scan that stopped at the first
    match did.
    """
    if not rel_path or not name:
        return []
    cache = {} if cache is None else cache
    if rel_path not in cache:
        cache[rel_path] = _bundle_controllers(inst, rel_path)
    return list(cache[rel_path].get(name, []))


def _role(inst: Install, cols: list[str], col_base: int, col_act: int,
          cache: dict, layer_lists: dict | None = None) -> HRole | None:
    path = _cell(cols, col_act)
    name = _cell(cols, col_act + 1)
    base_path = _cell(cols, col_base)
    base_name = _cell(cols, col_base + 1)
    if not path and base_path and base_name:
        # ``HSceneProc.ChangeAnimator`` loads the action controller only when
        # ``param*.path.assetpath`` is set; otherwise the character runs the
        # base one (``path*Base``) alone — it is loaded when the mode changes,
        # always for ``EMode`` 5‥8 (peeping, lesbian, the two 3P modes), and
        # otherwise exactly when the action path is empty.  The same three-way test guards the female, the second female
        # and the male.  So a row with a base and no action path is not a row
        # with no role: its base controller *is* the role, state machine and
        # motion both.  ハードルオナニー and the three レズ rows are written
        # that way, which is how 96 actions came out as 90 (``Animation_Progress``
        # D1e; the other two are D1d's blend trees).
        path, name = base_path, base_name
    if not path or not name:
        return None

    def states(rel: str, nm: str) -> list[str]:
        return _controller_states(inst, rel, nm, cache)

    return HRole(
        pack=Path(path).stem,
        controller=name,
        base_pack=Path(base_path).stem if base_path else "",
        base_controller=base_name,
        states=states(path, name),
        base_states=states(base_path, base_name) if base_path and base_name else [],
        base_speeds=(dict(_SPEEDS.get((base_path, base_name), {}))
                     if base_path and base_name else {}),
        base_clip_states=(dict(_CLIP_STATES.get((base_path, base_name), {}))
                          if base_path and base_name else {}),
        # ``HSceneProc.ChangeAnimator`` loads ``al_`` + the action's own file
        # name — the controller named in the action column, or the base when
        # the row names only a base (``khr_f_02_*``).
        layers=dict((layer_lists or {}).get(name, {})),
        id_layer=_id_layer(_cell(cols, col_base + 2)),
    )


def _id_layer(cell: str) -> int:
    """``lstIdLayer[0]`` from its cell: '/'-separated, each ``GetIntTryParse(x, -1)``.

    ``CreateAllAnimationList`` reads the column right after ``path*Base``'s two;
    ``ChangeAnimator`` only ever uses element 0, and skips it when it is -1.
    """
    for part in (cell or "").split("/"):
        if part.strip():
            try:
                return int(part)
            except ValueError:
                return -1
    return -1


def category_moves(cell: str, count: int) -> list[str]:
    """Each of a row's *count* categories' ``fileMove`` from the move column (:data:`COL_MOVES`).

    ``CreateAllAnimationList`` splits the cell on '/' and assigns the parts only when there are as
    many as categories (``bne.un`` at IL_022d); otherwise every ``fileMove`` stays empty.
    """
    parts = [p.strip() for p in cell.split("/")] if cell else []
    return parts if len(parts) == count else [""] * count


def parse_actions(inst: Install | None = None) -> list[HAction]:
    """Every H action the install defines, from the list assets.

    **A row updates the action with that ``id``; it does not add a second one.**
    ``CreateAllAnimationList`` does ``lstAnimInfo[mode].Find(x => x.id == id)``
    and only ``Add``s when that misses (IL_008f‥IL_00e0), then writes every
    column onto whichever entry it now holds — ``lstCategory.Clear()`` right
    before refilling it is the tell that the row is *replaced*, not merged.

    That matters because ``AnimationInfo_{mode}`` is not one asset:
    ``LoadAllListText`` takes ``GetAssetBundleNameListFromPath``, **sorts it**,
    and concatenates the same-named TextAsset out of every bundle.  On this
    install ``AnimationInfo_01`` lives in four of them (30 + 11 + 2 + 1 rows)
    and ``_02`` in five; the DLC bundles re-state ten ``houshi`` rows to widen
    ``lstCategory`` (``0/1/6/8`` → ``0/1/6/7/8``) and add a desk spot to
    ``pathMapObjectNull``.  Appending gave ten duplicate actions, so the Free H
    menu carried **two identical rows for each** — picking the second one
    loaded the same clip pair and nothing on screen changed.  Sorted bundle
    order is what makes "later wins" well-defined, and ``_read_list_rows``
    already walks ``sorted(glob)``.
    """
    inst = inst or _install()
    if inst is None:
        return []
    cache: dict = {}
    layer_lists = parse_layer_lists(inst)
    by_id: dict[tuple[int, int], HAction] = {}
    for mode, rows in sorted(_read_list_rows(inst).items()):
        mode_name = emode_name(mode)
        for cols in rows:
            cats = [int(x) for x in _cell(cols, COL_CATEGORY).split("/")
                    if x.strip().lstrip("-").isdigit()]
            act = HAction(
                mode=mode,
                mode_name=mode_name,
                id=_int(cols, COL_ID, -1),
                name=_cell(cols, COL_NAME),
                categories=cats,
                moves=category_moves(_cell(cols, COL_MOVES), len(cats)),
                posture=_int(cols, COL_POSTURE),
                kind_houshi=_int(cols, COL_KIND_HOUSHI),
                use_desk=_int(cols, COL_USE_DESK),
                use_chair=_int(cols, COL_USE_CHAIR),
                female=_role(inst, cols, COL_FEMALE_BASE, COL_FEMALE, cache, layer_lists),
                male=_role(inst, cols, COL_MALE_BASE, COL_MALE, cache, layer_lists),
                female1=_role(inst, cols, COL_FEMALE1_BASE, COL_FEMALE1, cache, layer_lists),
                male_son=_row_int(cols, COL_MALE_SON),
                male1_son=_row_int(cols, COL_MALE1_SON),
                num_ctrl=_int(cols, COL_NUM_CTRL, 0),
            )
            by_id[(mode, act.id)] = act      # insertion order kept on first sight
    return list(by_id.values())


# ---------------------------------------------------------------------------
# Our format
# ---------------------------------------------------------------------------
ACTIONS_VERSION = 2          # 2: each category's move list and the ``moves`` table


def bake_actions(dest: Path, *, inst: Install | None = None) -> dict:
    """Write ``actions.json`` (our format) next to the baked clips."""
    inst = inst or _install()
    if inst is None:
        return {"ok": False, "error": "no_origin_install", "actions": 0}
    _READ_FAILURES.clear()
    actions = parse_actions(inst)
    refused = _refuse_partial("actions")
    if refused:
        return refused
    if not actions:
        return {"ok": False, "error": "no_list_rows", "actions": 0}
    missing = [f"{a.mode_name}/{a.id}"
               for a in actions
               if any(r is not None and not r.states
                      for r in (a.female, a.male, a.female1))]
    named = sorted({m for a in actions for m in a.moves if m})
    moves = parse_moves(inst, named)
    refused = _refuse_partial("actions")
    if refused:
        return refused
    payload = {
        "version": ACTIONS_VERSION,
        # move list → ``[pos.x, pos.y, pos.z, rot.x, rot.y, rot.z]`` as the source game holds it (Unity's
        # axes, degrees); a name a row gives that no bundle holds is left out and listed in the result
        "moves": moves,
        "actions": [
            {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v)
             for k, v in asdict(a).items() if v is not None}
            for a in actions
        ],
    }
    write_json_atomic(dest, payload, indent=1)
    return {
        "ok": True,
        "actions": len(actions),
        "by_mode": {m: sum(1 for a in actions if a.mode_name == m)
                    for m in sorted({a.mode_name for a in actions})},
        "missing_states": missing,
        "moves": len(moves),
        "missing_moves": [n for n in named if n not in moves],
        "file": str(dest),
    }


# ---------------------------------------------------------------------------
# HFaceList — the H expression table (import side)
# ---------------------------------------------------------------------------
# ``FaceListCtrl.LoadInfo`` reads these columns with a running counter, so this
# is the read order, not a guess.  ``highlight``/``eyesblink``/``yure`` are
# ``Get(...) == "1"``; ``openMaxEye``/``openMinMouth``/``cheek`` are floats.
HFACE_COLUMNS = ("id", "name", "eyebrow", "eye", "openMaxEye", "mouth",
                 "openMinMouth", "tears", "cheek", "highlight", "eyesblink",
                 "yure")

# ``FaceListCtrl.LoadText(personality)`` fills ``facelib`` with exactly this:
#     for i in 0..9:  LoadInfo(pers, 0, i, facelib[0][i])
#     for i in 0..10: LoadInfo(pers, 1, i, facelib[1][i])
#                     LoadInfo(pers, 2, 0, facelib[2][0])
# so a file is ``HFaceList_c{personality}_{kind:00}_{index:00}``.  **Kind 0's
# index is the EMode**, which the data confirms rather than the name: kind 0 /
# index 1 (houshi) is the only group with ``mouth`` 21/22 (49 + 49 rows), and
# index 2 (sonyu) has none — 舐め and 咥え belong to the oral action family.
HFACE_KINDS = {0: 10, 1: 11, 2: 1}


def parse_hface(inst: Install | None = None) -> dict:
    """``{(personality, kind, index): {row_id: [cols]}}`` from ``HFaceList_*``.

    Rows are keyed by their own ``id`` because ``LoadInfo`` does
    ``dict[id] = info`` — same "later wins" as :func:`parse_actions`, and for
    the same reason: ``LoadAllListText`` concatenates the same-named asset out
    of every (sorted) bundle.  ``HFaceList_c00_00_01`` is 238 raw rows across
    two bundles and **136 distinct ids** once the overwrite is applied.
    """
    import re

    inst = inst or _install()
    if inst is None:
        return {}
    pat = re.compile(r"HFaceList_c(-?\d+)_(\d+)_(\d+)$")
    out: dict = {}
    for m, lines in _text_assets(inst, pat, "hface"):
        key = (m.group(1), int(m.group(2)), int(m.group(3)))
        rows = out.setdefault(key, {})
        for cols in lines:
            if len(cols) < len(HFACE_COLUMNS):
                continue
            try:
                rid = int(cols[0])
            except ValueError:
                continue
            rows[rid] = cols[:len(HFACE_COLUMNS)]
    return out


HFACE_VERSION = 1


def bake_hface(dest: Path, *, inst: Install | None = None) -> dict:
    """Write ``hface.json`` — our format for the H expression table."""
    inst = inst or _install()
    if inst is None:
        return {"ok": False, "error": "no_origin_install", "rows": 0}
    _READ_FAILURES.clear()
    table = parse_hface(inst)
    refused = _refuse_partial("rows")
    if refused:
        return refused
    if not table:
        return {"ok": False, "error": "no_hface_rows", "rows": 0}
    faces: dict = {}
    n_rows = 0
    for (pers, kind, index), rows in sorted(table.items()):
        slot = faces.setdefault(pers, {}).setdefault(str(kind), {})
        # rows stay column arrays: 80k dicts would be ~4x the bytes for the
        # same data, and the column order is pinned by ``HFACE_COLUMNS``.
        slot[str(index)] = [rows[k] for k in sorted(rows)]
        n_rows += len(rows)
    payload = {"version": HFACE_VERSION,
               "columns": list(HFACE_COLUMNS),
               "faces": faces}
    write_json_atomic(dest, payload, separators=(",", ":"))
    tongue = sum(1 for rows in table.values() for c in rows.values()
                 if c[5] in ("21", "22"))
    missing = [f"c{p}_{k}_{i}"
               for p in faces
               for k, n in HFACE_KINDS.items()
               for i in range(n)
               if str(i) not in faces[p].get(str(k), {})]
    return {
        "ok": True,
        "personalities": len(faces),
        "files": len(table),
        "rows": n_rows,
        "tongue_rows": tongue,
        "missing_files": missing,
        "file": str(dest),
    }


# ---------------------------------------------------------------------------
# breath — which HFaceList row plays when (import side)
# ---------------------------------------------------------------------------
# ``HFaceList`` says what a face *is*; these two say **when**.  The chain Origin
# walks every frame (``HVoiceCtrl.BreathProc``):
#
#   current base-layer state name
#     → dicBreathPtn[mode][anim]                 (breath_{mode}_00)
#       → each VoicePtnInfo passing IsPlayBreathVoicePtn
#         → lstVoice[n]                          (a *voice* id)
#           → dicBreathVoiceIntos[level][voice]  (personality_breath_c*_{mode}_00)
#             → faceWeak / lstFaceNormal         ← an HFaceList row id
#
# ``level`` is ``GetStateSelect(info.state, …)``, which maps the pattern's
# ``state`` onto the heroine's H experience (0‥3) — the same 0/100/200 blocks
# the HFaceList row ids fall into.
#
# ``LoadBreathPtnInfo``'s seven columns, in read order (``Get(row, col++)``):
BREATH_INFO_COLUMNS = ("lstAnimID", "state", "lstConditions", "lstVoice",
                       "linkIsLink", "linkNo", "linkConditionEtc")
# ``LoadBreathPtn``'s row header, then ``lstInfo`` repeats until the row ends.
BREATH_PTN_HEADER = ("id", "anim", "timeChangeFaceMin", "timeChangeFaceMax")
# ``LoadBreathBase``: ``id, group`` then ``LoadBreathBaseInfo`` **four times**
# (one per level).  Each is nine columns; an empty ``pathAsset`` skips the
# level.  We keep only what drives a face — the other five are the voice clip
# and its Japanese line.
BREATH_BASE_LEVEL_COLUMNS = ("pathAsset", "nameFile", "word",
                             "absoluteOverWrite", "isWeakPointJudge",
                             "faceWeak", "faceNormal0", "faceNormal1",
                             "faceNormal2")
BREATH_LEVELS = 4


def _ints(cell: str) -> list[int]:
    return [int(x) for x in str(cell).split(",")
            if x.strip().lstrip("-").isdigit()]


def _list_assets(inst: Install, pattern, what: str):
    """``(match, raw bytes)`` for every matching TextAsset, in sorted-bundle order.

    Sorted because ``LoadAllListText`` sorts the bundle list before
    concatenating, and every reader keys on an id — so later rows overwrite
    earlier ones and the order has to be Origin's (§7.130 一).  *what* labels a
    bundle that fails to load (:func:`_read_failed`).

    Every list reader goes through here, and each bundle's environment is
    dropped and collected before the next is opened: UnityPy environments hold
    reference cycles, so without the collection four passes over sixteen
    bundles stay resident together (§7.357).
    """
    import gc
    import UnityPy

    d = inst.abdata.joinpath(*_LIST_DIR)
    if not d.is_dir():
        return
    for p in sorted(d.glob("*.unity3d")):
        try:
            env = UnityPy.load(str(p))
        except Exception as exc:  # noqa: BLE001
            _read_failed(f"{what} bundle {p.name}", exc)
            continue
        found = []
        for obj in env.objects:
            if obj.type.name != "TextAsset":
                continue
            try:
                data = obj.read()
            except Exception as exc: # noqa: BLE001
                swallowed.note("origin.hlist._list_assets: obj.read", exc)
                continue
            m = pattern.match(str(getattr(data, "m_Name", "") or ""))
            if not m:
                continue
            raw = getattr(data, "m_Script", None)
            if raw is None:
                raw = getattr(data, "script", None)
            if raw is None:
                continue
            if isinstance(raw, str):
                raw = raw.encode("utf-8", "surrogateescape")
            found.append((m, bytes(raw)))
        del env
        gc.collect()
        yield from found


def _text_assets(inst: Install, pattern, what: str = "breath"):
    """``(match, rows)`` for every matching TextAsset: :func:`_list_assets`, split into tab-separated lines."""
    for m, raw in _list_assets(inst, pattern, what):
        yield m, [ln.split("\t") for ln in raw.decode("utf-8", "replace").splitlines() if ln.strip()]


def parse_breath_ptn(inst: Install | None = None) -> dict:
    """``{mode: {anim: {…, "infos": [...]}}}`` from ``breath_{mode}_00``.

    Keyed by ``anim`` because ``dicBreathPtn`` is
    ``Dictionary<int, Dictionary<string, BreathPtn>>`` and ``LoadBreathPtn``
    only ``Add``s when ``ContainsKey`` misses — later bundles overwrite.
    """
    import re

    inst = inst or _install()
    if inst is None:
        return {}
    out: dict = {}
    for m, rows in _text_assets(inst, re.compile(r"breath_(\d+)_00$")):
        mode = int(m.group(1))
        slot = out.setdefault(mode, {})
        for c in rows:
            if len(c) < len(BREATH_PTN_HEADER):
                continue
            infos = []
            i = len(BREATH_PTN_HEADER)
            # ``while (ncols > col)`` — a null/empty cell ends the row early,
            # which is ``LoadBreathPtnInfo`` returning null (IL_0017/IL_008e…).
            while i + len(BREATH_INFO_COLUMNS) <= len(c):
                blk = c[i:i + len(BREATH_INFO_COLUMNS)]
                if not blk[0].strip():
                    break
                link = blk[6].split("/")
                infos.append({
                    "anim_ids": _ints(blk[0]),
                    "state": int(blk[1]) if blk[1].strip().lstrip("-").isdigit() else 0,
                    "conditions": _ints(blk[2]),
                    "voices": _ints(blk[3]),
                    "link_is": blk[4] == "1",
                    "link_no": int(blk[5]) if blk[5].strip().lstrip("-").isdigit() else 0,
                    "link_cond": [int(x) for x in link
                                  if x.strip().lstrip("-").isdigit()]
                    if len(link) == 2 else [],
                })
                i += len(BREATH_INFO_COLUMNS)
            slot[c[1]] = {
                "id": int(c[0]) if c[0].strip().lstrip("-").isdigit() else -1,
                "anim": c[1],
                "time_min": float(c[2] or 0.0),
                "time_max": float(c[3] or 0.0),
                "infos": infos,
            }
    return out


def breath_file_mode(mode: int) -> int:
    """The ``personality_breath_c*_{NN}_00`` a mode's voices are read from.

    ``HVoiceCtrl.LoadBreathList`` builds the name for EMode ``i`` (0‥9) as
    ``IsRange(6, i, 7) ? (i - 5) : i`` — the two 3P modes (``houshi3P`` 6,
    ``sonyu3P`` 7) have no files of their own and load ``houshi`` / ``sonyu``'s
    (§7.408).  The MMF modes (8, 9) do have their own.
    """
    return mode - 5 if 6 <= mode <= 7 else mode


def parse_breath_base(inst: Install | None = None) -> dict:
    """``{(personality, mode): {voice_id: [level0..3]}}`` — the face rows.

    A level is ``None`` when its ``pathAsset`` is empty: ``LoadBreathBaseInfo``
    returns false there and ``LoadBreathBase`` **removes** the entry from that
    level's dictionary, so an absent level is "this voice does not exist at
    this experience", not "reuse level 0".
    """
    import re

    inst = inst or _install()
    if inst is None:
        return {}
    n_lvl = len(BREATH_BASE_LEVEL_COLUMNS)
    out: dict = {}
    for m, rows in _text_assets(
            inst, re.compile(r"personality_breath_c(-?\d+)_(\d+)_(\d+)$")):
        key = (m.group(1), int(m.group(2)))
        slot = out.setdefault(key, {})
        for c in rows:
            if len(c) < 2 or not c[0].strip().lstrip("-").isdigit():
                continue
            vid = int(c[0])
            levels: list = []
            for lv in range(BREATH_LEVELS):
                i = 2 + lv * n_lvl
                blk = c[i:i + n_lvl]
                if len(blk) < n_lvl or not blk[0].strip():
                    levels.append(None)
                    continue
                faces = [int(x) for x in blk[6:9]
                         if x.strip().lstrip("-").isdigit() and int(x) != -1]
                levels.append({
                    "weak_judge": blk[4] == "1",
                    "face_weak": int(blk[5]) if blk[5].strip().lstrip("-").isdigit() else -1,
                    "face_normal": faces,
                })
            slot[vid] = {"group": int(c[1]) if c[1].strip().lstrip("-").isdigit() else 0,
                         "levels": levels}
    return out


BREATH_VERSION = 1


def bake_breath(dest: Path, *, inst: Install | None = None) -> dict:
    """Write ``breath.json`` — patterns + per-personality face rows."""
    inst = inst or _install()
    if inst is None:
        return {"ok": False, "error": "no_origin_install", "patterns": 0}
    _READ_FAILURES.clear()
    ptn = parse_breath_ptn(inst)
    base = parse_breath_base(inst)
    refused = _refuse_partial("patterns")
    if refused:
        return refused
    if not ptn or not base:
        return {"ok": False,
                "error": "no_breath_rows" if not ptn else "no_breath_base",
                "patterns": len(ptn)}
    payload = {
        "version": BREATH_VERSION,
        "info_columns": list(BREATH_INFO_COLUMNS),
        # {mode: {anim: {...}}}
        "patterns": {str(k): v for k, v in sorted(ptn.items())},
        # {personality: {mode: {voice_id: {...}}}}
        "voices": {},
    }
    for (pers, mode), voices in sorted(base.items()):
        payload["voices"].setdefault(pers, {})[str(mode)] = {
            str(k): v for k, v in sorted(voices.items())}
    write_json_atomic(dest, payload, separators=(",", ":"))
    n_infos = sum(len(p["infos"]) for m in ptn.values() for p in m.values())
    n_lvl = sum(1 for v in base.values() for e in v.values()
                for lv in e["levels"] if lv)
    return {
        "ok": True,
        "modes": sorted(ptn),
        "states": sum(len(v) for v in ptn.values()),
        "infos": n_infos,
        "personalities": len({p for p, _ in base}),
        "voices": sum(len(v) for v in base.values()),
        "levels": n_lvl,
        "file": str(dest),
    }


def load_breath(path: Path) -> dict:
    """Runtime side — reads only our JSON, never a bundle."""
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[hlist] breath read fail: {exc}")
        return {}
    if int(d.get("version") or 0) != BREATH_VERSION:
        return {}
    return d


def load_hface(path: Path) -> dict:
    """Runtime side — reads only our JSON, never a bundle."""
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[hlist] hface read fail: {exc}")
        return {}
    if int(d.get("version") or 0) != HFACE_VERSION:
        return {}
    return dict(d.get("faces") or {})


def _read_actions_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[hlist] actions read fail: {exc}")
        return {}
    if int(d.get("version") or 0) != ACTIONS_VERSION:
        print(f"[hlist] {path.name} is version {d.get('version')}, this build reads {ACTIONS_VERSION}: "
              "convert the H animations again")
        return {}
    return d


def load_actions(path: Path) -> list[dict]:
    """Runtime side — reads only our JSON, never a bundle."""
    return list(_read_actions_file(path).get("actions") or [])


_SON_FLAGS: dict = {}


def son_flags(mode: int, action_id: int) -> tuple[bool, bool] | None:
    """``(male.visibleSon, male1.visibleSon)`` as ``ChangeAnimator`` sets them for a row, or None.

    ``numMaleSon == 1`` and ``numFemaleUpperCloth1 == 1`` from ``actions.json``; None when the row is
    not there or the file was written before these columns were read (convert the H animations again).
    """
    if not _SON_FLAGS:
        from .hanim import ACTIONS_FILE
        for a in load_actions(ACTIONS_FILE):
            if "male_son" in a and "male1_son" in a:
                _SON_FLAGS[(int(a.get("mode", -1)), int(a.get("id", -1)))] = (
                    int(a["male_son"]) == 1, int(a["male1_son"]) == 1)
        _SON_FLAGS.setdefault(None, None)            # read once, even when nothing in it says
    return _SON_FLAGS.get((int(mode), int(action_id)))


_NUM_CTRL: dict = {}


def num_ctrl(mode: int, action_id: int) -> int | None:
    """A row's ``numCtrl`` from ``actions.json``, or None when the file was written before it was read."""
    if not _NUM_CTRL:
        from .hanim import ACTIONS_FILE
        for a in load_actions(ACTIONS_FILE):
            if "num_ctrl" in a:
                _NUM_CTRL[(int(a.get("mode", -1)), int(a.get("id", -1)))] = int(a["num_ctrl"])
        _NUM_CTRL.setdefault(None, None)
    return _NUM_CTRL.get((int(mode), int(action_id)))


def load_moves(path: Path) -> dict:
    """Runtime side: ``{move list: [pos.x, pos.y, pos.z, rot.x, rot.y, rot.z]}`` (Unity's axes, degrees)."""
    return dict(_read_actions_file(path).get("moves") or {})


def parse_moves(inst: Install, names) -> dict:
    """``{name: [6 floats]}`` for the ``h/list`` move lists *names*.

    ``HSceneProc.LoadLocalMoveList`` reads every row of the (bundle-concatenated) list into the same
    two vectors, so the **last** row is the one that stands; a row of fewer than six numbers is not
    one it could parse.
    """
    import re

    wanted = set(names)
    if not wanted:
        return {}
    out: dict = {}
    for m, rows in _text_assets(inst, re.compile(r"^(move_\w+)$"), "move"):
        if m.group(1) not in wanted:
            continue
        for cols in rows:
            if len(cols) < 6:
                continue
            try:
                out[m.group(1)] = [float(c) for c in cols[:6]]
            except ValueError:
                continue
    return out


# ---------------------------------------------------------------------------
# HMotionEyeNeck — where a character looks in each H state (import side)
# ---------------------------------------------------------------------------
# ``fileMotionNeck`` of each role in an ``AnimationInfo_*`` row, read off a row
# (``neck_khh_m_00`` / ``neck_khh_f_00`` / ``neck_khr_f_00_01``) and matching
# the field order of ``HSceneProc.CreateAllAnimationList``.
COL_MALE_NECK = 20
COL_FEMALE_NECK = 34
COL_FEMALE1_NECK = 71

EYENECK_VERSION = 3          # 2: the men's lists read in their own column order; 3: the MMF second man's (role male1)


def parse_eyeneck_base(inst: Install) -> dict[int, dict]:
    """``h/list`` ``eyeneckbase`` → ``{id: {...}}`` (``HMotionEyeNeck.LoadEyeNeckPtn``).

    Columns in ``Get`` order: id, behaviour, targetNeck, targetEye, rateNeck,
    rateEye.  The asset is in several bundles and a later row of an id
    replaces the earlier one (``ContainsKey`` then every field reassigned).
    """
    import re

    out: dict[int, dict] = {}
    for _m, rows in _text_assets(inst, re.compile(r"^eyeneckbase$"), "eyeneck"):
        for c in rows:
            if len(c) < 6 or not _cell(c, 0).lstrip("-").isdigit():
                continue
            try:
                out[int(c[0])] = {"behaviour": int(c[1]), "target_neck": int(c[2]),
                                  "target_eye": int(c[3]),
                                  "rate_neck": float(c[4]), "rate_eye": float(c[5])}
            except ValueError:
                continue
    return out


def parse_eyeneck_files(inst: Install, names: set[str],
                        male: set[str] = frozenset()) -> dict[str, dict]:
    """``{file: {anim state: {...}}}`` for the per-action lists.

    Both readers take 16 columns in ``Get`` order, but not the same ones.
    ``HMotionEyeNeckFemale.Load``: state name, ``isConfigEyeDisregard`` ('1'),
    ``isConfigNeckDisregard`` ('1'), four ``idEyeNecks`` (one per
    ``HExperience``), ``idEyeNeckNoVoice``.  ``HMotionEyeNeckMale.Load`` (the
    files in *male*): state name, ``eyebrow``, ``eye``, ``mouth``,
    ``openEye``, ``openMouth``, ``tears``, one ``idEyeNeck``.  Then both have
    ``rangeNeck`` and ``rangeFace`` as up, down, left, right.  A row whose
    first cell is empty is skipped; a repeated state overwrites.
    """
    import re

    if not names:
        return {}
    pat = re.compile("^(" + "|".join(re.escape(n) for n in sorted(names)) + ")$")
    out: dict[str, dict] = {}
    for m, rows in _text_assets(inst, pat, "eyeneck"):
        table = out.setdefault(m.group(1), {})
        for c in rows:
            anim = _cell(c, 0)
            if not anim or len(c) < 16:
                continue
            try:
                if m.group(1) in male:
                    row = {"eyebrow": int(c[1]), "eye": int(c[2]), "mouth": int(c[3]),
                           "open_eye": int(c[4]), "open_mouth": int(c[5]),
                           "tears": int(c[6]), "id": int(c[7])}
                else:
                    row = {"eye_disregard": _cell(c, 1) == "1",
                           "neck_disregard": _cell(c, 2) == "1",
                           "ids": [int(c[3]), int(c[4]), int(c[5]), int(c[6])],
                           "no_voice": int(c[7])}
                row["range_neck"] = [float(v) for v in c[8:12]]
                row["range_face"] = [float(v) for v in c[12:16]]
                table[anim] = row
            except ValueError:
                continue
    return out


def bake_eyeneck(dest: Path, *, inst: Install | None = None) -> dict:
    """Write ``eyeneck.json``: the id table, every action's lists, and which role reads which.

    The role → file map lives here rather than on ``HRole`` so ``actions.json``
    keeps its version (a bump would make every earlier bake read as empty).

    The third column (``paramFemale1.fileMotionNeck``) is the second woman's
    list, except on the MMF modes: there ``ChangeAnimator`` hands it to
    ``eyeneckMale1`` -- ``HMotionEyeNeckMale.Load``, the men's reader -- when
    the scene has a second man (IL_0c42‥IL_0c64), so it is kept as ``male1``
    and read in the men's column order.
    """
    inst = inst or _install()
    if inst is None:
        return {"ok": False, "error": "no_origin_install"}
    _READ_FAILURES.clear()
    base = parse_eyeneck_base(inst)
    roles: dict[str, dict] = {}
    for mode, rows in sorted(_read_list_rows(inst).items()):
        third = "male1" if mode in MMF_MODES else "female1"
        for cols in rows:
            key = f"{mode}/{_int(cols, COL_ID, -1)}"
            entry = {r: _cell(cols, i) for r, i in (("male", COL_MALE_NECK),
                                                    ("female", COL_FEMALE_NECK),
                                                    (third, COL_FEMALE1_NECK))
                     if _cell(cols, i)}
            if entry:
                roles[key] = entry            # a later row of an id replaces it
    men = ("male", "male1")
    male = {f for e in roles.values() for r, f in e.items() if r in men}
    both = male & {f for e in roles.values() for r, f in e.items() if r not in men}
    if both:
        # One file read by both readers would need both column orders; none
        # does on this install, and if one ever does it must say so.
        return {"ok": False, "error": f"eyeneck file read as male and female: {sorted(both)[:3]}"}
    files = parse_eyeneck_files(inst, {f for e in roles.values() for f in e.values()}, male)
    refused = _refuse_partial("eyeneck")
    if refused:
        return refused
    if not base:
        return {"ok": False, "error": "no_eyeneckbase"}
    missing = sorted({f for e in roles.values() for f in e.values()} - set(files))
    payload = {"version": EYENECK_VERSION,
               "base": {str(k): v for k, v in sorted(base.items())},
               "files": files, "actions": roles}
    write_json_atomic(dest, payload, separators=(",", ":"))
    return {"ok": True, "ids": len(base), "files": len(files),
            "states": sum(len(t) for t in files.values()), "actions": len(roles),
            "missing_files": missing, "file": str(dest)}


def load_eyeneck(path: Path) -> dict:
    """Runtime side — reads only our JSON, never a bundle."""
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[hlist] eyeneck read fail: {exc}")
        return {}
    if int(d.get("version") or 0) != EYENECK_VERSION:
        return {}
    return d


# ---------------------------------------------------------------------------
# Lookat_dan — which ``dan_*`` list aims the shaft in each action (import side)
# ---------------------------------------------------------------------------
# ``paramMale.fileLookAt`` of an ``AnimationInfo_*`` row, in the field order of
# ``HSceneProc.CreateAllAnimationList`` (the column before ``fileMotionNeck``).
# ``ChangeAnimator`` hands it to ``Lookat_dan.LoadList("h/list/", file)``.
COL_MALE_LOOKAT = 19
# ``paramFemale1.fileDynamicBoneRef``: on the MMF modes (8, 9) ``ChangeAnimator``
# loads it into ``lookDan1``, whose male is ``male1`` — the second man, who
# stands in the third role (``pathFemale1``) on those rows.
COL_FEMALE1_DBREF = 70
MMF_MODES = (8, 9)

LOOKAT_VERSION = 2          # 2: ``actions_male1`` (the MMF second man's list)


def bake_lookat(dest: Path, *, inst: Install | None = None) -> dict:
    """Write ``lookat.json``: each action's ``dan_*`` list and the lists' lines.

    ``actions`` maps ``"{mode}/{id}"`` to the row's ``paramMale.fileLookAt`` (a
    later row of an id replaces it, as for the other per-action lists);
    ``actions_male1`` does the same for the MMF rows' second man
    (``paramFemale1.fileDynamicBoneRef``, loaded into ``lookDan1``).
    ``lists`` holds every named list's lines as ``LoadAllListText`` joins them:
    the assets of that name from every bundle, in sorted-bundle order, so a
    later row of a motion replaces an earlier one when the lines are parsed
    (``LoadList`` finds the row by motion and re-initialises it, §7.501).  The
    runtime reads this file, never a bundle.
    """
    import re

    inst = inst or _install()
    if inst is None:
        return {"ok": False, "error": "no_origin_install"}
    _READ_FAILURES.clear()
    actions: dict[str, str] = {}
    actions_male1: dict[str, str] = {}
    for mode, rows in sorted(_read_list_rows(inst).items()):
        for cols in rows:
            key = f"{mode}/{_int(cols, COL_ID, -1)}"
            name = _cell(cols, COL_MALE_LOOKAT)
            if name:
                actions[key] = name
            name1 = _cell(cols, COL_FEMALE1_DBREF) if mode in MMF_MODES else ""
            if name1:
                actions_male1[key] = name1
    names = sorted(set(actions.values()) | set(actions_male1.values()))
    lists: dict[str, list[str]] = {}
    if names:
        pat = re.compile("^(" + "|".join(re.escape(n) for n in names) + ")$")
        for m, rows in _text_assets(inst, pat, "lookat"):
            lists.setdefault(m.group(1), []).extend("\t".join(r) for r in rows)
    refused = _refuse_partial("lookat")
    if refused:
        return refused
    if not actions:
        return {"ok": False, "error": "no_lookat_column"}
    payload = {"version": LOOKAT_VERSION, "actions": actions,
               "actions_male1": actions_male1,
               "lists": {n: lists[n] for n in sorted(lists)}}
    write_json_atomic(dest, payload, separators=(",", ":"))
    return {"ok": True, "actions": len(actions), "actions_male1": len(actions_male1),
            "lists": len(lists),
            "rows": sum(len(v) for v in lists.values()),
            "missing_files": sorted(set(names) - set(lists)), "file": str(dest)}


# ---------------------------------------------------------------------------
# What the H scene hangs on a body's nulls — the row's items and the caress hands (import side)
# ---------------------------------------------------------------------------
#: ``paramFemale.fileItem``: ``CreateAllAnimationList`` stores it after ``fileReaction`` (column 38); ``ChangeAnimator``
#: hands only the woman's to ``ItemObject.LoadItem`` (IL_0b3d‥IL_0b6c), which loads ``h/list/<fileItem>``.
COL_FEMALE_ITEM = 39
#: ``ItemObject.LoadList``: id, manifest, the object's bundle and name, the animator's bundle and name, then
#: ``(numToWhomParent, nameParent, nameSelf, kindSetParent)`` groups until an empty ``numToWhomParent``.
ITEM_PARENTS_FROM = 6
#: ``ItemObject.LoadItem``: whose body ``nameParent`` is looked up under (``FindLoop`` on ``objBodyBone``) for each
#: ``numToWhomParent`` — 0 the man, 1 the first woman, 11 the second.  2 is the argument ``ChangeAnimator`` passes as
#: null, and 100 and up are another item's own transforms, not a body's.
ITEM_BODY_ROLES = {0: "male", 1: "female", 11: "female1"}
#: ``HandCtrl.LoadLayerInfo`` loads ``aibulayer_00`` … for each of ``dicAreaLayerInfos``' areas (an array of 6,
#: ``HandCtrl..ctor`` IL_0051); column 5 is ``nameParents``, '/'-separated, which ``HandCtrl.Init`` looks up under
#: the woman's body (``FindLoop`` on ``female.objBodyBone``) as the caress hands' parents.
AIBU_LAYERS = 6
AIBU_LAYER_PARENTS = 5
ATTACH_VERSION = 1


def item_parents(rows: list[list[str]]) -> list[list]:
    """``[[role, nameParent], …]`` of an item list's rows (``ItemObject.LoadList``) whose parent is on a body."""
    out: list[list] = []
    for cols in rows:
        for i in range(ITEM_PARENTS_FROM, len(cols), 4):
            whom = _cell(cols, i)
            if not whom:
                break
            role = ITEM_BODY_ROLES.get(int(whom)) if whom.lstrip("-").isdigit() else None
            name = _cell(cols, i + 1)
            if role and name:
                out.append([role, name])
    return out


def aibu_layer_parents(rows: list[list[str]]) -> list[str]:
    """The ``nameParents`` of an ``aibulayer_NN`` list (``HandCtrl.LoadLayerInfo``), in row order, each once."""
    out: list[str] = []
    for cols in rows:
        for name in _cell(cols, AIBU_LAYER_PARENTS).split("/"):
            if name.strip() and name.strip() not in out:
                out.append(name.strip())
    return out


def bake_attach(dest: Path, *, inst: Install | None = None) -> dict:
    """Write ``attach.json``: the body nulls the H scene parents things to, from the lists that name them.

    ``items`` maps ``"{mode}/{id}"`` to the row's ``paramFemale.fileItem``; ``item_parents`` maps each named
    list to its ``[role, nameParent]`` pairs on a body (:func:`item_parents`; ``LoadAllListText`` joins the
    assets of that name from every bundle in sorted order); ``aibu_parents`` maps ``aibulayer_NN`` to its
    ``nameParents``.  The H clip bake keeps these nulls' curves (``hanim._kept_bone``): the clips animate
    them, and an item or a caress hand sits where the Animator put the null.
    """
    import re

    inst = inst or _install()
    if inst is None:
        return {"ok": False, "error": "no_origin_install"}
    _READ_FAILURES.clear()
    items: dict[str, str] = {}
    for mode, rows in sorted(_read_list_rows(inst).items()):
        for cols in rows:
            name = _cell(cols, COL_FEMALE_ITEM)
            if name:
                items[f"{mode}/{_int(cols, COL_ID, -1)}"] = name
    names = sorted(set(items.values()))
    layers = [f"aibulayer_{i:02d}" for i in range(AIBU_LAYERS)]
    rows_by: dict[str, list] = {}
    pat = re.compile("^(" + "|".join(re.escape(n) for n in names + layers) + ")$")
    for m, rows in _text_assets(inst, pat, "attach"):
        rows_by.setdefault(m.group(1), []).extend(rows)
    refused = _refuse_partial("attach")
    if refused:
        return refused
    item_out = {n: item_parents(rows_by[n]) for n in names if n in rows_by}
    aibu_out = {n: aibu_layer_parents(rows_by[n]) for n in layers if n in rows_by}
    payload = {"version": ATTACH_VERSION, "items": items, "item_parents": item_out, "aibu_parents": aibu_out}
    write_json_atomic(dest, payload, separators=(",", ":"))
    return {"ok": True, "items": len(items), "item_lists": len(item_out), "aibu_layers": len(aibu_out),
            "nulls": sorted({p for v in item_out.values() for _r, p in v} | {p for v in aibu_out.values() for p in v}),
            "missing_files": sorted((set(names) | set(layers)) - set(rows_by)), "file": str(dest)}


ITEMS_VERSION = 3


def item_rows(rows: list[list[str]]) -> list[dict]:
    """``ItemObject.LoadList``: an item list's rows, each its prefab, its animator and its parent groups.

    Column 2 is the bundle and 3 the object (``pathAssetObject`` / ``nameObject``, what ``LoadItem`` hands
    ``LoadAsset``), 4 and 5 the animator's (``LoadAnimation``: none when either is empty), then
    ``(numToWhomParent, nameParent, nameSelf, kindSetParent)`` groups until an empty ``numToWhomParent``.
    A later row of an id replaces it (``dicInfo`` by id).
    """
    out: dict = {}
    for cols in rows:
        if not _cell(cols, 0).lstrip("-").isdigit():
            continue
        parents = []
        for i in range(ITEM_PARENTS_FROM, len(cols), 4):
            whom = _cell(cols, i)
            if not whom:
                break
            parents.append([_int(cols, i, 0), _cell(cols, i + 1), _cell(cols, i + 2), _int(cols, i + 3, 0)])
        out[int(_cell(cols, 0))] = {"bundle": _cell(cols, 2), "object": _cell(cols, 3),
                                    "animator_bundle": _cell(cols, 4), "animator": _cell(cols, 5),
                                    "parents": parents}
    return [dict(v, id=k) for k, v in sorted(out.items())]


def _prefab_nodes(bundle_path: Path, name: str) -> list | None:
    """``[[name, parent index, position, rotation, scale], …]`` of prefab *name*'s transforms, root first.

    Unity's own local values (``m_LocalPosition`` / ``m_LocalRotation`` (x, y, z, w) / ``m_LocalScale``);
    the runtime mirrors them as every prefab read does (``prefab._trs``).  None when the prefab is not there.
    """
    from .prefab import loader
    L = loader(bundle_path)
    go = L.find_prefab(name)
    tr = L._transform_of(go) if go is not None else None
    if tr is None:
        return None
    out: list = []
    seen = {tr.path_id}

    def v3(d, k, default):
        x = d.get(k) or {}
        return [float(x.get(a, default[i])) for i, a in enumerate("xyz")]

    def walk(t, parent: int) -> None:
        tt = t.read_typetree()
        go_ = L.by_id.get((tt.get("m_GameObject") or {}).get("m_PathID"))
        node = go_.read_typetree().get("m_Name", "") if go_ is not None else ""
        q = tt.get("m_LocalRotation") or {}
        out.append([node, parent, v3(tt, "m_LocalPosition", (0, 0, 0)),
                    [float(q.get(a, 1.0 if a == "w" else 0.0)) for a in "xyzw"],
                    v3(tt, "m_LocalScale", (1, 1, 1))])
        me = len(out) - 1
        for co in L._children(tt, seen):
            walk(co, me)

    walk(tr, -1)
    return out


def _node_paths(nodes: list) -> dict[int, str]:
    """``{crc32(path): node}`` of a prefab's transforms (:func:`_prefab_nodes`), relative to its root.

    The binding paths of an Animator on the prefab's root: ``m_hand_L``, ``m_hand_L/s_hand_R`` …
    """
    import zlib
    out: dict[int, str] = {}
    for i, node in enumerate(nodes):
        parts, j = [], i
        while j > 0:
            parts.append(nodes[j][0])
            j = int(nodes[j][1])
        out[zlib.crc32("/".join(reversed(parts)).encode("utf-8")) & 0xFFFFFFFF] = node[0]
    return out


def _tree_nodes(nodes: list, tos: dict, n_clips: int) -> list:
    """A state's blend tree as plain nodes: a leaf ``{"clip": index}``, else its type, float(s) and children.

    1D nodes (``m_BlendType`` 0) carry their thresholds; the 2D kinds carry their two floats and their
    children's positions, which the runtime reports rather than blends (none of them is played on an item).
    """
    out = []
    for node in nodes:
        kids = [int(k) for k in (unwrap(node.get("m_ChildIndices")) or [])]
        cid = int(node.get("m_ClipID", _NO_CLIP))
        if not kids:
            out.append({"clip": cid if 0 <= cid < n_clips else -1})
            continue
        kind = int(node.get("m_BlendType", -1))
        entry = {"type": kind, "param": tos.get(int(node.get("m_BlendEventID", _NO_CLIP)), ""), "children": kids}
        if kind == 0:
            entry["thresholds"] = [float(x) for x in
                                   unwrap(unwrap(node.get("m_Blend1dData") or {}).get("m_ChildThresholdArray")) or []]
        else:
            entry["param_y"] = tos.get(int(node.get("m_BlendEventYID", _NO_CLIP)), "")
        out.append(entry)
    return out


def _item_animation(bundle_path: Path, controller: str, nodes: list) -> tuple[dict | None, dict]:
    """``ItemObject.LoadAnimation``'s controller over the item's nodes: ``({state: entry}, report)``.

    ``HActionBase.SetPlay`` plays the bodies' state on it too (``ItemObject.SetPlay`` →
    ``animItem.Play(name, 0)``, its only caller), and ``HActionBase.SetAnimatorFloat`` hands it every float
    the bodies get.  An entry is ``{"clip": …}`` for a single-clip state, converted the way a body's is on the
    item's own binding paths; ``{"tree": [nodes], "clips": {index: …}}`` for a blend-tree state, every leaf
    converted (a leaf that moves none of the item's nodes is ``"rest"``); ``{"rest": true}`` for a state whose
    clip moves none of them, which with Unity's write-defaults puts the item back at its rest.  None when
    the controller is not in the bundle.
    """
    from . import hanim
    from .meshimport import _load_bundle
    report = {"states": 0, "blend_tree": 0, "tree_2d": 0, "rest": 0, "converted": 0}
    env = _load_bundle(bundle_path)
    clips: dict = {}
    tree = None
    for obj in env.objects:
        if obj.type.name == "AnimationClip":
            clips[int(obj.path_id)] = obj
        elif obj.type.name == "AnimatorController" and tree is None:
            t = obj.read_typetree()
            if t.get("m_Name") == controller:
                tree = t
    if tree is None:
        return None, report
    ptrs = tree.get("m_AnimationClips") or []
    path_map = _node_paths(nodes)
    trees = {label: (nds, tos) for label, nds, tos in _state_trees(tree)}
    converted: dict = {}

    def leaf(index: int, name: str):
        """A clip of the controller over the item's nodes, as json, or None when it moves none of them."""
        if index not in converted:
            ptr = ptrs[index] if 0 <= index < len(ptrs) else {}
            pid = int(ptr.get("m_PathID", 0)) if isinstance(ptr, dict) else int(getattr(ptr, "path_id", 0))
            obj = clips.get(pid)
            clip = hanim._unity_clip_to_ours(obj.read(), name=name, path_map=path_map) if obj is not None else None
            converted[index] = None if clip is None else hanim._clip_to_json(clip)
        return converted[index]

    out: dict = {}
    for stem, ids in controller_states(tree):
        report["states"] += 1
        if len(ids) == 1:
            j = leaf(ids[0], stem)
            out[stem] = {"clip": j} if j is not None else {"rest": True}
            report["converted" if j is not None else "rest"] += 1
            continue
        got = trees.get(stem)
        if not ids or got is None:
            out[stem] = {"rest": True}
            report["rest"] += 1
            continue
        nodes_out = _tree_nodes(got[0], got[1], len(ptrs))
        leaves = {n["clip"] for n in nodes_out if "clip" in n and n["clip"] >= 0}
        out[stem] = {"tree": nodes_out, "clips": {str(i): leaf(i, f"{stem}#{i}") for i in sorted(leaves)}}
        report["blend_tree"] += 1
        if any("type" in n and n["type"] != 0 for n in nodes_out):
            report["tree_2d"] += 1
    return out, report


def bake_items(dest: Path, *, inst: Install | None = None) -> dict:
    """Write ``items.json``: every item list the rows name (``paramFemale.fileItem``), each row's prefab tree.

    What ``ChangeAnimator`` → ``ItemObject.LoadItem`` builds: the prefab, an Animator when the row names
    one, and each ``nameSelf`` (the prefab itself when empty) put under ``nameParent`` on the body
    ``numToWhomParent`` names.  The transforms are what a MotionIK target names (``p_kotei_head_00``,
    ``m_hand_L`` …); the item bundles (``h/common``) are read from the install the lists are read from,
    or the external one when the synced cache has none.  An item with its own Animator carries its
    controller's states as ``anim`` (:func:`_item_animation`).
    """
    import re

    from .install import source_install
    inst = inst or _install()
    if inst is None:
        return {"ok": False, "error": "no_origin_install"}
    _READ_FAILURES.clear()
    names = sorted({_cell(cols, COL_FEMALE_ITEM) for rows in _read_list_rows(inst).values()
                    for cols in rows if _cell(cols, COL_FEMALE_ITEM)})
    rows_by: dict[str, list] = {}
    pat = re.compile("^(" + "|".join(re.escape(n) for n in names) + ")$")
    for m, rows in _text_assets(inst, pat, "items"):
        rows_by.setdefault(m.group(1), []).extend(rows)
    refused = _refuse_partial("items")
    if refused:
        return refused
    roots = [r for r in (inst.abdata, source_install().abdata) if r]
    lists: dict = {}
    missing: list = []
    anim_report: dict = {}
    for name in names:
        out = []
        for row in item_rows(rows_by.get(name, [])):
            path = next((r / row["bundle"] for r in roots if (r / row["bundle"]).is_file()), None)
            nodes = _prefab_nodes(path, row["object"]) if path is not None else None
            if nodes is None:
                missing.append(f"{name}:{row['bundle']}/{row['object']}")
                continue
            extra = {}
            if row["animator_bundle"] and row["animator"]:
                apath = next((r / row["animator_bundle"] for r in roots if (r / row["animator_bundle"]).is_file()),
                             None)
                anim, rep = _item_animation(apath, row["animator"], nodes) if apath is not None else (None, {})
                if anim is None:
                    missing.append(f"{name}:{row['animator_bundle']}/{row['animator']}")
                else:
                    extra["anim"] = anim
                    for k, v in rep.items():
                        anim_report[k] = anim_report.get(k, 0) + v
            out.append(dict(row, nodes=nodes, **extra))
        lists[name] = out
    write_json_atomic(dest, {"version": ITEMS_VERSION, "lists": lists}, separators=(",", ":"))
    return {"ok": True, "lists": len(lists), "items": sum(len(v) for v in lists.values()),
            "animated": anim_report, "missing": missing, "file": str(dest)}


def load_items(path: Path) -> dict:
    """``items.json`` (:func:`bake_items`): ``{list name: [item row, …]}``; ``{}`` when missing or of another version."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(d, dict) or int(d.get("version") or 0) != ITEMS_VERSION:
        return {}
    return dict(d.get("lists") or {})


def load_attach(path: Path) -> dict:
    """``attach.json`` (:func:`bake_attach`), or ``{}`` when it is missing, unreadable or of another version."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) and int(d.get("version") or 0) == ATTACH_VERSION else {}


# ---------------------------------------------------------------------------
# YureCtrl — which breast / hip chains and breast shapes each state switches (§7.560, B3)
# ---------------------------------------------------------------------------
# The action procs' flow (D13): what the rows and the clips say about it
# ---------------------------------------------------------------------------
#: ``CreateAllAnimationList``'s ``paramFemale.isAnal`` (``Get(…) == "1"``, IL_0709–IL_0724; the column
#: before ``nameCamera`` 41): whether ``HSprite.SetSonyuStart`` offers the anal inserts on the row
COL_FEMALE_IS_ANAL = 40
#: ``paramFemale.fileSiruPaste`` (after ``fileSe`` 35) and the second woman's (after her ``fileSe`` 72), which
#: ``ChangeAnimator`` IL_0b72–IL_0ba1 hands ``SiruPasteCtrl.Load("h/list/", …)``; the man's ``fileMetaballCtrl``
#: (after ``fileMotionNeck`` 20)
COL_FEMALE_SIRU, COL_FEMALE1_SIRU, COL_MALE_METABALL = 36, 73, 21
FLOW_VERSION = 1


def clip_loop_times(bundle: Path) -> dict:
    """``{clip m_Name: m_MuscleClip.m_LoopTime}`` of one H animation bundle.

    A clip whose ``m_LoopTime`` is off plays once and holds its last frame when its state runs on (what
    ``Animator`` does with a normalised time past 1); the rest wrap.
    """
    from .meshimport import _load_bundle
    env = _load_bundle(bundle)
    out: dict = {}
    for obj in env.objects:
        if obj.type.name != "AnimationClip":
            continue
        tree = obj.read_typetree()
        muscle = tree.get("m_MuscleClip") or {}
        if "m_LoopTime" not in muscle:
            continue                          # not read: left out, so the runtime says it does not know
        out[str(tree.get("m_Name") or "")] = bool(muscle["m_LoopTime"])
    return out


def siru_paste_rows(rows: list) -> dict:
    """``SiruPasteCtrl.Load`` (IL_001d–IL_0141) of one list: ``{anim: [[normalizeTime, [where, …]], …]}``.

    Column 0 is the state; then pairs of a time and a ``,``-separated list of ``SiruParts``; an empty
    time ends the row.  ``Proc`` stops at the first row whose state is playing (IL_0105–IL_0113), so a
    later row of the same state is never read and is not kept.
    """
    out: dict = {}
    for cols in rows:
        anim = _cell(cols, 0)
        if not anim:
            continue
        timings = []
        for j in range(1, len(cols), 2):
            t = _cell(cols, j)
            if not t:
                break
            try:
                at = float(t)
            except ValueError:
                break
            where = [int(w) for w in _cell(cols, j + 1).split(",") if w.strip().lstrip("-").isdigit()]
            timings.append([at, where])
        out.setdefault(anim, timings)
    return out


def bake_flow(dest: Path, *, inst: Install | None = None, bundles=None) -> dict:
    """Write ``flow.json``: each row's ``paramFemale.isAnal`` and fluid lists, each H clip's ``m_LoopTime``.

    Rows are keyed ``"<mode>/<id>"`` (``is_anal``; ``siru``: each woman's ``fileSiruPaste``; ``metaball``: the
    man's ``fileMetaballCtrl``; ``particles``: ``HParticleCtrl``'s systems, :mod:`particle_bake`, their maps in
    ``particles/``), clips ``"<f|m>/<pack>/<m_Name>"`` -- the manifest's ``sex``, ``pack`` and
    ``clip`` -- for every pack of ``h/anim`` (or *bundles*); ``siru`` holds each paste list
    (:func:`siru_paste_rows`).
    """
    from .hanim import list_h_bundle_candidates
    inst = inst or _install()
    if inst is None:
        return {"ok": False, "error": "no_origin_install"}
    _READ_FAILURES.clear()
    rows: dict = {}
    for mode, lines in sorted(_read_list_rows(inst).items()):
        for cols in lines:
            ident = _int(cols, COL_ID, -1)
            if ident >= 0:
                rows[f"{mode}/{ident}"] = {"is_anal": _cell(cols, COL_FEMALE_IS_ANAL) == "1",
                                           "siru": [_cell(cols, COL_FEMALE_SIRU), _cell(cols, COL_FEMALE1_SIRU)],
                                           "metaball": _cell(cols, COL_MALE_METABALL)}
    import re
    names = sorted({n for r in rows.values() for n in r["siru"] if n})
    siru: dict = {}
    if names:
        pattern = re.compile("^(" + "|".join(re.escape(n) for n in names) + ")$")
        for m, lines in _text_assets(inst, pattern, "siru"):
            siru[m.group(1)] = siru_paste_rows(lines)   # ``LoadAllFolderInOneFile``: the last bundle's
    metaball = _metaball_params(inst, {r["metaball"] for r in rows.values() if r.get("metaball")})
    abdata = _fluid_source()
    particles: dict = {}
    if abdata is not None:
        bullets, nulls, fluid_failed = _bullets_and_nulls(abdata, metaball, _attach_table(inst))
        # ``HParticleCtrl``'s list and prefabs: what ``MetaballCtrl`` plays when the fluid is particles
        from .particle_bake import bake_particles, particle_rows
        prow = [r for _m, lines in _text_assets(inst, re.compile(r"^h_particle$"), "particle")
                for r in particle_rows(lines)]
        particles, p_failed = bake_particles(prow, abdata, Path(dest).parent / "particles")
        fluid_failed += p_failed
    else:
        bullets, nulls, fluid_failed = {}, {}, ["h/common: in neither the cache nor the sync source"]
    refused = _refuse_partial("flow")
    if refused:
        return refused
    clips: dict = {}
    failed: list = []
    for path in (bundles if bundles is not None else list_h_bundle_candidates(inst, limit=10_000)):
        sex = "f" if "female" in str(path).lower() else "m"
        try:
            got = clip_loop_times(Path(path))
        except Exception as exc:  # noqa: BLE001 -- one unreadable bundle is reported, the rest still read
            failed.append(f"{Path(path).name}: {exc}")
            continue
        for name, loop in got.items():
            clips[f"{sex}/{Path(path).stem}/{name}"] = loop
    write_json_atomic(dest, {"version": FLOW_VERSION, "rows": rows, "loop_time": clips, "siru": siru,
                             "metaball": metaball, "bullets": bullets, "nulls": nulls, "particles": particles},
                      separators=(",", ":"))
    return {"ok": True, "rows": len(rows), "clips": len(clips), "one_pass": sum(not v for v in clips.values()),
            "siru_lists": len(siru), "siru_missing": sorted(set(names) - set(siru)),
            "metaball_lists": len(metaball), "bullets": len(bullets), "nulls": len(nulls),
            "particles": len(particles),
            "fluid_failed": fluid_failed,
            "failed": failed, "file": str(dest)}


#: ``MetaballParam.Param``'s fields the shot reads (``MetaballCtrl.Load`` IL_00ef–IL_0632)
METABALL_FIELDS = ("ID", "Enable", "Dir_parent_sex", "ObjDirParent", "Stop_parent_sex", "ObjStopParent",
                   "BulletAssetPath", "ObjBullet", "CondomAssetPath", "ObjCondom", "Interval",
                   "S_Speed_Min", "S_Speed_Max", "S_RandomDirX", "S_RandomDirY",
                   "M_Speed_Min", "M_Speed_Max", "M_RandomDirX", "M_RandomDirY",
                   "L_Speed_Min", "L_Speed_Max", "L_RandomDirX", "L_RandomDirY", "ShootCondition")


def _metaball_params(inst: Install, names: set) -> dict:
    """``{file: [param, …]}``: each row's ``MetaballParam`` (``LoadAllFolder<MetaballParam>("h/list/", file)``)."""
    import gc
    import UnityPy
    out: dict = {}
    d = inst.abdata.joinpath(*_LIST_DIR)
    for p in sorted(d.glob("*.unity3d")):
        try:
            env = UnityPy.load(str(p))
        except Exception as exc:  # noqa: BLE001
            _read_failed(f"metaball bundle {p.name}", exc)
            continue
        for obj in env.objects:
            if obj.type.name != "MonoBehaviour":
                continue
            try:
                tree = obj.read_typetree()
            except Exception as exc:  # noqa: BLE001 -- another behaviour without a type tree: not a param list
                swallowed.note("origin.hlist._metaball_params: read_typetree", exc)
                continue
            name = str(tree.get("m_Name") or "")
            if name in names and isinstance(tree.get("param"), list):
                out[name] = [{k: prm.get(k) for k in METABALL_FIELDS} for prm in tree["param"]]
        del env
        gc.collect()
    return out


def _fluid_source() -> Path | None:
    """The ``abdata`` that carries ``h/common`` (the bullet prefabs): the cache, else the sync source."""
    from .install import source_install
    for inst in (configured(), source_install()):
        if inst is not None and inst.valid and inst.abdata.joinpath("h", "common").is_dir():
            return inst.abdata
    return None


def _unity_local(t: dict):
    """A ``Transform`` type tree's local matrix (a 4×4 numpy array), in Unity's space."""
    import numpy as _np                     # numpy only on the import side
    p, q, s = t["m_LocalPosition"], t["m_LocalRotation"], t["m_LocalScale"]
    x, y, z, w = (float(q[k]) for k in "xyzw")
    r = _np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = _np.eye(4)
    m[:3, :3] = r * _np.array([float(s["x"]), float(s["y"]), float(s["z"])])
    m[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
    return m


def _bullets_and_nulls(abdata: Path, params: dict, attach: dict) -> tuple:
    """``(bullets, nulls, failed)`` out of the bullet bundles.

    A bullet is its start body (``start_mass``) and the metaball nodes chained after it by their
    ``MetaballJoint.transformJoint`` (``radius`` = ``MetaballNode.baseRadius``, ``limit`` =
    ``limitLength``), with the start body's ``MetaCollider`` drags and times.  A null is an emitter or
    stop name found in those bundles: its prefab root, the bone that root hangs on (*attach*, the
    ``parent_object_*`` lists) and its matrix under that bone, in Unity's space.
    """
    import gc
    import UnityPy
    import numpy as np
    want_b: dict = {}
    want_n: set = set()
    for prms in params.values():
        for prm in prms:
            if prm.get("Enable") and prm.get("ObjBullet"):
                want_b.setdefault(str(prm.get("BulletAssetPath") or ""), set()).add(str(prm["ObjBullet"]))
            for key in ("ObjDirParent", "ObjStopParent"):
                # a body bone (``cf_`` / ``cm_``) is the skeleton's own; only the H scene's nulls are looked up
                if prm.get(key) and not str(prm[key]).lower().startswith(("cf_", "cm_")):
                    want_n.add(str(prm[key]))
    bullets, nulls, failed = {}, {}, []
    paths = sorted(set(want_b) | {"h/common/00_00.unity3d"})
    for rel in paths:
        bundle = abdata.joinpath(*rel.split("/"))
        if not bundle.is_file():
            failed.append(f"{rel}: not found")
            continue
        env = UnityPy.load(str(bundle))
        by = {o.path_id: o for o in env.objects}
        trees: dict = {}

        def tree(pid):
            if pid not in trees:
                trees[pid] = by[pid].read_typetree()
            return trees[pid]

        go_of_tr, tr_of_go, comps = {}, {}, {}
        for o in env.objects:
            if o.type.name != "GameObject":
                continue
            g = tree(o.path_id)
            cs = [c["component"]["m_PathID"] for c in g.get("m_Component", [])]
            comps[o.path_id] = cs
            for c in cs:
                if c in by and by[c].type.name == "Transform":
                    tr_of_go[o.path_id], go_of_tr[c] = c, o.path_id

        def script(pid):
            try:
                return by[pid].read().m_Script.read().m_Name
            except Exception as exc:  # noqa: BLE001 -- a component with no readable script: not one of ours
                swallowed.note("origin.hlist._bullets_and_nulls: script", exc)
                return ""

        names = {}
        for gid in comps:
            names.setdefault(tree(gid).get("m_Name", ""), gid)
        for name in sorted(want_b.get(rel, ())):
            gid = names.get(name)
            if gid is None:
                failed.append(f"{rel}: no {name}")
                continue
            nodes, start = {}, None
            for ch in tree(tr_of_go[gid]).get("m_Children", []):
                cg = go_of_tr.get(ch["m_PathID"])
                info = {"tr": ch["m_PathID"]}
                for c in comps.get(cg, ()):
                    kind = by[c].type.name
                    if kind == "Rigidbody":
                        t = tree(c)
                        info["mass"], info["gravity"] = float(t["m_Mass"]), bool(t["m_UseGravity"])
                    elif kind == "MonoBehaviour":
                        s = script(c)
                        t = tree(c)
                        if s == "MetaballNode":
                            info["radius"] = float(t["baseRadius"])
                        elif s == "MetaballJoint":
                            info["limit"], info["joint"] = float(t["limitLength"]), t["transformJoint"]["m_PathID"]
                        elif s == "MetaCollider":
                            info["collider"] = {k: float(t[k]) for k in ("dragHit", "dragAir", "timeDropDown",
                                                                         "sleepTime")}
                if "radius" in info:
                    nodes[info.get("joint", "start")] = info    # a node with no joint sits on the start body
                    info.setdefault("limit", 0.0)
                elif "mass" in info:
                    start = info
            if start is None:
                failed.append(f"{rel}: {name} has no start body")
                continue
            if "start" in nodes:
                nodes[start["tr"]] = nodes.pop("start")
            chain, at = [], start["tr"]
            while at in nodes:
                chain.append({"radius": nodes[at]["radius"], "limit": nodes[at]["limit"]})
                at = nodes[at]["tr"]
            col = start.get("collider") or {}
            bullets[name] = {"start_mass": start["mass"], "nodes": chain, "drag_hit": col.get("dragHit", 10.0),
                             "drag_air": col.get("dragAir", 0.0), "time_drop_down": col.get("timeDropDown", 3.0),
                             "sleep_time": col.get("sleepTime", 3.0)}
        for name in sorted(want_n):
            gid = names.get(name)
            if gid is None or name in nulls:
                continue
            m, at, root = np.eye(4), tr_of_go[gid], gid
            while at:
                t = tree(at)
                m = _unity_local(t) @ m
                root = go_of_tr.get(at, root)
                at = t.get("m_Father", {}).get("m_PathID", 0)
            rname = tree(root).get("m_Name", "")
            bone = attach.get(rname)
            if bone is None:
                failed.append(f"{rel}: {name} hangs on {rname}, which no parent_object list places")
                continue
            nulls[name] = {"bone": bone, "root": rname, "matrix": [round(float(v), 7) for v in m.reshape(-1)]}
        del env
        gc.collect()
    return bullets, nulls, failed


def _attach_table(inst: Install) -> dict:
    """``{object: bone}`` from every ``parent_object_*`` list (``bone, bundle, object`` triples, one row)."""
    import re
    out: dict = {}
    for _m, rows in _text_assets(inst, re.compile(r"^parent_object_(base_)?(male|female)\w*$"), "attach"):
        for cols in rows:
            for i in range(0, len(cols) - 2, 3):
                bone, obj = cols[i].strip(), cols[i + 2].strip()
                if bone and obj:
                    out.setdefault(obj, bone)
    return out


def load_flow(path: Path) -> dict:
    """``flow.json`` (:func:`bake_flow`), or ``{}`` when it is missing, unreadable or of another version."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) and int(d.get("version") or 0) == FLOW_VERSION else {}


# ---------------------------------------------------------------------------
#: ``CreateAllAnimationList``'s columns: ``paramFemale.isYure`` / ``fileYure``, and the second woman's
COL_FEMALE_IS_YURE, COL_FEMALE_YURE = 31, 32
COL_FEMALE1_IS_YURE, COL_FEMALE1_YURE = 68, 69
YURE_VERSION = 1
#: ``YureExcel.Param``: the name, the four chains (``dictDynamicBoneBust`` 0‥3), then the nine breast
#: shapes (``cf_BustShapeMaskID``'s order) left and right in turn
YURE_CHAINS = ("Mune_L", "Mune_R", "Siri_L", "Siri_R")


def _yure_assets(inst: Install, names: set):
    """``(name, [row dict…])`` for every ``YureExcel`` MonoBehaviour named in *names*, in sorted-bundle order.

    ``YureCtrl.LoadAllExcel`` loads the asset of that name from each bundle
    of the folder in turn and ``LoadParam`` keys the rows by state name, so a
    later bundle's row replaces an earlier one's — the order has to be the
    game's.  The assets carry a typetree (``param``).
    """
    import gc
    import UnityPy

    d = inst.abdata.joinpath(*_LIST_DIR)
    if not d.is_dir():
        return
    for p in sorted(d.glob("*.unity3d")):
        try:
            env = UnityPy.load(str(p))
        except Exception as exc:  # noqa: BLE001
            _read_failed(f"yure bundle {p.name}", exc)
            continue
        found = []
        for obj in env.objects:
            if obj.type.name != "MonoBehaviour":
                continue
            try:
                tree = obj.read_typetree()
            except Exception as exc:  # noqa: BLE001
                swallowed.note("origin.hlist._yure_assets: read_typetree", exc)
                continue
            if tree.get("m_Name") in names and isinstance(tree.get("param"), list):
                found.append((tree["m_Name"], tree["param"]))
        del env
        gc.collect()
        yield from found


def yure_row(param: dict) -> tuple[list[bool], list[bool], list[bool]]:
    """``(isActives[4], breast shapes left[9], right[9])`` of one ``YureExcel.Param`` (``YureCtrl.LoadParam``).

    The shapes follow the chains in the asset's field order, left and right
    alternating (``UDPos_L``, ``UDPos_R``, ``LRRot_L``…); a true is a shape
    left on (``BreastShapeInfo``, which ``Shape`` turns into
    ``DisableShapeBodyID(…, !value)``).
    """
    keys = [k for k in param if k != "name"]
    if keys[:4] != list(YURE_CHAINS) or len(keys) != 22:
        raise ValueError(f"YureExcel.Param fields not as read from the IL: {keys}")
    shapes = keys[4:]
    left, right = shapes[0::2], shapes[1::2]
    if not all(k.endswith("L") for k in left) or not all(k.endswith("R") for k in right):
        raise ValueError(f"YureExcel.Param shapes not left / right in turn: {shapes}")
    return ([bool(param[k]) for k in YURE_CHAINS], [bool(param[k]) for k in left],
            [bool(param[k]) for k in right])


def bake_yure(dest: Path, *, inst: Install | None = None) -> dict:
    """Write ``yure.json``: each action's ``isYure`` / ``fileYure`` for each woman, and the tables' rows.

    ``actions`` maps ``"{mode}/{id}"`` to ``{"f": [isYure, file], "f1": [...]}``
    (``f1`` on the rows with a second woman), a later row of an id replacing
    it.  ``tables`` maps a file to ``{state: [actives, left, right]}`` as
    ``LoadParam`` keys it.  The runtime reads this file, never a bundle.
    """
    inst = inst or _install()
    if inst is None:
        return {"ok": False, "error": "no_origin_install"}
    _READ_FAILURES.clear()
    actions: dict = {}
    for mode, rows in sorted(_read_list_rows(inst).items()):
        for cols in rows:
            entry = {}
            for who, c_is, c_file in (("f", COL_FEMALE_IS_YURE, COL_FEMALE_YURE),
                                      ("f1", COL_FEMALE1_IS_YURE, COL_FEMALE1_YURE)):
                name = _cell(cols, c_file)
                if name:
                    entry[who] = [_cell(cols, c_is) == "1", name]
            if entry:
                actions[f"{mode}/{_int(cols, COL_ID, -1)}"] = entry
    names = {pair[1] for entry in actions.values() for pair in entry.values()}
    tables: dict = {}
    for name, params in _yure_assets(inst, names):
        table = tables.setdefault(name, {})
        for param in params:
            table[str(param.get("name") or "")] = [list(part) for part in yure_row(param)]
    refused = _refuse_partial("yure")
    if refused:
        return refused
    if not actions:
        return {"ok": False, "error": "no_yure_column"}
    write_json_atomic(dest, {"version": YURE_VERSION, "actions": actions,
                             "tables": {n: tables[n] for n in sorted(tables)}}, separators=(",", ":"))
    return {"ok": True, "actions": len(actions), "tables": len(tables),
            "states": sum(len(t) for t in tables.values()),
            "missing_files": sorted(names - set(tables)), "file": str(dest)}


# --------------------------------------------------------------------------
# ``MotionIK`` (Animation_Progress D14): the IK targets and bone corrections an
# H action applies after its animation.  ``HSceneProc.ChangeAnimator`` loads
# ``LoadAllFolderInOneFile("h/list/", <controller file>)`` into each role's
# ``MotionIK``, and the TextAsset is the binary ``MotionIKData.Read`` reads.

#: ``frameNo`` 0‥7: each limb's effector-target offset (2·i) and bend-goal offset (2·i + 1), limbs in
#: ``MOTION_IK_PARTS`` order (``MotionIK.IKTargetPair.IKTargetLength`` = 4 × 2, ``MotionIKData.InitFrame``)
MOTION_IK_LIMB_FRAMES = 8
#: ``FrameCorrect``'s bones: ``frameNo`` 8 onward, in ``FrameCorrect..cctor``'s order
MOTION_IK_FRAMES = ("cf_j_hips", "cf_j_spine01", "cf_j_spine02", "cf_j_spine03", "cf_j_neck", "cf_j_head",
                    "cf_j_waist01", "cf_j_waist02",
                    "cf_j_thumb01_L", "cf_j_index01_L", "cf_j_middle01_L", "cf_j_ring01_L", "cf_j_little01_L",
                    "cf_j_thumb01_R", "cf_j_index01_R", "cf_j_middle01_R", "cf_j_ring01_R", "cf_j_little01_R",
                    "cf_j_bust01_L", "cf_j_bust02_L", "cf_j_bust03_L", "cf_j_bust01_R", "cf_j_bust02_R", "cf_j_bust03_R")
#: ``MotionIKData.State``'s four parts, in the order ``Read`` fills them
MOTION_IK_PARTS = ("leftHand", "rightHand", "leftFoot", "rightFoot")
MOTION_IK_VERSION = 1


class _Reader:
    """``System.IO.BinaryReader`` over *raw*: little-endian ints and floats, 7-bit length-prefixed UTF-8 strings."""

    def __init__(self, raw: bytes):
        self.raw, self.at = raw, 0

    def take(self, n: int) -> bytes:
        """The next *n* bytes; ``ValueError`` past the end."""
        if self.at + n > len(self.raw):
            raise ValueError(f"read past the end at {self.at} (+{n} of {len(self.raw)})")
        out = self.raw[self.at:self.at + n]
        self.at += n
        return out

    def i32(self) -> int:
        """``ReadInt32``."""
        return int.from_bytes(self.take(4), "little", signed=True)

    def f32(self) -> float:
        """``ReadSingle``."""
        import struct
        return struct.unpack("<f", self.take(4))[0]

    def string(self) -> str:
        """``ReadString``: a 7-bit encoded length, then UTF-8."""
        n = shift = 0
        while True:
            b = self.take(1)[0]
            n |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return self.take(n).decode("utf-8")


def parse_motion_ik_data(raw: bytes) -> list[dict]:
    """``MotionIKData.Read``: the states of one controller's IK file.

    Each state is ``{"name", "parts": {part: {sex, target, weightPos, weightAng, chain, weight}},
    "frames": [{frameNo, editNo, shapes: [{shapeNo, small: [pos3, ang3], large: [pos3, ang3]}]}]}``.
    Raises ``ValueError`` when the bytes do not end exactly where the reader does.
    """
    r = _Reader(raw)
    states = []
    for _ in range(r.i32()):
        st = {"name": r.string(), "parts": {}, "frames": []}
        for part in MOTION_IK_PARTS:
            st["parts"][part] = {"sex": r.i32(), "target": r.string(), "weightPos": r.f32(),
                                 "weightAng": r.f32(), "chain": r.string(), "weight": r.f32()}
        for _f in range(r.i32()):
            fr = {"frameNo": r.i32(), "editNo": r.i32(), "shapes": []}
            for _s in range(r.i32()):
                sh = {"shapeNo": r.i32()}
                for key in ("small", "large"):
                    sh[key] = [[r.f32() for _ in range(3)], [r.f32() for _ in range(3)]]
                fr["shapes"].append(sh)
            st["frames"].append(fr)
        states.append(st)
    if r.at != len(raw):
        raise ValueError(f"{len(raw) - r.at} bytes left after {len(states)} states")
    return states


def parse_motion_ik(inst: Install, names: set[str]) -> dict[str, list[dict]]:
    """``{controller file: states}`` for every IK file in *names* (the actions' controller files)."""
    import re
    out: dict[str, list[dict]] = {}
    for m, raw in _list_assets(inst, re.compile(r"(k[a-z0-9]+_[fm]\w*)$"), "motion_ik"):
        name = m.group(1)
        if name not in names:
            continue
        try:
            out[name] = parse_motion_ik_data(raw)     # a later bundle's file replaces it
        except ValueError as exc:
            _read_failed(f"motion_ik {name}", exc)
    return out


def bake_motion_ik(dest: Path, *, inst: Install | None = None) -> dict:
    """Write ``motion_ik.json``: each action controller's ``MotionIKData``, keyed by its file name."""
    inst = inst or _install()
    if inst is None:
        return {"ok": False, "error": "no_origin_install"}
    _READ_FAILURES.clear()
    names = set()
    for _mode, rows in sorted(_read_list_rows(inst).items()):
        for cols in rows:
            for col in (COL_MALE, COL_FEMALE, COL_FEMALE1):
                file = _cell(cols, col + 1)
                if file:
                    names.add(file)
    files = parse_motion_ik(inst, names)
    refused = _refuse_partial("motion_ik")
    if refused:
        return refused
    if not files:
        return {"ok": False, "error": "no_motion_ik"}
    payload = {"version": MOTION_IK_VERSION, "limb_frames": MOTION_IK_LIMB_FRAMES,
               "frames": list(MOTION_IK_FRAMES), "files": files}
    write_json_atomic(dest, payload, separators=(",", ":"))
    return {"ok": True, "files": len(files), "states": sum(len(v) for v in files.values()),
            "missing_files": sorted(names - set(files)), "file": str(dest)}


# ---------------------------------------------------------------------------
# HVoiceCtrl — the H voice lines, their patterns and lengths (import side)
# ---------------------------------------------------------------------------
# What ``HVoiceCtrl.Init`` loads for a heroine whose ``Heroine.ChaName`` is
# ``"c" + FixCharaIDOrPersonality.MinusThroughToString("00")``:
#
# * ``LoadVoiceList`` -> ``LoadVoice``: ``personality_voice_<cha>_<m>_<nn>``
#   for the voice modes ``m`` 0..10 (``IsRange(7, m, 8)`` reads ``m - 5``'s
#   files) and ``nn`` 00..09, appended in that order; a row is the id and four
#   experience blocks of 23 cells (priority, pathAsset, nameFile, word, 15
#   ``isPlayConditions``, isOneShot, notOverwrite, face, eyeneck) -- an empty
#   pathAsset removes that block's entry and skips 21 cells (IL_0221..IL_036b);
# * ``LoadVoicePtn``: ``voice_<m>_00``, a row is the id then ``VoicePtnInfo``
#   blocks of 9 cells (animIDs, state, conditions, voices, second conditions,
#   second voices, isLink, link no, ``condition/etc``) until a block comes back
#   null (``LoadVoicePtnInfo``);
# * ``LoadShortBreath``: ``personality_shortbreath_<cha>_00``, the id and four
#   blocks of 5 (pathAsset, nameFile, word, notOverwrite, face);
#   ``LoadShortBreathPtn``: ``shortbreath_00``, the id then blocks of 3
#   (state, conditions, voices);
# * ``LoadBreathList`` -> ``LoadBreathBase``: ``personality_breath_<cha>_<e>_00``
#   for the modes ``e`` 0..9 (``IsRange(6, e, 7)`` reads ``e - 5``), the id, the
#   group and four blocks of 9 (pathAsset ... faceNormal x3); an empty pathAsset
#   removes the block (``LoadBreathBaseInfo`` returns false).
#
# Each line's length is the ``AudioClip.m_Length`` of ``nameFile`` in the
# bundle ``pathAsset`` -- read from the clip's header only, so no audio is
# decoded and none is written (``Manager.Voice.IsVoiceCheck`` is "that clip is
# still playing", which is all the flow needs from the sound).
VOICE_VERSION = 1
VOICE_MODES = 11                  # ``LoadVoiceList``: 0..10 (``VoicePtnConditionDelegate[11]``)
VOICE_LEVEL_CELLS = 23
VOICE_CONDITIONS = 15             # ``VoiceInfo.isPlayConditions = new bool[15]``
BREATH_MODES = 10                 # ``LoadBreathList``: 0..9
#: ``CreateAllAnimationList`` columns the voice conditions read off a row
#: (``nowAnimationInfo``), counted along its sequential ``Get`` calls; 19, 20,
#: 26, 34, 39 and 52 are the anchors the other readers here already use.
COL_KIND_HOUSHI = 4               # TryParse -> kindHoushi
COL_SYS_TAII = 43                 # TryParse -> sysTaii
COL_FEMALE_INITIATIVE = 47        # == "1"
COL_HOUSHI_LOOP_ACTION = 48       # "W/S", Split('/') of length 2
COL_SPLASH = 49                   # == "1"
COL_VOICE_IDS = 50                # "main,sub", Split(',') of length 2, else -1/-1
COL_VOICE_KIND = 51               # GetIntTryParse(..., 0)
COL_SHORT3P_MAIN = 80             # the row's extra cells: "a/b/c/d" (GetIntTryParse(..., -1))
COL_SHORT3P_SUB = 81


def voice_file_mode(mode: int) -> int:
    """``LoadVoice``'s file index for voice mode *mode*: 3P's 7 and 8 read 2 and 3."""
    return mode - 5 if 7 <= mode <= 8 else mode


def _voice_int(cell: str, default: int = 0) -> int:
    s = str(cell).strip()
    return int(s) if s.lstrip("-").isdigit() else default


def _voice_ptn_info(c: list, i: int) -> tuple[dict | None, int]:
    """``LoadVoicePtnInfo`` at cell *i*: ``(info, next cell)``; None when a required cell is empty."""
    def cell(k):
        return c[k] if k < len(c) else ""
    if not cell(i).strip():
        return None, i
    info = {"anim_ids": _ints(cell(i)), "state": _voice_int(cell(i + 1))}
    if not cell(i + 2).strip():
        return None, i
    info["conditions"] = _ints(cell(i + 2))
    if not cell(i + 3).strip():
        return None, i
    info["voices"] = _ints(cell(i + 3))
    if not cell(i + 4).strip():
        return None, i
    info["second_conditions"] = _ints(cell(i + 4))
    info["second_voices"] = _ints(cell(i + 5)) if cell(i + 5).strip() else []
    info["link"] = {"is_link": cell(i + 6) == "1", "no": _voice_int(cell(i + 7))}
    if not cell(i + 8).strip():
        return None, i
    parts = cell(i + 8).split("/")
    info["link"]["condition"], info["link"]["etc"] = (
        (_voice_int(parts[0]), _voice_int(parts[1])) if len(parts) == 2 else (-1, -1))
    return info, i + 9


def _voice_rows(texts: list) -> list:
    rows = []
    for raw in texts:
        rows += [ln.split("\t") for ln in raw.decode("utf-8", "replace").splitlines() if ln.strip()]
    return rows


def _voice_lines(rows: list) -> dict:
    """``LoadVoice``'s rows: ``{id: [block 0..3 or None]}``, a later row of an id overwriting its blocks."""
    table: dict = {}
    for c in rows:
        levels = table.setdefault(str(_voice_int(c[0], -1)), [None] * 4)
        for lv in range(4):
            blk = c[1 + lv * VOICE_LEVEL_CELLS:1 + (lv + 1) * VOICE_LEVEL_CELLS]
            blk = blk + [""] * (VOICE_LEVEL_CELLS - len(blk))
            if not blk[1].strip():
                levels[lv] = None                  # ``Remove(id)`` for this block
                continue
            levels[lv] = {"priority": _voice_int(blk[0]), "path": blk[1], "file": blk[2],
                          "cond": [x == "1" for x in blk[4:4 + VOICE_CONDITIONS]],
                          "one_shot": blk[19] == "1", "not_overwrite": blk[20] == "1",
                          "face": _voice_int(blk[21], -1), "eyeneck": _voice_int(blk[22], -1)}
    return table


def _breath_lines(rows: list) -> dict:
    """``LoadBreathBase``'s rows: ``{id: {"group", "levels": [block or None]}}``."""
    table: dict = {}
    for c in rows:
        entry = table.setdefault(str(_voice_int(c[0], -1)), {"group": 0, "levels": [None] * 4})
        entry["group"] = _voice_int(c[1] if len(c) > 1 else "", 0)
        for lv in range(4):
            blk = (c[2 + lv * 9:11 + lv * 9] + [""] * 9)[:9]
            if not blk[0].strip():
                entry["levels"][lv] = None
                continue
            entry["levels"][lv] = {
                "path": blk[0], "file": blk[1], "abs": blk[3] == "1", "weak": blk[4] == "1",
                "face_weak": _voice_int(blk[5], -1),
                "faces": [f for f in (_voice_int(x, -1) for x in blk[6:9]) if f != -1]}
    return table


def parse_voice_tables(inst: Install) -> dict:
    """Every voice / short-breath / breath table of the install, as ``HVoiceCtrl.Init`` reads them."""
    import re

    pat = re.compile(r"(personality_voice_(c-?\d+)_(\d\d)_(\d\d)|voice_(\d\d)_00|"
                     r"personality_shortbreath_(c-?\d+)_00|shortbreath_00|"
                     r"personality_breath_(c-?\d+)_(\d\d)_00)$")
    texts: dict = {}
    for m, raw in _list_assets(inst, pat, "voice"):
        texts.setdefault(m.group(1), []).append(raw)    # sorted-bundle order (``LoadAllListText``)
    chas = set()
    for name in texts:
        m = pat.match(name)
        chas |= {g for g in (m.group(2), m.group(6), m.group(7)) if g}
    ptn: dict = {}
    for mode in range(VOICE_MODES):
        table: dict = {}
        for c in _voice_rows(texts.get(f"voice_{mode:02d}_00", [])):
            infos, i = [], 1
            while True:
                info, i = _voice_ptn_info(c, i)
                if info is None:
                    break
                infos.append(info)
            table[str(_voice_int(c[0], -1))] = infos   # ``lstInfo.Clear()`` then filled again
        ptn[str(mode)] = table
    lines: dict = {}
    for cha in sorted(chas):
        voice: dict = {}
        for fmode in sorted({voice_file_mode(m) for m in range(VOICE_MODES)}):
            rows = []
            for nn in range(10):
                rows += _voice_rows(texts.get(f"personality_voice_{cha}_{fmode:02d}_{nn:02d}", []))
            if rows:
                voice[str(fmode)] = _voice_lines(rows)
        short: dict = {}
        for c in _voice_rows(texts.get(f"personality_shortbreath_{cha}_00", [])):
            blocks = []
            for lv in range(4):
                blk = (c[1 + lv * 5:6 + lv * 5] + [""] * 5)[:5]
                blocks.append({"path": blk[0], "file": blk[1], "not_overwrite": blk[3] == "1",
                               "face": _voice_int(blk[4], -1)})
            short[str(_voice_int(c[0], -1))] = blocks
        breath: dict = {}
        for emode in range(BREATH_MODES):
            fmode = emode - 5 if 6 <= emode <= 7 else emode
            table = _breath_lines(_voice_rows(texts.get(f"personality_breath_{cha}_{fmode:02d}_00", [])))
            if table:
                breath[str(emode)] = table
        lines[cha] = {"voice": voice, "short": short, "breath": breath}
    short_ptn: dict = {}
    for c in _voice_rows(texts.get("shortbreath_00", [])):
        infos, i = [], 1
        while True:
            blk = (c[i:i + 3] + ["", "", ""])[:3]
            if not blk[0].strip() or not blk[1].strip() or not blk[2].strip():
                break
            infos.append({"state": _voice_int(blk[0]), "conditions": _ints(blk[1]), "voices": _ints(blk[2])})
            i += 3
        short_ptn[str(_voice_int(c[0], -1))] = infos
    return {"ptn": ptn, "lines": lines, "short_ptn": short_ptn}


def parse_voice_actions(inst: Install) -> dict:
    """``{"mode/id": {...}}``: the ``AnimationListInfo`` fields the voice conditions read."""
    def ptns(cols, col):
        parts = _cell(cols, col).split("/") if col < len(cols) else []
        return [_voice_int(p, -1) for p in parts] if len(parts) == 4 else [0, 0, 0, 0]

    out: dict = {}
    for mode, rows in sorted(_read_list_rows(inst).items()):
        for cols in rows:
            loop = _cell(cols, COL_HOUSHI_LOOP_ACTION).split("/")
            ids = _cell(cols, COL_VOICE_IDS).split(",")
            main, sub = ((_voice_int(ids[0]), _voice_int(ids[1])) if len(ids) == 2 else (-1, -1))
            out[f"{mode}/{_int(cols, COL_ID, -1)}"] = {   # a later row of an id replaces it
                "kind_houshi": _voice_int(_cell(cols, COL_KIND_HOUSHI)),
                "sys_taii": _voice_int(_cell(cols, COL_SYS_TAII)),
                "female_initiative": _cell(cols, COL_FEMALE_INITIATIVE) == "1",
                "loop_w": _voice_int(loop[0]) if len(loop) == 2 else 0,
                "loop_s": _voice_int(loop[1]) if len(loop) == 2 else 0,
                "splash": _cell(cols, COL_SPLASH) == "1",
                "main_voice": main, "sub_voice": sub,
                "voice_kind": _voice_int(_cell(cols, COL_VOICE_KIND)),
                "short3p_main": ptns(cols, COL_SHORT3P_MAIN),
                "short3p_sub": ptns(cols, COL_SHORT3P_SUB),
            }
    return out


def parse_voice_lengths(inst: Install, wanted: dict) -> tuple:
    """``({"bundle|clip": seconds}, [missing])`` for *wanted* ``{bundle: {clip}}``, from the clips' headers."""
    import gc
    import UnityPy

    from .install import source_install

    # the synced cache carries ``h/list`` but not ``sound/``: the clips are read
    # where the bundle is, the cache first and then the game's own folder
    roots = [i.abdata for i in (inst, source_install()) if i is not None and i.valid]
    out, missing = {}, []
    for bundle, clips in sorted(wanted.items()):
        p = next((r / bundle for r in roots if (r / bundle).is_file()), roots[0] / bundle)
        if not p.is_file():
            missing += [f"{bundle}|{c}" for c in sorted(clips)]
            continue
        try:
            env = UnityPy.load(str(p))
        except Exception as exc:  # noqa: BLE001
            _read_failed(f"voice bundle {bundle}", exc)
            continue
        have = {}
        for obj in env.objects:
            if obj.type.name != "AudioClip":
                continue
            try:
                d = obj.read()
            except Exception as exc:  # noqa: BLE001
                swallowed.note("origin.hlist.parse_voice_lengths: obj.read", exc)
                continue
            have[str(getattr(d, "m_Name", "") or "")] = float(getattr(d, "m_Length", 0.0) or 0.0)
        del env
        gc.collect()
        for c in sorted(clips):
            if c in have:
                out[f"{bundle}|{c}"] = have[c]
            else:
                missing.append(f"{bundle}|{c}")
    return out, missing


def _voice_wanted(lines: dict) -> dict:
    """``{bundle: {clip}}`` every line of every table names."""
    wanted: dict = {}

    def add(lv):
        if lv and lv.get("path"):
            wanted.setdefault(lv["path"], set()).add(lv["file"])
    for per in lines.values():
        for table in per["voice"].values():
            for levels in table.values():
                for lv in levels:
                    add(lv)
        for levels in per["short"].values():
            for lv in levels:
                add(lv)
        for table in per["breath"].values():
            for entry in table.values():
                for lv in entry["levels"]:
                    add(lv)
    return wanted


def voice_lines_path(dest: Path, cha: str) -> Path:
    """Where one personality's lines go: ``voice/<cha>.json`` next to ``voice.json``."""
    return dest.parent / dest.stem / f"{cha}.json"


def bake_voice(dest: Path, *, inst: Install | None = None) -> dict:
    """Write ``voice.json`` (patterns, short-breath patterns, the action fields) and, per personality,
    ``voice/<cha>.json`` (its voice, short-breath and breath lines with their clips' lengths) -- one
    file per personality, so a scene reads only the heroines it has."""
    inst = inst or _install()
    if inst is None:
        return {"ok": False, "error": "no_origin_install"}
    _READ_FAILURES.clear()
    tables = parse_voice_tables(inst)
    actions = parse_voice_actions(inst)
    lengths, missing = parse_voice_lengths(inst, _voice_wanted(tables["lines"]))
    refused = _refuse_partial("voice")
    if refused:
        return refused
    if not any(tables["ptn"].values()) or not tables["lines"]:
        return {"ok": False, "error": "no_voice_lists"}
    for cha, per in tables["lines"].items():
        keys = {f"{b}|{c}" for b, clips in _voice_wanted({cha: per}).items() for c in clips}
        write_json_atomic(voice_lines_path(dest, cha),
                          {"version": VOICE_VERSION, "cha": cha, **per,
                           "lengths": {k: lengths[k] for k in sorted(keys) if k in lengths}},
                          separators=(",", ":"))
    payload = {"version": VOICE_VERSION, "ptn": tables["ptn"], "short_ptn": tables["short_ptn"],
               "actions": actions, "personalities": sorted(tables["lines"]),
               "missing_lengths": len(missing)}
    write_json_atomic(dest, payload, separators=(",", ":"))
    return {"ok": True, "personalities": len(tables["lines"]),
            "patterns": sum(len(t) for t in tables["ptn"].values()),
            "lines": sum(len(t) for p in tables["lines"].values() for t in p["voice"].values()),
            "lengths": len(lengths), "missing_lengths": missing[:20], "n_missing": len(missing),
            "file": str(dest)}


def load_voice(path: Path) -> dict:
    """Runtime side -- reads only our JSON, never a bundle (``voice.json`` or a ``voice/<cha>.json``)."""
    if not path.is_file():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[hlist] voice read fail: {exc}")
        return {}
    if int(d.get("version") or 0) != VOICE_VERSION:
        return {}
    return d
