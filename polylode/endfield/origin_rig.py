"""A converted character's fitted source-game rig as a stored, certified table (PROGRESS #169).

``origin.converted_rig`` computes the rig; this module gathers its inputs out
of the Endfield store, keeps the result as the named table
``table/endfield/origin_rig/<ident>.json`` (no item or node of the store is
rewritten, so the body's own certification stays valid) and puts a loaded
completed body on it (:func:`rig_actor`).

The table is bound to everything it was made from (:func:`capture`): the
reference rig's cache, the character's ``chara``, ``body`` and ``body_source``
items and tables, the two H bakes the goal table is read off, and the card its interim
body sliders come from.  When any of them changed the table is **refused**
(:class:`~.actor.ActorError`): a stale rig is not played, and neither is the
native skeleton in its place — the table is rebuilt (``tools/build_origin_rig.py``).

Built for the completed body (``category == "body"``) only: the landmarks
are the lent body's kept joints, which no other category has.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..mesh.skeleton import Skeleton
from ..origin import converted_rig as CR
from . import store
from .actor import ActorError, build_skeleton

#: The named table the rig is kept under, beside ``chara`` / ``body`` / ``helper_drive``.
TABLE = "origin_rig"
#: The table's own layout (the rig's is ``converted_rig.PAYLOAD_VERSION``); another is not read.
VERSION = 1
#: The stored item the rig is fitted to: the completed body (its kept joints are the landmarks).
CATEGORY = "body"
#: The items an H scene plays a character as (``hrole.category``), a table is built for: its completed body,
#: or a reconstruction standing on the source joints its body fit kept (Si, §7.1088).
PLAYED = (CATEGORY, "reconstruction")
#: The 44 body sliders a card has (``ChaFileBody.shapeValueBody``).
SHAPE_COUNT = 44
#: Where the interim slider values come from until they are measured (design note §3, stage 5).
SHAPE_INTERIM = "source_card"


def _sha(path: Path) -> str:
    from .body_archive import file_digest
    return file_digest(path)


def _store_file(root: Path, key: str) -> str:
    """The store file of manifest *key*, relative to the store; ``FileNotFoundError`` when it has none."""
    entry = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["items"].get(key) or {}
    if not entry.get("file"):
        raise FileNotFoundError(f"origin rig input missing: {key}")
    path = (root / entry["file"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"origin rig input missing or outside the store: {entry['file']}")
    return entry["file"]


def source_card_path(ident: str) -> Path | None:
    """The card the lent body was built from (``body.bake_body``), where it is now; None without one."""
    from ..core.paths import stored_card
    return stored_card((store.read_extra("body", ident) or {}).get("source_card"))


def capture(ident: str, category: str = CATEGORY) -> dict:
    """What the table of *ident* is made from, hashed: ``{reference, store, files}``.

    *reference*: the reference rig's cache version and SHA-256; *store*: the
    store files of ``chara`` / ``body`` / ``body_source`` (the lent body's own
    bones) and the two items' tables; *files*: the two H
    bakes MotionIK's goal table is read off, and the lent body's source card
    (its sliders are the interim values of the sliders not measured); *shape*:
    what the measured sliders were read through (:func:`shape_program_digest`).
    A table for a reconstruction (*category*) is also made from that item.
    ``FileNotFoundError`` when one is gone.
    """
    from ..origin import hanim, rig
    root = store.home(ident).resolve()
    kept = {}
    keys = [f"chara/{ident}", f"body/{ident}", f"body_source/{ident}", f"table/{store.SOURCE}/chara/{ident}",
            f"table/{store.SOURCE}/body/{ident}"]
    if category != CATEGORY:
        keys.append(f"{category}/{ident}")
    for key in keys:
        relative = _store_file(root, key)
        kept[relative] = _sha(root / relative)
    files = {}
    for path in (hanim.FBBIK_FILE, hanim.MOTION_IK_FILE, source_card_path(ident)):
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f"origin rig input missing: {path}")
        files[Path(path).name] = _sha(Path(path))
    if not rig.CACHE.is_file():
        raise FileNotFoundError(f"origin rig input missing: {rig.CACHE}")
    return {"reference": {"cache_version": rig.CACHE_VERSION, "sha256": _sha(rig.CACHE)},
            "store": dict(sorted(kept.items())), "files": dict(sorted(files.items())),
            "shape": shape_program_digest()}


def shape_program_digest() -> dict:
    """SHA-256 of what the measured sliders were read through: the slider→curve table, the curves, the program.

    ``FileNotFoundError`` when the install's shape tables are not converted.
    """
    from ..origin.figure import shape_anim, shape_apply, shape_cat, shape_update
    rows, curves = shape_cat.body_rows(), shape_anim.body_curves()
    if not rows or curves is None:
        raise FileNotFoundError("origin rig input missing: the body shape tables (bake the install)")
    parts = {"rows": shape_cat.to_payload(rows), "curves": shape_anim.to_payload(curves),
             "program": repr(shape_update.BODY_UPDATE) + repr(shape_apply.BODY_ALWAYS)}
    return {key: hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            for key, value in parts.items()}


def _digest(table: dict) -> str:
    """SHA-256 of *table* without its ``table_sha256``: the stored JSON as written, key order fixed."""
    body = {key: value for key, value in table.items() if key != "table_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# -- inputs ------------------------------------------------------------------------------------------
@dataclass
class Inputs:
    """What :func:`converted_rig.build` is given for one character, out of the store."""
    ident: str
    reference: Skeleton
    source: Skeleton
    humanoid: dict
    fingers: dict
    landmarks: dict
    goal_joints: dict
    #: the lent body's source card (its sliders are the interim shape values), or None
    card: object = None
    card_path: Path | None = None
    #: ``(the reference rig's mouth bone, the character's lip nodes)``, or None without lips
    mouth: tuple | None = None
    notes: list = field(default_factory=list)


def source_skeleton(ident: str, category: str = CATEGORY) -> Skeleton:
    """The played item's skeleton at its rest: the character's nodes with the lent body's kept joints.

    The skeleton :func:`endfield.body.load_complete_actor` draws on, built
    from the same two items; for a reconstruction (*category*) the one
    ``reconstruction.load_reconstructed_actor`` draws on — its own nodes with
    the source joints its completed body's fit kept (``reconstruction.source_joints``).
    """
    from .body import append_body_nodes
    if category != CATEGORY:
        from . import reconstruction
        item = store.read_item(category, ident)
        if item is None or not item.nodes:
            raise ActorError(f"{category}/{ident} is not in the store")
        skeleton = build_skeleton(item.nodes, ident)
        joints = reconstruction.source_joints(ident, skeleton)
        if not joints:
            raise ActorError(f"{category}/{ident} does not stand on the source joints its body fit kept")
        return append_body_nodes(skeleton, joints)
    native = store.read_item("chara", ident)
    body = store.read_item(CATEGORY, ident)
    if native is None or not native.nodes:
        raise ActorError(f"chara/{ident} is not in the store")
    if body is None or not body.nodes:
        raise ActorError(f"{CATEGORY}/{ident} is not in the store: the rig is fitted to the completed body")
    return append_body_nodes(build_skeleton(native.nodes, ident), body.nodes)


def reference_skeleton(ident: str) -> Skeleton:
    """The rig a card wearing *ident*'s lent body has: the reference rig at scale 1 with that body's own bones.

    The lent body is a card's body (``body_source``), and a card wearing it
    carries its bones beside the reference rig's — the vagina chains
    BetterPenetration pushes and the clips move, by name — so the stand-in
    is built the same way (``hrole.stand_in_skeleton``).  Left out, the lent
    body's joints for them stayed kept nodes under another name and neither
    the clip nor the push reached them.
    """
    from . import hrole
    source = store.read_item("body_source", ident)
    if source is None:
        raise ActorError(f"body_source/{ident} is not in the store: the lent body's own bones are unknown")
    return hrole.stand_in_skeleton(getattr(source, "implant", None))


def gather(ident: str, category: str = CATEGORY) -> Inputs:
    """The inputs of *ident*'s rig: its skeleton and tables, the fit's finger pairs and landmarks, the goals.

    The finger pairs are the H path's own (``hrole.finger_pairs``); the
    landmarks are the lent body's kept joints (``body.GRAFT`` + a reference
    bone name) — the fit already placed and certified them; the goal table is
    read off MotionIK's limb rows (``converted_rig.goal_joints_from_ik``).
    """
    from ..chara.cardimport import read_origin_card
    from ..origin import hanim, motion_ik
    from . import hrole
    from .body import GRAFT
    reference = reference_skeleton(ident)
    source = source_skeleton(ident, category)
    humanoid = (store.read_extra("chara", ident) or {}).get("humanoid") or {}
    if CR.HIPS not in humanoid:
        raise ActorError(f"chara/{ident} has no humanoid table with Hips")
    fingers = {origin: node for node, origin in hrole.finger_pairs(source, humanoid, reference).items()}
    landmarks = {b.name[len(GRAFT):]: b.name for b in source.bones
                 if b.name.startswith(GRAFT) and reference.has(b.name[len(GRAFT):])}
    try:
        settings = motion_ik.load_settings(json.loads(hanim.FBBIK_FILE.read_text(encoding="utf-8")))
        files = json.loads(hanim.MOTION_IK_FILE.read_text(encoding="utf-8")).get("files") or {}
    except (OSError, ValueError) as exc:
        raise ActorError(f"MotionIK is not baked (tools/bake_hanim.py): {exc}") from exc
    goals = CR.goal_joints_from_ik(settings, files)
    card_path = source_card_path(ident)
    card = read_origin_card(card_path) if card_path is not None else None
    from .morph import LIPS
    # the oral rows aim at points on the reference head, read against her own lips (``converted_rig.build``)
    has_lips = reference.has(hrole.SOURCE_MOUTH) and all(source.has(n) for n in LIPS)
    mouth = (hrole.SOURCE_MOUTH, LIPS) if has_lips else None
    if mouth is None and reference.has(hrole.SOURCE_MOUTH):
        # a converted source's face with no lip joints: its lips measured off its own surface and keys
        from .shape_face import measured_lips
        native = store.read_item("chara", ident)
        lips = measured_lips(native.pieces, build_skeleton(native.nodes, ident)) if native is not None else None
        mouth = None if lips is None else (hrole.SOURCE_MOUTH, lips)
    return Inputs(ident, reference, source, humanoid, fingers, landmarks, goals, card, card_path, mouth)


# -- the table ---------------------------------------------------------------------------------------
#: How a slider's value was found (``shape_evidence["sliders"][k]["rule"]``).
SHAPE_MEASURED = "measured"


def _source_piece(ident: str):
    """``body_source/<ident>``'s ``o_body_a``: the source game's body the lent skin is."""
    from .hrole import SOURCE_BODY
    source = store.read_item("body_source", ident)
    piece = next((p for p in (source.pieces if source is not None else []) if p.mesh.name == SOURCE_BODY), None)
    if piece is None:
        raise ActorError(f"body_source/{ident} has no {SOURCE_BODY}: the source game's body is unknown")
    return piece


def measured_bodies(inputs: Inputs, rig: CR.ConvertedRig) -> tuple:
    """``(the body as drawn, the source game's body on the same skin points, evidence)`` for ``shape_measure``.

    The drawn body is the stored ``body`` item's skin (the one
    :func:`body.load_body_part` draws) at its rest: the points its faces
    still use — the composite cut away the skin under the character's own,
    which is not her body as seen — inside the range the source body's skin
    occupies (``source_piece_ranges``: the fit keeps its points in order, so
    point *i* of the range is the source body's point *i*).  Both bodies
    read the same points as owned by the same bones (the source body's), so
    a spread compares like with like; the frames are the rig's binds (the
    character's own rest), the placed bones its anchors and landmarks, the
    scale its ``s``.
    """
    from ..origin import shape_measure as SM
    from .body import GRAFT
    ident = inputs.ident
    item = store.read_item(CATEGORY, ident)
    if item is None or not item.pieces:
        raise ActorError(f"{CATEGORY}/{ident} is not in the store")
    piece, source = item.pieces[0], _source_piece(ident)
    ranges = [r for r in (store.read_extra(CATEGORY, ident) or {}).get("source_piece_ranges") or ()
              if r.get("source_mesh") == source.mesh.name]
    if len(ranges) != 1 or int(ranges[0]["vertices"]) != source.mesh.vertex_count:
        raise ActorError(f"{CATEGORY}/{ident}: the source body's range in the drawn skin is not recorded")
    start, count = int(ranges[0]["vertex_start"]), int(ranges[0]["vertices"])
    used = np.unique(np.asarray(piece.mesh.indices, dtype=np.int64))
    used = used[(used >= start) & (used < start + count)]
    rows = used - start
    owner = SM.heaviest(source.skin, list(source.bones))
    own = SM.heaviest(piece.skin, [n[len(GRAFT):] if n.startswith(GRAFT) else n for n in piece.bones])
    agree = int((own[used] == owner[rows]).sum())
    frames = {b.name: b.bind_world for b in rig.bones}
    parents = {b.name: b.parent for b in rig.bones}
    placed = {b.name for b in rig.bones if b.rule in ("anchor", "landmark")}
    drawn = SM.Body(np.asarray(piece.mesh.positions, dtype=np.float64)[used], owner[rows], frames, parents,
                    placed, float(rig.scale))
    evidence = {"points": int(len(rows)), "of": count, "owners_agree": agree}
    return drawn, SM.OriginBody(source, inputs.reference, rows), evidence


def shape_values(inputs: Inputs, rig: CR.ConvertedRig) -> tuple[list, dict]:
    """``(the 44 body sliders this body reports, how each was found)``.

    The height (0) is the slider at which the source game's body is this
    size (``hrole.height_slider``, unclamped — owner 2026-10-17).  Every other
    slider is **measured** on the drawn body (``origin.shape_measure``,
    design note §3): the value at which the source game's body, at the same
    size, has this body's spread and joint places at the bones the slider
    drives.  A slider none of whose channels can be read on this body, that
    the body does not tell from the others, or whose shape the source game's
    body does not take there is not measured (``shape_measure.measure``); it
    keeps the lent body's source card's value, marked interim
    (``rule: source_card``, with why), or None without a card.
    """
    from ..origin import shape_measure as SM
    from ..origin.figure import shape_cat, shape_update
    from . import edits, hrole
    height = hrole.height_slider(rig.scale)
    if height is None:
        raise ActorError(f"{inputs.ident}: the height slider's curve cannot be read")
    card_values = edits.body_values(inputs.card) if inputs.card is not None else None
    if card_values is not None and len(card_values) != SHAPE_COUNT:
        raise ActorError(f"{inputs.ident}: the source card has {len(card_values)} body sliders, not {SHAPE_COUNT}")
    channels = SM.slider_channels(shape_cat.body_rows(), shape_update.BODY_UPDATE)
    # measured from the middle of every slider (the editor's default), the height being the body's: the
    # lent body's source skin is bound where the reference rig shaped at 0.5 puts its joints (EndminF's
    # 134 bones: median 0.0, p90 4.8, max 7.5 mm), not where its source card's sliders do (4.3 / 8.5 / 30.3)
    start = [0.5] * SHAPE_COUNT
    start[0] = float(height)
    drawn, model, points = measured_bodies(inputs, rig)
    found, readings = SM.measure(drawn, model, start, channels)
    values: list = [None] * SHAPE_COUNT
    values[0] = float(height)
    sliders = {}
    for reading in readings:
        k = reading.slider
        row = {"read": [f"{c.bone}:{c.kind}:{'xyz'[c.axis]}" for c in reading.read],
               "unread": sorted({f"{c.bone}:{c.kind}:{'xyz'[c.axis]} ({why})" for c, why in reading.unread})}
        if reading.alone_mm is not None:
            row["alone_mm_per_unit"] = round(reading.alone_mm, 6)
        if reading.measured:
            values[k] = float(found[k])
            row.update(rule=SHAPE_MEASURED, residual_mm=round(reading.residual_mm, 6), at_limit=reading.at_limit)
        else:
            values[k] = None if card_values is None else float(card_values[k])
            row.update(rule=SHAPE_INTERIM if card_values is not None else "none", interim=True, why=reading.why)
        sliders[str(k)] = row
    evidence = {"height": {"index": 0, "rule": "hrole.height_slider(scale), unclamped", "scale": float(rig.scale)},
                "measure": {"rule": "origin.shape_measure: spread and placed-joint places against the source "
                                    "game's body under the shape program", "start": 0.5,
                            "limits": list(SM.SLIDER_LIMITS), "skin_points": points},
                "card": inputs.card_path.name if inputs.card_path is not None else None,
                "measured": sorted(int(k) for k, row in sliders.items() if row["rule"] == SHAPE_MEASURED),
                "interim": sorted(int(k) for k, row in sliders.items() if row.get("interim")),
                "sliders": sliders}
    return values, evidence


def _evidence(rig: CR.ConvertedRig, inputs: Inputs) -> dict:
    """Numbers that say the rig stands where the character does (the stage-2 acceptance, kept with the table)."""
    rest = inputs.source.evaluate({})
    by = {b.name: b for b in rig.bones}
    from . import hrole
    tied = {hrole.H_BONES[h]: n for h, n in inputs.humanoid.items() if h in hrole.H_BONES}
    tied.update(inputs.fingers)
    tied.update({b: n for b, n in inputs.landmarks.items() if b not in rig.shadowed})
    apart = max(float(np.linalg.norm(by[b].bind_world[:3, 3] - rest[inputs.source.index[n]][:3, 3]))
                for b, n in tied.items())
    sk = rig.skeleton(inputs.ident, kept=True)
    skin = sk.skinning_matrices(rig.bind_pose())
    offsets = sorted(((n, float(np.linalg.norm(v))) for n, v in rig.offsets.items()), key=lambda row: -row[1])
    mouth = None
    if inputs.mouth is not None:
        bone, lips = inputs.mouth
        measured = isinstance(lips, np.ndarray)
        middle = (lips.reshape(-1, 3).mean(axis=0) if measured
                  else np.mean([rest[inputs.source.index[n]][:3, 3] for n in lips], axis=0))
        mouth = {"bone": bone, "lips": lips.tolist() if measured else list(lips),
                 **({"lips_measured": "shape_face.measured_lips"} if measured else {}),
                 "to_lips_mm": round(float(np.linalg.norm(by[bone].bind_world[:3, 3] - middle)) * 1000.0, 4)}
    return {"rules": dict(sorted(rig.rules().items())), "mouth": mouth,
            "bones": len(rig.bones), "source_nodes": len(inputs.source.bones),
            "replaced": len(rig.replace), "reparented": len(rig.reparent), "kept": len(rig.kept),
            "shadowed_landmarks": len(rig.shadowed),
            "tied_max_mm": round(apart * 1000.0, 4),
            "bind_pose_skinning_max_dev": float(np.abs(skin - np.eye(4)).max()),
            "offsets_over_1mm": sum(1 for _n, v in offsets if v > 1e-3),
            "offsets_largest_mm": [[n, round(v * 1000.0, 2)] for n, v in offsets[:6]]}


def build_table(ident: str) -> dict:
    """*ident*'s rig as the stored table (not written): the rig, its shape values, evidence, certification.

    Built for the item an H scene plays the character as (``hrole.category``).
    """
    from . import hrole
    category = hrole.category(ident)
    if category not in PLAYED:
        raise ActorError(f"{ident} is played as {category}, which no origin rig is fitted to")
    inputs = gather(ident, category)
    rig = CR.build(inputs.reference, inputs.source, inputs.humanoid, hrole.H_BONES,
                   fingers=inputs.fingers, landmarks=inputs.landmarks, goal_joints=inputs.goal_joints,
                   mouth=inputs.mouth)
    values, shape_evidence = shape_values(inputs, rig)
    table = {"version": VERSION, "source": store.SOURCE, "ident": ident, "category": category,
             "rig": CR.to_payload(rig),
             "shape_values": values, "shape_evidence": shape_evidence,
             "evidence": _evidence(rig, inputs),
             "certification": capture(ident, category)}
    table["table_sha256"] = _digest(table)
    return table


def played() -> list[str]:
    """Every character an H scene plays on the rig: ``hrole.eligible``, played as an item a table is built for."""
    from . import hrole
    return [ident for ident, _female in hrole.eligible() if hrole.category(ident) in PLAYED]


def refresh(ident: str) -> tuple[bool, str]:
    """Build and keep *ident*'s table when an H scene plays it; ``(whether it stands, a line saying what was done)``.

    A table is made from the character's played item, its lent body and the
    bakes it reads (:func:`capture`): whatever changes one of them leaves it
    stale, and a stale table is refused at load.  So the tools that finish
    changing a character call this after they write (owner 2026-10-18, §7.1232:
    a newly converted character goes straight onto the rig).  A character no
    H scene plays has nothing to build and stands; one whose table cannot be
    built is said, with why, and does not.
    """
    if ident not in played():
        return True, f"{ident}: no origin rig to build (not played by an H scene)"
    try:
        table = build_table(ident)
        write(ident, table)
        require_current(ident, read(ident))
    except (ActorError, CR.ConvertedRigError, FileNotFoundError, ValueError, KeyError) as refusal:
        return False, f"{ident}: origin rig NOT built: {refusal}"
    return True, (f"{ident}: origin rig built for {table['category']} "
                  f"(scale {table['rig']['scale']:.4f}, {len(table['rig']['bones'])} bones)")


def write(ident: str, table: dict) -> None:
    """Keep *table* as ``table/endfield/origin_rig/<ident>.json`` (nothing else in the store changes)."""
    from ..origin import bake
    with store.opened(ident, write=True):
        bake.write_named_table(store.SOURCE, TABLE, ident, table)


def read(ident: str) -> dict | None:
    """The stored table of *ident*, or None when none was ever built."""
    from ..origin import bake
    with store.opened(ident):
        recorded, payload = bake.named_table(store.SOURCE, TABLE, ident)
    return payload if recorded and payload else None


def require_current(ident: str, table: dict) -> None:
    """Refuse a table of another version, one edited after it was written, or one whose inputs changed."""
    def stale(why: str) -> ActorError:
        return ActorError(f"{TABLE}/{ident}: {why}; rebuild the origin rig (tools/build_origin_rig.py {ident})")
    if not isinstance(table, dict) or table.get("version") != VERSION:
        raise stale(f"the table is not version {VERSION}")
    if table.get("ident") != ident or table.get("source") != store.SOURCE:
        raise stale("the table is another character's")
    if table.get("table_sha256") != _digest(table):
        raise stale("the table does not match its own digest")
    try:
        actual = capture(ident, table.get("category", CATEGORY))
    except (OSError, ValueError, KeyError) as exc:
        raise stale(f"its inputs cannot be checked ({exc})") from exc
    expected = table.get("certification") or {}
    for part in ("reference", "store", "files", "shape"):
        if actual.get(part) != expected.get(part):
            changed = sorted(key for key in set(actual.get(part) or {}) | set(expected.get(part) or {})
                             if (actual.get(part) or {}).get(key) != (expected.get(part) or {}).get(key))
            raise stale(f"its {part} inputs changed ({', '.join(changed)})")


# -- loading -------------------------------------------------------------------------------------------
@dataclass
class Rigged:
    """A loaded character on its fitted rig: the rebinding and what the table says of the body."""
    rebound: CR.Rebound
    table: dict
    #: the humanoid table as the source game's names (``hrole.H_BONES``) and the native one
    humanoid: dict
    native_humanoid: dict
    #: the item's component and node tables as loaded (the native skeleton's names)
    native_extra: dict

    @property
    def scale(self) -> float:
        return float(self.rebound.rig.scale)

    def shape_value(self, index: int) -> float | None:
        """Body slider *index* as the table holds it; None for one it does not have."""
        values = self.table.get("shape_values") or []
        if not 0 <= index < len(values) or values[index] is None:
            return None
        return float(values[index])

    def name(self, node: str) -> str:
        """The rig's name for the native node *node*."""
        return self.rebound.rig.replace.get(node, node)


def load(category: str, ident: str) -> dict | None:
    """*ident*'s current table when *category* is the one it is built for; None without one; ActorError when stale.

    A table is built for the item an H scene plays the character as
    (:data:`PLAYED`, ``hrole.category``); loading another of its items (its
    plain ``chara``, the completed body of a character played by its
    reconstruction) is not that item and stays on its own skeleton.
    """
    if category not in PLAYED:
        return None
    table = read(ident)
    if table is None or table.get("category", CATEGORY) != category:
        return None
    require_current(ident, table)
    return table


def _renamed(value, names: dict):
    """*value* (a component table) with every ``"node"`` naming a replaced node renamed."""
    if isinstance(value, dict):
        return {key: (names.get(item, item) if key == "node" and isinstance(item, str) else _renamed(item, names))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_renamed(item, names) for item in value]
    return value


def _meshes(actor) -> list:
    """Every mesh of *actor* that holds bone indices: the parts', and the visibility's own copies."""
    found: dict = {}
    for part in actor.model.parts:
        found[id(part.mesh)] = part.mesh
    visibility = getattr(actor, "composite_visibility", None)
    if visibility is not None:
        for mesh in [getattr(visibility, "source", None), getattr(getattr(visibility, "body", None), "mesh", None),
                     *(getattr(visibility, "_meshes", {}) or {}).values(),
                     *(part.mesh for part in getattr(visibility, "outfit", ()) or ())]:
            if mesh is not None:
                found[id(mesh)] = mesh
    return list(found.values())


def rig_actor(actor, category: str, ident: str) -> Rigged | None:
    """Put the loaded *actor* on its fitted rig when it has a current table; None (as loaded) without one.

    The skeleton becomes the rig (origin bones + the kept source nodes), every
    mesh is re-pointed at it (:meth:`converted_rig.Rebound.remap`), and the
    humanoid and component tables name the rig's bones.  ``ActorError`` when
    the table is stale or does not fit the skeleton — never the native
    skeleton in its place.
    """
    table = load(category, ident)
    if table is None:
        return None
    native = actor.skeleton
    try:
        rig = CR.from_payload(table["rig"], native)
        rebound = CR.Rebound(rig, native, ident)
    except (CR.ConvertedRigError, KeyError, ValueError) as exc:
        raise ActorError(f"{TABLE}/{ident}: {exc}; rebuild the origin rig (tools/build_origin_rig.py {ident})") from exc
    if getattr(actor, "pose_correctives", None):
        # fitted against the native skeleton's rest frames, which the rig's binds do not keep
        raise ActorError(f"{ident}: pose-corrected parts are not carried onto the origin rig")
    for constructor in getattr(actor, "surface_constructors", None) or ():
        # a surface program reads skinning by the native skeleton's bone indices: each tied rig bone skins
        # as its native node does (``Rebound``), so one that can say its indices again comes along
        remap = getattr(constructor, "remap_bones", None)
        if remap is None:
            raise ActorError(f"{ident}: {type(constructor).__name__} is not carried onto the origin rig")
        try:
            remap(rebound.index)
        except ValueError as exc:
            raise ActorError(f"{ident}: {exc}") from exc
    for mesh in _meshes(actor):
        rebound.remap(mesh)
    native_extra = actor.extra
    native_humanoid = dict(native_extra.get("humanoid") or {})
    names = rig.replace
    humanoid = {human: names.get(node, node) for human, node in native_humanoid.items()}
    actor.extra = {**native_extra, "humanoid": humanoid,
                   "components": _renamed(native_extra.get("components") or {}, names)}
    actor.skeleton = rebound.skeleton
    actor.model.skeleton = rebound.skeleton
    return Rigged(rebound, table, humanoid, native_humanoid, native_extra)
