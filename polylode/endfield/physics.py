"""Endfield's swing chains and colliders for the FBX sidecar (``docs/sidecar_physics.md``, IslandLife §7.1512).

The prefab's ``BeyondBoneCloth`` components (MagicaCloth2's BoneCloth: the
``serializeData`` of its ``ClothSerializeData``) and its ``BeyondBone*Collider`` /
legacy ``Magica*Collider`` components are kept by :mod:`.components` with
every pointer turned into a node name.  The FBX holds the bones, not the
nodes: a component or a collider hanging on a node that is not a bone (the
``MC_*`` cloth nodes are children of the root) is said against the nearest
bone above it, or the model's root.

Geometry is computed where the game's numbers mean something — each node's
world matrix in this project's space (:func:`.rigscene.world_matrices`: Unity
with x mirrored, scale included) — and then said in the bone's FBX rest frame
(:func:`.rigscene.rest_matrix`: the scene's space, scale taken out), metres.
A collider is read as IslandLife's ``endfield.cloth.collider`` reads it.  The
painted points (``selectionData``) are matched to the chain's bones in the
same world, as ``endfield.cloth.Cloth._painted`` matches them.
"""
from __future__ import annotations

import copy

import numpy as np

from .rigscene import TO_Z_UP, world_matrices

SYSTEM = "magica_cloth2.bone_cloth"
CLOTH = "BeyondBoneCloth"
SPHERE, CAPSULE, PLANE = "BeyondBoneSphereCollider", "BeyondBoneCapsuleCollider", "BeyondBonePlaneCollider"
MAGICA_SPHERE, MAGICA_CAPSULE, MAGICA_PLANE = "MagicaSphereCollider", "MagicaCapsuleCollider", "MagicaPlaneCollider"
COLLIDERS = (SPHERE, CAPSULE, PLANE, MAGICA_SPHERE, MAGICA_CAPSULE, MAGICA_PLANE)
#: ``selectionData`` attribute values (MagicaCloth2's VertexAttribute) and their names in the sidecar
STATES = {0: "off", 1: "fixed", 2: "move"}
#: the serialized fields the chain's own entries state, left out of ``params``
RESOLVED_FIELDS = ("rootBones", "ignoreFromRootBones")
SPACE = ("geometry: the FBX bone's rest frame (as the FBX binds it, no scale), metres; world directions: the FBX's "
         "world, Y up, metres; params: the game's own units and conventions")
_MIRROR = np.array([-1.0, 1.0, 1.0])


class PhysicsError(ValueError):
    """The components name something the item does not hold."""


def _vec(d: dict) -> np.ndarray:
    """A serialized Unity vector in this project's handedness (x mirrored)."""
    return np.array([float(d["x"]), float(d["y"]), float(d["z"])]) * _MIRROR


def _point(world: np.ndarray, local) -> np.ndarray:
    return world[:3, :3] @ np.asarray(local, dtype=np.float64) + world[:3, 3]


def collider_shape(kind: str, record: dict) -> tuple[str, np.ndarray, np.ndarray, float, float] | None:
    """``(kind, start, end, radius start, radius end)`` in the node's own space (this project's handedness).

    The same reading as IslandLife ``endfield.cloth.collider``: a capsule's ``size`` is the start radius
    (x), the end radius (y, with ``radiusSeparation``) and the whole length caps included (z) along
    ``direction`` (``reverseDirection`` flips it), centred on ``center`` (``alignedOnCenter``) or starting
    there; a legacy capsule runs ``length`` along ``axis`` centred on ``center``; a plane keeps what it
    collides with on its node's +Y side.  None for a record that is switched off.
    """
    if not record.get("enabled", True):
        return None
    centre = _vec(record["center"]) if "center" in record else np.zeros(3)
    if kind == MAGICA_CAPSULE and "axis" in record:
        axis = np.zeros(3)
        axis[int(record["axis"])] = 1.0
        axis *= _MIRROR
        half = max(0.0, float(record["length"])) / 2.0
        return "capsule", centre - axis * half, centre + axis * half, float(record["startRadius"]), \
            float(record["endRadius"])
    if kind == MAGICA_SPHERE and "radius" in record:
        radius = float(record["radius"])
        return "sphere", centre, centre, radius, radius
    size = record.get("size") or {"x": 0.0, "y": 0.0, "z": 0.0}
    if kind in (SPHERE, MAGICA_SPHERE):
        return "sphere", centre, centre, float(size["x"]), float(size["x"])
    if kind in (PLANE, MAGICA_PLANE):
        return "plane", centre, np.array([0.0, 1.0, 0.0]), 0.0, 0.0
    axis = np.zeros(3)
    axis[int(record.get("direction", 0))] = -1.0 if record.get("reverseDirection") else 1.0
    axis = axis * _MIRROR
    first = float(size["x"])
    second = float(size["y"]) if record.get("radiusSeparation") else first
    reach = max(float(size["z"]) - first - second, 0.0)
    start = centre - axis * (reach / 2.0) if record.get("alignedOnCenter", 1) else centre + axis * first
    return "capsule", start, start + axis * reach, first, second


class _Rig:
    """The item's nodes against the FBX's bones: which bone stands for a node, and the frames."""

    def __init__(self, bones, nodes: dict, humanoid: dict):
        self.nodes = nodes
        self.world = world_matrices(nodes)
        self.bone_of = {bone.node: bone.name for bone in bones}
        self.rest = {bone.name: np.asarray(bone.rest, dtype=np.float64) for bone in bones}
        self.human_of = {node: human for human, node in (humanoid or {}).items()}

    def bone(self, node: str) -> str | None:
        """The FBX bone that is *node*, or the nearest above it; None for the root (the armature itself)."""
        if node not in self.nodes:
            raise PhysicsError(f"node {node!r} is not in the item")
        while node:
            if node in self.bone_of:
                return self.bone_of[node]
            node = self.nodes[node][0]
        return None

    def scene(self, node: str) -> np.ndarray:
        """*node*'s world matrix in the scene's space (Z up), its scale kept."""
        return TO_Z_UP @ self.world[node]

    def frame(self, bone: str | None) -> np.ndarray:
        """The rest frame geometry is said in: the bone's (no scale); for the root, the FBX's world (Y up: this
        project's space, which the scene's Z up is turned from)."""
        return self.rest[bone] if bone is not None else TO_Z_UP

    def humanoid(self, node: str) -> str | None:
        """The nearest human bone at or above *node* (Unity ``HumanBodyBones`` name)."""
        while node:
            if node in self.human_of:
                return self.human_of[node]
            node = self.nodes[node][0]
        return None

    def descendants(self, node: str) -> list[str]:
        children: dict[str, list] = {}
        for name, (parent, _local) in self.nodes.items():
            children.setdefault(parent, []).append(name)
        out, stack = [], [node]
        while stack:
            here = stack.pop()
            out.append(here)
            stack.extend(sorted(children.get(here, ()), reverse=True))
        return out


def _rename(value, rig: _Rig):
    """*value* with every ``{"node": name}`` pointing at the FBX bone that stands for it."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "node" and isinstance(item, str):
                out[key] = rig.bone(item) if item in rig.nodes else item
            else:
                out[key] = _rename(item, rig)
        return out
    if isinstance(value, list):
        return [_rename(item, rig) for item in value]
    return value


def _round(values) -> list:
    return [round(float(v), 7) for v in np.ravel(values)]


def colliders(components: dict, rig: _Rig, simulated: set[str]) -> tuple[list, list]:
    """``(rows, refused)``: every collider component as a sidecar row (``docs/sidecar_physics.md``)."""
    rows, refused = [], []
    for kind in COLLIDERS:
        for record in components.get(kind, ()):
            shape = collider_shape(kind, record)
            if shape is None:
                refused.append(f"{record.get('id')}: switched off")
                continue
            node = record["node"]
            bone = rig.bone(node)
            shape_kind, start, end, first, second = shape
            world = rig.scene(node)
            scale = float(np.linalg.norm(world[:3, 0]))
            into = np.linalg.inv(rig.frame(bone))
            a = _point(into @ world, start)
            if shape_kind == "plane":
                normal = into[:3, :3] @ (world[:3, :3] @ end)
                b = normal / np.linalg.norm(normal)
            else:
                b = _point(into @ world, end)
            rows.append({"id": str(record["id"]), "kind": shape_kind, "bone": bone, "start": _round(a),
                         "end": _round(b), "radius": [round(first * scale, 7), round(second * scale, 7)],
                         "inside": False,
                         "host": {"humanoid": rig.humanoid(node), "on_chain": node in simulated},
                         "source": {"class": kind, "node": node,
                                    "fields": {k: v for k, v in record.items() if k not in ("id", "node")}}})
    return rows, refused


def _states(selection: dict, rig: _Rig, chain: list[str]) -> list | None:
    """The painted attribute of each chain node with a painted point within reach (``Cloth._painted``)."""
    points = selection.get("positions") or []
    values = [int(a.get("Value", 2)) for a in selection.get("attributes") or []]
    if not points or len(values) != len(points):
        return None
    painted = np.array([_vec(p) for p in points])
    reach = float(selection.get("maxConnectionDistance") or 0.0)
    out = []
    for node in chain:
        gap = np.linalg.norm(painted - rig.world[node][:3, 3], axis=1)
        nearest = int(gap.argmin())
        if gap[nearest] <= reach and node in rig.bone_of:
            out.append({"node": rig.bone_of[node], "state": STATES.get(values[nearest], str(values[nearest]))})
    return out


def chains(components: dict, rig: _Rig) -> tuple[list, set[str]]:
    """``(rows, simulated nodes)``: every cloth component as a sidecar chain row."""
    rows, simulated = [], set()
    for record in components.get(CLOTH, ()):
        data = record.get("serializeData") or {}
        roots = [entry["node"] for entry in data.get("rootBones") or () if entry and entry.get("node")]
        ignored = set()
        for entry in data.get("ignoreFromRootBones") or ():
            if entry and entry.get("node") in rig.nodes:
                ignored.update(rig.descendants(entry["node"]))
        nodes = []
        for root in roots:
            if root not in rig.nodes:
                raise PhysicsError(f"cloth {record.get('id')}: root {root!r} is not in the item")
            nodes += [n for n in rig.descendants(root) if n not in nodes and n not in ignored]
        simulated.update(n for n in nodes if n not in roots)
        frame_node = record["node"]
        frame_bone = rig.bone(frame_node)
        local = np.linalg.inv(rig.frame(frame_bone)) @ rig.scene(frame_node)
        collision = data.get("colliderCollisionConstraint") or {}
        slots = collision.get("colliderList") or ()
        listed = [str(slot["component"]) for slot in slots if isinstance(slot, dict) and slot.get("component")]
        params = _rename({k: copy.deepcopy(v) for k, v in data.items() if k not in RESOLVED_FIELDS}, rig)
        params.get("colliderCollisionConstraint", {}).pop("colliderList", None)
        gravity = _vec(data.get("gravityDirection") or {"x": 0, "y": -1, "z": 0}) * float(data.get("gravity", 0.0))
        rows.append({
            "id": str(record["id"]), "system": SYSTEM, "enabled": bool(record.get("enabled", True)),
            "frame": {"bone": frame_bone, "node": frame_node, "local": [_round(row) for row in local]},
            "roots": [rig.bone(r) for r in roots],
            "exclude": sorted({rig.bone(e["node"]) for e in data.get("ignoreFromRootBones") or ()
                               if e and e.get("node") in rig.nodes} - {None}),
            "bones": [rig.bone_of[n] for n in nodes if n in rig.bone_of],
            "states": _states(record.get("selectionData") or {}, rig, nodes),
            "colliders": listed,
            "empty_slots": sum(slot is None for slot in slots),
            "params": params,
            # Unity's world (x mirrored) is this project's, which the FBX's world is (its Y up)
            "resolved": {"gravity_world": _round(gravity)}})
    return rows, simulated


def physics(scene, nodes: dict, extra: dict) -> dict:
    """The sidecar's ``physics`` section for an Endfield item (``docs/sidecar_physics.md``)."""
    components = extra.get("components") or {}
    basis = ("the prefab's BeyondBoneCloth and BeyondBone*/Magica* collider components "
             "(endfield.components.KEPT)")
    if not components.get(CLOTH):
        return {"format": 1, "status": "read", "system": SYSTEM, "basis": basis, "space": SPACE,
                "chains": [], "colliders": [], "unread": {},
                "reason": "the prefab holds no BeyondBoneCloth component"}
    rig = _Rig(scene.bones, nodes, extra.get("humanoid") or {})
    rows, simulated = chains(components, rig)
    found, refused = colliders(components, rig, simulated)
    known = {row["id"] for row in found}
    for row in rows:
        missing = [c for c in row["colliders"] if c not in known]
        if missing:
            row["colliders"] = [c for c in row["colliders"] if c in known]
            row["missing_colliders"] = missing
    return {"format": 1, "status": "read", "system": SYSTEM, "basis": basis, "space": SPACE,
            "chains": rows, "colliders": found, **({"refused": refused} if refused else {}),
            "unread": {"selfCollisionConstraint": "self collision: off (selfMode 0) on every prefab read (§7.1511)",
                       "springConstraint": "the documentation gives the spring block to another cloth type",
                       "cullingSettings": "culling and level of detail",
                       "clothSimulateWeight": "the game's own field (not in the public package); read as an output "
                                              "blend by IslandLife"}}


def add_to_document(document_path, scene, nodes: dict, extra: dict) -> None:
    """Write :func:`physics` into the interchange document at *document_path* (its ``physics`` key)."""
    import json
    from pathlib import Path
    path = Path(document_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["physics"] = physics(scene, nodes, extra)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
