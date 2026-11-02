"""What a prefab's components say beyond its renderers.

Read off the serialized objects and stored beside the item as plain JSON (the
item's *extra*): nothing here is interpreted, a run-time consumer gets the
game's own numbers with every pointer turned into a node name.

``humanoid``   the Avatar's human-bone table: Unity's bone name → node.  This is
               the game's own statement of which transform is the hips, the
               left upper arm and so on, so nothing has to infer it from how a
               rig happens to name its bones.
``human``      the Avatar's whole human definition as serialized (axes, limits,
               reference pose, masses) with ``nodes`` naming the node of each
               human-skeleton entry — what a muscle curve is defined against.
``model``      the asset path of the model file the Avatar belongs to
               (``…/actor/<body type>/<name>/models/…``): where the character's
               own clips are filed, and its body type.
``components`` the MonoBehaviours of the classes in :data:`KEPT`, by class:
               the cloth simulations and their colliders.

A component is identified by its **script** (the ``MonoScript`` its
``m_Script`` points at), not by the shape of its fields.
"""
from __future__ import annotations

from . import prefab as P
from .cabmap import CabIndex, Ref
from .staged import Staged

#: ``mecanim::human::BoneType`` — the order of ``m_HumanBoneIndex``.
HUMAN_BONES = (
    "Hips", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg",
    "LeftFoot", "RightFoot", "Spine", "Chest", "UpperChest", "Neck", "Head",
    "LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
    "LeftLowerArm", "RightLowerArm", "LeftHand", "RightHand", "LeftToes",
    "RightToes", "LeftEye", "RightEye", "Jaw")

#: Script classes whose components are kept, and the fields kept from each
#: (``None`` = every serialized field).  ``BeyondBoneCloth`` carries megabytes
#: of pre-built solver state in ``serializeData2.preBuildData``; what defines
#: the simulation is ``serializeData`` and the painted ``selectionData``.
KEPT: dict[str, tuple | None] = {
    "BeyondBoneCloth": ("serializeData", "selectionData"),
    "BeyondBoneCapsuleCollider": None,
    "BeyondBoneSphereCollider": None,
    "BeyondBonePlaneCollider": None,
    # Legacy Magica v1 colliders have axis/length/startRadius/endRadius,
    # rather than the newer direction/size schema. Keep their fields intact.
    "MagicaCapsuleCollider": None,
    "MagicaSphereCollider": None,
    "MagicaPlaneCollider": None,
    # the UI-model prefab's own statement of which character it is
    "CharUIModelMono": ("_charId",),
}
_OWN = frozenset({"m_GameObject", "m_Enabled", "m_Script", "m_Name"})


class ComponentError(ValueError):
    """A staged component that does not say what the converter needs."""


def humanoid(avatar: dict, prefab: P.Prefab, names: dict[int, str]) -> dict:
    """Human bone name → node name, for the bones the Avatar maps.

    ``m_HumanBoneIndex[bone]`` indexes the human skeleton, whose ``m_ID`` is a
    path hash the Avatar's ``m_TOS`` turns back into a transform path.  A bone
    whose path is not a node of this prefab is left out.
    """
    human = avatar["m_Avatar"]["m_Human"]
    ids = human["m_Skeleton"]["m_ID"]
    paths = avatar["m_TOS"]
    out = {}
    for bone, index in zip(HUMAN_BONES, human["m_HumanBoneIndex"]):
        if not 0 <= int(index) < len(ids):
            continue
        path = paths.get(str(ids[int(index)]))
        tid = None if path is None else P.find_path(prefab, path)
        if tid is not None:
            out[bone] = names[tid]
    return out


def human_definition(avatar: dict, prefab: P.Prefab, names: dict[int, str]) -> dict:
    """``m_Avatar.m_Human`` as serialized, plus the node of each skeleton entry.

    ``nodes[i]`` is the node at human-skeleton index *i* (``None`` when its
    path is not in this prefab).  Kept whole and uninterpreted: it is the
    input of the muscle arithmetic, not a result of it.
    """
    human = avatar["m_Avatar"]["m_Human"]
    paths = avatar["m_TOS"]
    nodes = []
    for value in human["m_Skeleton"]["m_ID"]:
        path = paths.get(str(value))
        tid = None if path is None else P.find_path(prefab, path)
        nodes.append(None if tid is None else names[tid])
    return {"definition": human, "nodes": nodes}


class _Resolver:
    """Turns the pointers inside a component into something a reader can use."""

    def __init__(self, prefab: P.Prefab, names: dict[int, str], cabs: CabIndex):
        self.cab = prefab.cab
        self.cabs = cabs
        self.names = names
        self.owner = {comp: names[tid] for tid, node in prefab.nodes.items()
                      for comp in node.components}

    def pointer(self, pptr: dict):
        ref = self.cabs.pointer(self.cab, pptr)
        if ref is None:
            return None
        if ref.cab == self.cab and ref.path_id in self.names:
            return {"node": self.names[ref.path_id]}
        if ref.cab == self.cab and ref.path_id in self.owner:
            return {"component": str(ref.path_id), "node": self.owner[ref.path_id]}
        return {"external": [ref.cab, str(ref.path_id)]}

    def value(self, v):
        if isinstance(v, dict):
            if set(v) >= {"m_FileID", "m_PathID"} and len(v) <= 4:
                return self.pointer(v)
            return {k: self.value(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self.value(x) for x in v]
        return v


def _fields(doc: dict, keep: tuple | None) -> dict:
    """The serialized fields of *doc* named by *keep*, searched one level down.

    ``selectionData`` sits inside ``serializeData2``; naming the field rather
    than its container is what lets the pre-built block beside it be left out.
    """
    own = {k: v for k, v in doc.items() if k not in _OWN and not k.startswith("$")}
    if keep is None:
        return own
    out = {}
    for name in keep:
        if name in own:
            out[name] = own[name]
            continue
        holders = [v for v in own.values() if isinstance(v, dict) and name in v]
        if len(holders) != 1:
            raise ComponentError(f"field {name!r} found {len(holders)} times")
        out[name] = holders[0][name]
    return out


def script_class(staged: Staged, cabs: CabIndex, cab: str, doc: dict) -> str:
    """The class name of a staged MonoBehaviour's script (``""`` if unstaged)."""
    ref = cabs.pointer(cab, doc.get("m_Script"))
    if ref is None or not staged.has(ref):
        return ""
    return str(staged.json(ref).get("m_ClassName") or "")


def kept_components(staged: Staged, cabs: CabIndex, prefab: P.Prefab,
                    names: dict[int, str]) -> tuple[dict, dict]:
    """``(components by class, count of every class seen)`` for *prefab*.

    Each kept component is ``{"id", "node", "enabled", …its fields…}`` with
    pointers resolved: ``{"node": name}`` for a transform or GameObject of the
    prefab, ``{"component": id, "node": name}`` for another component of it,
    ``None`` for a null pointer.
    """
    resolver = _Resolver(prefab, names, cabs)
    kept: dict[str, list] = {}
    seen: dict[str, int] = {}
    for ref in staged.of_type("MonoBehaviour", prefab.cab):
        doc = staged.json(ref)
        name = script_class(staged, cabs, prefab.cab, doc)
        seen[name] = seen.get(name, 0) + 1
        if name not in KEPT:
            continue
        node = resolver.owner.get(ref.path_id)
        if node is None:
            raise ComponentError(f"{name} {ref.path_id} is on no node")
        entry = {"id": str(ref.path_id), "node": node,
                 "enabled": bool(doc.get("m_Enabled", 1))}
        entry.update(resolver.value(_fields(doc, KEPT[name])))
        kept.setdefault(name, []).append(entry)
    return kept, seen


def animator_avatar(staged: Staged, cabs: CabIndex, prefab: P.Prefab) -> Ref | None:
    """The Avatar the prefab's root Animator names, or ``None``."""
    root = prefab.nodes[prefab.root]
    for ref in staged.of_type("Animator", prefab.cab):
        if ref.path_id in root.components:
            return cabs.pointer(prefab.cab, staged.json(ref).get("m_Avatar"))
    return None
