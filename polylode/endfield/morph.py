"""Endfield facial morphs: a character's face controls, and the poses that set them.

The game's faces are bone-driven.  Each character has a *config* (a
``SkeletalMorph`` asset pointing at its Avatar) that lists the face bones, a
base pose for them, and a few hundred named *controls*; a control holds, per
bone it moves, how far it takes that bone from the base at full weight.  An
*emotion* or a *pose* asset is a table of control names and weights.  Ears
have a config of the same shape.

What was read off the data (§7.570):

* a control's record is not decoded by the exporter; it gives the byte range.
  The bytes are a 20-byte header — the control's index, two hashes, a group
  flag, a bone count — and that many 44-byte bone entries (bone-name hash,
  bone id, position, rotation, scale), the same record the base pose uses.
  Every entry's hash is one of the config's own bones;
* a control that drives a material value instead has, after the same header,
  which component of the value it sets and the reference id of one of the
  config's shader parameters;
* positions are the engine's local positions.  **Rotations are Euler angles in
  degrees, turned about x, then y, then z, and they are already in this
  project's handedness** (the mirror of the engine's): read that way the base
  pose is the prefab's rest pose on most bones to a ten-thousandth of a
  degree, and under no other order or sign;
* an entry is a difference: weights times entries **add onto the base, the
  Euler angles as numbers**.  The clips that animate the face are those sums
  — fitted, they give weights like 0.48 on the same control left and right —
  and adding the angles leaves less unexplained than composing the turns
  either way round (``tools/ef_morph_check.py``).

The import half turns the exported documents into plain tables; the run-time
half (:class:`Face`) turns weights into a pose for the item's skeleton.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from ..core import math3d as m3
from ..core.paths import write_json_atomic
from . import store

#: Where the morph tables sit in the store.
FOLDER = "morph"
INDEX = "index.json"
POSES = "poses.json"
#: A config's ``dataType``.
FACE, EAR = 0, 1
KINDS = {FACE: "face", EAR: "ear"}
_HEAD = struct.Struct("<5i")          # index, two hashes, group, bone count
_ENTRY = struct.Struct("<2i9f")       # bone-name hash, bone id, position, rotation, scale
_DRIVEN = struct.Struct("<iq")        # after the header: component, parameter reference
_MAPPING = "SkeletalMorphMappingData"
_PARAM_MAPPING = "SkeletalMorphShaderPropMappingData"
#: The lists a pose document keeps its weights in.
_POSE_LISTS = ("_browValueL", "_browValueR", "_eyeValueL", "_eyeValueR", "_mouthValue",
               "_otherValue")


class MorphError(ValueError):
    """A morph document that is not laid out the way this module reads it."""


class Unread(MorphError):
    """A config that uses a record the exporter did not decode.

    A vehicle's screen "face" drives a four-component material value whose
    parameter record comes out of the exporter as bytes only.  Such a config
    is reported as what it is rather than as a failure.
    """


# -- import side ---------------------------------------------------------------
def _xyz(d: dict) -> list:
    return [float(d["x"]), float(d["y"]), float(d["z"])]


def _record(blob: bytes, index: int) -> tuple[int, list]:
    """One control's bytes → ``(group, [(bone hash, bone id, 9 floats)])``."""
    if len(blob) < _HEAD.size:
        raise MorphError(f"control {index}: {len(blob)} bytes, shorter than its header")
    own, _hash_a, _hash_b, group, count = _HEAD.unpack_from(blob)
    if own != index:
        raise MorphError(f"control {index}: its record says it is control {own}")
    if len(blob) != _HEAD.size + count * _ENTRY.size:
        raise MorphError(f"control {index}: {len(blob)} bytes for {count} bone entries")
    entries = []
    for k in range(count):
        row = _ENTRY.unpack_from(blob, _HEAD.size + k * _ENTRY.size)
        entries.append((row[0], row[1], list(row[2:])))
    return group, entries


def _driven(blob: bytes, index: int, name: str, refs: dict, params: dict) -> list:
    """A material-value control's bytes → ``[property, component]``."""
    if len(blob) != _HEAD.size + _DRIVEN.size:
        raise MorphError(f"control {name}: {len(blob)} bytes for a material-value record")
    own, _hash_a, _hash_b, _group, count = _HEAD.unpack_from(blob)
    if own != index or count:
        raise MorphError(f"control {name}: its record says control {own} with {count} bones")
    component, rid = _DRIVEN.unpack_from(blob, _HEAD.size)
    if rid not in refs:
        raise MorphError(f"control {name}: drives a parameter the config does not list")
    if rid not in params:
        raise Unread(f"control {name}: its {refs[rid]['type']['class']} parameter "
                     f"was not decoded by the exporter")
    return [params[rid], component]


def config(doc: dict, raw: bytes) -> dict:
    """An exported morph config and its raw bytes as a plain table.

    ``{"kind", "bones": [name…], "base": [[px, py, pz, rx, ry, rz, sx, sy, sz]…],
    "controls": {name: {"group", "bones": [[bone index, 9 differences]…]}},
    "params": {control name: [material property, component]}}``.

    Raises :class:`MorphError` when a record is not the layout described in
    this module, or names a bone the config does not list — never a guess.
    """
    data = doc["data"]
    names = [str(n) for n in data["allBoneNames"]]
    base = data["basePoseConfig"]
    if len(names) != len(base):
        raise MorphError(f"{len(names)} bone names for {len(base)} base entries")
    by_hash = {int(entry["boneNameHash"]): i for i, entry in enumerate(base)}
    refs = {int(ref["rid"]): ref for ref in doc["references"]["RefIds"]}
    params = {rid: str(ref["data"]["_paramName"]) for rid, ref in refs.items()
              if "_paramName" in ref["data"]}
    controls, driven = {}, {}
    control_names = data["morphMappingNames"]
    if len(control_names) != len(data["morphMappingCfgs"]):
        raise MorphError("control names and control records differ in number")
    for index, (name, cfg) in enumerate(zip(control_names, data["morphMappingCfgs"])):
        ref = refs.get(int(cfg["rid"]))
        if ref is None:
            raise MorphError(f"control {name}: its record is not in the document")
        blob = raw[int(ref["dataOffset"]):int(ref["dataOffset"]) + int(ref["dataLength"])]
        kind = ref["type"]["class"]
        if kind == _PARAM_MAPPING:
            driven[str(name)] = _driven(blob, index, str(name), refs, params)
            continue
        if kind != _MAPPING:
            raise MorphError(f"control {name}: a {kind} record, which was not read")
        group, entries = _record(blob, index)
        bones = []
        for bone_hash, _bone_id, values in entries:
            if bone_hash not in by_hash:
                raise MorphError(f"control {name}: moves a bone the config does not list")
            bones.append([by_hash[bone_hash], values])
        controls[str(name)] = {"group": group, "bones": bones}
    return {"kind": int(doc["dataType"]), "bones": names,
            "base": [_xyz(e["position"]) + _xyz(e["rotation"]) + _xyz(e["scale"]) for e in base],
            "controls": controls, "params": driven}


def pose(doc: dict) -> dict:
    """An exported pose or emotion as ``{"controls": {name: weight}, "additive": bool}``.

    A control listed twice keeps its last weight, as a dictionary would.
    """
    held = doc["_pose"]
    weights = {}
    for key in _POSE_LISTS:
        for entry in held.get(key) or ():
            weights[str(entry["_ctrlName"])] = float(entry["_value"])
    return {"controls": weights, "additive": bool(held.get("_bIsAdditivePose"))}


# -- the store -------------------------------------------------------------------
def folder() -> Path:
    return store.root() / FOLDER


def _read(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_config(name: str, model: str, table: dict) -> None:
    """Store config *name* and file it under the model asset its Avatar belongs to."""
    path = folder() / "config" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, table)
    listing = _read(folder() / INDEX, {})
    listing.setdefault(model, {})[KINDS.get(table["kind"], str(table["kind"]))] = name
    write_json_atomic(folder() / INDEX, listing, indent=1)


def write_poses(emotions: dict, poses: dict) -> None:
    """Store the shared tables: emotions and (mouth and other) poses, by asset name."""
    folder().mkdir(parents=True, exist_ok=True)
    write_json_atomic(folder() / POSES, {"emotions": emotions, "poses": poses}, indent=1)


def for_model(model: str) -> list[dict]:
    """The configs filed under model asset *model* (face first), ``[]`` when none."""
    if not model:
        return []
    listed = _read(folder() / INDEX, {}).get(model) or {}
    return [_read(folder() / "config" / f"{listed[kind]}.json", None)
            for kind in ("face", "ear") if kind in listed]


def for_actor(extra: dict) -> list[dict]:
    """A warehouse's registered face tables, else the original model's shared config.

    A transferred face is its own measured configuration: the template model's
    absolute positions are never applied directly to another character.
    """
    held = extra.get("face_morphs")
    if held:
        if not isinstance(held, list) or any(not isinstance(c, dict) for c in held):
            raise MorphError("registered face morphs must be a list of config tables")
        return held
    return for_model(extra.get("model") or "")


def emotions() -> dict:
    """``{asset name: {"controls": {…}, "additive": bool}}`` of the stored emotions."""
    return _read(folder() / POSES, {}).get("emotions") or {}


def poses() -> dict:
    """The stored pose assets (mouth shapes and the like), same shape as :func:`emotions`."""
    return _read(folder() / POSES, {}).get("poses") or {}


# -- run time ----------------------------------------------------------------------
def euler(degrees) -> np.ndarray:
    """A config's rotation as a quaternion in this project's space.

    Turned about x first, then y, then z, each about the fixed axes.
    """
    x, y, z = np.radians(np.asarray(degrees, dtype=np.float64))
    q = m3.quat_from_axis_angle((0.0, 0.0, 1.0), z)
    q = m3.quat_mul(q, m3.quat_from_axis_angle((0.0, 1.0, 0.0), y))
    return m3.quat_mul(q, m3.quat_from_axis_angle((1.0, 0.0, 0.0), x))


#: A config's position → this project's (the engine's X mirrored).
_MIRROR = np.array([-1.0, 1.0, 1.0])


#: The middle joints of the upper and the lower lip on the game's face rig (``hrole.StandIn.point``).
LIPS = ("lipMupJoint", "lipMdnJoint")


class Face:
    """A character's face (and ear) controls on its own skeleton.

    *configs* are the tables :func:`config` made; a bone a config lists that
    the skeleton does not have is an error, not a bone to skip — the config
    belongs to this model's Avatar.
    """

    def __init__(self, configs: list, skeleton):
        self.skeleton = skeleton
        self.bones: list = []
        base = []
        self.controls: dict = {}
        self.params: dict = {}
        for table in configs:
            offset = len(self.bones)
            for name in table["bones"]:
                if not skeleton.has(name):
                    raise MorphError(f"the skeleton has no bone {name} that its morph config moves")
                self.bones.append(name)
            base += table["base"]
            for name, control in table["controls"].items():
                held = self.controls.setdefault(name, [])
                held += [(offset + index, np.asarray(values, dtype=np.float64))
                         for index, values in control["bones"]]
            self.params.update(table["params"])
        self.base = np.asarray(base, dtype=np.float64).reshape(-1, 9)

    def known(self, name: str) -> bool:
        """Is *name* one of this face's controls (a bone one or a material one)?"""
        return name in self.controls or name in self.params

    def pose(self, weights: dict) -> dict:
        """Control weights → a pose for the face bones, as offsets from the rest.

        Every face bone is in it: the base pose is part of what a face is, and
        on a few bones it is not the prefab's rest.  A control this face does
        not have is left out (emotions are shared between characters whose
        faces differ; :meth:`missing` lists them).
        """
        values = self.base.copy()
        for name, weight in weights.items():
            for index, delta in self.controls.get(name, ()):
                values[index] += weight * delta
        out = {}
        for name, row in zip(self.bones, values):
            bone = self.skeleton.bones[self.skeleton.index[name]]
            out[name] = {
                "rot": m3.quat_mul(m3.quat_conj(bone.local_rot), euler(row[3:6])),
                "pos": row[0:3] * _MIRROR - bone.local_pos,
                "scale": row[6:9] / bone.local_scale}
        return out

    def missing(self, weights: dict) -> list:
        """The controls *weights* names that this face does not have."""
        return sorted(name for name in weights if not self.known(name))

    def material_values(self, weights: dict) -> dict:
        """Decoded scalar shader controls, separate from the bone pose.

        The imported mapping names the property and component directly;
        scalar parameters have component zero. Vector mappings are unread.
        """
        values = {}
        for name, weight in weights.items():
            if name not in self.params:
                continue
            prop, component = self.params[name]
            if component != 0:
                raise MorphError(f"{name}: non-scalar material component {component} was not read")
            value = float(weight)
            if not np.isfinite(value) or (prop in values and values[prop] != value):
                raise MorphError(f"{name}: nonfinite or conflicting material control")
            values[prop] = value
        return values
