"""A Wuthering Waves character's display poses, written into its assembled case as animations.

A character's role folder has ``CommonAnim/UiAnim``: the clips its character screens play (``Ui_Weapon_Start``,
``Ui_Weapon_Loop``, ``Ui_Weapon_End``, ``Team_Stand1``…).  Unreal's ``AnimSequence`` has no loop flag (the animation
blueprint decides), so what is read is the motion itself: **a held pose ends where it starts** — its last frame is
its first, within :data:`LOOP_CM` and :data:`LOOP_DEGREES` on every keyed bone — and a transition does not.  Every
clip is written, the transitions too (the owner wants every pose, 2026-11-04), the held ones first.  Only clips made
on the model's own skeleton are taken (``Stand1`` is made on another character's, and an additive clip is
a difference, not a pose).

The game compresses them with Unreal's ACL plugin, a codec CUE4Parse does not decode here (its ACL needs the native
library it builds with CMake).  So the reader hands over each sequence's bone stream as serialized (``wuwa.exe …
anim``: ``<k>.acl``, with which skeleton bone each track drives and the skeleton's ref pose), and it is decoded with
the ACL library the bundled AnimeStudio carries — the one Endfield's clips are decoded with (``endfield.acl``): the
buffer is ACL's ``compressed_tracks`` (tag ``0xAC11AC11``), every track a rotation, translation and scale in Unreal's
space (on Xin, a spine track's translation is its ref pose's, 5.62 cm).  A sequence CUE4Parse does decode comes
sampled instead (``<k>.bin``).  A frame is then made the way the case's rest is (``source_scene``): each bone's local transform —
the clip's for a bone it keys, the case's own native local otherwise — mirrored and scaled as
``source_scene.local_transform`` does, composed down the case's bones, and each world frame's uniform scale taken out
and its rotation made orthonormal (``bake_uniform``): unposed, that is ``bone_rest`` again.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .source_scene import C

#: how far a held pose's last frame may be from its first: centimetres, and degrees
LOOP_CM, LOOP_DEGREES = 0.5, 2.0
FOLDER = "CommonAnim/UiAnim/"


def ui_folder(package: str) -> str:
    """The display clips' folder of the model whose mesh is *package* (``…/<role>/<model>/Model/<mesh>``): the role
    folder's ``CommonAnim/UiAnim/``; empty when the path is not laid out so."""
    parts = package.replace("\\", "/").split("/")
    if "Model" not in parts or parts.index("Model") < 2:
        return ""
    return "/".join(parts[:parts.index("Model") - 1]) + "/" + FOLDER


def model_folder(package: str) -> str:
    parts = package.replace("\\", "/").split("/")
    return "/".join(parts[:parts.index("Model")]) + "/" if "Model" in parts else ""


def read(folder: Path, k: int, decoder=None) -> tuple[dict, np.ndarray]:
    """The reader's ``<k>.json`` and its frames ``(frames, bones, 10)``: quaternion xyzw, position, scale.

    A raw ACL buffer (``<k>.acl``) is decoded with *decoder* (an ``endfield.acl.Decoder``): each track lands on the
    skeleton bone it drives, every other bone holds the skeleton's ref pose; the record gets the buffer's own sample
    count and rate and which bones are keyed."""
    record = json.loads((folder / f"{k}.json").read_text(encoding="utf-8"))
    raw = folder / f"{k}.acl"
    if raw.is_file():
        from ..endfield import acl
        if decoder is None:
            raise ValueError(f"{record['name']}: an ACL buffer needs the ACL decoder")
        buffer = raw.read_bytes()
        _tracks, samples, rate = acl.header(buffer)
        decoded = decoder.decode(buffer, acl.TRANSFORM).astype(np.float64)
        bones = record["tracks"]
        if decoded.shape[1] != len(bones):
            raise ValueError(f"{record['name']}: {decoded.shape[1]} tracks for {len(bones)} mapped bones")
        values = np.repeat(np.asarray(record["reference"], dtype=np.float64)[None], samples, axis=0)
        values[:, bones] = decoded
        keyed = np.zeros(len(record["bones"]), dtype=bool)
        keyed[bones] = True
        record.update(frames=samples, rate=rate * float(record.get("rate_scale") or 1.0), keyed=keyed.tolist())
        return record, values
    data = np.fromfile(folder / f"{k}.bin", dtype="<f4")
    return record, data.reshape(int(record["frames"]), len(record["bones"]), 10).astype(np.float64)


def holds(record: dict, frames: np.ndarray) -> bool:
    """Does the clip end where it starts, on every bone it keys?"""
    keyed = np.asarray(record["keyed"], dtype=bool)
    if len(frames) < 2 or not keyed.any():
        return False
    first, last = frames[0, keyed], frames[-1, keyed]
    moved = np.linalg.norm(last[:, 4:7] - first[:, 4:7], axis=1).max()
    dot = np.abs(np.sum(last[:, :4] * first[:, :4], axis=1) / (np.linalg.norm(last[:, :4], axis=1)
                                                                * np.linalg.norm(first[:, :4], axis=1)))
    turned = np.degrees(2 * np.arccos(np.clip(dot, 0, 1))).max()
    return bool(moved <= LOOP_CM and turned <= LOOP_DEGREES)


def chosen(records: list[tuple[dict, np.ndarray]], package: str) -> list[tuple[dict, np.ndarray]]:
    """The poses among *records*, held ones first: on the model's own skeleton and not additive."""
    own = model_folder(package)
    found = [(r, f) for r, f in records
             if own and own.lower() in str(r.get("skeleton", "")).replace("\\", "/").lower()
             and not r.get("additive") and len(f)]
    return sorted(found, key=lambda item: not holds(*item))


def _native_local(transform: np.ndarray) -> np.ndarray:
    """An Unreal local (quaternion, position in cm, scale) in the case's space, as ``local_transform``."""
    out = np.eye(4)
    out[:3, :3] = C @ Rotation.from_quat(transform[:4]).as_matrix() @ np.diag(transform[7:10]) @ C.T
    out[:3, 3] = transform[4:7] @ C.T * .01
    return out


#: a clip's scale below this on any axis hides the bone (a transition does, ``Ui_Chip_End`` on Xin)
HIDDEN_SCALE = 1e-4


def _local(transform: np.ndarray, rest_local: np.ndarray) -> np.ndarray:
    """A clip's local for one bone, as :func:`_native_local`; a bone the clip scales to nothing keeps its rest scale.

    The exported animation carries translation and rotation only (``interchange.glb``), so a clip's scale never
    reaches the file; a scale of 0 would also leave its world frame no rotation to keep (:func:`_rigid`).  The
    bone takes the clip's rotation and position, at the scale it has at rest."""
    if np.abs(transform[7:10]).min() >= HIDDEN_SCALE:
        return _native_local(transform)
    out = _native_local(np.concatenate([transform[:7], [1.0, 1.0, 1.0]]))
    u, _s, vt = np.linalg.svd(rest_local[:3, :3])
    out[:3, :3] = out[:3, :3] @ (u @ vt).T @ rest_local[:3, :3]            # the rest's scale, its rotation taken out
    return out


def _rigid(world: np.ndarray) -> np.ndarray:
    """``bake_uniform``'s rest frame of one world matrix: its uniform scale out, its rotation orthonormal."""
    out = world.copy()
    out[:3, :3] /= np.cbrt(np.linalg.det(world[:3, :3]))
    u, _s, vt = np.linalg.svd(out[:3, :3])
    out[:3, :3] = u @ vt
    return out


def frames(record: dict, values: np.ndarray, bones: list[dict], native_rest: np.ndarray) -> np.ndarray:
    """The clip played on the case's bones: ``(frames, bones, 4, 4)`` in the space of ``bone_rest``."""
    by_name = {name: index for index, name in enumerate(record["bones"])}
    keyed = record["keyed"]
    native_rest = np.asarray(native_rest, dtype=np.float64)
    rest_local = [native_rest[i] if bone["parent"] < 0 else np.linalg.solve(native_rest[bone["parent"]], native_rest[i])
                  for i, bone in enumerate(bones)]
    driven = {i: by_name[bone["name"]] for i, bone in enumerate(bones)
              if bone["name"] in by_name and keyed[by_name[bone["name"]]]}
    out = np.empty((len(values), len(bones), 4, 4), dtype=np.float32)
    for frame in range(len(values)):
        world = []
        for i, bone in enumerate(bones):
            local = _local(values[frame, driven[i]], rest_local[i]) if i in driven else rest_local[i]
            world.append(world[bone["parent"]] @ local if bone["parent"] >= 0 else local)
        out[frame] = np.stack([_rigid(w) for w in world])
    return out


def add(case: Path, records: list[tuple[dict, np.ndarray]], package: str, log=print) -> int:
    """The poses among *records* (the reader's ``anim`` output for the model's ``UiAnim`` clips) added to the
    case's document as animations, as ``endfield.poses.add`` does; how many."""
    document_path = Path(case) / "source_document.json"
    document = json.loads(document_path.read_text(encoding="utf-8"))
    arrays_path = Path(case) / document["arrays"]
    arrays = dict(np.load(arrays_path))
    document["animations"] = []
    for k, (record, values) in enumerate(chosen(records, package)):
        log(f"寫入姿勢 {record['name']}（{len(values)} 格）")
        arrays[f"anim{k}"] = frames(record, values, document["bones"], arrays["bone_native_rest"])
        document["animations"].append({"name": record["name"], "clip": record["package"],
                                       "rate": float(record["rate"]), "frames": len(values)})
    if not document["animations"]:
        return 0
    np.savez_compressed(arrays_path, **arrays)
    document_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(document["animations"])
