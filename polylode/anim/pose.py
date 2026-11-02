"""Pose serialisation and the user pose library (UserData/studio/pose)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.paths import STUDIO_POSE, unique_path, write_json_atomic
from .clip import Pose
from .library import POSE_LABELS, POSES, get_pose, pose_names


def pose_to_dict(pose: Pose) -> dict:
    out = {}
    for bone, e in pose.items():
        d = {}
        if e.get("rot") is not None:
            d["r"] = [round(float(x), 6) for x in np.asarray(e["rot"]).ravel()[:4]]
        if e.get("pos") is not None:
            d["p"] = [round(float(x), 6) for x in np.asarray(e["pos"]).ravel()[:3]]
        s = e.get("scale")
        if s is not None:
            sv = np.asarray(np.broadcast_to(s, (3,)), dtype=float)
            d["s"] = [round(float(x), 6) for x in sv]
        if d:
            out[bone] = d
    return out


def pose_from_dict(d: dict) -> Pose:
    out: Pose = {}
    for bone, e in (d or {}).items():
        entry = {}
        if "r" in e:
            entry["rot"] = np.asarray(e["r"], dtype=np.float64)
        if "p" in e:
            entry["pos"] = np.asarray(e["p"], dtype=np.float64)
        if "s" in e:
            entry["scale"] = np.asarray(e["s"], dtype=np.float64)
        out[bone] = entry
    return out


def save_pose(pose: Pose, name: str, folder: Path | None = None,
              thumbnail=None, meta: dict | None = None) -> Path:
    """Write a pose as a PNG card (with embedded JSON) or plain JSON."""
    folder = folder or STUDIO_POSE
    payload = {"format": "polylode.pose", "version": 1, "name": name,
               "pose": pose_to_dict(pose)}
    if meta:
        payload["meta"] = meta
    if thumbnail is not None:
        from ..chara.card import write_card_png
        path = unique_path(folder, name, ".png")
        write_card_png(path, thumbnail, payload, marker=b"[IslandLifePose]")
        return path
    path = unique_path(folder, name, ".json")
    write_json_atomic(path, payload, indent=1)
    return path


def load_pose(path: Path) -> tuple[Pose, dict]:
    path = Path(path)
    if path.suffix.lower() == ".png":
        from ..chara.card import read_card_png
        payload = read_card_png(path, marker=b"[IslandLifePose]")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        return {}, {}
    return pose_from_dict(payload.get("pose", {})), payload


class PoseLibrary:
    """Built-in poses plus everything found in UserData/studio/pose."""

    def __init__(self, folder: Path | None = None):
        self.folder = Path(folder or STUDIO_POSE)
        self.user: dict[str, Path] = {}
        self.refresh()

    def refresh(self) -> None:
        self.user = {}
        if not self.folder.exists():
            return
        for p in sorted(self.folder.rglob("*")):
            if p.suffix.lower() in (".png", ".json") and p.is_file():
                rel = p.relative_to(self.folder)
                self.user[str(rel.with_suffix(""))] = p

    def names(self) -> list[str]:
        return pose_names() + sorted(self.user)

    def label(self, name: str) -> str:
        return POSE_LABELS.get(name, name)

    def categories(self) -> dict[str, list[str]]:
        cats: dict[str, list[str]] = {"內建": pose_names()}
        for name in sorted(self.user):
            folder = name.rsplit("/", 1)[0] if "/" in name else "自訂"
            cats.setdefault(folder, []).append(name)
        return cats

    def get(self, name: str) -> Pose:
        if name in POSES:
            return get_pose(name)
        p = self.user.get(name)
        if p is None:
            return {}
        pose, _ = load_pose(p)
        return pose

    def save(self, name: str, pose: Pose, thumbnail=None) -> Path:
        path = save_pose(pose, name, self.folder, thumbnail)
        self.refresh()
        return path

    def delete(self, name: str) -> bool:
        p = self.user.get(name)
        if p and p.exists():
            p.unlink()
            self.refresh()
            return True
        return False


def mirror_pose(pose: Pose) -> Pose:
    """Swap L/R bones and mirror their rotations across the YZ plane."""
    out: Pose = {}
    for bone, e in pose.items():
        if bone.endswith("_L"):
            target = bone[:-2] + "_R"
        elif bone.endswith("_R"):
            target = bone[:-2] + "_L"
        else:
            target = bone
        n = {}
        if e.get("rot") is not None:
            q = np.asarray(e["rot"], dtype=np.float64)
            # mirroring about X negates the y/z imaginary parts
            n["rot"] = np.array([q[0], -q[1], -q[2], q[3]])
        if e.get("pos") is not None:
            p = np.asarray(e["pos"], dtype=np.float64).copy()
            p[0] = -p[0]
            n["pos"] = p
        if e.get("scale") is not None:
            n["scale"] = np.asarray(e["scale"], dtype=np.float64)
        out[target] = n
    return out
