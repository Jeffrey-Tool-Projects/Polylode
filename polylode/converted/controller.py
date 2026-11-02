"""Source-qualified ordinary pose controller and complete native draw inputs.

The controller never assigns a generic material. A native renderer must accept
every source material and scene resource before a frame can be placed on stage.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..anim.drive import DRIVE_POSE, PoseDrive
from ..core import math3d as m3
from ..mesh import skinning
from ..mesh.mesh import Mesh
from ..mesh.skeleton import BONE_LABELS
from ..origin.converted_rig import Rebound
from . import bindings, donor_skin, geometry, store, surfaces
from .streams import NativeMesh

if TYPE_CHECKING:
    from .scene import NativeItem


@dataclass
class DenseBody:
    """Opaque body geometry and its complete dense skin, independent of the eight-weight mesh format."""
    positions: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    bones: tuple[str, ...]
    corner_uvs: tuple[np.ndarray, ...] = ()


@dataclass
class NativeActor:
    """Actual source meshes, complete donor and opaque inner surface; no placeholder Model/material."""
    geometry: geometry.CharacterGeometry
    meshes: list[NativeMesh]
    donor: Mesh
    inner: DenseBody
    inner_material: dict
    material_owners: list[dict]
    rebound: Rebound

    def donor_material(self) -> donor_skin.StoredSkin:
        """Require this body's own complete clean source material; numeric pose loading needs no texture fallback."""
        return donor_skin.read(self.geometry.folder, self.geometry.metadata)


@dataclass(frozen=True)
class NativeFrame:
    """A native renderer's explicit actor/pose/transform inputs, retaining all body weights."""
    actor: NativeActor
    rig_world: np.ndarray
    native_world: np.ndarray
    native_skin: np.ndarray
    transform: np.ndarray

    def __post_init__(self) -> None:
        for name in ('rig_world', 'native_world', 'native_skin', 'transform'):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            expected = (4, 4) if name == 'transform' else (len(value), 4, 4) if value.ndim == 3 else None
            if expected is None or value.shape != expected or not np.isfinite(value).all():
                raise ValueError('actor frame requires complete finite 4x4 matrices')
            if name == 'transform':
                if not np.array_equal(value[3], [0, 0, 0, 1]) or np.linalg.slogdet(value[:3, :3])[0] == 0:
                    raise ValueError('actor transform must be nonsingular and affine')
            elif not len(value):
                raise ValueError('actor frame requires a complete nonempty skeleton')
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.native_world.shape != self.native_skin.shape:
            raise ValueError('actor source bone frames and skin palette differ')

    def donor_positions(self) -> np.ndarray:
        """Evaluate source-bound donor skin once, without blending the fitted rest shape twice."""
        mesh = self.actor.donor
        return skinning.skinned(mesh.skin_rest()[0], mesh.bone_idx, mesh.bone_wt, self.native_skin)

    def inner_positions(self) -> np.ndarray:
        """Evaluate every retained influence of the opaque body; no top-k reduction."""
        return self.actor.geometry.opaque_positions(self.native_world)

    def donor_normals(self) -> np.ndarray:
        """Carry unblended authored donor normals once through their actual skin frames, in actor space."""
        mesh = self.actor.donor
        positions, normals = mesh.skin_rest()
        _points, frames = skinning.linear_frames(positions, mesh.bone_idx, mesh.bone_wt, self.native_skin)
        result = skinning.posed_normals(frames, normals)
        if not np.isfinite(result).all() or np.any(np.linalg.norm(result, axis=1) == 0):
            raise store.WarehouseError("donor skin has degenerate posed normals")
        return result

    def inner_normals(self) -> np.ndarray:
        """Carry the inner surface's authored normals with every retained influence, in actor space."""
        inner = self.actor.inner
        return geometry.dense_normals(inner.normals, inner.weights,
                                      self.actor.geometry.opaque_matrices(self.native_world))

    def native_positions(self, index: int) -> np.ndarray:
        """Evaluate one original renderer with its own untouched full skin weights."""
        mesh = self.actor.meshes[index].mesh
        return skinning.skinned(mesh.positions, mesh.bone_idx, mesh.bone_wt, self.native_skin)


def _owners(model: geometry.CharacterGeometry) -> list[dict]:
    if model.metadata["source"] == "gf2":
        name = "native_material_owners.json"
        if name not in model.metadata["files"]:
            raise store.WarehouseError("original GF2 material/shader ownership has not been prepared")
        record = json.loads(store.child(model.folder, name).read_text(encoding="utf-8"))
        if (record.get("schema") != "gf2_native_material_owners_v1"
                or record["source"] != "gf2" or record["ident"] != model.metadata["ident"]):
            raise store.WarehouseError("native material ownership names another actor")
        rows = record["renderers"]
        parts = model.document["parts"]
        if (len(rows) != len(parts) or any(row["renderer"] != part["name"]
                or [slot["material"] for slot in row["slots"]] != part["materials"]
                for row, part in zip(rows, parts))):
            raise store.WarehouseError("native material ownership differs from renderer slots")
        return rows
    return [{"renderer": part["name"], "slots": [{
        "material": name, "source_package": model.document["materials"][name]["source_package"],
        "source_json_sha256": model.document["materials"][name]["source_json_sha256"]}
        for name in part["materials"]]} for part in model.document["parts"]]


class ActorController(PoseDrive):
    """Drive a consolidated character on its fitted original-game rig, independently of another game's controller."""

    card = None

    def __init__(self, source: str, ident: str, *, base: Path | None = None):
        self._load(store.locate(source, ident, base=base))

    @classmethod
    def from_folder(cls, folder: Path) -> ActorController:
        """Load a verified package directly for preparation and independent numeric readback."""
        controller = cls.__new__(cls)
        controller._load(Path(folder))
        return controller

    def _load(self, folder: Path) -> None:
        model = geometry.load(folder)
        document, arrays = surfaces.read(folder, model.metadata)
        recipe = bindings.read(folder, model.metadata)
        donor = bindings.donor_mesh(model, recipe)
        inner = DenseBody(model.body["opaque_positions"], model.body["opaque_faces"], arrays["normals"],
                          model.body["opaque_weights"], tuple(model.body["opaque_bone_names"].tolist()),
                          surfaces.corner_uvs(document, arrays, len(model.body["opaque_faces"])))
        if len(inner.positions) != document["vertices"] or inner.normals.shape != inner.positions.shape:
            raise store.WarehouseError("runtime surface normals do not match the full opaque body")
        rebound = model.rebound()
        self.actor = NativeActor(model, model.native_meshes(), donor, inner,
                                 document["material"], _owners(model), rebound)
        self.source, self.ident = model.metadata["source"], model.metadata["ident"]
        self.category, self.sex, self.params = "body", 0, {}
        self.name = model.metadata["character"]
        self.transform = m3.identity()
        self._visible = [not row.hidden for row in self.actor.meshes]
        super().__init__(rebound.skeleton)
        self._world = self.skeleton.evaluate({})
        self._native_world = rebound.native_world(self._world)
        self._native_skin = model.source_skeleton.skinning_matrices(world=self._native_world)
        self.set_source_bind()
        low, high = self.bounds()
        self._height = float(high[1] - low[1])
        if not math.isfinite(self._height) or self._height <= 0:
            raise store.WarehouseError('complete actor has no positive source-bind height')

    def set_source_bind(self) -> None:
        """Restore the exact delivered bind, distinct from the animation library's reference T-pose.

        ConvertedRig intentionally keeps reference rest and source bind apart.
        Explicit affine entries preserve native scales and shear; a rotation-only
        bind pose cannot reproduce all retained helper frames exactly.
        """
        self.drive = DRIVE_POSE
        self.pose_name = "source_bind"
        self.base_pose = self.actor.rebound.pose_from_native(self.actor.geometry.source_skeleton.evaluate({}))
        self.clear_manual()
        self.apply_rest = False
        self._blend_t = 1.0
        self.update(0)

    def set_pose(self, name: str, immediate: bool = False) -> None:
        """Choose an existing pose; first leave the affine source bind directly, then blend ordinary poses."""
        if name not in self.poses.names():
            raise KeyError(f"unknown actor pose: {name}")
        immediate = immediate or self.pose_name == "source_bind"
        super().set_pose(name, immediate)

    def set_clip(self, clip_name: str) -> None:
        """Choose an existing ordinary clip, rejecting missing names instead of silently ignoring them."""
        if clip_name not in self.library:
            raise KeyError(f"unknown actor clip: {clip_name}")
        self.apply_rest = True
        super().set_clip(clip_name)

    def use_state_machine(self) -> None:
        """Play ordinary locomotion on the fitted reference rest."""
        self.apply_rest = True
        super().use_state_machine()

    def use_motion_matching(self) -> None:
        """Match ordinary locomotion on the fitted reference rest, retaining character lengths."""
        self.apply_rest = True
        super().use_motion_matching()

    def display_name(self) -> str:
        """The verified folder/table character name, never a translated model-id guess."""
        return self.name

    def update(self, dt: float) -> None:
        """Advance ordinary poses/clips/manual FK on the fitted rig and retain source-space bone frames."""
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("actor timestep must be finite and nonnegative")
        if not self.animated:
            return
        pose = self.step(dt)
        # Absolute source-bind locals take precedence over channels in Skeleton.
        # Apply manual FK explicitly in each local's own orientation, retaining
        # its affine basis instead of silently ignoring the user's rotation.
        for bone, entry in self.manual_pose.items():
            if "matrix" not in pose[bone]:
                continue
            local = pose[bone]["matrix"].copy()
            orient = m3.quat_to_mat4(m3.decompose(local)[1])[:3, :3]
            turn = m3.quat_to_mat4(entry["rot"])[:3, :3]
            local[:3, :3] = orient @ turn @ orient.T @ local[:3, :3]
            pose[bone] = {"matrix": local}
        self._world = self.skeleton.evaluate(pose)
        self._native_world = self.actor.rebound.native_world(self._world)
        self._native_skin = self.actor.geometry.source_skeleton.skinning_matrices(world=self._native_world)

    def set_bone_rotation(self, bone: str, euler_deg) -> None:
        """Turn an actual fitted-rig or retained native bone; reject unmapped/unknown bone names."""
        if not self.skeleton.has(bone):
            raise KeyError(f"actor has no active bone: {bone}")
        angles = np.asarray(euler_deg, dtype=np.float64)
        if angles.shape != (3,) or not np.isfinite(angles).all():
            raise ValueError("actor bone rotation requires three finite angles")
        super().set_bone_rotation(bone, angles)

    @property
    def world(self) -> np.ndarray:
        """Current world frames on the original-game hierarchy, including retained nonhuman/IK bones."""
        return self._world

    def frame(self) -> NativeFrame:
        """Return complete source/pose inputs without rendering, scene defaults or acceptance changes."""
        return NativeFrame(self.actor, self._world, self._native_world, self._native_skin, self.transform)

    def native_item(self) -> NativeItem:
        """Place this controller on the native scene path, without constructing a generic material Model."""
        from .scene import NativeItem
        return NativeItem(self)

    def parts(self) -> list[tuple[int, str, bool]]:
        """The actor's own original renderer visibility, keeping donor and opaque inner body present."""
        return [(index, row.mesh.name, self._visible[index]) for index, row in enumerate(self.actor.meshes)]

    def set_part_visible(self, index: int, visible: bool) -> None:
        """Change only an explicitly selected native renderer; complete inner body is never removed."""
        if type(index) is not int or not 0 <= index < len(self._visible) or type(visible) is not bool:
            raise ValueError("invalid native renderer visibility")
        self._visible[index] = visible

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Current complete character bounds for camera framing, including both rebuilt surfaces."""
        frame = self.frame()
        positions = [frame.donor_positions(), frame.inner_positions()]
        positions.extend(frame.native_positions(i) for i, visible in enumerate(self._visible) if visible)
        low = np.min([points.min(axis=0) for points in positions], axis=0)
        high = np.max([points.max(axis=0) for points in positions], axis=0)
        return low, high

    @property
    def height(self) -> float:
        """Complete source-bind height for the shared camera API, independent of the current pose."""
        return self._height

    def stats(self) -> dict[str, int]:
        """Complete actor counts for scene status, including both rebuilt surfaces."""
        return {'parts': len(self.actor.meshes) + 2,
                'tris': sum(row.mesh.tri_count for row in self.actor.meshes)
                        + self.actor.donor.tri_count + len(self.actor.inner.faces),
                'bones': len(self.skeleton)}

    def fk_bones(self) -> list[tuple[str, str]]:
        """List every retained fitted/native bone by its actual active skeleton identity."""
        return [(BONE_LABELS.get(bone.name, bone.name), bone.name) for bone in self.skeleton.bones]
