"""Explicit, certified reconstruction states on the original game skeleton.

Reconstructed parts carry their own material and rig provenance. An opaque
NPC underlayer is a separate source from an adult Origin body, and this module
does not replace a missing card or a missing original actor.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..origin import bake, textures
from ..mesh import pose_corrective, skinning
from ..mesh.mesh import Material
from . import shading, store
from .actor import Actor, ActorError, _rest_mesh, _texture_key, build_skeleton
from .body_archive import file_digest
from .body_scope import require_body_scope
from .certification import _item_maps
from .model import folder_of

CATEGORY = "reconstruction"
#: An adult body of the source game; what an earlier version called it is read as it
ADULT, LEGACY_ADULT = "adult_origin", "adult_kk"
SOURCE_KINDS = frozenset({ADULT, LEGACY_ADULT, "opaque_npc", "original_undercoat"})


def canonical(kind):
    """*kind* as this version names it (:data:`LEGACY_ADULT` is :data:`ADULT`)."""
    return ADULT if kind == LEGACY_ADULT else kind


def validate_scope(extra: dict, manifest: dict) -> None:
    """Require authoritative model scope and an explicit reconstruction source."""
    kind = manifest.get("source_kind")
    if kind not in SOURCE_KINDS:
        raise ActorError("reconstruction has no supported explicit body source")
    if not folder_of(str(extra.get("model") or "")):
        raise ActorError("reconstruction original model declaration is missing")
    if canonical(kind) == ADULT:
        require_body_scope(extra, "reconstruction")
    elif manifest.get("opaque_source_verified") is not True:
        raise ActorError("opaque reconstruction source verification is missing")


def validate_states(manifest: dict, count: int, *, reviewed: bool = True) -> None:
    """Check complete reversible part states before changing model visibility.

    *reviewed* is whether the complete-body review is required as well; only
    an explicit inspection of a candidate may go without it.
    """
    if manifest.get("version") != 1:
        raise ActorError("unsupported reconstruction state version")
    states = manifest.get("states")
    if not isinstance(states, dict) or set(states) != {"on", "off"}:
        raise ActorError("reconstruction requires both outfit states")
    for key, selected in states.items():
        if (not isinstance(selected, list) or not selected
                or any(type(index) is not int or not 0 <= index < count for index in selected)
                or len(set(selected)) != len(selected)):
            raise ActorError(f"invalid reconstruction {key} part state")
    if set(states["on"]) | set(states["off"]) != set(range(count)):
        raise ActorError("reconstruction contains a part in neither state")
    if reviewed and manifest.get("complete_body_reviewed") is not True:
        raise ActorError("reconstruction complete body review is missing")
    for key, faces in (manifest.get("dressed_left_out") or {}).items():
        if (not key.isdigit() or int(key) not in states["on"] or int(key) in states["off"]
                or not isinstance(faces, list) or len(set(faces)) != len(faces)
                or any(type(face) is not int or face < 0 for face in faces)):
            raise ActorError(f"invalid reconstruction dressed-skin list for part {key}")


class ReconstructionVisibility:
    """Switch exact reviewed part sets while preserving unsupported-piece flags."""

    def __init__(self, parts: list, manifest: dict, *, reviewed: bool = True, skin=(), own=()) -> None:
        validate_states(manifest, len(parts), reviewed=reviewed)
        self._parts = parts
        #: the original pieces drawn as skin (by index): never something that is worn
        self._skin = frozenset(skin)
        self._drawable = [bool(part.visible) for part in parts]
        self._states = {key: frozenset(indices) for key, indices in manifest["states"].items()}
        # the character's own pieces the scene drew dressed only (Si's tail, ``part_roles.tail_parts``):
        # drawn undressed as well, and so not worn
        kept = frozenset(own) & self._states["on"]
        self._states["off"] = self._states["off"] | kept
        #: the original pieces come first; the reconstructed surfaces after them
        self._originals = int(manifest.get("original_pieces", len(parts)))
        #: the parts taken off one by one, by ``id`` (:meth:`set_hidden`)
        self.hidden: frozenset = frozenset()
        #: the undressed body's surfaces whole, and the faces of each to leave out (:meth:`set_cover`)
        self._whole = {index: part.mesh for index, part in enumerate(parts)
                       if index >= self._originals and index in self._states["off"]}
        self.cover: dict | None = None
        self._body = None
        # a dressed skin is only ever drawn under the whole outfit: without the faces of it
        # that come through that outfit (``reconstruction_compose.dressed_left_out``)
        for key, faces in (manifest.get("dressed_left_out") or {}).items():
            part = parts[int(key)]
            keep = np.ones(part.mesh.tri_count, dtype=bool)
            if faces and max(faces) >= len(keep):
                raise ActorError(f"reconstruction dressed-skin list does not fit part {key}")
            keep[faces] = False
            cut = part.mesh.copy()
            cut.indices = part.mesh.indices.reshape(-1, 3)[keep].ravel().copy()
            part.mesh = cut
        self.outfit_visible = True
        self.set_outfit_visible(True)

    def whole_body(self):
        """The undressed body's surfaces as one mesh at rest, with their bones and weights; made once.

        What is solved against a body (an arm kept out of it, colliders for
        what hangs from the head) asks for one mesh; a reconstruction's body
        is several surfaces of its scene, so they are put end to end here.
        None when the item has no surface of its own to undress to.
        """
        meshes = [mesh for _index, mesh in sorted(self._whole.items()) if mesh.vertex_count]
        if self._body is None and meshes:
            from ..mesh.mesh import Mesh
            width = max(mesh.bone_idx.shape[1] for mesh in meshes)
            starts = np.cumsum([0] + [mesh.vertex_count for mesh in meshes])
            body = Mesh.from_arrays(
                np.concatenate([mesh.positions for mesh in meshes]),
                np.concatenate([mesh.indices.astype(np.int64) + start for mesh, start in zip(meshes, starts)]),
                normals=np.concatenate([mesh.normals for mesh in meshes]), name="reconstructed body")
            body.bone_idx = np.concatenate([np.pad(mesh.bone_idx, ((0, 0), (0, width - mesh.bone_idx.shape[1])))
                                            for mesh in meshes])
            body.bone_wt = np.concatenate([np.pad(mesh.bone_wt, ((0, 0), (0, width - mesh.bone_wt.shape[1])))
                                           for mesh in meshes])
            self._body = body
        return self._body

    def worn_parts(self) -> list:
        """The original pieces that are the outfit: what undressing takes away, less the skin it replaces.

        The scene's two states say what is worn — the head, the hair, a tail
        or ears of the character's own are drawn in both.  (Listed as worn,
        they were pieces to take off beside the clothes, and skin under the
        hair was left out as if a garment covered it, §7.1009.)
        """
        return [part for index, part in enumerate(self._parts)
                if index < self._originals and index in self._states["on"]
                and index not in self._states["off"] and index not in self._skin]

    def body_surfaces(self) -> list:
        """``(part, whole mesh, faces never drawn)`` of the undressed body's surfaces."""
        return [(self._parts[index], mesh, np.zeros(mesh.tri_count, dtype=bool))
                for index, mesh in sorted(self._whole.items())]

    def set_cover(self, masks: dict | None) -> None:
        """Leave out the body faces *masks* names (``id(part) → (F,) bool``) while dressed.

        ``None`` goes back to the two reviewed states.  With a cover the
        dressed state draws the undressed body's surfaces under the original
        pieces, less the faces a piece that is still on covers
        (``endfield/pieces.py``).
        """
        self.cover = None if masks is None else {key: np.asarray(mask, dtype=bool)
                                                 for key, mask in masks.items()}
        self._apply()

    def set_outfit_visible(self, visible: bool) -> None:
        """Retain each exact part mesh, material, weights and source visibility."""
        self.outfit_visible = bool(visible)
        self._apply()

    def set_hidden(self, parts) -> None:
        """Take parts off one by one (their ``id``); the rest stay as the state has them."""
        self.hidden = frozenset(parts)
        self._apply()

    def drawable(self) -> dict:
        """``id(part) → whether the item draws it in some state``."""
        shown = self._states["on"] | self._states["off"]
        return {id(part): drawable and index in shown
                for index, (part, drawable) in enumerate(zip(self._parts, self._drawable))}

    def _apply(self) -> None:
        state = self._states["on" if self.outfit_visible else "off"]
        taken = {index for index, part in enumerate(self._parts)
                 if id(part) in self.hidden and index in state}
        covered = self.outfit_visible and self.cover is not None
        if self.outfit_visible and (taken or covered):
            # the dressed state's skin is only what the whole outfit leaves in
            # view: with a part of the outfit off, the undressed body is under
            # the original pieces that are still on.  Not under the game's own
            # skin that body replaces, though: the two are the same surface
            # drawn twice, and where they are drawn differently (a clean
            # undressed skin, §7.1007) the one shows through the other in shards
            replaced = self._skin - self._states["off"]
            state = self._states["off"] | {index for index in state
                                           if index < self._originals and index not in replaced}
        for index, mesh in self._whole.items():
            part = self._parts[index]
            mask = self.cover.get(id(part)) if covered else None
            if mask is None or not mask.any():
                part.mesh = mesh
                continue
            cut = mesh.copy()
            cut.indices = mesh.indices.reshape(-1, 3)[~mask].ravel().copy()
            part.mesh = cut
        for index, (part, drawable) in enumerate(zip(self._parts, self._drawable)):
            part.visible = drawable and index in state and id(part) not in self.hidden


def capture(ident: str, extra: dict) -> dict:
    """Hash the complete item, original rig evidence, maps and source/state table."""
    root = store.home(ident).resolve()
    with store.opened(ident):
        index = bake.manifest()["items"]
    item = index.get(f"{CATEGORY}/{ident}")
    if not item or not item.get("file"):
        raise ActorError(f"reconstruction/{ident}: source archive is missing")
    import numpy as np
    archive_path = (root / item["file"]).resolve()
    if not archive_path.is_relative_to(root) or not archive_path.is_file():
        raise ActorError("reconstruction source archive is outside the local store")
    with np.load(archive_path, allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["meta"]).decode("utf-8"))
    maps, missing = _item_maps(metadata, index)
    if missing:
        raise ActorError(f"reconstruction source maps missing: {sorted(missing)}")
    manifest = extra["reconstruction"]
    paths = {item["file"], *maps, *manifest.get("source_evidence_files", [])}
    files = {}
    for relative in sorted(paths):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ActorError(f"reconstruction input missing/outside store: {relative}")
        files[relative] = file_digest(path)
    payload = dict(extra)
    payload["reconstruction"] = {key: value for key, value in manifest.items()
                                if key != "runtime_certification"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"version": 1, "files": files, "manifest_sha256": hashlib.sha256(encoded).hexdigest()}


def require_current(ident: str, extra: dict) -> None:
    """Reject changed source policy, rig, geometry, maps or state membership."""
    certificate = extra.get("reconstruction", {}).get("runtime_certification")
    if not isinstance(certificate, dict) or certificate.get("version") != 1:
        raise ActorError(f"reconstruction/{ident}: runtime certification is missing")
    if certificate.get("visibility_and_pose_reviewed") is not True:
        raise ActorError(f"reconstruction/{ident}: runtime visibility/pose review is missing")
    current = capture(ident, extra)
    if any(certificate.get(key) != value for key, value in current.items()):
        raise ActorError(f"reconstruction/{ident}: stale certification")


def certify(ident: str, report: dict, viewed: str) -> dict:
    """Write the runtime certification of candidate *ident*; returns it.

    The certification is a statement by whoever calls this: that the
    candidate's outfit states and poses were looked at as this runtime draws
    them (*viewed* says what was looked at), on top of what was measured —
    *report* is the conversion's own (``tools/import_endfield_reconstruction.py
    --report``).  It is written only for a candidate whose scene was accepted
    whole, whose measurements passed, and whose report is of this very scene;
    it binds the archive, its maps and the manifest, so any later change to
    one of them makes it stale (:func:`require_current`).
    """
    extra = store.read_extra(CATEGORY, ident) or {}
    manifest = extra.get("reconstruction") or {}
    validate_scope(extra, manifest)
    if manifest.get("complete_body_reviewed") is not True:
        raise ActorError(f"reconstruction/{ident}: its scene was not accepted as a whole character")
    item = store.read_item(CATEGORY, ident)
    if item is None:
        raise ActorError(f"reconstruction/{ident}: not in the store")
    validate_states(manifest, len(item.pieces), reviewed=True)
    measured = report.get("verified") or {}
    scene = (report.get("composed") or {}).get("source") or {}
    if measured.get("passed") is not True:
        raise ActorError(f"reconstruction/{ident}: the conversion's measurements did not pass")
    if not scene.get("sha256") or scene.get("sha256") != (manifest.get("scene") or {}).get("sha256"):
        raise ActorError(f"reconstruction/{ident}: the report is of another scene than the candidate")
    if not viewed.strip():
        raise ActorError("say what was looked at: the outfit states and poses, as this runtime draws them")
    manifest.pop("runtime_certification", None)
    certificate = {**capture(ident, extra), "visibility_and_pose_reviewed": True,
                   "viewed": viewed.strip(),
                   "max_fitted_offset_m": measured.get("max_fitted_offset_m"),
                   "fitted_tolerance_m": measured.get("fitted_tolerance_m")}
    manifest["runtime_certification"] = certificate
    with store.opened(ident, write=True):
        bake.write_named_table(store.SOURCE, CATEGORY, ident, extra)
    require_current(ident, store.read_extra(CATEGORY, ident) or {})
    return certificate


def _local_material(piece, descriptor: dict) -> Material:
    kind = descriptor.get("kind")
    if kind not in ("local_albedo", "local_solid"):
        raise ActorError("unsupported explicit reconstruction surface")
    if "color" not in descriptor or len(descriptor["color"]) != 4:
        raise ActorError("reconstruction surface has no declared color")
    material = Material(name=descriptor["name"], base_color=tuple(descriptor["color"]))
    if 'cast_outline' in descriptor:
        if type(descriptor['cast_outline']) is not bool:
            raise ActorError('reconstruction outline declaration is invalid')
        material.cast_outline = descriptor['cast_outline']
    if kind == "local_albedo":
        if piece.texture is None or descriptor.get("texture_srgb") not in (True, False):
            raise ActorError("reconstruction surface albedo/source color space is missing")
        material.texture = textures.store(piece.texture, srgb=descriptor["texture_srgb"])
    normal = descriptor.get('normal')
    if normal is not None:
        entry = (piece.shader.get('maps') or {}).get(normal.get('map'))
        if (normal.get('packing') != 'RG_positive_Z' or not entry
                or entry.get('srgb') is not False or not np.isfinite(normal.get('strength', np.nan))):
            raise ActorError('reconstruction local normal declaration is unsupported')
        pixels = store.read_texture(entry['cab'], entry['name'])
        if pixels is None:
            raise ActorError('reconstruction local normal map is missing')
        if pixels.srgb:
            raise ActorError('reconstruction local normal map has the wrong colour space')
        material.normal_map = textures.store(pixels.px, srgb=False, repeat=True)
        material.normal_encoding = 2
        material.normal_scale = normal['strength']
    return material


def load_reconstructed_actor(ident: str) -> Actor:
    """Load a certified source choice; missing/stale acceptance fails before maps."""
    return _load_reconstructed_actor(ident, certified=True)


def source_joints(ident: str, skeleton) -> dict:
    """``{node: (parent, local 4×4)}``: the source rig's joints the character's completed body keeps, for *skeleton*.

    An H clip is written for the source rig; what carries it onto a
    character (``hrole.StandIn``) reads where the fit put that rig's joints
    on the character's bones — the finger a finger bone follows, the points
    a row aims at.  That fit is of the rig, not of a skin, and the completed
    body of the same character stores it (``body.helper_nodes``); a
    reconstruction stands on the same bones and takes it from there.  Empty
    when there is no completed body with its source body stored, or when one
    of the joints hangs from a bone this skeleton does not have.
    """
    from .body import GRAFT
    body = store.read_item("body", ident) if store.has("body_source", ident) else None
    joints = {name: node for name, node in (body.nodes if body is not None else {}).items()
              if name.startswith(GRAFT)}
    if any(not parent.startswith(GRAFT) and not skeleton.has(parent) for parent, _local in joints.values()):
        return {}
    return joints


def load_inspection_actor(ident: str) -> Actor:
    """Explicitly inspect a source candidate; never used by the Studio loader."""
    return _load_reconstructed_actor(ident, certified=False)


def _load_reconstructed_actor(ident: str, *, certified: bool) -> Actor:
    extra = store.read_extra(CATEGORY, ident) or {}
    manifest = extra.get("reconstruction") or {}
    validate_scope(extra, manifest)
    if certified:
        require_current(ident, extra)
    item = store.read_item(CATEGORY, ident)
    if item is None or not item.nodes:
        raise ActorError(f"reconstruction/{ident}: original runtime rig is missing")
    validate_states(manifest, len(item.pieces), reviewed=certified)
    from ..mesh.mesh import Model, Part
    skeleton = build_skeleton(item.nodes, ident)
    joints = source_joints(ident, skeleton)
    if joints:
        from .body import append_body_nodes
        # after the character's own bones: no piece's bone index moves
        append_body_nodes(skeleton, joints)
    model = Model(ident, skeleton)
    undrawn, partial = [], []
    for index, piece in enumerate(item.pieces):
        shader = piece.shader or {}
        descriptor = shader.get("reconstruction_surface")
        if descriptor is not None:
            material, note = _local_material(piece, descriptor), ""
        else:
            material, note = shading.material(shader, _texture_key)
        part = Part(f"{shader['_rendName']}#{piece.mat_slot}", _rest_mesh(piece, skeleton),
                    group="endfield", layer=index)
        if material is None:
            part.visible = False
            undrawn.append(f"{part.name}: {note}")
        else:
            part.material, part.layer = material, material.render_queue
            part.skinning = shader.get("_skinning", skinning.LINEAR)
            if (shader.get(shading.VERTEX_FIELD) or shader.get(shading.VERTEX_TINT)
                    or (descriptor or {}).get("vertex_tint")):
                if len(part.mesh.vcolor) != part.mesh.vertex_count:
                    raise ActorError(f"{part.name}: its colour field is not in the archive")
                # the program reads the field from the vertex colour
                part.mesh.colors = part.mesh.vcolor.copy()
            # a renderer the prefab ships switched off stays off in both states
            part.visible = not (shader.get("_offNode") or shader.get("_rendOff"))
            if note:
                partial.append(f"{part.name}: {note}")
        model.add(part)
    actor = Actor(model, skeleton, 0.0, undrawn, extra, partial)
    for part, piece in zip(model.parts, item.pieces):
        source_motion = None
        declared_motion = (piece.shader or {}).get('_source_motion')
        if declared_motion is not None:
            from .source_motion import SourceSurface, load_program
            try:
                source_motion = load_program(declared_motion, store.home(ident), skeleton.index)
                vertices = np.asarray(declared_motion['source_vertices'], np.int64)
                if (vertices.shape != (part.mesh.vertex_count,) or np.any(vertices < 0)
                        or np.any(vertices >= len(source_motion.data['rest']))):
                    raise ValueError('declared source surface vertex mapping is invalid')
            except (KeyError, ValueError) as error:
                raise ActorError(f'{part.name}: {error}') from error
            actor.surface_constructors.append(SourceSurface(part, source_motion, vertices))
        construction = (piece.shader or {}).get('_construction')
        if construction is not None:
            if construction.get('kind') != 'convex_hull' or type(construction.get('smooth')) is not bool:
                raise ActorError(f'{part.name}: unsupported source surface construction')
            from .convex_surface import ConvexSurface
            actor.surface_constructors.append(ConvexSurface(part, part.mesh, smooth=construction['smooth']))
        found = pose_corrective.from_shapes(piece.shapes, skeleton.index)
        if found is not None and source_motion is not None:
            raise ActorError(f'{part.name}: a declared source program cannot also use fitted pose corrections')
        if found is not None:
            actor.pose_correctives.append((part, found, part.mesh.positions.copy()))
        descriptor = (piece.shader or {}).get("_normal_transport")
        if descriptor is not None:
            from .surface_normals import SourceNormals, load_source
            try:
                data, corners, corrective = load_source(descriptor, store.home(ident), skeleton,
                                                        part.mesh.vertex_count)
            except (KeyError, ValueError) as error:
                raise ActorError(f"{part.name}: {error}") from error
            parents = np.array([bone.parent for bone in skeleton.bones])
            transport = SourceNormals(part, data, corners, descriptor["skinning"], corrective, parents)
            if source_motion is not None:
                transport.source_motion = source_motion
            actor.surface_constructors.append(transport)
    skin = [index for index, piece in enumerate(item.pieces)
            if shading.SE.FAMILIES.get((piece.shader or {}).get("_shader")) == shading.SE.FAMILY_SKIN]
    from .part_roles import tail_parts
    actor.composite_visibility = ReconstructionVisibility(
        model.parts, manifest, reviewed=certified, skin=skin,
        own=[index for index in tail_parts(model.parts, skeleton) if index < int(manifest.get("original_pieces", 0))])
    #: whether the source rig's joints stand on this skeleton (:func:`source_joints`): what an H scene needs
    actor.source_joints = bool(joints)
    lo, hi = model.bounds()
    actor.height = float(hi[1] - lo[1])
    return actor
