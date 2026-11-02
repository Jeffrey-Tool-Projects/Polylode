"""Classify connected mesh islands from authored shader and skin bindings."""
from __future__ import annotations

import numpy as np

from ..mesh.skin import split_archive_skin


#: A converted source without this game's shaders declares each renderer's role in its conversion
#: request (``converted.runtime_items``, validated against the file): the key its pieces' shader
#: records carry it under, and the roles.  Endfield's pieces have no such key; theirs come from
#: their shaders and bindings (:func:`classify_faces`).
DECLARED_ROLE = "_role"
HEAD_SKIN_CONTINUATION = "_head_skin_continuation"
ROLES = ("native_skin", "face", "head", "tail", "outfit")
#: How a declared role classifies a renderer's faces, in :func:`classify_faces`' roles: the face and
#: what the head keeps are the head; a tail is the character's own but not skin (:func:`declared_tails`).
DECLARED_CLASS = {"native_skin": "native_skin", "face": "head", "head": "head", "tail": "outfit",
                  "outfit": "outfit"}


def declared_role(piece) -> str | None:
    """The role *piece*'s conversion declared, ``None`` for a piece classified by its shader."""
    return (getattr(piece, "shader", None) or {}).get(DECLARED_ROLE)


def connected_faces(triangles: np.ndarray) -> list[np.ndarray]:
    """Faces sharing a vertex form one island; preserve original face indices."""
    triangles = np.asarray(triangles).reshape(-1, 3)
    parent = list(range(len(triangles)))
    vertex_face = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for face, triangle in enumerate(triangles):
        for vertex in triangle:
            old = vertex_face.setdefault(int(vertex), face)
            a, b = find(face), find(old)
            parent[max(a, b)] = min(a, b)
    groups = {}
    for i in range(len(triangles)):
        groups.setdefault(find(i), []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


#: How far a texel of a skin part may be from the part's skin colour (as a fraction of that colour's
#: length, linear) and still be skin.  Over the four composed bodies every skin face samples within
#: 0.40 of it and the one painted garment (shorts drawn into a skin texture) from 0.60 up; nothing
#: falls between (§7.939).
PAINTED_FROM_SKIN = 0.5


def painted_garment_faces(positions, triangles, uvs, faces, colour_at, skin_reference=None) -> np.ndarray:
    """The faces among *faces* of a skin part that are a garment painted into its texture.

    A skin part is drawn with one texture, and a garment can be painted
    straight into it: the faces are skin by their shader and clothing by what
    they show.  A face is painted when a corner or its centre samples
    (``colour_at(uvs) -> linear RGB``) farther than :data:`PAINTED_FROM_SKIN`
    from the part's own skin colour — the median over the faces' centres.
    When that median is itself farther than :data:`PAINTED_FROM_SKIN` from
    *skin_reference* (the character's facial skin, linear RGB), most of the
    part is painted and its median is the garment's: the reference is the
    skin colour then (Si: her gloves' faces outnumber her skin's, and her
    skin read as the garment).  A connected painted region is a garment only
    when it wraps around the body: it has more than one boundary loop, or one
    rim and it closes over the end of a limb (a glove, a sock:
    :func:`_caps_over`).  Any other region with a single rim is a mark on
    the skin (a mole, a tattoo, a gem) and stays skin.
    """
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64)
    if not len(faces):
        return faces
    corner = np.asarray(uvs, dtype=np.float64)[triangles[faces]]
    samples = np.concatenate([corner, corner.mean(axis=1, keepdims=True)], axis=1)
    colour = np.asarray(colour_at(samples.reshape(-1, 2)), dtype=np.float64).reshape(len(faces), 4, 3)
    skin = np.median(colour[:, 3], axis=0)
    if skin_reference is not None:
        reference = np.asarray(skin_reference, dtype=np.float64)
        if np.linalg.norm(skin - reference) > PAINTED_FROM_SKIN * max(float(np.linalg.norm(reference)), 1e-9):
            skin = reference
    away = np.linalg.norm(colour - skin, axis=2) / max(float(np.linalg.norm(skin)), 1e-9)
    painted = np.flatnonzero((away > PAINTED_FROM_SKIN).any(axis=1))
    if not len(painted):
        return painted
    points = np.asarray(positions, dtype=np.float64)
    _, first, welded = np.unique(np.round(points / 1e-6).astype(np.int64), axis=0,
                                 return_index=True, return_inverse=True)
    welded, points = welded.reshape(-1), points[first]
    garment = []
    for component in connected_faces(welded[triangles[faces[painted]]]):
        corners = welded[triangles[faces[painted[component]]]]
        edges = np.sort(np.concatenate([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]]), axis=1)
        unique, count = np.unique(edges, axis=0, return_counts=True)
        rim = unique[count == 1]
        # the rim's edges as degenerate triangles: its loops are their connected components
        loops = len(connected_faces(np.column_stack([rim, rim[:, :1]]))) if len(rim) else 0
        if loops > 1 or (loops == 1 and _caps_over(points, corners, rim)):
            garment.extend(faces[painted[component]].tolist())
    return np.asarray(sorted(garment), dtype=np.int64)


#: How much more surface than its rim spans a painted region with one rim covers when it closes over
#: the end of a limb: a hemisphere covers twice the disc of its rim.  A mark lies on the skin like a
#: patch (Zhuangfy's two, Typhoea's one: 1.00-1.02); Si's gloves cover 14.3 times their wrist's disc.
CAPS_OVER = 2.0


def _caps_over(points: np.ndarray, corners: np.ndarray, rim: np.ndarray) -> bool:
    """Whether the region of *corners* (welded triangles) with the one rim *rim* closes over an end.

    A glove or a sock painted into the skin has one rim, as a mark does, but
    it wraps the limb's end: its area is more than :data:`CAPS_OVER` times
    the area its rim spans (the vector area of the rim's edges, directed as
    the region's triangles run).
    """
    tri = points[corners]
    area = 0.5 * float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
    directed = np.concatenate([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]])
    on_rim = {tuple(edge) for edge in rim.tolist()}
    keep = np.array([tuple(sorted(edge)) in on_rim for edge in directed.tolist()], dtype=bool)
    span = 0.5 * float(np.linalg.norm(np.cross(points[directed[keep, 0]], points[directed[keep, 1]]).sum(axis=0)))
    return area > CAPS_OVER * span


def classify_faces(triangles, shaders: list[dict], active_bones: list[set[str]],
                   head_bones: set[str], *, positions=None, facial_context=None) -> list[dict]:
    """Pure source-face classification shared by runtime and Blender.

    Authored hair, eye and overlay shaders mark retained original surfaces.
    Skin faces bound wholly to the Head subtree are retained with that head;
    they need not use the facial lighting variant of the skin shader.
    Their neck, cloth-helper or tail bindings must not turn those surfaces into
    removable garments. Non-head ancestry is recorded as evidence requiring
    inspection; it is not a reason to erase the surface.
    """
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if len(shaders) != len(triangles) or len(active_bones) != len(triangles):
        raise ValueError("classification requires shader and active bones for every source face")
    fallback = set()
    if facial_context is not None:
        if not isinstance(facial_context, dict):
            raise ValueError('facial context requires verified source metadata')
        validate_facial_context(facial_context, facial_context.get('bone_parents', {}))
        if positions is None:
            raise ValueError('facial fallback requires actual consumer positions')
        positions = np.asarray(positions, dtype=np.float64)
        if (positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all()
                or (triangles.size and (triangles.min() < 0 or triangles.max() >= len(positions)))):
            raise ValueError('facial fallback geometry has invalid positions or face indices')
        _, welded = np.unique(np.round(positions / 1e-6).astype(np.int64),
                              axis=0, return_inverse=True)
        materials = {}
        for face, shader in enumerate(shaders):
            if shader.get('_shader') == 'HGRP/CharacterNPR':
                materials.setdefault(shader.get('_material'), []).append(face)
        seeds, allowed = set(facial_context['seeds']), set(facial_context['allowed'])
        for selected in materials.values():
            selected = np.asarray(selected, dtype=np.int64)
            for island in connected_faces(welded[triangles[selected]]):
                faces = selected[island]
                active = set().union(*(active_bones[face] for face in faces))
                if active & seeds and active <= allowed:
                    fallback.update(faces.tolist())
    surfaces = {}
    retained_shaders = {"HGRP/CharacterNPR_Hair", "HGRP/CharacterNPR_Eye",
                        "HGRP/CharacterNPR_OverlayShadow"}
    for face, shader in enumerate(shaders):
        name = shader.get("_shader")
        facial = name == "HGRP/CharacterNPR_Skin" and "_SDFLIGHTMAP" in shader.get("keywords", [])
        head_skin = (name == "HGRP/CharacterNPR_Skin" and bool(active_bones[face])
                     and active_bones[face] <= head_bones)
        role = ("head" if facial or head_skin or face in fallback or name in retained_shaders else
                "native_skin" if name == "HGRP/CharacterNPR_Skin" else "outfit")
        surfaces.setdefault((name, facial, role, face in fallback, head_skin), []).append(face)
    result = []
    for (shader, facial, role, is_fallback, head_skin), selected in surfaces.items():
        selected = np.asarray(selected, dtype=np.int64)
        for group in connected_faces(triangles[selected]):
            faces = selected[group]
            vertices = np.unique(triangles[faces])
            active = set().union(*(active_bones[i] for i in faces))
            head_only = bool(active) and active <= head_bones
            outside = sorted(active - head_bones)
            ambiguity = ("authored retained surface has bindings outside the Head subtree"
                         if role == "head" and outside else
                         "authored retained surface has no active skin bindings"
                         if role == "head" and not active else "")
            result.append({"role": role, "faces": faces.tolist(), "vertices": vertices.tolist(),
                           "shader": shader, "face_lighting": facial,
                           "skin_bones": sorted(active), "head_only": head_only,
                           "bones_outside_head": outside, "binding_ambiguity": ambiguity,
                           "classification_evidence": (
                               'matching Avatar facial morph and axial island bindings' if is_fallback
                               else 'authored skin shader and bindings wholly inside the Head subtree' if head_skin
                               else "authored shader and facial lighting keywords"),
                           **({'facial_reference': True} if is_fallback else {})})
    return result


#: The first word the source rig gives its tail joints (Si ``tail_01_jnt``, Typhoea
#: ``tail_M_stone_a_01_jnt``, Wulfa ``tail_base_L_a_01_jnt``); their cloth components say the same
#: (``MC_jsspsi_Tail``, ``MBC_Typhoea_Tail``, ``MC_tail``).
TAIL_JOINT = "tail"
#: The share of a part's vertices that hang mostly from tail joints for it to be the tail.
TAIL_SHARE = 0.5
#: How far along its chain (deepest joint over the chain's joints) a tail's skin reaches.  What hangs
#: only on the chain's first joints is worn there (owner 2026-10-22, §7.1298): Wulfa's rings and her
#: shorts' root reach the 2nd–3rd of 7 (0.29–0.43); the tails reach 5–9 of 10 (Zhuangfy's dress, 0.5),
#: 9 of 10 (Si's dress), 7 of 8 (Typhoea) and 6 of 7 (Wulfa's fur).
TAIL_REACH = 0.5


def _tail_bones(skeleton) -> np.ndarray:
    return np.array([bone.name.lower().split("_", 1)[0] == TAIL_JOINT for bone in skeleton.bones], dtype=bool)


def _tail_depths(skeleton, tail: np.ndarray) -> np.ndarray:
    """Per bone, how far along its tail chain it is over the chain's joints (0 off a tail)."""
    depth = np.zeros(len(tail), dtype=np.int64)
    root = np.full(len(tail), -1, dtype=np.int64)
    for i in np.flatnonzero(tail):
        steps, top, parent = 1, i, skeleton.bones[i].parent
        while parent is not None and parent >= 0 and tail[parent]:
            steps, top, parent = steps + 1, parent, skeleton.bones[parent].parent
        depth[i], root[i] = steps, top
    out = np.zeros(len(tail))
    for top in set(root[tail].tolist()):
        chain = root == top
        out[chain] = depth[chain] / depth[chain].max()
    return out


def _tail_share(mesh, tail: np.ndarray, rows=None) -> float:
    rows = np.arange(mesh.vertex_count) if rows is None else rows
    heaviest = mesh.bone_idx[rows, np.argmax(mesh.bone_wt[rows], axis=1)]
    return float(np.mean(tail[np.clip(heaviest, 0, len(tail) - 1)])) if len(rows) else 0.0


def _is_tail(mesh, tail: np.ndarray, reach: np.ndarray, rows=None) -> bool:
    """Whether the vertices *rows* of *mesh* are the tail: mostly on its joints, and out along its chain."""
    rows = np.arange(mesh.vertex_count) if rows is None else rows
    if not len(rows) or _tail_share(mesh, tail, rows) < TAIL_SHARE:
        return False
    heaviest = np.clip(mesh.bone_idx[rows, np.argmax(mesh.bone_wt[rows], axis=1)], 0, len(tail) - 1)
    return float(reach[heaviest].max()) >= TAIL_REACH


def tail_pieces(outfit, skeleton) -> set[tuple[int, int]]:
    """``(part index, island)`` of an outfit's pieces that are the tail (:func:`tail_parts`, piece by piece).

    A renderer can carry the tail with the clothes: Si's is 35 pieces of her dress.
    """
    tail = _tail_bones(skeleton)
    if not tail.any():
        return set()
    reach = _tail_depths(skeleton, tail)
    out = set()
    for index in sorted(outfit.parts):
        mesh = outfit.meshes[index]
        if not mesh.vertex_count or mesh.bone_idx is None:
            continue
        faces = mesh.indices.reshape(-1, 3)
        labels = outfit.labels[index]
        for island in range(int(labels.max()) + 1 if len(labels) else 0):
            if _is_tail(mesh, tail, reach, np.unique(faces[labels == island])):
                out.add((index, island))
    return out


def tail_parts(parts, skeleton) -> set[int]:
    """The indices of *parts* that are the character's tail: most of their vertices hang mostly from tail joints.

    A tail is drawn with the general program, as clothes are, and the
    game's renderer calls it ``cloth_*``; but it is the character's own, as
    hair is: undressing left Si and Typhoea without theirs, and it was listed
    with the clothes (owner 2026-10-15, §7.1089).  Its joints are the source
    rig's own word for it (:data:`TAIL_JOINT`), not a guess from a renderer's
    name; a part that only touches the tail's root (Wulfa's shorts, 37 %)
    stays what it is, and so does one that hangs only on the chain's first
    joints (:data:`TAIL_REACH`): the rings worn at the root of Wulfa's tail.
    """
    tail = _tail_bones(skeleton)
    if not tail.any():
        return set()
    reach = _tail_depths(skeleton, tail)
    return {index for index, part in enumerate(parts)
            if part.mesh.vertex_count and part.mesh.bone_idx is not None and _is_tail(part.mesh, tail, reach)}


def declared_faces(triangles, role: str, active_bones: list[set[str]], head_bones: set[str],
                   shader: dict | None) -> list[dict]:
    """:func:`classify_faces`' records for a renderer whose role its conversion declared.

    Every face takes the declared role (:data:`DECLARED_CLASS`), island by
    island, with the same binding evidence the shader rule records.
    """
    if role not in DECLARED_CLASS:
        raise ValueError(f"undeclared surface role {role!r}")
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    kind = DECLARED_CLASS[role]
    result = []
    for faces in connected_faces(triangles):
        active = set().union(*(active_bones[i] for i in faces))
        outside = sorted(active - head_bones)
        result.append({"role": kind, "faces": faces.tolist(), "vertices": np.unique(triangles[faces]).tolist(),
                       "shader": (shader or {}).get("_shader"), "face_lighting": False,
                       "skin_bones": sorted(active), "head_only": bool(active) and active <= head_bones,
                       "bones_outside_head": outside,
                       "binding_ambiguity": ("declared head surface has bindings outside the Head subtree"
                                             if kind == "head" and outside else ""),
                       "classification_evidence": f"declared role table: {role}"})
    return result


def declared_tails(parts, roles: dict) -> set[int]:
    """The indices of *parts* whose renderer the conversion declared the character's tail.

    A part's name is ``<renderer>#<slot>`` (``endfield/actor.load_actor``), with
    what a split appended after ``|``.
    """
    if not roles:
        return set()
    return {index for index, part in enumerate(parts)
            if roles.get(str(part.name).split("|", 1)[0].rpartition("#")[0]) == "tail"}


def classify_piece(piece, skeleton, humanoid: dict, *, facial_faces=()) -> list[dict]:
    """Return canonical source-face islands with material and rig evidence."""
    head = humanoid.get("Head")
    head_bones = ({skeleton.bones[i].name for i in skeleton.descendants(head)}
                  if head and skeleton.has(head) else set())
    triangles = piece.mesh.indices.reshape(-1, 3)
    weights, indices = split_archive_skin(piece.skin, bone_count=len(piece.bones))
    active = []
    for face in triangles:
        active.append({piece.bones[i] for i in np.unique(indices[face][weights[face] > 0])})
    declared = declared_role(piece)
    result = (classify_faces(triangles, [piece.shader or {}] * len(triangles), active, head_bones)
              if declared is None else declared_faces(triangles, declared, active, head_bones, piece.shader))
    facial_faces = set(facial_faces)
    if not facial_faces:
        return result
    classified = []
    for row in result:
        selected = sorted(set(row['faces']) & facial_faces)
        remaining = sorted(set(row['faces']) - facial_faces)
        if remaining:
            classified.append({**row, 'faces': remaining,
                               'vertices': np.unique(triangles[remaining]).tolist()})
        if selected:
            classified.append({**row, 'role': 'head', 'faces': selected,
                               'vertices': np.unique(triangles[selected]).tolist(),
                               'classification_evidence': 'matching Avatar facial morph and axial island bindings',
                               'facial_reference': True})
    return classified


def facial_evidence(source_extra):
    """Read only the morph config linked to the source model's authored Avatar."""
    from . import morph
    configs = [table for table in morph.for_model(source_extra.get('model', ''))
               if isinstance(table, dict) and table.get('kind') == 0]
    if len(configs) > 1:
        raise ValueError('ambiguous authored facial morph evidence')
    if not configs:
        return None
    result = dict(configs[0])
    index = morph.folder() / morph.INDEX
    if index.is_file():
        import json
        listing = json.loads(index.read_text(encoding='utf-8'))
        name = (listing.get(source_extra.get('model', '')) or {}).get('face')
        if name:
            result['source_evidence_files'] = [
                f'{morph.FOLDER}/{morph.INDEX}', f'{morph.FOLDER}/config/{name}.json']
    return result


def has_sdf_face(pieces):
    """Return whether the original source already declares a facial SDF surface."""
    return any((piece.shader or {}).get('_shader') == 'HGRP/CharacterNPR_Skin'
               and '_SDFLIGHTMAP' in (piece.shader or {}).get('keywords', [])
               for piece in pieces)


def facial_context(skeleton, source_extra, face_evidence):
    """Validate source Avatar/model/rig identity and serialize bounded facial bindings."""
    if not isinstance(face_evidence, dict) or face_evidence.get('kind') != 0:
        raise ValueError('no authored facial morph evidence')
    avatar = source_extra.get('avatar')
    if (not isinstance(avatar, dict) or not isinstance(avatar.get('cab'), str)
            or not avatar['cab'] or type(avatar.get('path_id')) is not int
            or face_evidence.get('avatar') != avatar):
        raise ValueError('facial morph Avatar evidence is missing or mismatched')
    model = source_extra.get('model')
    if not isinstance(model, str) or not model or face_evidence.get('model') != model:
        raise ValueError('facial morph model evidence is mismatched')
    seeds = set(face_evidence.get('bones') or [])
    if not seeds or any(not skeleton.has(name) for name in seeds):
        raise ValueError('facial morph bones are missing from the source rig')
    humanoid = source_extra.get('humanoid') or {}
    head, chest = humanoid.get('Head'), humanoid.get('UpperChest')
    if not skeleton.has(head) or not skeleton.has(chest):
        raise ValueError('facial morph requires authored Head and UpperChest bones')
    head_names = {skeleton.bones[index].name for index in skeleton.descendants(head)}
    if not seeds <= head_names:
        raise ValueError('facial morph bones extend outside the Head subtree')
    allowed = set(head_names)
    index, seen = skeleton.index[head], set()
    while index >= 0 and skeleton.bones[index].name != chest:
        if index in seen:
            raise ValueError('facial source rig contains a cyclic Head ancestor chain')
        seen.add(index)
        allowed.add(skeleton.bones[index].name)
        index = skeleton.bones[index].parent
    if index < 0:
        raise ValueError('facial Head is not a descendant of authored UpperChest')
    allowed.add(chest)
    parents = {name: (skeleton.bones[skeleton.bones[skeleton.index[name]].parent].name
                     if skeleton.bones[skeleton.index[name]].parent >= 0 else None)
               for name in sorted(allowed)}
    context = {'avatar': dict(avatar), 'model': model, 'seeds': sorted(seeds),
               'allowed': sorted(allowed), 'bone_parents': parents}
    validate_facial_context(context, parents)
    return context


def validate_facial_context(context, bone_parents):
    """Reject consumer bones whose bounded names or parent links differ from the source."""
    if not isinstance(context, dict) or not isinstance(bone_parents, dict):
        raise ValueError('facial context requires source and consumer bone parent mappings')
    avatar = context.get('avatar')
    if (not isinstance(avatar, dict) or not isinstance(avatar.get('cab'), str)
            or not avatar['cab'] or type(avatar.get('path_id')) is not int
            or not isinstance(context.get('model'), str) or not context['model']):
        raise ValueError('facial context is missing exact Avatar/model evidence')
    seeds, allowed = context.get('seeds'), context.get('allowed')
    if (not isinstance(seeds, list) or not seeds or not isinstance(allowed, list)
            or not allowed or any(not isinstance(name, str) or not name for name in seeds + allowed)
            or not set(seeds) <= set(allowed)):
        raise ValueError('facial context requires bounded source morph bone names')
    expected = context.get('bone_parents')
    if not isinstance(expected, dict) or set(expected) != set(allowed):
        raise ValueError('facial context must retain every bounded source bone parent')
    for name, parent in expected.items():
        if parent is not None and (not isinstance(parent, str) or not parent):
            raise ValueError(f'invalid facial source parent: {name}')
        if name not in bone_parents or bone_parents[name] != parent:
            raise ValueError(f'facial source/consumer bone parent mismatch: {name}')
        seen, current = set(), name
        while current in expected:
            if current in seen:
                raise ValueError('facial context contains a cyclic bone parent chain')
            seen.add(current)
            current = expected[current]


def facial_reference_islands(pieces, skeleton, source_extra, face_evidence=None):
    """Select exact source faces using SDF or matching Avatar/morph evidence.

    A morph-driven island may include the authored axial neck bindings. Limb
    or garment-helper branches disqualify it even when they share a jaw bone.
    The resulting candidates still require the normal attachment gates.
    """
    sdf = [{"piece": index, "faces": list(range(piece.mesh.tri_count)),
            "evidence": "authored facial SDF"}
           for index, piece in enumerate(pieces)
           if (piece.shader or {}).get("_shader") == "HGRP/CharacterNPR_Skin"
           and "_SDFLIGHTMAP" in (piece.shader or {}).get("keywords", [])]
    if sdf:
        return sdf
    declared = [{"piece": index, "faces": list(range(piece.mesh.tri_count)), "evidence": "declared face"}
                for index, piece in enumerate(pieces) if declared_role(piece) == "face"]
    if declared:
        return declared
    context = facial_context(skeleton, source_extra, face_evidence)
    head_names = {skeleton.bones[index].name
                  for index in skeleton.descendants(source_extra['humanoid']['Head'])}
    result = []
    for piece_index, piece in enumerate(pieces):
        if (piece.shader or {}).get("_shader") != "HGRP/CharacterNPR":
            continue
        if piece.skin is None:
            continue
        triangles = piece.mesh.indices.reshape(-1, 3)
        weights, indices = split_archive_skin(piece.skin, bone_count=len(piece.bones))
        active = []
        for vertices in triangles:
            active.append({piece.bones[index] for index in np.unique(indices[vertices][weights[vertices] > 0])})
        records = classify_faces(triangles, [piece.shader or {}] * len(triangles), active,
                                 head_names, positions=piece.mesh.positions, facial_context=context)
        for record in records:
            if record.get('facial_reference'):
                result.append({"piece": piece_index, "faces": record['faces'],
                               "evidence": "matching Avatar facial morph and axial island bindings",
                               "avatar": dict(context['avatar']), "model": context['model'],
                               "morph_bones": list(context['seeds'])})
    if not result:
        raise ValueError("facial morph evidence identifies no compatible source island")
    return result


def face_reference(pieces: list, skeleton, source_extra=None, face_evidence=None):
    """Authored facial surfaces in rest space, without renderer-name matching."""
    from ..mesh.mesh import Mesh
    from .actor import _rest_mesh
    selected = facial_reference_islands(pieces, skeleton, source_extra or {}, face_evidence)
    mesh = Mesh("native_face_reference")
    for record in selected:
        part = _rest_mesh(pieces[record["piece"]], skeleton)
        part.indices = part.indices.reshape(-1, 3)[record["faces"]].ravel().copy()
        vertices, inverse = np.unique(part.indices, return_inverse=True)
        original_count = part.vertex_count
        for field in ('positions', 'normals', 'uvs', 'colors', 'bone_idx', 'bone_wt',
                      *[name for name, _ in Mesh.OPTIONAL_STREAMS]):
            stream = getattr(part, field)
            if len(stream) == original_count:
                setattr(part, field, stream[vertices].copy())
        part.indices = inverse.astype(np.uint32)
        mesh.append(part)
    return mesh


#: The body's regions, by the human bone each hangs under (:func:`face_regions`).
REGIONS = ("Head", "Neck", "UpperChest", "Chest", "Spine", "Hips",
           "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
           "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
           "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "RightUpperLeg", "RightLowerLeg", "RightFoot")


def face_regions(mesh, skeleton, humanoid: dict) -> np.ndarray:
    """Per face, the index in :data:`REGIONS` of the human bone its heaviest bone hangs under (-1: none)."""
    named = {skeleton.index[node]: REGIONS.index(human) for human, node in humanoid.items() if human in REGIONS}
    of_bone = np.full(len(skeleton.bones), -1, dtype=np.int64)
    for index, bone in enumerate(skeleton.bones):            # parents come first
        of_bone[index] = named.get(index, of_bone[bone.parent] if bone.parent >= 0 else -1)
    heavy = mesh.bone_idx[np.arange(mesh.vertex_count), np.argmax(mesh.bone_wt, axis=1)].astype(np.int64)
    corner = of_bone[heavy][mesh.indices.reshape(-1, 3).astype(np.int64)]
    return np.where(corner[:, 0] == corner[:, 1], corner[:, 0], corner[:, 2])
