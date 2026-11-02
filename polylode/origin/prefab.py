"""Assemble a Unity prefab into a single project Mesh.

An Illusion item is a prefab GameObject holding several renderers, each with its
own transform under the prefab root.  This walks that hierarchy, accumulates the
transforms and merges every renderer into one mesh in project space.

Coordinate handling: Unity is left-handed, this project is right-handed, and
both put the character's face along +Z.  The handedness change therefore mirrors
**X** (mirroring Z instead would put the fringe on the back of the head);
rotation quaternions become (x, -y, -z, w) and winding is reversed to match.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..core import math3d as m3
from ..mesh.mesh import Mesh
from . import colorspace as CS
from .meshimport import _load_bundle, _parse_obj
from ..core import swallowed


#: The GameObject names ``ChaReference.CreateReferenceInfo`` looks up inside a
#: clothing prefab for the ``S_C*`` / ``S_UW*`` / ``S_PANST`` / ``S_TPARTS``
#: state keys (``RefObjKey`` 92‥116), which ``ChaControl.UpdateVisible``
#: switches per clothes state (``outfit.state_node_visible``).  A renderer's
#: nearest such ancestor is stored as ``_stateNode`` (§7.379).  ``o_bra_*`` /
#: ``o_inner_*`` (117‥120) are looked up too but ``UpdateVisible`` never
#: switches them, so they are not state nodes.
CLOTHES_STATE_NODES = frozenset({
    "n_top_a", "n_top_b", "n_bot_a", "n_bot_b", "n_bot_c",
    "n_panst_a", "n_panst_b",
})


def _submesh_triangle_counts(text: str) -> list[int]:
    """Triangles per submesh in UnityPy's OBJ export (§7.379).

    ``export_mesh_obj`` writes one ``g <name>_<i>`` group per entry of
    ``get_triangles()`` — i.e. per submesh, in order — after the header group,
    and every face line is a triangle.
    """
    counts: list[int] = []
    header = False
    for line in text.splitlines():
        if line.startswith("g "):
            if header:
                counts.append(0)
            header = True
            continue
        if line.startswith("f ") and counts:
            counts[-1] += 1
    return counts


def _unorm(arr: np.ndarray) -> np.ndarray | None:
    """Vertex colours as 0..1, or ``None`` when the encoding is unknown.

    UnityPy normalises ``m_Colors`` on its legacy and compressed paths but
    *not* on the interleaved vertex-data one, which is the path every Origin mesh
    takes: it hands back the raw channel components, and Unity stores COLOR as
    ``Color32``, so they arrive as integers 0..255.  Reading them as if they
    were already 0..1 makes ``main_skin``'s ``COLOR.r`` a factor of 255 on the
    areola's uv, which samples several hundred UV-units off the map.

    The float / integer split is the encoding's, not a look at the numbers —
    a float channel is already 0..1.  Anything integral outside 0..255 is some
    other packing we have not seen; there is no safe reading of it, so the
    stream is dropped and the caller falls back to Unity's white.
    """
    if np.issubdtype(arr.dtype, np.integer):
        if arr.size and (arr.min() < 0 or arr.max() > 255):
            return None
        arr = arr.astype(np.float32) / 255.0
    # An all-white stream says nothing a missing one does not: Unity feeds
    # white to a mesh with no COLOR, and every reader here falls back to 1.0.
    # Dropping it *at import* is what keeps the two load paths identical —
    # ``Mesh.append`` discards an optional stream when only one side carries
    # it, so "present but white" on one path and "absent" on the other stops
    # agreeing the moment two pieces merge (``origin_bra_2_0``, five parts across
    # two cards in ``bake_verify``).
    return None if np.array_equal(arr, np.ones_like(arr)) else arr


# ``<ChangeAccessoryAsync>`` files two nodes out of the accessory prefab and
# nothing else:
#
#     objAcsMove[slot, 0] = FindLoop(objAccessory[slot].transform, "N_move")
#     objAcsMove[slot, 1] = FindLoop(objAccessory[slot].transform, "N_move2")
#
# ``TransformFindEx.FindLoop`` is a pre-order depth-first search by exact
# GameObject name that tests the node itself before its children, so "the first
# one in that order" is the whole rule — :meth:`PrefabLoader.move_chain`
# matches it.  The order here is also the order the card groups ``addMove`` in.
MOVE_NODES = ("N_move", "N_move2")


class Piece(NamedTuple):
    """One renderer of a prefab: geometry, rigging and the three Origin maps.

    Illusion splits a surface across three textures and never bakes them
    together (see ``docs/origin_load_path.md``):

    * ``texture`` — ``MainTex`` (``*_t``), near-white; shading and stitching.
      **Hair has none at all**, its colour is entirely the card's.
    * ``mask``    — ``ColorMask`` (``*_mc``); R/G/B pick colour 1/2/3.
    * ``detail``  — ``DetailMask`` (``*_md``); line and groove shading.

    Feeding ``_md`` in as if it were an albedo is what made hair render at a
    third of its brightness: that map averages ~77/255, so every strand came
    out three times too dark no matter what colour the card asked for.
    """
    mesh: Mesh
    bones: list
    skin: np.ndarray | None
    binds: np.ndarray | None
    texture: np.ndarray | None = None
    mask: np.ndarray | None = None
    bone_world: list | None = None    # bone transforms inside the prefab
    detail: np.ndarray | None = None
    shader: dict | None = None        # _outline / _linewidth / colours …
    tex_set: int = 0                  # 0 = MainTex, 1 = MainTex02
    normal: np.ndarray | None = None  # _NormalMap
    alpha_mask: np.ndarray | None = None   # _AlphaMask
    ramp: np.ndarray | None = None    # _AnotherRamp (toon band shape)
    mat_slot: int = 0                 # which of the renderer's materials
    shapes: list | None = None        # (name, indices, deltas) blend shapes
    line_mask: np.ndarray | None = None    # _LineMask (line work, not albedo)
    # ``bone_world`` after the prefab's DynamicBone chains have been run to
    # rest; ``None`` per bone that no chain touches.  See ``sway_rest``.
    bone_rest: list | None = None
    # Which of the prefab's two ``N_move`` nodes this renderer hangs under, as
    # a bitmask (1 = ``N_move``, 2 = ``N_move2``).  ``ChangeAccessory``
    # overwrites those two nodes' local TRS from the card, so a piece under one
    # of them moves with it — see :meth:`PrefabLoader.move_chain`.  A pair
    # prefab puts its left half under ``N_move`` and its right half under
    # ``N_move2``, so this is genuinely per-renderer and not an item-wide flag.
    # Measured over the 357 base accessories: 6 have a renderer whose *own*
    # transform sits outside ``N_move`` (all six are skinned and their bones are
    # inside it, so nothing is left unmoved), and 5 have one with neither its
    # transform nor any of its bones under ``N_move`` — those move with
    # ``N_move2`` alone.
    move_mask: int = 0
    # the same, per skin bone — the geometry of a skinned piece follows its
    # bones, so that is where the override has to land
    move_bones: np.ndarray | None = None
    # ``Texture2D.m_ColorSpace`` per map above, by field name.  A map with no
    # entry has no known colour space — which is a failure, not a default: the
    # local ``*_mc`` masks split 62 linear / 40 sRGB and a mask is a lerp
    # weight, so picking one would move the colour of every garment that
    # guessed wrong (:mod:`colorspace`).
    map_cs: dict | None = None
    # ``_GlassRamp`` — ``toon_glasses_lod0``'s reflection ramp, sampled by
    # ``|R·H| * (N.x + N.y + N.z)``; its own field because it is not
    # ``_AnotherRamp`` and a field that meant both would be read wrong by one
    glass_ramp: np.ndarray | None = None


@dataclass
class PrefabPart:
    mesh_name: str
    node: str
    matrix: np.ndarray
    bones: int = 0
    mesh_id: int = 0        # path_id: names are not unique within a bundle
    # ``m_FileID`` of the reference.  A path id only identifies an object
    # *within one file*, and a bundle can hold several: every mesh in a map
    # scene sits in ``buildplayer-<map>.sharedassets`` and is referenced with
    # file id 1.  Looking those up in the main file's table does not miss —
    # it returns whatever object happens to carry that number there, which on
    # ``myroom`` was a MeshFilter for all 61 of them.
    mesh_file: int = 0


def _orthonormal(m3x3: np.ndarray) -> np.ndarray:
    """The rotation part of a bone matrix, with any scale divided out.

    A frame is transposed to get into it, which is only the inverse for an
    orthonormal one.
    """
    m = np.asarray(m3x3, dtype=np.float64).copy()
    for c in range(3):
        n = float(np.linalg.norm(m[:, c]))
        m[:, c] = m[:, c] / n if n > 1e-12 else (0.0, 0.0, 0.0)
        if n <= 1e-12:
            m[c, c] = 1.0
    return m


def _quat_unity_to_local(q) -> np.ndarray:
    """Mirror a Unity quaternion through the X plane."""
    x, y, z, w = float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])
    return m3.quat_normalize(np.array([x, -y, -z, w]))


def _trs(tree: dict) -> np.ndarray:
    p = tree.get("m_LocalPosition") or {"x": 0, "y": 0, "z": 0}
    r = tree.get("m_LocalRotation") or {"x": 0, "y": 0, "z": 0, "w": 1}
    s = tree.get("m_LocalScale") or {"x": 1, "y": 1, "z": 1}
    pos = np.array([-float(p["x"]), float(p["y"]), float(p["z"])])
    return m3.compose(pos, _quat_unity_to_local(r),
                      np.array([float(s["x"]), float(s["y"]), float(s["z"])]))


class PrefabLoader:
    def __init__(self, bundle_path: Path, env=None):
        """Open *bundle_path*, or wrap an *env* UnityPy already loaded.

        The second form is for bundles that do not sit on disk as a file of
        their own: a zipmod member (``load_mesh._open_zipmod``) or a resource
        embedded in a plugin DLL (``figure.lewdcrest``).  Both used to set the
        attributes by hand after ``__new__``, and that copy had drifted —
        ``_extern`` was missing from it.
        """
        self.path = Path(bundle_path)
        self.env = env if env is not None else _load_bundle(self.path)
        self.by_id = {o.path_id: o for o in self.env.objects}
        # mesh path_id -> triangles per submesh, for meshes with more than one
        # (``submesh_piece``, §7.379)
        self.submesh_tris: dict = {}
        # Objects reached through a non-zero ``m_FileID``, keyed by the pair
        # that actually identifies them.  See ``PrefabPart.mesh_file``.
        self._extern: dict[tuple[int, int], object] = {}
        self._tex_index = None
        self._dynbones = None
        self.last_sway: dict = {}
        # the same chains' solver parameters, so the run time can solve them
        # against the current pose (``sway_chain_specs``, §7.540)
        self.last_sway_spec: list = []
        self.last_moves: list = [None, None]
        # ``LoadCharaFbxDataAsync``'s ``copyDynamicBone`` argument — see
        # :meth:`_colliders_for`.  True is the clothes loaders' value and stays
        # the default because clothes are what this loader was written for;
        # every caller that is *not* a garment says so (``load_mesh``'s
        # ``_pieces_from_loader``, and the head in ``body_head``).
        self.copy_dynamic_bone = True

    # ------------------------------------------------------------------
    def _name(self, obj) -> str:
        try:
            return obj.read().m_Name
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader._name: return obj.read().m_Name", exc)
            return ""

    def _deref(self, ref):
        if not isinstance(ref, dict):
            return None
        inner = ref.get("component", ref)
        if not isinstance(inner, dict):
            return None
        return self.by_id.get(inner.get("m_PathID"))

    def _container_prefab(self, name: str):
        """The GameObject the bundle's own table of contents files under *name*.

        Origin loads a prefab through ``AssetBundleManager.LoadAsset`` →
        ``AssetBundle.LoadAsset(name, type)``, which looks the name up in the
        bundle's ``m_Container`` (entries keyed by asset path; the lookup takes
        the file name without extension, case-blind) — not in the scene of
        objects.  The two differ whenever the name is also a node somewhere
        else: one local hair pack has three ``f_18R`` GameObjects and the first
        one in object order is a bone under ``f_18L``'s armature, with no
        renderer under it (§7.476).
        """
        want = name.lower()
        try:
            items = list(self.env.container.items())
        except (AttributeError, KeyError, ValueError) as exc:
            swallowed.note("origin.prefab.PrefabLoader._container_prefab", exc)
            return None
        for path, ptr in items:
            base = str(path).replace("\\", "/").rsplit("/", 1)[-1]
            stem = base.rsplit(".", 1)[0] if "." in base else base
            if stem.lower() != want:
                continue
            if int(getattr(ptr, "file_id", getattr(ptr, "m_FileID", 0)) or 0) != 0:
                continue
            obj = self.by_id.get(getattr(ptr, "path_id", None))
            if obj is not None and obj.type.name == "GameObject":
                return obj
        return None

    def find_prefab(self, name: str):
        """The GameObject the bundle's container lists under *name*, or None.

        The container is what ``AssetBundle.LoadAsset`` asks
        (:meth:`_container_prefab`), and it is the only thing asked here.  The
        name guesses that used to follow it (``p_`` variants, GameObject names,
        substrings) found nothing the game would: 3,285 clothing, hair and
        accessory rows and every body, head, uncensor and Studio caller
        resolve through the container, and the one that did not — the penis
        prefab — is named by the uncensor's manifest instead (§7.498).
        """
        if not name:
            return None
        return self._container_prefab(name.lower())

    def _transform_of(self, game_object) -> object | None:
        for c in game_object.read_typetree().get("m_Component", []):
            o = self._deref(c)
            if o is not None and o.type.name == "Transform":
                return o
        return None

    def _children(self, tree: dict, seen: set):
        """Child transforms of this node, each reached once.

        Every walk over a prefab hierarchy in this file used to stop at a depth
        counter, described in the one place it was written down as "a guard
        against a cyclic hierarchy, not a real limit".  As a cycle guard a
        counter is both loose and lossy: it lets a cycle run round as many times
        as the ceiling allows, and it silently truncates any tree that is simply
        deeper than the number somebody picked.  A visited set is the exact
        guard — a node is entered once, a cycle terminates on its second edge,
        and nothing correct is ever cut off.

        The truncation was not theoretical.  ``_collect_worlds``'s ceiling of 12
        left **14,199 of 40,404** skinning bones (35.1%) with no world matrix
        across 376 prefab roots in the base bundles plus 30 sampled zipmods, and
        219 of those roots (58.2%) are deeper than 12.  What that costs is
        invisible on a garment, which is retargeted onto our own armature and
        never reads these transforms, and fatal on anything resolved in its own
        prefab space: ``rest_pose`` skips a bone with no matrix, re-normalises
        the remaining weights, and leaves a vertex that had no other bone
        exactly where the vertex buffer put it.  ``Centaur's Body`` hangs
        ``legbackRbot`` / ``legbackLbot`` / ``Tail4`` at depth **13**, one past
        the ceiling, so the horse's hind legs stayed 1.2 m above the rest of the
        animal as two straight columns while the front legs were placed
        correctly — the shape §7.91 spent a round narrowing down (§7.96).
        """
        for ch in tree.get("m_Children", []):
            co = self.by_id.get(ch.get("m_PathID"))
            if co is None or co.path_id in seen:
                continue
            seen.add(co.path_id)
            yield co

    # ------------------------------------------------------------------
    def parts(self, prefab_name: str) -> list[PrefabPart]:
        root = self.find_prefab(prefab_name)
        if root is None:
            return []
        troot = self._transform_of(root)
        if troot is None:
            return []
        out: list[PrefabPart] = []
        self._walk_parts(troot, out)
        return out

    def scene_parts(self) -> list[PrefabPart]:
        """Every renderer in the bundle, walked from every root transform.

        A map bundle is a Unity **scene**, not a prefab: its container is a
        ``.unity`` file and there is no single root GameObject whose name could
        be handed to :meth:`parts`.  Roots are the transforms with no father.
        """
        out: list[PrefabPart] = []
        for o in self.env.objects:
            if o.type.name != "Transform":
                continue
            try:
                tt = o.read_typetree()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader.scene_parts: o.read_typetree", exc)
                continue
            if (tt.get("m_Father") or {}).get("m_PathID"):
                continue
            self._walk_parts(o, out)
        return out

    def _mesh_of(self, comp, ref: dict):
        """The Mesh a renderer points at, honouring the reference's file id.

        Type-checked on the way out.  The old lookup took ``m_PathID`` alone
        and accepted whatever came back; a wrong file id then produced an
        object of some other type rather than nothing, so the failure looked
        like "this bundle has no geometry" instead of a bad reference.
        """
        pid = (ref or {}).get("m_PathID")
        if not pid:
            return None
        fid = int((ref or {}).get("m_FileID") or 0)
        if fid == 0:
            mo = self.by_id.get(pid)
            return mo if mo is not None and mo.type.name == "Mesh" else None
        got = self._extern.get((fid, pid))
        if got is None:
            try:
                # UnityPy resolves the pointer against the files it loaded
                # alongside this one; ``deref`` hands back a reader, which is
                # what ``_mesh_from_object`` expects.
                got = comp.read().m_Mesh.deref()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader._mesh_of: got = comp.read().m_Mesh.deref()", exc)
                return None
            if got is None:
                return None
            self._extern[(fid, pid)] = got
        return got if got.type.name == "Mesh" else None

    def mesh_reader(self, part: PrefabPart):
        """The reader for a part's mesh, wherever the part's reference pointed."""
        if part.mesh_file:
            return self._extern.get((part.mesh_file, part.mesh_id))
        return self.by_id.get(part.mesh_id)

    def _walk_parts(self, troot, out: list) -> None:
        seen = {troot.path_id}

        def walk(t, parent: np.ndarray) -> None:
            tt = t.read_typetree()
            world = parent @ _trs(tt)
            go = self.by_id.get((tt.get("m_GameObject") or {}).get("m_PathID"))
            if go is not None:
                gt = go.read_typetree()
                node = gt.get("m_Name", "")
                for c in gt.get("m_Component", []):
                    o = self._deref(c)
                    if o is None or o.type.name not in (
                            "SkinnedMeshRenderer", "MeshFilter"):
                        continue
                    rt = o.read_typetree()
                    ref = rt.get("m_Mesh") or {}
                    mo = self._mesh_of(o, ref)
                    if mo is None:
                        continue
                    out.append(PrefabPart(self._name(mo), node, world.copy(),
                                          len(rt.get("m_Bones", []) or []),
                                          mo.path_id,
                                          int(ref.get("m_FileID") or 0)))
            for co in self._children(tt, seen):
                walk(co, world)

        walk(troot, m3.identity())

    # ------------------------------------------------------------------
    def _mesh_from_object(self, obj) -> Mesh | None:
        """Decode the Mesh a renderer actually points at.

        Never look these up by name: one bundle holds every variant of an item
        and they all share a name -- ``co_bra_00.unity3d`` has seven meshes
        called ``o_bra_a``.  Taking the first match returns another prefab's
        geometry, whose vertex count then disagrees with the skin weights, so
        the piece silently loses its rigging and lands in the wrong place.
        """
        try:
            d = obj.read()
            text = d.export()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader._mesh_from_object: d = obj.read()", exc)
            return None
        if not isinstance(text, str):
            # UnityPy answers ``False`` for a mesh with no vertices — the SAO
            # Alice cloak's second renderer is one — and Unity draws nothing
            # for it.  Parsing that raised, and the raise took the whole
            # prefab's other, real renderers with it (§7.494).
            return None
        pos, nrm, uv, idx = _parse_obj(text)
        if len(pos) == 0 or len(idx) == 0:
            return None
        counts = _submesh_triangle_counts(text)
        if len(counts) > 1 and sum(counts) * 3 == len(idx):
            self.submesh_tris[int(getattr(obj, "path_id", 0))] = counts
        # UnityPy's OBJ writer already negates X on positions and normals and
        # emits faces as (c, b, a) -- the handedness change is done.  Mirroring
        # again here silently swaps left and right, which only shows up once
        # asymmetric bone weights are applied.
        if len(nrm) != len(pos):
            nrm = None
        m = Mesh.from_arrays(pos, idx, normals=nrm,
                             uvs=uv if len(uv) == len(pos) else None,
                             name=getattr(d, "m_Name", "mesh"))
        for name, arr in self._streams(d, len(pos)).items():
            setattr(m, name, arr)
        return m

    def submesh_piece(self, mesh_obj, mesh: Mesh, slot: int) -> Mesh:
        """The part of *mesh* material slot *slot* draws (§7.379).

        Unity draws submesh *i* with material *i*, and a material past the last
        submesh draws the **last** submesh again (which is how ``cf_O_eyeline``'s
        lid shadow is a second pass over the lash).  Handing every slot the
        whole mesh drew each submesh once per material — 5.7% of sampled zipmod
        renderers carry more than one.  A single-submesh mesh comes back as is.
        """
        counts = self.submesh_tris.get(int(getattr(mesh_obj, "path_id", 0)))
        if not counts:
            return mesh
        s = min(int(slot), len(counts) - 1)
        start = sum(counts[:s])
        out = mesh.copy()
        out.indices = mesh.indices[3 * start:3 * (start + counts[s])].copy()
        return out

    @staticmethod
    def _streams(mesh_obj, count: int) -> dict:
        """The vertex streams the OBJ export drops: UV1..UV3 and COLOR.

        The exporter only carries UV0, so this reads the decoded vertex buffer
        directly.  It is the same buffer the exporter walks, in the same order,
        so index *i* here is index *i* in the mesh we just built — the same
        alignment ``_blend_shapes`` relies on.

        Who ships what, measured on the stock assets: hair has UV1 (the layout
        that places each strand on the ``_HairGloss`` sheet, which is where
        ``main_hair``'s highlight band comes from); ``o_body_a`` has all three
        sets *and* a COLOR stream; ``cf_O_face`` has all three sets and **no**
        COLOR.  Most clothing has none of them, so nothing is stored for it.

        COLOR lands in ``vcolor``, never in ``Mesh.colors`` — see the note on
        ``Mesh.OPTIONAL_STREAMS``.  Absent is left absent: the decal bake reads
        1.0 for a mesh without one, which is what Unity feeds a shader whose
        mesh has none.  That is not a guess — ``main_skin`` multiplies the lip
        decal's uv by COLOR.r, so a face reading zero there could never draw
        lipstick at all, and Origin draws it.
        """
        out: dict = {}
        try:
            from UnityPy.helpers.MeshHelper import MeshHandler
            h = MeshHandler(mesh_obj)
            h.process()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader._streams: from UnityPy.helpers.MeshHelper import MeshHandl", exc)
            return out
        for attr, name, width in (("m_UV1", "uv1", 2), ("m_UV2", "uv2", 2),
                                  ("m_UV3", "uv3", 2),
                                  ("m_Colors", "vcolor", 4)):
            raw = getattr(h, attr, None)
            if raw is None:
                continue
            arr = np.asarray(raw)
            if arr.size != count * width:
                continue
            if name == "vcolor":
                arr = _unorm(arr)
                if arr is None:
                    continue
            out[name] = arr.astype(np.float32).reshape(-1, width)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _blend_shapes(mesh_tree: dict) -> list:
        """Unity ``m_Shapes`` → ``[(name, indices, deltas, normal_deltas)]``.

        The layout is three parallel lists: ``channels`` names each shape,
        ``shapes`` gives its slice of the shared sparse ``vertices`` array, and
        each entry is ``{index, vertex, normal, tangent}``.  Only single-frame
        channels appear in Illusion's heads, so a channel maps to one slice.

        Indices address the mesh's own vertex order, which survives the OBJ
        round-trip intact — ``cf_O_face`` reports 2695 vertices on both sides
        and the largest shape index is 2694.  X is negated to match the
        handedness change UnityPy's writer already applied to the positions.

        ``normal_deltas`` is the frame's ``normal`` column, X negated the same
        way, or ``None`` when the frame says ``hasNormals`` is false — Unity
        adds it to the mesh normal before normalising, so a morphed lid keeps
        its authored shading instead of the rest pose's (§7.379).
        """
        sh = mesh_tree.get("m_Shapes") or {}
        verts = sh.get("vertices") or []
        shapes = sh.get("shapes") or []
        channels = sh.get("channels") or []
        if not (verts and shapes and channels):
            return []
        out = []
        for ch in channels:
            first = int(ch.get("frameIndex", 0))
            n = int(ch.get("frameCount", 1))
            if n < 1 or first >= len(shapes):
                continue
            # one channel, one frame in every Illusion head; take the last
            # frame so a multi-frame channel still lands on full weight
            spec = shapes[min(first + n - 1, len(shapes) - 1)]
            a = int(spec.get("firstVertex", 0))
            k = int(spec.get("vertexCount", 0))
            if k <= 0:
                continue
            block = verts[a:a + k]
            idx = np.fromiter((int(v["index"]) for v in block),
                              dtype=np.int32, count=len(block))
            d = np.empty((len(block), 3), dtype=np.float32)
            for i, v in enumerate(block):
                q = v["vertex"]
                d[i] = (-float(q["x"]), float(q["y"]), float(q["z"]))
            nd = None
            if spec.get("hasNormals", True):
                nd = np.empty((len(block), 3), dtype=np.float32)
                for i, v in enumerate(block):
                    q = v.get("normal") or {}
                    nd[i] = (-float(q.get("x", 0.0)), float(q.get("y", 0.0)),
                             float(q.get("z", 0.0)))
                if not np.any(nd):
                    nd = None       # a frame of zeros changes nothing
            out.append((str(ch.get("name", "")), idx, d, nd))
        return out

    # ------------------------------------------------------------------
    def _renderer_bones(self, renderer, worlds: dict | None = None,
                        settled: dict | None = None,
                        move_mask: dict | None = None):
        """(bone names, world matrices, settled ones, ``N_move`` bitmasks).

        An accessory rigs its own sway bones, which exist nowhere in the body
        armature.  Keeping their prefab-space transforms lets the piece be
        resolved to its rest shape locally and then hung off an attach node,
        instead of being retargeted against bones nobody has.

        ``settled`` is the same list after the prefab's ``DynamicBone`` chains
        have been run to rest (:meth:`sway_rest`); ``None`` for bones that no
        chain touches.  ``move_mask`` is :meth:`move_chain`'s path_id table, so
        each bone comes back with the ``N_move`` nodes it hangs under.
        """
        names, mats, rest, move = [], [], [], []
        for b in (renderer.read_typetree().get("m_Bones") or []):
            pid = b.get("m_PathID")
            t = self.by_id.get(pid)
            if t is None:
                names.append("")
                mats.append(None)
                rest.append(None)
                move.append(0)
                continue
            tt = t.read_typetree()
            go = self.by_id.get((tt.get("m_GameObject") or {}).get("m_PathID"))
            names.append(self._name(go) if go is not None else "")
            mats.append(None if worlds is None else worlds.get(pid))
            rest.append(None if settled is None else settled.get(pid))
            move.append(0 if move_mask is None else move_mask.get(pid, 0))
        return names, mats, rest, np.asarray(move, dtype=np.uint8)

    def _texture_index(self) -> dict:
        if self._tex_index is None:
            idx = {}
            for o in self.env.objects:
                if o.type.name != "Texture2D":
                    continue
                try:
                    idx.setdefault(o.read().m_Name, o)
                except Exception as exc:             # noqa: BLE001
                    swallowed.note("origin.prefab.PrefabLoader._texture_index: o.read", exc)
                    continue
            self._tex_index = idx
        return self._tex_index

    @staticmethod
    def _texture_base(material_name: str) -> str:
        """``cf_m_top_tsyatu01_a_low`` -> ``cf_top_tsyatu01``.

        Base-game materials leave ``_MainTex`` empty because the game composites
        the albedo at runtime from the card's colours; the maps themselves are
        loose Texture2D assets in the same bundle, found by this convention.
        """
        name = material_name
        for marker in ("_m_", "_M_"):
            if marker in name:
                name = name.replace(marker, "_", 1)
                break
        for suffix in ("_low", "_high"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        parts = name.split("_")
        while len(parts) > 2 and len(parts[-1]) == 1:    # trailing _a / _b variant
            parts.pop()
        return "_".join(parts)

    @staticmethod
    def _texture_name_candidates(material_name: str) -> list[str]:
        """All plausible albedo names for a material (clothing + underwear).

        ``_md`` is deliberately absent: that is the detail/line map, not an
        albedo, and an item that has no ``_t`` genuinely has no albedo.
        """
        base = PrefabLoader._texture_base(material_name)
        out = [f"{base}_t", f"{base}_t_low", base]
        # Underwear maps as ``cf_under_08_top_t`` while the material is
        # ``cf_m_bra_08`` / ``cf_m_shorts_02`` — the plain base never hits.
        import re
        low = material_name.lower()
        m = re.search(r"(?:bra|shorts|panst|under)[_\-]?(\d+)", low)
        if m:
            n = m.group(1).zfill(2) if len(m.group(1)) <= 2 else m.group(1)
            for region in ("top", "bot", "a", "b", ""):
                stem = f"cf_under_{n}" + (f"_{region}" if region else "")
                out.extend([f"{stem}_t", f"{stem}_t_low"])
        # shorts / panty materials
        m2 = re.search(r"shorts[_\-]?(\d+)", low)
        if m2:
            n = m2.group(1).zfill(2)
            for stem in (f"cf_shorts_{n}", f"cf_under_{n}_bot", f"cf_o_shorts_{n}"):
                out.extend([f"{stem}_t", f"{stem}_t_low"])
        # Mod shoes / gloves packs often name albedos ``09_t`` while the
        # material is ``mf_m_shoes9`` / ``mf_m_shoes09`` — trailing digits only.
        m3 = re.search(r"(?:shoes?|boots?|gloves?|socks?)[_\-]?(\d+)", low)
        if m3:
            raw = m3.group(1)
            nums = {raw, raw.zfill(2), raw.lstrip("0") or "0"}
            for n in nums:
                for stem in (n, f"{n}", f"mf_{n}", f"shoes_{n}", f"mf_shoes{n}"):
                    out.extend([f"{stem}_t", f"{stem}_t_low", stem])
        # bare trailing number on any material: ``…09`` / ``…_9``
        m4 = re.search(r"(\d+)$", low.replace("_low", "").replace("_high", ""))
        if m4:
            raw = m4.group(1)
            for n in {raw, raw.zfill(2), raw.lstrip("0") or "0"}:
                out.extend([f"{n}_t", f"{n}_t_low", f"{n}_md", f"{n}_md_low"])
        # de-dupe preserve order
        seen: set[str] = set()
        uniq = []
        for c in out:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    def _decode(self, tex):
        """A :class:`colorspace.Map` — the only place a Texture2D is read.

        ``m_ColorSpace`` comes off the asset here because this is the one point
        that has it: past this line the map is a PNG's worth of bytes and the
        flag is unrecoverable.  Illusion's import settings are not consistent
        (62 of 102 local ``*_mc`` masks are linear, 96 of 107 ``*_t`` albedos are
        sRGB), so it cannot be re-derived from the name either — see
        :mod:`colorspace`.

        A texture that reads but has no ``m_ColorSpace`` is **not** given one.
        The field is not optional in this Unity version, so its absence means
        the read went wrong, and a guess here would be a coin toss on the
        lerp weights of every colour mask (CLAUDE.md 一之一).
        """
        try:
            data = tex.read()
            img = data.image
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader._decode: data = tex.read()", exc)
            return None
        if img is None:
            return None
        cs = getattr(data, "m_ColorSpace", None)
        if cs is None:
            try:
                cs = tex.read_typetree().get("m_ColorSpace")
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader._decode: cs = tex.read_typetree().get('m_ColorSpa…", exc)
                cs = None
        if cs is None:
            return None
        # Keep PIL's row order (row 0 = v = 1).  ``TextureCache.upload`` is the
        # one place that flips for GL's bottom-left origin, so flipping here
        # too handed every imported map to the GPU upside down — the iris
        # gradient came out light-on-top, and every ``_mc`` colour region and
        # ``_md`` line map was mirrored about the garment's waist.
        # The wrap mode is the asset's too (``m_TextureSettings``): a shader that
        # moves the UV off the unit square — a tiled ``_NormalMap_ST`` — tiles
        # or smears depending on it, and past this line it is gone (§7.383).
        ts = getattr(data, "m_TextureSettings", None)
        wrap = getattr(ts, "m_WrapMode", None)
        if wrap is None:
            wrap = getattr(ts, "m_WrapU", None)
        return CS.Map(np.asarray(img.convert("RGBA"), dtype=np.uint8),
                      int(cs) == CS.SRGB,
                      None if wrap is None else int(wrap))

    def _main_texture(self, renderer, slot: int = 0,
                      guess: bool = True):
        """The renderer's albedo as a :class:`colorspace.Map`, bottom-up.

        Kept as uint8: a 1024² albedo is 4MB that way and 16MB as float, and a
        card can pull in a couple of dozen of them.

        ``guess`` controls the name-convention fallback below.  It exists for
        base-game *clothing*, whose materials leave ``_MainTex`` empty because
        the game assigns it from a list row at run time.  A rigid accessory
        binds its albedo on the material, so guessing there only finds the
        wrong texture — it is what turned a light-blue beret green.
        """
        mats = self._materials(renderer)
        for mat in (mats[slot:slot + 1] if slot < len(mats) else []):
            if mat is None:
                continue
            try:
                tree = mat.read_typetree()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader._main_texture: mat.read_typetree", exc)
                continue
            props = tree.get("m_SavedProperties") or {}
            for entry in (props.get("m_TexEnvs") or []):
                name, val = (entry if isinstance(entry, (list, tuple))
                             else (entry.get("first"), entry.get("second")))
                if name != "_MainTex" or not isinstance(val, dict):
                    continue
                tex = self.by_id.get((val.get("m_Texture") or {}).get("m_PathID"))
                if tex is not None and tex.type.name == "Texture2D":
                    return self._decode(tex)         # mods do set this
            index = self._texture_index()
            mat_name = tree.get("m_Name", "") or ""
            # Clothing albedos use ``_t``; underwear uses ``cf_under_NN_*``
            # while materials are named ``cf_m_bra_NN``.  Nothing here may
            # return an ``_md``: that is the detail map, not an albedo.
            if not guess:
                return None
            for cand in self._texture_name_candidates(mat_name):
                tex = index.get(cand)
                if tex is not None:
                    return self._decode(tex)
            base = self._texture_base(mat_name)
            for cand_name, tex in index.items():     # last resort: same family
                if not base or not cand_name.startswith(base[: max(6, len(base) // 2)]):
                    continue
                # ``_md`` must never come back from here.  It is the detail /
                # line map, and handing it over as an albedo is exactly the
                # trap §7.6 records: ``cf_m_tang`` binds no ``_MainTex`` (the
                # tongue is meant to be white × ``_Color``), so this loop found
                # ``cf_tang_md`` and painted the tongue green.  That was then
                # papered over by hard-coding the tongue's colour.
                if cand_name.endswith(("_t", "_t_low")):
                    return self._decode(tex)
        return None

    def texture_by_name(self, *names: str):
        """A :class:`colorspace.Map` for a Texture2D named outright.

        For assets whose material we cannot name — the texture-only lists point
        straight at an image rather than at a prefab.
        """
        index = self._texture_index()
        for n in names:
            tex = index.get(n)
            if tex is not None:
                return self._decode(tex)
        return None

    def texture_wrap_by_name(self, *names: str) -> int | None:
        """``m_TextureSettings.m_WrapMode`` of a Texture2D named outright.

        0 is Repeat, 1 Clamp; ``None`` when no such texture is here or the field
        cannot be read.  It matters wherever a shader moves the UV off the unit
        square — ``OverBraMask`` is sampled through an ``_AlphaMask_ST`` of
        ``(1, 2, 0, -1)`` on some bras, and 126 of the local bra renderers put
        vertices outside [0, 1] after it (§7.367).  Repeat and Clamp answer those
        texels differently, so the mode travels with the map rather than being
        assumed (一之一).
        """
        index = self._texture_index()
        for n in names:
            tex = index.get(n)
            if tex is None:
                continue
            try:
                ts = tex.read().m_TextureSettings
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader.texture_wrap_by_name: ts = tex.read().m_TextureSe…", exc)
                return None
            mode = getattr(ts, "m_WrapMode", None)
            if mode is None:
                mode = getattr(ts, "m_WrapU", None)
            return None if mode is None else int(mode)
        return None

    def _detail_map(self, renderer, slot: int = 0):
        """The ``_md`` map bound as ``_DetailMask``, as a :class:`colorspace.Map`.

        For hair this is the *only* shading there is — the material binds no
        ``_MainTex`` at all, so the strands are card colour modulated by this
        and nothing else.
        """
        return self._material_texture(renderer, "_DetailMask", "_md", slot)

    def _material_slots(self, renderer) -> list[int]:
        """Indices of the renderer's **filled** material slots.

        Unity draws nothing for an empty slot, so a renderer carrying one
        material and one blank entry is drawn once.  Counting the entries drew
        it twice: ``o_top_jacket04_a`` / ``_b`` (the cardigan) are the only two
        in this install, and the phantom pass put a second 5,829-triangle copy
        of the garment on top of the real one wearing a texture the generic
        path had picked — nothing reported anything (§7.168).

        An empty slot serialises as ``{m_FileID: 0, m_PathID: 0}``, the same
        shape the empty collider lists have (§7.71): the field is there, its
        contents are not.  A non-zero ``m_FileID`` is a material in **another
        file** and stays in the list — that is a real pass we would otherwise
        drop silently, and :meth:`_materials` already answers ``None`` for it
        so the caller reports rather than invents.
        """
        out = [i for i, mp in
               enumerate(renderer.read_typetree().get("m_Materials") or [])
               if int(mp.get("m_PathID") or 0)]
        return out or [0]

    def _materials(self, renderer) -> list:
        """The renderer's materials, in order, as readable objects."""
        out = []
        for mp in (renderer.read_typetree().get("m_Materials") or []):
            mat = self.by_id.get(mp.get("m_PathID"))
            out.append(mat if (mat is not None
                               and mat.type.name == "Material") else None)
        return out

    def _material_floats(self, renderer, slot: int = 0) -> dict:
        """Shader settings of the renderer's material — scalars *and* colours.

        ``_outline`` / ``_linewidth`` are what decide whether an item is drawn
        with an inverted-hull edge and how thick.  Illusion turns the outline
        **off for hair** (``_outline = 0``) and on for clothing (``= 1``);
        drawing one anyway put a dark rim on every strand and dragged the whole
        head a shade darker than the card.
        """
        mats = self._materials(renderer)
        for mat in (mats[slot:slot + 1] if slot < len(mats) else []):
            if mat is None:
                continue
            try:
                tree = mat.read_typetree()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader._material_floats: mat.read_typetree", exc)
                continue
            props = tree.get("m_SavedProperties") or {}
            out = {}
            # ``m_ShaderKeywords`` decides which compiled variant runs, and for
            # these shaders that is not decoration: ``main_opaque`` ships both a
            # discarding and a non-discarding forward base, and
            # ``_ALPHA_A_ON`` / ``_ALPHA_B_ON`` are what select the one that
            # ends in ``discard_nz``.  95 of the 100 local ``main_opaque``
            # materials carry them (and 6/6 ``main_alpha``, 2/2 ``main_skin``),
            # so "does this garment cut out" is a fact on the asset — it was
            # being guessed from the composited pixels instead (§7.88).
            kw = str(tree.get("m_ShaderKeywords") or "")
            if kw:
                out["_keywords"] = kw
            # The material's own name: MaterialEditor's float / colour /
            # texture / shader / queue rows address a material by it
            # (``MaterialAPI`` matches ``NameFormatted()``), and nothing else on
            # a piece says which material it was drawn with (§7.379).
            out["_matName"] = str(tree.get("m_Name") or "")
            for entry in (props.get("m_Floats") or []):
                name, val = (entry if isinstance(entry, (list, tuple))
                             else (entry.get("first"), entry.get("second")))
                try:
                    out[str(name)] = float(val)
                except (TypeError, ValueError):
                    continue
            # the toon colours live in m_Colors, not m_Floats — reading only
            # the scalars left _ShadowColor / _LineColor behind and the loader
            # fell back to its own much darker defaults
            for entry in (props.get("m_Colors") or []):
                name, val = (entry if isinstance(entry, (list, tuple))
                             else (entry.get("first"), entry.get("second")))
                seq = (list(val.values()) if isinstance(val, dict)
                       else list(val) if isinstance(val, (list, tuple)) else None)
                if seq is None or len(seq) < 3:
                    continue
                try:
                    out[str(name)] = [float(x) for x in seq[:4]]
                except (TypeError, ValueError):
                    continue
            # Every sampler carries a scale/offset the fragment program applies
            # before sampling.  ``_HairGloss_ST`` is the one that matters most:
            # it is how each hair style places the shared highlight sheet, and
            # every piece has a different offset (the sampled materials run
            # from (0, -0.12) to (-0.06, 0.02)).
            for entry in (props.get("m_TexEnvs") or []):
                nm, val = (entry if isinstance(entry, (list, tuple))
                           else (entry.get("first"), entry.get("second")))
                if not isinstance(val, dict):
                    continue
                sc = val.get("m_Scale") or {}
                of = val.get("m_Offset") or {}
                try:
                    out[f"{nm}_ST"] = (float(sc.get("x", 1.0)), float(sc.get("y", 1.0)),
                                       float(of.get("x", 0.0)), float(of.get("y", 0.0)))
                except (TypeError, ValueError):
                    continue
            # Which shader this is decides things no material float can: only
            # the bytecode says whether a pass clips, and ``_alphaTest`` is
            # present on materials whose shader never reads it (toon_eye_lod0
            # ships 0.5 and has no discard at all).
            name = self._shader_name(mat)
            if name:
                out["_shader"] = name
            # Which order the blended passes draw in (§7.379): the material's
            # ``m_CustomRenderQueue`` when it is set (the wedding veil ships
            # 3600), else the shader's ``Queue`` tag (-1 means "use it").
            try:
                out["_queue"] = int(tree.get("m_CustomRenderQueue", -1))
            except (TypeError, ValueError):
                out["_queue"] = -1
            sq = self._shader_queue(mat)
            if sq is not None:
                out["_shaderQueue"] = sq
            if out:
                return out
        return {}

    #: Unity's named render queues (``Shader`` ``Queue`` tag).
    _QUEUE_BASE = {"background": 1000, "geometry": 2000, "alphatest": 2450,
                   "transparent": 3000, "overlay": 4000}

    @classmethod
    def _parse_queue(cls, text) -> int | None:
        """``"Transparent+1010"`` → 4010; ``None`` for anything unparseable."""
        s = str(text or "").strip().lower().replace(" ", "")
        if not s:
            return None
        for name, base in cls._QUEUE_BASE.items():
            if s.startswith(name):
                rest = s[len(name):]
                try:
                    return base + (int(rest) if rest else 0)
                except ValueError:
                    return None
        try:
            return int(s)
        except ValueError:
            return None

    def _shader_queue(self, mat) -> int | None:
        """The ``Queue`` tag of the material's shader (first SubShader), if any."""
        try:
            d = mat.read() if hasattr(mat, "read") else mat
            sh = getattr(d, "m_Shader", None)
            if sh is None:
                return None
            form = getattr(sh.read(), "m_ParsedForm", None)
            for sub in (getattr(form, "m_SubShaders", None) or [])[:1]:
                tags = getattr(getattr(sub, "m_Tags", None), "tags", None) or {}
                for k, v in dict(tags).items():
                    if str(k).lower() == "queue":
                        return self._parse_queue(v)
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader._shader_queue: d = mat.read() if hasattr(mat, 'read')…", exc)
            return None
        return None

    @staticmethod
    def _shader_name(mat) -> str:
        try:
            d = mat.read() if hasattr(mat, "read") else mat
            sh = getattr(d, "m_Shader", None)
            if sh is None:
                return ""
            s = sh.read()
            pf = getattr(s, "m_ParsedForm", None)
            return str(getattr(pf, "m_Name", "") or getattr(s, "m_Name", ""))
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader._shader_name: d = mat.read() if hasattr(mat, 'read')…", exc)
            return ""

    def _material_texture(self, renderer, prop: str,
                          suffix: str, slot: int = 0):
        """The texture bound to *prop* as a :class:`colorspace.Map`,
        falling back to ``<material><suffix>``."""
        mats = self._materials(renderer)
        for mat in (mats[slot:slot + 1] if slot < len(mats) else []):
            if mat is None:
                continue
            try:
                tree = mat.read_typetree()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader._material_texture: mat.read_typetree", exc)
                continue
            props = tree.get("m_SavedProperties") or {}
            for entry in (props.get("m_TexEnvs") or []):
                name, val = (entry if isinstance(entry, (list, tuple))
                             else (entry.get("first"), entry.get("second")))
                if name != prop or not isinstance(val, dict):
                    continue
                tex = self.by_id.get((val.get("m_Texture") or {}).get("m_PathID"))
                if tex is not None and tex.type.name == "Texture2D":
                    return self._decode(tex)
            index = self._texture_index()
            base = self._texture_base(tree.get("m_Name", ""))
            for cand in (f"{base}{suffix}", f"{base}{suffix}_low"):
                tex = index.get(cand)
                if tex is not None:
                    return self._decode(tex)
        return None

    def _bound_texture(self, renderer, prop: str, slot: int = 0) -> tuple:
        """``(Map | None, state)`` for a slot that must not be guessed at.

        *state* is ``"bound"``, ``"none"`` (the material leaves the slot empty,
        so the shader reads its declared default) or ``"unresolved"`` (a
        reference this loader cannot follow — another file's ``m_FileID``, or a
        path id not in the bundle).  Unlike :meth:`_material_texture` there is
        no fall-back to ``<material><suffix>``: a slot like ``_GlassRamp`` is
        read by value, and "no map" is a legitimate answer that a guessed map
        would hide (``CLAUDE.md`` 一之一).
        """
        mats = self._materials(renderer)
        mat = mats[slot] if slot < len(mats) else None
        if mat is None:
            return None, "none"
        try:
            tree = mat.read_typetree()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader._bound_texture: tree = mat.read_typetree()", exc)
            return None, "unresolved"
        for entry in ((tree.get("m_SavedProperties") or {}).get("m_TexEnvs") or []):
            name, val = (entry if isinstance(entry, (list, tuple))
                         else (entry.get("first"), entry.get("second")))
            if name != prop or not isinstance(val, dict):
                continue
            ref = val.get("m_Texture") or {}
            if not ref.get("m_PathID"):
                return None, "none"
            if ref.get("m_FileID"):
                return None, "unresolved"
            tex = self.by_id.get(ref.get("m_PathID"))
            if tex is None or tex.type.name != "Texture2D":
                return None, "unresolved"
            return self._decode(tex), "bound"
        return None, "none"

    def _colour_mask(self, renderer, slot: int = 0):
        """The ``_mc`` region mask as a :class:`colorspace.Map`.

        Which card colour goes where.

        Illusion's clothing albedo is nearly white on purpose: the visible
        colour comes from the card, selected per-region by this mask.  Without
        it every imported garment renders as a white shell.
        """
        mats = self._materials(renderer)
        for mat in (mats[slot:slot + 1] if slot < len(mats) else []):
            if mat is None:
                continue
            try:
                tree = mat.read_typetree()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader._colour_mask: mat.read_typetree", exc)
                continue
            props = tree.get("m_SavedProperties") or {}
            for entry in (props.get("m_TexEnvs") or []):
                name, val = (entry if isinstance(entry, (list, tuple))
                             else (entry.get("first"), entry.get("second")))
                if name != "_ColorMask" or not isinstance(val, dict):
                    continue
                tex = self.by_id.get((val.get("m_Texture") or {}).get("m_PathID"))
                if tex is not None and tex.type.name == "Texture2D":
                    return self._decode(tex)
            index = self._texture_index()
            base = self._texture_base(tree.get("m_Name", ""))
            for cand in (f"{base}_mc", f"{base}_mc_low"):
                tex = index.get(cand)
                if tex is not None:
                    return self._decode(tex)
        return None

    @staticmethod
    def _skin_array(mesh_tree: dict) -> np.ndarray | None:
        """m_Skin -> (N,8) of [w0..w3, i0..i3]."""
        skin = mesh_tree.get("m_Skin")
        if not skin:
            return None
        out = np.zeros((len(skin), 8), dtype=np.float64)
        for r, e in enumerate(skin):
            for k in range(4):
                out[r, k] = float(e.get(f"weight[{k}]", 0.0))
                out[r, 4 + k] = float(e.get(f"boneIndex[{k}]", 0))
        return out

    @staticmethod
    def _bind_poses(mesh_tree: dict) -> np.ndarray | None:
        """m_BindPose -> (B,4,4), converted into this project's handedness.

        A skinned mesh's vertices live in *bind space*, not world space; the bind
        pose maps them into each bone's frame.  Skipping it puts the garment at
        the origin instead of on the body.
        """
        bp = mesh_tree.get("m_BindPose")
        if not bp:
            return None
        mirror = np.diag([-1.0, 1.0, 1.0, 1.0])
        out = np.zeros((len(bp), 4, 4))
        for i, m in enumerate(bp):
            if isinstance(m, dict):
                a = np.array([[float(m.get(f"e{r}{c}", 1.0 if r == c else 0.0))
                               for c in range(4)] for r in range(4)])
            else:
                a = np.asarray(m, dtype=np.float64).reshape(4, 4)
            out[i] = mirror @ a @ mirror
        return out

    # Enough to identify ``ChaAccessoryComponent`` and no more — the same rule
    # as ``_CLOTHES_FIELDS`` below, and for a second reason on top of the null
    # array one: a zipmod serialises the component as it existed in the Origin
    # build it was made against, so a field added later is simply not in the
    # tree.  ``noOutline`` is one of those, and requiring it threw the whole
    # component away for every older mod — the renderer lost its role, the
    # card's colours were never applied, and the accessory drew in
    # ``make_material``'s default grey.  The three ``useColor0N`` flags plus
    # ``defColor01`` are value fields (always written) and belong to no other
    # component on an accessory root; ``ChaCustomHairComponent`` sits on the
    # same object and shares ``rendHair`` / ``initialize`` / ``setcolor``.
    _ACC_FIELDS = ("useColor01", "useColor02", "useColor03", "defColor01")
    #: ``ChaCustomHairComponent``: two value fields (always written) that the
    #: accessory component does not have
    _HAIR_COMP_FIELDS = ("lengthRate", "baseLength")

    def accessory_roles(self, root) -> dict:
        """What ``ChaAccessoryComponent`` says about this prefab's renderers.

        The component is **serialised on the prefab root**, already pointing at
        the renderers by PPtr, so none of this needs guessing:

        * ``rendNormal`` takes the card's accessory colours 1-3, but only where
          ``useColor01/02/03`` is set.  Where it is not, the material's own
          ``_ColorN`` is what the item is supposed to look like — a straw hat
          leaves the third off so its band stays dark brown, and a swim cap
          leaves it off so its trim stays near-black.  Painting the card's
          (white) third colour over those is what bleached them.
        * ``rendAlpha`` takes ``_Color4`` from the fourth colour and is
          switched off entirely when that colour's alpha is 0.
        * ``rendHair`` ignores the accessory's colours and takes the
          *character's* ``hair.parts[0].startColor`` in all three slots — the
          hair a swim cap or a wig brings with it belongs to the wearer.

        (``Initialize`` can also rebuild the three lists from GameObject name
        prefixes ``on_`` / ``oa_`` / ``cf_hair``, but base-game prefabs ship
        the lists filled in and use none of those names.)

        ``ChaCustomHairComponent`` on the same root is read too, because
        HairAccessoryCustomizer works off *its* lists and they are not the
        accessory component's: over 46 worn hair accessories the accessory
        component's ``rendHair`` was empty every time and the hair renderers sat
        in ``rendNormal`` (§7.446).  ``hair_comp`` maps its ``rendHair`` to
        ``"hair"`` and its ``rendAccessory`` to ``"accessory"``; it is ``None``
        when the root carries no such component.

        A hair *item*'s root carries the same component, and the game reads
        the same two lists there: ``ChangeSettingHairColor`` /
        ``ChangeSettingHairOutlineColor`` write ``rendHair``,
        ``ChangeSettingHairAcsColor`` writes the first ``len(acsDefColor)``
        colour slots of every ``rendAccessory`` (``hair_acs_n``).

        Returns ``{"roles": {path_id: role}, "use": (a, b, c), "outline": bool,
        "hair_comp": {path_id: role} | None, "hair_acs_n": int}``.
        """
        out = {"roles": {}, "use": (1, 1, 1), "outline": True, "hair_comp": None,
               "hair_acs_n": 0, "hair_len": None}
        try:
            gt = root.read_typetree()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader.accessory_roles: gt = root.read_typetree()", exc)
            return out
        for c in gt.get("m_Component", []):
            o = self._deref(c)
            if o is None or o.type.name != "MonoBehaviour":
                continue
            try:
                tt = o.read_typetree()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader.accessory_roles: o.read_typetree", exc)
                continue
            if all(k in tt for k in self._HAIR_COMP_FIELDS):
                comp = {}
                for key, role in (("rendHair", "hair"),
                                  ("rendAccessory", "accessory")):
                    for ref in (tt.get(key) or []):
                        pid = (ref or {}).get("m_PathID") if isinstance(ref, dict) else 0
                        if pid:
                            comp[pid] = role
                out["hair_comp"] = comp
                out["hair_acs_n"] = len(tt.get("acsDefColor") or [])
                # ``Update`` rewrites each ``trfLength[i]``'s local y every
                # frame (:meth:`length_chain`, §7.516)
                out["hair_len"] = {
                    "trf": [(ref or {}).get("m_PathID", 0) if isinstance(ref, dict) else 0
                            for ref in (tt.get("trfLength") or [])],
                    "base": [float(v) for v in (tt.get("baseLength") or [])],
                    "add": float(tt.get("addLength", 0) or 0),
                    "rate": float(tt.get("lengthRate", 0) or 0),
                    # a disabled component's Update never runs
                    "on": bool(tt.get("m_Enabled", 1))}
                continue
            if not all(k in tt for k in self._ACC_FIELDS) or out["roles"]:
                continue
            for key, role in (("rendNormal", "normal"), ("rendAlpha", "alpha"),
                              ("rendHair", "hair")):
                for ref in (tt.get(key) or []):
                    pid = (ref or {}).get("m_PathID")
                    if pid:
                        out["roles"][pid] = role
            out["use"] = tuple(int(tt.get(f"useColor0{i}", 1) or 0)
                               for i in (1, 2, 3))
            out["outline"] = not int(tt.get("noOutline", 0) or 0)
        return out

    # Enough to identify ``ChaClothesComponent`` and no more.  A null
    # ``Renderer[]`` is simply **absent** from the serialised object -- every
    # base-game top leaves ``rendNormal03`` out -- so a signature that lists
    # every field the type declares matches nothing at all.
    _CLOTHES_FIELDS = ("rendNormal01", "rendEmblem01", "useColorN01")

    def clothes_roles(self, root) -> dict:
        """``path_id -> role`` from ``ChaClothesComponent``, serialised on the root.

        Which of a garment's renderers gets which map is decided here and
        nowhere else.  ``InitBaseCustomTextureClothes`` builds **three**
        composites from three separate column pairs of the item's list row --

            ct[0]  MainTex   (33) / ColorMaskTex   (7)     KeyType, and the
            ct[1]  MainTex02 (34) / ColorMask02Tex (5)     ``*AB`` column beside
            ct[2]  MainTex03(104) / ColorMask03Tex(102)    each, else MainAB

        -- and ``ChangeCustomClothes`` hands ct[0] to ``rendNormal01``,
        ``rendAlpha01`` and ``rendAccessory``, ct[1] to ``rendNormal02`` and
        ct[2] to ``rendNormal03``.  The loader used to pick between the sets by
        asking whether the renderer's *mesh name* contained the texture's stem,
        which put the blouse's albedo on the sailor top's emblem quad (a black
        rectangle on the chest) and had no way at all to reach set 3.

        The two emblem renderers take neither: ``ChangeCustomEmblem`` gives
        them the card's ``emblemeId`` texture out of CategoryNo 431.

        All three ct's get the *same* three colours, so a renderer's texture
        set says nothing about which ``colorInfo`` entries it reads.
        """
        out: dict = {}
        try:
            gt = root.read_typetree()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader.clothes_roles: gt = root.read_typetree()", exc)
            return out
        for c in gt.get("m_Component", []):
            o = self._deref(c)
            if o is None or o.type.name != "MonoBehaviour":
                continue
            try:
                tt = o.read_typetree()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader.clothes_roles: o.read_typetree", exc)
                continue
            if not all(k in tt for k in self._CLOTHES_FIELDS):
                continue
            for key, role in (("rendNormal01", "normal01"),
                              ("rendNormal02", "normal02"),
                              ("rendNormal03", "normal03"),
                              ("rendAlpha01", "alpha01"),
                              ("rendAlpha02", "alpha02"),
                              ("rendAccessory", "accessory"),
                              # ``ChangeCustomEmblem(kind, no)`` switches on
                              # *no*, not on the field number: no 0 writes
                              # rendEmblem01, rendEmblem02 **and**
                              # exRendEmblem01, and only exRendEmblem02 takes
                              # no 1 (the card's ``emblemeId2``).  So the role
                              # is named after the id it reads.
                              ("rendEmblem01", "emblem0"),
                              ("exRendEmblem01", "emblem0"),
                              ("rendEmblem02", "emblem0"),
                              ("exRendEmblem02", "emblem1")):
                # single Renderer fields and Renderer[] fields both appear
                val = tt.get(key)
                refs = val if isinstance(val, list) else [val]
                for ref in refs:
                    pid = (ref or {}).get("m_PathID") if isinstance(ref, dict) else 0
                    if pid:
                        out[pid] = role
            break
        return out

    def clothes_options(self, root) -> dict:
        """``GameObject path_id -> 1 | 2`` from ``ChaClothesComponent.objOpt01/02``.

        ``ChaControl.DrawOption(kind)`` switches these two groups with
        ``SetActive(hideOpt[0] == 0)`` / ``SetActive(hideOpt[1] == 0)`` —
        ``hideBraOpt`` for the bra slot, ``hideShortsOpt`` for the panty, the
        slot's own ``hideOpt`` for every other (§7.379).  A renderer belongs to
        an option when its own GameObject **or an ancestor** is one of them
        (``SetActive`` hides the subtree).  43 of the 629 base clothes
        components name any.
        """
        return self._clothes_groups(root, ("objOpt01", "objOpt02"))

    def clothes_sleeves(self, root) -> dict:
        """``GameObject path_id -> 1 | 2 | 3`` from ``objSleeves01/02/03``.

        ``ChaControl.DrawSode``, which ``UpdateVisible`` calls right after
        ``DrawOption(0)``, switches the three groups of the **top** slot's
        component (``cusClothesCmp[0]``) with ``SetActiveControl(obj,
        sleevesType == N - 1)``: exactly one sleeve length is drawn, the one
        the card's ``parts[0].sleevesType`` names (``MemberInit`` 0).
        """
        return self._clothes_groups(
            root, ("objSleeves01", "objSleeves02", "objSleeves03"))

    def _clothes_groups(self, root, keys: tuple) -> dict:
        """``GameObject path_id -> i + 1`` for the ``GameObject[]`` fields *keys*
        of the root's ``ChaClothesComponent`` (empty when it has none)."""
        out: dict = {}
        try:
            gt = root.read_typetree()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader._clothes_groups: gt = root.read_typetree()", exc)
            return out
        for c in gt.get("m_Component", []):
            o = self._deref(c)
            if o is None or o.type.name != "MonoBehaviour":
                continue
            try:
                tt = o.read_typetree()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader._clothes_groups: o.read_typetree", exc)
                continue
            if not all(k in tt for k in self._CLOTHES_FIELDS):
                continue
            for k, key in enumerate(keys, 1):
                for ref in tt.get(key) or []:
                    pid = (ref or {}).get("m_PathID") if isinstance(ref, dict) else 0
                    if pid:
                        out[pid] = k
            break
        return out

    # The tuning ``EyeLookMaterialControll.Update`` reads.  Every one of these
    # is serialised on the iris renderer's own GameObject and **none of it is
    # reset at run time**: ``Initialize`` writes a different set of waits
    # (156/100/156/100) but nothing calls it -- neither ``Initialize`` nor the
    # ``initializeL`` / ``initializeR`` flags beside it are referenced anywhere
    # in ``Assembly-CSharp``.  ``Start`` only fills in the shader property ids.
    # ``YureTime`` is how often ``Update`` re-rolls the highlights' shake
    # (``ChangeEyesShaking``, the H face rows' ``yure``, §7.403); the four
    # ``Yure*`` ints beside it are serialised but ``Update`` never reads them.
    _EYELOOK_FIELDS = ("InsideWait", "OutsideWait", "UpWait", "DownWait",
                       "InsideLimit", "OutsideLimit", "UpLimit", "DownLimit",
                       "power", "eyeLR", "YureTime")

    def face_blend_shape(self, root) -> dict:
        """The head-wide half of ``FaceBlendShape``, off the prefab root.

        Only the three gaze corrections are taken.  ``LateUpdate`` folds them
        into how far open the lids sit::

            hi = min(1 - EyeLookUpCorrect, EyesCtrl.OpenMax)
            c  = (v > 0 ? LerpAccel(0, Up, v) : -LerpAccel(0, Down, -v))
               - LerpAccel(0, Side, |h|)
            EyesCtrl.SetCorrectOpenMax(hi + Clamp(c, -max(Down, Side), Up)
                                            * EyesCtrl.OpenMax)

        so with the gaze at rest the lids sit at ``hi`` and nothing else in the
        component matters.  ``EyeLookUpCorrect`` is 0 on the stock female head
        and 0.1 on the male one, which is a real 2% difference in openness.
        """
        out: dict = {}
        try:
            gt = root.read_typetree()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader.face_blend_shape: gt = root.read_typetree()", exc)
            return out
        for c in gt.get("m_Component", []):
            o = self._deref(c)
            if o is None or o.type.name != "MonoBehaviour":
                continue
            try:
                tt = o.read_typetree()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader.face_blend_shape: o.read_typetree", exc)
                continue
            if "EyeLookUpCorrect" not in tt:
                continue
            for key, name in (("EyeLookUpCorrect", "upCorrect"),
                              ("EyeLookDownCorrect", "downCorrect"),
                              ("EyeLookSideCorrect", "sideCorrect")):
                try:
                    out[name] = float(tt.get(key) or 0.0)
                except (TypeError, ValueError):
                    pass
            eyes = tt.get("EyesCtrl") or {}
            for key, name in (("OpenMin", "openMin"), ("OpenMax", "openMax"),
                              ("FixedRate", "fixedRate")):
                try:
                    out[name] = float(eyes.get(key))
                except (TypeError, ValueError):
                    pass
            # The blink rhythm and whether the brow follows it (§7.351).  Both
            # are serialised per head: 532 of the 533 components on this
            # install's head mods carry the constructor's 30 / 0.15 / true and
            # one carries 60 / 0.3, so they are read, not assumed.
            blink = tt.get("BlinkCtrl") or {}
            try:
                out["blinkFrequency"] = int(blink["BlinkFrequency"])
                out["blinkSpeed"] = float(blink["BaseSpeed"])
            except (KeyError, TypeError, ValueError):
                pass
            brow = tt.get("EyebrowCtrl") or {}
            if "SyncBlink" in brow:
                out["eyebrowSyncBlink"] = bool(brow.get("SyncBlink"))
            break
        return out

    def _eye_look(self, game_object) -> dict | None:
        """``EyeLookMaterialControll`` on this renderer's GameObject, if any.

        Reading it rather than hard-coding it matters: 46 of the 1,068
        components in the locally installed head mods differ from the stock
        values (waits down to ±20, limits down to ±30), and the waits are what
        turn the card's ``pupilX`` / ``pupilY`` into the iris map's ST.
        """
        try:
            gt = game_object.read_typetree()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader._eye_look: gt = game_object.read_typetree()", exc)
            return None
        for c in gt.get("m_Component", []):
            o = self._deref(c)
            if o is None or o.type.name != "MonoBehaviour":
                continue
            try:
                tt = o.read_typetree()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader._eye_look: o.read_typetree", exc)
                continue
            if "InsideWait" not in tt or "isYure" in tt:
                continue
            out = {k: tt.get(k) for k in self._EYELOOK_FIELDS}
            states = tt.get("texStates") or []
            out["isYure"] = [int((s or {}).get("isYure", 0) or 0)
                             for s in states]
            return out
        return None

    # ------------------------------------------------------------------
    # sway bones
    # ------------------------------------------------------------------
    def dynamic_bones(self) -> dict:
        """``m_Root`` path_id → its ``DynamicBone`` parameters.

        Illusion drives skirts, sleeves, ribbons and long hair with Will Hong's
        ``DynamicBone``.  UnityPy returns nothing for these objects — the
        script has no type tree — so :func:`dynbone.parse_component`
        deserialises them by the field order in ``Assembly-CSharp``.  A parse
        that does not consume the object's bytes exactly is rejected, which is
        what keeps this from quietly mis-reading an unrelated MonoBehaviour:
        of the 158 components in ``co_bot_00`` exactly the 96 DynamicBones fit.
        """
        if self._dynbones is None:
            from . import dynbone as DB
            out: dict = {}
            for o in self.env.objects:
                if o.type.name != "MonoBehaviour":
                    continue
                try:
                    spec = DB.parse_component(bytes(o.get_raw_data()))
                except Exception as exc:             # noqa: BLE001
                    swallowed.note("origin.prefab.PrefabLoader.dynamic_bones: o.get_raw_data", exc)
                    continue
                if spec is None:
                    continue
                pid = spec.root[1] if spec.root else 0
                if pid:
                    out[pid] = spec
            self._dynbones = out
        return self._dynbones

    def _colliders_for(self, spec, worlds: dict) -> list:
        """The colliders this chain is solved against.

        A prefab's own ``m_Colliders`` is never the answer: all 96
        ``DynamicBone`` components of ``co_bot_00`` carry the field and every
        entry is null, and so are all 75 entries across ``p_cf_hair_f_31``'s
        five chains.  The list is filled in at load time —
        ``<LoadCharaFbxDataAsync>c__Iterator13.MoveNext`` does

            if (copyDynamicBone) {
                …
                db.m_Colliders.Clear();
                foreach (c in dbc) db.m_Colliders.Add(c);
            }

        where ``dbc`` is ``objBodyBone.GetComponentsInChildren<DynamicBoneCollider>
        (true)``.  That method is the **only** writer of ``m_Colliders`` in
        ``Assembly-CSharp`` (the other two references are the two solvers
        reading it), so whatever the flag does not fill stays null.

        **And only the clothes loaders pass the flag.**  Read off the four call
        sites, ``copyDynamicBone`` is ``true`` from ``<ChangeClothes*Async>``
        and ``false`` from ``<ChangeHairAsync>``, ``<ChangeHeadAsync>`` and
        ``<ChangeAccessoryAsync>`` — so in the game **hair, the head and
        accessories are solved against nothing at all**.  Handing them the
        body's set, ``cf_hit_head`` enlarged 1.2× included, was ours: a chain
        that the game lets pass through the skull would get pushed off it here.
        Measured on four cards it changes nothing today (only the skirt moves,
        by 5.2 mm, and the skirt is clothing), because the static settle never
        brings a hair chain into contact — but it is not ours to decide, and it
        stops being inert the moment the solver runs per frame.
        """
        if not self.copy_dynamic_bone:
            return []
        if getattr(spec, "colliders", None) is None:
            return []
        return list(body_colliders())

    def _child_pids(self, pid: int) -> list:
        """A transform's children in order, every one (``childCount`` counts them all)."""
        cache = self.__dict__.setdefault("_kids", {})
        hit = cache.get(pid)
        if hit is None:
            t = self.by_id.get(pid)
            tt = t.read_typetree() if t is not None else {}
            hit = [c.get("m_PathID") for c in tt.get("m_Children", []) if (c or {}).get("m_PathID")]
            cache[pid] = hit
        return hit

    def _parent_pid(self, pid: int):
        t = self.by_id.get(pid)
        if t is None:
            return None
        return (t.read_typetree().get("m_Father") or {}).get("m_PathID") or None

    def _go_transform(self, worlds: dict, go) -> int:
        """The transform path_id of a GameObject PPtr, 0 when it is not in *worlds*."""
        cached = self.__dict__.get("_go_tf")
        if cached is None or cached[0] is not worlds:
            table = {}
            for pid in worlds:
                t = self.by_id.get(pid)
                if t is not None:
                    table[(t.read_typetree().get("m_GameObject") or {}).get("m_PathID")] = pid
            cached = self._go_tf = (worlds, table)
        return cached[1].get((go or (0, 0))[1], 0)

    def _sway_trees(self, worlds: dict) -> list:
        """Every ``DynamicBone`` under this prefab, as ``AppendParticles`` builds it.

        ``[(spec, nodes, parents, end_offsets, bone_length, scale, problem)]``:
        *nodes* are transform path_ids, ``None`` for an end particle; *scale*
        is ``m_ObjectScale`` (the component's own ``lossyScale.x``).  Read once
        per ``worlds`` (:meth:`sway_rest` and :meth:`sway_chain_specs` both
        need it).
        """
        cached = self.__dict__.get("_trees")
        if cached is not None and cached[0] is worlds:
            return cached[1]
        from . import dynbone as DB
        out = []
        for root_pid, spec in self.dynamic_bones().items():
            if root_pid not in worlds:
                continue
            comp = self._go_transform(worlds, spec.game_object)
            comp_world = worlds.get(comp) if comp else None
            problem = ""
            if comp_world is None:
                problem = "讀不到搖擺骨元件自己的 Transform"
                nodes, parents, ends, lengths = [root_pid], [-1], np.zeros((1, 3)), np.zeros(1)
            else:
                try:
                    nodes, parents, ends, lengths, problem = DB.particle_tree(
                        root_pid, self._child_pids, lambda pid: worlds[pid], self._parent_pid,
                        spec, excluded={pp[1] for pp in spec.exclusions or () if pp and pp[1]},
                        not_rolls={pp[1] for pp in spec.not_rolls or () if pp and pp[1]},
                        component_world=comp_world)
                except KeyError as exc:
                    problem = f"鏈上有讀不到的 Transform（{exc}）"
                    nodes, parents, ends, lengths = [root_pid], [-1], np.zeros((1, 3)), np.zeros(1)
            scale = float(np.linalg.norm(np.asarray(comp_world)[:3, 0])) if comp_world is not None else 1.0
            out.append((spec, nodes, parents, ends, lengths, scale, problem))
        self._trees = (worlds, out)
        return out

    @staticmethod
    def _tree_frames(nodes: list, parents: list, worlds: dict) -> np.ndarray:
        """Each particle's transform; an end particle's is its leaf's."""
        frames: list = []
        for node, par in zip(nodes, parents):
            frames.append(frames[par] if node is None else np.asarray(worlds[node], dtype=np.float64))
        return np.array(frames)

    def _tree_chain(self, spec, nodes, parents, ends, lengths, worlds) -> dict:
        """One tree as :class:`dynbone.Tree` and the bake take it."""
        from . import dynbone as DB
        dist = DB.distribute(spec, lengths)
        chain = {
            "bones": ["" if n is None else self._go_name(n) for n in nodes],
            "parent": [int(x) for x in parents],
            "end": [n is None for n in nodes],
            "end_offset": np.asarray(ends, dtype=np.float32),
            # whether the game hands this tree the body's colliders at all
            # (``copyDynamicBone``: clothes yes, hair and accessories no) —
            # :meth:`_colliders_for`, the same answer the settle gets
            "collide": bool(self._colliders_for(spec, worlds)),
            "gravity": np.asarray(spec.gravity, dtype=np.float32),
            "force": np.asarray(spec.force, dtype=np.float32),
            "freeze_axis": int(getattr(spec, "freeze_axis", 0) or 0),
            # ``m_UpdateRate``: how many steps a frame's time holds (``UpdateDynamicBones``, §7.418)
            "update_rate": float(spec.update_rate),
        }
        for k, v in dist.items():
            chain[k] = v.astype(np.float32)
        return chain

    def sway_rest(self, worlds: dict) -> dict:
        """path_id → the world matrix every bone under a sway chain hangs in.

        :class:`dynbone.Tree` on the particles ``AppendParticles`` makes, run to
        rest with the prefab held in its own rest pose — the same solver the
        run time steps every frame (``figure.sway_live``).  A particle's bone
        takes the turn and the position ``ApplyParticlesToTransforms`` gives
        it; a bone that is not a particle (``m_notRolls``, ``m_Exclusions``)
        rides on its parent, which is all the game does to it.
        """
        from . import dynbone as DB
        out: dict = {}
        for spec, nodes, parents, ends, lengths, scale, problem in self._sway_trees(worlds):
            if problem or len(nodes) < 2:
                continue
            chain = self._tree_chain(spec, nodes, parents, ends, lengths, worlds)
            T = self._tree_frames(nodes, parents, worlds)
            single = [n is None or len(self._child_pids(n)) <= 1 for n in nodes]
            tree = DB.Tree([chain], single, T)
            tree.settle(T, DB.PlacedColliders(self._colliders_for(spec, worlds)), scale)
            solved, _gaps = tree.turn(T)
            R = DB.rotations(T)
            new: dict = {}
            for k, pid in enumerate(nodes):
                if pid is None:
                    continue
                m = np.asarray(worlds[pid], dtype=np.float64).copy()
                m[:3, :3] = solved[k] @ R[k].T @ m[:3, :3]
                if parents[k] >= 0:
                    m[:3, 3] = tree.pos[k]
                new[pid] = m
            self._carry(nodes[0], worlds, new)
            out.update(new)
        return out

    def _carry(self, root: int, worlds: dict, new: dict) -> None:
        """Hang every bone under *root* that is not in *new* on its moved parent."""
        stack = [root]
        while stack:
            pid = stack.pop()
            for c in self._child_pids(pid):
                if c not in worlds:
                    continue
                if c not in new and pid in new:
                    new[c] = new[pid] @ np.linalg.inv(np.asarray(worlds[pid], dtype=np.float64)) \
                        @ np.asarray(worlds[c], dtype=np.float64)
                stack.append(c)

    def sway_local_deltas(self, worlds: dict, settled: dict) -> dict:
        """bone name → the rotation the solve added, in that bone's parent frame.

        A skirt's sway bones are ``cf_j_sk_NN_MM``, which the body armature has
        too, so the garment is retargeted onto *our* skeleton and never goes
        through the prefab-rest path.  Settling its own copy of the chain would
        therefore change nothing — the shape has to be pushed onto the bones the
        mesh is actually skinned to, which is what the game does when its
        ``DynamicBone`` writes back to the character's transforms.
        """
        out: dict = {}
        if not settled:
            return out
        parent_of: dict = {}
        for pid in settled:
            t = self.by_id.get(pid)
            if t is None:
                continue
            try:
                tt = t.read_typetree()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab.PrefabLoader.sway_local_deltas: t.read_typetree", exc)
                continue
            parent_of[pid] = ((tt.get("m_Father") or {}).get("m_PathID") or 0)
        for pid, new in settled.items():
            old = worlds.get(pid)
            ppid = parent_of.get(pid, 0)
            if old is None or ppid not in worlds:
                continue
            p_old, p_new = worlds[ppid], settled.get(ppid, worlds[ppid])
            a = _orthonormal(np.asarray(p_old)[:3, :3]).T @ _orthonormal(
                np.asarray(old)[:3, :3])
            b = _orthonormal(np.asarray(p_new)[:3, :3]).T @ _orthonormal(
                np.asarray(new)[:3, :3])
            t = self.by_id.get(pid)
            tt = t.read_typetree()
            go = self.by_id.get((tt.get("m_GameObject") or {}).get("m_PathID"))
            name = self._name(go) if go is not None else ""
            if name:
                out[name] = a.T @ b          # a⁻¹ b, both orthonormal
        return out

    def _go_name(self, pid: int) -> str:
        """The GameObject name behind a Transform path_id ("" if unreadable)."""
        t = self.by_id.get(pid)
        if t is None:
            return ""
        try:
            tt = t.read_typetree()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader._go_name: tt = t.read_typetree()", exc)
            return ""
        go = self.by_id.get((tt.get("m_GameObject") or {}).get("m_PathID"))
        return self._name(go) if go is not None else ""

    def sway_chain_specs(self, worlds: dict, objects: bool = False) -> list:
        """Per-tree solver parameters, in a form that can be baked.

        *objects*: also name each tree's component object (``object``: the
        GameObject the ``DynamicBone`` sits on, whose travel and scale are its
        ``m_ObjectMove`` / ``m_ObjectScale``) — for a body pack's own chains, whose
        component is on one of the pack's bones (the run time keys hair on the
        tag, so garments and hair, whose objects the dress names, go without).

        ``sway_rest`` solves these trees **once, in the prefab's own rest
        pose**, and the result is baked as a fixed shape.  The game instead
        runs the solver every ``LateUpdate`` against the character's *current*
        pose, which is why a skirt gets out of the way of a raised thigh (§7.118).
        To do that at run time the parameters have to travel in our own format,
        because ``dynamic_bones()`` reads them out of a ``.unity3d`` — which the
        run time is not allowed to open (``CLAUDE.md`` 四之一).

        Stored per tree: the particle bones' **names** ("" for an end particle;
        the body armature owns a skirt's ``cf_j_sk_*``, so the reader looks them
        up on the skeleton), the parent table, the end particles' offsets and
        the **already-distributed** per-particle values — what ``SetupParticles``
        computes, so no Unity curve goes into the bake.  A tree the solver does
        not take carries ``unsolved`` and the reason, for the dress to report.
        """
        out: list = []
        for spec, nodes, parents, ends, lengths, _scale, problem in self._sway_trees(worlds):
            if len(nodes) < 2 and not problem:
                continue
            names = ["" if n is None else self._go_name(n) for n in nodes]
            if not problem and any(n is not None and not nm for n, nm in zip(nodes, names)):
                problem = "鏈上有讀不出名字的骨頭"
            if problem:
                out.append({"bones": names, "unsolved": problem})
                continue
            chain = self._tree_chain(spec, nodes, parents, ends, lengths, worlds)
            if objects:
                comp = self._go_transform(worlds, spec.game_object)
                chain["object"] = self._go_name(comp) if comp else ""
            out.append(chain)
        return out

    def _collect_worlds(self, t, parent, out: dict,
                        seen: set | None = None) -> None:
        """Every Transform's world matrix within the prefab, by path_id.

        **Every** one: the walk is bounded by a visited set (:meth:`_children`),
        not by a depth ceiling.  Three callers used to raise the old ceiling of
        12 to 64 because the body armature is 20 deep, and the one that did not
        — :meth:`build_skinned`, the path every garment, hair and accessory
        goes through — lost a third of its bones without saying so (§7.96).
        """
        if seen is None:
            seen = {t.path_id}
        tt = t.read_typetree()
        world = parent @ _trs(tt)
        out[t.path_id] = world
        for co in self._children(tt, seen):
            self._collect_worlds(co, world, out, seen)

    def move_chain(self, troot):
        """(per-move-node ``(parent_world, node_world)``, path_id → bitmask).

        ``ChaControl.UpdateAccessoryMoveFromInfo`` writes the card's ``addMove``
        into those two nodes with ``set_localPosition`` / ``EulerAngles`` /
        ``Scale`` — it **replaces** their local TRS in the middle of the
        hierarchy rather than multiplying anything onto the finished item.  The
        world matrix the game ends up with is therefore

            world = A · move0 · B · move1 · C

        for the three prefab-local stretches A (root → ``N_move``'s parent),
        B (``N_move`` → ``N_move2``'s parent) and C (``N_move2`` → the mesh),
        and what this returns is what a reader needs to rebuild it: with
        ``P`` the node's parent world and ``W`` the node's own world, replacing
        that node's local by ``M`` is the world-space correction
        ``P · M · W⁻¹`` — see :class:`figure.attach.AccessoryMove`.

        Two consequences that are not obvious from the outside:

        * the prefab's own local TRS on those nodes is **discarded**, because a
          card with no manual offset still writes identity.  Six of the 357 base
          accessories author a non-identity one (millimetres, so nothing here
          hinges on it, but it is not ours to keep).
        * the chain *above* ``N_move`` is not identity on 131 of those 357 — 41
          carry a scale (``p_acs_valletta_heart1``'s ``nc`` is 1.5×,
          ``p_acs_ear_weddingshizuku`` 5×) and 50 a rotation.  Multiplying
          ``addMove`` onto the outside of the finished item, which is what this
          used to do, drops exactly that factor: the card's −2 cm becomes −3 cm
          in the game and stayed −2 cm here (§7.66 二).
        """
        found: dict[str, tuple] = {}
        mask_of: dict[int, int] = {}
        seen = {troot.path_id}

        def walk(t, parent_world, mask):
            tt = t.read_typetree()
            world = parent_world @ _trs(tt)
            go = self.by_id.get((tt.get("m_GameObject") or {}).get("m_PathID"))
            name = self._name(go) if go is not None else ""
            for bit, nm in enumerate(MOVE_NODES):
                # only the first hit in pre-order is the one FindLoop returns;
                # a second node of the same name is just an ordinary child
                if name == nm and nm not in found:
                    found[nm] = (parent_world, world)
                    mask |= 1 << bit
            mask_of[t.path_id] = mask
            # recursion, not a stack: ``FindLoop`` is pre-order and the mask is
            # inherited down a branch, so both the visit order and the parent's
            # accumulated bits have to be the ones this node was reached by
            for co in self._children(tt, seen):
                walk(co, world, mask)

        walk(troot, m3.identity(), 0)
        return [found.get(nm) for nm in MOVE_NODES], mask_of

    def length_chain(self, troot, trf_pids) -> tuple:
        """``(groups, path_id → bitmask)`` for ``ChaCustomHairComponent.trfLength``.

        ``Update`` sets ``trfLength[i].localPosition.y = baseLength[i] +
        Mathf.Lerp(0, addLength, lengthRate)`` every frame, and
        ``ChangeSettingHairLength`` gives a hair slot the card's
        ``hair.parts[slot].length`` as ``lengthRate`` (§7.516).  Replacing a
        local y moves the node and its subtree by the parent's world rotation
        and scale applied to ``(0, Δy, 0)``, so each group is that direction
        per unit y and the node's own rest y (``Δy`` is the target minus it);
        the bitmask says which groups a node sits in or under (bit *i* =
        ``trfLength[i]``), the same shape as :meth:`move_chain`.

        ``dyn`` marks a node that is an enabled ``DynamicBone``'s particle (its
        ``m_Root`` or under it, not cut off by ``m_Exclusions``):
        ``DynamicBone.Update`` runs ``InitTransforms`` — every particle's
        ``localPosition`` back to the one ``SetupParticles`` took at ``Start``,
        the serialised one — and its execution order (12250) is after this
        component's (0), so the write never reaches the chain.  None of the
        stock hair's 392 ``trfLength`` nodes is one (368 sit above a chain's
        root, 24 on no chain).
        """
        want = {pid: i for i, pid in enumerate(trf_pids or ()) if pid}
        groups: list = [None] * len(trf_pids or ())
        mask_of: dict[int, int] = {}
        seen = {troot.path_id}
        chains = {pid: {p[1] for p in (spec.exclusions or ()) if p and p[1]}
                  for pid, spec in self.dynamic_bones().items() if spec.enabled}

        def walk(t, parent_world, mask, active):
            tt = t.read_typetree()
            local = _trs(tt)
            world = parent_world @ local
            # AppendParticles skips an excluded child and its subtree, chain by chain
            active = tuple(ex for ex in active if t.path_id not in ex)
            if t.path_id in chains:
                active += (chains[t.path_id],)
            i = want.get(t.path_id)
            if i is not None and groups[i] is None:
                groups[i] = {"dir": [float(v) for v in parent_world[:3, :3] @ (0.0, 1.0, 0.0)],
                             "y0": float(local[1, 3]), "dyn": bool(active)}
                mask |= 1 << i
            mask_of[t.path_id] = mask
            for co in self._children(tt, seen):
                walk(co, world, mask, active)

        walk(troot, m3.identity(), 0, ())
        return groups, mask_of

    def node_hierarchy(self, prefab_name: str) -> dict:
        """``name → (parent name, local 4×4)`` for every node in the prefab.

        The bones a renderer is skinned to are only the nodes that *carry
        vertices*.  A prefab's hierarchy also holds the ones nothing is weighted
        to, and on a head those are the ``a_n_*`` attachment points — the frames
        every hat, hairpin, earring and pair of glasses is parented to.  They
        are part of the head the card wears, not of some common rig: the twelve
        of them sit up to **92.6 mm** apart between ``oo_base``'s generic
        ``p_cf_head_bone`` and the head list's own ``p_cf_head_00``, and on the
        200/201 heads twenty-seven ``cf_J_*`` joints move as well (§7.102).

        Parents by name rather than by path_id so the table survives the bake,
        which is where a reader gets it from — the runtime does not open
        bundles (CLAUDE.md 四).
        """
        root = self.find_prefab(prefab_name)
        troot = self._transform_of(root) if root is not None else None
        if troot is None:
            return {}
        out: dict[str, tuple[str, np.ndarray]] = {}
        seen = {troot.path_id}

        def walk(t, parent_name: str):
            tt = t.read_typetree()
            go = self.by_id.get((tt.get("m_GameObject") or {}).get("m_PathID"))
            name = self._name(go) if go is not None else ""
            # First name wins, same rule the reference rig walks under: a
            # prefab that ships two nodes of one name has no answer to "which
            # one", and picking the later one silently would be a lottery.
            if name and name not in out:
                out[name] = (parent_name, _trs(tt))
            for co in self._children(tt, seen):
                walk(co, name or parent_name)

        walk(troot, "")
        return out

    def build_skinned(self, prefab_name: str,
                      want_textures: bool = True) -> list["Piece"]:
        """Like ``build`` but keeps each renderer's skin data and albedo.

        ``skin`` is None for renderers that carry no weights (rigid props).
        Every node of the prefab is read: Origin filters none by name (§7.500).
        """
        root = self.find_prefab(prefab_name)
        if root is None:
            return []
        troot = self._transform_of(root)
        if troot is None:
            return []
        results = []
        worlds: dict = {}
        self._collect_worlds(troot, m3.identity(), worlds)
        self.last_moves, move_mask = self.move_chain(troot)
        acc = self.accessory_roles(root)
        cloth = self.clothes_roles(root)
        hair_len = acc.get("hair_len")
        len_groups, len_mask = (self.length_chain(troot, hair_len["trf"])
                                if hair_len else ([], {}))
        options = self.clothes_options(root)
        sleeves = self.clothes_sleeves(root)
        fbs = self.face_blend_shape(root)
        try:
            settled = self.sway_rest(worlds)
            self.last_sway = self.sway_local_deltas(worlds, settled)
            self.last_sway_spec = self.sway_chain_specs(worlds)
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.PrefabLoader.build_skinned: settled = self.sway_rest(worlds)", exc)
            settled, self.last_sway, self.last_sway_spec = {}, {}, []

        walked = {troot.path_id}

        def walk(t, parent, opt=0, state="", sleeve=0, off=""):
            tt = t.read_typetree()
            world = parent @ _trs(tt)
            go_pid = (tt.get("m_GameObject") or {}).get("m_PathID")
            go = self.by_id.get(go_pid)
            opt = options.get(go_pid, opt)
            sleeve = sleeves.get(go_pid, sleeve)
            gt = go.read_typetree() if go is not None else None
            if gt is not None and gt.get("m_Name", "") in CLOTHES_STATE_NODES:
                state = gt["m_Name"]
            # Unity draws a renderer only while its GameObject and every
            # ancestor are active (§7.509).  The garment switches above are
            # the nodes Origin's ``UpdateVisible`` turns on and off at run time,
            # and so is the prefab's own root (``objClothes`` / ``objHair`` /
            # ``objAccessory[i]``); an inactive node that is none of them
            # stays off in the game.
            own_off = (not off and gt is not None and t.path_id != troot.path_id
                       and not gt.get("m_IsActive", 1)
                       and gt.get("m_Name", "") not in CLOTHES_STATE_NODES
                       and go_pid not in options and go_pid not in sleeves)
            if own_off:
                off = str(gt.get("m_Name", "")) or "?"
            if go is not None:
                node = gt.get("m_Name", "")
                # A rigid object splits the two halves across siblings: the
                # mesh hangs off ``MeshFilter`` and the materials off
                # ``MeshRenderer``.  Asking the MeshFilter for materials
                # returns nothing, which is why every rigid accessory —
                # glasses, hairpins, bags — arrived with no texture, no
                # colours and no shader settings and drew as a flat slab.
                rigid_mats = None
                for c in gt.get("m_Component", []):
                    ro = self._deref(c)
                    if ro is not None and ro.type.name == "MeshRenderer":
                        rigid_mats = ro
                        break
                for c in gt.get("m_Component", []):
                    o = self._deref(c)
                    if o is None or o.type.name not in (
                            "SkinnedMeshRenderer", "MeshFilter"):
                        continue
                    mat_src = o if o.type.name == "SkinnedMeshRenderer"                             else (rigid_mats if rigid_mats is not None else o)
                    rt = o.read_typetree()
                    mo = self.by_id.get((rt.get("m_Mesh") or {}).get("m_PathID"))
                    if mo is None or mo.type.name != "Mesh":
                        continue
                    mesh = self._mesh_from_object(mo)
                    if mesh is None or mesh.vertex_count == 0:
                        continue
                    bones, skin, binds, bworld = [], None, None, None
                    brest, bmove = None, None
                    if o.type.name == "SkinnedMeshRenderer":
                        bones, bworld, brest, bmove = self._renderer_bones(
                            o, worlds, settled, move_mask)
                        mt = mo.read_typetree()
                        skin = self._skin_array(mt)
                        binds = self._bind_poses(mt)
                        if skin is not None and len(skin) != mesh.vertex_count:
                            skin = None      # export order didn't line up
                    if skin is None:
                        mesh.transform(world)
                    # A renderer with more materials than submeshes draws
                    # the same geometry once per extra material — Unity's
                    # own rule, and how the eyelid shadow is done:
                    # ``cf_O_eyeline`` has one submesh and two materials
                    # (``cf_m_eyeline_00_up`` + ``cf_m_eyeline_kage``).
                    # Reading only ``m_Materials[0]`` dropped that pass.
                    shapes = self._blend_shapes(mo.read_typetree())
                    slots = (self._material_slots(mat_src)
                             if want_textures else [0])
                    role = acc["roles"].get(getattr(mat_src, "path_id", 0), "")
                    # ``ChangeAccessoryColor`` switches a ``rendAlpha``
                    # renderer's own GameObject (on while colour 4's alpha is
                    # not 0), so that one inactive node is Origin's to decide.
                    hidden_by = "" if own_off and role == "alpha" else off
                    try:
                        rend = mat_src.read_typetree()
                        enabled = bool(rend.get("m_Enabled", 1))
                        cast = int(rend.get("m_CastShadows", 1))
                    except Exception as exc:         # noqa: BLE001
                        swallowed.note("origin.prefab.PrefabLoader.build_skinned: m_Enabled", exc)
                        enabled, cast = True, 1
                    eyelook = self._eye_look(go)
                    for pass_i, si in enumerate(slots):
                        m_i = self.submesh_piece(mo, mesh, si)
                        if m_i is mesh and pass_i:
                            m_i = mesh.copy()
                        sh = (self._material_floats(mat_src, si)
                              if want_textures else None)
                        if sh is not None and acc["roles"]:
                            # travels with the material dict so the bake
                            # keeps it without a schema of its own
                            sh["_accRole"] = role
                            sh["_useColor"] = list(acc["use"])
                            sh["_accOutline"] = 1 if acc["outline"] else 0
                        if sh is not None and (acc["roles"]
                                               or acc["hair_comp"] is not None):
                            # ``ChaCustomHairComponent``'s list this renderer
                            # is on: "hair" / "accessory" / "" (neither, or
                            # no such component).  The key's *absence*
                            # marks an archive baked before it was read
                            # (§7.446).
                            sh["_hairComp"] = (acc["hair_comp"] or {}).get(
                                getattr(mat_src, "path_id", 0), "")
                            sh["_hairAcsN"] = int(acc["hair_acs_n"])
                        if sh is not None and cloth:
                            sh["_clothRole"] = cloth.get(
                                getattr(mat_src, "path_id", 0), "")
                            # 0 = always drawn, 1 / 2 = the objOpt group
                            # the card's hideOpt byte switches (§7.379)
                            sh["_clothOpt"] = int(opt)
                            # the state node this renderer hangs under
                            # (``CLOTHES_STATE_NODES``), "" for none
                            sh["_stateNode"] = state
                            # 0 = always drawn, 1 / 2 / 3 = the objSleeves
                            # group ``DrawSode`` shows for sleevesType 0 / 1 / 2
                            sh["_clothSleeve"] = int(sleeve)
                        if sh is not None and hair_len is not None:
                            # every piece of an item with the component, so
                            # its presence also says "baked after §7.516"
                            sh["_hairLen"] = {
                                "base": list(hair_len["base"]),
                                "add": hair_len["add"], "rate": hair_len["rate"],
                                "on": hair_len.get("on", True), "groups": len_groups}
                            if o.type.name == "SkinnedMeshRenderer":
                                bits = [int(len_mask.get((b or {}).get("m_PathID"), 0))
                                        for b in (rt.get("m_Bones") or [])]
                                if any(bits):
                                    sh["_lenBones"] = bits
                            elif len_mask.get(t.path_id, 0):
                                sh["_lenMask"] = int(len_mask[t.path_id])
                        if sh is not None and hidden_by:
                            # the inactive node nothing in Origin switches on:
                            # Unity never draws this renderer (§7.509)
                            sh["_offNode"] = hidden_by
                        if sh is not None and not enabled:
                            # ``Renderer.enabled`` false in the prefab; only a
                            # MaterialEditor "Enabled 1" row turns it on
                            sh["_rendOff"] = 1
                        if sh is not None and cast != 1:
                            # ``ShadowCastingMode`` other than On: 0 casts none,
                            # 3 is only a shadow (#21 b); absent is On
                            sh["_castShadows"] = cast
                        if sh is not None:
                            # The renderer's own GameObject name, which is
                            # what ``Renderer.name`` returns at run time.
                            # MaterialEditor addresses renderers by it —
                            # ``MaterialAPI.SetRendererEnabled`` matches
                            # ``NameFormatted(r) == RendererName`` — and
                            # nothing else on this side carried it, so a
                            # card that switches a renderer off had no way
                            # to say which one (§7.138).  It rides in the
                            # material dict for the same reason
                            # ``_accRole`` and ``_clothRole`` do: the bake
                            # keeps it without a schema of its own.
                            sh["_rendName"] = str(node)
                            # How many submeshes the renderer's mesh has.
                            # A material appended past them draws the
                            # last one again (MaterialEditor's copies,
                            # §7.381), and a submesh no material draws is
                            # in no piece — this says whether it exists.
                            sh["_submeshes"] = len(self.submesh_tris.get(
                                int(getattr(mo, "path_id", 0) or 0)) or [0])
                        if sh is not None and eyelook is not None:
                            # same trick for the iris: the ST the eye map is
                            # sampled with is this component's arithmetic.
                            # ``FaceBlendShape`` lives on the prefab root
                            # and has no renderer of its own, so its share
                            # rides along here rather than inventing a
                            # second carrier for one float.
                            sh["_eyeLook"] = {**eyelook, **fbs}
                        # Each map arrives as a ``colorspace.Map``; the flags
                        # travel in ``map_cs`` keyed by the field they land
                        # in, because past this point the map is a PNG and
                        # ``m_ColorSpace`` is unrecoverable.
                        maps, cs, wraps = {}, {}, {}
                        glass = None
                        if want_textures and sh is not None and str(
                                sh.get("_shader") or "").endswith(
                                "toon_glasses_lod0"):
                            # read by value, never guessed; the state rides
                            # in the shader dict so a reader can tell "the
                            # material has none" from "baked before this
                            # was read" (§7.413)
                            glass, sh["_glassRamp"] = self._bound_texture(
                                mat_src, "_GlassRamp", si)
                        if want_textures:
                            for field, got in (
                                ("texture", self._main_texture(
                                    mat_src, si,
                                    guess=rigid_mats is None)),
                                ("mask", self._colour_mask(mat_src, si)),
                                ("detail", self._detail_map(mat_src, si)),
                                ("normal", self._material_texture(
                                    mat_src, "_NormalMap", "_n", si)),
                                ("alpha_mask", self._material_texture(
                                    mat_src, "_AlphaMask", "_am", si)),
                                ("ramp", self._material_texture(
                                    mat_src, "_AnotherRamp", "_ramp", si)),
                                ("line_mask", self._material_texture(
                                    mat_src, "_LineMask", "_line", si)),
                                ("glass_ramp", glass),
                            ):
                                if got is None:
                                    continue
                                maps[field] = got.px
                                cs[field] = got.space
                                if got.wrap is not None:
                                    wraps[field] = got.wrap
                        if sh is not None:
                            # 0 Repeat / 1 Clamp per map field (§7.383)
                            sh["_mapWrap"] = wraps
                        results.append(Piece(
                            m_i, bones, skin, binds,
                            maps.get("texture"),
                            maps.get("mask"),
                            bworld,
                            maps.get("detail"),
                            sh,
                            0,
                            maps.get("normal"),
                            maps.get("alpha_mask"),
                            maps.get("ramp"),
                            si,
                            shapes if si == 0 else None,
                            maps.get("line_mask"),
                            brest,
                            move_mask.get(t.path_id, 0),
                            bmove,
                            cs,
                            maps.get("glass_ramp")))
            for co in self._children(tt, walked):
                walk(co, world, opt, state, sleeve, off)

        walk(troot, m3.identity())
        return results

    def build(self, prefab_name: str, scale: float = 1.0) -> Mesh:
        """Merge every renderer under the prefab into one mesh (every node, §7.500)."""
        out = Mesh(prefab_name)
        for part in self.parts(prefab_name):
            mo = self.mesh_reader(part)
            m = self._mesh_from_object(mo) if mo is not None else None
            if m is None or m.vertex_count == 0:
                continue
            m.transform(part.matrix)
            if scale != 1.0:
                m.scale(scale)
            out.append(m)
        if out.vertex_count:
            out.compute_normals()
        return out


_loaders: "OrderedDict[str, PrefabLoader]" = OrderedDict()
MAX_LOADERS = 6          # each pins a decompressed bundle in memory


def loader(bundle_path: Path) -> PrefabLoader:
    key = str(bundle_path)
    L = _loaders.get(key)
    if L is not None:
        _loaders.move_to_end(key)
        return L
    while len(_loaders) >= MAX_LOADERS:
        _loaders.popitem(last=False)
    L = PrefabLoader(bundle_path)
    _loaders[key] = L
    return L


def clear_loaders() -> None:
    import gc
    _loaders.clear()
    gc.collect()


def text_asset_bytes(env, name: str) -> bytes | None:
    """The bytes of the TextAsset called *name* in a loaded bundle, or ``None``.

    Binary-safe: UnityPy hands a TextAsset's ``m_Script`` back as ``str``, and
    the shape tables are binary, so the round trip goes through
    ``surrogateescape`` — decoding it any other way replaces every byte above
    0x7F and destroys the float stream.  One owner for the three shape loaders
    that each carried a copy of this loop.
    """
    for obj in env.objects:
        if obj.type.name != "TextAsset":
            continue
        data = obj.read()
        if getattr(data, "m_Name", "") != name:
            continue
        raw = data.m_Script
        return (raw.encode("utf-8", "surrogateescape")
                if isinstance(raw, str) else bytes(raw))
    return None


# ---------------------------------------------------------------------------
# the character armature's own colliders
# ---------------------------------------------------------------------------
# ``ChaControl.<LoadAsync>`` instantiates ``hiPoly ? "p_cf_body_bone" :
# "p_cf_body_bone_low"`` from ``chara/oo_base.unity3d`` as ``objBodyBone``, and
# every garment loaded afterwards has its DynamicBone collider list replaced
# with everything hanging off it.  There is no male variant anywhere in the
# install — the same armature serves both sexes, exactly as the head does —
# and the low-poly one carries **no colliders at all**, so a low-detail
# character has no sway collision in the game either.
BODY_BONE_PREFAB = {True: "p_cf_body_bone", False: "p_cf_body_bone_low"}
BODY_BONE_BUNDLE = "chara/oo_base.unity3d"
_BODY_COLLIDERS: dict[tuple, tuple] = {}

# ``ChaReference.CreateReferenceInfo``'s body-bone group files
# ``RefObjKey.CORRECT_HEAD_DBCOL`` (80) under this name, and
# ``ChaControl.UpdateShapeFace`` gives it a uniform localScale — see
# :func:`body_colliders`.
HEAD_COLLIDER = "cf_hit_head"


def head_collider_scale(type_bone: bool) -> float:
    """The ``localScale`` ``UpdateShapeFace`` writes on :data:`HEAD_COLLIDER`."""
    return 1.08 if type_bone else 1.2

# The card's ``custom.body.typeBone``, parked here because the one caller that
# needs the colliders (``_colliders_for``, inside a garment's sway solve) has no
# card in reach.  ``dress_character`` sets it per character, the same way
# ``hi_poly`` is read from config rather than threaded through every call.
_TYPE_BONE = False


def set_type_bone(flag: bool) -> None:
    global _TYPE_BONE
    _TYPE_BONE = bool(flag)


def _script_name(loader_, obj) -> str:
    """The class behind a MonoBehaviour, or ``""``."""
    try:
        script = loader_.by_id.get(
            (obj.read_typetree().get("m_Script") or {}).get("m_PathID"))
        return "" if script is None else str(script.read().m_ClassName)
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.prefab._script_name: script = loader_.by_id.get((obj.read_typetree().", exc)
        return ""


def _collider_script(loader_, obj) -> bool:
    """Is this MonoBehaviour a ``DynamicBoneCollider``?"""
    return _script_name(loader_, obj) == "DynamicBoneCollider"


#: Returned by :func:`_baked_body_table` when the converter never wrote the table.
_NOT_BAKED = object()
#: Table names of the three armature reads in ``bake`` (§7.483).
EYE_LOOK_TABLE = "p_cf_head_bone#eyelook"


def collider_table(hi_poly: bool, type_bone: bool) -> str:
    """Bake name of :func:`body_colliders`' answer for one ``(hi_poly, type_bone)``.

    ``#hit_colliders`` rather than the ``#colliders`` the tables were first
    written under: those are in the bundle's object order, not the game's
    (:func:`_read_body_colliders`), and a new name is what makes every
    existing bake convert them again instead of reading the old order.
    """
    return f"{BODY_BONE_PREFAB[bool(hi_poly)]}#hit_colliders{'+typebone' if type_bone else ''}"


def bust_table(hi_poly: bool) -> str:
    """Bake name of :func:`bust_dynamic_bones`' answer for one armature."""
    return f"{BODY_BONE_PREFAB[bool(hi_poly)]}#bust"


def table_payload(value) -> dict | None:
    """The bake's JSON of one of the three armature reads (``tablecodec``)."""
    from . import tablecodec as TC
    return None if value is None else {"value": TC.encode(value)}


def _baked_body_table(name: str, types=()):
    """An armature read the converter stored, :data:`_NOT_BAKED` when it did not.

    The body armature bundle is opened once per process for the colliders, the
    bust's dynamic bones and the gaze (``runtime_reads``, §7.483); the
    converter keeps each answer under ``bake.write_named_table``.
    """
    from . import bake
    from . import tablecodec as TC
    recorded, payload = bake.named_table(bake.INSTALL, BODY_BONE_BUNDLE, name)
    if not recorded:
        return _NOT_BAKED
    try:
        return None if payload is None else TC.decode(payload["value"], types)
    except (KeyError, TypeError, ValueError) as exc:
        swallowed.note("origin.prefab._baked_body_table", exc)
        return _NOT_BAKED


def body_colliders(hi_poly: bool | None = None,
                   type_bone: bool | None = None, *,
                   from_bundle: bool = False) -> tuple:
    """The placed colliders every sway chain is solved against.

    Read from the armature prefab rather than from the garment, because that is
    where they are: a garment's ``m_Colliders`` is null in every shipped prefab
    and the loader overwrites it with these.  The result is in the prefab's own
    bind space, which the garments share exactly — ``p_o_bot_skirt02`` and
    ``p_cf_body_bone`` agree to the last bit on all 8 skirt roots, and differ by
    at most 12 mm on the ``cf_hit_*`` bones, where the armature is the one the
    game actually uses.

    One of the twenty-four is **not** left at its authored size.  The last
    thing ``ChaControl.UpdateShapeFace`` does before running the face shaper is

        GetReferenceInfo(RefObjKey.CORRECT_HEAD_DBCOL).transform.localScale
            = typeBone ? 1.08 : 1.2

    and key 80 resolves, in ``ChaReference.CreateReferenceInfo``'s body-bone
    group, to the object named ``cf_hit_head``.  A capsule's world radius is
    ``m_Radius * |lossyScale.z|``, so the head's collision volume is a fifth
    bigger than the prefab says — which is what keeps hair off the face.
    *type_bone* is the card's ``custom.body.typeBone``; the cache is keyed on it
    because it changes the answer.  From the bake when the converter wrote it
    (§7.483); ``from_bundle`` is the converter's read, past the bake and the
    cache.
    """
    if hi_poly is None:
        try:                                         # already imported by now
            from .figure.body_head import hi_poly as _hi
            hi_poly = bool(_hi())
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.body_colliders: from .figure.body_head import hi_poly as _hi", exc)
            hi_poly = True
    hi_poly = bool(hi_poly)
    type_bone = bool(_TYPE_BONE if type_bone is None else type_bone)
    key = (hi_poly, type_bone)
    from . import dynbone as DB
    if not from_bundle:
        cached = _BODY_COLLIDERS.get(key)
        if cached is not None:
            return cached
        got = _baked_body_table(collider_table(hi_poly, type_bone), (DB.Collider,))
        if got is not _NOT_BAKED:
            _BODY_COLLIDERS[key] = tuple(got or ())
            return _BODY_COLLIDERS[key]

    out = [DB.place_collider(col, world)
           for _name, _parent, world, _local, col in _read_body_colliders(hi_poly, type_bone)]
    if not from_bundle:
        _BODY_COLLIDERS[key] = tuple(out)
    return tuple(out)


def _read_body_colliders(hi_poly: bool, type_bone: bool) -> list:
    """``[(name, parent name, world, local, Collider), …]`` off the armature prefab.

    *world* is the collider object's bind world in the prefab and *local* the
    same object relative to its parent bone — one walk, read by
    :func:`body_colliders` (placed once, in bind space) and
    :func:`body_collider_rig` (placed every frame on the posed parent).  The
    head collider's ``localScale`` is the one ``UpdateShapeFace`` writes
    (1.08 with ``typeBone``, else 1.2), in both.

    **In hierarchy order**, because the solver pushes a particle out of one
    collider after another and a particle wedged between two (the front of a
    skirt between the thighs) ends where the last push left it.  The game's
    list is ``objBodyBone.GetComponentsInChildren<DynamicBoneCollider>(true)``,
    which is a pre-order walk of the transforms; the bundle's object table is
    not, and the two disagree on all 24 (§7.540).
    """
    from . import dynbone as DB
    from .install import configured
    out: list = []
    path = configured().bundle(BODY_BONE_BUNDLE)
    if not path.exists():
        return out
    try:
        L = loader(path)
        root = L.find_prefab(BODY_BONE_PREFAB[hi_poly])
        troot = L._transform_of(root) if root is not None else None
        if troot is None:
            return out
        worlds: dict = {}
        L._collect_worlds(troot, m3.identity(), worlds)
        by_go, names, father = {}, {}, {}
        for pid in worlds:
            node = L.by_id.get(pid)
            if node is None:
                continue
            try:
                tt = node.read_typetree()
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab._read_body_colliders: node.read_typetree", exc)
                continue
            gid = (tt.get("m_GameObject") or {}).get("m_PathID")
            father[pid] = (tt.get("m_Father") or {}).get("m_PathID")
            by_go[gid] = pid
            go = L.by_id.get(gid)
            try:
                names[pid] = "" if go is None else str(go.read().m_Name)
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab._read_body_colliders: names[pid] = '' if go is None else str…", exc)
                names[pid] = ""
        head_scale = head_collider_scale(type_bone)
        # ``_collect_worlds`` fills *worlds* parent first, children in
        # ``m_Children`` order: the order ``GetComponentsInChildren`` returns
        walk = {pid: k for k, pid in enumerate(worlds)}
        found: list = []
        for obj in L.env.objects:
            if obj.type.name != "MonoBehaviour" or not _collider_script(L, obj):
                continue
            tree = obj.read_typetree()
            pid = by_go.get((tree.get("m_GameObject") or {}).get("m_PathID"))
            if pid is None:
                continue
            col = DB.collider_from_tree(tree)
            if col is None:                          # no type tree in this build
                try:
                    col = DB.parse_collider(bytes(obj.get_raw_data()))
                except Exception as exc:             # noqa: BLE001
                    swallowed.note("origin.prefab._read_body_colliders: DB.parse_collider", exc)
                    col = None
            if col is None:
                continue
            # ``m_Center`` is a point in the collider's own local frame, which
            # ``_trs`` mirrored through X like every transform here (``S·M·S``):
            # ``TransformPoint(c)`` is then ``(S·M·S)·(S·c)``, so the centre is
            # mirrored too — ``cf_hit_shoulder_L``'s ``x = −0.05`` sat 0.1 × its
            # scale the wrong way along the bone (``cf_hit_bust02_*`` 0.01)
            col = DB.mirrored(col)
            world = np.asarray(worlds[pid], dtype=np.float64)
            if names.get(pid) == HEAD_COLLIDER:
                # UpdateShapeFace writes localScale, so the 3x3 scales and
                # the origin does not
                world = world.copy()
                world[:3, :3] = world[:3, :3] * head_scale
            dad = father.get(pid)
            parent_world = worlds.get(dad)
            local = (np.linalg.inv(np.asarray(parent_world, dtype=np.float64)) @ world
                     if parent_world is not None else world.copy())
            found.append((walk[pid], (names.get(pid, ""), names.get(dad, ""), world, local, col)))
        out = [row for _k, row in sorted(found, key=lambda kv: kv[0])]
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.prefab._read_body_colliders", exc)
        return []
    return out


def collider_rig_table(hi_poly: bool, type_bone: bool) -> str:
    """Bake name of :func:`body_collider_rig`'s answer for one ``(hi_poly, type_bone)``."""
    return f"{BODY_BONE_PREFAB[bool(hi_poly)]}#collider_rig{'+typebone' if type_bone else ''}"


_COLLIDER_RIGS: dict[tuple, tuple] = {}


def body_collider_rig(hi_poly: bool | None = None,
                      type_bone: bool | None = None, *,
                      from_bundle: bool = False) -> tuple:
    """The armature's colliders as ``(collider node name, Collider)``, in the game's order.

    ``DynamicBone`` places its colliders every frame from their own transforms,
    which hang off the character's bones — ``cf_hit_thigh02_L`` under
    ``cf_s_thigh02_L`` — and which the body shape pass moves and scales
    (``cf_a_hit_*``).  The graft carries the 24 nodes (``rig.body_bone_names``),
    so the run time places each collider on its node as the pose and the
    shape left it.  :func:`body_colliders` is the same 24 placed in the
    prefab's bind space, which only the load-time settle can use.  From the
    bake when the converter wrote it (``from_bundle`` is the converter's read).
    """
    from . import dynbone as DB
    if hi_poly is None:
        try:
            from .figure.body_head import hi_poly as _hi
            hi_poly = bool(_hi())
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.body_collider_rig: hi_poly", exc)
            hi_poly = True
    hi_poly = bool(hi_poly)
    type_bone = bool(_TYPE_BONE if type_bone is None else type_bone)
    key = (hi_poly, type_bone)
    if not from_bundle:
        cached = _COLLIDER_RIGS.get(key)
        if cached is not None:
            return cached
        got = _baked_body_table(collider_rig_table(hi_poly, type_bone), (DB.Collider,))
        if got is not _NOT_BAKED:
            _COLLIDER_RIGS[key] = tuple((str(n), c) for n, c in (got or ()))
            return _COLLIDER_RIGS[key]
    out = tuple((name, col) for name, _parent, _world, _local, col
                in _read_body_colliders(hi_poly, type_bone) if name)
    if not from_bundle:
        _COLLIDER_RIGS[key] = out
    return out


# ``ChaControl.<LoadAsync>c__Iterator0`` builds ``dictDynamicBoneBust`` by
# walking every ``DynamicBone_Ver02`` under the loaded character and filing it
# under the index its ``Comment`` matches.  These four strings are the whole
# mapping — there is no other key.
BUST_COMMENTS = {"左胸": 0, "右胸": 1, "左尻": 2, "右尻": 3}
_BUST_BONES: dict[bool, dict] = {}


def bust_dynamic_bones(hi_poly: bool | None = None, *,
                       from_bundle: bool = False) -> dict:
    """``dictDynamicBoneBust`` — index 0..3 → its ``DynamicBone_Ver02``.

    0/1 are the breasts and 2/3 the buttocks; the specs come back mirrored into
    this project's space, keyed by bone **name** so a solver can drive our
    armature with them.

    The low-poly armature carries **none** of these (338 transforms and zero
    components against 580 and four), so a low-detail character has no bust
    physics in the game either — the same story as its missing colliders.
    From the bake when the converter wrote it (§7.483).
    """
    if hi_poly is None:
        try:                                         # already imported by now
            from .figure.body_head import hi_poly as _hi
            hi_poly = bool(_hi())
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.bust_dynamic_bones: from .figure.body_head import hi_poly as _hi", exc)
            hi_poly = True
    hi_poly = bool(hi_poly)
    from . import dynbone2 as DB2
    if not from_bundle:
        cached = _BUST_BONES.get(hi_poly)
        if cached is not None:
            return cached
        got = _baked_body_table(bust_table(hi_poly),
                                (DB2.Ver02Spec, DB2.Pattern, DB2.BoneParam))
        if got is not _NOT_BAKED:
            _BUST_BONES[hi_poly] = got or {}
            return _BUST_BONES[hi_poly]

    from .install import configured
    out: dict = {}
    path = configured().bundle(BODY_BONE_BUNDLE)
    if path.exists():
        try:
            L = loader(path)
            root = L.find_prefab(BODY_BONE_PREFAB[hi_poly])
            troot = L._transform_of(root) if root is not None else None
            if troot is not None:
                worlds: dict = {}
                L._collect_worlds(troot, m3.identity(), worlds)
                names, hosts = {}, set()
                for pid in worlds:
                    node = L.by_id.get(pid)
                    if node is None:
                        continue
                    try:
                        gid = (node.read_typetree().get("m_GameObject")
                               or {}).get("m_PathID")
                    except Exception as exc:         # noqa: BLE001
                        swallowed.note("origin.prefab.bust_dynamic_bones: node.read_typetree", exc)
                        continue
                    go = L.by_id.get(gid)
                    if go is not None:
                        names[pid] = L._name(go)
                        hosts.add(gid)
                for obj in L.env.objects:
                    if obj.type.name != "MonoBehaviour":
                        continue
                    if _script_name(L, obj) != "DynamicBone_Ver02":
                        continue
                    tree = obj.read_typetree()
                    # the bundle holds both armatures; only the components
                    # hanging off *this* one belong to it
                    if (tree.get("m_GameObject") or {}).get("m_PathID") \
                            not in hosts:
                        continue
                    spec = DB2.from_tree(tree, names.get)
                    if spec is None:                 # no type tree in this build
                        try:
                            spec = DB2.parse_component(
                                bytes(obj.get_raw_data()), names.get)
                        except Exception as exc:     # noqa: BLE001
                            swallowed.note("origin.prefab.bust_dynamic_bones: spec = DB2.parse_component(byt…", exc)
                            spec = None
                    idx = BUST_COMMENTS.get(getattr(spec, "comment", ""))
                    if spec is not None and idx is not None:
                        out[idx] = spec.mirror_x()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.bust_dynamic_bones", exc)
            out = {}
    if not from_bundle:
        _BUST_BONES[hi_poly] = out
    return out


# The gaze is **not** on the head prefab.  ``EyeLookMaterialControll.script``
# is null there, and ``<ChangeHeadAsync>`` fills it in with the ``EyeLookCalc``
# that ships on the *body* bundle's head armature — one component for every
# head a card can name, which is why the eight patterns below are not per-head
# data the way the iris waits are (see :meth:`PrefabLoader._eye_look`).
HEAD_BONE_PREFAB = "p_cf_head_bone"

# ``EyeTypeState``'s fields, in the order ``origin_il_dump --fields`` gives them.
# ``lookType`` is an ``EYE_LOOK_TYPE``; the eight rows are what ``ptnNo``
# indexes, and their ``comment`` (正面 / こっち / 制御 / そらす, then the four H
# ones) is the only label the game gives them.
_EYE_STATE_FIELDS = ("comment", "thresholdAngleDifference", "bendingMultiplier",
                     "maxAngleDifference", "upBendingAngle", "downBendingAngle",
                     "minBendingAngle", "maxBendingAngle", "leapSpeed",
                     "forntTagDis", "nearDis", "hAngleLimit", "vAngleLimit",
                     "lookType")
_EYE_LOOK_CALC: list = []           # one slot, filled on first read


def eye_look_calc(*, from_bundle: bool = False) -> dict | None:
    """``EyeLookCalc`` + ``EyeLookController``, off ``p_cf_head_bone``.

    Everything ``EyeUpdateCalc`` reads that is *data* rather than code:

    * ``states``  — the eight ``EyeTypeState`` rows.  The bending angles are
      not symmetric (the stock rows run −36..23 horizontally and −30..10
      vertically), and that asymmetry is the whole reason a character looking
      straight ahead does **not** have ``angleHRate`` 0 — see
      :mod:`figure.eye_look`.
    * ``ptn``     — ``EyeLookController.ptnNo``, 1 (こっち) on the prefab.
    * ``eyes``    — per eye: the node ``EyeUpdateCalc`` measures from, its
      ``eyeLR``, and the two reference directions ``Init`` freezes at the bind
      pose (``Inverse(parent.rotation) * rootNode.rotation * headLookVector``).
      Both eye nodes are bare leaf transforms — no children, no renderer — so
      the rotation the method writes onto them moves nothing on screen.  The
      gaze is visible only through the two rates.
    * ``root`` / ``center`` — ``rootNode`` (the armature root, parented onto
      ``cf_s_head``) and ``trfCenter`` (``cf_J_Eye_tz``), by bone name.

    Vectors come back in **this project's** space (X mirrored), like everything
    else the loader returns.  ``None`` when the bundle is not installed.  From
    the bake when the converter wrote it (§7.483).
    """
    if not from_bundle:
        if _EYE_LOOK_CALC:
            return _EYE_LOOK_CALC[0]
        got = _baked_body_table(EYE_LOOK_TABLE)
        if got is not _NOT_BAKED:
            _EYE_LOOK_CALC.append(got)
            return got

    from .install import configured
    out: dict | None = None
    path = configured().bundle(BODY_BONE_BUNDLE)
    if path.exists():
        try:
            out = _read_eye_look_calc(loader(path))
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab.eye_look_calc", exc)
            out = None
    if not from_bundle:
        _EYE_LOOK_CALC.append(out)
    return out


def _read_eye_look_calc(L: "PrefabLoader") -> dict | None:
    root = L.find_prefab(HEAD_BONE_PREFAB)
    troot = L._transform_of(root) if root is not None else None
    if troot is None:
        return None
    worlds: dict = {}
    L._collect_worlds(troot, m3.identity(), worlds)
    names, hosts = {}, set()
    for pid in worlds:
        node = L.by_id.get(pid)
        gid = None
        if node is not None:
            try:
                gid = (node.read_typetree().get("m_GameObject")
                       or {}).get("m_PathID")
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.prefab._read_eye_look_calc: gid = (node.read_typetree().get('m_GameOb…", exc)
                gid = None
        go = L.by_id.get(gid) if gid else None
        if go is not None:
            names[pid] = L._name(go)
            hosts.add(gid)

    calc = ctrl = None
    for obj in L.env.objects:
        if obj.type.name != "MonoBehaviour":
            continue
        cls = _script_name(L, obj)
        if cls not in ("EyeLookCalc", "EyeLookController"):
            continue
        try:
            tree = obj.read_typetree()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.prefab._read_eye_look_calc: obj.read_typetree", exc)
            continue
        if (tree.get("m_GameObject") or {}).get("m_PathID") not in hosts:
            continue
        if cls == "EyeLookCalc":
            calc = tree
        else:
            ctrl = tree
    if calc is None:
        return None

    def node_of(ref):
        """(bone name, world matrix) behind a Transform PPtr."""
        pid = (ref or {}).get("m_PathID")
        return names.get(pid, ""), worlds.get(pid)

    root_name, root_world = node_of(calc.get("rootNode"))
    if root_world is None:                           # rootNode null => Init
        root_name, root_world = names.get(troot.path_id, ""), worlds[troot.path_id]
    centre_name, _ = node_of(calc.get("trfCenter"))

    look = _vec3(calc.get("headLookVector"), (0.0, 0.0, 1.0))
    up = _vec3(calc.get("headUpVector"), (0.0, 1.0, 0.0))
    eyes = []
    for eo in (calc.get("eyeObjs") or []):
        name, world = node_of(eo.get("eyeTransform"))
        if world is None:
            continue
        # ``Init`` runs on the freshly instantiated prefab, so these are the
        # *bind* directions -- the face sliders rotate ``cf_J_Eye_tx_*`` later
        # and the reference deliberately does not follow.  The parent's
        # rotation is the eye node's own: its local rotation is identity, and
        # ``EyeUpdateCalc`` resets it to that before reading anything.
        rel = _orthonormal(world[:3, :3]).T @ _orthonormal(root_world[:3, :3])
        eyes.append({"bone": name, "eyeLR": int(eo.get("eyeLR", 0) or 0),
                     "look": tuple(rel @ _unit(look)),
                     "up": tuple(rel @ _unit(up))})

    states = []
    for st in (calc.get("eyeTypeStates") or []):
        row = {}
        for key in _EYE_STATE_FIELDS:
            v = st.get(key)
            row[key] = v if key == "comment" else float(v or 0.0)
        row["lookType"] = int(st.get("lookType", 0) or 0)
        states.append(row)
    return {
        "states": states,
        "ptn": int((ctrl or {}).get("ptnNo", 0) or 0),
        "correct": bool(calc.get("correct")),
        "centerEyeLength": float(calc.get("centerEyeLength") or 0.0),
        "sorasiRate": float(calc.get("sorasiRate") or 0.0),
        "root": root_name,
        "center": centre_name,
        "eyes": eyes,
    }


# ---------------------------------------------------------------------------
# FaceBlendShape — the expression *patterns* (eyebrow / eyes / mouth)
# ---------------------------------------------------------------------------
FACE_BLEND_BUNDLE = "chara/bo_head_00.unity3d"

# ``FaceBlendShape``'s three ``FBSBase`` slots, in the order ``SetFace`` drives
# them.  Each is an ``FBSCtrl*`` with an ``FBSTarget[]``; a target pairs one
# mesh with a ``PtnSet`` — ``CloseOpen{Close, Open}`` **per pattern number**.
_FBS_SLOTS = (("eyebrow", "EyebrowCtrl"), ("eyes", "EyesCtrl"),
              ("mouth", "MouthCtrl"))


def face_blend_patterns(bundle: str = FACE_BLEND_BUNDLE,
                        prefab: str = "p_cf_head_00") -> dict | None:
    """``FaceBlendShape``'s pattern → blend-shape table, off the head prefab.

    ``ChaControl.ChangeMouthPtn(ptn, open)`` is only bookkeeping — it stores
    ``fileStatus.mouthPtn``, flips ``tongueState`` when ``ptn == 21`` and turns
    ``UseAdjustWidthScale`` off for 21/22 — and then hands the number to
    ``FBSBase.ChangePtn``, which does ``dictNowFace = {ptn: 1}``.  The shapes
    come out of ``FBSBase.CalculateBlendShape``, transcribed line for line:

        openMax  = correctOpenMax >= 0 ? correctOpenMax : OpenMax
        rate     = Lerp(OpenMin, openMax, openRate)     # openRate is the voice
        if FixedRate >= 0: rate = FixedRate             # -0.1 prefab / -1 off
        openPct  = (int)Clamp(rate * 100, 0, 100)
        closePct = 100 - openPct
        for target in FBSTarget:                        # 5 meshes, same PtnSet
            for i, co in enumerate(target.PtnSet):
                w = dictNowFace.get(i, 0)               # blended with dictBackFace
                if co.Close >= 0: weight[co.Close] += w * closePct
                if co.Open  >= 0: weight[co.Open]  += w * openPct

    So a pattern is **a pair of shapes**, and the mouth's openness picks how
    much of each — which is why ``HFaceList``'s ``openMinMouth`` of 1.0 on
    every 舐め/咥え row (``ptn`` 21/22) forces the open half regardless of the
    voice, and is exactly what a tongue-out motion needs.

    ``Close``/``Open`` are indices into the mesh's own blend-shape channel
    list, so they are returned as **names**: the five meshes of one pattern
    (``cf_O_face``, ``cf_O_noseline``, ``cf_O_tooth``, ``cf_O_canine``,
    ``o_tang``) carry the same suffix (``kuti_face.f00_name_op`` /
    ``kuti_sita.t00_name_op``), which is the key
    :mod:`figure.face_morph` already registers them under.

    ``None`` when the bundle is missing — the caller reports it rather than
    substituting a pattern table (``CLAUDE.md`` 一之一).
    """
    from .install import configured
    from . import sideloader as SL

    if SL.overrides_at(bundle):
        # the component rides on the head *prefab*, and the game loads that by
        # name — so from the source Sideloader hands *prefab* out of (§7.489)
        env = SL.open_asset_bundle(bundle, prefab)
        if env is None:
            return None
        L = PrefabLoader(Path(bundle), env=env)
    else:
        path = configured().bundle(bundle)
        if not path.exists():
            return None
        try:
            L = loader(path)
        except Exception as exc:                         # noqa: BLE001
            swallowed.note("origin.prefab.face_blend_patterns: L = loader(path)", exc)
            return None

    fbs = None
    for obj in L.env.objects:
        if obj.type.name == "MonoBehaviour" and _script_name(L, obj) == "FaceBlendShape":
            try:
                fbs = obj.read_typetree()
            except Exception as exc:                     # noqa: BLE001
                swallowed.note("origin.prefab.face_blend_patterns: fbs = obj.read_typetree()", exc)
                fbs = None
            break
    if not fbs:
        return None

    # GameObject path-id → its skinned mesh's blend-shape channel names
    shape_names: dict[int, list[str]] = {}
    for obj in L.env.objects:
        if obj.type.name != "SkinnedMeshRenderer":
            continue
        try:
            t = obj.read_typetree()
        except Exception as exc:                         # noqa: BLE001
            swallowed.note("origin.prefab.face_blend_patterns: obj.read_typetree", exc)
            continue
        gid = (t.get("m_GameObject") or {}).get("m_PathID")
        mesh = L.by_id.get((t.get("m_Mesh") or {}).get("m_PathID"))
        if gid is None or mesh is None:
            continue
        try:
            ch = ((mesh.read_typetree().get("m_Shapes") or {}).get("channels")
                  or [])
        except Exception as exc:                         # noqa: BLE001
            swallowed.note("origin.prefab.face_blend_patterns: mesh.read_typetree", exc)
            continue
        shape_names[gid] = [str(c.get("name") or "") for c in ch]

    out: dict = {}
    for key, field in _FBS_SLOTS:
        ctrl = fbs.get(field) or {}
        targets = []
        for tgt in (ctrl.get("FBSTarget") or []):
            gid = (tgt.get("ObjTarget") or {}).get("m_PathID")
            names = shape_names.get(gid) or []
            go = L.by_id.get(gid)
            ptns = []
            for co in (tgt.get("PtnSet") or []):
                def nm(i):
                    i = int(i if i is not None else -1)
                    return names[i] if 0 <= i < len(names) else ""
                ptns.append((nm(co.get("Close")), nm(co.get("Open"))))
            targets.append({
                "object": L._name(go) if go is not None else "",
                "patterns": ptns,
            })
        if not targets:
            continue
        out[key] = {
            "targets": targets,
            "open_min": float(ctrl.get("OpenMin") or 0.0),
            "open_max": float(ctrl.get("OpenMax") or 0.0),
            # −0.1 on the prefab, i.e. off; H scenes write a positive value.
            "fixed_rate": float(ctrl.get("FixedRate") if
                                ctrl.get("FixedRate") is not None else -0.1),
        }
    return out or None


def _vec3(value, fallback) -> np.ndarray:
    """A serialised Vector3, mirrored into this project's space."""
    if not isinstance(value, dict):
        return np.asarray(fallback, dtype=np.float64)
    return np.array([-float(value.get("x", 0.0)), float(value.get("y", 0.0)),
                     float(value.get("z", 0.0))])


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v
