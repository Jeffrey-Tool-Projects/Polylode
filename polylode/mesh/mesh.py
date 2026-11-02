"""Mesh / Part / Model containers.

A ``Mesh`` is raw geometry with skinning attributes.  A ``Part`` pairs a mesh
with a material and a visibility flag (so clothing pieces can be toggled).  A
``Model`` is a list of parts plus the skeleton they are bound to.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field, fields

import numpy as np

from .skin import DEFAULT_INFLUENCES, MAX_INFLUENCES as MAX_INFLUENCES, pad_stream, validate_streams
from . import tangent


class Mesh:
    """Triangle mesh with position/normal/uv/color and up to 8 skin influences."""

    __slots__ = ("positions", "normals", "uvs", "colors", "bone_idx", "bone_wt",
                 "indices", "name", "uv1", "uv2", "uv3", "vcolor", "tangents", "skin_dual",
                 "skin_positions", "skin_normals", "outline_normals", "skin_outline_normals", "_bounds")

    #: Vertex streams a mesh only has if it shipped one, with their widths.
    #: Empty otherwise, and every editing operation keeps them that way — a
    #: half-filled channel would hand the tail of the buffer somebody else's
    #: values, silently.
    #:
    #: ``vcolor`` is deliberately *not* ``colors``: this project's ``colors``
    #: is a vertex tint the renderer multiplies into the albedo, while an
    #: imported mesh's COLOR stream is a shader input whose meaning is whatever
    #: that shader decides.  On ``o_body_a`` it is a decal mask that reads 0
    #: over most of the body, so filing it under ``colors`` renders the arms
    #: and neck pure black (§7.59 八).
    #:
    #: ``skin_dual`` is each vertex's share of the dual-quaternion blend on a
    #: part whose ``skinning`` is mixed (``mesh/skinning.py``); no other part
    #: has it.
    OPTIONAL_STREAMS = (("uv1", 2), ("uv2", 2), ("uv3", 2), ("vcolor", 4), ("tangents", 4),
                        ("skin_dual", 1))

    def __init__(self, name: str = "mesh"):
        self.name = name
        self.positions = np.zeros((0, 3), dtype=np.float32)
        self.normals = np.zeros((0, 3), dtype=np.float32)
        self.uvs = np.zeros((0, 2), dtype=np.float32)
        # Illusion's hair lays each strand out on UV1 for the _HairGloss sheet
        # (the anime highlight band); the body and the face carry all three
        # sets, because main_skin samples its areola / under-hair / eyeshadow
        # decals on UV1, UV2 and UV3 rather than on the albedo's own layout,
        # and gates the first two on COLOR.r / COLOR.b (§7.50 六, §7.59).
        for s, width in Mesh.OPTIONAL_STREAMS:
            setattr(self, s, np.zeros((0, width), dtype=np.float32))
        self.colors = np.zeros((0, 4), dtype=np.float32)
        self.bone_idx = np.zeros((0, DEFAULT_INFLUENCES), dtype=np.int32)
        self.bone_wt = np.zeros((0, DEFAULT_INFLUENCES), dtype=np.float32)
        self.indices = np.zeros((0,), dtype=np.uint32)
        # What the skinning is given in place of ``positions`` / ``normals`` when the bones a
        # vertex hangs on do not all bind it where it rests (:meth:`skin_rest`).  Empty on every
        # mesh but one that says so; not a stream the editing operations know.
        self.skin_positions = np.zeros((0, 3), dtype=np.float32)
        self.skin_normals = np.zeros((0, 3), dtype=np.float32)
        # The normal an outline hull is pushed along where the material asks for the averaged one
        # (``mesh/outline_normals.py``), and that normal as the skinning takes it (:meth:`outline_rest`).
        # Made when a converted character is put together, never stored; like ``skin_*``, not a stream
        # the editing operations know, so a mesh they change has none (its outline goes by the normal).
        self.outline_normals = np.zeros((0, 3), dtype=np.float32)
        self.skin_outline_normals = np.zeros((0, 3), dtype=np.float32)
        # ``(positions, lo, hi)`` of the last :meth:`bounds` — see there
        self._bounds = None

    # -- construction ------------------------------------------------------
    @staticmethod
    def from_arrays(positions, indices, normals=None, uvs=None, colors=None,
                    name: str = "mesh") -> "Mesh":
        m = Mesh(name)
        m.positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        m.indices = np.asarray(indices, dtype=np.uint32).ravel()
        n = len(m.positions)
        m.normals = (np.asarray(normals, dtype=np.float32).reshape(-1, 3)
                     if normals is not None else np.zeros((n, 3), dtype=np.float32))
        m.uvs = (np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
                 if uvs is not None else np.zeros((n, 2), dtype=np.float32))
        m.colors = (np.asarray(colors, dtype=np.float32).reshape(-1, 4)
                    if colors is not None else np.ones((n, 4), dtype=np.float32))
        m.bone_idx = np.zeros((n, DEFAULT_INFLUENCES), dtype=np.int32)
        m.bone_wt = np.zeros((n, DEFAULT_INFLUENCES), dtype=np.float32)
        m.bone_wt[:, 0] = 1.0
        if normals is None:
            m.compute_normals()
        return m

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    @property
    def tri_count(self) -> int:
        return len(self.indices) // 3

    def skin_rest(self) -> tuple[np.ndarray, np.ndarray]:
        """``(positions, normals)`` as the skinning takes them.

        The rest shape itself, except on a mesh whose bones bind it elsewhere
        than it rests (a body lent from another rig, ``endfield/body.unblend``):
        there each vertex is stored where its bones' binds, blended by its
        weights, put it *at* its rest position.  Everything that asks where
        the surface is at rest still reads ``positions``.
        """
        if len(self.skin_positions) == self.vertex_count and self.vertex_count:
            return self.skin_positions, self.skin_normals
        return self.positions, self.normals

    def outline_rest(self) -> np.ndarray:
        """``outline_normals`` as the skinning takes them (:meth:`skin_rest`'s frame); empty when there are none.

        A mesh whose skin is stored elsewhere than it rests needs them carried
        there too (``endfield/body.unblend``): without that it has none.
        """
        n = self.vertex_count
        if not n or len(self.outline_normals) != n:
            return np.zeros((0, 3), dtype=np.float32)
        if len(self.skin_positions) == n:
            return self.skin_outline_normals if len(self.skin_outline_normals) == n \
                else np.zeros((0, 3), dtype=np.float32)
        return self.outline_normals

    def copy(self) -> "Mesh":
        m = Mesh(self.name)
        for s in Mesh.__slots__:
            if s in ("name", "_bounds"):
                continue
            setattr(m, s, getattr(self, s).copy())
        return m

    # -- editing -----------------------------------------------------------
    def append(self, other: "Mesh") -> "Mesh":
        """Merge ``other`` into self (in place); returns self."""
        if other.vertex_count == 0:
            return self
        base = self.vertex_count
        if base and bool(len(self.bone_wt)) != bool(len(other.bone_wt)):
            raise ValueError("cannot append bound and unbound skin streams")
        validate_streams(self.bone_idx, self.bone_wt, base if len(self.bone_wt) else 0)
        validate_streams(other.bone_idx, other.bone_wt, other.vertex_count if len(other.bone_wt) else 0)
        width = max(self.bone_wt.shape[1], other.bone_wt.shape[1])
        merged_indices = np.vstack([pad_stream(self.bone_idx, width), pad_stream(other.bone_idx, width)])
        merged_weights = np.vstack([pad_stream(self.bone_wt, width), pad_stream(other.bone_wt, width)])
        # The optional streams are optional, so merging two meshes only keeps
        # one when *both* sides have it.
        for s, width in Mesh.OPTIONAL_STREAMS:
            mine, theirs = getattr(self, s), getattr(other, s)
            setattr(self, s,
                    np.vstack([mine, theirs])
                    if len(mine) == base and len(theirs) == other.vertex_count
                    else np.zeros((0, width), dtype=np.float32))
        self.positions = np.vstack([self.positions, other.positions])
        self.normals = np.vstack([self.normals, other.normals])
        self.uvs = np.vstack([self.uvs, other.uvs])
        self.colors = np.vstack([self.colors, other.colors])
        self.bone_idx = merged_indices
        self.bone_wt = merged_weights
        self.indices = np.concatenate([self.indices, other.indices + base]).astype(np.uint32)
        return self

    def transform(self, matrix: np.ndarray) -> "Mesh":
        m = np.asarray(matrix, dtype=np.float64)
        tangent.validate(self.tangents, self.vertex_count)
        moved_tangent = tangent.affine(self.tangents, m[:3, :3]) if len(self.tangents) else None
        nm = np.linalg.inv(m[:3, :3]).T
        self.positions = (self.positions @ m[:3, :3].T + m[:3, 3]).astype(np.float32)
        n = self.normals @ nm.T
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        self.normals = (n / np.maximum(ln, 1e-9)).astype(np.float32)
        if moved_tangent is not None:
            self.tangents = moved_tangent
        return self

    def translate(self, offset) -> "Mesh":
        self.positions = (self.positions + np.asarray(offset, dtype=np.float32)).astype(np.float32)
        return self

    def scale(self, s) -> "Mesh":
        s = np.asarray(s if not np.isscalar(s) else (s, s, s), dtype=np.float32)
        if len(self.tangents):
            matrix = np.eye(4)
            matrix[:3, :3] = np.diag(s)
            return self.transform(matrix)
        self.positions = (self.positions * s).astype(np.float32)
        inv = 1.0 / np.maximum(np.abs(s), 1e-9)
        n = self.normals * inv
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        self.normals = (n / np.maximum(ln, 1e-9)).astype(np.float32)
        return self

    def mirror_x(self) -> "Mesh":
        """Return an X-mirrored copy with winding flipped."""
        m = self.copy()
        m.positions = m.positions * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
        m.normals[:, 0] *= -1.0
        if len(m.tangents):
            tangent.validate(m.tangents, m.vertex_count)
            m.tangents[:, 0] *= -1.0
            m.tangents[:, 3] *= -1.0
        idx = m.indices.reshape(-1, 3)[:, ::-1]
        m.indices = idx.ravel().astype(np.uint32)
        return m

    def flip_winding(self) -> "Mesh":
        self.indices = self.indices.reshape(-1, 3)[:, ::-1].ravel().astype(np.uint32)
        return self

    def set_color(self, rgba) -> "Mesh":
        self.colors = np.tile(np.asarray(rgba, dtype=np.float32), (self.vertex_count, 1))
        return self

    def set_bone(self, bone_index: int) -> "Mesh":
        """Rigidly bind every vertex to a single bone."""
        self.bone_idx[:] = 0
        self.bone_wt[:] = 0.0
        self.bone_idx[:, 0] = bone_index
        self.bone_wt[:, 0] = 1.0
        return self

    def compute_normals(self, smooth: bool = True) -> "Mesh":
        p = self.positions.astype(np.float64)
        tri = self.indices.reshape(-1, 3)
        if len(tri) == 0:
            return self
        v0, v1, v2 = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        n = np.zeros_like(p)
        # area-weighted accumulation
        np.add.at(n, tri[:, 0], fn)
        np.add.at(n, tri[:, 1], fn)
        np.add.at(n, tri[:, 2], fn)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        self.normals = (n / np.maximum(ln, 1e-9)).astype(np.float32)
        if not smooth:
            self.split_faceted()
        return self

    def split_faceted(self) -> "Mesh":
        """Duplicate vertices per triangle to get flat shading."""
        tri = self.indices.reshape(-1, 3)
        flat = tri.ravel()
        for s, _ in Mesh.OPTIONAL_STREAMS:
            if len(getattr(self, s)) == len(self.positions):
                setattr(self, s, getattr(self, s)[flat])
        self.positions = self.positions[flat]
        self.uvs = self.uvs[flat]
        self.colors = self.colors[flat]
        self.bone_idx = self.bone_idx[flat]
        self.bone_wt = self.bone_wt[flat]
        self.indices = np.arange(len(flat), dtype=np.uint32)
        self.compute_normals(smooth=True)
        return self

    def weld(self, tol: float = 1e-5) -> "Mesh":
        """Merge coincident vertices (keeps first occurrence's attributes)."""
        if self.vertex_count == 0:
            return self
        q = np.round(self.positions / tol).astype(np.int64)
        _, first, inverse = np.unique(q, axis=0, return_index=True, return_inverse=True)
        order = np.argsort(first)
        remap = np.zeros(len(first), dtype=np.int64)
        remap[order] = np.arange(len(first))
        keep = first[order]
        for s, _ in Mesh.OPTIONAL_STREAMS:
            if len(getattr(self, s)) == len(self.positions):
                setattr(self, s, getattr(self, s)[keep])
        self.positions = self.positions[keep]
        self.normals = self.normals[keep]
        self.uvs = self.uvs[keep]
        self.colors = self.colors[keep]
        self.bone_idx = self.bone_idx[keep]
        self.bone_wt = self.bone_wt[keep]
        self.indices = remap[inverse[self.indices]].astype(np.uint32)
        self.compute_normals()
        return self

    def drop_unused(self) -> np.ndarray:
        """Drop vertices no triangle references; returns the boolean keep mask.

        A vertex in no triangle is **not surface**, and this body ships 237 of
        them: :func:`polylode.mesh.weld.weld` trims interior triangles away
        and leaves their vertices behind.  They cost nothing to draw and are
        invisible, which is why they survived twenty sections — but they are
        not free to *measure*: ``compute_normals`` leaves a zero normal on a
        vertex with no face, and ``--vs-shape`` reads a zero normal as **90°
        out** — 24 of the 28 vertices its ``nipple`` band counted over 30° were
        these, and so were all 98 of the ``arm`` band's "buried" ones (§7.304).  They are also counted in every
        "our vertices against Origin's" ratio, which is the number ``PROGRESS`` N20
        spends on what to build next.

        The caller keeps the mask because a per-vertex sidecar (the segment
        labels) has to come along: a compaction that leaves ``seg`` the old
        length is a silent misalignment of every ruler's section selection.
        """
        if self.vertex_count == 0:
            return np.zeros(0, dtype=bool)
        keep = np.zeros(self.vertex_count, dtype=bool)
        keep[np.unique(np.asarray(self.indices, dtype=np.int64))] = True
        if keep.all():
            return keep
        remap = np.cumsum(keep) - 1
        for s, _ in Mesh.OPTIONAL_STREAMS:
            if len(getattr(self, s)) == len(keep):
                setattr(self, s, getattr(self, s)[keep])
        self.positions = self.positions[keep]
        self.normals = self.normals[keep]
        self.uvs = self.uvs[keep]
        self.colors = self.colors[keep]
        self.bone_idx = self.bone_idx[keep]
        self.bone_wt = self.bone_wt[keep]
        self.indices = remap[np.asarray(self.indices,
                                        dtype=np.int64)].astype(np.uint32)
        return keep

    def bounds(self):
        """``(lo, hi)`` of the positions, cached on the positions array itself.

        ``BuiltCharacter.height`` asks every frame for every part (§7.340),
        and a ``min``/``max`` over each mesh was 3 ms of it (§7.420).  The
        cache holds the array it was computed from and is reused only while
        ``positions`` **is** that array — every writer in the project replaces
        the array (skinning, blend shapes, cloth) rather than writing into it,
        and that is the invariant this relies on: an in-place write must be
        followed by assigning a new array.
        """
        if self.vertex_count == 0:
            return np.zeros(3), np.zeros(3)
        c = self._bounds
        if c is None or c[0] is not self.positions:
            c = self._bounds = (self.positions, self.positions.min(axis=0),
                                self.positions.max(axis=0))
        return c[1].copy(), c[2].copy()

    def normalize_weights(self) -> "Mesh":
        s = self.bone_wt.sum(axis=1, keepdims=True)
        bad = s[:, 0] < 1e-6
        if np.any(bad):
            self.bone_wt[bad] = 0.0
            self.bone_wt[bad, 0] = 1.0
            s = self.bone_wt.sum(axis=1, keepdims=True)
        self.bone_wt = (self.bone_wt / s).astype(np.float32)
        return self

    def __repr__(self):
        return f"<Mesh {self.name} v={self.vertex_count} t={self.tri_count}>"


@dataclass
class Material:
    """Toon material parameters consumed by the cel shader."""
    name: str = "default"
    base_color: tuple = (1.0, 1.0, 1.0, 1.0)
    shadow_color: tuple = (0.78, 0.70, 0.76, 1.0)   # multiplied into shaded areas
    specular_color: tuple = (1.0, 1.0, 1.0, 1.0)
    specular_power: float = 32.0
    specular_strength: float = 0.25
    rim_color: tuple = (1.0, 1.0, 1.0, 1.0)
    rim_strength: float = 0.35
    rim_power: float = 3.0
    # ---- Illusion's own specular, which is *not* the Blinn pair above -----
    # ``_SpecularPower`` is a 0..1 strength that also sets the exponent
    # (256 * P) and the whole lobe is scaled by ``_SpecularColor.a``; the three
    # below carry the rest of main_opaque's specular block.  Keeping them apart
    # from ``specular_power`` / ``specular_strength`` matters because those two
    # mean an exponent and a mix weight in this project's own model, and
    # feeding Origin's numbers into them is what put a highlight on every garment.
    origin_specular: float = 0.0        # _SpecularPower
    spec_nail: float = 0.0          # _SpecularPowerNail
    no_tex_spec: float = 0.0        # _notusetexspecular
    rim_v: float = 0.0              # _rimV, the scale on main_opaque's rim
    outline_color: tuple = (0.20, 0.13, 0.17, 1.0)
    outline_width: float = 1.0
    shade_steps: float = 2.0        # number of quantisation bands
    shade_feather: float = 0.03     # softness of each band edge
    shade_offset: float = 0.0       # pushes the terminator (face uses a high value)
    texture: str | None = None      # key into the texture cache
    normal_map: str | None = None   # Illusion _NormalMap
    normal_scale: float = 1.0       # Illusion _BumpScale
    normal_encoding: int = 0       # 0: RGB; 1: GF2 R*A/inverted G; 2: declared RG/positive Z
    specular_threshold: float = 0.0  # Illusion _SpeclarHeight; 0 = use exponent
    # ---- Illusion's toon model (see origin/figure/toon.py) --------------
    # ``toon_family`` says which of Origin's shaders this material declares; 0 is
    # this project's own procedural model and keeps the old banded lighting.
    toon_family: int = 0
    #: Converted source semantics must survive part/material copies.
    declared_role: str | None = None
    edit_color_target: str | None = None
    # The diffuse ramp Origin samples with max(N·L, 0) — a global (``_RampG``),
    # not a per-material choice, but carried here so a Part can be drawn
    # without the renderer having to know whose character it belongs to.
    ramp: str | None = None
    # ``_AnotherRamp``: a *second* ramp, sampled by the view angle (hair) or
    # |N·H| (clothing).  It is not a diffuse ramp — reading it as one blows
    # the surface out, which is why it used to be gated behind a guess.
    view_ramp: str | None = None
    view_ramp_transform: tuple = (1.0, 1.0, 0.0, 0.0)
    # ``_NormalMap_ST`` (scale u, scale v, offset u, offset v): the normal map
    # is sampled at ``uv * scale + offset`` like every Unity ``TRANSFORM_TEX``
    # (§7.383); 20 of the 5,901 baked materials tile theirs, up to 16×
    normal_transform: tuple = (1.0, 1.0, 0.0, 0.0)
    # ``_LineMask``/``_DetailMask`` resolved into what the shader actually
    # does with them: R = the multiplier on the diffuse term, G = the ink that
    # darkens line work, B = outline width scale, A = the ``_AnotherRamp``
    # blend weight (which doubles as the scale on the shadow tint — one
    # register serves both, see :func:`create_tex.shade_mask`).
    shade_mask: str | None = None
    # ``_DetailMask`` itself, resampled through its own ``_ST``.  main_opaque
    # reads four channels the packing above has no room for — the specular
    # mask (.r), the rim mask (.g), the rim's own remap of (.b) and the
    # specular map (.a) — and reads .r a *second* time at a parallax-offset uv,
    # which no packing can carry at all.  Bound for the clothing family only.
    detail_mask: str | None = None
    # ``_HairGloss`` + its per-piece ``_ST``; ``gloss_height`` is
    # ``_SpeclarHeight``, which shifts the band with the viewing angle.
    gloss: str | None = None
    gloss_transform: tuple = (1.0, 1.0, 0.0, 0.0)
    gloss_height: float = 1.0
    # ``1 - _ShadowExtend``: the floor under main_hair's second tint term.
    shade_floor: float = 0.0
    pattern: str | None = None
    pattern_color: tuple = (1.0, 1.0, 1.0, 1.0)
    pattern_transform: tuple = (1.0, 1.0, 0.0, 0.0)  # scale u, scale v, off u, off v
    alpha_test: float = 0.0
    #: ``main_item_ditherd``'s ``_Cutoff`` for its screen-space dither; negative when the shader is another
    dither_cut: float = -1.0
    # Illusion's ``EyeLookMaterialControll.offset`` — a per-material UV shift.
    # The iris uses it for gaze; everything else leaves it at zero.
    uv_offset: tuple = (0.0, 0.0)
    # ``toon_eye_lod0``'s two highlights, as one map on UV1 sampled at
    # ``uv1 * xy + zw`` (``body_head.eye_highlight`` / ``eye_highlight_xf``):
    # the xform follows the gaze and the shake, ``highlight_on`` is
    # ``_isHighLight``.  Bound for the iris only (§7.403).
    highlight: str | None = None
    highlight_xf: tuple = (1.0, 1.0, 0.0, 0.0)
    highlight_on: float = 1.0
    # ``toon_glasses_lod0`` (§7.413): its own ``_Color`` / ``_Color4`` (the
    # accessory's fourth card colour lands in the latter), ``(_ColorInverse,
    # _ColorSort, _RimPower)`` and ``_GlassRamp`` with its ``_ST``.
    glass_color: tuple = (1.0, 1.0, 1.0, 1.0)
    glass_color4: tuple = (1.0, 1.0, 1.0, 1.0)
    glass_params: tuple = (0.0, 0.0, 1.0)
    glass_ramp: str | None = None
    glass_ramp_xf: tuple = (1.0, 1.0, 0.0, 0.0)
    # ``main_skin``'s liquid layer (``origin.figure.siru``, D13): the material's own ``_liquidmask``,
    # ``_Texture2`` (the drops) and ``_Texture3`` (their normals) with their transforms, the levels
    # ``UpdateSiru`` writes (``_liquidftop``, ``fbot``, ``btop``, ``bbot``; ``_liquidface``), the liquid's
    # colour at thin and full coverage and its opacity (``siru.FluidLook``), and ``_rimpower``, which on the
    # skin only the liquid's rim reads.  No levels, no layer.
    liquid_mask: str | None = None
    liquid_drop: str | None = None
    liquid_normal: str | None = None
    liquid_levels: tuple = (0.0, 0.0, 0.0, 0.0)
    liquid_face: float = 0.0
    liquid_tiling: tuple = (1.0, 1.0, 0.0, 0.0)
    liquid_mask_xf: tuple = (1.0, 1.0, 0.0, 0.0)
    liquid_drop_xf: tuple = (1.0, 1.0, 0.0, 0.0)
    liquid_normal_xf: tuple = (1.0, 1.0, 0.0, 0.0)
    liquid_look: tuple = (0.5, 0.397059, 0.242647, 0.85, 0.85, 0.85, 1.0)
    liquid_rim_power: float = 0.0
    liquid_solid: bool = False      # the whole surface is liquid (``game.intimate.fluid_shot``'s mesh)
    blend: bool = False             # Illusion _Mode 3: blended pass, no ZWrite
    # A blended pass that still writes depth (``ZWrite On`` in the shader's
    # own render state, ``textures.BLEND_ZWRITE``, §7.376).  Ignored unless
    # ``blend``: the opaque passes write depth anyway.
    depth_write: bool = False
    # Unity's render queue for this material: ``m_CustomRenderQueue`` when set,
    # else the shader's ``Queue`` tag; -1 when the bake predates it (§7.379).
    # The blended pass draws in this order, as Unity's transparent queue does.
    render_queue: int = -1
    # Illusion's one use of the stencil buffer, transcribed from the shaders'
    # serialised render state (``toon.STENCIL_WRITERS`` / ``STENCIL_MASKED``):
    # the eye and eyebrow sheets stamp ref 2 where they draw, and the *front*
    # hair alone refuses to draw over it.  Both are 0 on every other material.
    stencil_write: int = 0          # Comp Always, Pass Replace with this ref
    stencil_reject: int = 0         # Comp NotEqual against this ref, Pass Keep
    stencil_clip: float = 0.0       # the writer's own discard, so the stamp
                                    # has the shape of the ink not of the quad
    two_sided: bool = False
    cast_outline: bool = True
    unlit: float = 0.0              # 0 = fully shaded, 1 = flat colour (eyes/highlights)
    # ---- a shading family with inputs of its own (``toon_family`` ≥ 20:
    # ``gl/shaders_endfield.py``, filled by ``endfield/shading.py``) ---------
    #: ``((kind, uniform, args), …)`` sent beside the fields above
    extra_uniforms: tuple = ()
    #: GL ``(source, destination)`` factors of a blended pass that is not plain
    #: alpha; ``None`` = ``SrcAlpha, OneMinusSrcAlpha``
    blend_func: tuple | None = None
    #: ``(ref, compare, read mask, write mask)`` with compare ``"always"`` /
    #: ``"equal"`` / ``"gequal"``: tested when the part draws, and the ref is
    #: written (through the write mask) where it passes
    stencil_state: tuple | None = None
    #: a blended pass that tests ``Less`` instead of ``LEqual``
    depth_less: bool = False
    #: the outline as that family states it: the base colour's (brightness,
    #: saturation), and how far the hull is pushed back
    outline_tone: tuple = (1.0, 1.0, 0.0)
    #: the hull is pushed along the mesh's averaged normal (``_OutlineAverageNormal``, ``Mesh.outline_normals``)
    outline_average: bool = False

    def __setattr__(self, name: str, value) -> None:
        # Every set moves ``_rev``.  The renderer keeps a material's uniform
        # values on the material and recomputes them only when this moved
        # (``Renderer._material_uniforms``, §7.555) -- so a field has to be
        # *assigned*, never mutated in place, for a draw to see it.
        object.__setattr__(self, name, value)
        d = self.__dict__
        d["_rev"] = d.get("_rev", 0) + 1

    def copy(self) -> "Material":
        return Material(**{f.name: getattr(self, f.name) for f in fields(self)})


#: ``UnityEngine.Rendering.ShadowCastingMode``'s Off and ShadowsOnly (On is 1, TwoSided 2)
CAST_OFF, CAST_ONLY = 0, 3


@dataclass
class Part:
    """A drawable chunk: geometry + material + toggles."""
    name: str
    mesh: Mesh
    material: Material = field(default_factory=Material)
    visible: bool = True
    group: str = "body"     # body / hair / clothes / accessory / face ...
    layer: int = 0          # draw order hint (higher = later)
    dynamic: bool = False   # participates in the cloth/hair solver
    skinned: bool = True    # False once the solver writes model-space positions
    # Vertex data rewritten now and then rather than every frame (a rest shape
    # that follows the pose): whoever rewrites it sets this, and the renderer
    # uploads the part once more and clears it.
    stale: bool = False
    # How the part follows its bones (``mesh/skinning.py``): the weighted sum
    # of their matrices, or of their rigid motions, or — "mixed" — each vertex
    # its own share of the two (``Mesh.skin_dual``).  A surface built against
    # one keeps its shape only under that one.
    skinning: str = "linear"
    # the source renderer's ``m_CastShadows`` (:data:`CAST_OFF` … :data:`CAST_ONLY`)
    cast_shadows: int = 1
    # Illusion ships facial expressions as blend shapes on the imported mesh;
    # a Origin head part carries them here for ``build_face_morphs`` to register.
    origin_shapes: list | None = None
    # Set on the genital shells that come out of the body prefab: they are
    # handed the body's own albedo key, so they have to follow it every time it
    # is re-stored (undress, budget eviction — §7.104).  Recorded where the
    # sharing is created; a name test cannot tell them from an uncensor mesh,
    # which brings its own albedo and must keep it.
    body_skin: bool = False
    # ``ChaForegroundComponent`` put this sheet's render queue *below* the front
    # hair's 2850, so it gets to stamp the stencil before the fringe is drawn
    # and the fringe is cut out around it.  Which of the ten sheets are in this
    # state is per character (the card's tri-state × the player's setting), so
    # it lives on the Part and not on the material — see ``toon.foreground``.
    foreground: bool = False
    # The card itself switched this renderer off (MaterialEditor's character
    # rows, KK_InvisibleBody).  Kept apart from ``visible`` because Free H,
    # life and undress set ``visible = True`` on whole groups every scene —
    # they decide what a *scene* shows, and none of them may bring back what
    # the card hid.  :meth:`Model.visible_parts` is the renderer's only door.
    card_hidden: bool = False
    # The person at the screen took this part off (Free H's body, face and hair
    # pages).  Apart from ``visible`` for the same reason ``card_hidden`` is:
    # undress, the accessory states and the expressions write ``visible`` every
    # scene, and none of them may bring back what was taken off by hand.
    user_hidden: bool = False
    # The face's state switched this off: ``ChangeTongueState``'s tongues and the
    # ``tearsLv`` sheets.  Apart from ``visible`` for the same reason again: the
    # scene passes that show whole groups would bring back a tongue the mouth
    # pattern put away, and all three tear sheets at once (§7.1130).
    face_hidden: bool = False

    def copy(self) -> "Part":
        part = Part(self.name, self.mesh.copy(), self.material.copy(),
                    self.visible, self.group, self.layer, self.dynamic,
                    self.skinned, self.origin_shapes)
        part.skinning = self.skinning
        part.body_skin = self.body_skin
        part.foreground = self.foreground
        part.card_hidden = self.card_hidden
        part.user_hidden = self.user_hidden
        part.face_hidden = self.face_hidden
        return part


#: Every :class:`Model` alive in this process.  The texture store asks it which
#: maps are still referenced before it lets one go: a map that a live part's
#: material points at is being drawn, and dropping it painted that part white
#: (§7.499).  Weak, so a character nobody holds any more stops counting.
LIVE_MODELS: "weakref.WeakSet[Model]" = weakref.WeakSet()


class Model:
    """A skinned model: parts bound to a skeleton.

    Registers itself in :data:`LIVE_MODELS` for as long as it lives.
    """

    def __init__(self, name: str = "model", skeleton=None):
        self.name = name
        self.parts: list[Part] = []
        self.skeleton = skeleton
        LIVE_MODELS.add(self)

    def add(self, part: Part) -> Part:
        self.parts.append(part)
        return part

    def add_mesh(self, name: str, mesh: Mesh, material: Material | None = None,
                 group: str = "body", layer: int = 0) -> Part:
        return self.add(Part(name, mesh, material or Material(name), True, group, layer))

    def get(self, name: str) -> Part | None:
        for p in self.parts:
            if p.name == name:
                return p
        return None

    def remove(self, name: str) -> None:
        self.parts = [p for p in self.parts if p.name != name]

    def by_group(self, group: str) -> list[Part]:
        return [p for p in self.parts if p.group == group]

    def visible_parts(self, shadow: bool = False) -> list[Part]:
        """The parts drawn, in layer order; with *shadow*, the ones the shadow pass draws instead.

        ``Renderer.shadowCastingMode`` (:attr:`Part.cast_shadows`): an Off part casts nothing and a
        ShadowsOnly one is drawn only as a shadow.
        """
        cast = (lambda p: p.cast_shadows != CAST_OFF) if shadow else (lambda p: p.cast_shadows != CAST_ONLY)
        return sorted((p for p in self.parts
                       if p.visible and not p.card_hidden and not p.user_hidden and not p.face_hidden
                       and p.mesh.tri_count and cast(p)),
                      key=lambda p: p.layer)

    def total_tris(self) -> int:
        return sum(p.mesh.tri_count for p in self.parts)

    def bounds(self):
        lo = np.array([1e9, 1e9, 1e9])
        hi = -lo
        for p in self.parts:
            if p.mesh.vertex_count:
                a, b = p.mesh.bounds()
                lo = np.minimum(lo, a)
                hi = np.maximum(hi, b)
        if lo[0] > 1e8:
            return np.zeros(3), np.zeros(3)
        return lo, hi

    def __repr__(self):
        return f"<Model {self.name} parts={len(self.parts)} tris={self.total_tris()}>"
