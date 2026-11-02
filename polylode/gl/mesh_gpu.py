"""GPU-side mesh: one interleaved VBO + IBO per mesh."""
from __future__ import annotations

import ctypes

import numpy as np
from OpenGL import GL

from ..mesh.mesh import Mesh
from ..mesh.skin import MAX_INFLUENCES, pad_stream, validate_streams
from ..mesh.tangent import validate as validate_tangents

# pos(3) normal(3) uv(2) color(4) boneIdx(4) boneWt(4) uv1(2) boneIdx1(4) boneWt1(4)
# uv1 is always present in the buffer even though only Illusion's hair fills
# it: a per-mesh vertex layout would mean a second VAO shape and a second
# shader permutation for two floats.
# Authored tangent xyzw and presence follow the existing fields. A present zero
# direction is preserved and is distinct from a mesh without a source stream.
# Then a vertex's share of the dual-quaternion blend (``Mesh.skin_dual``):
# zero unless the part's skinning is mixed, where the vertex shader reads it.
# Last, the averaged normal the outline hull is pushed along and its presence
# (``Mesh.outline_rest``, ``_OutlineAverageNormal``): zero on a mesh without one.
STRIDE_FLOATS = 40
STRIDE = STRIDE_FLOATS * 4
LAYOUT = [(0, 3, 0), (1, 3, 3), (2, 2, 6), (3, 4, 8), (4, 4, 12), (5, 4, 16),
          (6, 2, 20), (7, 4, 22), (8, 4, 26), (9, 4, 30), (10, 1, 34), (11, 1, 35), (12, 4, 36)]


def interleave(mesh: Mesh) -> np.ndarray:
    """Upload complete authored xyzw and an explicit presence flag; absence stays absent."""
    n = mesh.vertex_count
    validate_streams(mesh.bone_idx, mesh.bone_wt, n)
    validate_tangents(mesh.tangents, n)
    joints = pad_stream(mesh.bone_idx, MAX_INFLUENCES)
    weights = pad_stream(mesh.bone_wt, MAX_INFLUENCES)
    buf = np.zeros((n, STRIDE_FLOATS), dtype=np.float32)
    buf[:, 0:3], buf[:, 3:6] = mesh.skin_rest()
    buf[:, 6:8] = mesh.uvs
    buf[:, 8:12] = mesh.colors
    buf[:, 12:16] = joints[:, :4]
    buf[:, 16:20] = weights[:, :4]
    buf[:, 22:26] = joints[:, 4:]
    buf[:, 26:30] = weights[:, 4:]
    uv1 = getattr(mesh, "uv1", None)
    if uv1 is not None and len(uv1) == n:
        buf[:, 20:22] = uv1
    if len(mesh.tangents):
        buf[:, 30:34] = mesh.tangents
        buf[:, 34] = 1.0
    if len(mesh.skin_dual) == n:
        buf[:, 35] = mesh.skin_dual[:, 0]
    outline = mesh.outline_rest()
    if len(outline) == n:
        buf[:, 36:39] = outline
        buf[:, 39] = 1.0
    return buf


def index_buffer(mesh: Mesh) -> tuple[np.ndarray, int, int]:
    """``(indices, surface count, hull count)``: the surface's triangles, then the outline hull's.

    A mesh with the averaged outline normal (``Mesh.outline_normals``, the
    ``HGRP/CharacterNPR`` outline) draws its hull without its folded faces
    (``mesh.outline_normals.hull_faces``); every other mesh draws its hull with
    the surface's own triangles, and the buffer holds them once.
    """
    from ..mesh.outline_normals import hull_faces
    idx = np.ascontiguousarray(mesh.indices, dtype=np.uint32)
    if len(mesh.outline_normals) != mesh.vertex_count or not mesh.vertex_count:
        return idx, len(idx), len(idx)
    hull = hull_faces(mesh).astype(np.uint32).ravel()
    if len(hull) == len(idx):
        return idx, len(idx), len(idx)
    return np.ascontiguousarray(np.concatenate([idx, hull])), len(idx), len(hull)


class GPUMesh:
    def __init__(self, mesh: Mesh, dynamic: bool = False):
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.ibo = GL.glGenBuffers(1)
        self.count = 0
        self._nbytes = 0
        #: the interleaved vertices as last sent (:meth:`refresh` compares against them)
        self._buf = None
        #: the mesh object last uploaded whole: a part handed another one is uploaded again
        self.source = None
        self.upload(mesh, dynamic)

    def refresh(self, mesh: Mesh) -> None:
        """Re-upload vertex data in place (cloth-solved parts change per frame).

        What a solve moves is the rest shape -- the positions and normals the
        skinning takes, the authored tangents, the outline's averaged normals
        -- so only those are put into the buffer kept from the last upload,
        and only the span of vertices that changed goes to the GPU: a lifted
        hair (``endfield.hair_lift``) moves a few hundred of its thousands.
        The streams that do not move (bones, weights, UVs, colours) are the
        ones the last whole upload put there; a mesh that gained or lost one
        is uploaded whole.  The buffer is kept from the first refresh on, so a
        mesh that is never refreshed keeps no copy.
        """
        buf = self._buf
        n = mesh.vertex_count
        outline = mesh.outline_rest()
        tangents = mesh.tangents
        if buf is None:
            # the first refresh: the whole buffer, kept from here on (a mesh never refreshed keeps none)
            data = interleave(mesh)
            if data.nbytes != self._nbytes:
                self.upload(mesh, dynamic=True)
                return
            self._buf = data
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, data.nbytes, data)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            return
        if (len(buf) != n or (len(tangents) == n) != bool(n and buf[0, 34] == 1.0)
                or (len(outline) == n) != bool(n and buf[0, 39] == 1.0)):
            self.upload(mesh, dynamic=True)
            return
        pos, nrm = mesh.skin_rest()
        streams = [(0, pos), (3, nrm)]
        if len(tangents) == n:
            streams.append((30, tangents))
        if len(outline) == n:
            streams.append((36, outline))
        changed = np.zeros(n, dtype=bool)
        for col, value in streams:
            value = np.asarray(value, dtype=np.float32)
            width = value.shape[1]
            moved = np.any(buf[:, col:col + width] != value, axis=1)
            if moved.any():
                buf[moved, col:col + width] = value[moved]
                changed |= moved
        rows = np.flatnonzero(changed)
        if not len(rows):
            return
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, lo * STRIDE, (hi - lo) * STRIDE, buf[lo:hi])
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def upload(self, mesh: Mesh, dynamic: bool = False) -> None:
        self.source = mesh
        data = interleave(mesh)
        self._buf = None
        idx, self.count, self.hull_count = index_buffer(mesh)
        #: where the hull's triangles start in the index buffer: after the surface's, or the same ones
        self.hull_offset = self.count if len(idx) > self.count else 0
        self._nbytes = data.nbytes
        usage = GL.GL_DYNAMIC_DRAW if dynamic else GL.GL_STATIC_DRAW
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, usage)
        for loc, size, off in LAYOUT:
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, GL.GL_FALSE, STRIDE,
                                     ctypes.c_void_p(off * 4))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)
        GL.glBindVertexArray(0)

    def draw(self) -> None:
        if not self.count:
            return
        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self.count, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

    def draw_hull(self) -> None:
        """The outline hull's triangles (:func:`index_buffer`): the surface's but for its folds."""
        if not self.hull_count:
            return
        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self.hull_count, GL.GL_UNSIGNED_INT,
                          ctypes.c_void_p(self.hull_offset * 4))
        GL.glBindVertexArray(0)

    def release(self) -> None:
        if self.vao:
            GL.glDeleteVertexArrays(1, [self.vao])
            GL.glDeleteBuffers(1, [self.vbo])
            GL.glDeleteBuffers(1, [self.ibo])
            self.vao = self.vbo = self.ibo = 0


class LineBatch:
    """Dynamic coloured line list for grids, gizmos and bone overlays."""

    def __init__(self):
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.count = 0
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 28, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 28, ctypes.c_void_p(12))
        GL.glBindVertexArray(0)

    def upload(self, verts: np.ndarray, colors: np.ndarray) -> None:
        n = len(verts)
        data = np.zeros((n, 7), dtype=np.float32)
        data[:, 0:3] = verts
        data[:, 3:7] = colors
        self.count = n
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
        GL.glBindVertexArray(0)

    def draw(self) -> None:
        if not self.count:
            return
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self.count)
        GL.glBindVertexArray(0)

    def release(self) -> None:
        if self.vao:
            GL.glDeleteVertexArrays(1, [self.vao])
            GL.glDeleteBuffers(1, [self.vbo])
            self.vao = self.vbo = 0


def make_grid(size: float = 4.0, step: float = 0.25, y: float = 0.0,
              color=(0.35, 0.35, 0.42, 0.45), axis_color=(0.55, 0.45, 0.60, 0.8)):
    verts, cols = [], []
    n = int(size / step)
    for i in range(-n, n + 1):
        x = i * step
        major = (i == 0)
        c = axis_color if major else color
        verts += [[x, y, -size], [x, y, size], [-size, y, x], [size, y, x]]
        cols += [c, c, c, c]
    return np.array(verts, dtype=np.float32), np.array(cols, dtype=np.float32)
