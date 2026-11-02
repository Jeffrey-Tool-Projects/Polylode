"""Framebuffer objects with optional multisampling and a sampleable depth.

The depth attachment is **depth24 + stencil8**, not plain depth24.  Illusion's
character shaders use the stencil buffer for exactly one thing — the eye and
eyebrow sheets stamp ref 2 and the front hair refuses to draw over it, which is
how the fringe gets cut out around them (``figure.toon.STENCIL_WRITERS``).  A
framebuffer with no stencil bits does not fail: ``glStencilFunc`` is accepted
and every test simply passes, so the fringe would draw straight over the
eyebrow and nothing anywhere would report a problem.

The colour attachment is **sRGB** (``srgb=True``) for the scene targets, which
is the encode boundary Koikatsu's own camera has: a Linear project renders
through ``RenderTextureReadWrite.Default`` = sRGB, so the bytes handed to the
image effects are gamma-encoded while the shaders wrote linear (§7.126).  With
``GL_FRAMEBUFFER_SRGB`` on, the GPU encodes on write, decodes for blending and
decodes again when the post pass samples the texture — all three are what the
engine does, and none of them can be reproduced by encoding in the fragment
shader (that one blends in the wrong space).
"""
from __future__ import annotations

from OpenGL import GL

from ..core import swallowed

# PyOpenGL's ``glTexImage2D`` wrapper maps the pixel *type* to an array type so
# it can size the upload, and its table has no entry for ``GL_UNSIGNED_INT_24_8``
# — allocating an empty packed depth-stencil texture raises ``KeyError`` before
# it ever reaches the driver.  One entry fixes it; the packed 24_8 texel is one
# 32-bit word, which is what ``GL_UNSIGNED_INT`` already says.
try:                                                     # pragma: no cover
    from OpenGL import images as _gl_images
    _gl_images.TYPE_TO_ARRAYTYPE.setdefault(GL.GL_UNSIGNED_INT_24_8,
                                            GL.GL_UNSIGNED_INT)
    _gl_images.TYPE_TO_BITS.setdefault(GL.GL_UNSIGNED_INT_24_8, 32)
except Exception as exc:                                 # noqa: BLE001
    swallowed.note("gl.framebuffer: TYPE_TO_ARRAYTYPE.setdefault", exc)


class Framebuffer:
    def __init__(self, width: int, height: int, samples: int = 0,
                 depth_texture: bool = False, name: str = "fbo",
                 mipmap: bool = False, srgb: bool = False):
        self.name = name
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.samples = max(0, int(samples))
        self.depth_texture = depth_texture and self.samples == 0
        self.mipmap = bool(mipmap) and self.samples == 0
        self.srgb = bool(srgb)
        self.fbo = 0
        self.color = 0
        self.depth = 0
        self.valid = False
        self._create()

    # ------------------------------------------------------------------
    def _create(self) -> None:
        self._destroy()
        w, h = self.width, self.height
        fmt = GL.GL_SRGB8_ALPHA8 if self.srgb else GL.GL_RGBA8
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)

        if self.samples > 0:
            self.color = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.color)
            GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, self.samples,
                                                fmt, w, h)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                         GL.GL_RENDERBUFFER, self.color)
            self.depth = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth)
            GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, self.samples,
                                                GL.GL_DEPTH24_STENCIL8, w, h)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,
                                         GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                         GL.GL_RENDERBUFFER, self.depth)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)
        else:
            self.color = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.color)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, fmt, w, h, 0,
                            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            minf = GL.GL_LINEAR_MIPMAP_LINEAR if self.mipmap else GL.GL_LINEAR
            for pn, pv in ((GL.GL_TEXTURE_MIN_FILTER, minf),
                           (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                           (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE),
                           (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
                GL.glTexParameteri(GL.GL_TEXTURE_2D, pn, pv)
            if self.mipmap:
                GL.glGenerateMipmap(GL.GL_TEXTURE_2D)     # allocate the chain
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                      GL.GL_TEXTURE_2D, self.color, 0)
            if self.depth_texture:
                self.depth = GL.glGenTextures(1)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth)
                # A packed depth-stencil texture still samples as depth in a
                # plain ``sampler2D``; the stencil half is only reachable
                # through the attachment, which is all the stencil test needs.
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH24_STENCIL8,
                                w, h, 0, GL.GL_DEPTH_STENCIL,
                                GL.GL_UNSIGNED_INT_24_8, None)
                for pn, pv in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST),
                               (GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST),
                               (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE),
                               (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, pn, pv)
                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,
                                          GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                          GL.GL_TEXTURE_2D, self.depth, 0)
            else:
                self.depth = GL.glGenRenderbuffers(1)
                GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth)
                GL.glRenderbufferStorage(GL.GL_RENDERBUFFER,
                                         GL.GL_DEPTH24_STENCIL8, w, h)
                GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,
                                             GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                             GL.GL_RENDERBUFFER, self.depth)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        self.valid = (status == GL.GL_FRAMEBUFFER_COMPLETE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _destroy(self) -> None:
        if self.fbo:
            GL.glDeleteFramebuffers(1, [self.fbo])
            self.fbo = 0
        if self.color:
            if self.samples > 0:
                GL.glDeleteRenderbuffers(1, [self.color])
            else:
                GL.glDeleteTextures(1, [self.color])
            self.color = 0
        if self.depth:
            if self.samples > 0 or not self.depth_texture:
                GL.glDeleteRenderbuffers(1, [self.depth])
            else:
                GL.glDeleteTextures(1, [self.depth])
            self.depth = 0

    # ------------------------------------------------------------------
    def resize(self, width: int, height: int) -> None:
        w, h = max(1, int(width)), max(1, int(height))
        if (w, h) != (self.width, self.height):
            self.width, self.height = w, h
            self._create()

    def bind(self) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, self.width, self.height)

    def regenerate_mipmaps(self) -> None:
        if not self.mipmap or not self.color:
            return
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def blit_to(self, other: "Framebuffer | int", color: bool = True,
                depth: bool = False) -> None:
        target = other.fbo if isinstance(other, Framebuffer) else int(other)
        tw = other.width if isinstance(other, Framebuffer) else self.width
        th = other.height if isinstance(other, Framebuffer) else self.height
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, target)
        mask = 0
        if color:
            mask |= GL.GL_COLOR_BUFFER_BIT
        if depth:
            mask |= GL.GL_DEPTH_BUFFER_BIT
        if mask:
            GL.glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, tw, th,
                                 mask, GL.GL_NEAREST if depth else GL.GL_LINEAR)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def release(self) -> None:
        self._destroy()
        self.valid = False
