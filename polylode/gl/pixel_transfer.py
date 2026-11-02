"""Scoped CPU texture uploads independent of another pass's pixel-store state."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from OpenGL import GL


@contextmanager
def cpu_unpack() -> Iterator[None]:
    """Use tightly packed CPU bytes, then restore PBO and every unpack parameter.

    Both raw pixels and original compressed blocks must ignore any previous
    renderer's row skips, strides or unpack buffer. No texture data is converted.
    """
    parameters = [GL.GL_UNPACK_ALIGNMENT, GL.GL_UNPACK_ROW_LENGTH, GL.GL_UNPACK_IMAGE_HEIGHT,
                  GL.GL_UNPACK_SKIP_PIXELS, GL.GL_UNPACK_SKIP_ROWS, GL.GL_UNPACK_SKIP_IMAGES]
    version = tuple(int(GL.glGetIntegerv(key)) for key in (GL.GL_MAJOR_VERSION, GL.GL_MINOR_VERSION))
    compressed = version >= (4, 2)
    if not compressed:
        compressed = any(GL.glGetStringi(GL.GL_EXTENSIONS, i) == b'GL_ARB_compressed_texture_pixel_storage'
                         for i in range(int(GL.glGetIntegerv(GL.GL_NUM_EXTENSIONS))))
    if compressed:
        parameters += [GL.GL_UNPACK_COMPRESSED_BLOCK_WIDTH, GL.GL_UNPACK_COMPRESSED_BLOCK_HEIGHT,
                       GL.GL_UNPACK_COMPRESSED_BLOCK_DEPTH, GL.GL_UNPACK_COMPRESSED_BLOCK_SIZE]
    previous = {key: int(GL.glGetIntegerv(key)) for key in parameters}
    buffer = int(GL.glGetIntegerv(GL.GL_PIXEL_UNPACK_BUFFER_BINDING))
    try:
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        for key in parameters:
            GL.glPixelStorei(key, 1 if key == GL.GL_UNPACK_ALIGNMENT else 0)
        yield
    finally:
        for key, value in previous.items():
            GL.glPixelStorei(key, value)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, buffer)
