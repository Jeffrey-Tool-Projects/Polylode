"""``ShaderArchive-*.ushaderbytecode`` (Unreal 4.26's FSerializedShaderArchive): its table of contents and code.

Layout: a version (2), then arrays — shader map hashes (SHA-1), shader hashes, shader map entries (first index
into the shader indices, count, first preload entry, preload count), shader entries (offset, size,
uncompressed size, frequency), preload entries, shader indices — then the code, each shader LZ4-compressed
when its size differs from its uncompressed size.  Wuthering Waves' character materials are in the
``Client_Chunk`` archive.
"""
from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass, field
from pathlib import Path

#: a pixel shader's frequency in this archive (5; 0 is the vertex shader)
PIXEL = 5


@dataclass
class Archive:
    path: Path
    map_hashes: list[bytes]
    entries: list[tuple[int, int, int, int]]
    map_entries: list[tuple[int, int, int, int]]
    indices: list[int]
    code_start: int
    _by_hash: dict[bytes, int] = field(default_factory=dict, repr=False)

    def map_index(self, digest: bytes) -> int | None:
        if not self._by_hash:
            self._by_hash = {h: i for i, h in enumerate(self.map_hashes)}
        return self._by_hash.get(digest)

    def shaders(self, map_index: int) -> list[int]:
        first, count, _preload, _preloads = self.map_entries[map_index]
        return self.indices[first:first + count]

    def frequency(self, shader: int) -> int:
        return self.entries[shader][3]

    def code(self, shader: int) -> bytes:
        """Shader *shader*'s code, decompressed: Unreal's header, the DXBC, Unreal's optional data."""
        import lz4.block
        offset, size, uncompressed, _frequency = self.entries[shader]
        with open(self.path, "rb") as stream:
            stream.seek(self.code_start + offset)
            data = stream.read(size)
        return data if size == uncompressed else lz4.block.decompress(data, uncompressed_size=uncompressed)


def read(path: Path) -> Archive:
    """*path*'s table of contents (the code stays in the file)."""
    with open(path, "rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        at = 0

        def take(fmt: str):
            nonlocal at
            values = struct.unpack_from("<" + fmt, data, at)
            at += struct.calcsize(fmt)
            return values

        def array(fmt: str, size: int) -> list:
            nonlocal at
            (count,) = take("i")
            items = [struct.unpack_from("<" + fmt, data, at + size * i) for i in range(count)]
            at += size * count
            return items

        (version,) = take("I")
        if version != 2:
            raise ValueError(f"{path}: shader archive version {version}, not 2")
        (count,) = take("i")
        map_hashes = [bytes(data[at + 20 * i:at + 20 * i + 20]) for i in range(count)]
        at += 20 * count
        (count,) = take("i")
        at += 20 * count                                  # the shaders' own hashes: not needed here
        map_entries = array("IIII", 16)
        entries = array("QIIB", 17)
        array("qq", 16)                                   # preload entries
        indices = [i[0] for i in array("I", 4)]
        return Archive(Path(path), map_hashes, entries, map_entries, indices, at)
