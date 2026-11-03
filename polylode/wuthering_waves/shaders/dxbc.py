"""A Wuthering Waves shader's code as Unreal 4.26 stores it for Direct3D 11: what it reads and its assembly.

The stored code is Unreal's header, then the DXBC container, then Unreal's optional data.  The header is the
shader's ``FShaderResourceTable``: a bit per uniform buffer that holds resources, then arrays of packed
entries — shader resource views, samplers, UAVs, layout hashes, textures — each entry
``uniform buffer << 24 | resource index << 8 | register``, a buffer's run ending in 0xFFFFFFFF.  The optional
data names the uniform buffers by their index, which is also their constant buffer's register
(``View``, ``Primitive``, ``ShadowDepthPass``, ``Material``…).  A uniform buffer's resources are its textures
and samplers by their order in the buffer: a material's 2D texture *i* is resource 2·i, its sampler 2·i + 1.

The assembly comes from ``d3dcompiler_47.dll`` (``D3DDisassemble``), on every Windows.
"""
from __future__ import annotations

import ctypes
import re
import struct
from dataclasses import dataclass, field


@dataclass
class Shader:
    asm: str
    textures: dict[int, tuple[int, int]]       # register → (uniform buffer, resource index)
    samplers: dict[int, tuple[int, int]]
    buffers: dict[str, int]                    # uniform buffer name → constant buffer register
    inputs: dict[int, list[tuple[str, int, str]]] = field(default_factory=dict)  # v# → [(semantic, index, mask)]


def disassemble(dxbc: bytes) -> str:
    library = ctypes.WinDLL("d3dcompiler_47.dll")
    blob = ctypes.c_void_p()
    result = library.D3DDisassemble(ctypes.c_char_p(dxbc), ctypes.c_size_t(len(dxbc)), 0, None, ctypes.byref(blob))
    if result != 0:
        raise OSError(f"D3DDisassemble 0x{result & 0xffffffff:08x}")
    table = ctypes.cast(ctypes.cast(blob, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    pointer = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(table[3])
    size = ctypes.WINFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(table[4])
    release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(table[2])
    text = ctypes.string_at(pointer(blob), size(blob)).decode("ascii", "replace").replace("\x00", "")
    release(blob)
    return text


def _table(entries: list[int]) -> dict[int, tuple[int, int]]:
    bound = {}
    for entry in entries:
        if entry == 0xFFFFFFFF or entry < 0x01000000:
            continue                                   # a buffer's run end, or the runs' offsets
        bound[entry & 0xFF] = (entry >> 24, (entry >> 8) & 0xFFFF)
    return bound


def _inputs(asm: str) -> dict[int, list[tuple[str, int, str]]]:
    signature = asm.split("// Input signature:", 1)[-1].split("// Output signature:", 1)[0]
    inputs: dict[int, list[tuple[str, int, str]]] = {}
    for semantic, index, mask, register in re.findall(r"//\s+(\w+)\s+(\d+)\s+([xyzw]+)\s+(\d+)\s", signature):
        inputs.setdefault(int(register), []).append((semantic, int(index), mask))
    return inputs


def read(code: bytes) -> Shader:
    """The stored code *code* (``archive.Archive.code``) as a :class:`Shader`."""
    start = code.find(b"DXBC")
    if start < 0:
        raise ValueError("no DXBC in this shader's code")
    at = 4                                             # the resource-table bits
    arrays = []
    for _ in range(5):
        (count,) = struct.unpack_from("<i", code, at)
        arrays.append(list(struct.unpack_from(f"<{count}I", code, at + 4)))
        at += 4 + 4 * count
    srv, samplers, _uav, _hashes, textures = arrays
    (size,) = struct.unpack_from("<I", code, start + 24)
    tail = code[start + size:]
    buffers = {}
    for match in re.finditer(rb"([\x02-\x40])\x00\x00\x00([A-Za-z_][A-Za-z0-9_]*)\x00", tail):
        if len(match.group(2)) + 1 == match.group(1)[0]:
            buffers.setdefault(match.group(2).decode(), len(buffers))
    asm = disassemble(code[start:start + size])
    return Shader(asm=asm, textures=_table(textures) or _table(srv), samplers=_table(samplers), buffers=buffers,
                  inputs=_inputs(asm))
