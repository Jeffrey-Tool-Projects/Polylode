"""A Wuthering Waves material instance's inline shader map, out of its ``.uexp``: its uniform expressions.

An instance with static switches of its own (``bHasStaticPermutationResource``) carries its compiled material
resources: per resource, a table of the names it uses, then a frozen memory image (Unreal 4.26's
FMemoryImage: a size, the bytes, then vtable, script-name and minimal-name patches — each script name with the
offsets of the frozen bytes that name it).  In the image, the uniform expression set's arrays lie end to end:

    vector preshader headers (8 bytes: opcode offset, size) · scalar preshader headers · scalar parameters
    (28 bytes; the default at +24) · vector parameters (40; the default at +24) · 2D texture parameters
    (32; the referenced texture's index at +24) · the other texture kinds' parameters · the preshader opcodes

Each array is a frozen ``TMemoryImageArray`` header (a pointer stored as ``offset << 1 | 1`` from where it
is, a count, a capacity) at 16-byte steps — eight texture kinds' arrays from the 2D one on, the opcodes' at
+192 from the first — and the chain of ends meeting starts is how the set is found.  A
parameter's name is the script name whose patch points at its record.  None of this is in CUE4Parse's dump
(its ``LoadedMaterialResources`` come out empty for these instances).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass
class Uniforms:
    """One resource's uniform expressions: what the Material uniform buffer holds and the textures it binds."""
    vector_code: list[bytes]          # one preshader per float4 slot
    scalar_code: list[bytes]          # one per scalar, four to a float4 after the vectors
    scalars: list[tuple[str, float]]  # the scalar parameters the code names by index: name, default
    vectors: list[tuple[str, tuple]]  # the vector parameters: name, default RGBA
    textures: list[tuple[str | None, int]]  # 2D texture parameters, Material_Texture2D_<i>: name, referenced index


def _names(data: bytes, at: int) -> tuple[list[str], int] | None:
    """A name table at *at* (count, then length, chars, NUL and a 4-byte hash each): its names and end."""
    (count,) = struct.unpack_from("<i", data, at)
    if not 0 < count < 20000:
        return None
    names, p = [], at + 4
    for _ in range(count):
        if p + 4 > len(data):
            return None
        (length,) = struct.unpack_from("<i", data, p)
        if not 1 < length < 512 or p + 8 + length > len(data) or data[p + 3 + length] != 0:
            return None
        text = data[p + 4:p + 3 + length]
        if not all(32 <= c < 127 for c in text):
            return None
        names.append(text.decode("ascii"))
        p += 8 + length
    return names, p


def _frozen(data: bytes, names: list[str], after: int):
    """The frozen image that uses *names*, after *after*: (its bytes, {frozen offset: name})."""
    count = len(names)
    for q in range(after, len(data) - 12, 4):
        vtables, script, minimal = struct.unpack_from("<3i", data, q)
        if script != count or not (0 <= vtables < 1000 and 0 <= minimal < 1000):
            continue
        for p in range(q - 8, after - 4, -1):
            (size,) = struct.unpack_from("<i", data, p)
            if p + 4 + size == q:
                break
        else:
            continue
        r = q + 12
        for _ in range(vtables):                          # type hash, patch count, (vtable offset, offset)…
            (n,) = struct.unpack_from("<i", data, r + 8)
            r += 12 + 8 * n
        named = {}
        for _ in range(script):
            index, _number, n = struct.unpack_from("<3i", data, r)
            r += 12
            for k in range(n):
                named[struct.unpack_from("<i", data, r + 4 * k)[0]] = names[index]
            r += 4 * n
        return data[p + 4:q], named, r
    return None


def _header(image: bytes, at: int):
    value, count, capacity = struct.unpack_from("<qii", image, at)
    if not value & 1 or not 0 <= count <= capacity:
        return None
    return at + (value >> 1), count


def _uniforms(image: bytes, named: dict[int, str]) -> Uniforms | None:
    for at in range(0, min(len(image) - 96, 4096), 8):
        heads = [_header(image, at + 16 * k) for k in range(5)]
        if None in heads:
            continue
        (vp, vn), (sp, sn), (pp, pn), (qp, qn), (tp, tn) = heads
        if not (vn and sn and vp + 8 * vn == sp and sp + 8 * sn == pp and pp + 28 * pn == qp and qp + 40 * qn == tp):
            continue
        # eight texture-kind arrays (2D first) from +64, then the opcodes' array at +192, after them all
        head = _header(image, at + 192)
        if head is None or head[0] < tp + 32 * tn:
            continue
        code_at, code_count = head
        code = image[code_at:code_at + code_count]
        pieces = lambda base, n: [code[o:o + s] for o, s in (struct.unpack_from("<II", image, base + 8 * i)
                                                               for i in range(n))]
        return Uniforms(
            vector_code=pieces(vp, vn), scalar_code=pieces(sp, sn),
            scalars=[(named.get(pp + 28 * i, ""), struct.unpack_from("<f", image, pp + 28 * i + 24)[0])
                     for i in range(pn)],
            vectors=[(named.get(qp + 40 * i, ""), struct.unpack_from("<4f", image, qp + 40 * i + 24))
                     for i in range(qn)],
            textures=[(named.get(tp + 32 * i), struct.unpack_from("<i", image, tp + 32 * i + 24)[0])
                      for i in range(tn)])
    return None


def read(uexp: bytes) -> list[Uniforms]:
    """Every resource's uniform expressions in a material instance's ``.uexp``, in file order."""
    found, at = [], 0
    while at < len(uexp) - 8:
        table = _names(uexp, at) if uexp[at + 4:at + 8] and struct.unpack_from("<i", uexp, at)[0] > 0 else None
        if table is None:
            at += 1
            continue
        names, end = table
        frozen = _frozen(uexp, names, end)
        if frozen is None:
            at = end
            continue
        image, named, after = frozen
        uniforms = _uniforms(image, named)
        if uniforms is not None:
            found.append(uniforms)
        at = after
    return found
