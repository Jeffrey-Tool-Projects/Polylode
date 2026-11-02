"""The managed resources embedded in a .NET assembly, without an IL library.

Some BepInEx plugins ship their assets inside the DLL (``KK_LewdCrestX``'s
crest bundle is ``GetManifestResourceStream("KK_LewdCrestX.crests")``).  The
converter runs on the user's machine and the release does not carry
``dnfile`` (``build.py`` keeps it to ``tools/``), so this reads the structures
it needs straight from the file format (ECMA-335 II.22–25):

* PE: ``e_lfanew`` → COFF header → optional header (PE32 / PE32+) → data
  directory 14, the CLI header; the section table maps RVAs to file offsets.
* CLI header: the metadata directory (offset 8) and the ``Resources``
  directory (offset 24).
* Metadata root → the ``#~`` (or ``#-``) stream → row counts and row sizes of
  every table up to ``ManifestResource`` (0x28), whose rows are
  ``Offset, Flags, Name, Implementation``; ``Implementation == 0`` means the
  bytes are in this file, at ``Resources + Offset`` as ``uint32 length`` then
  the data.

The offsets have to come from that table: the resources are *not* reliably
back to back — a Roslyn build pads each to 8 bytes, older compilers do not
(``KK_StudioTextPlugin``'s second resource is at 5927), so walking the
directory is a guess that works on most files and misreads the rest.
Checked against ``dnfile`` on every plugin of the local install (§7.452).
"""
from __future__ import annotations

import struct
from pathlib import Path

#: ECMA-335 II.24.2.6 coded indices: (tag bits, tables)
_CODED = {
    "TypeDefOrRef": (2, (2, 1, 27)),
    "TypeOrMethodDef": (1, (2, 6)),
    "HasConstant": (2, (4, 8, 23)),
    "HasCustomAttribute": (5, (6, 4, 1, 2, 8, 9, 10, 0, 14, 23, 20, 17, 26,
                               27, 32, 35, 38, 39, 40, 42, 44, 43)),
    "HasFieldMarshal": (1, (4, 8)),
    "HasDeclSecurity": (2, (2, 6, 32)),
    "MemberRefParent": (3, (2, 1, 26, 6, 27)),
    "HasSemantics": (1, (20, 23)),
    "MethodDefOrRef": (1, (6, 10)),
    "MemberForwarded": (1, (4, 6)),
    "Implementation": (2, (38, 35, 39)),
    "CustomAttributeType": (3, (6, 10)),
    "ResolutionScope": (2, (0, 26, 35, 1)),
}

#: ECMA-335 II.22, tables 0x00–0x28: an int is a fixed 2- or 4-byte field,
#: ``"S"`` / ``"G"`` / ``"B"`` a heap index, ``("T", n)`` a simple index into
#: table *n*, any other str a coded index (:data:`_CODED`)
_SCHEMA = {
    0: (2, "S", "G", "G", "G"),
    1: ("ResolutionScope", "S", "S"),
    2: (4, "S", "S", "TypeDefOrRef", ("T", 4), ("T", 6)),
    3: (("T", 4),),
    4: (2, "S", "B"),
    5: (("T", 6),),
    6: (4, 2, 2, "S", "B", ("T", 8)),
    7: (("T", 8),),
    8: (2, 2, "S"),
    9: (("T", 2), "TypeDefOrRef"),
    10: ("MemberRefParent", "S", "B"),
    11: (2, "HasConstant", "B"),
    12: ("HasCustomAttribute", "CustomAttributeType", "B"),
    13: ("HasFieldMarshal", "B"),
    14: (2, "HasDeclSecurity", "B"),
    15: (2, 4, ("T", 2)),
    16: (4, ("T", 4)),
    17: ("B",),
    18: (("T", 2), ("T", 20)),
    19: (("T", 20),),
    20: (2, "S", "TypeDefOrRef"),
    21: (("T", 2), ("T", 23)),
    22: (("T", 23),),
    23: (2, "S", "B"),
    24: (2, ("T", 6), "HasSemantics"),
    25: (("T", 2), "MethodDefOrRef", "MethodDefOrRef"),
    26: ("S",),
    27: ("B",),
    28: (2, "MemberForwarded", "S", ("T", 26)),
    29: (4, ("T", 4)),
    30: (4, 4),
    31: (4,),
    32: (4, 2, 2, 2, 2, 4, "B", "S", "S"),
    33: (4,),
    34: (4, 4, 4),
    35: (2, 2, 2, 2, 4, "B", "S", "S", "B"),
    36: (4, ("T", 35)),
    37: (4, 4, 4, ("T", 35)),
    38: (4, "S", "B"),
    39: (4, 4, "S", "S", "Implementation"),
    40: (4, 4, "S", "Implementation"),
}
MANIFEST_RESOURCE = 40


def _sections(data: bytes, pe: int):
    n_sec = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dirs = opt + (96 if magic == 0x10B else 112)
    secs = []
    for i in range(n_sec):
        o = opt + opt_size + 40 * i
        vsize, vaddr, rsize, raddr = struct.unpack_from("<IIII", data, o + 8)
        secs.append((vaddr, max(vsize, rsize), raddr))
    return dirs, secs


def _offset(secs, rva: int) -> int:
    for vaddr, size, raddr in secs:
        if vaddr <= rva < vaddr + size:
            return rva - vaddr + raddr
    raise ValueError(f"RVA {rva:#x} 不在任何 section 裡")


def _streams(data: bytes, root: int) -> dict:
    if data[root:root + 4] != b"BSJB":
        raise ValueError("metadata root 沒有 BSJB 簽章")
    vlen = struct.unpack_from("<I", data, root + 12)[0]
    p = root + 16 + vlen + 2
    count = struct.unpack_from("<H", data, p)[0]
    p += 2
    out = {}
    for _ in range(count):
        off, size = struct.unpack_from("<II", data, p)
        end = data.index(b"\0", p + 8)
        name = data[p + 8:end].decode("ascii")
        out[name] = (root + off, size)
        p = p + 8 + ((end - (p + 8) + 4) & ~3)
    return out


def _strings(data: bytes, heap: int, idx: int) -> str:
    end = data.index(b"\0", heap + idx)
    return data[heap + idx:end].decode("utf-8", "replace")


def _manifest_rows(data: bytes, secs, cli: int) -> tuple[list, int]:
    """``([(offset, name, implementation), …], #Strings offset)``."""
    md_rva = struct.unpack_from("<I", data, cli + 8)[0]
    streams = _streams(data, _offset(secs, md_rva))
    tab = streams.get("#~") or streams.get("#-")
    if tab is None:
        raise ValueError("metadata 沒有 #~ stream")
    t = tab[0]
    heap_sizes = data[t + 6]
    valid = struct.unpack_from("<Q", data, t + 8)[0]
    p = t + 24
    rows = {}
    for i in range(64):
        if valid >> i & 1:
            rows[i] = struct.unpack_from("<I", data, p)[0]
            p += 4
    if heap_sizes & 0x40:
        p += 4                      # "extra data" after the row counts
    if not valid >> MANIFEST_RESOURCE & 1:
        return [], streams["#Strings"][0]
    heap = {"S": 4 if heap_sizes & 1 else 2, "G": 4 if heap_sizes & 2 else 2,
            "B": 4 if heap_sizes & 4 else 2}

    def width(col) -> int:
        if isinstance(col, int):
            return col
        if isinstance(col, tuple):
            return 2 if rows.get(col[1], 0) < 0x10000 else 4
        if col in heap:
            return heap[col]
        bits, tables = _CODED[col]
        big = max((rows.get(x, 0) for x in tables), default=0)
        return 2 if big < (1 << (16 - bits)) else 4

    sizes = {t: sum(width(c) for c in _SCHEMA[t])
             for t in range(MANIFEST_RESOURCE + 1) if t in rows}
    for table in range(MANIFEST_RESOURCE):
        p += rows.get(table, 0) * sizes.get(table, 0)
    s_heap = streams["#Strings"][0]
    out = []
    impl_w = width("Implementation")
    for _ in range(rows[MANIFEST_RESOURCE]):
        offset = struct.unpack_from("<I", data, p)[0]
        q = p + 8
        name_idx = struct.unpack_from("<I" if heap["S"] == 4 else "<H", data, q)[0]
        q += heap["S"]
        impl = struct.unpack_from("<I" if impl_w == 4 else "<H", data, q)[0]
        out.append((offset, _strings(data, s_heap, name_idx), impl))
        p += sizes[MANIFEST_RESOURCE]
    return out, s_heap


def manifest_resources(path: str | Path) -> dict[str, bytes]:
    """``{name: bytes}`` of every resource embedded in the assembly at *path*.

    Raises ``ValueError`` on a file that is not a .NET image — a reported
    failure, not an empty answer (``CLAUDE.md`` 一之一).
    """
    data = Path(path).read_bytes()
    if data[:2] != b"MZ":
        raise ValueError("不是 PE 檔（沒有 MZ）")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("不是 PE 檔（沒有 PE 簽章）")
    dirs, secs = _sections(data, pe)
    cli_rva = struct.unpack_from("<I", data, dirs + 8 * 14)[0]
    if not cli_rva:
        raise ValueError("不是 .NET 組件（沒有 CLI header）")
    cli = _offset(secs, cli_rva)
    rows, _ = _manifest_rows(data, secs, cli)
    res_rva = struct.unpack_from("<I", data, cli + 24)[0]
    out: dict[str, bytes] = {}
    for offset, name, impl in rows:
        if impl or not res_rva:
            continue                # lives in another file / assembly
        base = _offset(secs, res_rva) + offset
        n = struct.unpack_from("<I", data, base)[0]
        out[name] = data[base + 4:base + 4 + n]
    return out
