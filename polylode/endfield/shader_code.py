"""An Endfield shader's compiled code, as the game ships it: a pass's variants, their DXBC and parameter blocks.

The ``Shader`` object the exporter writes as JSON holds, per sub-shader, an LZ4 blob per platform; its D3D11 blob
is an entry table (``(offset, length, segment)``) over segments, each entry a program (a variant's stages, each a
DXBC container) or a parameter block (its constant buffers — every variable's byte offset — and the slot each
buffer, texture and sampler has in the pixel and in the vertex stage).  A pass lists, per variant, its keywords and
the entries of its program and parameter block.  Read here so a material whose shader this program does not draw
can be run as the game compiled it (``vfx_bake``); the shaders are fetched once from the installation, into a
cache beside the converted store.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path

PLATFORM = "D3D11"
#: resource kinds in a variant's binding sets
SAMPLER, TEXTURE, CBUFFER, BUFFER = 0, 2, 6, 7
UNBOUND = 0xFF


class ShaderCodeError(ValueError):
    """A shader, pass or variant not found, or a blob not read."""


def cache_dir() -> Path:
    from . import store
    return store.root() / "shaders"


class _Reader:
    """Little-endian ints and length-prefixed, 4-aligned strings."""

    def __init__(self, data: bytes, at: int = 0):
        self.data, self.at = data, at

    def int(self) -> int:
        value = struct.unpack_from("<i", self.data, self.at)[0]
        self.at += 4
        return value

    def uint(self) -> int:
        return self.int() & 0xFFFFFFFF

    def str(self) -> str:
        size = self.int()
        if not 0 <= size <= 4096:
            raise ShaderCodeError(f"string length {size} at {self.at - 4}")
        text = self.data[self.at:self.at + size].decode("utf-8")
        self.at += (size + 3) & ~3
        return text


def blob_entries(doc: dict, lod: int = 0) -> list[bytes]:
    import lz4.block
    sub = doc["subShaderBlobs"][lod]
    platform = doc["platforms"].index(PLATFORM)
    packed = base64.b64decode(sub["m_CompressedBlob"])
    segments = [lz4.block.decompress(packed[off:off + size], uncompressed_size=full)
                for off, size, full in zip(sub["m_Offsets"][platform], sub["m_CompressedLengths"][platform],
                                           sub["m_DecompressedLengths"][platform])]
    head = segments[0]
    count = struct.unpack_from("<I", head, 0)[0]
    out = []
    for i in range(count):
        off, size, segment = struct.unpack_from("<III", head, 4 + 12 * i)
        out.append(segments[segment][off:off + size])
    return out


def dxbc_containers(data: bytes) -> list[bytes]:
    """Every DXBC container inside a program entry."""
    found = []
    for match in re.finditer(b"DXBC", data):
        start = match.start()
        if start + 32 > len(data):
            continue
        size = int.from_bytes(data[start + 24:start + 28], "little")
        if 0 < size <= len(data) - start:
            found.append(data[start:start + size])
    return found


@dataclass
class Params:
    """One variant's parameter block."""
    #: buffer name → (byte size, [(variable, byte offset, rows, columns, array)])
    buffers: dict = field(default_factory=dict)
    #: resource name → (kind, pixel-stage slot, vertex-stage slot)
    slots: dict = field(default_factory=dict)


def _parse_params(entry: bytes, params: Params) -> Params:
    reader = _Reader(entry, 4)
    buffers = reader.int() - 1
    reader.at = 24
    for _ in range(buffers):
        name, size = reader.str(), reader.int()
        variables = params.buffers.setdefault(name, (size, []))[1]
        for _v in range(reader.int()):
            variable = reader.str()
            _kind, rows, columns, _matrix, array, offset = (reader.int() for _i in range(6))
            variables.append((variable, offset, rows, columns, array))
        if reader.int():
            raise ShaderCodeError(f"{name}: struct parameters are not read")
    for _ in range(reader.int()):                 # the material's own bindings
        reader.str()
        texture = reader.int() == 0
        reader.at += 12 if texture else 8
    for _ in range(reader.int()):                 # binding sets: Global, PerMaterial…
        reader.str()
        _index, count, _last = reader.int(), reader.int(), reader.int()
        for _r in range(count):
            name = reader.str()
            _slot, kind, where, _flags = reader.int(), reader.int(), reader.uint(), reader.int()
            params.slots[name] = (kind, (where >> 16) & 0xFF, (where >> 24) & 0xFF)
    return params


def _common_params(program: dict, names: dict, params: Params) -> Params:
    for cb in program["m_CommonParameters"]["m_ConstantBuffers"]:
        name = names[cb["m_NameIndex"]]
        variables = params.buffers.setdefault(name, (cb["m_Size"], []))[1]
        for p in cb["m_VectorParams"]:
            variables.append((names[p["m_NameIndex"]], p["m_Index"], 1, p["m_Dim"], p["m_ArraySize"]))
        for p in cb["m_MatrixParams"]:
            variables.append((names[p["m_NameIndex"]], p["m_Index"], p["m_RowCount"], 4, p["m_ArraySize"]))
    return params


@dataclass
class Variant:
    """One compiled variant of a pass."""
    keywords: list[str]
    pixel: bytes                       # DXBC
    params: Params
    state: dict                        # the pass's render state (blend, culling…)
    tags: dict = field(default_factory=dict)   # the pass's tags (LIGHTMODE, RENDERTYPE…)
    vertex: bytes | None = None        # DXBC of the vertex stage

    @property
    def vertex_asm(self) -> str | None:
        from ..wuthering_waves.shaders.dxbc import disassemble
        return disassemble(self.vertex) if self.vertex is not None else None

    def registers(self, stage: str = "ps") -> tuple[dict[int, tuple[str, list]], dict[int, str], dict[int, str]]:
        """A stage's ``{cb register: (buffer, variables)}``, ``{t#: texture}``, ``{s#: sampler}``."""
        buffers, textures, samplers = {}, {}, {}
        for name, (kind, pixel, vertex) in self.params.slots.items():
            slot = pixel if stage == "ps" else vertex
            if slot == UNBOUND:
                continue
            if kind == CBUFFER and name in self.params.buffers:
                buffers[slot] = (name, self.params.buffers[name][1])
            elif kind in (TEXTURE, BUFFER):
                textures[slot] = name
            elif kind == SAMPLER:
                samplers[slot] = name
        return buffers, textures, samplers


def load(name: str) -> dict | None:
    for path in sorted(cache_dir().glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["m_ParsedForm"]["m_Name"] == name:
            return doc
    return None


def fetch(pipeline, names: list[str], log=print) -> None:
    """The shaders *names* exported from the installation into the cache."""
    rows = [row for name in names for row in pipeline.assets.shaders(name)]
    entries = {}
    for _name, source, offset, _path_id in rows:
        cab = pipeline.cabs.at(source, offset)
        if cab:
            entries[cab] = pipeline.cabs.entry(cab)
    if not entries:
        return
    from .pipeline import stage_root
    stage = stage_root() / "shader_stage"
    shutil.rmtree(stage, ignore_errors=True)
    exported = pipeline.exporter.run(list(entries.values()), stage, ["Shader"], log=log)
    cache_dir().mkdir(parents=True, exist_ok=True)
    wanted = {name for name, *_ in rows}
    for row in exported:
        path = Path(row["output"])
        if path.suffix == ".json" and row.get("name") in wanted:
            shutil.copyfile(path, cache_dir() / path.name)
    shutil.rmtree(stage, ignore_errors=True)


def tags_of(chosen: dict) -> dict[str, str]:
    """A pass's tags, keys upper-cased (the blob spells ``LIGHTMODE``, ``RenderType``…)."""
    return {t["Key"].upper(): t["Value"] for t in (chosen["m_State"].get("m_Tags", {}).get("tags") or [])}


def _pass(doc: dict, light_modes: tuple[str, ...]) -> dict:
    passes = doc["m_ParsedForm"]["m_SubShaders"][0]["m_Passes"]
    for mode in light_modes:
        for candidate in passes:
            if tags_of(candidate).get("LIGHTMODE") == mode or candidate["m_State"]["m_Name"] == mode:
                return candidate
    have = ", ".join(p["m_State"]["m_Name"] for p in passes)
    raise ShaderCodeError(f"{doc['m_ParsedForm']['m_Name']}: no pass of {light_modes} (has {have})")


def variant(doc: dict, keywords: list[str], light_modes: tuple[str, ...] = ("ForwardOnly", "UniversalForward",
                                                                                "SRPDefaultUnlit")) -> Variant:
    """The pass variant the game draws a material of *keywords* with: compiled for every one of them the pass
    knows and the fewest others (a material carries the keywords of all its passes)."""
    chosen = _pass(doc, light_modes)
    names_list = doc["m_ParsedForm"]["m_KeywordNames"]
    program = chosen["progFragment"] if chosen.get("progFragment", {}).get("m_PlayerSubPrograms") else chosen["progVertex"]
    options = []
    for tier, subs in enumerate(program["m_PlayerSubPrograms"]):
        for sub, block in zip(subs, program["m_ParameterBlobIndices"][tier]):
            options.append((sorted(names_list[k] for k in sub["m_KeywordIndices"]), sub["m_BlobIndex"], block))
    if not options:
        raise ShaderCodeError("the pass has no compiled variant")
    known = set().union(*(set(o[0]) for o in options))
    asked = set(keywords) & known
    fits = [o for o in options if asked <= set(o[0])]
    if not fits:
        raise ShaderCodeError(f"no variant has all of: {' '.join(sorted(asked))}")
    words, program_entry, block = min(fits, key=lambda o: (len(o[0]), o[0]))
    entries = blob_entries(doc)
    names = {e["Value"]: e["Key"] for e in chosen["m_NameIndices"]}
    params = _parse_params(entries[block], _common_params(chosen["progVertex"], names, Params()))
    from ..wuthering_waves.shaders.dxbc import disassemble
    stages = {}
    for code in dxbc_containers(entries[program_entry]):
        kind = re.search(r"^(ps|vs|gs|hs|ds)_\d", disassemble(code), re.M)
        if kind:
            stages.setdefault(kind[1], code)
    if "ps" not in stages:
        raise ShaderCodeError("the variant has no pixel stage")
    return Variant(words, stages["ps"], params, chosen["m_State"], tags_of(chosen), stages.get("vs"))


def property_defaults(doc: dict) -> dict[str, list[float]]:
    """Each property's default value, as the shader declares it."""
    out = {}
    for prop in doc["m_ParsedForm"]["m_PropInfo"]["m_Props"]:
        values = prop["m_DefValue"]
        out[prop["m_Name"]] = [float(v) for v in (values if isinstance(values, list) else [values])]
    return out
