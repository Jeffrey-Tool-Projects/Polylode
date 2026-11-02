"""Preserve binary FBX payloads while setting explicit mesh visibility."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from ..endfield import fbxfile as F


@dataclass
class _Record:
    name: bytes
    props: list
    payload: bytes
    children: list
    ending: bytes


def _decode(data: bytes, start: int, end: int) -> tuple[list, bytes]:
    records, cursor = [], start
    for name, props, inner, stop in F._records(data, start, end):
        children, ending = _decode(data, inner, stop)
        payload = data[cursor + 13 + len(name):inner]
        records.append(_Record(name, props, payload, children, ending))
        cursor = stop
    return records, data[cursor:end]


def _encode(records: list, offset: int) -> bytes:
    result = bytearray()
    for record in records:
        body_start = offset + len(result) + 13 + len(record.name) + len(record.payload)
        body = _encode(record.children, body_start) + record.ending
        stop = body_start + len(body)
        result.extend(struct.pack('<III', stop, len(record.props), len(record.payload)))
        result.extend(bytes([len(record.name)]) + record.name + record.payload + body)
    return bytes(result)


def _string(value: bytes) -> bytes:
    return b'S' + struct.pack('<I', len(value)) + value


def _property(value: float) -> _Record:
    strings = [b'Visibility', b'Visibility', b'', b'A']
    props = [(b'S', s, 0) for s in strings] + [(b'D', value, 0)]
    payload = b''.join(_string(s) for s in strings) + b'D' + struct.pack('<d', value)
    return _Record(b'P', props, payload, [], b'')


def _meshes(records: list) -> dict:
    objects = [r for r in records if r.name == b'Objects']
    if len(objects) != 1:
        raise ValueError('FBX has no unique Objects record')
    meshes = {}
    for record in objects[0].children:
        if record.name != b'Model' or record.props[2][1] != b'Mesh':
            continue
        name = record.props[1][1].split(b'\x00')[0].decode('utf8')
        properties = [r for r in record.children if r.name == b'Properties70']
        if name in meshes or len(properties) != 1:
            raise ValueError('FBX has ambiguous mesh identity or properties')
        meshes[name] = properties[0]
    return meshes


def _visibility(properties: _Record) -> _Record | None:
    fields = [r for r in properties.children if r.name == b'P' and r.props[0][1] == b'Visibility']
    if len(fields) > 1 or fields and (len(fields[0].props) != 5 or fields[0].props[4][0] != b'D'):
        raise ValueError('unsupported FBX Visibility property')
    return fields[0] if fields else None


def _fingerprint(records: list, excluded: set[int]) -> str:
    h = hashlib.sha256()

    def visit(record):
        if id(record) in excluded:
            return
        h.update(struct.pack('<II', len(record.name), len(record.payload)))
        h.update(record.name + record.payload)
        for child in record.children:
            visit(child)
        h.update(b'end')

    for record in records:
        visit(record)
    return h.hexdigest()


def read_visibility(data: bytes) -> dict[str, float]:
    """Read mesh Visibility; its FBX default when omitted is 1.0."""
    F._top(data)
    records, _ = _decode(data, len(F.MAGIC) + 4, len(data))
    return {name: float(prop.props[4][1]) if (prop := _visibility(properties)) else 1.0
            for name, properties in _meshes(records).items()}


def fbx_visibility(data: bytes, hidden: dict[str, bool]) -> tuple[bytes, dict]:
    """Set only mesh Visibility, retaining every other property/array payload."""
    F._top(data)
    start = len(F.MAGIC) + 4
    records, footer = _decode(data, start, len(data))
    meshes = _meshes(records)
    if set(meshes) != set(hidden) or any(type(v) is not bool for v in hidden.values()):
        raise ValueError('FBX visibility profile must cover exactly its mesh instances')
    fields = {n: _visibility(p) for n, p in meshes.items()}
    before = _fingerprint(records, {id(v) for v in fields.values() if v is not None})
    changed, added, excluded = [], [], set()
    for name, properties in meshes.items():
        prop, wanted = fields[name], float(not hidden[name])
        if prop is None and wanted == 1.0:
            continue
        if prop is None:
            prop = _property(wanted)
            properties.children.append(prop)
            if not properties.ending:
                properties.ending = bytes(13)
            added.append(name)
            changed.append(name)
        else:
            if prop.props[4][1] != wanted:
                changed.append(name)
            replacement = _property(wanted)
            prop.props, prop.payload = replacement.props, replacement.payload
        excluded.add(id(prop))
    if _fingerprint(records, excluded) != before:
        raise ValueError('FBX visibility update changed unrelated content')
    result = data[:start] + _encode(records, start) + footer
    if read_visibility(result) != {n: float(not v) for n, v in hidden.items()}:
        raise ValueError('FBX visibility read-back differs from profile')
    return result, {'changed_meshes': changed, 'added_visibility_properties': added,
                    'non_visibility_content_unchanged': True,
                    'non_visibility_bytes_unchanged': not added,
                    'non_visibility_content_sha256': before}
