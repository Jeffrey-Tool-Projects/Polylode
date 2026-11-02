"""Read native authored FBX frames and their lossless explicit source handedness."""
from __future__ import annotations

import base64
import zlib
from pathlib import Path

import numpy as np

from ..endfield import fbxfile as F
from .fbx_tangents import SIGN_PROPERTY, FRAME_PROPERTY


def _array(data: bytes, prop: tuple) -> np.ndarray:
    kind, (count, encoding, stored), offset = prop
    dtype = {b'd': '<f8', b'i': '<i4'}.get(kind)
    if dtype is None or encoding not in (0, 1):
        raise F.FbxError('unsupported native authored frame array')
    payload = data[offset:offset + stored]
    if encoding:
        payload = zlib.decompress(payload)
    if len(payload) != count * np.dtype(dtype).itemsize:
        raise F.FbxError('native authored frame array length changed')
    return np.frombuffer(payload, dtype=dtype).copy()


def _fields(data: bytes, start: int, stop: int) -> dict:
    return {name: (props, child_start, child_stop)
            for name, props, child_start, child_stop in F._records(data, start, stop)}


def _payload(value: bytes, count: int) -> np.ndarray:
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(value, validate=True), count * 4 + 1)
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or len(raw) != count * 4:
        raise F.FbxError('source frame payload length changed')
    return np.frombuffer(raw, dtype='<f4').copy()


def _vectors(data: bytes, element: tuple, name: bytes) -> np.ndarray:
    _, start, stop = element
    fields = _fields(data, start, stop)
    if fields[b'MappingInformationType'][0][0][1] != b'ByPolygonVertex':
        raise F.FbxError('authored FBX vector mapping changed')
    reference = fields[b'ReferenceInformationType'][0][0][1]
    vectors = _array(data, fields[name][0][0]).reshape(-1, 3)
    if reference == b'IndexToDirect' and name == b'Normals':
        index = _array(data, fields[b'NormalsIndex'][0][0])
        if index.size and (index.min() < 0 or index.max() >= len(vectors)):
            raise F.FbxError('authored FBX normal index is outside its array')
        return vectors[index]
    if reference != b'Direct':
        raise F.FbxError('authored FBX vector reference changed')
    return vectors


def read(path: Path) -> list[dict]:
    """Read each authored geometry by native uid, corner indices, N/T/B and point signs."""
    data = path.read_bytes()
    top = F._top(data)
    models = {int(props[0][1]): props[1][1].split(b'\x00')[0].decode('utf8')
              for name, props, _, _ in F._records(data, *top[b'Objects']) if name == b'Model'}
    connections = [(props[1][1], props[2][1]) for _, props, _, _ in F._records(data, *top[b'Connections'])
                   if props[0][1] == b'OO']
    results = []
    for name, props, start, stop in F._records(data, *top[b'Objects']):
        if name != b'Geometry' or len(props) < 3 or props[2][1] != b'Mesh':
            continue
        fields = _fields(data, start, stop)
        properties = fields.get(b'Properties70')
        if properties is None:
            continue
        signs = [values[-1][1] for _, values, _, _ in F._records(data, properties[1], properties[2])
                 if values[0][1] == SIGN_PROPERTY]
        if not signs:
            continue
        if len(signs) != 1:
            raise F.FbxError('duplicate source tangent sign property')
        points = _array(data, fields[b'Vertices'][0][0]).reshape(-1, 3)
        sign = _payload(signs[0], len(points))
        frames = [values[-1][1] for _, values, _, _ in F._records(data, properties[1], properties[2])
                  if values[0][1] == FRAME_PROPERTY]
        if len(frames) != 1:
            raise F.FbxError('missing or duplicate source point frame property')
        point_frame = _payload(frames[0], len(points) * 6).reshape(-1, 6)
        indices = _array(data, fields[b'PolygonVertexIndex'][0][0])
        indices = np.where(indices < 0, -indices - 1, indices)
        if indices.size and (indices.min() < 0 or indices.max() >= len(points)):
            raise F.FbxError('authored FBX polygon index is outside its vertex array')
        uid = int(props[0][1])
        vectors = {key: _vectors(data, fields[element], values) for key, element, values in
                   (('normal', b'LayerElementNormal', b'Normals'), ('tangent', b'LayerElementTangent', b'Tangents'),
                    ('binormal', b'LayerElementBinormal', b'Binormals'))}
        if (any(len(value) != len(indices) or not np.isfinite(value).all() for value in vectors.values())
                or not np.isfinite(sign).all() or not np.isfinite(point_frame).all()):
            raise F.FbxError('native authored frame count or finite values changed')
        if (not np.array_equal(vectors['normal'].astype(np.float32), point_frame[indices, :3])
                or not np.array_equal(vectors['tangent'].astype(np.float32), point_frame[indices, 3:])):
            raise F.FbxError('native corner layers differ from source point frame')
        if not np.array_equal(vectors['binormal'], np.cross(vectors['normal'], vectors['tangent']) * sign[indices, None]):
            raise F.FbxError('native binormal differs from authored normal, tangent and sign')
        results.append(dict(geometry_uid=uid, geometry_name=props[1][1].split(b'\x00')[0].decode('utf8'),
            model_names=[models[target] for source, target in connections if source == uid and target in models],
            positions=points, corner_vertex_indices=indices, point_sign=sign,
            point_normal=point_frame[:, :3], point_tangent=point_frame[:, 3:], **vectors))
    return results
