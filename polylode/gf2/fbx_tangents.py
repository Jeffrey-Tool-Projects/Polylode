"""Export authored frames as native FBX layers instead of recomputing UV tangent space."""
from __future__ import annotations

import base64
import struct
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..mesh.tangent import directions

SIGN_PROPERTY = b'SourceTangentSignF32LEZlibBase64'
FRAME_PROPERTY = b'SourceNormalTangentF32LEZlibBase64'


def _attribute(mesh, name: str, field: str, width: int) -> np.ndarray:
    attribute = mesh.attributes.get(name)
    if attribute is None or attribute.domain != 'POINT' or len(attribute.data) != len(mesh.vertices):
        raise ValueError(f'{mesh.name}: missing complete original {name} point stream')
    values = np.empty(len(mesh.vertices) * width, dtype=np.float32)
    attribute.data.foreach_get(field, values)
    if not np.isfinite(values).all():
        raise ValueError(f'{mesh.name}: nonfinite original {name}')
    return values.reshape(-1, width)


def _frame(mesh, me_obj, scene_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = _attribute(mesh, 'SourceNormal', 'vector', 3).astype(float)
    tangent = _attribute(mesh, 'SourceTangent', 'vector', 3).astype(float)
    sign = _attribute(mesh, 'SourceTangentSign', 'value', 1)[:, 0]
    if me_obj.use_bake_space_transform(scene_data):
        linear = np.asarray(scene_data.settings.global_matrix, dtype=float)[:3, :3]
        determinant = float(np.linalg.det(linear))
        if determinant == 0:
            raise ValueError('singular baked FBX geometry transform')
        normal = directions(normal, np.linalg.inv(linear).T)
        tangent = directions(tangent, linear)
        sign = sign * np.sign(determinant)
    return normal, tangent, sign


def _layer(exporter, geometry, kind: bytes, data_name: bytes, vectors: np.ndarray) -> None:
    layer = exporter.elem_data_single_int32(geometry, kind, 0)
    version = (exporter.FBX_GEOMETRY_TANGENT_VERSION if kind == b'LayerElementTangent'
               else exporter.FBX_GEOMETRY_BINORMAL_VERSION)
    exporter.elem_data_single_int32(layer, b'Version', version)
    exporter.elem_data_single_string(layer, b'Name', b'SourceTangent')
    exporter.elem_data_single_string(layer, b'MappingInformationType', b'ByPolygonVertex')
    exporter.elem_data_single_string(layer, b'ReferenceInformationType', b'Direct')
    exporter.elem_data_single_float64_array(layer, data_name, vectors.ravel())


def _patch(exporter, geometry, mesh, frame) -> None:
    normal, tangent, sign = frame
    corner = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get('vertex_index', corner)
    geometry.elems[:] = [element for element in geometry.elems
                         if element.id not in (b'LayerElementNormal', b'LayerElementTangent', b'LayerElementBinormal')]
    source_normal = exporter.elem_data_single_int32(geometry, b'LayerElementNormal', 0)
    exporter.elem_data_single_int32(source_normal, b'Version', exporter.FBX_GEOMETRY_NORMAL_VERSION)
    exporter.elem_data_single_string(source_normal, b'Name', b'SourceNormal')
    exporter.elem_data_single_string(source_normal, b'MappingInformationType', b'ByPolygonVertex')
    exporter.elem_data_single_string(source_normal, b'ReferenceInformationType', b'IndexToDirect')
    exporter.elem_data_single_float64_array(source_normal, b'Normals', normal[corner].ravel())
    exporter.elem_data_single_int32_array(source_normal, b'NormalsIndex', np.arange(len(corner), dtype=np.int32))
    _layer(exporter, geometry, b'LayerElementTangent', b'Tangents', tangent[corner])
    _layer(exporter, geometry, b'LayerElementBinormal', b'Binormals', (np.cross(normal, tangent) * sign[:, None])[corner])
    layers = [element for element in geometry.elems if element.id == b'Layer']
    for layer in layers:
        layer.elems[:] = [element for element in layer.elems if not any(
            field.id == b'Type' and field.props[0][4:] in (b'LayerElementTangent', b'LayerElementBinormal')
            for field in element.elems)]
    layer = next(element for element in layers if struct.unpack('<i', element.props[0])[0] == 0)
    for kind in (b'LayerElementTangent', b'LayerElementBinormal'):
        reference = exporter.elem_empty(layer, b'LayerElement')
        exporter.elem_data_single_string(reference, b'Type', kind)
        exporter.elem_data_single_int32(reference, b'TypedIndex', 0)
    properties = next(element for element in geometry.elems if element.id == b'Properties70')
    # A parallel authored N/T produces B=0, so sign cannot be inferred from B.
    # Preserve it in an explicit lossless custom property, not TangentsW with unknown SDK semantics.
    payload = base64.b64encode(zlib.compress(np.asarray(sign, dtype='<f4').tobytes())).decode('ascii')
    exporter.elem_props_set(properties, 'p_string', SIGN_PROPERTY, payload, custom=True)
    # Native corner layers cannot represent vertices unused by any polygon.
    points = np.concatenate((normal, tangent), axis=1).astype('<f4')
    payload = base64.b64encode(zlib.compress(points.tobytes())).decode('ascii')
    exporter.elem_props_set(properties, 'p_string', FRAME_PROPERTY, payload, custom=True)


@contextmanager
def authored_export() -> Iterator[None]:
    """Scope the installed exporter's mesh callback; restore it even after an export failure."""
    from io_scene_fbx import export_fbx_bin as exporter
    original = exporter.fbx_data_mesh_elements

    def write(root, me_obj, scene_data, done_meshes):
        key, mesh, _ = scene_data.data_meshes[me_obj]
        if key in done_meshes or mesh.attributes.get('SourceTangent') is None:
            return original(root, me_obj, scene_data, done_meshes)
        if scene_data.settings.use_mesh_edges and any(edge.is_loose for edge in mesh.edges):
            raise ValueError(f'{mesh.name}: authored corner export does not support loose edge polygons')
        frame = _frame(mesh, me_obj, scene_data)
        original(root, me_obj, scene_data, done_meshes)
        uid = exporter.get_fbx_uuid_from_key(key)
        geometry = next(element for element in root.elems
                        if element.id == b'Geometry' and struct.unpack('<q', element.props[0])[0] == uid)
        _patch(exporter, geometry, mesh, frame)

    exporter.fbx_data_mesh_elements = write
    try:
        yield
    finally:
        exporter.fbx_data_mesh_elements = original


def export_scene(**options: Any) -> set[str]:
    """Run Blender's ordinary FBX export with exact authored T/B layers where supplied."""
    import bpy
    with authored_export():
        return bpy.ops.export_scene.fbx(**options)


@contextmanager
def authored_import(path: Path) -> Iterator[None]:
    """Restore point streams by native geometry uid during Blender's own import."""
    from .fbx_streams import read
    from io_scene_fbx import import_fbx as importer
    frames = {frame['geometry_uid']: frame for frame in read(path)}
    original = importer.blen_read_geom
    restored = set()

    def geometry(template, element, settings):
        mesh = original(template, element, settings)
        uid = int(element.props[0])
        frame = frames.get(uid)
        if frame is None:
            return mesh
        positions = frame['positions']
        normal, tangent, sign = frame['point_normal'], frame['point_tangent'], frame['point_sign']
        if settings.bake_space_transform:
            positions = importer.vcos_transformed(
                positions.ravel(), settings.global_matrix, np.float32).reshape(-1, 3)
            linear = np.asarray(settings.global_matrix, dtype=float)[:3, :3]
            normal = directions(normal, np.linalg.inv(linear).T)
            tangent = directions(tangent, linear)
            sign = sign * np.sign(np.linalg.det(linear))
        actual = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get('co', actual)
        if not np.array_equal(actual.reshape(-1, 3), positions.astype(np.float32)):
            raise ValueError(f'{mesh.name}: native FBX vertex order or positions changed')
        for name, kind, field, values in (
                ('SourceNormal', 'FLOAT_VECTOR', 'vector', normal),
                ('SourceTangent', 'FLOAT_VECTOR', 'vector', tangent),
                ('SourceTangentSign', 'FLOAT', 'value', sign)):
            attribute = mesh.attributes.new(name, kind, 'POINT')
            attribute.data.foreach_set(field, np.asarray(values, dtype=np.float32).ravel())
        mesh.normals_split_custom_set_from_vertices(np.asarray(normal, dtype=np.float32).tolist())
        restored.add(uid)
        return mesh

    importer.blen_read_geom = geometry
    try:
        yield
        if restored != frames.keys():
            raise ValueError('Blender did not import every authored FBX geometry')
    finally:
        importer.blen_read_geom = original


def import_scene(**options: Any) -> set[str]:
    """Import with Blender and restore native authored point frames exactly."""
    import bpy
    with authored_import(Path(options['filepath'])):
        return bpy.ops.import_scene.fbx(**options)
