"""Added-skin vertex tints calibrated against immutable native seam UV colours."""
from __future__ import annotations

import numpy as np

from ..origin import colorspace as CS
from . import store


def restore_tints(mesh, manifest: dict | None) -> None:
    """Restore the render tint stream kept separately from authored game vcolor."""
    if not manifest or 'render_tints' not in manifest:
        return
    colors = np.asarray(manifest['render_tints'], dtype=np.float32)
    if colors.shape != (mesh.vertex_count, 4) or not np.isfinite(colors).all():
        raise ValueError('composite render tints do not match finite body RGBA vertices')
    mesh.colors = colors.copy()


def uv_colors(texture, uvs: np.ndarray) -> np.ndarray:
    """Decode texels before bilinear filtering, following the texture's flag."""
    pixels = texture.px
    height, width = pixels.shape[:2]
    uvs = np.asarray(uvs)
    x = (uvs[:, 0] % 1) * width - .5
    y = (1 - uvs[:, 1] % 1) * height - .5
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - ix)[:, None], (y - iy)[:, None]
    def at(rows, columns):
        # Decode each sampled corner before interpolation, rather than an
        # entire 4096² map for a sparse seam/garment query (§7.1352).
        return CS.decode(pixels[rows, columns][:, None, :], texture.srgb)[:, 0, :3]

    a = at(iy % height, ix % width)
    b = at(iy % height, (ix + 1) % width)
    c = at((iy + 1) % height, ix % width)
    d = at((iy + 1) % height, (ix + 1) % width)
    return (a * (1-fx) + b * fx) * (1-fy) + (c * (1-fx) + d * fx) * fy


#: What the body's tint away from the seams is the median correction of (``tint_baseline`` in the seams'
#: record): every seam's samples, or only the body skin's own seams (the limbs').  A head seam's samples are
#: the face's texture at the jaw, where faces paint its shadow: Lappland's read 0.96/0.92/0.98 of the lent
#: skin against her hands' 1.10/1.26/1.35, and with the head seam's 20 samples outvoting the hands' 18
#: the whole lent body came out pink beside her white hands and face.
BASELINES = ("every seam", "body skin seams")


def seam_tints(positions: np.ndarray, source_colors: np.ndarray, seam_indices: np.ndarray,
               targets: np.ndarray, band: float, baseline_rows: np.ndarray | None = None) -> np.ndarray:
    """Match seam albedo exactly and blend its correction into a calibrated body.

    *baseline_rows* (a mask over the seam samples) are the samples whose
    median correction the body carries away from the seams; all of them
    when ``None``.
    """
    gain = targets / np.maximum(source_colors[seam_indices], 1e-8)
    baseline = np.median(gain if baseline_rows is None else gain[np.asarray(baseline_rows, dtype=bool)], axis=0)
    colors = np.ones((len(positions), 4), dtype=np.float32)
    colors[:, :3] = baseline
    for start in range(0, len(positions), 256):
        distance = np.linalg.norm(positions[start:start+256, None] - positions[seam_indices][None], axis=2)
        nearest = distance.argmin(axis=1)
        fade = np.clip(1 - distance[np.arange(len(nearest)), nearest] / max(band, 1e-8), 0, 1)
        fade = (fade * fade * (3 - 2 * fade))[:, None]
        colors[start:start+256, :3] = baseline + (gain[nearest] - baseline) * fade
    colors[seam_indices, :3] = gain
    return colors


def reviewed_native_skin_color(texture, uvs: np.ndarray, reference_vertices: np.ndarray,
                               base_color: np.ndarray) -> np.ndarray:
    """Estimate linear skin colour only from explicitly reviewed exposed native vertices.

    The caller must certify that these vertices sample skin without clothing or
    decorative marks. No automatic body-part or colour-based classification is used.
    The texture, UVs and authored material colour remain unchanged.
    """
    uvs = np.asarray(uvs)
    vertices = np.asarray(reference_vertices)
    base_color = np.asarray(base_color, dtype=float)
    if uvs.ndim != 2 or uvs.shape[1] != 2 or not np.isfinite(uvs).all():
        raise ValueError('native skin reference requires finite UV pairs')
    if (vertices.ndim != 1 or not len(vertices) or vertices.dtype.kind not in 'iu'
            or np.any(vertices < 0) or np.any(vertices >= len(uvs))):
        raise ValueError('native skin reference requires valid reviewed vertex indices')
    if base_color.shape != (4,) or not np.isfinite(base_color).all() or np.any(base_color < 0):
        raise ValueError('native skin reference requires finite nonnegative authored RGBA')
    colors = uv_colors(texture, uvs[np.unique(vertices)]) * CS.colour_to_linear(base_color)[:3]
    if not np.isfinite(colors).all() or np.any(colors < 0):
        raise ValueError('reviewed native skin samples contain invalid linear colours')
    return np.median(colors, axis=0).astype(np.float32)


def _material_colors(shader: dict, uvs: np.ndarray) -> np.ndarray:
    """Linear albedo at *uvs* in *shader*'s authored base map times its base colour."""
    descriptor = shader.get('maps', {}).get('_BaseMap')
    if descriptor is None:
        raise ValueError('native seam skin has no authored base map')
    texture = store.read_texture(descriptor['cab'], descriptor['name'])
    if texture is None:
        raise FileNotFoundError(descriptor)
    base = shader.get('colors', {}).get('_BaseColor', [1, 1, 1, 1])
    return uv_colors(texture, uvs) * CS.colour_to_linear(base)[:3]


def _native_targets(item: dict, seam: dict, shader: dict) -> np.ndarray:
    """Sample immutable endpoint UVs with the authored material colour.

    A surface joined from several drawn meshes (``samples._join_head_skin``)
    lists the material of each vertex range in ``vertex_materials``; every
    endpoint is sampled with the material that draws it, since its UVs
    address that material's texture.  An endpoint outside every range is an
    error, not a sample of some other texture.
    """
    vertices = np.asarray(seam['native_vertices'], dtype=np.int64)
    uvs = item['mesh'].uvs
    spans = item.get('vertex_materials') or [(0, len(uvs), shader)]
    targets = np.full((len(vertices), 3), np.nan)
    for start, stop, own in spans:
        inside = (vertices >= start) & (vertices < stop)
        if inside.any():
            targets[inside] = _material_colors(own, uvs[vertices[inside]])
    if np.isnan(targets).any():
        raise ValueError('a native seam endpoint lies on no drawn surface of its head skin')
    return targets


def baseline_rows(seams: dict) -> np.ndarray | None:
    """The seam samples the body's tint is the median of (:data:`BASELINES`), in the order the seams list them."""
    if seams.get("tint_baseline", BASELINES[0]) == BASELINES[0]:
        return None
    limbs = [len(row["render_native_vertices"]) for row in seams.get("seams", [])]
    heads = [len(row["render_native_vertices"]) for row in seams.get("head_seams", [])]
    return np.r_[np.ones(sum(limbs), dtype=bool), np.zeros(sum(heads), dtype=bool)]


def calibrate_skin(mesh, native: list, seams: dict, albedo, *, head_surfaces: list | None = None) -> dict:
    """Tint added body/Head seams from native endpoint UVs, preserving original data.

    Records in *seams* (``tint_baseline``) what the body's tint is the median
    of: the body skin's own seams when there are any, else every seam.
    """
    seams["tint_baseline"] = BASELINES[1] if seams.get("seams") else BASELINES[0]
    indices, targets, bands = [], [], []
    records = [(seam, native[seam['piece_index']], native[seam['piece_index']]['piece'].shader or {})
               for seam in seams['seams']]
    for seam in seams.get('head_seams', []):
        surface = (head_surfaces or [])[seam['head_surface_index']]
        records.append((seam, surface, surface['shader']))
    for seam, item, shader in records:
        targets.extend(_native_targets(item, seam, shader))
        indices.extend(seam['render_native_vertices'])
        bands.append(seam['bridge_band_m'])
    if not indices:
        return {'status': 'no native seams to calibrate'}
    indices, targets = np.asarray(indices), np.asarray(targets)
    source = uv_colors(albedo, mesh.uvs)
    mesh.colors = seam_tints(mesh.positions, source, indices, targets, max(bands) * 8, baseline_rows(seams))
    residual = np.max(np.abs(source[indices] * mesh.colors[indices, :3] - targets))
    return {'status': 'native seam albedo calibrated in linear space',
            'seam_samples': len(indices), 'max_linear_albedo_error': float(residual),
            'tint_baseline': seams["tint_baseline"],
            'limitation': 'albedo matching; native game lighting and shader responses are separate'}
