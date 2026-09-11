"""Cache lightweight display meshes without changing the source geometry."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def load_display_mesh(source, *, triangles=20000, cache_dir=None):
    import open3d as o3d

    if triangles < 4:
        raise ValueError('mesh triangles must be at least 4')
    source = Path(source).resolve()
    stat = source.stat()
    identity = json.dumps([str(source), stat.st_size, stat.st_mtime_ns, triangles,
                           o3d.__version__, 1])
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    cache_dir = Path(cache_dir) if cache_dir else (
        Path(__file__).resolve().parents[3] / 'data/cache/display_meshes')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f'{source.stem}-{key}.ply'
    if cached.exists():
        mesh = o3d.io.read_triangle_mesh(str(cached))
        if not mesh.is_empty() and mesh.has_triangles():
            return mesh
    print(f'Preparing display mesh ({triangles:,} triangles); subsequent loads use cache.', flush=True)
    mesh = o3d.io.read_triangle_mesh(str(source))
    if mesh.is_empty() or not mesh.has_triangles():
        raise ValueError(f'Cannot load mesh: {source}')
    original = len(mesh.triangles)
    if original > triangles:
        mesh = mesh.simplify_quadric_decimation(triangles)
    mesh.compute_vertex_normals()
    descriptor, temporary = tempfile.mkstemp(suffix='.ply', dir=cache_dir)
    os.close(descriptor)
    try:
        if not o3d.io.write_triangle_mesh(temporary, mesh, write_ascii=False):
            raise OSError(f'Cannot save display mesh: {cached}')
        os.replace(temporary, cached)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f'Display mesh: {original:,} -> {len(mesh.triangles):,} triangles; {cached}', flush=True)
    return mesh
