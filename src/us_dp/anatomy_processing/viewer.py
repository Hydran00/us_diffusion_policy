"""Open3D inspector for the actual ultrasound Skin/Liver meshes and liver PCA."""

import json
from pathlib import Path

import numpy as np

from us_dp.anatomy_processing.assets import sha256
from us_dp.anatomy_processing.frames import estimate_surface_frame, probe_target_in_world
from us_dp.common.geometry import validate_poses

# Confirmed by visual inspection (blue arrow in view_anatomy): in the raw
# Skin/Liver mesh frame extracted from the ultrasound Docker image, -Y points
# from the liver toward the chest/skin contact side. Not documented upstream;
# see the "Orientamenti di riferimento" section of us_dp/README.md.
ANTERIOR_AXIS_MESH_FRAME = np.array([0.0, -1.0, 0.0])


def open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Install the viewer in this environment: python -m pip install -e '.[viewer]'"
        ) from error
    return o3d


def _project_probe_target(o3d, liver_center, anterior_direction, transverse_direction, vertices, triangles):
    """Ray-cast the liver center toward the skin along the confirmed anterior axis.

    Returns one rigid 4x4 pose in mesh frame: position is the first Skin
    intersection along anterior_direction; the third rotation column points
    INTO the tissue (opposite the outward surface normal); the first column is
    transverse_direction projected into the local tangent plane. This is a
    controller-calibration convention, not a clinical one.
    """
    direction = np.asarray(anterior_direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(triangles, dtype=np.uint32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    ray = np.concatenate((liver_center, direction)).astype(np.float32).reshape(1, 6)
    hit = scene.cast_rays(o3d.core.Tensor(ray))
    distance = float(hit["t_hit"].numpy()[0])
    if not np.isfinite(distance):
        raise ValueError("Anterior ray from the liver center did not hit the Skin mesh")
    position = liver_center + distance * direction
    outward_normal = hit["primitive_normals"].numpy()[0].astype(np.float64)
    normal_length = np.linalg.norm(outward_normal)
    if normal_length < 1e-8:
        raise ValueError("Degenerate skin normal at the projected contact point")
    insertion_axis = -outward_normal / normal_length
    tangential = transverse_direction - np.dot(transverse_direction, insertion_axis) * insertion_axis
    tangential_length = np.linalg.norm(tangential)
    if tangential_length < 1e-6:
        raise ValueError("Transverse axis is degenerate at the projected skin point")
    transverse_axis = tangential / tangential_length
    second_axis = np.cross(insertion_axis, transverse_axis)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((transverse_axis, second_axis, insertion_axis))
    pose[:3, 3] = position
    validate_poses(pose)
    return pose


def prepare_anatomy(
    mesh_dir, output, *, mesh_to_world=None, mesh_units="mm", display_voxel_m=0.003
):
    """Estimate on full-resolution liver and phantom; simplify ONLY display meshes."""
    o3d = open3d()
    root, output = Path(mesh_dir), Path(output)
    for name in ("Skin.obj", "Liver.obj"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing {root / name}; run us-dp extract-anatomy first")
    if mesh_units not in ("mm", "m"):
        raise ValueError("mesh_units must be mm or m")
    if not np.isfinite(display_voxel_m) or display_voxel_m <= 0:
        raise ValueError("display_voxel_m must be positive")
    pose = np.eye(4) if mesh_to_world is None else np.asarray(mesh_to_world, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError("Expected one mesh_to_world matrix")
    validate_poses(pose)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "mesh_units_original": mesh_units,
        "output_units": "m",
        "frame": "acoustic_mesh" if mesh_to_world is None else "world",
        "mesh_to_world": pose.tolist(),
        "display_voxel_m": display_voxel_m,
        "meshes": {},
    }
    source = root / "source.json"
    if source.exists():
        report["source"] = json.loads(source.read_text())
    liver_center = skin_vertices = skin_triangles = skin_transverse_axis = None
    for name in ("Liver", "Skin"):
        path = root / f"{name}.obj"
        print(f"Loading {name}: {path}", flush=True)
        # OBJ material/texture files are unnecessary: the viewer supplies colors.
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        if mesh.is_empty() or not mesh.has_triangles():
            raise ValueError(f"No triangle surface in {path}")
        if mesh_units == "mm":
            mesh.scale(0.001, center=np.zeros(3))
        vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        if not np.isfinite(vertices).all():
            raise ValueError(f"Nonfinite mesh: {path}")
        report["meshes"][name] = {
            "source_path": str(path.resolve()),
            "sha256": sha256(path),
            "vertices": len(vertices),
            "triangles": len(triangles),
            "bounds_mesh_m": [vertices.min(0).tolist(), vertices.max(0).tolist()],
        }
        frame = estimate_surface_frame(vertices, triangles)
        prefix = "liver" if name == "Liver" else "phantom"
        report[f"{prefix}_mesh_frame"] = frame.to_dict()
        report[f"{prefix}_display_frame"] = frame.in_world(pose).to_dict()
        print(f"{name} center in mesh frame [m]:", frame.center.tolist(), flush=True)
        if frame.ambiguous_axes:
            print(
                f"Warning: {name} has similar eigenvalues; PCA directions may be unstable.",
                flush=True,
            )
        if name == "Liver":
            liver_center = frame.center
        else:
            skin_vertices, skin_triangles, skin_transverse_axis = vertices, triangles, frame.axes[:, 1]
        display_mesh = mesh.simplify_vertex_clustering(display_voxel_m)
        display_mesh.remove_degenerate_triangles()
        display_mesh.remove_duplicated_triangles()
        report["meshes"][name]["display_triangles"] = len(display_mesh.triangles)
        if not o3d.io.write_triangle_mesh(
            str(output / f"{name}.ply"), display_mesh, write_ascii=False
        ):
            raise RuntimeError("Could not save display mesh")
    local_target_pose = _project_probe_target(
        o3d, liver_center, ANTERIOR_AXIS_MESH_FRAME, skin_transverse_axis, skin_vertices, skin_triangles
    )
    report["probe_target_mesh_frame"] = {
        "pose_m": local_target_pose.tolist(),
        "method": "ray_cast_fixed_anterior_axis",
        "anterior_axis_mesh_frame": ANTERIOR_AXIS_MESH_FRAME.tolist(),
    }
    report["probe_target_display_frame"] = {
        "pose_m": probe_target_in_world(local_target_pose, pose).tolist()
    }
    print("Probe target (skin contact) in mesh frame [m]:", local_target_pose[:3, 3].tolist(), flush=True)
    (output / "landmarks.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _arrow(o3d, center, direction, length, color):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.002,
        cone_radius=0.004,
        cylinder_height=0.85 * length,
        cone_height=0.15 * length,
    )
    z = np.asarray(direction)
    helper = np.eye(3)[np.argmin(np.abs(z))]
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    rotation = np.column_stack((x, np.cross(z, x), z))
    arrow.rotate(rotation, center=np.zeros(3)).translate(center)
    arrow.paint_uniform_color(color)
    arrow.compute_vertex_normals()
    return arrow


def view_anatomy(directory, *, screenshot=None, close_after_capture=False):
    """Interactive desktop viewer. P cycles skin; L toggles liver; S saves PNG."""
    o3d = open3d()
    root = Path(directory)
    report = json.loads((root / "landmarks.json").read_text())
    # Upgrade old reports from their original full-resolution Skin source, never
    # from the simplified display mesh (which would alter the estimated axes).
    report_dirty = False
    if "phantom_mesh_frame" not in report:
        source = Path(report["meshes"]["Skin"]["source_path"])
        if not source.is_file() or sha256(source) != report["meshes"]["Skin"]["sha256"]:
            raise ValueError("Original Skin mesh missing/changed: regenerate with us-dp anatomy")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            full_skin = o3d.io.read_triangle_mesh(str(source), enable_post_processing=False)
        if report["mesh_units_original"] == "mm":
            full_skin.scale(0.001, center=np.zeros(3))
        phantom_frame = estimate_surface_frame(
            np.asarray(full_skin.vertices), np.asarray(full_skin.triangles)
        )
        report["phantom_mesh_frame"] = phantom_frame.to_dict()
        report["phantom_display_frame"] = phantom_frame.in_world(report["mesh_to_world"]).to_dict()
        report_dirty = True
    if "probe_target_mesh_frame" not in report:
        source = Path(report["meshes"]["Skin"]["source_path"])
        if not source.is_file() or sha256(source) != report["meshes"]["Skin"]["sha256"]:
            raise ValueError("Original Skin mesh missing/changed: regenerate with us-dp anatomy")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            full_skin = o3d.io.read_triangle_mesh(str(source), enable_post_processing=False)
        if report["mesh_units_original"] == "mm":
            full_skin.scale(0.001, center=np.zeros(3))
        liver_center = np.asarray(report["liver_mesh_frame"]["center_m"])
        transverse_axis = np.asarray(report["phantom_mesh_frame"]["axes_columns"])[:, 1]
        local_target_pose = _project_probe_target(
            o3d,
            liver_center,
            ANTERIOR_AXIS_MESH_FRAME,
            transverse_axis,
            np.asarray(full_skin.vertices),
            np.asarray(full_skin.triangles),
        )
        report["probe_target_mesh_frame"] = {
            "pose_m": local_target_pose.tolist(),
            "method": "ray_cast_fixed_anterior_axis",
            "anterior_axis_mesh_frame": ANTERIOR_AXIS_MESH_FRAME.tolist(),
        }
        report["probe_target_display_frame"] = {
            "pose_m": probe_target_in_world(local_target_pose, report["mesh_to_world"]).tolist()
        }
        report_dirty = True
    if report_dirty:
        (root / "landmarks.json").write_text(json.dumps(report, indent=2) + "\n")
    pose = np.asarray(report["mesh_to_world"])
    validate_poses(pose)
    liver, skin = [
        o3d.io.read_triangle_mesh(str(root / f"{name}.ply")) for name in ("Liver", "Skin")
    ]
    for mesh in (liver, skin):
        mesh.transform(pose)
        mesh.compute_vertex_normals()
    liver.paint_uniform_color([0.8, 0.26, 0.08])
    skin.paint_uniform_color([0.65, 0.68, 0.72])
    wire = o3d.geometry.LineSet.create_from_triangle_mesh(skin.simplify_vertex_clustering(0.012))
    wire.paint_uniform_color([0.18, 0.22, 0.27])
    frame = report["phantom_display_frame"]
    # Orientations come from the entire phantom, anchored at the liver target.
    center = np.asarray(report["liver_display_frame"]["center_m"])
    axes = np.asarray(frame["axes_columns"])
    lengths = np.maximum(1.8 * np.sqrt(frame["variances_m2"]), 0.025)
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
    marker.translate(center).paint_uniform_color([1.0, 0.95, 0.05])
    marker.compute_vertex_normals()
    colors = ([1.0, 0.1, 0.1], [0.1, 1.0, 0.2])
    arrows = [_arrow(o3d, center, axes[:, i], lengths[i], colors[i]) for i in range(2)]
    # Full signed axes make their intersection evident even through the liver.
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(
        np.array([center + sign * axes[:, i] * lengths[i] for i in range(2) for sign in (-1, 1)])
    )
    lines.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3]])
    lines.colors = o3d.utility.Vector3dVector(colors)
    # Confirmed by visual inspection: -Y in the raw mesh frame is anterior.
    anterior_world = pose[:3, :3] @ ANTERIOR_AXIS_MESH_FRAME
    anterior_arrow = _arrow(o3d, center, anterior_world, lengths[2], [0.15, 0.55, 1.0])
    target_pose = np.asarray(report["probe_target_display_frame"]["pose_m"])
    validate_poses(target_pose)
    target_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
    target_marker.translate(target_pose[:3, 3]).paint_uniform_color([1.0, 0.15, 0.85])
    target_marker.compute_vertex_normals()
    standoff_arrow = _arrow(
        o3d, target_pose[:3, 3], -target_pose[:3, 2], lengths[2], [1.0, 0.55, 0.05]
    )
    # Liver wireframe keeps the internal centroid visible; L toggles solid surface.
    liver_wire = o3d.geometry.LineSet.create_from_triangle_mesh(
        liver.simplify_vertex_clustering(0.008)
    )
    liver_wire.paint_uniform_color([0.6, 0.25, 0.06])
    viewer = o3d.visualization.VisualizerWithKeyCallback()
    if not viewer.create_window(
        "US-DP | Phantom PCA: 1=RED 2=GREEN | Liver center=YELLOW | Anterior=BLUE"
        " | Probe target=MAGENTA | P:skin L:liver S:PNG",
        1280,
        900,
    ):
        raise RuntimeError(
            "Open3D could not open a window. Run from a graphical terminal or use anatomy --no-view."
        )
    try:
        for geometry in [
            wire,
            liver_wire,
            marker,
            lines,
            *arrows,
            anterior_arrow,
            target_marker,
            standoff_arrow,
        ]:
            viewer.add_geometry(geometry)
        options = viewer.get_render_option()
        options.background_color = np.array([0.025, 0.03, 0.045])
        options.mesh_show_back_face = True
        options.line_width = 1.0
        view = viewer.get_view_control()
        view.set_lookat((skin.get_min_bound() + skin.get_max_bound()) / 2)
        # LPS: look toward the anterior torso, superior axis up; rotate with mesh.
        view.set_front(pose[:3, :3] @ np.array([1.2, -1.0, 0.3]))
        view.set_up(pose[:3, :3] @ np.array([0.0, 0.0, 1.0]))
        view.set_zoom(0.75)
        modes = {"skin": 0, "liver": 0}
        skin_modes = (wire, skin, None)
        liver_modes = (liver_wire, liver, None)

        def cycle(key, geometries):
            def callback(vis):
                old = geometries[modes[key]]
                if old is not None:
                    vis.remove_geometry(old, reset_bounding_box=False)
                modes[key] = (modes[key] + 1) % len(geometries)
                new = geometries[modes[key]]
                if new is not None:
                    vis.add_geometry(new, reset_bounding_box=False)
                return False

            return callback

        target = Path(screenshot) if screenshot else root / "viewer.png"

        def capture(vis):
            target.parent.mkdir(parents=True, exist_ok=True)
            vis.capture_screen_image(str(target), do_render=True)
            print(f"Screenshot: {target}", flush=True)
            return False

        viewer.register_key_callback(ord("P"), cycle("skin", skin_modes))
        viewer.register_key_callback(ord("L"), cycle("liver", liver_modes))
        viewer.register_key_callback(ord("S"), capture)
        for _ in range(10):
            viewer.poll_events()
            viewer.update_renderer()
        if screenshot or close_after_capture:
            capture(viewer)
        print(
            "Yellow: liver surface centroid. Red/green: phantom principal axes 1/2 (longitudinal/transverse candidates).\n"
            "Blue: confirmed anterior axis (-Y in raw mesh frame). Magenta: liver center ray-cast onto the"
            " skin along that axis -- the fixed reach/sweep target. Orange: standoff direction there.\n"
            "P: phantom wire/solid/hidden; L: liver wire/solid/hidden; S: save PNG; Q: close.",
            flush=True,
        )
        if not close_after_capture:
            viewer.run()
    finally:
        viewer.destroy_window()
