"""Kinematic, Isaac-free generator and viewer for randomized reach demonstrations.

Approximates the acoustic mesh frame as coincident with the organ rigid-body
frame: it applies NO i4h mesh_to_organ_transform calibration offset, and never
runs inside Isaac -- it fabricates no ultrasound or robot dynamics. It exists
to visually and numerically validate anatomy.probe_target_in_world and
oracle.straight_line_reach before wiring a real Isaac controller through
collection.collect_oracle_reach, which uses the exact calibrated frame via
anatomy.read_isaac_mesh_to_world instead.
"""

import json
from pathlib import Path

import numpy as np

from us_dp.anatomy_processing.frames import probe_target_in_world
from us_dp.anatomy_processing.viewer import open3d
from us_dp.common.geometry import validate_poses
from us_dp.dataset_generation.oracle import random_phantom_pose, straight_line_reach


def _load_local_target_pose(landmarks_path):
    report = json.loads(Path(landmarks_path).read_text())
    local_pose = np.asarray(report["probe_target_mesh_frame"]["pose_m"])
    validate_poses(local_pose)
    return local_pose


class _ReachViewer:
    """Open3D animation loop: persistent geometries moved via relative delta transforms."""

    def __init__(self, o3d, landmarks_dir):
        self.viewer = o3d.visualization.VisualizerWithKeyCallback()
        if not self.viewer.create_window("US-DP | Kinematic reach demonstrations", 1280, 900):
            raise RuntimeError(
                "Open3D could not open a window. Run with visualize=False for a headless run."
            )
        self.skin = o3d.io.read_triangle_mesh(str(Path(landmarks_dir) / "Skin.ply"))
        self.skin.paint_uniform_color([0.65, 0.68, 0.72])
        self.skin.compute_vertex_normals()
        self.target_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        self.target_marker.paint_uniform_color([1.0, 0.15, 0.85])
        self.target_marker.compute_vertex_normals()
        self.probe_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        self.probe_marker.compute_vertex_normals()
        for geometry in (self.skin, self.target_marker, self.probe_marker):
            self.viewer.add_geometry(geometry)
        options = self.viewer.get_render_option()
        options.background_color = np.array([0.025, 0.03, 0.045])
        options.mesh_show_back_face = True
        self._phantom_pose = np.eye(4)
        self._target_pose = np.eye(4)
        self._probe_pose = np.eye(4)

    def _move(self, mesh, new_pose, current_attr):
        current = getattr(self, current_attr)
        mesh.transform(new_pose @ np.linalg.inv(current))
        setattr(self, current_attr, new_pose)

    def show_phantom(self, phantom_pose, target_pose_world):
        self._move(self.skin, phantom_pose, "_phantom_pose")
        target_only = np.eye(4)
        target_only[:3, 3] = target_pose_world[:3, 3]
        self._move(self.target_marker, target_only, "_target_pose")
        self.viewer.update_geometry(self.skin)
        self.viewer.update_geometry(self.target_marker)
        self.viewer.poll_events()
        self.viewer.update_renderer()

    def animate(self, positions, rotations):
        for position, rotation in zip(positions, rotations):
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = rotation, position
            self._move(self.probe_marker, pose, "_probe_pose")
            self.viewer.update_geometry(self.probe_marker)
            self.viewer.poll_events()
            self.viewer.update_renderer()

    def close(self):
        self.viewer.destroy_window()


def generate_reach_demonstrations(landmarks_dir, output, config, episodes, *, visualize=True):
    """Save one privileged, kinematic-only .npz per demonstration.

    For each episode: randomize the phantom pose (oracle.random_phantom_pose,
    translation/yaw ranges from config), project it to a world-frame target
    (anatomy.probe_target_in_world), then generate one randomized straight-line
    reach (oracle.straight_line_reach). Each saved file holds positions_world
    (samples,3), rotations_world (samples,3,3), timestamps (samples,),
    phantom_pose (4,4), target_pose_world (4,4), episode_id, group_id.

    There is no ultrasound or robot_state: this is a geometry/kinematics
    validation tool, not a substitute for a real Isaac rollout recorded with
    collection.collect_oracle_reach.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    landmarks_dir = Path(landmarks_dir)
    local_target_pose = _load_local_target_pose(landmarks_dir / "landmarks.json")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config.seed)
    viewer = _ReachViewer(open3d(), landmarks_dir) if visualize else None
    saved = []
    try:
        for episode in range(episodes):
            phantom_pose = random_phantom_pose(
                rng,
                config.nominal_phantom_pose(),
                translation_xy=config.phantom_translation_xy_m,
                yaw_radians=np.radians(config.phantom_yaw_range_deg),
            )
            target_pose_world = probe_target_in_world(local_target_pose, phantom_pose)
            reference = straight_line_reach(
                target_pose_world,
                rng,
                samples=config.samples,
                start_radius_m=config.start_radius_m,
                end_radius_m=config.end_radius_m,
                orientation_cone_rad=np.radians(config.orientation_cone_deg),
            )
            episode_id = f"reach_{episode:04d}"
            path = root / f"{episode_id}.npz"
            np.savez_compressed(
                path,
                positions_world=reference["positions_world"],
                rotations_world=reference["rotations_world"],
                timestamps=(np.arange(config.samples) / config.sample_hz).astype(np.float32),
                phantom_pose=phantom_pose.astype(np.float32),
                target_pose_world=target_pose_world.astype(np.float32),
                episode_id=np.array(episode_id),
                group_id=np.array(f"reach_phantom_{episode:04d}"),
            )
            saved.append(path)
            if viewer is not None:
                viewer.show_phantom(phantom_pose, target_pose_world)
                viewer.animate(reference["positions_world"], reference["rotations_world"])
    finally:
        if viewer is not None:
            viewer.close()
    return saved
