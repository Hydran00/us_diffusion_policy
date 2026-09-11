"""Synchronized Open3D playback of measured TCP poses and recorded ultrasound."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from us_dp.common.geometry import rotation_matrix, validate_poses

WORKSPACE = Path(__file__).resolve().parents[4]
DEFAULT_RUN = WORKSPACE / 'runs/ultrasound_liver_scan/acq_001'
DEFAULT_SKIN = WORKSPACE / 'us_dp/data/assets/abdphantom/Skin.obj'


def nominal_mesh_pose():
    """panda_phantom nominal root Rz(pi), then acoustic mesh offset Rx(pi/2)."""
    pose = np.eye(4)
    pose[:3, :3] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    pose[:3, 3] = [0.6, 0, 0.09]
    return pose


class Recording:
    """Load poses per episode; read only the currently displayed image from disk."""

    def __init__(self, path, sample_hz=50.0, mesh_poses=None):
        import h5py

        if not np.isfinite(sample_hz) or sample_hz <= 0:
            raise ValueError('sample_hz must be positive and finite')
        path = Path(path)
        if path.is_dir():
            path = path / 'data/raw/demos.hdf5'
        self.handle = h5py.File(path, 'r')
        self.sample_hz = sample_hz
        self.mesh_poses = mesh_poses or {}
        try:
            self.names = sorted(
                (n for n in self.handle['data'] if n.startswith('demo_')),
                key=lambda n: int(n.removeprefix('demo_')),
            )
            if not self.names:
                raise ValueError('No data/demo_* episodes found')
            self.initial_episode = next((i for i, name in enumerate(self.names)
                                         if 'obs/mesh_pose' in self.handle['data'][name]), 0)
            self.select(self.initial_episode)
        except Exception:
            self.close()
            raise

    def select(self, index):
        self.episode_index = index % len(self.names)
        self.name = self.names[self.episode_index]
        demo = self.handle['data'][self.name]
        poses = demo['obs/measured_ee_pose'][()]
        if poses.ndim != 2 or poses.shape[1] != 7 or not len(poses):
            raise ValueError(f'{self.name}: expected nonempty measured_ee_pose (T,7)')
        self.positions = poses[:, :3]
        self.rotations = rotation_matrix(poses[:, 3:], 'wxyz')
        if not np.isfinite(self.positions).all():
            raise ValueError(f'{self.name}: nonfinite measured positions')
        # Loaded fully rather than kept as an h5py dataset reference: a
        # per-frame disk read during playback was a second big contributor
        # to low FPS, alongside the per-frame geometry rebuilds in Playback.
        # One episode's worth of uint8 frames is a few tens of MB at most.
        self.images = demo['obs/ultrasound'][()]
        if len(self.images) != len(poses) or self.images.dtype != np.uint8:
            raise ValueError(f'{self.name}: ultrasound must be uint8 and match pose count')
        if self.images.ndim not in (3, 4) or (
            self.images.ndim == 4 and self.images.shape[-1] not in (1, 3)
        ):
            raise ValueError(f'{self.name}: expected grayscale or RGB ultrasound')
        self.times = (
            np.asarray(demo['obs/timestamps'][()], dtype=float)
            if 'obs/timestamps' in demo else np.arange(len(poses)) / self.sample_hz
        )
        if (self.times.shape != (len(poses),) or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times) <= 0)):
            raise ValueError(f'{self.name}: invalid timestamps')
        self.times -= self.times[0]
        self.time_source = 'recorded timestamps' if 'obs/timestamps' in demo else (
            f'assumed {self.sample_hz:g} Hz (acq_001 log: dt=.005 x decimation=4)'
        )
        supplied = self.mesh_poses.get(self.name, self.mesh_poses.get('mesh_to_world'))
        self.mesh_pose_frames = None
        self.exact_mesh = (supplied is not None or 'mesh_to_world' in demo.attrs
                           or 'obs/mesh_pose' in demo)
        if supplied is not None:
            self.mesh_pose = np.asarray(supplied, dtype=float)
            self.mesh_source = 'Skin: supplied mesh-to-world transform'
        elif 'mesh_to_world' in demo.attrs:
            self.mesh_pose = np.asarray(demo.attrs['mesh_to_world'], dtype=float)
            self.mesh_source = 'Skin: recorded mesh-to-world transform'
        elif 'obs/mesh_pose' in demo:
            recorded = demo['obs/mesh_pose']
            values = recorded[()]
            if values.shape != (len(poses), 7):
                raise ValueError(f'{self.name}: mesh_pose must have shape (T,7)')
            self.mesh_pose_frames = np.tile(np.eye(4), (len(poses), 1, 1))
            self.mesh_pose_frames[:, :3, :3] = rotation_matrix(
                values[:, 3:], recorded.attrs.get('quaternion_order', 'wxyz'))
            self.mesh_pose_frames[:, :3, 3] = values[:, :3]
            validate_poses(self.mesh_pose_frames)
            self.mesh_pose = self.mesh_pose_frames[0]
            self.mesh_source = 'Skin: recorded acoustic mesh pose (per frame)'
        else:
            self.mesh_pose = nominal_mesh_pose()
            self.mesh_source = 'Skin: NOMINAL pose (episode has no recorded mesh pose)'
        if self.mesh_pose.shape != (4, 4):
            raise ValueError('mesh_to_world must be a 4x4 matrix in metres')
        validate_poses(self.mesh_pose)
        self.status = str(demo.attrs.get('status', 'unknown'))

    def frame_at(self, elapsed):
        return int(np.clip(np.searchsorted(self.times, elapsed, side='right') - 1,
                           0, len(self.times) - 1))

    def image(self, index):
        image = self.images[index]
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        return np.ascontiguousarray(image)

    def close(self):
        self.handle.close()


class Playback:
    def __init__(self, recording, skin, mesh_units='mm'):
        import open3d as o3d
        from open3d.visualization import gui, rendering

        self.o3d, self.gui = o3d, gui
        self.recording = recording
        self.paused, self.elapsed, self.frame = False, 0.0, -1
        self.last_tick = time.monotonic()
        self.app = gui.Application.instance
        self.window = self.app.create_window('US-DP | HDF5 ultrasound playback', 1400, 850)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.035, 0.04, 0.05, 1])
        self.panel = gui.Vert(10, gui.Margins(12, 12, 12, 12))
        self.info = gui.Label('')
        self.alignment = gui.Label('')
        self.image_widget = gui.ImageWidget()
        self.panel.add_child(self.info)
        self.panel.add_child(self.alignment)
        self.panel.add_child(gui.Label('SPACE play/pause | N/P episode\nLEFT/RIGHT frame (pause) | R restart\nHOME reset camera | Q quit'))
        self.panel.add_child(self.image_widget)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self.layout)
        self.window.set_on_key(self.on_key)
        self.window.set_on_tick_event(self.tick)
        self.skin = o3d.io.read_triangle_mesh(str(skin))
        if self.skin.is_empty() or not self.skin.has_triangles():
            raise ValueError(f'Cannot load skin mesh: {skin}')
        self.skin.compute_vertex_normals()
        self.skin.scale(0.001 if mesh_units == 'mm' else 1.0, center=(0, 0, 0))
        # The standalone Skin.obj (extracted separately for us_dp's anatomy
        # tooling) does not share its local axis convention with the USD
        # phantom asset actually calibrated by mesh_to_organ_transform in the
        # real Isaac scene: confirmed by watching real Isaac (belly at +Z, as
        # expected for a supine phantom in Isaac's Z-up world with the robot
        # reaching from above) versus this viewer (belly at -Z) using the
        # same recorded mesh_pose in both. 180 degrees about local X is a
        # first guess at the missing correction, applied only to this
        # standalone mesh, before the per-frame mesh_pose transform -- verify
        # against base_frame that both the up/down AND the left/right heading
        # now match real Isaac; if the heading flips instead, try Y.
        self.skin.rotate(np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]), center=(0, 0, 0))
        self.skin_material = rendering.MaterialRecord()
        self.skin_material.shader = 'defaultLitTransparency'
        self.skin_material.base_color = [0.65, 0.72, 0.78, 0.35]
        self.line_material = rendering.MaterialRecord()
        self.line_material.shader = 'unlitLine'
        self.line_material.line_width = 3
        self.solid_material = rendering.MaterialRecord()
        self.solid_material.shader = 'defaultUnlit'
        # World-origin reference (robot base frame): identity transform, never
        # moved. Compare the skin/trajectory against this fixed frame to tell
        # whether the mesh, the recorded poses, or the camera view is what's
        # actually wrong.
        self.base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.base_frame.compute_vertex_normals()
        self.select(recording.initial_episode)

    def layout(self, _context):
        rect = self.window.content_rect
        width = min(450, int(rect.width * 0.36))
        self.scene_widget.frame = self.gui.Rect(rect.x, rect.y, rect.width - width, rect.height)
        self.panel.frame = self.gui.Rect(rect.get_right() - width, rect.y, width, rect.height)

    def select(self, index):
        self.recording.select(index)
        scene = self.scene_widget.scene
        scene.clear_geometry()
        scene.add_geometry('base_frame', self.base_frame, self.solid_material)
        scene.add_geometry('skin', self.skin, self.skin_material)
        scene.set_geometry_transform('skin', self.recording.mesh_pose)
        # Persistent geometry, moved via set_geometry_transform instead of
        # remove+recreate+add every frame (was the majority of the per-frame
        # cost together with the trajectory rebuild below).
        self.tcp = self.o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.025)
        self.tcp.compute_vertex_normals()
        scene.add_geometry('tcp', self.tcp, self.solid_material)
        self.trajectory_shown = 0
        self.alignment.text = self.recording.mesh_source
        self.elapsed, self.frame = 0.0, -1
        self.last_tick = time.monotonic()
        self.draw_frame(0)
        self.reset_camera()
        self.last_tick = time.monotonic()

    def reset_camera(self):
        points = self.recording.positions
        points = np.vstack((points,
                            np.asarray(self.skin.vertices) @ self.recording.mesh_pose[:3, :3].T
                            + self.recording.mesh_pose[:3, 3],
                            np.zeros((1, 3))))  # keep the base_frame origin in view too
        bounds = self.o3d.geometry.AxisAlignedBoundingBox(points.min(0), points.max(0))
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())

    # Rebuilding the trajectory LineSet costs O(index) (points + lines
    # re-uploaded to the GPU) and index grows every frame, so doing it on
    # every single frame was the dominant cost of the low framerate, worst
    # near the end of a long episode. Redrawing it only every few frames
    # keeps the visible trailing edge within a frame or two of the true
    # position -- not noticeable at playback speed -- for a large cut in
    # total rebuild work. Always redraws immediately when scrubbing backward
    # (LEFT arrow / R restart), so paused frame-stepping stays exact.
    TRAJECTORY_REBUILD_EVERY = 3

    def draw_frame(self, index):
        o3d, scene = self.o3d, self.scene_widget.scene
        if self.recording.mesh_pose_frames is not None:
            scene.set_geometry_transform(
                'skin', self.recording.mesh_pose_frames[index])
        if index == 0 or index < self.trajectory_shown or index - self.trajectory_shown >= self.TRAJECTORY_REBUILD_EVERY:
            if scene.has_geometry('trajectory'):
                scene.remove_geometry('trajectory')
            if index:
                trajectory = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(self.recording.positions[:index + 1]),
                    o3d.utility.Vector2iVector(np.column_stack((np.arange(index), np.arange(1, index + 1)))),
                )
                trajectory.paint_uniform_color([0.1, 0.95, 0.65])
                scene.add_geometry('trajectory', trajectory, self.line_material)
            self.trajectory_shown = index
        pose = np.eye(4)
        pose[:3, :3] = self.recording.rotations[index]
        pose[:3, 3] = self.recording.positions[index]
        scene.set_geometry_transform('tcp', pose)
        self.image_widget.update_image(o3d.geometry.Image(self.recording.image(index)))
        self.frame = index
        self.update_label()
        self.window.post_redraw()

    def update_label(self):
        r = self.recording
        self.info.text = (f'{r.name} | episode {r.episode_index + 1}/{len(r.names)} | {r.status}\n'
                          f'Frame {self.frame + 1}/{len(r.times)} | t={r.times[self.frame]:.2f}s\n'
                          f'{"PAUSED" if self.paused else "PLAYING"} | 1x simulation time\n'
                          f'{r.time_source}\nMeasured TCP trajectory (world, metres)')

    def tick(self):
        now = time.monotonic()
        if not self.paused:
            self.elapsed = min(self.elapsed + now - self.last_tick, self.recording.times[-1])
            index = self.recording.frame_at(self.elapsed)
            if self.elapsed >= self.recording.times[-1]:
                self.paused = True
            if index != self.frame:
                self.draw_frame(index)
        self.last_tick = now
        return False

    def on_key(self, event):
        """Window callbacks return bool; handled keys stop dispatch to the scene."""
        gui = self.gui
        if event.type != gui.KeyEvent.DOWN or event.is_repeat:
            return False
        key = event.key
        if key == gui.KeyName.SPACE:
            self.paused = not self.paused
            self.last_tick = time.monotonic()
        elif key in (gui.KeyName.N, gui.KeyName.P):
            self.select(self.recording.episode_index + (1 if key == gui.KeyName.N else -1))
            self.paused = False
        elif key in (gui.KeyName.LEFT, gui.KeyName.RIGHT):
            self.paused = True
            index = int(np.clip(self.frame + (1 if key == gui.KeyName.RIGHT else -1),
                                0, len(self.recording.times) - 1))
            self.elapsed = self.recording.times[index]
            self.draw_frame(index)
        elif key == gui.KeyName.R:
            self.elapsed, self.paused = 0.0, False
            self.last_tick = time.monotonic()
            self.draw_frame(0)
        elif key == gui.KeyName.HOME:
            self.reset_camera()
        elif key == gui.KeyName.Q:
            self.window.close()
        else:
            return False
        self.update_label()
        return True


def view_hdf5(path=DEFAULT_RUN, skin=DEFAULT_SKIN, *, sample_hz=50, mesh_poses=None,
              mesh_units='mm'):
    from open3d.visualization import gui

    poses = json.loads(Path(mesh_poses).read_text()) if mesh_poses else None
    recording = Recording(path, sample_hz, poses)
    try:
        gui.Application.instance.initialize()
        viewer = Playback(recording, skin, mesh_units)
        gui.Application.instance.run()
        del viewer
    finally:
        recording.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(DEFAULT_RUN), help='Run directory or HDF5 file')
    parser.add_argument('--skin', default=str(DEFAULT_SKIN))
    parser.add_argument('--sample-hz', type=float, default=50, help='Fallback without timestamps')
    parser.add_argument('--mesh-poses', help='JSON: demo_N -> mesh-to-world 4x4, or mesh_to_world')
    parser.add_argument('--mesh-units', choices=('mm', 'm'), default='mm')
    args = parser.parse_args()
    view_hdf5(args.input, args.skin, sample_hz=args.sample_hz,
              mesh_poses=args.mesh_poses, mesh_units=args.mesh_units)


if __name__ == '__main__':
    main()
