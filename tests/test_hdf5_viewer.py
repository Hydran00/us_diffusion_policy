import numpy as np
import pytest

from us_dp.dataset_generation.hdf5_viewer import Recording, nominal_mesh_pose


@pytest.fixture
def recording_path(tmp_path):
    h5py = pytest.importorskip('h5py')
    path = tmp_path / 'demos.hdf5'
    with h5py.File(path, 'w') as handle:
        for name in ('demo_10', 'demo_2', 'demo_0'):
            obs = handle.create_group(f'data/{name}/obs')
            poses = np.zeros((3, 7))
            poses[:, 0] = [0.1, 0.2, 0.3]
            poses[:, 3] = 1  # Recording quaternion convention is wxyz.
            obs['measured_ee_pose'] = poses
            obs['ultrasound'] = np.stack([np.full((8, 8, 3), i, np.uint8) for i in range(3)])
    return path


def test_order_time_and_image_pose_synchronization(recording_path):
    r = Recording(recording_path, sample_hz=50)
    try:
        assert r.names == ['demo_0', 'demo_2', 'demo_10']
        np.testing.assert_allclose(r.rotations, np.tile(np.eye(3), (3, 1, 1)))
        assert r.frame_at(0.019) == 0
        assert r.frame_at(0.02) == 1
        assert r.frame_at(100) == 2
        assert r.frame_at(-1) == 0
        i = r.frame_at(0.02)
        assert r.positions[i, 0] == 0.2
        assert np.all(r.image(i) == 1)
        r.select(-1)
        assert r.name == 'demo_10'
        r.select(3)
        assert r.name == 'demo_0'
        assert not r.exact_mesh
    finally:
        r.close()


def test_recorded_timestamps_and_per_episode_mesh(recording_path):
    import h5py

    with h5py.File(recording_path, 'a') as handle:
        handle['data/demo_0/obs/timestamps'] = [12.0, 12.1, 12.4]
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    r = Recording(recording_path, mesh_poses={'demo_0': pose.tolist()})
    try:
        assert r.frame_at(0.2) == 1
        np.testing.assert_allclose(r.times, [0, 0.1, 0.4])
        np.testing.assert_allclose(r.mesh_pose, pose)
        assert r.exact_mesh
        r.select(1)
        assert not r.exact_mesh
        np.testing.assert_allclose(r.mesh_pose, nominal_mesh_pose())
    finally:
        r.close()


def test_reject_mismatched_images(recording_path):
    import h5py

    with h5py.File(recording_path, 'a') as handle:
        del handle['data/demo_0/obs/ultrasound']
        handle['data/demo_0/obs/ultrasound'] = np.zeros((2, 8, 8, 3), np.uint8)
    with pytest.raises(ValueError, match='match pose count'):
        Recording(recording_path)


@pytest.mark.parametrize('hz', [0, -1, float('nan')])
def test_invalid_fallback_rate(recording_path, hz):
    with pytest.raises(ValueError, match='sample_hz'):
        Recording(recording_path, sample_hz=hz)


def test_recorded_mesh_moves_with_episode_frames(recording_path):
    import h5py

    with h5py.File(recording_path, 'a') as handle:
        ds = handle.create_dataset('data/demo_0/obs/mesh_pose', data=[
            [1, 2, 3, 1, 0, 0, 0], [4, 5, 6, 0, 1, 0, 0], [7, 8, 9, 1, 0, 0, 0]])
        ds.attrs['quaternion_order'] = 'wxyz'
    r = Recording(recording_path)
    try:
        assert r.exact_mesh
        np.testing.assert_allclose(r.mesh_pose[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(r.mesh_pose_frames[1, :3, 3], [4, 5, 6])
        np.testing.assert_allclose(r.mesh_pose_frames[1, :3, :3], np.diag([1, -1, -1]))
        r.select(1)
        assert r.mesh_pose_frames is None
        assert not r.exact_mesh
    finally:
        r.close()


def test_window_keyboard_callback_returns_bool_and_handles_episode_once():
    from types import SimpleNamespace

    pytest.importorskip('open3d')
    from open3d.visualization import gui

    from us_dp.dataset_generation.hdf5_viewer import Playback

    # Exercise the callback with the installed pybind enums, without a display.
    player = Playback.__new__(Playback)
    player.gui = gui
    player.paused = False
    player.recording = SimpleNamespace(episode_index=0)
    selections = []
    player.select = selections.append
    player.update_label = lambda: None

    def event(key, kind=gui.KeyEvent.DOWN, repeat=False):
        return SimpleNamespace(key=key, type=kind, is_repeat=repeat)

    assert player.on_key(event(gui.KeyName.SPACE)) is True
    assert player.paused
    assert player.on_key(event(gui.KeyName.N)) is True
    assert selections == [1]
    assert not player.paused
    assert player.on_key(event(gui.KeyName.N, repeat=True)) is False
    assert player.on_key(event(gui.KeyName.N, kind=gui.KeyEvent.UP)) is False
    assert player.on_key(event(gui.KeyName.A)) is False
    assert selections == [1]


def test_start_with_first_calibrated_episode(recording_path):
    import h5py

    with h5py.File(recording_path, 'a') as handle:
        handle['data/demo_2/obs/mesh_pose'] = np.tile([1., 2., 3., 1., 0., 0., 0.], (3, 1))
    r = Recording(recording_path)
    try:
        assert r.name == 'demo_2'
        assert r.initial_episode == 1
        assert r.exact_mesh
        r.select(0)
        assert not r.exact_mesh
        assert 'NOMINAL' in r.mesh_source
    finally:
        r.close()
