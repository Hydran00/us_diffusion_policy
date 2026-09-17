import h5py
import numpy as np

from us_dp.training.simulator_compare import summarize_recording


def test_summarize_complete_simulator_episodes(tmp_path):
    path = tmp_path / "rollouts.hdf5"
    with h5py.File(path, "w") as recording:
        data = recording.create_group("data")
        for index, success in enumerate((True, False)):
            demo = data.create_group(f"demo_{index}")
            demo.attrs["episode_index"] = index
            demo.attrs["success"] = success
            obs = demo.create_group("obs")
            measured = np.zeros((3, 7), dtype=np.float32)
            measured[:, 0] = [0, 0.01, 0.02]
            commanded = measured.copy()
            commanded[:, 0] += 0.001 * (index + 1)
            obs.create_dataset("measured_ee_pose", data=measured)
            obs.create_dataset("commanded_ee_pose", data=commanded)
            obs.create_dataset("timestamps", data=[0, 0.02, 0.04])
    summary = summarize_recording(path, expected_episodes=2)
    assert summary["successes"] == 1
    assert summary["success_rate_percent"] == 50
    assert np.isclose(summary["mean_duration_s"], 0.04)
    assert np.isclose(summary["mean_tcp_path_length_m"], 0.02)
    assert np.isclose(summary["mean_tracking_error_m"], 0.0015)
