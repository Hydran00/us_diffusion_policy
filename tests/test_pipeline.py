import numpy as np
import pytest
import torch

from us_dp.config import Config
from us_dp.data import WindowDataset, load_episode, prepare, save_episode
from us_dp.demo import synthetic_episodes
from us_dp.geometry import rotation_matrix, to_local, to_world
from us_dp.spline import SplineCodec


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def config():
    return Config(
        image_size=16,
        down_dims=(16, 32),
        feature_dim=16,
        diffusion_steps=4,
        inference_steps=2,
        epochs=1,
        batch_size=8,
    )


def test_upstream_spline_continuity_fit_and_endpoint():
    torch.manual_seed(3)
    codec = SplineCodec(4, 21)
    w = torch.randn(2, 6, 3)
    w[:, 0] = 0
    cp = codec.control_points(w)
    torch.testing.assert_close(cp[:, :-1, 2], cp[:, 1:, 0])
    torch.testing.assert_close(cp[:, :-1, 2] - cp[:, :-1, 1], cp[:, 1:, 1] - cp[:, 1:, 0])
    dense = codec.decode(w)
    torch.testing.assert_close(dense[:, 0], torch.zeros(2, 3))
    torch.testing.assert_close(dense[:, -1], cp[:, -1, 2])
    torch.testing.assert_close(
        codec.sample(w, torch.linspace(0, 1, 21)), dense, atol=3e-6, rtol=1e-5
    )
    torch.testing.assert_close(codec.decode(codec.fit(dense)), dense, atol=3e-6, rtol=1e-5)
    weights = w.clone().requires_grad_()
    codec.decode(weights).square().mean().backward()
    assert weights.grad.isfinite().all() and weights.grad.abs().sum() > 0


def test_frame_invariance_and_quaternion_conventions():
    angle = 0.6
    r = rotation_matrix([0, 0, np.sin(angle / 2), np.cos(angle / 2)], "xyzw")
    np.testing.assert_allclose(
        r, rotation_matrix([np.cos(angle / 2), 0, 0, np.sin(angle / 2)], "wxyz")
    )
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3], pose[:3, 3] = r, [0.4, -0.1, 0.2]
    local = np.random.default_rng(2).normal(size=(21, 3))
    world = to_world(local, pose)
    np.testing.assert_allclose(to_local(world, pose), local, atol=1e-6)
    global_pose = np.eye(4, dtype=np.float32)
    global_pose[:3, :3], global_pose[:3, 3] = r, [-0.2, 0.5, 0.9]
    np.testing.assert_allclose(
        to_local(to_world(world, global_pose), global_pose @ pose), local, atol=1e-6
    )
    with pytest.raises(ValueError, match="Zero"):
        rotation_matrix([0, 0, 0, 0], "xyzw")


def test_group_split_causal_windows_and_normalization(tmp_path, config):
    raw = synthetic_episodes(tmp_path / "raw", config)
    manifest = prepare(raw, tmp_path / "data", config)
    assert manifest["fit_rmse_m"] < 1e-4
    group_splits = {}
    for episode in manifest["episodes"]:
        group_splits.setdefault(episode["group_id"], set()).add(episode["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    dataset = WindowDataset(tmp_path / "data", "train")
    sample = dataset[0]
    name, row = dataset.index[0]
    stored = dataset.episode(name)
    t = stored["anchors"][row]
    np.testing.assert_array_equal(
        sample["robot_state"], stored["robot_state"][t - config.history + 1 : t + 1]
    )
    assert sample["ultrasound"].shape == (3, 1, 16, 16)
    torch.testing.assert_close(sample["trajectory"][0], torch.zeros(3))
    stats = dataset.statistics()
    assert all(torch.isfinite(mean).all() and (std > 0).all() for mean, std in stats.values())
    with pytest.raises(ValueError, match="training split"):
        WindowDataset(tmp_path / "data", "validation").statistics()
    # A sentinel introduced exclusively into held-out states cannot affect stats.
    held = WindowDataset(tmp_path / "data", "test")
    held_file = tmp_path / "data" / held.entries[0]["file"]
    with np.load(held_file) as f:
        arrays = {key: f[key] for key in f.files}
    arrays["robot_state"][:] = 1e9
    np.savez_compressed(held_file, **arrays)
    again = WindowDataset(tmp_path / "data", "train").statistics()
    for key in stats:
        torch.testing.assert_close(stats[key][0], again[key][0])


def test_invalid_episode_and_timing(tmp_path, config):
    raw = synthetic_episodes(tmp_path / "raw", config, 3)
    arrays, meta = load_episode(raw / "demo_0000.npz")
    meta["state_fields"][0] = "organ_pose"
    with pytest.raises(ValueError, match="privileged"):
        save_episode(tmp_path / "bad.npz", arrays, meta)
    arrays, meta = load_episode(raw / "demo_0000.npz")
    arrays["timestamps"][2] += 0.02
    # Timing irregularity still increases monotonically but cannot use uniform targets.
    (raw / "demo_0000.npz").unlink()
    save_episode(raw / "demo_0000.npz", arrays, meta)
    with pytest.raises(ValueError, match="timestamps"):
        prepare(raw, tmp_path / "data", config)


def test_unet_padding_epsilon_gradient_and_sampling(config):
    from us_dp.model import UltrasoundSplinePolicy

    torch.manual_seed(10)
    policy = UltrasoundSplinePolicy(config, 23)
    batch = {
        "ultrasound": torch.rand(2, 3, 1, 16, 16),
        "robot_state": torch.randn(2, 3, 23),
        "spline_params": torch.randn(2, 6, 3),
    }
    assert policy.free_count == 5 and policy.padded_count == 6
    loss = policy.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    for component in (policy.image_encoder, policy.state_encoder, policy.denoiser):
        assert (
            sum(float(p.grad.abs().sum()) for p in component.parameters() if p.grad is not None) > 0
        )
    output = policy.eval().predict(
        batch["ultrasound"], batch["robot_state"], torch.Generator().manual_seed(3)
    )
    assert output["trajectory"].shape == (2, 21, 3)
    assert output["trajectory"].isfinite().all()
    torch.testing.assert_close(output["trajectory"][:, 0], torch.zeros(2, 3))


def test_end_to_end_checkpoint_and_replanning(tmp_path, config):
    from us_dp.inference import RecedingHorizonPolicy
    from us_dp.train import evaluate, train

    raw = synthetic_episodes(tmp_path / "raw", config, 3)
    prepare(raw, tmp_path / "data", config)
    checkpoint = train(tmp_path / "data", tmp_path / "training")
    report = evaluate(checkpoint, tmp_path / "data")
    assert report["samples"] > 0 and np.isfinite(report["position_rmse_m"])
    runner = RecedingHorizonPolicy(checkpoint)
    episode, _ = load_episode(raw / "demo_0000.npz")
    with pytest.raises(RuntimeError, match="history"):
        runner.plan()
    for i in range(3):
        runner.observe(
            episode["ultrasound"][i],
            episode["robot_state"][i],
            episode["probe_pose"][i],
            episode["timestamps"][i],
        )
    plan = runner.plan(control_hz=50, generator=torch.Generator().manual_seed(0))
    assert plan["positions_world"].shape == (20, 3)
    assert plan["time_from_start"][0] == pytest.approx(0.02)
    assert plan["time_from_start"][-1] == pytest.approx(0.4)
    np.testing.assert_allclose(plan["probe_pose_at_plan"], episode["probe_pose"][2])
    with pytest.raises(ValueError, match="consecutive"):
        runner.observe(
            episode["ultrasound"][3],
            episode["robot_state"][3],
            episode["probe_pose"][3],
            1.5,
        )
    runner.reset()
    assert len(runner.images) == 0 and runner.last_timestamp is None


def test_surface_oracle_uses_skin_roi():
    from us_dp.oracle import surface_sweep

    x = np.linspace(-0.05, 0.05, 40)
    points = np.stack((x, np.zeros_like(x), 0.2 + 0.1 * x * x), axis=-1)
    normals = np.tile([0.0, 0.0, 1.0], (len(x), 1))
    pose = np.eye(4)
    pose[:3, 3] = [0.5, 0, 0.1]
    result = surface_sweep(points, normals, pose, samples=31)
    assert result["positions_world"].shape == (31, 3)
    assert result["positions_world"][:, 2].min() >= 0.3 - 1e-6
    np.testing.assert_allclose(np.linalg.norm(result["normals_world"], axis=-1), 1, atol=1e-6)


def test_hdf5_mapping_measured_pose(tmp_path):
    h5py = pytest.importorskip("h5py")
    from us_dp.convert import import_hdf5

    path = tmp_path / "recording.h5"
    with h5py.File(path, "w") as f:
        demo = f.create_group("data/demo_0")
        demo.attrs["group_id"] = "phantom_0"
        for key, value in {
            "image": np.zeros((4, 16, 16, 3), np.uint8),
            "q": np.zeros((4, 9)),
            "dq": np.zeros((4, 9)),
            "pos": np.zeros((4, 3)),
            "quat": np.tile([0, 0, 0, 1], (4, 1)),
        }.items():
            demo.create_dataset("obs/" + key, data=value)
    mapping = {
        "ultrasound": "obs/image",
        "image_format": "rgb_uint8",
        "joint_position": "obs/q",
        "joint_velocity": "obs/dq",
        "arm_joint_indices": list(range(7)),
        "probe_position": "obs/pos",
        "probe_quaternion": "obs/quat",
        "quaternion_order": "xyzw",
        "sample_hz": 10,
    }
    import_hdf5(path, tmp_path / "raw", mapping)
    arrays, meta = load_episode(tmp_path / "raw/demo_0.npz")
    assert arrays["robot_state"].shape == (4, 23)
    assert meta["group_id"] == "phantom_0"


def test_collection_records_executed_not_commanded_poses(tmp_path):
    from us_dp.collection import EpisodeRecorder, collect_oracle_episode

    time = 0.0
    pose = np.eye(4, dtype=np.float32)

    def observe():
        return {
            "timestamp": time,
            "ultrasound": np.zeros((16, 16), np.uint8),
            "robot_state": np.zeros(23, np.float32),
            "probe_pose": pose.copy(),
        }

    def controller(position, normal, dt):
        nonlocal time
        time += dt
        pose[:3, 3] = position * 0.8  # deliberate tracking error

    reference = {
        "positions_world": np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]]),
        "normals_world": np.tile([0, 0, 1], (3, 1)),
    }
    recorder = EpisodeRecorder(tmp_path / "episode.npz", "0", "phantom_0")
    collect_oracle_episode(recorder, reference, observe, controller, sample_hz=10)
    arrays, _ = load_episode(tmp_path / "episode.npz")
    np.testing.assert_allclose(arrays["probe_pose"][-1, :3, 3], [0.016, 0, 0])
    assert not np.allclose(arrays["probe_pose"][-1, :3, 3], reference["positions_world"][-1])


def test_three_level_unet_and_config_validation(config):
    from us_dp.model import UltrasoundSplinePolicy

    c = Config(**(config.to_dict() | {"down_dims": (16, 32, 64)}))
    policy = UltrasoundSplinePolicy(c, 23)
    assert policy.padded_count == 8
    output = policy.predict(torch.zeros(1, 3, 1, 16, 16), torch.zeros(1, 3, 23))
    assert output["trajectory"].shape == (1, 21, 3)
    for override in (
        {"execution_seconds": 0.45},
        {"inference_steps": 5},
        {"validation_fraction": 0},
        {"num_segments": 30},
    ):
        with pytest.raises(ValueError):
            Config(**(config.to_dict() | override))
