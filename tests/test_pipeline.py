import json
from pathlib import Path

import numpy as np
import pytest
import torch

from us_dp.common.geometry import rotation_matrix, to_local, to_world
from us_dp.common.spline import SplineCodec
from us_dp.config import Config
from us_dp.dataset.processing import WindowDataset, load_episode, prepare, save_episode
from us_dp.dataset_generation.synthetic import synthetic_episodes


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


@pytest.mark.parametrize("pose_only", [False, True])
def test_group_split_causal_windows_and_normalization(tmp_path, config, pose_only):
    raw = synthetic_episodes(tmp_path / "raw", config)
    if pose_only:
        for path in raw.glob("*.npz"):
            arrays, metadata = load_episode(path)
            arrays["robot_state"] = arrays["robot_state"][:, 14:]
            metadata["state_fields"] = metadata["state_fields"][14:]
            path.unlink()
            save_episode(path, arrays, metadata)
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
    assert sample["robot_state"].shape == (3, 9)
    raw_arrays, _ = load_episode(raw / f"demo_{int(dataset.entries[0]["episode_id"]):04d}.npz")
    np.testing.assert_array_equal(stored["robot_state"], raw_arrays["robot_state"][:, -9:])
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
    arrays["timestamps"][2] += 0.005
    # Timing irregularity still increases monotonically but cannot use uniform targets.
    (raw / "demo_0000.npz").unlink()
    save_episode(raw / "demo_0000.npz", arrays, meta)
    with pytest.raises(ValueError, match="timestamps"):
        prepare(raw, tmp_path / "data", config)


def test_unet_padding_epsilon_gradient_and_sampling(config):
    from us_dp.training.model import UltrasoundSplinePolicy

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
    assert output["trajectory"].shape == (2, config.future_steps + 1, 3)
    assert output["trajectory"].isfinite().all()
    torch.testing.assert_close(output["trajectory"][:, 0], torch.zeros(2, 3))


def test_end_to_end_checkpoint_and_replanning(tmp_path, config):
    from us_dp.deployment.inference import RecedingHorizonPolicy
    from us_dp.training.train import evaluate, train

    raw = synthetic_episodes(tmp_path / "raw", config, 3)
    prepare(raw, tmp_path / "data", config)
    checkpoint = train(tmp_path / "data", tmp_path / "training")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(tmp_path / "training" / "tensorboard")).Reload()
    recorded = json.loads((tmp_path / "training" / "metrics.jsonl").read_text().splitlines()[-1])
    for name in ("train_noise_mse", "validation_noise_mse"):
        scalar = events.Scalars(f"loss/{name}")[-1]
        assert scalar.step == config.epochs
        assert scalar.value == pytest.approx(recorded[name])
    steps = events.Scalars("step/train_noise_mse")
    assert [event.step for event in steps] == list(range(1, len(steps) + 1))
    assert np.isfinite([event.value for event in steps]).all()
    assert events.Scalars("step/grad_norm_before_clip")
    assert events.Scalars("validation/mean_position_error_m")
    report = evaluate(checkpoint, tmp_path / "data")
    assert report["samples"] > 0 and np.isfinite(report["position_rmse_m"])
    runner = RecedingHorizonPolicy(checkpoint)
    episode, _ = load_episode(raw / "demo_0000.npz")
    with pytest.raises(RuntimeError, match="history"):
        runner.plan()
    for i in range(3):
        runner.observe(
            episode["ultrasound"][i],
            episode["robot_state"][i, 14:],
            episode["probe_pose"][i],
            episode["timestamps"][i],
        )
    plan = runner.plan(control_hz=50, generator=torch.Generator().manual_seed(0))
    assert plan["positions_world"].shape == (20, 3)
    assert plan["time_from_start"][0] == pytest.approx(0.02)
    assert plan["time_from_start"][-1] == pytest.approx(0.4)
    np.testing.assert_allclose(plan["probe_pose_at_plan"], episode["probe_pose"][2])
    np.testing.assert_allclose(plan["orientations_world"], np.repeat(episode["probe_pose"][:1, :3, :3], 20, axis=0))
    changed = episode["probe_pose"][3].copy()
    changed[:3, :3] = rotation_matrix([np.cos(0.3), 0, 0, np.sin(0.3)], "wxyz")
    runner.observe(episode["ultrasound"][3], episode["robot_state"][3, 14:], changed, episode["timestamps"][3])
    replanned = runner.plan(control_hz=50)
    np.testing.assert_allclose(replanned["orientations_world"], plan["orientations_world"])
    np.testing.assert_allclose(replanned["probe_pose_at_plan"], changed)

    # horizon_seconds=prediction_seconds: the whole trajectory in one shot
    # (open-loop deployment), not just the short execution_seconds slice.
    full = runner.plan(control_hz=50, horizon_seconds=config.prediction_seconds)
    assert full["positions_world"].shape == (round(config.prediction_seconds * 50), 3)
    assert full["time_from_start"][-1] == pytest.approx(config.prediction_seconds)
    with pytest.raises(ValueError, match="horizon_seconds"):
        runner.plan(control_hz=50, horizon_seconds=config.prediction_seconds + 1)
    with pytest.raises(ValueError, match="consecutive"):
        runner.observe(
            episode["ultrasound"][3],
            episode["robot_state"][3, 14:],
            episode["probe_pose"][3],
            1.5,
        )
    runner.reset()
    assert len(runner.images) == 0 and runner.last_timestamp is None


def test_surface_oracle_uses_skin_roi():
    from us_dp.dataset_generation.oracle import surface_sweep

    x = np.linspace(-0.05, 0.05, 40)
    points = np.stack((x, np.zeros_like(x), 0.2 + 0.1 * x * x), axis=-1)
    normals = np.tile([0.0, 0.0, 1.0], (len(x), 1))
    pose = np.eye(4)
    pose[:3, 3] = [0.5, 0, 0.1]
    result = surface_sweep(points, normals, pose, samples=31)
    assert result["positions_world"].shape == (31, 3)
    assert result["positions_world"][:, 2].min() >= 0.3 - 1e-6
    np.testing.assert_allclose(np.linalg.norm(result["normals_world"], axis=-1), 1, atol=1e-6)


def test_straight_line_reach_converges_near_a_fixed_target():
    from us_dp.dataset_generation.oracle import straight_line_reach

    target = np.eye(4)
    target[:3, 3] = [0.5, -0.08, 0.05]
    rng = np.random.default_rng(0)
    reference = straight_line_reach(
        target, rng, samples=41, start_radius_m=0.10, end_radius_m=0.02
    )
    positions = reference["positions_world"]
    rotations = reference["rotations_world"]
    assert positions.shape == (41, 3) and rotations.shape == (41, 3, 3)
    distances = np.linalg.norm(positions - target[:3, 3], axis=-1)
    # The start radius is a true circumference: exactly start_radius_m, not <=.
    np.testing.assert_allclose(distances[0], 0.10, atol=1e-6)
    assert distances[-1] <= 0.02 + 1e-6
    np.testing.assert_allclose(
        np.einsum("nij,nkj->nik", rotations, rotations), np.tile(np.eye(3), (41, 1, 1)), atol=1e-5
    )
    with pytest.raises(ValueError, match="end_radius_m"):
        straight_line_reach(target, rng, start_radius_m=0.05, end_radius_m=0.10)
    with pytest.raises(ValueError, match="orientation_cone_rad"):
        straight_line_reach(target, rng, orientation_cone_rad=0)


def test_straight_line_reach_orientation_stays_within_cone():
    from us_dp.dataset_generation.oracle import straight_line_reach

    target = np.eye(4)
    target[:3, 3] = [0.5, -0.08, 0.05]
    cone = np.radians(30)
    max_error = 0.0
    for seed in range(50):
        reference = straight_line_reach(
            target, np.random.default_rng(seed), samples=11, orientation_cone_rad=cone
        )
        for rotation in (reference["rotations_world"][0], reference["rotations_world"][-1]):
            relative = target[:3, :3].T @ rotation
            cosine = np.clip((np.trace(relative) - 1) / 2, -1, 1)
            max_error = max(max_error, np.arccos(cosine))
    assert max_error <= cone + 1e-6


def test_collect_oracle_reach_records_executed_poses(tmp_path):
    from us_dp.dataset_generation.collection import EpisodeRecorder, collect_oracle_reach
    from us_dp.dataset_generation.oracle import straight_line_reach

    time = 0.0
    pose = np.eye(4, dtype=np.float32)

    def observe():
        return {
            "timestamp": time,
            "ultrasound": np.zeros((16, 16), np.uint8),
            "robot_state": np.zeros(23, np.float32),
            "probe_pose": pose.copy(),
        }

    def controller(position, rotation, dt):
        nonlocal time
        time += dt
        pose[:3, 3] = position
        pose[:3, :3] = rotation

    target = np.eye(4)
    target[:3, 3] = [0.5, -0.08, 0.05]
    reference = straight_line_reach(target, np.random.default_rng(1), samples=10)
    recorder = EpisodeRecorder(tmp_path / "reach.npz", "0", "phantom_0")
    collect_oracle_reach(recorder, reference, observe, controller, sample_hz=10)
    arrays, _ = load_episode(tmp_path / "reach.npz")
    np.testing.assert_allclose(arrays["probe_pose"][-1, :3, 3], reference["positions_world"][-1])


def test_hdf5_mapping_measured_pose(tmp_path):
    h5py = pytest.importorskip("h5py")
    from us_dp.dataset.convert import import_hdf5

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


def test_hdf5_mapping_combined_pose_and_cartesian_state(tmp_path):
    """Combined measured poses work with or without joint observations."""
    h5py = pytest.importorskip("h5py")
    from us_dp.dataset.convert import import_hdf5

    path = tmp_path / "recording.h5"
    pos = np.array([[0.5, 0.0, 0.1], [0.5, 0.01, 0.1], [0.5, 0.02, 0.1], [0.5, 0.03, 0.1]])
    quat_wxyz = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
    with h5py.File(path, "w") as f:
        demo = f.create_group("data/demo_0")
        demo.attrs["episode_index"] = 7
        demo.create_dataset("obs/ultrasound", data=np.zeros((4, 16, 16, 3), np.uint8))
        demo.create_dataset("obs/measured_ee_pose", data=np.concatenate([pos, quat_wxyz], axis=-1))
        demo.create_dataset("obs/timestamps", data=np.arange(4) / 10.0)
    mapping = {
        "group_attribute": "episode_index",
        "ultrasound": "obs/ultrasound",
        "image_format": "rgb_uint8",
        "probe_pose": "obs/measured_ee_pose",
        "quaternion_order": "wxyz",
        "timestamps": "obs/timestamps",
    }
    import_hdf5(path, tmp_path / "cartesian", mapping)
    arrays, meta = load_episode(tmp_path / "cartesian/demo_0.npz")
    assert arrays["robot_state"].shape == (4, 9)
    np.testing.assert_allclose(arrays["robot_state"][:, :3], pos)
    with h5py.File(path, "a") as f:
        f["data/demo_0"].create_dataset("obs/q", data=np.ones((4, 7)))
        f["data/demo_0"].create_dataset("obs/dq", data=np.full((4, 7), 0.2))
    mapping.update(joint_position="obs/q", joint_velocity="obs/dq", arm_joint_indices=list(range(7)))
    import_hdf5(path, tmp_path / "raw", mapping)
    arrays, meta = load_episode(tmp_path / "raw/demo_0.npz")
    assert arrays["robot_state"].shape == (4, 23)
    np.testing.assert_allclose(arrays["robot_state"][:, :7], 1)
    np.testing.assert_allclose(arrays["robot_state"][:, 7:14], 0.2)
    np.testing.assert_allclose(arrays["robot_state"][:, 14:17], pos)
    np.testing.assert_allclose(arrays["probe_pose"][:, :3, 3], pos)
    assert meta["group_id"] == "7"


def test_collection_records_executed_not_commanded_poses(tmp_path):
    from us_dp.dataset_generation.collection import EpisodeRecorder, collect_oracle_episode

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


def test_usfm_encoder_gradient_flow_and_freezing(config):
    from us_dp.training.model import UltrasoundSplinePolicy

    torch.manual_seed(11)
    c = Config(**(config.to_dict() | {"usfm_freeze": False}))
    policy = UltrasoundSplinePolicy(c, 23)
    batch = {
        "ultrasound": torch.rand(2, 3, 1, 16, 16),
        "robot_state": torch.randn(2, 3, 23),
        "spline_params": torch.randn(2, 6, 3),
    }
    policy.compute_loss(batch).backward()
    for component in (policy.image_encoder.backbone, policy.image_encoder.project):
        assert (
            sum(float(p.grad.abs().sum()) for p in component.parameters() if p.grad is not None) > 0
        )
    output = policy.eval().predict(
        batch["ultrasound"], batch["robot_state"], torch.Generator().manual_seed(3)
    )
    assert output["trajectory"].isfinite().all()

    # A frozen backbone must not update, and must stay in eval mode even under policy.train().
    frozen = UltrasoundSplinePolicy(config, 23)
    frozen.train()
    assert not frozen.image_encoder.backbone.training
    before = frozen.image_encoder.backbone.patch_embed.proj.weight.clone()
    frozen.compute_loss(batch).backward()
    assert all(p.grad is None for p in frozen.image_encoder.backbone.parameters())
    torch.testing.assert_close(before, frozen.image_encoder.backbone.patch_embed.proj.weight)


def test_usfm_encoder_loads_released_checkpoint():
    checkpoint = Path(__file__).resolve().parents[2] / "USFM" / "USFM_latest.pth"
    if not checkpoint.exists():
        pytest.skip("USFM_latest.pth not downloaded")
    from us_dp.training.usfm_encoder import USFMEncoder

    config = Config(image_size=128, feature_dim=32, history=2)
    encoder = USFMEncoder(config, pretrained=str(checkpoint), freeze=True)
    out = encoder(torch.rand(2, config.history, 128, 128))
    assert out.shape == (2, 32) and out.isfinite().all()


def test_three_level_unet_and_config_validation(config):
    from us_dp.training.model import UltrasoundSplinePolicy

    c = Config(**(config.to_dict() | {"down_dims": (16, 32, 64)}))
    policy = UltrasoundSplinePolicy(c, 23)
    assert policy.padded_count == 8
    output = policy.predict(torch.zeros(1, 3, 1, 16, 16), torch.zeros(1, 3, 23))
    assert output["trajectory"].shape == (1, c.future_steps + 1, 3)
    for override in (
        {"execution_seconds": 0.45},
        {"inference_steps": 5},
        {"validation_fraction": 0},
        {"num_segments": config.future_steps + 1},
    ):
        with pytest.raises(ValueError):
            Config(**(config.to_dict() | override))


def test_reach_config_nominal_pose_and_validation():
    from us_dp.common.geometry import validate_poses
    from us_dp.config import ReachConfig

    reach = ReachConfig()
    pose = reach.nominal_phantom_pose()
    validate_poses(pose)
    np.testing.assert_allclose(pose[:3, 3], [0.6, 0.0, 0.09])
    np.testing.assert_allclose(pose[:3, :3] @ [1, 0, 0], [-1, 0, 0], atol=1e-6)  # 180 deg yaw
    for override in (
        {"end_radius_m": 0.2, "start_radius_m": 0.1},
        {"orientation_cone_deg": 0},
        {"phantom_yaw_range_deg": 200},
    ):
        with pytest.raises(ValueError):
            ReachConfig(**override)


def test_generate_reach_demonstrations_headless(tmp_path):
    from us_dp.config import ReachConfig
    from us_dp.dataset_generation.reach_demo import generate_reach_demonstrations

    landmarks_dir = tmp_path / "anatomy"
    landmarks_dir.mkdir()
    local_target_pose = np.eye(4).tolist()
    (landmarks_dir / "landmarks.json").write_text(
        json.dumps({"probe_target_mesh_frame": {"pose_m": local_target_pose}})
    )
    config = ReachConfig(samples=11, seed=0)
    saved = generate_reach_demonstrations(
        landmarks_dir, tmp_path / "reach", config, episodes=3, visualize=False
    )
    assert len(saved) == 3
    data = np.load(saved[0])
    assert data["positions_world"].shape == (11, 3)
    assert data["rotations_world"].shape == (11, 3, 3)
    start_distance = np.linalg.norm(data["positions_world"][0] - data["target_pose_world"][:3, 3])
    end_distance = np.linalg.norm(data["positions_world"][-1] - data["target_pose_world"][:3, 3])
    np.testing.assert_allclose(start_distance, config.start_radius_m, atol=1e-6)
    assert end_distance <= config.end_radius_m + 1e-6
    with pytest.raises(ValueError, match="episodes"):
        generate_reach_demonstrations(landmarks_dir, tmp_path / "reach2", config, 0, visualize=False)


def test_flat_decoder_seconds_and_autograd():
    codec = SplineCodec(4, 21)
    free = torch.randn(2, 15, requires_grad=True)
    times = torch.linspace(0, 2, 101)
    xyz = codec(free, times)
    assert xyz.shape == (2, 101, 3)
    torch.testing.assert_close(xyz[:, 0], torch.zeros(2, 3))
    torch.testing.assert_close(xyz[:, -1], codec.decode(codec.unpack(free))[:, -1])
    xyz.square().mean().backward()
    assert free.grad.isfinite().all() and free.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        codec(torch.zeros(2, 18), times)
    with pytest.raises(ValueError):
        codec(free, torch.tensor([2.01]))


def test_parameter_denormalization_and_metric_separation(config):
    from us_dp.training.model import UltrasoundSplinePolicy
    from us_dp.training.train import batch_metrics

    policy = UltrasoundSplinePolicy(config, 23)
    policy.param_mean.copy_(torch.tensor([0.1, -0.2, 0.3]))
    policy.param_std.copy_(torch.tensor([0.02, 0.1, 0.4]))
    free = torch.randn(2, 5, 3)
    torch.testing.assert_close(policy.denormalize_params(policy.normalize_params(free)), free)
    params = policy.codec.unpack(free.flatten(1))
    trajectory = policy.codec.decode(params)
    batch = {"ultrasound": torch.rand(2, 3, 1, 16, 16), "robot_state": torch.randn(2, 3, 23),
             "spline_params": params, "trajectory": trajectory + 0.01}
    # Perfect parameter prediction still retains the representation's fitting error.
    policy.predict = lambda *args: {"spline_params": params, "trajectory": trajectory}
    metrics = batch_metrics(policy, batch)
    assert metrics["spline_parameter_mse_m2"] == 0
    assert metrics["trajectory_mse_m2"] == pytest.approx(0.0003, rel=1e-4)
    assert metrics["spline_fit_mse_m2"] == pytest.approx(metrics["trajectory_mse_m2"])
    with pytest.raises(ValueError, match="23-value"):
        UltrasoundSplinePolicy(config, 8)


@pytest.mark.parametrize("bound", [0, -1, float("nan"), float("inf")])
def test_sampling_clip_range_rejects_invalid_values(bound):
    with pytest.raises(ValueError, match="sampling_clip_range"):
        Config(sampling_clip_range=bound)


def test_sampling_bounds_clean_estimate_before_reverse_chain(config):
    from dataclasses import replace
    from us_dp.training.model import UltrasoundSplinePolicy

    policy = UltrasoundSplinePolicy(replace(config, diffusion_steps=100, inference_steps=100), 23)
    scheduler = policy.scheduler
    scheduler.set_timesteps(100)
    latent = torch.ones(1, policy.free_count, 3)
    # Deliberately inaccurate epsilon prediction: the unclipped estimate is ~203.
    noise = latent * 0.9
    bounded = scheduler.step(noise, 99, latent, generator=torch.Generator().manual_seed(1))
    assert bounded.pred_original_sample.abs().max() <= 4
    scheduler.register_to_config(clip_sample=False)
    unbounded = scheduler.step(noise, 99, latent, generator=torch.Generator().manual_seed(1))
    assert unbounded.pred_original_sample.abs().max() > 100
    assert bounded.prev_sample.abs().max() < unbounded.prev_sample.abs().max()
    # A coefficient above the old diagnostic limit 3 remains representable.
    scheduler.register_to_config(clip_sample=True)
    a = scheduler.alphas_cumprod[50]
    clean = torch.full_like(latent, 3.55)
    xt = a.sqrt() * clean + (1-a).sqrt() * noise
    result = scheduler.step(noise, 50, xt)
    torch.testing.assert_close(result.pred_original_sample, clean)


def test_prepare_run_imports_successful_hdf5_and_groups_phantom(tmp_path, config):
    import json
    h5py = pytest.importorskip("h5py")
    run = tmp_path / "run"
    run.mkdir()
    source = run / "demos.hdf5"
    n = config.history + config.future_steps + 2
    with h5py.File(source, "w") as f:
        for i in range(5):
            demo = f.create_group(f"data/demo_{i}")
            demo.attrs['success'] = i != 4
            pose = np.tile([0.5, 0, 0.2, 1, 0, 0, 0], (n, 1))
            pose[:, 0] += np.arange(n) * 0.001
            demo.create_dataset('obs/measured_ee_pose', data=pose)
            demo.create_dataset('obs/ultrasound', data=np.zeros((n, 16, 16, 3), np.uint8))
            demo.create_dataset('obs/timestamps', data=np.arange(n) / config.sample_hz)
            phantom = np.tile([float(i % 3), 0, 0, 1, 0, 0, 0], (n, 1))
            demo.create_dataset('obs/phantom_pose', data=phantom)
    (run / 'run.json').write_text(json.dumps({'recording': 'demos.hdf5'}))
    manifest = prepare(run, tmp_path / 'prepared', config)
    assert len(manifest['episodes']) == 4
    assert manifest['preprocessing']['excluded_failed_episodes'] == 1
    episodes = {e['episode_id']: e for e in manifest['episodes']}
    assert episodes['demo_0']['group_id'] == episodes['demo_3']['group_id']
    assert episodes['demo_0']['split'] == episodes['demo_3']['split']
    assert manifest['state_fields'] == ['px', 'py', 'pz', 'r00', 'r10', 'r20', 'r01', 'r11', 'r21']
    assert all(e['source_file'] == str(source) for e in manifest['episodes'])
    with pytest.raises(FileExistsError):
        prepare(source, tmp_path / 'prepared', config)


def test_pose_only_training_sampling_and_checkpoint(config, tmp_path, monkeypatch):
    from dataclasses import replace
    from us_dp.training import model
    from us_dp.training.train import train, load_policy

    def forbidden_encoder(*args, **kwargs):
        raise AssertionError("Pose-only mode must not construct the image encoder")
    monkeypatch.setattr(model, 'USFMEncoder', forbidden_encoder)
    c = replace(config, use_image_conditioning=False, usfm_pretrained='/nonexistent.pth')
    policy = model.UltrasoundSplinePolicy(c, 9)
    state = torch.randn(2, c.history, 9)
    black = torch.zeros(2, c.history, 1, c.image_size, c.image_size)
    image = torch.rand_like(black)
    cond = policy.condition(image, state)
    torch.testing.assert_close(cond, policy.condition(black, state), rtol=0, atol=0)
    assert cond.shape[-1] == c.feature_dim
    cond.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.state_encoder.parameters())
    policy.eval()
    first = policy.predict(image, state, torch.Generator().manual_seed(3))
    second = policy.predict(black, state, torch.Generator().manual_seed(3))
    torch.testing.assert_close(first['trajectory'], second['trajectory'], rtol=0, atol=0)
    raw = synthetic_episodes(tmp_path / 'raw', config, 6)
    prepare(raw, tmp_path / 'prepared', config)
    checkpoint = train(tmp_path / 'prepared', tmp_path / 'training', epochs=1,
                       use_image_conditioning=False)
    loaded, payload = load_policy(checkpoint)
    assert payload['config']['use_image_conditioning'] is False
    assert loaded.image_encoder is None
    assert not any(k.startswith('image_encoder.') for k in payload['model'])
    assert Config(**{k: v for k, v in config.to_dict().items() if k != 'use_image_conditioning'}).use_image_conditioning
