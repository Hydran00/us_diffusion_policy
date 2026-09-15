"""Deterministic synthetic fixtures to exercise software, not simulate ultrasound physics."""

import json
from pathlib import Path

import numpy as np
import torch

from us_dp.common.geometry import to_world
from us_dp.config import Config
from us_dp.dataset.processing import load_episode, prepare
from us_dp.dataset_generation.collection import STATE_FIELDS, EpisodeRecorder
from us_dp.dataset_generation.oracle import random_phantom_pose


def synthetic_episodes(directory, config, episodes=6):
    if episodes < 3:
        raise ValueError("At least three episodes required")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config.seed)
    n = config.history + config.future_steps + 12
    yy, xx = np.mgrid[-1 : 1 : complex(config.image_size), -1 : 1 : complex(config.image_size)]
    for episode in range(episodes):
        nominal = np.eye(4, dtype=np.float32)
        nominal[:3, 3] = [0.5, 0, 0.2]
        phantom = random_phantom_pose(rng, nominal)
        t = np.arange(n) / config.sample_hz
        phase = np.linspace(-1, 1, n)
        local = np.stack((0.04 * phase, 0.006 * (1 - phase**2), 0.002 * np.sin(phase)), axis=-1)
        positions = to_world(local, phantom)
        record = EpisodeRecorder(
            root / f"demo_{episode:04d}.npz",
            str(episode),
            f"synthetic_phantom_{episode}",
            source="synthetic_smoke",
            phantom_pose=phantom.tolist(),
        )
        for i in range(n):
            pose = phantom.copy()
            pose[:3, 3] = positions[i]
            signal = np.exp(-((xx - 0.35 * phase[i]) ** 2 / 0.15 + (yy + 0.15) ** 2 / 0.3))
            gray = np.rint(np.clip(signal + rng.normal(0, 0.02, signal.shape), 0, 1) * 255).astype(
                np.uint8
            )
            state = np.concatenate((np.zeros(14), positions[i], pose[:3, 0], pose[:3, 1])).astype(
                np.float32
            )
            assert len(state) == len(STATE_FIELDS)
            record.append(timestamp=t[i], ultrasound=gray, robot_state=state, probe_pose=pose)
        record.save()
    return root


def smoke(output, repo=None):
    from us_dp.deployment.inference import RecedingHorizonPolicy
    from us_dp.training.train import evaluate, train

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    config = Config(
        image_size=32,
        down_dims=(16, 32),
        feature_dim=16,
        diffusion_steps=4,
        inference_steps=2,
        epochs=1,
        batch_size=8,
    )
    torch.set_num_threads(2)
    (root / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    synthetic_episodes(root / "raw", config)
    manifest = prepare(root / "raw", root / "dataset", config, repo)
    checkpoint = train(root / "dataset", root / "training", repo=repo)
    metrics = evaluate(checkpoint, root / "dataset", repo=repo)
    episode, _ = load_episode(root / "raw" / "demo_0000.npz")
    runner = RecedingHorizonPolicy(checkpoint, repo=repo)
    for i in range(config.history):
        runner.observe(
            episode["ultrasound"][i],
            episode["robot_state"][i, 14:],
            episode["probe_pose"][i],
            episode["timestamps"][i],
        )
    prediction = runner.plan(control_hz=50)
    np.savez_compressed(root / "prediction.npz", **prediction)
    report = {
        "fixture": "synthetic_smoke",
        "fit_rmse_m": manifest["fit_rmse_m"],
        "evaluation": metrics,
        "execution_points": len(prediction["positions_world"]),
        "checkpoint": str(checkpoint),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
