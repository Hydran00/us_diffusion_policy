"""Run and summarize complete closed-loop policy episodes in Isaac."""

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from uuid import uuid4

import h5py
import numpy as np


def summarize_recording(path, expected_episodes):
    episodes = []
    with h5py.File(path, "r") as recording:
        for name, demo in recording["data"].items():
            if not name.startswith("demo_"):
                continue
            obs = demo["obs"]
            measured = np.asarray(obs["measured_ee_pose"][:, :3], dtype=np.float64)
            commanded = np.asarray(obs["commanded_ee_pose"][:, :3], dtype=np.float64)
            timestamps = np.asarray(obs["timestamps"], dtype=np.float64)
            valid = np.isfinite(commanded).all(axis=1)
            if not len(measured) or not np.isfinite(measured).all() or not valid.any():
                raise ValueError(f"{path}: incomplete TCP recording in {name}")
            episodes.append({
                "episode_index": int(demo.attrs["episode_index"]),
                "success": bool(demo.attrs["success"]),
                "steps": len(measured),
                "duration_s": float(timestamps[-1] - timestamps[0]),
                "tcp_path_length_m": float(np.linalg.norm(np.diff(measured, axis=0), axis=1).sum()),
                "mean_tracking_error_m": float(np.linalg.norm(commanded[valid] - measured[valid], axis=1).mean()),
                "initial_tcp_position_m": measured[0].tolist(),
            })
    episodes.sort(key=lambda episode: episode["episode_index"])
    if len(episodes) != expected_episodes:
        raise ValueError(f"{path}: expected {expected_episodes} recorded episodes, found {len(episodes)}")
    return {
        "episodes": episodes,
        "successes": sum(episode["success"] for episode in episodes),
        "success_rate_percent": 100 * np.mean([episode["success"] for episode in episodes]),
        "mean_duration_s": float(np.mean([episode["duration_s"] for episode in episodes])),
        "mean_tcp_path_length_m": float(np.mean([episode["tcp_path_length_m"] for episode in episodes])),
        "mean_tracking_error_m": float(np.mean([episode["mean_tracking_error_m"] for episode in episodes])),
    }


def run_simulator(checkpoint, episodes, seed, device, workflow_root):
    root = Path(workflow_root).resolve()
    launcher = root / "run.sh"
    if not launcher.is_file():
        raise FileNotFoundError(f"Isaac workflow launcher not found: {launcher}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = root / "runs" / "ultrasound_liver_scan" / f"{stamp}_us_dp_compare_{uuid4().hex[:8]}"
    command = [
        str(launcher), "ultrasound_liver_scan", "--policy",
        "--task-id", "us_dp/ultrasound_liver_scan",
        "--checkpoint", str(Path(checkpoint).resolve()),
        "--ultrasound", "--record", "--record-failures",
        "--episodes", str(episodes), "--attempts", "1", "--seed", str(seed),
        "--run-dir", str(run_dir), "--device", device,
    ]
    env = os.environ.copy()
    env["US_DP_DEVICE"] = device
    env["UV_NO_SYNC"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(root / "tasks" / "us_dp"),
        str(root / "common"),
        str(root.parent / "USFM"),
        env.get("PYTHONPATH", ""),
    )))
    result = subprocess.run(command, cwd=root, env=env, check=False)
    recording = run_dir / "demos.hdf5"
    if not recording.is_file():
        raise RuntimeError(f"Simulator exited with code {result.returncode} without a recording: {run_dir}")
    summary = summarize_recording(recording, episodes)
    summary.update({"run_dir": str(run_dir), "recording": str(recording), "exit_code": result.returncode})
    return summary


def compare_simulator(checkpoint_image, checkpoint_no_image, episodes, seed, device, workflow_root):
    if episodes < 1:
        raise ValueError("simulator episodes must be positive")
    image = run_simulator(checkpoint_image, episodes, seed, device, workflow_root)
    no_image = run_simulator(checkpoint_no_image, episodes, seed, device, workflow_root)
    starts_image = {item["episode_index"]: item for item in image["episodes"]}
    starts_no_image = {item["episode_index"]: item for item in no_image["episodes"]}
    start_gaps = [
        np.linalg.norm(np.asarray(starts_image[index]["initial_tcp_position_m"])
                       - np.asarray(starts_no_image[index]["initial_tcp_position_m"]))
        for index in starts_image.keys() & starts_no_image.keys()
    ]
    return {
        "seed": seed,
        "requested_episodes_per_checkpoint": episodes,
        "with_image": image,
        "without_image": no_image,
        "mean_initial_tcp_gap_m": float(np.mean(start_gaps)) if start_gaps else None,
    }
