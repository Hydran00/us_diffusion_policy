"""Inspect one prepared episode and the history/future window at an anchor.

Run from the workspace root, for example:
    python3 us_dp/tests/inspect_episode.py data/prepared_acq_c/episode_000050.npz
    python3 us_dp/tests/inspect_episode.py data/prepared_acq_d/episode_000050.npz --window 17
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Prepared episode_*.npz")
    parser.add_argument("--window", type=int, default=0, help="Window row to inspect (default: 0)")
    args = parser.parse_args()

    manifest = json.loads((args.episode.parent / "manifest.json").read_text())
    config = manifest["config"]
    entry = next((e for e in manifest["episodes"] if e["file"] == args.episode.name), None)
    if entry is None:
        parser.error(f"{args.episode.name} is absent from manifest.json")

    with np.load(args.episode, allow_pickle=False) as episode:
        for key in ("ultrasound", "robot_state", "anchors", "spline_params", "trajectory"):
            value = episode[key]
            print(f"{key:14} shape={str(value.shape):18} dtype={value.dtype}")

        anchors = episode["anchors"]
        if not 0 <= args.window < len(anchors):
            parser.error(f"--window must be between 0 and {len(anchors) - 1}")
        t = int(anchors[args.window])
        history = int(config["history"])
        future_steps = round(config["prediction_seconds"] * config["sample_hz"])
        expected = np.arange(history - 1, len(episode["ultrasound"]) - future_steps)
        print(f"\nepisode_id={entry['episode_id']}  split={entry['split']}")
        print(f"sample_hz={config['sample_hz']:g}  history={history}  "
              f"prediction_seconds={config['prediction_seconds']:g}  "
              f"future_steps={future_steps}  image_size_at_training={config['image_size']}")
        print(f"windows={len(anchors)}  anchors={anchors[0]}..{anchors[-1]}  "
              f"expected_windows={len(expected)}")
        print(f"state_fields={manifest['state_fields']}")
        print(f"\nwindow[{args.window}] anchor t={t}")
        print(f"  input history rows:   {t - history + 1}..{t} (inclusive)")
        print(f"  target future rows:   {t}..{t + future_steps} (inclusive)")
        print(f"  first state at t:     {episode['robot_state'][t]}")
        print(f"  local target start:   {episode['trajectory'][args.window, 0]}")
        print(f"  local target end:     {episode['trajectory'][args.window, -1]}")
        print(f"  fixed spline origin:  {episode['spline_params'][args.window, 0]}")
        print(f"  predicted variables:  {episode['spline_params'].shape[1] - 1} XYZ vectors")
        if not np.array_equal(anchors, expected):
            print("WARNING: anchor indices differ from the current prepare() formula")


if __name__ == "__main__":
    main()
