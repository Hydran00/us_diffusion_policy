# Ultrasound Diffusion Spline Policy (US-DP)

US-DP learns measured TCP **position** trajectories from ultrasound and TCP pose history. It fits demonstrations to a locally anchored C1 quadratic spline, trains a conditional DDPM on its free XYZ coefficients, and produces timed world-frame Cartesian references. Expert anatomy is excluded from policy observations.

![US-DP framework: acquisition, preparation, training, and deployment](docs/figures/framework.png)

Read [the implementation guide](docs/data_flow.md) for the full data flow, schemas, formulas, assumptions, parameters, metrics and module index. See [usage](usage.md) for commands.

## Quick start

Install the project with `pip install -e '.[hdf5]'`, make the sibling `spline_policy` checkout available, and install `USFM/usdsgen` for image conditioning. Set `usfm_pretrained` in the config to a USFM checkpoint for demonstration training.

```bash
us-dp prepare --raw /path/to/run-or-raw-npz --output data/prepared --config configs/default.json
us-dp train --dataset data/prepared --output runs/train_001 --device cuda
us-dp evaluate --checkpoint runs/train_001/best.pt --dataset data/prepared --device cuda
```

`prepare` persists the effective config in `manifest.json`. `train --no-image-conditioning` creates a pose-only model. `us-dp smoke --output /tmp/us-dp-smoke` checks the software path using fabricated images.

Training stops after five consecutive epochs without an improvement in `loss/validation_noise_mse` (configurable with `early_stopping_patience`). `best.pt` retains the lowest-validation-loss model; `last.pt` records the final completed epoch.

## Current policy contract

With defaults, each input window contains three 128-square grayscale frames and three 9-value TCP pose states. The model samples 15 free XYZ spline scalars for a 1.0 s path (50 steps at 50 Hz). The controller normally executes 0.8 s (40 steps), then replans from the measured TCP. Desired orientation remains the first one observed after `reset()`; no orientation or joint trajectory is learned.

Raw recordings may have seven arm positions and velocities before the pose. `prepare` discards these 14 fields. Older 23D checkpoints remain loadable. See [the precise field and frame conventions](docs/data_flow.md#2-raw-episode-contract-and-hdf5-import).

The `src/us_dp` subpackages separate acquisition, anatomy, data preparation, training, and deployment. `_compat` forwards old Python import paths. `us-dp` and `python -m us_dp` enter the same CLI.
