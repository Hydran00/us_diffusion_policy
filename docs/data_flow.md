# US-DP: data flow, mathematics, and assumptions

This is the implementation reference for `src/us_dp/`. Paths below are relative to
the `us_dp` directory. The policy predicts **TCP position only**. Anatomical
geometry may guide an expert demonstration; the trained policy receives only
ultrasound images and measured TCP pose. A Cartesian controller, sensor timing,
contact handling, and simulator stepping remain the caller's responsibility.

## End-to-end map

```text
Skin/Liver OBJ -> anatomy landmarks -> privileged reach/sweep reference
                                         | caller's simulator/controller
                                         v
                              measured TCP + B-mode + time
                                         |
                   EpisodeRecorder or HDF5 import -> raw NPZ
                                         |
                        prepare: validate, group split, local spline fit
                                         v
                         manifest.json + prepared episode NPZ
                                         |
                         WindowDataset -> training statistics
                                         |
                   USFM + state MLP -> conditional U-Net -> DDPM
                                         |
                             best.pt / last.pt checkpoint
                                         |
                       RecedingHorizonPolicy.observe / plan
                                         v
                       timed world-frame Cartesian references
```

The `synthetic` and `smoke` commands exercise software with fabricated images;
`collect-reach` produces geometry-only NPZ files. Neither is a source of real
ultrasound demonstrations. `view-hdf5` and `view-anatomy` are inspection paths,
not training transforms.

## 1. Anatomy and expert references

| Stage | Code | Input and output |
| --- | --- | --- |
| Mesh extraction | `anatomy_processing/assets.py` | Copies `Skin.obj` and `Liver.obj` from the Docker image and records image ID and SHA256. |
| Landmarks | `anatomy_processing/viewer.py`, `frames.py` | Reads full resolution meshes, converts mm to m by default, writes `landmarks.json` and simplified display PLYs. |
| Reach | `dataset_generation/oracle.py`, `reach_demo.py` | Builds a randomized kinematic approach to a fixed anatomical target. `reach_demo.py` has no Isaac physics or ultrasound. |
| Sweep | `dataset_generation/oracle.py` | Builds a privileged skin-surface reference with positions and normals. |
| Recording | `dataset_generation/collection.py` | Caller steps the simulator; recorder saves synchronized **measured** data. |

`estimate_surface_frame` integrates each triangle by area. For triangle vertices
`v_i` and area `A`, its uniform-surface first moment is `A(sum v_i)/3` and second
moment is `A[sum(v_i v_i^T) + (sum v_i)(sum v_i)^T]/12`. Divide accumulated
moments by total area; covariance is `E[xx^T] - mean mean^T`. Eigenvectors ordered
by decreasing eigenvalue form the PCA axes. This is a **surface centroid**, not a
volume center. Similar eigenvalues (gap <= `0.001 * largest variance`) mark axes
ambiguous. Calculations use full meshes; `display_voxel_m=0.003` simplifies only
display copies.

`prepare_anatomy` casts a ray from the liver surface centroid along the fixed
`ANTERIOR_AXIS_MESH_FRAME = (0,-1,0)` to the first Skin intersection. Its probe target uses
the negative outward skin normal as insertion axis and the Skin transverse PCA
axis projected into the tangent plane as its first axis. These are geometric
controller conventions, not inferred clinical targets. At reset, use
`read_isaac_mesh_to_world`: the phantom root omits the ultrasound renderer's
`mesh_to_organ_transform` calibration. The world target is
`T_world_target = T_world_mesh @ T_mesh_target`.

`straight_line_reach` starts on an exact radius circle in the target tangent
plane and ends uniformly within a smaller disk (`r = R sqrt(U)`). Start and end
rotations are independently perturbed within an orientation cone; position and
SO(3) rotation use quintic ease `s(u)=10u^3-15u^4+6u^5`. `surface_sweep` takes a
preselected **external Skin ROI** (at least six points/normals), takes its first
PCA axis, fits quadratic least-squares curves to positions and normals versus
axis coordinate, then queries between its 5th and 95th percentiles with the
same easing. Its optional `offset_m` is along the fitted normal. Neither
routine enforces contact or solves robot dynamics.

`collect_oracle_episode` and `collect_oracle_reach` call user-provided `observe`
and `execute_reference` callbacks at `1/sample_hz`. They first record the
measured scan/reach start and then one observation after each command. They
reject cadence mismatch. `read_isaac_observation` reads arm q/dq, calibrated TCP
pose and B-mode; dB display conversion is
`uint8(round(255 * clip((db + 60)/60, 0, 1)))`. The caller must accept only a
**new** ultrasound `frame_id` and reject failed/reset/contact-lost episodes.
Spline labels later come from measured TCP positions, never the oracle commands.

## 2. Raw episode contract and HDF5 import

`dataset/processing.py:save_episode` writes a compressed NPZ with equal-length
streams: `timestamps` `[T]` in strictly increasing simulation seconds,
`ultrasound` `[T,H,W]` grayscale `uint8`, `robot_state` `[T,D]`, and `probe_pose`
`[T,4,4]` in metres with proper local-to-world rotation. JSON metadata names
`episode_id`, `group_id`, ordered `state_fields`, and `source`; optional
privileged metadata is saved but never put in policy observations. `group_id`
must identify an independent phantom/anatomy configuration.

The two accepted ordered raw state schemas are:

| Schema | Column order | Use after `prepare` |
| --- | --- | --- |
| 9D | `px,py,pz,r00,r10,r20,r01,r11,r21` | All nine columns. |
| 23D | `q0..q6,dq0..dq6` then the 9D pose | Only the final nine pose columns. |

The rotation entries are the first two **columns** of local-to-world `R`.
Joint values can be recorded but are currently discarded by `prepare`. Normal
force/contact are allowed in a raw recorder schema, but `prepare` accepts only
the exact 9D or 23D ordering above. The image and pose for a row are assumed
synchronized; a timestamp alone cannot prove sensor freshness.

`import-hdf5` (`dataset/convert.py`) reads `data/demo_*` with explicit mapping
JSON. It requires a measured probe pose, either one `[T,7]` position/quaternion
dataset or separate position and quaternion datasets. Quaternion order is
declared (`xyzw` or `wxyz`), normalized by `rotation_matrix`, never guessed.
`image_format=db` applies the fixed `[-60,0]` dB window above;
`rgb_uint8` uses `round(mean(R,G,B))`; `gray_uint8` uses values as supplied.
Timestamps come from the mapped dataset or `arange(T)/sample_hz`. A complete
q/dq/joint-index mapping produces 23D; omitting all three produces 9D; an
explicit `robot_state` mapping must declare ordered `state_fields`.

The ordinary `prepare --raw <HDF5 or i4h run directory>` path uses a built-in
mapping: `obs/ultrasound` RGB, `obs/measured_ee_pose` wxyz, and
`obs/timestamps`. It selects only `success=True` episodes, which does **not**
prove continuous contact. It uses the `group_id` attribute or SHA256 of the
initial `obs/phantom_pose` rounded to five decimals. The latter groups by
initial configuration even if the phantom later moves. The HDF5 is converted
to temporary raw NPZ before normal preparation. A run directory is resolved
through its `run.json` recording path.

## 3. Preparation: windows, geometry, and splits

`prepare` requires timestamps spaced at `1/sample_hz` within 1% relative and
`1e-5` s absolute tolerance. Episodes cannot mix synthetic and simulator
sources; all must share the same ordered raw state schema. It splits sorted
unique group IDs after seeded shuffling, **before** creating windows. The first
`max(1,int(G*validation_fraction))` groups are validation, the next analogous
count test, and the rest train; at least three groups are needed. Multiple
episodes with one group stay in the same split.

For every anchor index `t` from `history-1` to `T-future_steps-1`, observations
are indices `t-history+1 ... t`; the measured position target is indices
`t ... t+future_steps` (inclusive). Defaults are 3 history frames and 50 future
intervals at 50 Hz, hence `[3,1,128,128]` images and `[51,3]` target positions.
`image_tensor` rescales uint8 to `[0,1]` and bilinearly resizes to
`image_size x image_size` with `align_corners=False`. No image augmentation or
temporal interpolation is performed.

For anchor TCP transform `(R_t,p_t)`, the local target is
`x_i = R_t^T(p_{t+i}-p_t)` (row-vector implementation:
`(p_{t+i}-p_t) @ R_t`). Thus `x_0=0`. `SplineCodec.fit` uses the upstream
quadratic C1 Bernstein basis `Phi` and fixes its first independent XYZ vector
to zero. The remaining coefficients solve
`W_free = pinv(Phi[:,1:]) X` in least squares. `decode(W)=Phi W`.
`manifest.fit_rmse_m = sqrt(sum((decode(W)-X)^2)/(number of XYZ scalars))`;
this is a **per-coordinate** RMSE. Fitting is batched in groups of 256 anchors.

Prepared `episode_*.npz` contains original uint8 images, only the 9D pose
state, anchor indices, fitted `spline_params` `[windows, K+2,3]` including the
fixed zero row, and measured local `trajectory` `[windows,future_steps+1,3]`.
`manifest.json` records config, upstream checkout/revision, schema, group
assignment, and fit RMSE. `WindowDataset` lazily forms image/state histories
and caches at most two episode NPZ files. No duplicated image windows are
stored.

`WindowDataset.statistics()` runs only on train episodes. For each state
column, and for each XYZ spline coefficient coordinate across free parameter
positions, it computes population `mean=E[x]` and
`std=max(sqrt(E[x^2]-mean^2),1e-4)`. The fixed origin is excluded. These
statistics are saved as model buffers in the checkpoint.

## 4. Spline representation

`common/spline.py` imports `QuadraticSpline` and `ConditionalUnet1D` from the
sibling `spline_policy` checkout (override with `SPLINE_POLICY_ROOT` or CLI
`--spline-policy-root`). With `K=4` quadratic segments, three XYZ control
points per segment give 12 vectors. Three C0 joins and three C1 joins remove
six vectors; fixing the local origin removes one more. There are `K+1=5`
predicted XYZ vectors, or **15 scalars**. The stored representation has
`K+2=6` vectors because row zero is inserted deterministically. Continuity
comes from the upstream control matrix `C`, not a soft loss.

Within a segment, `u in [0,1]`, the decoded position is
`(1-u)^2 P0 + 2u(1-u) P1 + u^2 P2`. Global phase is
`time_seconds/prediction_seconds`; the final point is explicitly assigned to
the last segment. `SplineCodec.forward` accepts `[B,15]` free scalars and a
vector of query times in seconds, then inserts the zero row. Query frequency
can differ from the 50 Hz fitting frequency. C1 holds within one plan; no
cross-plan velocity continuity constraint is implemented.

## 5. Conditioning, diffusion, and optimization

`training/model.py` accepts prepared `[B,H,1,S,S]` images and `[B,H,9]`
state. The 23D model input is retained only for older checkpoints or direct
construction. The current `prepare` always writes 9D. State history is
standardized and flattened, then a `9H -> 256 -> feature_dim` SiLU MLP
produces its feature. With image conditioning, `USFMEncoder` replicates each
grayscale frame to three channels, applies ImageNet mean/std, runs a ViT-B/16
USFM backbone per frame, concatenates its 768D frame vectors, and projects
`768H -> 256 -> feature_dim` with SiLU. Image and state features concatenate
to `2*feature_dim`. By default the pretrained backbone is frozen/eval, while
its projection head trains. Demonstration training with images requires
`usfm_pretrained`; the policy checkpoint embeds the loaded backbone weights.

`--no-image-conditioning` creates a state-only model and denoiser condition
width `feature_dim`; no USFM encoder is constructed. Config fields
`pose_only_compact_conditioning` and `use_image_conditioning` are persisted,
but the current model's branch decision uses `use_image_conditioning`.

Free coefficient XYZ values are standardized using train moments. At training
time draw `epsilon ~ N(0,I)` and a uniform discrete diffusion step `t`, then
form `x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon` with the
Diffusers `squaredcos_cap_v2` beta schedule. The upstream 1D conditional U-Net
predicts epsilon. The **only** optimized loss is
`mean((epsilon_hat-epsilon)^2)` over `[B,K+1,3]`. If needed, the free-vector
length is zero-padded internally to a multiple of `2^(len(down_dims)-1)` for
the U-Net; padded outputs are sliced away before the loss. With default widths
`[128,256,512]`, five vectors are padded to eight.

`train.py` seeds Python/NumPy/Torch, uses shuffled mini-batches, AdamW with
`learning_rate`, clips gradient norm to 1.0, and runs `epochs` passes. Each
epoch logs training/validation metrics and saves `last.pt`; `best.pt` is chosen
by **validation noise MSE**, not trajectory error. Training stops after
`early_stopping_patience` (default five) consecutive epochs without a strictly
lower validation noise MSE; the final epoch remains in `last.pt`, while `best.pt`
retains the best model. Validation forks and seeds
the Torch RNG with `seed+1` for a repeatable noise/timestep stream. A checkpoint
contains model and optimizer states, full config, state schema, upstream
revision, epoch and metrics. `load_policy` reconstructs without reopening the
original pretrained USFM file.

At inference, DDPM starts with standard Gaussian `[B,K+1,3]`, runs
`inference_steps` reverse scheduler steps, denormalizes XYZ coefficients, and
inserts the fixed origin. `sampling_clip_range=4.0` bounds the reconstructed
clean coefficients **in standardized units at each reverse step** through the
Diffusers scheduler. `None` disables this for diagnostics. This is a sampler
stability setting, not a physical workspace bound.

## 6. Evaluation and diagnostics

`training/train.py:batch_metrics` computes:

| Metric | Formula and interpretation |
| --- | --- |
| `noise_mse` | Mean squared predicted vs sampled epsilon in normalized coefficient space. |
| `spline_parameter_mse_m2` | Mean squared difference between predicted and fitted **free** XYZ coefficients in metres squared, averaged per scalar. |
| `trajectory_mse_m2` | Mean over samples/times of `||predicted_local - measured_local||²`. Includes fitting error. |
| `mean_position_error_m` | Mean over samples/times of the same Euclidean distance. |
| `spline_fit_mse_m2` | Mean `||fitted_local - measured_local||²`; measures representation error independent of sampling. |

`evaluate` also reports `position_rmse_m=sqrt(trajectory_mse_m2)` and
`spline_fit_coordinate_rmse_m=sqrt(spline_fit_mse_m2/3)`. It verifies key
dataset/checkpoint geometry settings and state schema. `compare-image-conditioning`
evaluates one image and one pose-only checkpoint on the same prepared split;
positive percentage `(error_without-error_with)/error_without*100` favors
images. It does not establish causal benefit if training settings differ.
`image-sensitivity` replaces images with random or black frames while reusing
the same diffusion seed per batch, then reports accuracy and prediction shifts.
Its 1 mm/10% text verdict is a heuristic; shifted predictions alone do not
establish better scanning performance. `training/dry_run.py` and `smoke` are
software checks, not simulation validation.

## 7. Deployment and timing

`deployment/inference.py:RecedingHorizonPolicy` loads a checkpoint. Call
`reset()` between episodes; call `observe(image,state,pose,timestamp)` on each
new synchronized sample, including while the previous reference is executing.
It requires a full history and consecutive timestamps at `sample_hz` (1%
relative, `1e-5` s absolute tolerance). The first observed orientation after
reset is retained as `R0` for all commanded orientations in that episode.

`plan(control_hz)` defaults to the current deployment `Config.execution_seconds`
(0.8 s), including when an older checkpoint stores 0.4 s, or accepts an explicit
`horizon_seconds <= prediction_seconds`. It samples the full local spline,
queries at `j/control_hz` for `j=1..round(horizon_seconds*control_hz)` (omitting
the `t=0` anchor), and maps every point with the **latest measured** pose:
`p_world = R_t p_local + p_t`. Returned `orientations_world` are copies of
`R0`, while translation uses current `R_t`; orientation is neither predicted
nor interpolated. Defaults give 40 references at 50 Hz over 0.8 s from each 1.0 s plan, replanning
from a fresh measured TCP pose after each prefix. The caller must execute
these references and enforce tracking, contact, limits and collision safety.

`deployment/plot_trajectory.py` saves/plots world-frame prediction NPZ and PNG artifacts. `dataset_generation/mesh_cache.py` is an optional display-only decimation cache (default target 20,000 triangles); the HDF5 viewer currently loads the supplied Skin OBJ directly rather than calling it. The viewer in
`dataset_generation/hdf5_viewer.py` replays measured TCP and ultrasound from
HDF5. It uses recorded `obs/timestamps`, or assumes `sample_hz` (default 50)
when absent. It loads a selected episode's images into RAM and uses its
recorded per-frame `obs/mesh_pose`, a supplied 4x4 matrix, an episode attribute,
or a clearly labelled **nominal** pose. A nominal placement does not verify
mesh alignment. Old recordings without sensor frame IDs cannot establish the
age of repeated images.

## Configuration and important defaults

`configs/default.json` overrides a subset of `Config` in `config.py`; absent
keys use dataclass defaults. The *effective* default policy is:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `history`, `sample_hz`, `image_size` | `3`, `50`, `128` | Observation window and resized B-mode. |
| `num_segments`, `prediction_seconds`, `execution_seconds` | `4`, `1.0`, `0.8` | C1 quadratic spline; 50 future sample intervals; 40 sample intervals executed per replan. |
| `diffusion_steps`, `inference_steps`, `sampling_clip_range` | `100`, `100`, `4.0` | DDPM schedule, reverse iterations, standardized clean-sample bound. |
| `down_dims`, `feature_dim` | `(128,256,512)`, `128` | U-Net widths and each modality's feature width. |
| `batch_size`, `epochs`, `early_stopping_patience`, `learning_rate` | `64`, `100`, `5`, `1e-4` | AdamW training; stop after five validation epochs without improvement. |
| `validation_fraction`, `test_fraction`, `seed` | `0.15`, `0.15`, `42` | Group allocation and random seed. |
| `use_image_conditioning`, `usfm_freeze` | `true`, `true` | Use USFM features; freeze pretrained backbone. |
| `usfm_pretrained` | `null` | Must be set for image-conditioned demonstration training. |

`Config` validates positive rates and sizes, at least two U-Net widths divisible
by eight, `inference_steps <= diffusion_steps`, `0 < execution_seconds <=
prediction_seconds`, integral horizon sample counts, and enough future samples
to fit the spline. `ReachConfig` is separate: nominal phantom `(0.6,0,0.09)` m
and yaw `180°`, XY translation range `±0.05` m, yaw range `±180°`, start radius
`0.10` m, end disk radius `0.03` m, orientation cone `30°`, 101 samples at 10 Hz,
seed 0. These defaults affect only `collect-reach`, not policy training.

## Module index and reproducibility limits

| Module | Responsibility |
| --- | --- |
| `config.py`, `cli.py`, `__main__.py` | Validated parameters and public commands. |
| `common/state.py`, `geometry.py`, `spline.py`, `upstream.py` | Ordered state schema, rigid transforms, anchored spline, external checkout import/revision. |
| `dataset_generation/{collection,oracle,reach_demo,synthetic,mesh_cache,hdf5_viewer}.py` | Recorder/adapters, expert geometry, software fixtures, mesh cache and playback. |
| `dataset/{convert,processing}.py` | HDF5 import, raw validation, split, fitting, prepared windows/statistics. |
| `anatomy_processing/{assets,frames,viewer}.py` | Mesh provenance, surface landmarks/transforms, Open3D inspection. |
| `training/{usfm_encoder,model,train,compare,image_sensitivity,dry_run}.py` | Image encoder, DDPM policy, checkpoints, metrics and diagnostics. |
| `deployment/{inference,plot_trajectory}.py` | Receding-horizon references and trajectory plot. |
| `_compat/*.py` | Old import names forwarding to canonical modules. |

Training provenance records a `spline_policy` commit, but exact replay also
depends on the USFM weights, Diffusers/Torch versions, simulation scene,
controller, synchronized sensor frames and selected data. `best.pt` includes
the trained model and USFM weights. Group splitting reduces leakage between
known configurations but does not measure generalization to other patients or
scanners. No code here certifies contact, coverage, tracking, ultrasound image
freshness, or real-world safety.
