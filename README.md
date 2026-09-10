# Ultrasound-Guided Diffusion Spline Policy in Isaac Lab

## 1. Project Goal

The goal is to train a Franka robot in Isaac Lab to perform an **ultrasound-guided scan of a simulated organ inside a simulated phantom**.

At run time, the policy should use only information that would realistically be available to the robot:

- real-time ultrasound image(s);
- robot proprioception/state;
- optionally contact/force information if available.

The policy should output a **smooth future probe motion represented by spline parameters**, following the *Spline Policy* formulation rather than a fixed discrete action chunk.

The proposed learning problem is:

$$
p_\theta(\mathbf{w}_{t:t+H}\mid \mathbf{o}_t),
$$

where:

- $\mathbf{o}_t$ is the current multimodal observation;
- $\mathbf{w}$ contains the spline parameters describing the future probe trajectory;
- the distribution over $\mathbf{w}$ is modeled with a conditional Diffusion Policy.

The key architectural idea from *Spline Policy: A Structured Representation for Robot Policies* is preserved:

> keep the perception/policy backbone, but replace the discrete action-chunk output with spline parameters.

---

## 2. Proposed Task

### Task: Ultrasound-Guided Organ Sweep

The robot starts with the ultrasound probe in contact with, or close to, the phantom. The phantom is spawned at a randomized pose.

A target organ is embedded in the phantom. The robot must:

1. detect/localize the organ implicitly from the ultrasound observation;
2. generate a smooth local scan trajectory over the organ;
3. maintain suitable probe contact;
4. repeatedly replan as new ultrasound images arrive;
5. cover a predefined region of the organ.

A good first version should **not** attempt a complete clinical examination. It should solve a more controlled problem:

> Given the current ultrasound view and robot state, predict the next smooth probe trajectory segment that continues an oracle organ-scan path.

The full oracle scan path can be generated using privileged simulator information, but the learned policy must not receive this privileged information at inference time.

---

## 3. Why Use Local Replanning Instead of Predicting One Entire Scan

Generating one spline for the entire scan from one ultrasound frame is unnecessarily difficult and may make the ultrasound observation insufficient.

Instead, first generate a **global oracle organ scan**, then convert it into many local training examples.

At time $t$:

$$
\mathbf{o}_t
\longrightarrow
\mathbf{w}_t
\longrightarrow
\mathbf{x}_{t:t+H}.
$$

The predicted spline covers only the next short horizon $H$.

The robot executes the first part of that spline, acquires a new ultrasound image, and predicts a new spline.

This gives a receding-horizon system:

```text
US + robot state
       |
       v
Diffusion Spline Policy
       |
       v
future local spline
       |
       v
execute first part
       |
       v
new US + robot state
       |
       +------> replan
```

This is preferable because ultrasound information changes continuously with probe motion and contact.

---

## 4. Observation Space

Define the observation as

$$
\mathbf{o}_t =
\left(
I^{US}_{t-L+1:t},
\mathbf{s}_{t-L+1:t}
\right),
$$

where $L$ is a short history length.

A reasonable first choice is:

$$
L=2 \text{ or } 3.
$$

### 4.1 Ultrasound observation

Use a short sequence of grayscale ultrasound frames:

$$
I^{US}_t \in \mathbb{R}^{1\times H_I\times W_I}.
$$

For example:

```text
1 x 128 x 128
```

A history of three frames may be stacked as channels:

```text
3 x 128 x 128
```

or encoded frame-by-frame and aggregated temporally.

The history is useful because a single ultrasound image may not reveal how the observed anatomy is changing with probe motion.

### 4.2 Robot state

Use robot information that will also be available at inference time.

Recommended baseline state:

$$
\mathbf{s}_t =
[
\mathbf{q}_t,
\dot{\mathbf{q}}_t,
\mathbf{p}^{EE}_t,
\mathbf{r}^{EE}_t,
f_t
].
$$

Possible components:

- 7 Franka joint positions;
- 7 joint velocities;
- end-effector/probe Cartesian position;
- probe orientation;
- contact state;
- normal contact force or wrench, if simulated.

Do **not** provide the policy with:

- phantom ground-truth pose;
- organ ground-truth pose;
- organ mesh;
- segmentation unavailable at inference;
- simulator-only privileged anatomical coordinates.

Those values may be used by the **expert/data generator**, but not by the learned policy.

---

## 5. Action Representation

For the first version, avoid predicting raw joint trajectories.

Predict a probe trajectory in Cartesian space.

A particularly clean first problem is to learn only the position trajectory:

$$
\mathbf{x}(t)
=
[x(t),y(t),z(t)].
$$

Probe orientation can initially be computed by the expert/controller from the local surface normal.

Contact force can initially be handled by a classical force controller.

This creates a useful decomposition:

```text
Diffusion Spline Policy
    -> tangential / Cartesian scan path

classical controller
    -> IK
    -> contact regulation
    -> joint control
```

Once this baseline works, the spline output can be extended to include orientation.

---

# 6. Spline Parameterization

Use the piecewise quadratic Bernstein spline representation used in the Spline Policy paper.

For spline segment $i$, define local phase

$$
\tau \in [0,1].
$$

The trajectory for that segment is

$$
\mathbf{f}_i(\tau)
=
(1-\tau)^2\mathbf{w}_i^1
+
2(1-\tau)\tau\mathbf{w}_i^2
+
\tau^2\mathbf{w}_i^3,
$$

where

$$
\mathbf{w}_i^1,\quad
\mathbf{w}_i^2,\quad
\mathbf{w}_i^3
$$

are the three spline control points.

For a 3-D Cartesian trajectory,

$$
\mathbf{w}_i^j \in \mathbb{R}^3.
$$

For $K$ concatenated spline segments, collect the parameters into

$$
\mathbf{w}
=
[
\mathbf{w}_1^1,\mathbf{w}_1^2,\mathbf{w}_1^3,
\dots,
\mathbf{w}_K^1,\mathbf{w}_K^2,\mathbf{w}_K^3
].
$$

The decoded continuous motion can be written compactly as

$$
\mathbf{f}_{\mathbf{w}}(t)
=
\boldsymbol{\phi}(t)\mathbf{w}.
$$

The policy therefore predicts $\mathbf{w}$, not individual future Cartesian samples.

---

## 7. Continuity Constraints

Neighboring spline segments should at least satisfy positional continuity.

### $C^0$ continuity

$$
\mathbf{w}_i^3
=
\mathbf{w}_{i+1}^1.
$$

For smoother replanning, use $C^1$ continuity.

For segment durations $\Delta t_i$ and $\Delta t_{i+1}$:

$$
\frac{\mathbf{w}_i^3-\mathbf{w}_i^2}{\Delta t_i}
=
\frac{\mathbf{w}_{i+1}^2-\mathbf{w}_{i+1}^1}{\Delta t_{i+1}}.
$$

The first implementation should use equal segment duration:

$$
\Delta t_i = \Delta t.
$$

This simplifies the constraint to

$$
\mathbf{w}_i^3-\mathbf{w}_i^2
=
\mathbf{w}_{i+1}^2-\mathbf{w}_{i+1}^1.
$$

### Recommended implementation strategy

Fit the demonstration splines with explicit $C^1$ constraints.

For the policy output, either:

1. predict all control points and project the result back into the valid constraint subspace; or
2. preferably, predict only a set of independent spline variables and reconstruct the constrained control points deterministically.

The second option makes every decoded prediction continuous by construction.

---

# 8. Coordinate Frame for the Spline

Do not initially predict spline control points directly in the global Isaac world frame.

Randomizing the phantom pose would otherwise force the network to learn unnecessary global transformations.

Represent future control points relative to the current probe pose:

$$
{}^{P_t}\mathbf{w}.
$$

For position:

$$
{}^{P_t}\mathbf{p}_{future}
=
R_{WP_t}^{T}
\left(
{}^{W}\mathbf{p}_{future}
-
{}^{W}\mathbf{p}_{P_t}
\right).
$$

This makes the target approximately invariant to global translation and rotation.

The policy then learns:

$$
p_\theta
\left(
{}^{P_t}\mathbf{w}
\mid
I^{US}_{t-L+1:t},
\mathbf{s}_{t-L+1:t}
\right).
$$

At execution time, the predicted local spline is transformed back into the world frame.

This is especially useful when the phantom is spawned at many different poses.

---

# 9. Data Acquisition

## 9.1 Core idea

Use the simulator as a privileged expert.

For each episode:

1. randomize the phantom;
2. access the simulated organ geometry;
3. construct an oracle scan path over the organ;
4. execute the oracle trajectory with the Franka;
5. record ultrasound images and robot states;
6. fit local quadratic splines to the expert trajectory;
7. store observation-to-spline training pairs.

The dataset therefore contains **supervised spline demonstrations**, even though the final policy does not see the ground-truth organ geometry.

---

## 9.2 Randomize the phantom

For every rollout sample a phantom transform

$$
{}^{W}T_P.
$$

Randomize at least:

- translation in $x,y$;
- optionally small $z$ changes;
- yaw;
- moderate roll/pitch if physically meaningful.

Later add domain randomization for:

- organ location inside the phantom;
- organ scale;
- organ shape;
- tissue/acoustic parameters;
- ultrasound gain;
- image noise;
- probe contact force;
- surface deformation.

Start with pose randomization only.

---

## 9.3 Obtain a privileged organ map

Because the organ is simulated, use its ground-truth geometry to generate expert demonstrations.

Possible representations:

- organ surface mesh;
- point cloud sampled from the mesh;
- organ centerline;
- implicit surface;
- predefined anatomical ROI.

This map exists only inside the expert/data-generation pipeline.

For example, sample a surface point cloud

$$
\mathcal{M}_{organ}
=
\{\mathbf{p}_1,\dots,\mathbf{p}_N\}.
$$

Also compute or obtain surface normals

$$
\mathbf{n}_i.
$$

---

## 9.4 Generate an oracle scan curve

Define an organ scan trajectory in the organ/phantom coordinate system before applying the randomized world transform.

Examples:

### Option A — single sweep

Generate one curve following the long axis of the organ.

### Option B — raster scan

Generate several approximately parallel passes over the target ROI.

### Option C — centerline tracking

If the organ has a meaningful centerline, define the probe path from its surface projection.

For a first experiment, use **one smooth sweep**.

Let the oracle surface path be

$$
\gamma_{organ}(u),
\qquad
u\in[0,1].
$$

At each point, compute a desired probe position

$$
\mathbf{p}_{probe}(u)
=
\gamma_{organ}(u)
+
d\,\mathbf{n}(u),
$$

where $d$ is the chosen probe offset/compression convention.

The desired probe orientation should align its acoustic axis with the surface normal.

The global trajectory is obtained by applying the randomized phantom transform:

$$
{}^{W}\mathbf{p}_{probe}(u)
=
{}^{W}T_P\,
{}^{P}\mathbf{p}_{probe}(u).
$$

---

# 10. Important Recommendation: Generate a Global Expert Path, Train on Local Splines

Do not store only:

```text
one ultrasound image -> one full organ spline
```

Instead, execute the full oracle scan and generate sliding local targets.

Suppose the expert rollout contains

$$
\mathbf{x}_0,\mathbf{x}_1,\ldots,\mathbf{x}_T.
$$

At each time $t$, take a future window

$$
\mathbf{X}_t
=
[
\mathbf{x}_t,
\mathbf{x}_{t+1},
\dots,
\mathbf{x}_{t+H}
].
$$

Fit a local spline:

$$
\mathbf{X}_t
\longrightarrow
\mathbf{w}_t.
$$

The corresponding observation is

$$
\mathbf{o}_t
=
(
I^{US}_{t-L+1:t},
\mathbf{s}_{t-L+1:t}
).
$$

Store

$$
(\mathbf{o}_t,\mathbf{w}_t).
$$

A single oracle rollout therefore generates many supervised training samples.

---

# 11. Fitting the Demonstration Spline

Given sampled expert points

$$
\mathbf{x}(t_1),\dots,\mathbf{x}(t_N),
$$

find spline parameters by solving

$$
\mathbf{w}^\star
=
\arg\min_{\mathbf{w}}
\frac{1}{N}
\sum_{j=1}^{N}
\left\|
\boldsymbol{\phi}(t_j)\mathbf{w}
-
\mathbf{x}(t_j)
\right\|_2^2
$$

subject to the desired continuity constraints.

For example:

$$
\mathbf{w}_i^3=\mathbf{w}_{i+1}^1
$$

and optionally

$$
\mathbf{w}_i^3-\mathbf{w}_i^2
=
\mathbf{w}_{i+1}^2-\mathbf{w}_{i+1}^1.
$$

This is a constrained least-squares problem.

The target stored in the dataset is

$$
\mathbf{w}_t^\star.
$$

### Fit the executed trajectory, not only the ideal geometric path

Prefer fitting the trajectory actually executed by the expert controller.

That means the demonstrations include the behavior of the complete expert system:

- Cartesian tracking;
- contact regulation;
- IK;
- robot dynamics.

This avoids training on trajectories that the simulated robot never actually executed.

---

# 12. Dataset Structure

A possible sample is:

```python
sample = {
    "ultrasound":      # [L, 1, H, W]
    "robot_state":     # [L, D_state]
    "spline_params":   # [K, 3, D_action] or free constrained parameters
}
```

Episode metadata can contain:

```python
metadata = {
    "phantom_pose":        # privileged, not policy input
    "organ_pose":          # privileged
    "organ_geometry_id":   # privileged
    "expert_path_id":
    "episode_id":
}
```

Keeping privileged metadata is useful for evaluation and debugging, even if the policy never receives it.

---

# 13. Avoid Data Leakage

Random train/validation splitting at the frame level is insufficient.

Frames from the same trajectory are strongly correlated.

Split by:

- episode;
- phantom pose;
- preferably organ geometry/configuration.

Example:

```text
70% training phantom configurations
15% validation phantom configurations
15% test phantom configurations
```

The test set should contain phantom poses not seen during training.

A stronger generalization test should also contain unseen organ geometry variations.

---

# 14. Diffusion Policy Over Spline Parameters

The clean spline target is

$$
\mathbf{w}^0.
$$

Instead of diffusing an action chunk, diffuse the spline-parameter vector.

For diffusion step $k$:

$$
\mathbf{w}^{k}
=
\sqrt{\bar{\alpha}_k}\mathbf{w}^{0}
+
\sqrt{1-\bar{\alpha}_k}\boldsymbol{\epsilon},
$$

with

$$
\boldsymbol{\epsilon}\sim\mathcal{N}(0,I).
$$

The denoising network receives

$$
(
\mathbf{w}^{k},
k,
\mathbf{o}_t
)
$$

and predicts the added noise:

$$
\hat{\boldsymbol{\epsilon}}
=
\epsilon_\theta
(
\mathbf{w}^{k},
k,
\mathbf{o}_t
).
$$

The standard diffusion objective becomes

$$
\mathcal{L}_{diff}
=
\left\|
\boldsymbol{\epsilon}
-
\epsilon_\theta
(
\mathbf{w}^{k},
k,
\mathbf{o}_t
)
\right\|_2^2.
$$

The important difference from a standard robot Diffusion Policy is simply:

```text
standard DP:
noise discrete future action chunk

this project:
noise future spline parameters
```

The final denoised sample is therefore a spline:

$$
\mathbf{w}^{K}
\rightarrow
\mathbf{w}^{K-1}
\rightarrow
\dots
\rightarrow
\mathbf{w}^{0}.
$$

Then decode:

$$
\mathbf{x}(t)
=
\boldsymbol{\phi}(t)\mathbf{w}^{0}.
$$

---

# 15. Recommended Network Architecture

## 15.1 Ultrasound encoder

Start with a small CNN rather than a very large vision model.

Recommended baseline:

```text
ResNet-18
```

Modify the first convolution for grayscale input.

For a short image history, either:

### Simple baseline

Stack frames along the channel dimension.

For $L=3$:

```text
input shape = [3, 128, 128]
```

### Better later version

Encode frames independently, then use:

- temporal 1-D convolution;
- GRU;
- small Transformer.

Start with frame stacking.

Let:

$$
\mathbf{z}_{US}=E_{US}(I^{US}_{t-L+1:t}).
$$

---

## 15.2 Robot-state encoder

Use an MLP:

$$
\mathbf{z}_{robot}
=
E_s(\mathbf{s}_{t-L+1:t}).
$$

For example:

```text
D_state * L
    ->
256
    ->
128
```

---

## 15.3 Multimodal fusion

Start with concatenation:

$$
\mathbf{c}_t
=
[
\mathbf{z}_{US};
\mathbf{z}_{robot}
].
$$

Do not introduce cross-attention until the simple baseline works.

---

## 15.4 Diffusion backbone

Use a conditional 1-D U-Net over the spline/control-point sequence.

Input:

$$
\mathbf{w}^{k}.
$$

Condition on:

- multimodal feature $\mathbf{c}_t$;
- diffusion timestep embedding $e(k)$.

Output:

$$
\hat{\boldsymbol{\epsilon}}
$$

with exactly the same dimension as $\mathbf{w}^{k}$.

If the spline representation is extremely low-dimensional, an MLP denoiser is also a useful ablation.

---

# 16. Training Pipeline

For each minibatch:

### Step 1 — Load clean data

Load:

$$
(
I^{US}_{t-L+1:t},
\mathbf{s}_{t-L+1:t},
\mathbf{w}^{0}_t
).
$$

### Step 2 — Normalize

Normalize:

- robot states;
- spline parameters;
- Cartesian positions;
- optional force values.

Normalization is particularly important because the diffusion process assumes comparable numerical scales.

### Step 3 — Encode observations

$$
\mathbf{z}_{US}
=
E_{US}(I^{US})
$$

$$
\mathbf{z}_{robot}
=
E_s(\mathbf{s})
$$

$$
\mathbf{c}
=
[\mathbf{z}_{US};\mathbf{z}_{robot}].
$$

### Step 4 — Sample diffusion timestep

$$
k\sim U\{1,\dots,K_D\}.
$$

Use $K_D$ for the number of diffusion steps to avoid confusing it with the number of spline segments $K_S$.

### Step 5 — Sample Gaussian noise

$$
\boldsymbol{\epsilon}
\sim
\mathcal{N}(0,I).
$$

### Step 6 — Corrupt spline parameters

$$
\mathbf{w}^{k}
=
\sqrt{\bar{\alpha}_{k}}\mathbf{w}^{0}
+
\sqrt{1-\bar{\alpha}_{k}}\boldsymbol{\epsilon}.
$$

### Step 7 — Predict noise

$$
\hat{\boldsymbol{\epsilon}}
=
\epsilon_\theta
(
\mathbf{w}^{k},
k,
\mathbf{c}
).
$$

### Step 8 — Diffusion loss

$$
\mathcal{L}_{diff}
=
\|
\boldsymbol{\epsilon}
-
\hat{\boldsymbol{\epsilon}}
\|^2.
$$

### Step 9 — Backpropagation

Update:

- ultrasound encoder;
- robot-state encoder;
- multimodal fusion;
- diffusion denoiser.

Train the complete policy end-to-end.

---

# 17. Optional Trajectory-Level Auxiliary Loss

The Spline Policy paper also emphasizes that the spline decoder is differentiable.

After estimating a clean spline

$$
\hat{\mathbf{w}}^0,
$$

decode it at sampled times:

$$
\hat{\mathbf{x}}(t_j)
=
\boldsymbol{\phi}(t_j)\hat{\mathbf{w}}^0.
$$

An optional auxiliary loss is

$$
\mathcal{L}_{traj}
=
\frac{1}{N}
\sum_j
\|
\hat{\mathbf{x}}(t_j)
-
\mathbf{x}^{expert}(t_j)
\|^2.
$$

A possible total loss is

$$
\mathcal{L}
=
\mathcal{L}_{diff}
+
\lambda_{traj}\mathcal{L}_{traj}.
$$

Treat this as an experiment rather than the initial baseline.

The cleanest first implementation uses only the standard diffusion noise-prediction loss.

---

# 18. Inference

At control time:

1. acquire the latest ultrasound history;
2. read the latest robot state history;
3. build condition $\mathbf{c}_t$;
4. initialize random spline parameters

$$
\mathbf{w}^{K_D}
\sim
\mathcal{N}(0,I);
$$

5. denoise iteratively:

$$
\mathbf{w}^{K_D}
\rightarrow
\dots
\rightarrow
\mathbf{w}^{0};
$$

6. enforce/project spline constraints if they are not built into the parameterization;
7. decode the spline;
8. transform it from the current probe frame to the world frame;
9. execute only its first part;
10. reacquire ultrasound and replan.

---

# 19. Execution Horizon

For example:

```text
prediction horizon: 2.0 s
execution horizon:  0.25-0.50 s
```

The policy predicts a longer smooth path than it actually executes.

This allows continuous correction from new ultrasound observations.

Do not initially execute the complete predicted spline open-loop.

---

# 20. Probe Control

The learned spline should preferably act as a reference for a lower-level controller.

Suggested hierarchy:

```text
Diffusion Spline Policy
        |
        v
Cartesian spline reference
        |
        v
trajectory sampler
        |
        v
Cartesian / operational-space controller
        |
        +---- contact / force controller
        |
        v
Franka joint torques or position targets
```

This keeps learning focused on **where to scan**, rather than forcing the diffusion model to learn all low-level robot dynamics.

---

# 21. How to Ensure the Policy Actually Uses Ultrasound

This is critical.

If all expert trajectories are nearly identical in world coordinates, the policy may ignore the image.

Therefore randomize enough that robot state alone is insufficient.

Useful variation:

- phantom translation;
- phantom rotation;
- organ pose relative to phantom;
- starting probe pose;
- scan direction;
- anatomy shape.

Then evaluate three policies:

### A. Robot state only

$$
p(\mathbf{w}\mid \mathbf{s})
$$

### B. Ultrasound only

$$
p(\mathbf{w}\mid I^{US})
$$

### C. Ultrasound + robot state

$$
p(\mathbf{w}\mid I^{US},\mathbf{s})
$$

The multimodal model should outperform the state-only baseline on randomized/unseen phantom configurations.

Otherwise there is no evidence that ultrasound is contributing meaningfully.

---

# 22. Suggested Initial Hyperparameters

These are starting points, not requirements from the paper.

```yaml
ultrasound_resolution: [128, 128]
observation_history: 3

spline:
  type: quadratic_bernstein
  num_segments: 4
  continuity: C1
  coordinate_frame: current_probe
  dimensions: 3

trajectory:
  prediction_horizon_seconds: 2.0
  execution_horizon_seconds: 0.4

vision_encoder:
  type: resnet18
  pretrained: false   # baseline; ultrasound differs strongly from natural RGB

state_encoder:
  type: mlp
  hidden_dims: [256, 128]

fusion:
  type: concatenation

diffusion:
  backbone: conditional_1d_unet
  training_steps: 100
  prediction_type: epsilon

optimizer:
  type: AdamW
  learning_rate: 1.0e-4
```

Tune these after establishing a working baseline.

---

# 23. Recommended Development Phases

## Phase 0 — Validate the simulator

Before learning anything:

- verify ultrasound changes correctly with probe pose;
- verify organ ground truth is aligned with ultrasound geometry;
- verify contact and coordinate frames;
- visualize probe trajectory, organ surface, and image plane.

---

## Phase 1 — Generate one deterministic oracle scan

Use one phantom pose.

Generate one organ surface sweep.

Fit the expert trajectory with quadratic splines.

Verify:

$$
\text{expert points}
\approx
\text{decoded spline}.
$$

Measure spline fitting error.

---

## Phase 2 — Randomized expert dataset

Randomize phantom pose.

For each pose:

- generate organ map;
- generate oracle scan;
- execute scan;
- record US + robot state;
- produce sliding windows;
- fit local spline labels.

Goal:

```text
many observations -> many supervised local spline targets
```

---

## Phase 3 — Behavior cloning baseline

Before diffusion, train a deterministic network:

$$
\mathbf{o}_t
\rightarrow
\mathbf{w}_t.
$$

Use MSE either on spline parameters or decoded trajectories.

If this cannot learn the task, debugging the diffusion model will be much harder.

---

## Phase 4 — Diffusion Spline Policy

Replace deterministic spline regression with conditional diffusion:

$$
p_\theta(\mathbf{w}\mid\mathbf{o}).
$$

Train using noise prediction.

Compare against deterministic behavior cloning.

---

## Phase 5 — Closed-loop receding-horizon execution

Run the policy inside Isaac Lab.

Repeatedly:

```text
observe -> predict spline -> execute short prefix -> observe -> replan
```

Measure closed-loop success.

---

## Phase 6 — Stronger domain randomization

Add:

- organ geometry variation;
- ultrasound noise;
- gain variation;
- pose perturbations;
- contact variation;
- simulated patient/phantom displacement.

---

## Phase 7 — Spline-specific experiments

Compare:

- discrete Diffusion Policy action chunks;
- Diffusion Spline Policy;
- $C^0$ vs $C^1$ continuity;
- different numbers of spline segments;
- local-frame vs world-frame spline parameters.

---

# 24. Evaluation Metrics

## Task metrics

- organ coverage percentage;
- scan completion rate;
- distance to oracle scan path;
- final localization error;
- time to complete scan.

## Ultrasound-related metrics

If a target feature is visible:

- distance of organ/feature from desired image location;
- percentage of frames where the organ is visible;
- segmentation overlap if ground truth segmentation is available;
- image-quality proxy defined from simulator ground truth.

## Robot/control metrics

- mean Cartesian tracking error;
- maximum contact force;
- force variance;
- trajectory length;
- peak velocity;
- peak acceleration;
- jerk.

## Spline-specific metrics

- spline fitting error;
- continuity error at segment junctions;
- number of output parameters;
- inference latency;
- replanning discontinuity.

---

# 25. Important Ablations

At minimum:

| Experiment | US | Robot State | Spline | Diffusion |
|---|---:|---:|---:|---:|
| State BC | No | Yes | Yes | No |
| US + State BC | Yes | Yes | Yes | No |
| Standard DP | Yes | Yes | No | Yes |
| Diffusion Spline Policy | Yes | Yes | Yes | Yes |

Additional useful comparisons:

- one ultrasound frame vs frame history;
- world-frame vs probe-frame targets;
- $C^0$ vs $C^1$;
- 2 vs 4 vs 8 spline segments;
- deterministic regression vs diffusion;
- state-only vs US-only vs US+state.

---

# 26. Potential Failure Modes

## Policy ignores ultrasound

Cause:

- phantom randomization too small;
- trajectories predictable from robot state.

Fix:

- increase phantom/organ randomization;
- evaluate state-only baseline.

## Poor spline fitting

Cause:

- too few spline segments;
- expert path has sharp changes.

Fix:

- increase segment count;
- smooth expert path;
- use a trajectory more appropriate for probe scanning.

## Diffusion unstable because parameter scales differ

Fix:

- normalize every spline dimension;
- use local probe coordinates;
- normalize robot state.

## Large discontinuities during replanning

Fix:

- impose $C^1$ constraints;
- anchor the first spline point to the current probe position;
- optionally match the initial spline tangent to the current probe velocity.

## Ultrasound does not uniquely determine the desired motion

Fix:

- include robot history;
- include ultrasound history;
- reduce prediction horizon;
- use receding-horizon replanning.

---

# 27. Recommended First Concrete Experiment

A minimal scientifically useful experiment is:

### Environment

- Franka robot;
- one simulated phantom;
- one simulated organ;
- ultrasound probe;
- phantom randomized in planar $x,y,\mathrm{yaw}$.

### Expert

Use ground-truth organ geometry to generate a single smooth scan across the organ.

### Data

Collect approximately:

```text
1,000-5,000 randomized expert episodes
```

if simulation throughput allows it.

Each rollout produces many sliding-window samples.

### Policy input

```text
3 ultrasound frames
+
robot state history
```

### Policy output

```text
4-segment, C1-continuous,
quadratic 3-D Cartesian spline
in the current probe frame
```

### Execution

Decode the predicted spline at the controller rate, execute approximately the first 20% of the prediction, then replan.

### Main question

> Can the policy infer, from ultrasound appearance and robot proprioception, a smooth local scan path that follows the privileged oracle path under unseen phantom poses?

This gives a clear task, a clear supervised data-generation method, and a clear reason for using both ultrasound and spline-based Diffusion Policy.

---

# 28. Conceptual Summary

The complete pipeline is:

```text
              DATA GENERATION

random phantom pose
       |
       v
privileged organ geometry
       |
       v
oracle organ scan planner
       |
       v
Franka executes scan
       |
       +---------------------+
       |                     |
       v                     v
ultrasound history       robot state
       |                     |
       +----------+----------+
                  |
                  v
          observation o_t

executed future robot path
       |
       v
constrained quadratic
spline fitting
       |
       v
clean spline parameters w^0

-------------------------------------------------

                  TRAINING

o_t ----------------------+
                          |
w^0 -> add diffusion noise|
                          |
                          v
             conditional denoiser
                          |
                          v
                 predicted noise
                          |
                          v
                       MSE loss

-------------------------------------------------

                 INFERENCE

US + robot state
       |
       v
condition encoder
       |
random spline parameters
       |
       v
iterative diffusion denoising
       |
       v
predicted spline parameters
       |
       v
continuous Cartesian trajectory
       |
       v
low-level robot/contact controller
       |
       v
execute short prefix
       |
       v
new observation -> replan
```

The core learning problem is therefore

$$
\boxed{
p_\theta
\left(
\mathbf{w}_{future}
\mid
I^{US}_{history},
\mathbf{s}_{robot,history}
\right)
}
$$

rather than

$$
p_\theta
\left(
\text{discrete action chunk}
\mid
\text{observation}
\right).
$$

The simulator provides privileged anatomical information only to create expert spline demonstrations. The final policy must infer the appropriate motion from ultrasound and robot state alone.

---

# 29. References

- M. Tian, Y. Li, S. Liu, A. Ijspeert, and S. Calinon, **“Spline Policy: A Structured Representation for Robot Policies,”** 2026.
- C. Chi et al., **“Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,”** IJRR, 2023.
- J. Ho, A. Jain, and P. Abbeel, **“Denoising Diffusion Probabilistic Models,”** NeurIPS, 2020.
