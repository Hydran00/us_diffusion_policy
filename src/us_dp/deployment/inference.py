"""History-aware local prediction and timed world-frame Cartesian references."""

from collections import deque

import numpy as np
import torch

from us_dp.common.geometry import to_world, validate_poses
from us_dp.dataset.processing import image_tensor
from us_dp.training.train import load_policy


class RecedingHorizonPolicy:
    def __init__(self, checkpoint, device="cpu", repo=None):
        self.policy, self.metadata = load_policy(checkpoint, device, repo)
        self.device = device
        self.config = self.policy.config
        self.reset()

    def reset(self):
        self.images = deque(maxlen=self.config.history)
        self.states = deque(maxlen=self.config.history)
        self.last_timestamp = None
        self.pose = None

    def observe(self, ultrasound, robot_state, probe_pose, timestamp):
        """Call at sample_hz, including while executing the previously planned prefix."""
        if np.asarray(probe_pose).shape != (4, 4):
            raise ValueError("Expected one probe pose (4,4)")
        validate_poses(probe_pose)
        if np.asarray(ultrasound).ndim != 2:
            raise ValueError("Expected one grayscale ultrasound frame (H,W)")
        image_tensor(ultrasound, self.config.image_size)
        state = np.asarray(robot_state, dtype=np.float32)
        if state.shape != (self.policy.state_dim,) or not np.isfinite(state).all():
            raise ValueError("Invalid robot state")
        if not np.isfinite(timestamp):
            raise ValueError("Invalid observation timestamp")
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if not np.isclose(dt, 1 / self.config.sample_hz, rtol=0.01, atol=1e-5):
                raise ValueError(
                    "Observations must be consecutive at the training sample_hz; reset history after a gap"
                )
        self.images.append(np.array(ultrasound, copy=True))
        self.states.append(state.copy())
        self.pose = np.array(probe_pose, copy=True)
        self.last_timestamp = float(timestamp)

    @torch.no_grad()
    def plan(self, control_hz=None, generator=None):
        if len(self.images) < self.config.history:
            raise RuntimeError("Collect a full observation history before planning")
        hz = self.config.sample_hz if control_hz is None else control_hz
        steps = round(self.config.execution_seconds * hz)
        if hz <= 0 or steps < 1 or not np.isclose(steps, hz * self.config.execution_seconds):
            raise ValueError("Execution duration must contain whole controller intervals")
        image = image_tensor(np.stack(self.images), self.config.image_size)[None].to(self.device)
        state = torch.from_numpy(np.stack(self.states))[None].to(self.device)
        result = self.policy.predict(image, state, generator)
        # Exclude t=0: the measured current position is an anchor, not an action.
        seconds = torch.arange(1, steps + 1, device=self.device) / hz
        prefix = (
            self.policy.codec.sample(
                result["spline_params"], seconds / self.config.prediction_seconds
            )[0]
            .cpu()
            .numpy()
        )
        if not np.isfinite(prefix).all():
            raise RuntimeError("Policy generated nonfinite Cartesian references")
        return {
            "positions_world": to_world(prefix, self.pose),
            "time_from_start": seconds.cpu().numpy(),
            "observation_timestamp": self.last_timestamp,
            "probe_pose_at_plan": self.pose.copy(),
            "spline_params_local": result["spline_params"][0].cpu().numpy(),
        }
