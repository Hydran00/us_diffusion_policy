import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    use_image_conditioning: bool = True
    pose_only_compact_conditioning: bool = True
    history: int = 3
    image_size: int = 128
    num_segments: int = 4
    sample_hz: float = 50.0
    prediction_seconds: float = 1.0
    execution_seconds: float = 0.8
    diffusion_steps: int = 100
    inference_steps: int = 100
    # Bound predicted clean coefficients in standardized units at every DDPM step.
    # None reproduces historical unbounded sampling for diagnostics only.
    sampling_clip_range: float | None = 4.0
    down_dims: tuple = (128, 256, 512)
    feature_dim: int = 128
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 5
    learning_rate: float = 1e-4
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    # E_US (README.md section 15.1) is USFM's pretrained ViT-B/16; see
    # training/usfm_encoder.py.
    usfm_pretrained: str | None = None
    # Freeze the pretrained backbone by default: 200 demonstration episodes is
    # too little data to fine-tune an 85M-parameter ViT without overfitting or
    # forgetting the pretrained ultrasound features; only the projection head
    # on top trains.
    usfm_freeze: bool = True

    def __post_init__(self):
        if type(self.pose_only_compact_conditioning) is not bool:
            raise ValueError("pose_only_compact_conditioning must be a boolean")
        if type(self.use_image_conditioning) is not bool:
            raise ValueError("use_image_conditioning must be a boolean")
        for key in (
            "history",
            "image_size",
            "num_segments",
            "diffusion_steps",
            "inference_steps",
            "feature_dim",
            "batch_size",
            "epochs",
            "early_stopping_patience",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.sampling_clip_range is not None and (
            not np.isfinite(self.sampling_clip_range) or self.sampling_clip_range <= 0
        ):
            raise ValueError("sampling_clip_range must be finite and positive, or None")
        if self.image_size < 16 or self.num_segments < 2:
            raise ValueError("image_size >= 16 and num_segments >= 2 required")
        if self.sample_hz <= 0 or self.learning_rate <= 0:
            raise ValueError("sample_hz and learning_rate must be positive")
        if not 0 < self.execution_seconds <= self.prediction_seconds:
            raise ValueError("Require 0 < execution_seconds <= prediction_seconds")
        for seconds in (self.prediction_seconds, self.execution_seconds):
            if abs(seconds * self.sample_hz - round(seconds * self.sample_hz)) > 1e-6:
                raise ValueError("Horizons must contain an integer number of sampling intervals")
        if self.future_steps < self.num_segments + 1 or self.execution_steps < 1:
            raise ValueError("Insufficient future samples for spline fitting or execution")
        if not 1 <= self.inference_steps <= self.diffusion_steps:
            raise ValueError("inference_steps must be <= diffusion_steps")
        if (
            not 0 < self.validation_fraction < 1
            or not 0 < self.test_fraction < 1
            or self.validation_fraction + self.test_fraction >= 1
        ):
            raise ValueError("Require positive validation/test fractions summing to less than one")
        if len(self.down_dims) < 2 or any(d < 8 or d % 8 for d in self.down_dims):
            raise ValueError("At least two U-Net widths, each divisible by 8, required")

    @property
    def future_steps(self):
        return round(self.prediction_seconds * self.sample_hz)

    @property
    def execution_steps(self):
        return round(self.execution_seconds * self.sample_hz)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text())) if path else cls()


@dataclass(frozen=True)
class ReachConfig:
    """Randomization ranges for the kinematic reach demonstration generator.

    phantom_position_m/phantom_yaw_deg default to the panda_phantom scene's
    own nominal organ placement (i4h_arena.assets.panda_phantom.make_assets):
    pos=[0.6, 0, 0.09], yaw=180deg in the robot base frame.
    """

    phantom_position_m: tuple = (0.6, 0.0, 0.09)
    phantom_yaw_deg: float = 180.0
    phantom_translation_xy_m: float = 0.05
    phantom_yaw_range_deg: float = 180.0
    start_radius_m: float = 0.10
    end_radius_m: float = 0.03
    orientation_cone_deg: float = 30.0
    samples: int = 101
    sample_hz: float = 10.0
    seed: int = 0

    def __post_init__(self):
        if len(self.phantom_position_m) != 3:
            raise ValueError("phantom_position_m must have 3 components")
        if not 0 <= self.phantom_yaw_range_deg <= 180:
            raise ValueError("phantom_yaw_range_deg must be within [0, 180]")
        if self.phantom_translation_xy_m < 0:
            raise ValueError("phantom_translation_xy_m must be non-negative")
        if not 0 <= self.end_radius_m <= self.start_radius_m:
            raise ValueError("end_radius_m must be within [0, start_radius_m]")
        if not 0 < self.orientation_cone_deg <= 180:
            raise ValueError("orientation_cone_deg must be within (0, 180]")
        if self.samples < 3:
            raise ValueError("samples must be at least 3")
        if self.sample_hz <= 0:
            raise ValueError("sample_hz must be positive")

    def nominal_phantom_pose(self):
        yaw = np.radians(self.phantom_yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        pose = np.eye(4)
        pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        pose[:3, 3] = self.phantom_position_m
        return pose

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text())) if path else cls()
