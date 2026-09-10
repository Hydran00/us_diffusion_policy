import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    history: int = 3
    image_size: int = 128
    num_segments: int = 4
    sample_hz: float = 10.0
    prediction_seconds: float = 2.0
    execution_seconds: float = 0.4
    diffusion_steps: int = 100
    inference_steps: int = 100
    down_dims: tuple = (128, 256, 512)
    feature_dim: int = 128
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-4
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42

    def __post_init__(self):
        for key in (
            "history",
            "image_size",
            "num_segments",
            "diffusion_steps",
            "inference_steps",
            "feature_dim",
            "batch_size",
            "epochs",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
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
