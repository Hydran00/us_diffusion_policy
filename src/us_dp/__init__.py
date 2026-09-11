"""Ultrasound Diffusion Spline Policy. Isaac imports remain in the simulator process."""

__version__ = "0.1.0"

# Keep historical imports available without loading optional dependencies.
from pathlib import Path as _Path

__path__.append(str(_Path(__file__).parent / "_compat"))
