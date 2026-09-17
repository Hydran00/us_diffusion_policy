"""Import the existing checkout as a dependency, without copying its source."""

import importlib
import os
import subprocess
import sys
from pathlib import Path


def use_spline_policy(repo=None):
    root = (
        Path(
            repo
            or os.environ.get(
                "SPLINE_POLICY_ROOT",
                Path(__file__).resolve().parents[3] / "spline_policy",
            )
        )
        .expanduser()
        .resolve()
    )
    source = root / "spline_policy" / "policy"
    expected = source / "diffusion_policy" / "planning" / "quadratic_spline.py"
    if not expected.is_file():
        raise FileNotFoundError(
            f"Spline Policy checkout missing at {root}; set SPLINE_POLICY_ROOT or --spline-policy-root"
        )
    for name in ("diffusion_policy", "diffusion_policy.planning.quadratic_spline"):
        loaded = sys.modules.get(name)
        if (
            loaded is not None
            and getattr(loaded, "__file__", None)
            and not Path(loaded.__file__).resolve().is_relative_to(source)
        ):
            raise RuntimeError(
                f"{name} is already imported from another checkout; use a separate process"
            )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return root


def classes(repo=None):
    use_spline_policy(repo)
    spline = importlib.import_module("diffusion_policy.planning.quadratic_spline").QuadraticSpline
    unet = importlib.import_module(
        "diffusion_policy.model.diffusion.conditional_unet1d"
    ).ConditionalUnet1D
    return spline, unet


def revision(repo=None):
    root = use_spline_policy(repo)
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {"root": str(root), "commit": result.stdout.strip() or "unknown"}
