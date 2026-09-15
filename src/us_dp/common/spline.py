"""Anchored least squares using upstream's C1-constrained Bernstein basis."""

import torch
from torch import nn

from us_dp.common.upstream import classes


class SplineCodec(nn.Module):
    def __init__(self, segments=4, samples=21, repo=None):
        super().__init__()
        spline_cls, _ = classes(repo)
        # Upstream basis construction is CPU-only; register the resulting matrices
        # as buffers so .to(device) also migrates the spline decoder correctly.
        self.poly = spline_cls(nbSeg=segments, nbDim=3, nPoints=samples, device="cpu")
        self.segments = segments
        self.samples = samples
        _, _, phi = self.poly.computePsiList1D(torch.linspace(0, 1, samples))
        self.register_buffer("phi", phi)
        self.register_buffer("control_matrix", self.poly.C)
        self.register_buffer("fit_matrix", torch.linalg.pinv(phi[:, 1:]))
        if not torch.allclose(phi[0], torch.eye(segments + 2)[0], atol=1e-6):
            raise RuntimeError("Upstream spline start is no longer its first independent parameter")

    def fit(self, local_trajectory):
        """Fit measured samples with p(0)=0 fixed in the current probe frame."""
        if local_trajectory.shape[-2:] != (self.samples, 3):
            raise ValueError("Unexpected trajectory shape")
        if not torch.allclose(
            local_trajectory[..., 0, :],
            torch.zeros_like(local_trajectory[..., 0, :]),
            atol=1e-5,
        ):
            raise ValueError("The current probe position must be the first trajectory sample")
        free = torch.einsum("kn,...nd->...kd", self.fit_matrix, local_trajectory)
        return torch.cat((torch.zeros_like(free[..., :1, :]), free), dim=-2)

    def decode(self, params):
        return torch.einsum("nk,...kd->...nd", self.phi, params)

    def control_points(self, params):
        return torch.einsum("nk,...kd->...nd", self.control_matrix, params).reshape(
            *params.shape[:-2], self.segments, 3, 3
        )

    def sample(self, params, phase):
        """Continuous evaluation, including the exact right endpoint."""
        phase = torch.as_tensor(phase, device=params.device, dtype=params.dtype)
        if (
            phase.ndim != 1
            or not torch.isfinite(phase).all()
            or torch.any((phase < 0) | (phase > 1))
        ):
            raise ValueError("phase must be a finite vector in [0,1]")
        index = (phase * self.segments).long().clamp(max=self.segments - 1)
        u = (phase * self.segments - index)[..., None]
        cp = self.control_points(params)[..., index, :, :]
        return (1 - u) ** 2 * cp[..., 0, :] + 2 * u * (1 - u) * cp[..., 1, :] + u**2 * cp[..., 2, :]

    def unpack(self, params):
        """Insert the fixed origin into flattened independent XYZ parameters."""
        if params.ndim != 2 or params.shape[1] != 3 * (self.segments + 1):
            raise ValueError("Expected [B, 3*(segments+1)] free spline parameters")
        free = params.reshape(params.shape[0], self.segments + 1, 3)
        return torch.cat((torch.zeros_like(free[:, :1]), free), dim=1)

    def forward(self, params, query_times, horizon_seconds=2.0):
        """Differentiable [B,D_param] -> [B,T,3]; query_times are seconds."""
        if not 0 < horizon_seconds < float("inf"):
            raise ValueError("horizon_seconds must be finite and positive")
        times = torch.as_tensor(query_times, device=params.device, dtype=params.dtype)
        return self.sample(self.unpack(params), times / horizon_seconds)
