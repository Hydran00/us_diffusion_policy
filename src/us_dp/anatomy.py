"""Privileged liver landmarks, estimated once in the acoustic mesh frame.

No Isaac or Open3D imports here: the same landmarks can be transformed at each
simulator reset without loading the policy or re-running PCA in world space.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import rotation_matrix, validate_poses


@dataclass(frozen=True)
class LiverFrame:
    center: np.ndarray
    axes: np.ndarray  # columns, ordered by decreasing surface variance
    variances: np.ndarray
    surface_area: float
    ambiguous_axes: bool

    def to_dict(self):
        return {
            "center_m": self.center.tolist(),
            "axes_columns": self.axes.tolist(),
            "variances_m2": self.variances.tolist(),
            "surface_area_m2": self.surface_area,
            "ambiguous_axes": self.ambiguous_axes,
            "method": "exact_area_weighted_surface_PCA",
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("method") != "exact_area_weighted_surface_PCA":
            raise ValueError("Unsupported liver landmark method")
        center = np.asarray(data["center_m"], dtype=np.float64)
        axes = np.asarray(data["axes_columns"], dtype=np.float64)
        variances = np.asarray(data["variances_m2"], dtype=np.float64)
        if center.shape != (3,) or axes.shape != (3, 3) or variances.shape != (3,):
            raise ValueError("Invalid liver landmark dimensions")
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = axes, center
        validate_poses(pose)
        if not np.isfinite(variances).all() or np.any(variances < 0):
            raise ValueError("Invalid PCA variances")
        return cls(
            center, axes, variances, float(data["surface_area_m2"]), bool(data["ambiguous_axes"])
        )

    def in_world(self, mesh_to_world):
        pose = np.asarray(mesh_to_world, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError("mesh_to_world must be one (4,4) rigid transform in metres")
        validate_poses(pose)
        return LiverFrame(
            pose[:3, :3] @ self.center + pose[:3, 3],
            pose[:3, :3] @ self.axes,
            self.variances.copy(),
            self.surface_area,
            self.ambiguous_axes,
        )


def estimate_surface_frame(vertices, triangles, batch_size=100_000):
    """Exact first/second moments of uniform triangle surface area (metres).

    Integrates each triangle rather than averaging vertices or random samples.
    The center is a SURFACE centroid, not a volumetric center of mass. Works
    with open meshes and is invariant to subdivision of the same surface.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("vertices must be (N,3)")
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh contains nonfinite vertices")
    if (
        triangles.ndim != 2
        or triangles.shape[1] != 3
        or not len(triangles)
        or not np.issubdtype(triangles.dtype, np.integer)
        or triangles.min() < 0
        or triangles.max() >= len(vertices)
    ):
        raise ValueError("triangles must contain valid integer vertex indices (M,3)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    # Offset reduces cancellation when a mesh has a large translated origin.
    origin = vertices.mean(axis=0)
    first = np.zeros(3)
    second = np.zeros((3, 3))
    area_sum = 0.0
    for start in range(0, len(triangles), batch_size):
        p = vertices[triangles[start : start + batch_size]] - origin
        area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2
        sums = p.sum(axis=1)
        # E[xx^T] on a triangle = (sum_i vi vi^T + (sum_i vi)(sum_i vi)^T)/12.
        moment = (np.einsum("nki,nkj->nij", p, p) + np.einsum("ni,nj->nij", sums, sums)) / 12
        first += np.einsum("n,ni->i", area, sums / 3)
        second += np.einsum("n,nij->ij", area, moment)
        area_sum += float(area.sum())
    if area_sum <= 1e-16:
        raise ValueError("Mesh has no nondegenerate surface")
    mean = first / area_sum
    covariance = second / area_sum - np.outer(mean, mean)
    covariance = (covariance + covariance.T) / 2
    variances, axes = np.linalg.eigh(covariance)
    order = np.argsort(variances)[::-1]
    variances, axes = np.maximum(variances[order], 0), axes[:, order]
    # Resolve sign only in the original mesh frame. Transform this frame at
    # reset; recomputing signs in world coordinates would introduce sign flips.
    for i in (0, 1):
        if axes[np.argmax(np.abs(axes[:, i])), i] < 0:
            axes[:, i] *= -1
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    gaps = variances[:-1] - variances[1:]
    ambiguous = bool(np.any(gaps <= max(float(variances[0]), 1e-16) * 1e-3))
    return LiverFrame(origin + mean, axes, variances, area_sum, ambiguous)


def read_isaac_mesh_to_world(env, *, quaternion_order, env_index=0):
    """Use the EXACT frame used by the current i4h ultrasound renderer.

    The phantom rigid body's root pose alone is not sufficient: it omits the
    calibrated mesh_to_organ_transform offset. Call after reset/physics update.
    """
    from i4h_arena.tensor_utils import to_torch

    data = env.unwrapped.scene["mesh_to_organ_transform"].data
    position = to_torch(data.target_pos_w).detach().cpu().numpy()[env_index, 0]
    quaternion = to_torch(data.target_quat_w).detach().cpu().numpy()[env_index, 0]
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation_matrix(quaternion, quaternion_order)
    pose[:3, 3] = position
    validate_poses(pose)
    return pose


def estimate_liver_frame(vertices, triangles, batch_size=100_000):
    """Backward-compatible entry point for liver surface landmarks."""
    return estimate_surface_frame(vertices, triangles, batch_size)
