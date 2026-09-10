"""Privileged surface-sweep planner. Geometry is never a policy input."""

import numpy as np

from .geometry import to_world, validate_poses


def random_phantom_pose(rng, nominal, translation_xy=0.03, yaw_radians=0.25):
    validate_poses(nominal)
    angle = rng.uniform(-yaw_radians, yaw_radians)
    c, s = np.cos(angle), np.sin(angle)
    pose = np.array(nominal, copy=True)
    pose[:3, :3] = nominal[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pose[:2, 3] += rng.uniform(-translation_xy, translation_xy, 2)
    return pose


def surface_sweep(surface_points, surface_normals, phantom_pose, samples=101, offset_m=0.0):
    """Fit a smooth quadratic sweep through a preselected skin ROI over the organ.

    Inputs must be the externally accessible phantom contact surface in phantom
    metres, with outward normals. An internal organ mesh is NOT a contact surface.
    The ROI is expected to be a narrow band along the desired sweep direction.
    The calibrated controller maps returned surface normals to the probe axis.
    """
    validate_poses(phantom_pose)
    points, normals = np.asarray(surface_points), np.asarray(surface_normals)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape or len(points) < 6:
        raise ValueError("At least six surface points and matching normals required")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(normals).all()
        or samples < 3
        or not np.isfinite(offset_m)
    ):
        raise ValueError("Invalid surface sweep inputs")
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError("Surface normals must be nonzero")
    normals = normals / lengths
    center = points.mean(0)
    _, singular, vh = np.linalg.svd(points - center, full_matrices=False)
    if singular[0] < 1e-6:
        raise ValueError("Degenerate scan ROI")
    axis = vh[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    u = (points - center) @ axis
    # Stay inside the sampled ROI and ease the scan in/out with a quintic phase.
    phase = np.linspace(0, 1, samples)
    phase = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    query = np.quantile(u, 0.05) + (np.quantile(u, 0.95) - np.quantile(u, 0.05)) * phase
    design = np.stack((np.ones_like(u), u, u * u), axis=-1)
    query_design = np.stack((np.ones_like(query), query, query * query), axis=-1)
    curve = query_design @ np.linalg.lstsq(design, points, rcond=None)[0]
    normal = query_design @ np.linalg.lstsq(design, normals, rcond=None)[0]
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError("Inconsistent surface normals")
    normal /= norm
    return {
        "positions_world": to_world(curve + offset_m * normal, phantom_pose).astype(np.float32),
        "normals_world": (normal @ phantom_pose[:3, :3].T).astype(np.float32),
    }
