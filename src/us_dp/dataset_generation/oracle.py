"""Privileged surface-sweep planner. Geometry is never a policy input."""

import numpy as np

from us_dp.common.geometry import to_world, validate_poses


def _random_point_on_circle(rng, radius_m, x_axis, y_axis):
    """Exact-radius sample in the plane spanned by x_axis/y_axis (a true circumference)."""
    angle = rng.uniform(0, 2 * np.pi)
    return radius_m * (np.cos(angle) * x_axis + np.sin(angle) * y_axis)


def _random_point_in_disk(rng, radius_m, x_axis, y_axis):
    """Area-uniform sample anywhere within radius_m, in the same tangent plane."""
    if radius_m == 0:
        return np.zeros(3)
    angle = rng.uniform(0, 2 * np.pi)
    radius = radius_m * np.sqrt(rng.uniform())
    return radius * (np.cos(angle) * x_axis + np.sin(angle) * y_axis)


def _skew(axis):
    x, y, z = axis
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def _rodrigues(axis, angle):
    k = _skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _perturb_rotation(rng, rotation, max_angle_rad):
    """Tilt rotation by a random axis, with the angle sampled uniformly over the
    solid angle of the cone (not just uniform in angle, which would bias toward
    the rim): cos(angle) ~ U[cos(max_angle_rad), 1]."""
    if max_angle_rad == 0:
        return rotation.copy()
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(rng.uniform(np.cos(max_angle_rad), 1.0))
    return _rodrigues(axis, angle) @ rotation


def _slerp_rotation(start, end, phase):
    """Geodesic SO(3) interpolation; phase is (N,) in [0,1]."""
    relative = start.T @ end
    cosine = np.clip((np.trace(relative) - 1) / 2, -1, 1)
    angle = np.arccos(cosine)
    if angle < 1e-8:
        return np.tile(start, (len(phase), 1, 1))
    axis = np.array(
        [relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]]
    ) / (2 * np.sin(angle))
    phased_angles = phase * angle
    k = _skew(axis)
    k2 = k @ k
    increments = (
        np.eye(3) + np.sin(phased_angles)[:, None, None] * k + (1 - np.cos(phased_angles))[:, None, None] * k2
    )
    return np.einsum("ij,njk->nik", start, increments)


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


def straight_line_reach(
    target_pose,
    rng,
    *,
    samples=101,
    start_radius_m=0.10,
    end_radius_m=0.03,
    orientation_cone_rad=np.radians(30),
):
    """Randomized initial positioning that converges near a fixed anatomical target.

    Precedes surface_sweep: a straight-line position interpolation and a
    geodesic SO(3) interpolation (both eased with the same quintic timing as
    surface_sweep), from a randomized start to a randomized end pose that
    lands close to, but not exactly on, target_pose.

    Position sampling uses target_pose's own tangent plane (its rotation
    columns 0/1, perpendicular to the insertion axis in column 2): the start
    is on the exact circle of radius start_radius_m around the target (a true
    circumference, not a filled disk), the end is anywhere inside the disk of
    radius end_radius_m. Sampling a different end pose per episode -- rather
    than the exact fixed target -- keeps recorded demonstrations from
    collapsing onto one terminal state.

    Both the start and end orientation are independently drawn from the same
    orientation_cone_rad around target_pose's orientation (uniform over the
    cone's solid angle), then geodesically interpolated.
    """
    validate_poses(target_pose)
    if not 0 <= end_radius_m <= start_radius_m:
        raise ValueError("end_radius_m must be within [0, start_radius_m]")
    if not 0 < orientation_cone_rad <= np.pi:
        raise ValueError("orientation_cone_rad must be within (0, pi]")
    if samples < 3:
        raise ValueError("samples must be at least 3")
    target_position, target_rotation = target_pose[:3, 3], target_pose[:3, :3]
    x_axis, y_axis = target_rotation[:, 0], target_rotation[:, 1]
    start_position = target_position + _random_point_on_circle(rng, start_radius_m, x_axis, y_axis)
    end_position = target_position + _random_point_in_disk(rng, end_radius_m, x_axis, y_axis)
    start_rotation = _perturb_rotation(rng, target_rotation, orientation_cone_rad)
    end_rotation = _perturb_rotation(rng, target_rotation, orientation_cone_rad)
    phase = np.linspace(0, 1, samples)
    eased = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    positions = start_position + eased[:, None] * (end_position - start_position)
    rotations = _slerp_rotation(start_rotation, end_rotation, eased)
    poses = np.tile(np.eye(4), (samples, 1, 1))
    poses[:, :3, :3], poses[:, :3, 3] = rotations, positions
    validate_poses(poses)
    return {
        "positions_world": positions.astype(np.float32),
        "rotations_world": rotations.astype(np.float32),
    }
