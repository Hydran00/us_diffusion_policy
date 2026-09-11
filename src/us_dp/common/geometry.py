import numpy as np


def rotation_matrix(quaternion, order):
    """Quaternion -> local-to-world matrix. The ordering is never inferred."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("Expected finite quaternions (...,4)")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Zero quaternion")
    q = q / norm
    if order == "wxyz":
        q = q[..., [1, 2, 3, 0]]
    elif order != "xyzw":
        raise ValueError("quaternion order must be xyzw or wxyz")
    x, y, z, w = np.moveaxis(q, -1, 0)
    return (
        np.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ),
            axis=-1,
        )
        .reshape(q.shape[:-1] + (3, 3))
        .astype(np.float32)
    )


def validate_poses(poses):
    poses = np.asarray(poses)
    if poses.shape[-2:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("Expected finite homogeneous probe poses (...,4,4)")
    r = poses[..., :3, :3]
    if not np.allclose(r.swapaxes(-1, -2) @ r, np.eye(3), atol=1e-4) or not np.allclose(
        np.linalg.det(r), 1, atol=1e-4
    ):
        raise ValueError("Pose rotation must be in SO(3)")
    if not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous pose bottom row")


def to_local(points, pose):
    return (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]


def to_world(points, pose):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]
