"""Explicit HDF5 mapping for existing i4h recordings; never fits commanded actions."""

from pathlib import Path

import numpy as np

from us_dp.common.geometry import rotation_matrix
from us_dp.dataset.processing import save_episode


def import_hdf5(path, output, mapping, *, successful_only=False, min_samples=None):
    """mapping declares relative obs paths, state_fields, quaternion order and timebase.

    Standard recordings that lack measured TCP pose cannot be silently
    promoted to expert demonstrations. Missing fields are errors.
    """
    import h5py

    from us_dp.common.state import POSE_FIELDS, STATE_FIELDS, validate_state_fields

    joint_keys = {"joint_position", "joint_velocity", "arm_joint_indices"}
    if "robot_state" in mapping:
        validate_state_fields(mapping.get("state_fields"))
    elif joint_keys & mapping.keys() and not joint_keys <= mapping.keys():
        raise ValueError("Map all of joint_position, joint_velocity and arm_joint_indices, or omit all for Cartesian pose")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(path, "r") as handle:
        if "data" not in handle or not any(name.startswith("demo_") for name in handle["data"]):
            raise ValueError("HDF5 file contains no data/demo_* episodes")
        for name, demo in handle["data"].items():
            if not name.startswith("demo_"):
                continue
            if successful_only:
                if "success" not in demo.attrs:
                    raise ValueError(f"{name}: missing success flag")
                if not bool(demo.attrs["success"]):
                    continue
            if min_samples is not None and len(demo[mapping["ultrasound"]]) < min_samples:
                raise ValueError(f"{name}: successful episode too short; requires {min_samples} frames")
            group = demo.attrs.get(mapping.get("group_attribute", "group_id"))
            if group is None and mapping.get("group_pose"):
                import hashlib

                phantom = np.asarray(demo[mapping["group_pose"]][()], dtype=np.float64)
                if not np.isfinite(phantom).all() or len(phantom) == 0:
                    raise ValueError(f"{name}: invalid phantom pose")
                # Group by initial configuration; physics can move the phantom later.
                canonical = np.round(phantom[0], 5)
                canonical[canonical == 0] = 0
                group = hashlib.sha256(canonical.astype("<f8").tobytes()).hexdigest()
            if group is None:
                raise ValueError(
                    f"{name}: missing configuration/group attribute; assign before splitting"
                )
            if isinstance(group, bytes):
                group = group.decode()
            image = demo[mapping["ultrasound"]][()]
            image_format = mapping["image_format"]
            if image_format == "db":
                if image.ndim == 4 and image.shape[-1] == 1:
                    image = image[..., 0]
                if not np.isfinite(image).all():
                    raise ValueError(f"{name}: nonfinite B-mode")
                image = np.rint(np.clip((image + 60) / 60, 0, 1) * 255).astype(np.uint8)
            elif image_format == "rgb_uint8":
                if image.dtype != np.uint8 or image.ndim != 4 or image.shape[-1] != 3:
                    raise ValueError("Expected (T,H,W,3) uint8 ultrasound display")
                image = np.rint(image.mean(axis=-1)).astype(np.uint8)
            elif image_format != "gray_uint8":
                raise ValueError("image_format must be db, rgb_uint8 or gray_uint8")
            if "probe_pose" in mapping:
                # A single (T, 7) [pos_xyz, quat] dataset, e.g. i4h_arena's
                # measured_ee_pose, instead of two separate datasets.
                combined = demo[mapping["probe_pose"]][()]
                if combined.ndim == 3 and combined.shape[1] == 1:
                    combined = combined[:, 0]
                position, quaternion = combined[..., :3], combined[..., 3:]
            else:
                position = demo[mapping["probe_position"]][()]
                quaternion = demo[mapping["probe_quaternion"]][()]
                if position.ndim == 3 and position.shape[1] == 1:
                    position, quaternion = position[:, 0], quaternion[:, 0]
            rotation = rotation_matrix(quaternion, mapping["quaternion_order"])
            poses = np.tile(np.eye(4, dtype=np.float32), (len(position), 1, 1))
            poses[:, :3, :3], poses[:, :3, 3] = rotation, position
            if "robot_state" in mapping:
                state = demo[mapping["robot_state"]][()]
                fields = mapping["state_fields"]
            elif joint_keys <= mapping.keys():
                q, dq = (
                    demo[mapping["joint_position"]][()],
                    demo[mapping["joint_velocity"]][()],
                )
                joint_indices = mapping["arm_joint_indices"]
                if len(joint_indices) != 7 or len(set(joint_indices)) != 7:
                    raise ValueError(
                        "arm_joint_indices must identify the seven Franka arm joints in order"
                    )
                state = np.concatenate(
                    (
                        q[:, joint_indices],
                        dq[:, joint_indices],
                        position,
                        rotation[:, :, 0],
                        rotation[:, :, 1],
                    ),
                    axis=-1,
                )
                fields = STATE_FIELDS
            else:
                state = np.concatenate((position, rotation[:, :, 0], rotation[:, :, 1]), axis=-1)
                fields = POSE_FIELDS
            times = (
                demo[mapping["timestamps"]][()]
                if "timestamps" in mapping
                else np.arange(len(image)) / float(mapping["sample_hz"])
            )
            save_episode(
                output / f"{name}.npz",
                {
                    "timestamps": times,
                    "ultrasound": image,
                    "robot_state": state.astype(np.float32),
                    "probe_pose": poses,
                },
                {
                    "episode_id": name,
                    "group_id": str(group),
                    "state_fields": fields,
                    "source": "i4h_hdf5",
                    "source_file": str(Path(path).resolve()),
                },
            )
    return output
