"""Explicit HDF5 mapping for existing i4h recordings; never fits commanded actions."""

from pathlib import Path

import numpy as np

from .data import save_episode
from .geometry import rotation_matrix


def import_hdf5(path, output, mapping):
    """mapping declares relative obs paths, state_fields, quaternion order and timebase.

    Standard recordings that lack measured TCP pose/velocity cannot be silently
    promoted to expert demonstrations. Missing fields are errors.
    """
    import h5py

    from .collection import STATE_FIELDS

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(path, "r") as handle:
        if "data" not in handle or not any(name.startswith("demo_") for name in handle["data"]):
            raise ValueError("HDF5 file contains no data/demo_* episodes")
        for name, demo in handle["data"].items():
            if not name.startswith("demo_"):
                continue
            group = demo.attrs.get(mapping.get("group_attribute", "group_id"))
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
            else:
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
