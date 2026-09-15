"""Recording hooks for an existing simulator loop; never owns env.step()."""

from pathlib import Path

import numpy as np

from us_dp.common.geometry import rotation_matrix
from us_dp.dataset.processing import save_episode

from us_dp.common.state import STATE_FIELDS


class EpisodeRecorder:
    def __init__(
        self,
        path,
        episode_id,
        group_id,
        source="isaac",
        state_fields=None,
        **privileged_metadata,
    ):
        self.path = Path(path)
        self.metadata = dict(
            privileged_metadata,
            episode_id=episode_id,
            group_id=group_id,
            source=source,
            state_fields=state_fields or STATE_FIELDS,
        )
        self.rows = []

    def append(self, *, timestamp, ultrasound, robot_state, probe_pose):
        if self.rows and timestamp <= self.rows[-1]["timestamps"]:
            raise ValueError(
                "Record each synchronized observation once, with increasing simulation time"
            )
        self.rows.append(
            {
                "timestamps": float(timestamp),
                "ultrasound": np.array(ultrasound, copy=True),
                "robot_state": np.array(robot_state, dtype=np.float32, copy=True),
                "probe_pose": np.array(probe_pose, dtype=np.float32, copy=True),
            }
        )

    def save(self):
        if len(self.rows) < 2:
            raise ValueError("Episode is empty or incomplete")
        save_episode(
            self.path,
            {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]},
            self.metadata,
        )
        return self.path


def read_isaac_observation(
    env,
    *,
    timestamp,
    quaternion_order,
    env_index=0,
    probe_sensor="ee_frame",
    ultrasound_sensor="ultrasound",
):
    """Read panda_phantom's calibrated TCP and B-mode after the simulator step.

    Explicit quaternion_order supports xyzw in this i4h checkout and wxyz in
    other Isaac versions. Invoke only when a NEW ultrasound frame is available;
    caller owns sampling/stepping and checks the returned frame_id.
    """
    from i4h_arena.tensor_utils import to_torch

    scene = env.unwrapped.scene

    def array(value):
        return to_torch(value).detach().cpu().numpy()

    robot = scene["robot"]
    ids = [list(robot.joint_names).index(f"panda_joint{i}") for i in range(1, 8)]
    q = array(robot.data.joint_pos)[env_index, ids]
    dq = array(robot.data.joint_vel)[env_index, ids]
    tcp = scene[probe_sensor].data
    pos = array(tcp.target_pos_w)[env_index, 0]
    rotation = rotation_matrix(array(tcp.target_quat_w)[env_index, 0], quaternion_order)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3], pose[:3, 3] = rotation, pos
    sensor = scene[ultrasound_sensor].data
    db = array(sensor.output["bmode_db"])[env_index, ..., 0]
    if not np.isfinite(db).all():
        raise ValueError("Nonfinite B-mode image")
    image = np.rint(np.clip((db + 60) / 60, 0, 1) * 255).astype(np.uint8)
    frame_id = int(array(sensor.frame_id)[env_index])
    state = np.concatenate((q, dq, pos, rotation[:, 0], rotation[:, 1])).astype(np.float32)
    return {
        "timestamp": timestamp,
        "ultrasound": image,
        "robot_state": state,
        "probe_pose": pose,
    }, frame_id


def collect_oracle_episode(recorder, reference, observe, execute_reference, *, sample_hz):
    """Execute a global oracle through caller-owned simulator/controller callbacks.

    observe() returns the canonical observation dict (timestamp, ultrasound,
    robot_state, probe_pose). execute_reference(position_world, normal_world, dt)
    runs the calibrated IK/contact controller for exactly dt simulation seconds;
    it must raise on reset, termination, loss of contact or a failed command.
    The first observation is taken at the already-reached scan start. All stored
    spline labels are subsequently fitted from measured poses, never references.
    """
    positions = np.asarray(reference["positions_world"])
    normals = np.asarray(reference["normals_world"])
    if (
        sample_hz <= 0
        or positions.ndim != 2
        or positions.shape[1] != 3
        or len(positions) < 2
        or normals.shape != positions.shape
    ):
        raise ValueError("Invalid oracle reference or sample rate")
    if not np.isfinite(positions).all() or not np.isfinite(normals).all():
        raise ValueError("Nonfinite oracle reference")
    if recorder.rows:
        raise ValueError("Use a fresh recorder for each episode")
    observation = observe()
    recorder.append(**observation)
    for position, normal in zip(positions[1:], normals[1:]):
        previous_time = observation["timestamp"]
        execute_reference(position, normal, 1 / sample_hz)
        observation = observe()
        if not np.isclose(
            observation["timestamp"] - previous_time,
            1 / sample_hz,
            rtol=0.01,
            atol=1e-5,
        ):
            raise ValueError("Controller/sensor cadence differs from the dataset sampling rate")
        recorder.append(**observation)
    return recorder.save()


def collect_oracle_reach(recorder, reference, observe, execute_reference, *, sample_hz):
    """Execute a privileged straight-line reach (oracle.straight_line_reach) before a sweep.

    Same contract as collect_oracle_episode, except execute_reference takes a
    full orientation (position_world, rotation_world, dt): the reach happens
    in free space/first contact, where the controller cannot yet regulate
    from a measured surface normal.
    """
    positions = np.asarray(reference["positions_world"])
    rotations = np.asarray(reference["rotations_world"])
    if (
        sample_hz <= 0
        or positions.ndim != 2
        or positions.shape[1] != 3
        or len(positions) < 2
        or rotations.shape != (len(positions), 3, 3)
    ):
        raise ValueError("Invalid oracle reference or sample rate")
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError("Nonfinite oracle reference")
    if recorder.rows:
        raise ValueError("Use a fresh recorder for each episode")
    observation = observe()
    recorder.append(**observation)
    for position, rotation in zip(positions[1:], rotations[1:]):
        previous_time = observation["timestamp"]
        execute_reference(position, rotation, 1 / sample_hz)
        observation = observe()
        if not np.isclose(
            observation["timestamp"] - previous_time,
            1 / sample_hz,
            rtol=0.01,
            atol=1e-5,
        ):
            raise ValueError("Controller/sensor cadence differs from the dataset sampling rate")
        recorder.append(**observation)
    return recorder.save()
