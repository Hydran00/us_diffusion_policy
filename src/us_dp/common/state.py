"""Ordered observation contracts shared by datasets and checkpoints."""

POSE_FIELDS = ["px", "py", "pz", "r00", "r10", "r20", "r01", "r11", "r21"]
STATE_FIELDS = [f"q{i}" for i in range(7)] + [f"dq{i}" for i in range(7)] + POSE_FIELDS


def validate_state_fields(fields):
    if fields != POSE_FIELDS and fields != STATE_FIELDS:
        raise ValueError("Expected ordered 9-value Cartesian pose or full 23-value STATE_FIELDS contract")
