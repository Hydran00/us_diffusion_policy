"""Raw episodes and lazy observation windows, split by phantom configuration."""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from us_dp.common.geometry import to_local, validate_poses
from us_dp.common.spline import SplineCodec
from us_dp.common.upstream import revision
from us_dp.config import Config
from us_dp.common.state import POSE_FIELDS, validate_state_fields


def image_tensor(images, size):
    """Canonical uint8 grayscale (...,H,W) to (...,1,size,size), in [0,1]."""
    value = np.asarray(images)
    if value.dtype != np.uint8 or value.ndim < 2:
        raise ValueError(
            "Ultrasound must be uint8 grayscale; convert dB using a fixed window at collection"
        )
    shape = value.shape[:-2]
    value = torch.from_numpy(value.copy()).float().reshape(-1, 1, *value.shape[-2:]) / 255
    return F.interpolate(value, (size, size), mode="bilinear", align_corners=False).reshape(
        *shape, 1, size, size
    )


def validate_episode(arrays, metadata):
    required = ("timestamps", "ultrasound", "robot_state", "probe_pose")
    if any(key not in arrays for key in required):
        raise ValueError(f"Episode requires {required}")
    times = arrays["timestamps"]
    n = len(times)
    if times.shape != (n,) or n < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite, strictly increasing simulation seconds")
    if any(len(arrays[k]) != n for k in required):
        raise ValueError("Episode streams must be synchronized and equally long")
    images, state, poses = (arrays[k] for k in required[1:])
    if images.ndim != 3 or images.dtype != np.uint8 or min(images.shape[1:]) < 1:
        raise ValueError("ultrasound must have shape (T,H,W), dtype uint8")
    if state.ndim != 2 or state.shape[1] < 1 or not np.isfinite(state).all():
        raise ValueError("robot_state must be finite (T,D)")
    if poses.shape != (n, 4, 4):
        raise ValueError("probe_pose must be (T,4,4)")
    validate_poses(poses)
    for key in ("episode_id", "group_id", "state_fields", "source"):
        if key not in metadata:
            raise ValueError(f"Missing metadata: {key}")
    if not isinstance(metadata["group_id"], str) or not metadata["group_id"]:
        raise ValueError("group_id must identify the phantom/anatomy configuration")
    fields = metadata["state_fields"]
    if len(fields) != state.shape[1] or len(set(fields)) != len(fields):
        raise ValueError("state_fields must name every state column uniquely")
    # Explicit allowlist: arbitrary task_obs can contain privileged target poses.
    allowed = (
        {f"q{i}" for i in range(7)}
        | {f"dq{i}" for i in range(7)}
        | {
            "px",
            "py",
            "pz",
            "r00",
            "r10",
            "r20",
            "r01",
            "r11",
            "r21",
            "normal_force",
            "contact",
        }
    )
    if not set(fields) <= allowed:
        raise ValueError(f"Unsupported/privileged state fields: {set(fields) - allowed}")


def save_episode(path, arrays, metadata):
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    validate_episode(arrays, metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays, metadata=np.array(json.dumps(metadata)))


def load_episode(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        arrays = {k: data[k] for k in data.files if k != "metadata"}
    validate_episode(arrays, meta)
    return arrays, meta


def split_groups(groups, config):
    names = sorted(set(groups))
    if len(names) < 3:
        raise ValueError(
            "At least three independent group_id values are required for train/validation/test"
        )
    np.random.default_rng(config.seed).shuffle(names)
    nval = max(1, int(len(names) * config.validation_fraction))
    ntest = max(1, int(len(names) * config.test_fraction))
    if nval + ntest >= len(names):
        raise ValueError("Not enough training groups after split")
    return {
        name: ("validation" if i < nval else "test" if i < nval + ntest else "train")
        for i, name in enumerate(names)
    }


def prepare(raw_dir, output_dir, config, repo=None):
    source = Path(raw_dir)
    if source.is_dir() and (source / "run.json").is_file():
        run = json.loads((source / "run.json").read_text())
        source = Path(run["recording"])
        if not source.is_absolute():
            source = Path(raw_dir) / source
    if source.is_file():
        return prepare_recording(source, output_dir, config, repo)
    paths = sorted(source.glob("*.npz"))
    if not paths:
        raise ValueError(f"No raw episodes in {raw_dir}")
    metadata = []
    for path in paths:
        _, meta = load_episode(path)
        metadata.append(meta)
    if len({m["episode_id"] for m in metadata}) != len(metadata):
        raise ValueError("episode_id must be unique")
    state_fields = metadata[0]["state_fields"]
    validate_state_fields(state_fields)
    pose_columns = [state_fields.index(field) for field in POSE_FIELDS]
    if any(m["state_fields"] != state_fields for m in metadata):
        raise ValueError("Every episode must have the same ordered state_fields")
    sources = {m["source"] for m in metadata}
    if "synthetic_smoke" in sources and len(sources) > 1:
        raise ValueError("Do not mix synthetic smoke fixtures with simulator demonstrations")
    assignment = split_groups([m["group_id"] for m in metadata], config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    codec = SplineCodec(config.num_segments, config.future_steps + 1, repo)
    manifest = {
        "schema_version": 1,
        "config": config.to_dict(),
        "upstream": revision(repo),
        "state_fields": POSE_FIELDS,
        "episodes": [],
    }
    squared_error, count = 0.0, 0
    for index, (path, meta) in enumerate(zip(paths, metadata)):
        arrays, _ = load_episode(path)
        if not np.allclose(
            np.diff(arrays["timestamps"]), 1 / config.sample_hz, rtol=0.01, atol=1e-5
        ):
            raise ValueError(
                f"{path}: timestamps must be sampled at {config.sample_hz} Hz; synchronize/resample first"
            )
        anchors = np.arange(config.history - 1, len(arrays["timestamps"]) - config.future_steps)
        if not len(anchors):
            raise ValueError(
                f"{path}: episode too short for observation history and prediction horizon"
            )
        params, targets = [], []
        # Bounded fitting batches; image histories are never duplicated on disk.
        for offset in range(0, len(anchors), 256):
            batch = anchors[offset : offset + 256]
            local = np.stack(
                [
                    to_local(
                        arrays["probe_pose"][t : t + config.future_steps + 1, :3, 3],
                        arrays["probe_pose"][t],
                    )
                    for t in batch
                ]
            )
            target = torch.from_numpy(local).float()
            w = codec.fit(target)
            squared_error += float((codec.decode(w) - target).square().sum())
            count += target.numel()
            params.append(w.numpy())
            targets.append(local.astype(np.float32))
        filename = f"episode_{index:06d}.npz"
        np.savez_compressed(
            output / filename,
            ultrasound=arrays["ultrasound"],
            robot_state=arrays["robot_state"][:, pose_columns].astype(np.float32),
            anchors=anchors,
            spline_params=np.concatenate(params),
            trajectory=np.concatenate(targets),
        )
        manifest["episodes"].append(
            {
                "file": filename,
                "episode_id": meta["episode_id"],
                "group_id": meta["group_id"],
                "split": assignment[meta["group_id"]],
                "samples": len(anchors),
                "source": meta["source"],
                **({"source_file": meta["source_file"]} if "source_file" in meta else {}),
            }
        )
    manifest["fit_rmse_m"] = (squared_error / count) ** 0.5
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class WindowDataset(Dataset):
    def __init__(self, directory, split):
        self.root = Path(directory)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        validate_state_fields(self.manifest["state_fields"])
        self.config = Config(**self.manifest["config"])
        self.entries = [e for e in self.manifest["episodes"] if e["split"] == split]
        self.index = [(e["file"], i) for e in self.entries for i in range(e["samples"])]
        if not self.index:
            raise ValueError(f"Empty {split} split")
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.index)

    def episode(self, filename):
        if filename not in self.cache:
            with np.load(self.root / filename, allow_pickle=False) as f:
                self.cache[filename] = {k: f[k] for k in f.files}
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(filename)
        return self.cache[filename]

    def __getitem__(self, index):
        filename, row = self.index[index]
        data = self.episode(filename)
        t = int(data["anchors"][row])
        history = slice(t - self.config.history + 1, t + 1)
        return {
            "ultrasound": image_tensor(data["ultrasound"][history], self.config.image_size),
            "robot_state": torch.from_numpy(data["robot_state"][history].copy()),
            "spline_params": torch.from_numpy(data["spline_params"][row].copy()),
            "trajectory": torch.from_numpy(data["trajectory"][row].copy()),
        }

    def statistics(self):
        """Training-only streaming moments, shared over parameter/time positions."""
        if any(e["split"] != "train" for e in self.entries):
            raise ValueError("Fit normalization on the training split only")

        def moments(key):
            total = total_sq = None
            n = 0
            for entry in self.entries:
                data = self.episode(entry["file"])
                values = data[key]
                if key == "spline_params":
                    values = values[:, 1:, :]  # fixed anchor is not a stochastic variable
                values = torch.from_numpy(values).double().reshape(-1, values.shape[-1])
                s, sq = values.sum(0), values.square().sum(0)
                total = s if total is None else total + s
                total_sq = sq if total_sq is None else total_sq + sq
                n += len(values)
            mean = total / n
            std = (total_sq / n - mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
            return mean.float(), std.float()

        return {key: moments(key) for key in ("robot_state", "spline_params")}


def prepare_recording(source, output_dir, config, repo=None):
    """Prepare canonical i4h HDF5 recordings without persistent intermediate data."""
    import tempfile
    import h5py
    from us_dp.dataset.convert import import_hdf5

    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    mapping = {
        "ultrasound": "obs/ultrasound",
        "image_format": "rgb_uint8",
        "probe_pose": "obs/measured_ee_pose",
        "quaternion_order": "wxyz",
        "timestamps": "obs/timestamps",
        "group_attribute": "group_id",
        "group_pose": "obs/phantom_pose",
    }
    with h5py.File(source, "r") as handle:
        episodes = [d for name, d in handle["data"].items() if name.startswith("demo_")]
        if any("success" not in d.attrs for d in episodes):
            raise ValueError("Automatic HDF5 preparation requires a success flag on every episode")
        total = len(episodes)
        selected = sum(bool(d.attrs["success"]) for d in episodes)
    if not selected:
        raise ValueError("No successful episodes in recording")
    with tempfile.TemporaryDirectory(prefix="us_dp_prepare_") as temporary:
        raw = Path(temporary) / "raw"
        import_hdf5(source, raw, mapping, successful_only=True,
                    min_samples=config.history + config.future_steps)
        manifest = prepare(raw, output_dir, config, repo)
    manifest["preprocessing"] = {
        "source_file": str(Path(source).resolve()),
        "mapping": mapping,
        "selection": "success=True (does not guarantee continuous contact)",
        "recorded_episodes": total,
        "selected_episodes": selected,
        "excluded_failed_episodes": total - selected,
        "grouping": "group_id attribute, otherwise SHA256 of initial phantom pose rounded to 5 decimals",
    }
    (Path(output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
