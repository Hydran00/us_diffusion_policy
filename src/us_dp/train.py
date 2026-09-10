import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import WindowDataset
from .model import UltrasoundSplinePolicy
from .upstream import revision


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def on_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def load_policy(checkpoint, device="cpu", repo=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    config = Config(**payload["config"])
    policy = UltrasoundSplinePolicy(config, len(payload["state_fields"]), repo)
    policy.load_state_dict(payload["model"])
    return policy.to(device).eval(), payload


def train(dataset, output_dir, device="cpu", repo=None, epochs=None, batch_size=None):
    training, validation = (
        WindowDataset(dataset, "train"),
        WindowDataset(dataset, "validation"),
    )
    values = training.config.to_dict()
    if epochs is not None:
        values["epochs"] = epochs
    if batch_size is not None:
        values["batch_size"] = batch_size
    config = Config(**values)
    seed_everything(config.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    policy = UltrasoundSplinePolicy(config, len(training.manifest["state_fields"]), repo).to(device)
    policy.set_statistics(training.statistics())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(training, batch_size=config.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(validation, batch_size=config.batch_size)
    provenance = revision(repo)
    (output / "dataset_manifest.json").write_text(json.dumps(training.manifest, indent=2) + "\n")
    best = float("inf")
    for epoch in range(config.epochs):
        policy.train()
        train_sum, train_n = 0.0, 0
        for batch in loader:
            batch = on_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = policy.compute_loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            train_sum += loss.item() * len(batch["ultrasound"])
            train_n += len(batch["ultrasound"])
        policy.eval()
        val_sum, val_n = 0.0, 0
        # Reuse a fixed validation noise/timestep stream without perturbing training.
        cuda_devices = (
            [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
        )
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.manual_seed(config.seed + 1)
            for batch in val_loader:
                batch = on_device(batch, device)
                loss = policy.compute_loss(batch)
                val_sum += loss.item() * len(batch["ultrasound"])
                val_n += len(batch["ultrasound"])
        metrics = {
            "epoch": epoch + 1,
            "train_noise_mse": train_sum / train_n,
            "validation_noise_mse": val_sum / val_n,
        }
        if not np.isfinite(list(metrics.values())).all():
            raise RuntimeError("Nonfinite validation metrics")
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        checkpoint = {
            "schema_version": 1,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "state_fields": training.manifest["state_fields"],
            "upstream": provenance,
            "epoch": epoch + 1,
            "metrics": metrics,
        }
        torch.save(checkpoint, output / "last.pt")
        if metrics["validation_noise_mse"] < best:
            best = metrics["validation_noise_mse"]
            torch.save(checkpoint, output / "best.pt")
        print(json.dumps(metrics), flush=True)
    return output / "best.pt"


@torch.no_grad()
def evaluate(checkpoint, dataset, split="test", device="cpu", repo=None):
    policy, payload = load_policy(checkpoint, device, repo)
    data = WindowDataset(dataset, split)
    for key in (
        "history",
        "image_size",
        "num_segments",
        "sample_hz",
        "prediction_seconds",
    ):
        if getattr(data.config, key) != getattr(policy.config, key):
            raise ValueError(f"Dataset/checkpoint mismatch: {key}")
    if data.manifest["state_fields"] != payload["state_fields"]:
        raise ValueError("Dataset/checkpoint state_fields mismatch")
    generator = torch.Generator(device=device).manual_seed(policy.config.seed)
    distance_sum, squared_sum, n, fit_squared, coordinate_count = 0.0, 0.0, 0, 0.0, 0
    for batch in DataLoader(data, batch_size=policy.config.batch_size):
        batch = on_device(batch, device)
        prediction = policy.predict(batch["ultrasound"], batch["robot_state"], generator)[
            "trajectory"
        ]
        delta = prediction - batch["trajectory"]
        distance_sum += float(delta.norm(dim=-1).sum())
        squared_sum += float(delta.square().sum())
        n += delta.shape[0] * delta.shape[1]
        fit_squared += float(
            (policy.codec.decode(batch["spline_params"]) - batch["trajectory"]).square().sum()
        )
        coordinate_count += delta.numel()
    return {
        "split": split,
        "samples": len(data),
        "mean_position_error_m": distance_sum / n,
        "position_rmse_m": (squared_sum / n) ** 0.5,
        "spline_fit_coordinate_rmse_m": (fit_squared / coordinate_count) ** 0.5,
    }
