import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from us_dp.common.upstream import revision
from us_dp.config import Config
from us_dp.common.state import validate_state_fields
from us_dp.dataset.processing import WindowDataset
from us_dp.training.model import UltrasoundSplinePolicy


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
    validate_state_fields(payload["state_fields"])
    config = Config(**payload["config"])
    if "sampling_clip_range" not in payload["config"]:
        logging.getLogger(__name__).warning(
            "Legacy checkpoint: applying sampling_clip_range=%s to normalized x0; "
            "historical unbounded sampling metrics are not directly comparable.",
            config.sampling_clip_range,
        )
    policy = UltrasoundSplinePolicy(config, len(payload["state_fields"]), repo, load_pretrained=False)
    policy.load_state_dict(payload["model"])
    return policy.to(device).eval(), payload


@torch.no_grad()
def batch_metrics(policy, batch, generator=None):
    """Four distinct errors; physical parameter/trajectory metrics use metres."""
    prediction = policy.predict(batch["ultrasound"], batch["robot_state"], generator)
    delta = prediction["trajectory"] - batch["trajectory"]
    fit_delta = policy.codec.decode(batch["spline_params"]) - batch["trajectory"]
    param_delta = prediction["spline_params"][:, 1:] - batch["spline_params"][:, 1:]
    return {
        "noise_mse": float(policy.compute_loss(batch)),
        "spline_parameter_mse_m2": float(param_delta.square().mean()),
        "trajectory_mse_m2": float(delta.square().sum(-1).mean()),
        "mean_position_error_m": float(delta.norm(dim=-1).mean()),
        "spline_fit_mse_m2": float(fit_delta.square().sum(-1).mean()),
    }


def train(dataset, output_dir, device="cpu", repo=None, epochs=None, batch_size=None,
          *, use_image_conditioning=None):
    from torch.utils.tensorboard import SummaryWriter

    training, validation = (
        WindowDataset(dataset, "train"),
        WindowDataset(dataset, "validation"),
    )
    values = training.config.to_dict()
    if epochs is not None:
        values["epochs"] = epochs
    if batch_size is not None:
        values["batch_size"] = batch_size
    if use_image_conditioning is not None:
        values["use_image_conditioning"] = use_image_conditioning
    config = Config(**values)
    if config.use_image_conditioning and config.usfm_pretrained is None and any(e["source"] != "synthetic_smoke" for e in training.entries):
        raise ValueError("Set usfm_pretrained to the USFM checkpoint before training on demonstrations")
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
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
    tqdm.write(
        f"[train] device={device} epochs={config.epochs} batch_size={config.batch_size} "
        f"train_episodes={len(training.entries)} validation_episodes={len(validation.entries)} "
        f"train_windows={len(training)} val_windows={len(validation)} "
        f"params_trainable={trainable:,} params_frozen={frozen:,} "
        f"image_conditioning={config.use_image_conditioning}"
    )
    tqdm.write(f"[train] TensorBoard logs: {output / 'tensorboard'}")
    best = float("inf")
    global_step = 0
    with SummaryWriter(log_dir=str(output / "tensorboard"), flush_secs=10) as writer:
        writer.add_text("config", "```json\n" + json.dumps(config.to_dict(), indent=2) + "\n```")
        for epoch in range(config.epochs):
            epoch_start = time.monotonic()
            policy.train()
            train_sum, train_n = 0.0, 0
            progress = tqdm(loader, desc=f"epoch {epoch + 1}/{config.epochs} [train]", leave=False)
            for step, batch in enumerate(progress, start=1):
                batch = on_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = policy.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                global_step += 1
                writer.add_scalar("step/train_noise_mse", loss.item(), global_step)
                writer.add_scalar("step/grad_norm_before_clip", float(grad_norm), global_step)
                writer.add_scalar("step/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                train_sum += loss.item() * len(batch["ultrasound"])
                train_n += len(batch["ultrasound"])
                progress.set_postfix(
                    iter=f"{step}/{len(loader)}",
                    loss=f"{loss.item():.4f}",
                    mse=f"{train_sum / train_n:.4f}",
                    grad_norm=f"{float(grad_norm):.3f}",
                )
            policy.eval()
            val_sum, val_n = 0.0, 0
            val_metrics = {}
            # Reuse a fixed validation noise/timestep stream without perturbing training.
            cuda_devices = (
                [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
            )
            with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
                torch.manual_seed(config.seed + 1)
                progress = tqdm(val_loader, desc=f"epoch {epoch + 1}/{config.epochs} [val]", leave=False)
                for batch in progress:
                    batch = on_device(batch, device)
                    measured = batch_metrics(policy, batch)
                    for key, value in measured.items():
                        val_metrics[key] = val_metrics.get(key, 0.0) + value * len(batch["ultrasound"])
                    val_sum += measured["noise_mse"] * len(batch["ultrasound"])
                    val_n += len(batch["ultrasound"])
                    progress.set_postfix(mse=f"{val_sum / val_n:.4f}")
            metrics = {
                "epoch": epoch + 1,
                "train_noise_mse": train_sum / train_n,
                **{f"validation_{key}": value / val_n for key, value in val_metrics.items()},
            }
            if not np.isfinite(list(metrics.values())).all():
                raise RuntimeError("Nonfinite validation metrics")
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            writer.add_scalar("loss/train_noise_mse", metrics["train_noise_mse"], epoch + 1)
            writer.add_scalar("loss/validation_noise_mse", metrics["validation_noise_mse"], epoch + 1)
            for key in val_metrics:
                if key != "noise_mse":
                    writer.add_scalar(f"validation/{key}", metrics[f"validation_{key}"], epoch + 1)
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
            improved = metrics["validation_noise_mse"] < best
            if improved:
                best = metrics["validation_noise_mse"]
                torch.save(checkpoint, output / "best.pt")
            tqdm.write(
                f"epoch {epoch + 1}/{config.epochs} "
                f"train_mse={metrics['train_noise_mse']:.4f} "
                f"val_mse={metrics['validation_noise_mse']:.4f} "
                f"best_val_mse={best:.4f}{' (new best)' if improved else ''} "
                f"elapsed={time.monotonic() - epoch_start:.1f}s"
            )
            writer.add_scalar("loss/best_validation_noise_mse", best, epoch + 1)
            writer.add_scalar("timing/epoch_seconds", time.monotonic() - epoch_start, epoch + 1)
            writer.flush()
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
    totals, n = {}, 0
    progress = tqdm(DataLoader(data, batch_size=policy.config.batch_size), desc=f"evaluate [{split}]")
    for batch in progress:
        batch = on_device(batch, device)
        values = batch_metrics(policy, batch, generator)
        size = len(batch["ultrasound"])
        n += size
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value * size
    metrics = {"split": split, "samples": n, **{key: value / n for key, value in totals.items()}}
    metrics["sampling_clip_range"] = policy.config.sampling_clip_range
    metrics["position_rmse_m"] = metrics["trajectory_mse_m2"] ** 0.5
    metrics["spline_fit_coordinate_rmse_m"] = (metrics["spline_fit_mse_m2"] / 3) ** 0.5
    tqdm.write(json.dumps(metrics, indent=2))
    return metrics
