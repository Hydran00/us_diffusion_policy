"""Does an image-conditioned checkpoint actually use the ultrasound image?

Feeds the same trained policy real frames and then substitutes noise/black
frames for them, keeping the diffusion sampling noise identical between the
two runs (same per-batch generator seed) so any change in the predicted
trajectory is attributable only to the swapped image. If the policy ignores
the image, real and substituted predictions coincide even though they are
far from identical, and both track the ground truth about as well.
"""
import json

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from us_dp.dataset.processing import WindowDataset
from us_dp.training.train import load_policy, on_device

SUBSTITUTES = ("noise", "black")


@torch.no_grad()
def image_sensitivity(checkpoint, dataset, split="test", device="cpu", repo=None):
    policy, payload = load_policy(checkpoint, device, repo)
    if policy.image_encoder is None:
        raise ValueError(
            "This checkpoint has use_image_conditioning=False; there is no image "
            "pathway to test. Pass an image-conditioned checkpoint instead."
        )
    data = WindowDataset(dataset, split)
    for key in ("history", "image_size", "num_segments", "sample_hz", "prediction_seconds"):
        if getattr(data.config, key) != getattr(policy.config, key):
            raise ValueError(f"Dataset/checkpoint mismatch: {key}")
    if data.manifest["state_fields"] != payload["state_fields"]:
        raise ValueError("Dataset/checkpoint state_fields mismatch")

    totals = {name: {"error_sum": 0.0} for name in ("real",) + SUBSTITUTES}
    shift_sum = {name: 0.0 for name in SUBSTITUTES}
    shift_max = {name: 0.0 for name in SUBSTITUTES}
    n = 0
    progress = tqdm(
        DataLoader(data, batch_size=policy.config.batch_size), desc=f"image-sensitivity [{split}]"
    )
    for step, batch in enumerate(progress):
        batch = on_device(batch, device)
        size = len(batch["ultrasound"])
        n += size
        seed = policy.config.seed + step
        real = policy.predict(
            batch["ultrasound"], batch["robot_state"], torch.Generator(device=device).manual_seed(seed)
        )["trajectory"]
        totals["real"]["error_sum"] += float(
            (real - batch["trajectory"]).norm(dim=-1).sum()
        )
        for name in SUBSTITUTES:
            substituted = (
                torch.rand_like(batch["ultrasound"]) if name == "noise" else torch.zeros_like(batch["ultrasound"])
            )
            prediction = policy.predict(
                substituted, batch["robot_state"], torch.Generator(device=device).manual_seed(seed)
            )["trajectory"]
            totals[name]["error_sum"] += float((prediction - batch["trajectory"]).norm(dim=-1).sum())
            per_point = (prediction - real).norm(dim=-1)
            shift_sum[name] += float(per_point.sum())
            shift_max[name] = max(shift_max[name], float(per_point.max()))

    report = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "split": split,
        "samples": n,
    }
    for name in ("real",) + SUBSTITUTES:
        report[f"mean_position_error_m_{name}_image"] = totals[name]["error_sum"] / (n * (policy.config.future_steps + 1))
    for name in SUBSTITUTES:
        points = n * (policy.config.future_steps + 1)
        report[f"prediction_shift_mean_m_real_vs_{name}"] = shift_sum[name] / points
        report[f"prediction_shift_max_m_real_vs_{name}"] = shift_max[name]

    real_error = report["mean_position_error_m_real_image"]
    ignores_image = all(
        report[f"prediction_shift_mean_m_real_vs_{name}"] < 1e-3
        and abs(report[f"mean_position_error_m_{name}_image"] - real_error) < 0.1 * real_error
        for name in SUBSTITUTES
    )
    report["verdict"] = (
        "The trained policy's predictions barely change when the image is replaced by noise/black "
        "frames, and accuracy is unaffected: this checkpoint learned to ignore the image."
        if ignores_image else
        "The trained policy's predictions or accuracy change meaningfully when the image is "
        "replaced by noise/black frames: this checkpoint does use the image."
    )
    tqdm.write(json.dumps(report, indent=2))
    return report
