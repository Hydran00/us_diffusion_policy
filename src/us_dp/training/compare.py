"""Evaluate an image-conditioned and a pose-only checkpoint on the same test set.

Both checkpoints must come from training runs on the same prepared dataset, so
the "test" (or "validation") split -- fixed by group hashing at `prepare` time --
is identical for both. This isolates the effect of `--no-image-conditioning`
from any difference in data.
"""
import json

from tqdm.auto import tqdm

from us_dp.training.train import evaluate, load_policy

# All current evaluate() metrics are errors: smaller is a better prediction.
LOWER_IS_BETTER = (
    "noise_mse",
    "spline_parameter_mse_m2",
    "trajectory_mse_m2",
    "mean_position_error_m",
    "spline_fit_mse_m2",
    "position_rmse_m",
    "spline_fit_coordinate_rmse_m",
)
PRIMARY_METRIC = "mean_position_error_m"


def compare(checkpoint_a, checkpoint_b, dataset, split="test", device="cpu", repo=None):
    policy_a, _ = load_policy(checkpoint_a, device, repo)
    policy_b, _ = load_policy(checkpoint_b, device, repo)
    image_a = policy_a.config.use_image_conditioning
    image_b = policy_b.config.use_image_conditioning
    if image_a == image_b:
        raise ValueError(
            "Both checkpoints have use_image_conditioning="
            f"{image_a}; pass one image-conditioned and one pose-only checkpoint"
        )
    image_checkpoint, no_image_checkpoint = (
        (checkpoint_a, checkpoint_b) if image_a else (checkpoint_b, checkpoint_a)
    )

    tqdm.write(f"[compare] with image:    {image_checkpoint}")
    image_metrics = evaluate(image_checkpoint, dataset, split, device, repo)
    tqdm.write(f"[compare] without image: {no_image_checkpoint}")
    no_image_metrics = evaluate(no_image_checkpoint, dataset, split, device, repo)
    if image_metrics["samples"] != no_image_metrics["samples"]:
        raise ValueError(
            "Checkpoints evaluated a different number of samples "
            f"({image_metrics['samples']} vs {no_image_metrics['samples']}); "
            "confirm both were trained on this same prepared dataset"
        )

    metrics = []
    for key in LOWER_IS_BETTER:
        with_image, without_image = image_metrics[key], no_image_metrics[key]
        # Positive => the image-conditioned model has lower error.
        error_reduction_percent = (
            (without_image - with_image) / without_image * 100 if without_image else float("nan")
        )
        metrics.append({
            "metric": key,
            "with_image": with_image,
            "without_image": without_image,
            "error_reduction_percent_with_image": error_reduction_percent,
            "image_wins": with_image < without_image,
        })

    primary = next(m for m in metrics if m["metric"] == PRIMARY_METRIC)
    verdict = (
        f"Image conditioning {'HELPS' if primary['image_wins'] else 'DOES NOT HELP'} "
        f"({PRIMARY_METRIC}: with_image={primary['with_image']:.6f} m, "
        f"without_image={primary['without_image']:.6f} m, "
        f"{primary['error_reduction_percent_with_image']:+.1f}% error change from using the image)"
    )

    report = {
        "dataset": str(dataset),
        "split": split,
        "samples": image_metrics["samples"],
        "image_checkpoint": str(image_checkpoint),
        "no_image_checkpoint": str(no_image_checkpoint),
        "metrics": metrics,
        "verdict": verdict,
    }
    tqdm.write(json.dumps(report, indent=2))
    return report
