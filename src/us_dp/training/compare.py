"""Evaluate an image-conditioned and a pose-only checkpoint on the same test set.

Both checkpoints must come from training runs on the same prepared dataset, so
the "test" (or "validation") split -- fixed by group hashing at `prepare` time --
is identical for both. This isolates the effect of `--no-image-conditioning`
from any difference in data.
"""
import json
from pathlib import Path

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


def markdown_report(report):
    seconds = report["prediction_window_seconds"]
    lines = [
        "# Image conditioning comparison",
        "",
        f"- Dataset: `{report['dataset']}`" if report.get("dataset") else "- Offline evaluation: skipped",
        f"- Split: `{report['split']}`" if report.get("dataset") else "",
        f"- Prediction window: {seconds:g} second{'s' if seconds != 1 else ''} "
        f"({report['prediction_intervals']} intervals, {report['prediction_positions']} positions)",
        f"- With image: `{report['image_checkpoint']}`",
        f"- Without image: `{report['no_image_checkpoint']}`",
    ]
    if "metrics" in report:
        lines.extend((
            f"- Evaluation windows: {report['samples']}",
            "",
            "| Metric | With image | Without image | Error reduction with image | Image wins |",
            "| --- | ---: | ---: | ---: | :---: |",
        ))
        for metric in report["metrics"]:
            outcome = "Tie" if metric["with_image"] == metric["without_image"] else "Yes" if metric["image_wins"] else "No"
            lines.append(
                f"| `{metric['metric']}` | {metric['with_image']:.6g} | "
                f"{metric['without_image']:.6g} | "
                f"{metric['error_reduction_percent_with_image']:+.1f}% | "
                f"{outcome} |"
            )
        lines.extend(("", f"**Verdict:** {report['verdict']}", ""))
    if "simulator" in report:
        sim = report["simulator"]
        with_image, without_image = sim["with_image"], sim["without_image"]
        lines.extend((
            "",
            "## Full simulator evaluation",
            "",
            f"{sim['requested_episodes_per_checkpoint']} complete episodes per checkpoint; seed {sim['seed']}.",
            "",
            "| Metric | With image | Without image |",
            "| --- | ---: | ---: |",
            f"| Successful episodes | {with_image['successes']} | {without_image['successes']} |",
            f"| Success rate | {with_image['success_rate_percent']:.1f}% | {without_image['success_rate_percent']:.1f}% |",
            f"| Mean episode duration (s) | {with_image['mean_duration_s']:.3f} | {without_image['mean_duration_s']:.3f} |",
            f"| Mean TCP path length (m) | {with_image['mean_tcp_path_length_m']:.6f} | {without_image['mean_tcp_path_length_m']:.6f} |",
            f"| Mean commanded-to-measured TCP error (m) | {with_image['mean_tracking_error_m']:.6f} | {without_image['mean_tracking_error_m']:.6f} |",
            "",
            f"- With-image recording: `{with_image['recording']}`",
            f"- Without-image recording: `{without_image['recording']}`",
            f"- Mean initial TCP gap between matching episode indices: {sim['mean_initial_tcp_gap_m']:.6f} m",
            "",
        ))
    return "\n".join(lines)


def compare(checkpoint_a, checkpoint_b, dataset, split="test", device="cpu", repo=None,
            output="compare_image_conditioning.md", sim_episodes=0, sim_seed=42, skip_offline=False):
    policy_a, _ = load_policy(checkpoint_a, device, repo)
    policy_b, _ = load_policy(checkpoint_b, device, repo)
    image_a = policy_a.config.use_image_conditioning
    image_b = policy_b.config.use_image_conditioning
    if image_a == image_b:
        raise ValueError(
            "Both checkpoints have use_image_conditioning="
            f"{image_a}; pass one image-conditioned and one pose-only checkpoint"
        )
    if policy_a.config.prediction_seconds != policy_b.config.prediction_seconds:
        raise ValueError("Checkpoints have different prediction windows")
    image_checkpoint, no_image_checkpoint = (
        (checkpoint_a, checkpoint_b) if image_a else (checkpoint_b, checkpoint_a)
    )
    if skip_offline:
        if sim_episodes < 1:
            raise ValueError("--skip-offline requires --sim-episodes >= 1")
        from us_dp.training.simulator_compare import compare_simulator

        report = {
            "dataset": None,
            "prediction_window_seconds": policy_a.config.prediction_seconds,
            "prediction_intervals": policy_a.config.future_steps,
            "prediction_positions": policy_a.config.future_steps + 1,
            "image_checkpoint": str(image_checkpoint),
            "no_image_checkpoint": str(no_image_checkpoint),
        }
        workflow_root = Path(__file__).resolve().parents[3] / "i4h-workflows"
        report["simulator"] = compare_simulator(
            image_checkpoint, no_image_checkpoint, sim_episodes, sim_seed, device, workflow_root
        )
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown_report(report), encoding="utf-8")
        report["markdown_report"] = str(output_path)
        tqdm.write(json.dumps(report, indent=2))
        return report
    if dataset is None:
        raise ValueError("--dataset is required unless --skip-offline is set")


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
        "prediction_window_seconds": policy_a.config.prediction_seconds,
        "prediction_intervals": policy_a.config.future_steps,
        "prediction_positions": policy_a.config.future_steps + 1,
        "image_checkpoint": str(image_checkpoint),
        "no_image_checkpoint": str(no_image_checkpoint),
        "metrics": metrics,
        "verdict": verdict,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_report(report), encoding="utf-8")
    report["markdown_report"] = str(output_path)
    if sim_episodes:
        from us_dp.training.simulator_compare import compare_simulator

        workflow_root = Path(__file__).resolve().parents[3] / "i4h-workflows"
        report["simulator"] = compare_simulator(
            image_checkpoint, no_image_checkpoint, sim_episodes, sim_seed, device, workflow_root
        )
        output_path.write_text(markdown_report(report), encoding="utf-8")
    tqdm.write(json.dumps(report, indent=2))
    return report
