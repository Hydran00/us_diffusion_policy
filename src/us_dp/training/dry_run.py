"""One synthetic batch through the actual encoders, DDPM loss and inference.

Run: python -m us_dp.training.dry_run --pretrained ../USFM/USFM_latest.pth
These random observations exercise software, not scan quality.
"""
import argparse
import json

import torch

from us_dp.config import Config
from us_dp.training.model import UltrasoundSplinePolicy
from us_dp.training.train import batch_metrics


def dry_run(config, device="cpu", batch_size=2):
    torch.manual_seed(config.seed)
    policy = UltrasoundSplinePolicy(config, 23).to(device)
    policy.set_statistics({
        "robot_state": (torch.zeros(23), torch.ones(23)),
        "spline_params": (torch.tensor([0.01, -0.02, 0.03]), torch.tensor([0.02, 0.03, 0.04])),
    })
    free = torch.randn(batch_size, 15, device=device) * 0.01
    params = policy.codec.unpack(free)
    times = torch.linspace(0, config.prediction_seconds, config.future_steps + 1, device=device)
    batch = {
        "ultrasound": torch.rand(batch_size, config.history, 1, config.image_size, config.image_size, device=device),
        "robot_state": torch.randn(batch_size, config.history, 23, device=device),
        "spline_params": params,
        "trajectory": policy.codec(free, times, config.prediction_seconds),
    }
    shapes = {}
    handles = [module.register_forward_hook(
        lambda _module, _inputs, output, name=name: shapes.update({name: list(output.shape)})
    ) for name, module in (
        ("visual_features", policy.image_encoder), ("state_features", policy.state_encoder)
    )]
    policy.train()
    clean, noise, noisy, predicted, cond = policy.diffusion_batch(batch)
    loss = (predicted - noise).square().mean()
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=config.learning_rate)
    loss.backward()
    optimizer.step()
    policy.eval()
    result = policy.predict(batch["ultrasound"], batch["robot_state"])
    for name, value in {
        "images": batch["ultrasound"], "robot_state": batch["robot_state"],
        "global_cond": cond, "clean_spline_params": clean, "noisy_spline_params": noisy,
        "predicted_noise": predicted, "flat_free_params": result["spline_params"][:, 1:].flatten(1),
        "decoded_xyz": result["trajectory"],
    }.items():
        shapes[name] = list(value.shape)
    for handle in handles:
        handle.remove()
    report = {"synthetic": True, "pretrained": config.usfm_pretrained, "shapes": shapes,
              "training_noise_mse": float(loss.detach()), "metrics": batch_metrics(policy, batch),
              "prediction_seconds": config.prediction_seconds, "execution_seconds": config.execution_seconds}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pretrained")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    dry_run(Config(usfm_pretrained=args.pretrained), args.device)
