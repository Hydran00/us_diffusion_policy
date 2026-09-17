from types import SimpleNamespace

import pytest

from us_dp.training import compare as comparison
from us_dp.training import simulator_compare


def test_skip_offline_uses_only_simulator(monkeypatch, tmp_path):
    def policy(image):
        return SimpleNamespace(config=SimpleNamespace(
            use_image_conditioning=image, prediction_seconds=1.0, future_steps=50
        ))

    monkeypatch.setattr(comparison, "load_policy",
                        lambda checkpoint, device, repo: (policy(checkpoint == "image.pt"), None))
    monkeypatch.setattr(comparison, "evaluate",
                        lambda *args: pytest.fail("offline evaluation must be skipped"))
    calls = []

    def fake_simulator(*args):
        calls.append(args)
        summary = {
            "successes": 1, "success_rate_percent": 100.0,
            "mean_duration_s": 4.0, "mean_tcp_path_length_m": 0.1,
            "mean_tracking_error_m": 0.002, "recording": "rollout.hdf5",
        }
        return {
            "requested_episodes_per_checkpoint": 1, "seed": 42,
            "with_image": summary, "without_image": summary,
            "mean_initial_tcp_gap_m": 0.0,
        }

    monkeypatch.setattr(simulator_compare, "compare_simulator", fake_simulator)
    output = tmp_path / "comparison.md"
    report = comparison.compare("image.pt", "pose.pt", None, output=output,
                                sim_episodes=1, skip_offline=True)
    assert len(calls) == 1
    assert "metrics" not in report
    text = output.read_text()
    assert "Offline evaluation: skipped" in text
    assert "Full simulator evaluation" in text
    assert "| Metric | With image | Without image | Error reduction" not in text


def test_skip_offline_requires_simulator_episodes(monkeypatch, tmp_path):
    monkeypatch.setattr(comparison, "load_policy",
                        lambda checkpoint, device, repo: (
                            SimpleNamespace(config=SimpleNamespace(
                                use_image_conditioning=checkpoint == "image.pt",
                                prediction_seconds=1.0, future_steps=50,
                            )), None
                        ))
    with pytest.raises(ValueError, match="--sim-episodes"):
        comparison.compare("image.pt", "pose.pt", None, output=tmp_path / "report.md",
                           skip_offline=True)
