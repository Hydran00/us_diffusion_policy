"""Legacy imports must resolve to the canonical module, including monkeypatches."""

import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    'legacy,canonical',
    [
        ('anatomy', 'anatomy_processing.frames'),
        ('anatomy_assets', 'anatomy_processing.assets'),
        ('anatomy_viewer', 'anatomy_processing.viewer'),
        ('collection', 'dataset_generation.collection'),
        ('oracle', 'dataset_generation.oracle'),
        ('reach_demo', 'dataset_generation.reach_demo'),
        ('demo', 'dataset_generation.synthetic'),
        ('data', 'dataset.processing'),
        ('convert', 'dataset.convert'),
        ('train', 'training.train'),
        ('model', 'training.model'),
        ('inference', 'deployment.inference'),
        ('spline', 'common.spline'),
        ('geometry', 'common.geometry'),
        ('upstream', 'common.upstream'),
    ],
)
def test_legacy_import_is_same_module(legacy, canonical):
    assert importlib.import_module(f'us_dp.{legacy}') is importlib.import_module(
        f'us_dp.{canonical}'
    )


def test_default_upstream_checkout(monkeypatch):
    from us_dp.common.upstream import use_spline_policy

    monkeypatch.delenv('SPLINE_POLICY_ROOT', raising=False)
    assert use_spline_policy() == Path(__file__).resolve().parents[2] / 'spline_policy'
