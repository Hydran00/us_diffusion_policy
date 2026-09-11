"""Compatibility alias for :mod:`us_dp.dataset_generation.reach_demo`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("us_dp.dataset_generation.reach_demo")
