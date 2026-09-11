"""Compatibility alias for :mod:`us_dp.dataset_generation.collection`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("us_dp.dataset_generation.collection")
