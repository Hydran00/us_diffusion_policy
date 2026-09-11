"""Compatibility alias for :mod:`us_dp.training.train`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("us_dp.training.train")
