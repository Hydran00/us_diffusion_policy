"""Compatibility alias for :mod:`us_dp.common.spline`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("us_dp.common.spline")
