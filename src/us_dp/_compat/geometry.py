"""Compatibility alias for :mod:`us_dp.common.geometry`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("us_dp.common.geometry")
