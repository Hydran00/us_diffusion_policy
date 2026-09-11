"""Compatibility alias for :mod:`us_dp.anatomy_processing.frames`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("us_dp.anatomy_processing.frames")
