"""
This file makes the 'runners' directory a Python package and exposes
the main functions from its submodules for easier access.
"""
from .cbir_region import main as cbir_region
from .cbir_whole_brain import main as cbir_whole_brain