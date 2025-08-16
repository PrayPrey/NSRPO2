"""
NSPO2 (Null Space Preference Optimization v2) Module

A production-ready module for variance reduction in policy gradient and preference optimization
through null space projection with bias management.
"""

from .config import NSPO2Config, get_default_config
from .core import NSPO2
from .optimizer_wrapper import NSPO2OptimizerWrapper, create_nspo2_optimizer

__version__ = "0.1.0"
__all__ = ["NSPO2", "NSPO2Config", "get_default_config", "NSPO2OptimizerWrapper", "create_nspo2_optimizer"]