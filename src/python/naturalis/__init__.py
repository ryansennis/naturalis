"""
Naturalis - A high-performance Python framework for orbital simulation and GNC.
"""

__version__ = "0.1.0"
__author__ = "Ryan Ennis"
__email__ = "ryansennis@hotmail.com"

def __getattr__(name):
    if name == "_naturalis":
        from . import _naturalis
        return _naturalis
    elif name in ["dynamics", "mathematics", "solvers", "constants", "plotting"]:
        return __import__(f"naturalis.{name}", fromlist=[name])
    raise AttributeError(f"module 'naturalis' has no attribute '{name}'")

__all__ = []