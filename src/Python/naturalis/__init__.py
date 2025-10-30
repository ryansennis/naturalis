"""
Naturalis - High-performance orbital simulation and GNC framework
"""

# Try to import C++ backend via Cython
try:
    from ._cnaturalis import hello_cpp, PyPropagator as _PyPropagator, PyOrbitalState as _PyOrbitalState
    _HAS_C_BACKEND = True
except ImportError:
    _HAS_C_BACKEND = False
    def hello_cpp():
        return "C++ backend not available"

# Export public API
from .propagator import Propagator, PropagatorType
from .dynamics import OrbitalState, ForceModel
from .plotting import (
    plot_trajectory_3d,
    plot_trajectory_2d,
    plot_trajectory_all_planes
)

__all__ = [
    'Propagator',
    'PropagatorType',
    'OrbitalState',
    'ForceModel',
    'plot_trajectory_3d',
    'plot_trajectory_2d',
    'plot_trajectory_all_planes',
    'hello_cpp',
]