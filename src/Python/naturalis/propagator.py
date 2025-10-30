from enum import Enum
import numpy as np
from typing import List, Tuple

# Try to import Cython bindings
try:
    from ._cnaturalis import PyPropagator, PyOrbitalState
    _HAS_C_BACKEND = True
except ImportError:
    _HAS_C_BACKEND = False
    PyPropagator = None
    PyOrbitalState = None

# Re-export OrbitalState from dynamics for compatibility
from .dynamics import OrbitalState as PyOrbitalStateCompat, ForceModel

class PropagatorType(Enum):
    RK4 = 0
    DP45 = 1

class Propagator:
    """
    Propagator class using C++ backend.
    Args:
        propagator_type: Propagator type (RK4 or DP45).
        mu: Gravitational parameter. (km^3/s^2)
    """
    def __init__(self, propagator_type: PropagatorType, mu: float):
        if not _HAS_C_BACKEND:
            raise RuntimeError("C++ backend not available. Please rebuild the Cython extension.")
        
        self.propagator_type = propagator_type
        self.mu = mu
        # Map PropagatorType enum to integer for Cython
        ptype_int = propagator_type.value
        self._cpp_propagator = PyPropagator(ptype_int, mu)

    def propagate(self, initial_state: PyOrbitalStateCompat, time_step: float, end_time: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Propagate the initial state to the end time.
        Args:
            initial_state: Initial orbital state (OrbitalState with r and v numpy arrays).
            time_step: Time step in units of [s].
            end_time: End time in units of [s].
        Returns:
            List of (r, v) tuples representing the trajectory.
        """
        if not isinstance(initial_state, PyOrbitalStateCompat):
            raise ValueError("Initial state must be an OrbitalState object.")
        if not isinstance(time_step, (int, float)) or time_step <= 0:
            raise ValueError("Time step must be a positive number.")
        if not isinstance(end_time, (int, float)) or end_time <= 0:
            raise ValueError("End time must be a positive number.")
        if end_time < time_step:
            raise ValueError("End time must be greater than or equal to time step.")
        
        # Convert Python OrbitalState to Cython PyOrbitalState
        r = np.asarray(initial_state.r, dtype=np.float64)
        v = np.asarray(initial_state.v, dtype=np.float64)
        cpp_state = PyOrbitalState(r, v)
        
        # Propagate using C++ backend
        trajectory = self._cpp_propagator.propagate(cpp_state, float(time_step), float(end_time))
        
        return trajectory