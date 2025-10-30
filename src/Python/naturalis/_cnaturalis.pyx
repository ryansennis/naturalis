# distutils: language=c++
# cython: language_level=3

import numpy as np
cimport numpy as cnp

from libcpp.vector cimport vector

# Forward declare Eigen Vector3d - we'll access it through the struct
cdef extern from "Eigen/Dense" namespace "Eigen":
    cdef cppclass Vector3d:
        Vector3d()
        Vector3d(double, double, double)
        double& operator[](int)

cdef extern from "orbit.hpp" namespace "naturalis":
    cdef cppclass OrbitalState:
        OrbitalState()
        Vector3d r
        Vector3d v
        OrbitalState operator+(OrbitalState)
        OrbitalState operator*(double)

cdef extern from "dynamics.hpp" namespace "naturalis":
    cdef enum ForceModel:
        TWO_BODY
        LUNAR
        SOLAR
        J2
        ATMOSPHERIC
        SOLAR_RADIATION

cdef extern from "propagator.hpp" namespace "naturalis":
    cdef enum PropagatorType:
        RK4
        DP45

    cdef cppclass Propagator:
        Propagator(PropagatorType type, double mu)
        Propagator(PropagatorType type, double mu, vector[ForceModel])
        vector[OrbitalState] propagate(OrbitalState const& initial, double timeStep, double endTime) const

# Conversion helpers - manually construct OrbitalState with Eigen Vector3d
cdef OrbitalState numpy_to_orbital_state(cnp.ndarray[double, ndim=1] r, cnp.ndarray[double, ndim=1] v):
    """Convert numpy arrays to C++ OrbitalState"""
    cdef OrbitalState state
    state.r = Vector3d(r[0], r[1], r[2])
    state.v = Vector3d(v[0], v[1], v[2])
    return state

cdef tuple orbital_state_to_numpy(const OrbitalState& state):
    """Convert C++ OrbitalState to numpy arrays"""
    cdef double[3] r_arr = [state.r[0], state.r[1], state.r[2]]
    cdef double[3] v_arr = [state.v[0], state.v[1], state.v[2]]
    return (np.array([r_arr[0], r_arr[1], r_arr[2]]), 
            np.array([v_arr[0], v_arr[1], v_arr[2]]))

cdef class PyOrbitalState:
    """Python wrapper for OrbitalState"""
    cdef OrbitalState _state

    def __cinit__(self, cnp.ndarray r, cnp.ndarray v):
        r_array = np.ascontiguousarray(r, dtype=np.float64)
        v_array = np.ascontiguousarray(v, dtype=np.float64)
        if len(r_array) != 3 or len(v_array) != 3:
            raise ValueError("Position and velocity must be 3D vectors")
        self._state = numpy_to_orbital_state(r_array, v_array)

    @property
    def r(self):
        """Get position vector as numpy array"""
        return orbital_state_to_numpy(self._state)[0]

    @property
    def v(self):
        """Get velocity vector as numpy array"""
        return orbital_state_to_numpy(self._state)[1]

cdef class PyPropagator:
    """Python wrapper for Propagator"""
    cdef Propagator* _p

    def __cinit__(self, int ptype, double mu):
        self._p = new Propagator(<PropagatorType>ptype, mu)

    def __dealloc__(self):
        if self._p is not None:
            del self._p
            self._p = NULL

    def propagate(self, PyOrbitalState init, double time_step, double end_time):
        """Propagate orbital state and return trajectory as list of (r, v) tuples"""
        cdef vector[OrbitalState] trajectory = self._p.propagate(init._state, time_step, end_time)
        cdef list result = []
        cdef size_t i
        for i in range(trajectory.size()):
            r, v = orbital_state_to_numpy(trajectory[i])
            result.append((r, v))
        return result

# Simple hello function (no longer needs C++ implementation)
def hello_cpp() -> str:
    return "Hello from Naturalis C++"