from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "naturalis/orbit.hpp" namespace "naturalis":
    cdef cppclass OrbitalState:
        OrbitalState()
        # members are Eigen types; we will expose via helper functions in pyx

cdef extern from "naturalis/propagator.hpp" namespace "naturalis":
    cdef enum PropagatorType:
        RK4
        DP45

    cdef cppclass Propagator:
        Propagator(PropagatorType type, double mu)
        vector[OrbitalState] propagate(OrbitalState const& initial, double timeStep, double endTime) const

cdef extern from "naturalis/dynamics.hpp" namespace "naturalis":
    cdef enum ForceModel:
        TWO_BODY
        LUNAR
        SOLAR
        J2
        ATMOSPHERIC
        SOLAR_RADIATION
