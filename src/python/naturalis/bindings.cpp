#include <../c++/Dynamics/Orbit/Orbit.hpp>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(naturalis, m) {
    

    py::class_<OrbitalState>(m, "OrbitalState")
        .def_readwrite("time", &OrbitalState::time)
        .def_readwrite("position", &OrbitalState::position)
        .def_readwrite("velocity", &OrbitalState::velocity);
}