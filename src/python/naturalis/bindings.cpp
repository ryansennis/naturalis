#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(_naturalis, m) {
    m.doc() = "Naturalis";
}
