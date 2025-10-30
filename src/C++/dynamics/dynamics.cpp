#include "dynamics.hpp"

namespace naturalis {

OrbitalState twoBodyDynamics(double mu, const OrbitalState& state) {
  const double rnorm = state.r.norm();
  const Eigen::Vector3d a = (-mu / (rnorm * rnorm * rnorm)) * state.r;
  return OrbitalState{state.v, a};
}

OrbitalState lunarPerturbation(const OrbitalState& state) {
  return OrbitalState{state.v, Eigen::Vector3d::Zero()};
}

OrbitalState solarPerturbation(const OrbitalState& state) {
  return OrbitalState{state.v, Eigen::Vector3d::Zero()};
}

OrbitalState j2Perturbation(double /*mu*/, const OrbitalState& state) {
  return OrbitalState{state.v, Eigen::Vector3d::Zero()};
}

OrbitalState atmosphericPerturbation(const OrbitalState& state) {
  return OrbitalState{state.v, Eigen::Vector3d::Zero()};
}

OrbitalState solarRadiationPerturbation(const OrbitalState& state) {
  return OrbitalState{state.v, Eigen::Vector3d::Zero()};
}

} // namespace naturalis


