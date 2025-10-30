#include "integrators.hpp"

namespace naturalis {

OrbitalState rk4Step(const DynamicsFn& f, const OrbitalState& x) {
  const OrbitalState k1 = f(x);
  const OrbitalState k2 = f(x + 0.5 * k1);
  const OrbitalState k3 = f(x + 0.5 * k2);
  const OrbitalState k4 = f(x + k3);
  return (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (1.0 / 6.0);
}

std::vector<OrbitalState> rk4(const std::vector<DynamicsFn>& dynamics,
                              const OrbitalState& initialState,
                              double timeStep,
                              double endTime) {
  std::vector<OrbitalState> states;
  OrbitalState state = initialState;
  for (double t = 0.0; t < endTime; t += timeStep) {
    states.push_back(state);
    OrbitalState delta{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    for (const auto& model : dynamics) {
      delta = delta + rk4Step(model, state) * timeStep;
    }
    state = state + delta;
  }
  return states;
}

OrbitalState dp45Step(const DynamicsFn& f, const OrbitalState& x) {
  const OrbitalState k1 = f(x);
  const OrbitalState k2 = f(x + (1.0/5.0) * k1);
  const OrbitalState k3 = f(x + (3.0/40.0) * k1 + (9.0/40.0) * k2);
  const OrbitalState k4 = f(x + (44.0/45.0) * k1 - (56.0/15.0) * k2 + (32.0/9.0) * k3);
  const OrbitalState k5 = f(x + (19372.0/6561.0) * k1 - (25360.0/2187.0) * k2 + (64448.0/6561.0) * k3 - (212.0/729.0) * k4);
  const OrbitalState k6 = f(x + (9017.0/3168.0) * k1 - (355.0/33.0) * k2 + (46732.0/5247.0) * k3 + (49.0/176.0) * k4 - (5103.0/18656.0) * k5);
  return (35.0/384.0) * k1 + (500.0/1113.0) * k3 + (125.0/192.0) * k4 - (2187.0/6784.0) * k5 + (11.0/84.0) * k6;
}

std::vector<OrbitalState> dp45(const DynamicsFn& dynamics,
                               const OrbitalState& initialState,
                               double timeStep,
                               double endTime) {
  std::vector<OrbitalState> states;
  OrbitalState state = initialState;
  for (double t = 0.0; t < endTime; t += timeStep) {
    states.push_back(state);
    state = state + dp45Step(dynamics, state) * timeStep;
  }
  return states;
}

} // namespace naturalis


