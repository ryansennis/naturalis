#include "propagator.hpp"

namespace naturalis {

Propagator::Propagator(PropagatorType type, double mu, std::vector<ForceModel> models)
  : type_(type), mu_(mu) {
  dynamics_.clear();
  for (const auto m : models) {
    switch (m) {
      case ForceModel::TWO_BODY:
        dynamics_.push_back([this](const OrbitalState& s) { return twoBodyDynamics(mu_, s); });
        break;
      case ForceModel::LUNAR:
        dynamics_.push_back([](const OrbitalState& s) { return lunarPerturbation(s); });
        break;
      case ForceModel::SOLAR:
        dynamics_.push_back([](const OrbitalState& s) { return solarPerturbation(s); });
        break;
      case ForceModel::J2:
        dynamics_.push_back([this](const OrbitalState& s) { return j2Perturbation(mu_, s); });
        break;
      case ForceModel::ATMOSPHERIC:
        dynamics_.push_back([](const OrbitalState& s) { return atmosphericPerturbation(s); });
        break;
      case ForceModel::SOLAR_RADIATION:
        dynamics_.push_back([](const OrbitalState& s) { return solarRadiationPerturbation(s); });
        break;
    }
  }
}

std::vector<OrbitalState> Propagator::propagate(const OrbitalState& initial,
                                                double timeStep,
                                                double endTime) const {
  if (type_ == PropagatorType::RK4) {
    return rk4(dynamics_, initial, timeStep, endTime);
  }
  return dp45(
    [this](const OrbitalState& s) { return twoBodyDynamics(mu_, s); },
    initial,
    timeStep,
    endTime
  );
}

} // namespace naturalis


