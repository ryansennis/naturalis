#pragma once

#include <Eigen/Dense>
#include "../orbit/orbit.hpp"

namespace naturalis {

enum class ForceModel {
  TWO_BODY,
  LUNAR,
  SOLAR,
  J2,
  ATMOSPHERIC,
  SOLAR_RADIATION
};

OrbitalState twoBodyDynamics(double mu, const OrbitalState& state);

// Placeholders for future perturbations
OrbitalState lunarPerturbation(const OrbitalState& state);
OrbitalState solarPerturbation(const OrbitalState& state);
OrbitalState j2Perturbation(double /*mu*/, const OrbitalState& state);
OrbitalState atmosphericPerturbation(const OrbitalState& state);
OrbitalState solarRadiationPerturbation(const OrbitalState& state);

} // namespace naturalis


