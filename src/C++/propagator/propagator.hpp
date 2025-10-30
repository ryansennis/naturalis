#pragma once

#include <vector>
#include "../dynamics/dynamics.hpp"
#include "../integrators/integrators.hpp"

namespace naturalis {

enum class PropagatorType { RK4, DP45 };

class Propagator {
public:
  explicit Propagator(PropagatorType type, double mu, std::vector<ForceModel> models = {ForceModel::TWO_BODY});

  std::vector<OrbitalState> propagate(const OrbitalState& initial,
                                      double timeStep,
                                      double endTime) const;

private:
  PropagatorType type_;
  double mu_;
  std::vector<DynamicsFn> dynamics_;
};

} // namespace naturalis


