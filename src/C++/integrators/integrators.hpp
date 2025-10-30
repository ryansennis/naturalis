#pragma once

#include <functional>
#include <vector>
#include "../orbit/orbit.hpp"

namespace naturalis {

using DynamicsFn = std::function<OrbitalState(const OrbitalState&)>;

std::vector<OrbitalState> rk4(const std::vector<DynamicsFn>& dynamics,
                              const OrbitalState& initialState,
                              double timeStep,
                              double endTime);

OrbitalState rk4Step(const DynamicsFn& dynamics, const OrbitalState& state);

std::vector<OrbitalState> dp45(const DynamicsFn& dynamics,
                               const OrbitalState& initialState,
                               double timeStep,
                               double endTime);

OrbitalState dp45Step(const DynamicsFn& dynamics, const OrbitalState& state);

} // namespace naturalis


