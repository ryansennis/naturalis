#pragma once

#include <Eigen/Dense>

namespace naturalis {

struct OrbitalState {
  Eigen::Vector3d r; // position (km)
  Eigen::Vector3d v; // velocity (km/s)

  inline std::size_t dimension() const noexcept { return static_cast<std::size_t>(r.size() + v.size()); }

  inline OrbitalState operator+(const OrbitalState& other) const {
    return OrbitalState{r + other.r, v + other.v};
  }

  inline OrbitalState operator*(double scalar) const {
    return OrbitalState{r * scalar, v * scalar};
  }
};

inline OrbitalState operator*(double scalar, const OrbitalState& s) { return s * scalar; }

} // namespace naturalis


