#include <Eigen/Dense>
#include <optional>

#pragma once

/** @struct OrbitalState */
struct OrbitalState {
    /** The time the state occurs at */
    float time;

    /** The state position in [km, km, km] */
    Eigen::Vector3d position;

    /** The state velocity in [km/s, km/s, km/s] */
    Eigen::Vector3d velocity;
};

struct Segment {
    OrbitalState initialState;

    OrbitalState finalState;
};

struct Trajectory {
    std::optional<Segment> initialCoast;

    std::optional<Segment> finalCoast;

    std::vector<Segment> segments;
};

/** @struct Burn - Impulsive burn for orbital dyanmics */
struct Burn {
    /** @property time - The time the burn occurs at */
    float time;

    /** @property value - The burn vector in [km/s, km/s, km/s] */
    Eigen::Vector3d value;
};
