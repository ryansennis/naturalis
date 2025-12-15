#include <Eigen/Dense>
#include <optional>

/** @struct Burn - Impulsive burn for orbital dyanmics */
struct Burn {
    /** @property time - The time the burn occurs at */
    float time;

    /** @property value - The burn vector in [km/s, km/s, km/s] */
    Eigen::Vector3d value;
};

/** @struct OrbitalState */
struct OrbitalState {
    float time;

    Eigen::Vector3d position;
    
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