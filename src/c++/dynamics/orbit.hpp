#include <Eigen/Dense>

struct OrbitalState {
    float time;

    Eigen::Vector3d position;
    
    Eigen::Vector3d velocity;
}