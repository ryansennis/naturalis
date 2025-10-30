import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Callable

class ForceModel(Enum):
    TWO_BODY = "TWO_BODY"
    LUNAR = "LUNAR"
    SOLAR = "SOLAR"
    J2 = "J2"
    ATMOSPHERIC = "ATMOSPHERIC"
    SOLAR_RADIATION = "SOLAR_RADIATION"

@dataclass
class OrbitalState:
    """
    Orbital state class.
    Args:
        r: Position vector.
        v: Velocity vector.
    """
    r: np.ndarray
    v: np.ndarray

    @property
    def shape(self) -> int:
        return len(self.r) + len(self.v)

    def __add__(self, other: "OrbitalState") -> "OrbitalState":
        return OrbitalState(self.r + other.r, self.v + other.v)

def collect_force_models(force_models: list[ForceModel]) -> list[Callable]:
    """
    Collect force models.
    Args:
        force_models: List of force models.
    Returns:
        List of force model functions.
    """
    force_models_functions = []
    for force_model in force_models:
        if force_model == ForceModel.TWO_BODY:
            force_models_functions.append(two_body_dynamics)
        elif force_model == ForceModel.LUNAR:
            force_models_functions.append(lunar_perturbation)
        elif force_model == ForceModel.SOLAR:
            force_models_functions.append(solar_perturbation)
        elif force_model == ForceModel.ATMOSPHERIC:
            force_models_functions.append(atmospheric_perturbation)
        elif force_model == ForceModel.SOLAR_RADIATION:
            force_models_functions.append(solar_radiation_perturbation)
    return force_models_functions

def two_body_dynamics(mu: float, orbital_state: OrbitalState) -> OrbitalState:
    """
    Two-body dynamics equation.
    Args:
        mu: Gravitational parameter. (km^3/s^2)
        orbital_state: Orbital state.
    Returns:
        Orbital state of the derivative.
    """
    r = orbital_state.r
    v = orbital_state.v
    norm_r = np.linalg.norm(r)
    return OrbitalState(v, -mu * r / norm_r**3)

def lunar_perturbation(orbital_state: OrbitalState) -> OrbitalState:
    """
    Placeholder for lunar perturbation dynamics.
    Args:
        orbital_state: Orbital state.
    Returns:
        An OrbitalState with placeholder perturbation.
    """
    # TODO: Implement lunar perturbation using ephemeris data.
    a_lunar = np.zeros_like(orbital_state.r)
    return OrbitalState(orbital_state.v, a_lunar)

def solar_perturbation(orbital_state: OrbitalState) -> OrbitalState:
    """
    Placeholder for solar perturbation dynamics.
    Args:
        orbital_state: Orbital state.
    Returns:
        An OrbitalState with placeholder perturbation.
    """
    # TODO: Implement solar perturbation using ephemeris data.
    a_solar = np.zeros_like(orbital_state.r)
    return OrbitalState(orbital_state.v, a_solar)

def j2_perturbation(mu: float, orbital_state: OrbitalState) -> OrbitalState:
    """
    Placeholder for J2 perturbation dynamics.
    Args:
        mu: Gravitational parameter (not used).
        orbital_state: Orbital state.
    Returns:
        An OrbitalState with placeholder perturbation.
    """
    # TODO: Implement J2 perturbation computation.
    a_j2 = np.zeros_like(orbital_state.r)
    return OrbitalState(orbital_state.v, a_j2)

def atmospheric_perturbation(orbital_state: OrbitalState) -> OrbitalState:
    """
    Placeholder for atmospheric drag perturbation (Jacchia model).
    Args:
        orbital_state: Orbital state.
    Returns:
        An OrbitalState with placeholder atmospheric drag acceleration.
    """
    # TODO: Implement Jacchia atmospheric drag computation.
    a_atm = np.zeros_like(orbital_state.r)
    return OrbitalState(orbital_state.v, a_atm)

def solar_radiation_perturbation(orbital_state: OrbitalState) -> OrbitalState:
    """
    Placeholder for solar radiation pressure perturbation dynamics.
    Args:
        orbital_state: Orbital state.
    Returns:
        An OrbitalState with placeholder solar radiation pressure perturbation.
    """
    # TODO: Implement solar radiation pressure acceleration computation.
    a_srp = np.zeros_like(orbital_state.r)
    return OrbitalState(orbital_state.v, a_srp)
