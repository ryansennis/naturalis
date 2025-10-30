import numpy as np
from typing import Callable, List
from .dynamics import OrbitalState

def rk4(dynamics: List[Callable], initial_state: OrbitalState, time_step: float, end_time: float) -> list[OrbitalState]:
    """
    Runge-Kutta 4 propagator.
    Args:
        dynamics: Dynamics function.
        initial_state: Initial orbital state.
        time_step: Time step in units of [s].
        end_time: End time in units of [s].
    Returns:
        List of orbital states.
    """
    states = []
    for _ in np.arange(0, end_time, time_step):
        states.append(initial_state)
        temp_state = initial_state
        for model in dynamics:
            temp_state = temp_state + time_step * rk4_step(model, temp_state)
        initial_state = temp_state
            
    return states

def dp45(dynamics: Callable, initial_state: OrbitalState, time_step: float, end_time: float) -> list[OrbitalState]:
    """
    Dormand-Prince 4(5) propagator.
    Args:
        dynamics: Dynamics function.
        initial_state: Initial orbital state.
        time_step: Time step in units of [s].
        end_time: End time in units of [s].
    Returns:
        List of orbital states.
    """
    states = []
    for _ in np.arange(0, end_time, time_step):
        states.append(initial_state)
        initial_state = initial_state + time_step * dp45_step(dynamics, initial_state)
    return initial_state

def rk4_step(dynamics: Callable, state: OrbitalState) -> OrbitalState:
    """
    Runge-Kutta 4 step.
    Args:
        dynamics: Dynamics function.
        state: Orbital state.
    Returns:
        Orbital state of the derivative.
    """
    k1 = dynamics(state)
    k2 = dynamics(state + 0.5 * k1)
    k3 = dynamics(state + 0.5 * k2)
    k4 = dynamics(state + k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6

def dp45_step(dynamics: Callable, state: OrbitalState) -> OrbitalState:
    """
    Dormand-Prince 4(5) step.
    Args:
        dynamics: Dynamics function.
        state: Orbital state.
    Returns:
        Orbital state of the derivative.
    """
    k1 = dynamics(state)
    k2 = dynamics(state + (1/5) * k1)
    k3 = dynamics(state + (3/40) * k1 + (9/40) * k2)
    k4 = dynamics(state + (44/45) * k1 - (56/15) * k2 + (32/9) * k3)
    k5 = dynamics(state + (19372/6561) * k1 - (25360/2187) * k2 + (64448/6561) * k3 - (212/729) * k4)
    k6 = dynamics(state + (9017/3168) * k1 - (355/33) * k2 + (46732/5247) * k3 + (49/176) * k4 - (5103/18656) * k5)
    return (35/384) * k1 + (500/1113) * k3 + (125/192) * k4 - (2187/6784) * k5 + (11/84) * k6