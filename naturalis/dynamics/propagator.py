from dataclasses import dataclass
from enum import Enum
from naturalis.dynamics.orbit import OrbitalState
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from typing import Callable, List, Tuple

import numpy as np
    
class PropagationMethod(Enum):
    """
    Represents a propagation method to use within the propagator.
    """
    RK23 = 'RK23'
    RK45 = 'RK45'
    DOP83 = 'DOP853'
    RADAU = 'Radau'
    BDF = 'BDF'
    LSODA = 'LSODA'

def two_body_dynamics(
    state: OrbitalState
) -> Tuple[NDArray, NDArray]:
    """
    Dynamics for two-body central gravitation.

    Args:
        state (OrbitalState): The state of the objective body about the primary graviational body.

    Returns:
        velocity, accelaration (Tuple[NDArray, NDArray]): The velocity and acceleration at that state.
    """
    r = state.position
    r_mag = np.linalg.norm(state.position)
    a = -(state.mu/r_mag**3) * r
    return state.velocity, a

class ForceModel(Enum):
    """
    Represents a force model to use within the propagator.
    """
    TWO_BODY = two_body_dynamics
    J2 = lambda state: state
    SOLAR = lambda state: state
    LUNAR = lambda state: state
    RADIATION = lambda state: state
    ATMOSPHERIC = lambda state: state

    @property
    def value(self: 'ForceModel') -> Callable[[OrbitalState], Tuple[NDArray, NDArray]]:
        return self.value

@dataclass 
class PropagatorSolution:
    """
    Wrapper for solutions that come out of scipy's `solve_ivp' function.

    Parameters:
        t (NDArray): The time history of the solution.
        y (NDArray): The solution.
    """
    t: NDArray
    y: NDArray

    @property
    def times(self: 'PropagatorSolution') -> NDArray:
        return self.t
    
    @property
    def solution(self: 'PropagatorSolution') -> NDArray:
        return self.y

    @staticmethod
    def from_solve_ivp_output(output) -> 'PropagatorSolution':
        """
        Static method to convert a `solve_ivp` function output to this class.

        Args:
            output (Bundle): A Bundle object output from `solve_ivp`.
        """
        return PropagatorSolution(
            t=output.t,
            y=output.y
        )
    
class OrbitalPropagator:
    def __init__(
        self: 'OrbitalPropagator',
        forces: List[ForceModel],
        method: PropagationMethod
    ) -> None:
        self.forces = forces
        self.method = method

    def _propagate(
        self: 'OrbitalPropagator',
        state: OrbitalState,
        time: float,
        rtol: float = 1e-10,
        atol: float = 1e-10
    ) -> PropagatorSolution:
        '''
        Helper function to propagate a state by some time.
        
        Args:
        state (OrbitalState): The state to propagate.
        time (float): The time to propagate by.
        rtol (float): Relative tolerance for internal solver. Defaults to 1e-10.
        atol (float): Absolute tolerance for internal solver. Defaults to 1e-10.

        Returns:
            solution (PropagatorSolution): The solution coming out of the internal solver.
        '''
        def dynamics(
            t: float,
            y: NDArray
        ) -> NDArray:
            v_f, a_f = 0, 0

            for model in self.forces:
                v, a = model.value(OrbitalState(state.mu, t, y[0:3], y[3:6]))
                v_f += v
                a_f += a

            return np.concatenate([v_f, a_f])
        
        y0 = np.concatenate([state.position, state.velocity])
        
        solution = PropagatorSolution.from_solve_ivp_output(
            solve_ivp(
                dynamics,
                (state.time, time),
                y0,
                method=self.method.value,
                rtol=rtol,
                atol=atol
            )
        )

        return solution

    def propagate_to_time(
        self: 'OrbitalPropagator',
        state: OrbitalState,
        time: float,
        rtol: float = 1e-10,
        atol: float = 1e-10
    ) -> OrbitalState:
        '''
        Propagates a state to some absolute time.
        
        Args:
        state (OrbitalState): The state to propagate.
        time (float): The time to propagate to.
        rtol (float): Relative tolerance for internal solver. Defaults to 1e-10.
        atol (float): Absolute tolerance for internal solver. Defaults to 1e-10.

        Returns:
            state (OrbitalState): The propagated state.
        '''
        output = self._propagate(
            state,
            time,
            rtol,
            atol
        )

        return OrbitalState(
            state.mu,
            output.times[-1],
            output.solution[0:3, -1],
            output.solution[3:6, -1]
        )

    def propagate_by_time(
        self: 'OrbitalPropagator',
        state: OrbitalState,
        time: float,
        rtol: float = 1e-10,
        atol: float = 1e-10
    ) -> OrbitalState:
        '''
        Propagates a state by some relative time.
        
        Args:
        state (OrbitalState): The state to propagate.
        time (float): The time to propagate by.
        rtol (float): Relative tolerance for internal solver. Defaults to 1e-10.
        atol (float): Absolute tolerance for internal solver. Defaults to 1e-10.

        Returns:
            state (OrbitalState): The propagated state.
        '''
        output = self._propagate(
            state,
            state.time + time,
            rtol,
            atol
        )

        return OrbitalState(
            state.mu,
            output.times[-1],
            output.solution[0:3, -1],
            output.solution[3:6, -1]
        )
    
    def propagate_trajectory_to_time(
        self: 'OrbitalPropagator',
        state: OrbitalState,
        time: float,
        rtol: float = 1e-10,
        atol: float = 1e-10
    ) -> List[OrbitalState]:
        
        '''
        Propagates a state to some absolute time and returns the entire trajectory.
        
        Args:
        state (OrbitalState): The state to propagate.
        time (float): The time to propagate to.
        rtol (float): Relative tolerance for internal solver. Defaults to 1e-10.
        atol (float): Absolute tolerance for internal solver. Defaults to 1e-10.

        Returns:
            state (List[OrbitalState]): The propagated trajectory.
        '''
        output = self._propagate(
            state,
            time,
            rtol,
            atol
        )

        trajectory: List[OrbitalState] = []

        for i in range(output.times.shape[0]):
            time = output.times[i]
            position = output.solution[0:3, i]
            velocity = output.solution[3:6, i]

            trajectory.append(
                OrbitalState(
                    state.mu,
                    time,
                    position,
                    velocity
                )
            )

        return trajectory
    
    def propagate_trajectory_by_time(
        self: 'OrbitalPropagator',
        state: OrbitalState,
        time: float,
        rtol: float = 1e-10,
        atol: float = 1e-10
    ) -> List[OrbitalState]:
        
        '''
        Propagates a state by some relative time and returns the entire trajectory.
        
        Args:
        state (OrbitalState): The state to propagate.
        time (float): The time to propagate by.
        rtol (float): Relative tolerance for internal solver. Defaults to 1e-10.
        atol (float): Absolute tolerance for internal solver. Defaults to 1e-10.

        Returns:
            state (List[OrbitalState]): The propagated trajectory.
        '''
        output = self._propagate(
            state,
            state.time + time,
            rtol,
            atol
        )

        trajectory: List[OrbitalState] = []

        for i in range(output.times.shape[0]):
            time = output.times[i]
            position = output.solution[0:3, i]
            velocity = output.solution[3:6, i]

            trajectory.append(
                OrbitalState(
                    state.mu,
                    time,
                    position,
                    velocity
                )
            )

        return trajectory
