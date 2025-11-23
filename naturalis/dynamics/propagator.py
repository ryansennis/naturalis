from dataclasses import dataclass
from enum import Enum
from naturalis.dynamics.orbit import OrbitalState, Segment, Trajectory, Burn
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from typing import List, Tuple, Optional

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

    def __call__(self, state) -> Tuple[NDArray, NDArray]:
        return self.value(state)

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
                v, a = model(OrbitalState(state.mu, t, y[0:3], y[3:6]))
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

    def propagate_state_to_time(
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

    def propagate_state_by_time(
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
            state=state,
            time=state.time + time,
            rtol=rtol,
            atol=atol
        )

        return OrbitalState(
            mu=state.mu,
            time=output.times[-1],
            position=output.solution[0:3, -1],
            velocity=output.solution[3:6, -1]
        )
    
    def propagate_state_to_segment(
        self: 'OrbitalPropagator',
        initial_state: OrbitalState,
        time: float,
        rtol: float = 1e-10,
        atol: float = 1e-10
    ) -> Segment:
        final_state = self.propagate_state_to_time(
            state=initial_state,
            time=time,
            rtol=rtol,
            atol=atol
        )

        return Segment(initial_state=initial_state, final_state=final_state)
    
    def propagate_state_by_segment(
        self: 'OrbitalPropagator',
        initial_state: OrbitalState,
        time: float,
        rtol: float = 1e-10,
        atol: float = 1e-10
    ) -> Segment:
        final_state = self.propagate_state_by_time(
            state=initial_state,
            time=time,
            rtol=rtol,
            atol=atol
        )

        return Segment(initial_state=initial_state, final_state=final_state)
    
    def propagate_segment(
        self: 'OrbitalPropagator',
        segment: Segment,
        rtol: float = 1e-10,
        atol: float = 1e-10
    ) -> List[OrbitalState]:
        output = self._propagate(
            state=segment.initial_state,
            time=segment.final_state.time,
            rtol=rtol,
            atol=atol
        )

        states: List[OrbitalState] = []

        for i in range(output.y.shape[1]):
            time: float = output.t[i]
            position: NDArray = output.y[0:3, i]
            velocity: NDArray = output.y[3:6, i]

            states.append(
                OrbitalState(
                    mu=segment.initial_state.mu,
                    time=time,
                    position=position,
                    velocity=velocity
                )
            )

        return states
    
    def propagate_segment_to_time(
        self: 'OrbitalPropagator',
        segment: Segment,
        time: float,
        rtol = 1e-10,
        atol = 1e-10
    ) -> OrbitalState:
        if not segment.initial_state.time < time < segment.final_state.time:
            raise ValueError(f"Time {time} does not exist within segment.")

        output = self._propagate(
            state=segment.initial_state,
            time=time,
            rtol=rtol,
            atol=atol
        )

        return OrbitalState(
            mu=segment.initial_state.mu,
            time=time,
            position=output.y[0:3, -1],
            velocity=output.y[3:6, -1]
        )
    
    def propagate_trajectory(
        self: 'OrbitalPropagator',
        trajectory: Trajectory,
        atol = 1e-10,
        rtol = 1e-10
    ) -> List[List[OrbitalState]]:
        states: List[List[OrbitalState]] = []

        for segment in trajectory.segments:
            states.append(self.propagate_segment(segment, rtol, atol))

        return states
    
    def propagate_state_to_trajectory(
        self: 'OrbitalPropagator',
        initial_state: OrbitalState,
        burns: List[Burn],
        time: float,
        rtol = 1e-10,
        atol = 1e-10
    ) -> Trajectory:
        assert burns[0].time >= initial_state.time
        assert burns[-1].time <= time

        segments: List[Segment] = []
        initial_states: List[OrbitalState] = []

        initial_coast: Optional[Segment] = None
        final_coast: Optional[Segment] = None

        if burns[0].time > initial_state.time:
            initial_coast = self.propagate_state_to_segment(
                initial_state=initial_state,
                time=burns[0].time,
                rtol=rtol,
                atol=atol
            )
            initial_states.append(initial_coast.final_state)
            burns[0].position = initial_coast.final_state.position
        else:
            initial_states.append(initial_state)

        for i in range(len(burns) - 1):
            burn = burns[i]
            burn.position = initial_states[i].position
            state = initial_states[i]
            state.velocity = state.velocity + burn.delta_v
            segment = self.propagate_state_to_segment(
                initial_state=state,
                time=burns[i + 1].time,
                rtol=rtol,
                atol=atol
            )
            segments.append(segment)
            initial_states.append(segment.final_state)

        if time > burns[-1].time:
            state = initial_states[-1]
            state.velocity = state.velocity + burns[-1].delta_v
            final_coast = self.propagate_state_to_segment(
                initial_state=state,
                time=time,
                rtol=rtol,
                atol=atol
            )

            burns[-1].position = final_coast.initial_state.position
        else:
            burns[-1].position = segments[-1].final_state.position


        return Trajectory(segments=segments, burns=burns, initial_coast=initial_coast, final_coast=final_coast)