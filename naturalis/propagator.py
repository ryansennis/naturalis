import numpy as np

from dataclasses import dataclass
from enum import Enum
from numpy.typing import NDArray
from scipy.integrate import solve_ivp, OdeSolution
from typing import Callable, List, Tuple

@dataclass
class OrbitalState:
    mu: float
    time: float
    position: NDArray
    velocity: NDArray

    def __post_init__(self):
        assert self.mu >= 0
        assert self.time >= 0
        assert len(self.position) == 3
        assert len(self.velocity) == 3

        self.position = np.array(self.position)
        self.velocity = np.array(self.velocity)
    
@dataclass
class OrbitalParameters:
    """
    Keplerian orbital elements that define an orbit's shape and orientation:
        mu = standard gravitational parameter (km^3 s^-2)
        a = semi-major axis (km)
        e = eccentricity
        i = inclination (rad)
        raan = right ascension of ascending node (rad)
        aop = argument of periapsis (rad)
    """
    mu: float
    a: float
    e: float
    i: float
    raan: float
    aop: float

    @property
    def period(self) -> float:
        """Orbital period (s)"""
        return 2 * np.pi * np.sqrt(self.a**3 / self.mu)
    
    @property 
    def h(self) -> float:
        """Specific angular momentum (km^2 s^-1)"""
        return np.sqrt(self.mu * self.a * (1 - self.e**2))

    @property
    def p(self) -> float:
        """Semilatus rectum (km)"""
        return self.a * (1 - self.e**2)
    
    def radius(self, nu: float) -> float:
        """
        Get radius at a given true anomaly nu.
        
        Args:
            nu: True anomaly (rad)
        Returns:
            Radius (km)
        """
        return self.p / (1 + self.e * np.cos(nu))

    def nu(self, t: float, t0: float) -> float:
        """
        Get true anomaly at time t given initial time t0.
        
        Args:
            t (float): Time to evaluate at (s)
            t0 (float): Initial time (s)
            
        Returns:
            True anomaly (rad)
        """
        n = np.sqrt(self.mu / self.a**3)
        
        M = n * (t - t0)
        
        E = M
        for _ in range(10):
            E_next = M + self.e * np.sin(E)
            if abs(E_next - E) < 1e-12:
                break
            E = E_next
            
        return 2 * np.arctan(np.sqrt((1 + self.e)/(1 - self.e)) * np.tan(E/2))

    def to_state(
        self,
        nu: float,
        time: float
    ) -> OrbitalState:
        """
        Convert orbital parameters and true anomaly to an orbital state.
        
        Args:
            nu (float): True anomaly (rad).
            time (float): The time of the orbital state (s).
            
        Returns:
            (OrbitalState): The state representation of the parameters at nu.
        """
        r = self.radius(nu)
        r_pqw = r * np.array([np.cos(nu), np.sin(nu), 0])
        v_pqw = self.mu/self.h * np.array([-np.sin(nu), self.e + np.cos(nu), 0])
        
        R_w = np.array([
            [np.cos(self.aop), np.sin(self.aop), 0],
            [-np.sin(self.aop), np.cos(self.aop), 0],
            [0, 0, 1]
        ])
        
        R_i = np.array([
            [1, 0, 0],
            [0, np.cos(self.i), -np.sin(self.i)],
            [0, np.sin(self.i), np.cos(self.i)]
        ])
        
        R_W = np.array([
            [np.cos(self.raan), -np.sin(self.raan), 0],
            [np.sin(self.raan), np.cos(self.raan), 0],
            [0, 0, 1]
        ])
        
        R = R_W @ R_i @ R_w
        r_eci = R @ r_pqw
        v_eci = R @ v_pqw
        
        return OrbitalState(self.mu, time, r_eci, v_eci)
    
class PropagationMethod(Enum):
    RK23 = 'RK23'
    RK45 = 'RK45'
    DOP83 = 'DOP853'

def two_body_dynamics(
    state: OrbitalState
) -> Tuple[NDArray, NDArray]:
    r = state.position
    r_mag = np.linalg.norm(state.position)
    a = -(state.mu/r_mag**3) * r
    return state.velocity, a

class ForceModel(Enum):
    TWO_BODY = 1
    J2 = 2
    SOLAR = 3
    LUNAR = 4
    RADIATION = 5
    ATMOSPHERIC = 6

def get_force_model(
    model: ForceModel
) -> Callable[[OrbitalState], Tuple[NDArray, NDArray]]:
    '''
    Gets a force model function given the specific enum value.

    Args:
        model (ForceModel): The model to get.

    Returns:
        function (Callable): The returned force model function.
    '''
    temp = lambda state: state
    match model:
        case ForceModel.TWO_BODY:
            return two_body_dynamics
        case ForceModel.J2:
            return temp
        case ForceModel.SOLAR:
            return temp
        case ForceModel.LUNAR:
            return temp
        case ForceModel.RADIATION:
            return temp
        case ForceModel.ATMOSPHERIC:
            return temp
        case _:
            raise ValueError(f"Force model {model} does not exist.")

@dataclass 
class PropagatorSolution:
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
        def dynamics(
            t: float,
            y: NDArray
        ) -> NDArray:
            v_f, a_f = 0, 0

            for model in self.forces:
                force = get_force_model(model)
                v, a = force(OrbitalState(state.mu, t, y[0:3], y[3:6]))
                v_f += v
                a_f += a

            return np.concatenate([v_f, a_f])
        
        y0 = np.concatenate([state.position, state.velocity])
        
        output = PropagatorSolution.from_solve_ivp_output(
            solve_ivp(
                dynamics,
                (state.time, time),
                y0,
                method=self.method.value,
                rtol=rtol,
                atol=atol
            )
        )

        return output

    def propagate_to_time(
        self: 'OrbitalPropagator',
        state: OrbitalState,
        time: float,
        return_trajectory: bool = False,
        rtol: float = 1e-10,
        atol: float = 1e-10
    ) -> OrbitalState | List[OrbitalState]:
        output = self._propagate(
            state,
            time,
            rtol,
            atol
        )

        if not return_trajectory:
            return OrbitalState(
                state.mu,
                output.times[-1],
                output.solution[0:3, -1],
                output.solution[3:6, -1]
            )
        else:
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
        

    def propagate_by_time(
        self: 'OrbitalPropagator',
        state: OrbitalState,
        time: float,
        return_trajectory: bool = False,
        rtol: float = 1e-10,
        atol: float = 1e-10
    ) -> OrbitalState | List[OrbitalState]:
        output = self._propagate(
            state,
            state.time + time,
            rtol,
            atol
        )

        if not return_trajectory:
            return OrbitalState(
                state.mu,
                output.times[-1],
                output.solution[0:3, -1],
                output.solution[3:6, -1]
            )
        else:
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
