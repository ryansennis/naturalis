from enum import Enum
import numpy as np

class PropagatorType(Enum):
    RK4 = "RK4"
    DP45 = "DP45"

class Propagator:
    def __init__(self, propagator_type: PropagatorType, mu: float):
        self.propagator_type = propagator_type
        self.mu = mu

    def propagate(self, initial_state: np.ndarray, time_step: float, end_time: float) -> np.ndarray:
        if self.propagator_type == PropagatorType.RK4:
            return self.rk4(initial_state, time_step, end_time)
        elif self.propagator_type == PropagatorType.DP45:
            return self.dp45(initial_state, time_step, end_time)

    def dynamics(self, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:]
        norm_r = np.linalg.norm(r)
        a = -self.mu * r / norm_r**3
        return np.concatenate((v, a))

    def rk4(self, initial_state: np.ndarray, time_step: float, end_time: float) -> np.ndarray:
        return initial_state + time_step * self.rk4_step(initial_state)

    def dp45(self, initial_state: np.ndarray, time_step: float, end_time: float) -> np.ndarray:
        return initial_state + time_step * self.dp45_step(initial_state)

    def rk4_step(self, state: np.ndarray) -> np.ndarray:
        k1 = self.dynamics(state)
        k2 = self.dynamics(state + 0.5 * k1)
        k3 = self.dynamics(state + 0.5 * k2)
        k4 = self.dynamics(state + k3)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def dp45_step(self, state: np.ndarray) -> np.ndarray:
        k1 = self.dynamics(state)
        k2 = self.dynamics(state + (1/5) * k1)
        k3 = self.dynamics(state + (3/40) * k1 + (9/40) * k2)
        k4 = self.dynamics(state + (44/45) * k1 - (56/15) * k2 + (32/9) * k3)
        k5 = self.dynamics(state + (19372/6561) * k1 - (25360/2187) * k2 + (64448/6561) * k3 - (212/729) * k4)
        k6 = self.dynamics(state + (9017/3168) * k1 - (355/33) * k2 + (46732/5247) * k3 + (49/176) * k4 - (5103/18656) * k5)

        return (35/384) * k1 + (500/1113) * k3 + (125/192) * k4 - (2187/6784) * k5 + (11/84) * k6