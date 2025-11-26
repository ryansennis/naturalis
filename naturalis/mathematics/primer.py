from dataclasses import dataclass
from naturalis.dynamics.orbit import Segment, Trajectory, Burn
from numpy.linalg import norm, inv
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from typing import List

import numpy as np

@dataclass
class PrimerVector:
    time: float
    value: NDArray
    derivative: NDArray

    @property
    def magnitude(self) -> float:
        return float(norm(self.value))

@dataclass
class PrimerVectorSegment:
    vectors: List[PrimerVector]

    @property
    def maximum(self) -> PrimerVector:
        return max(self.vectors, key=lambda v: v.magnitude)

@dataclass
class PrimerVectorTrajectory:
    segments: List[PrimerVectorSegment]

    @property
    def maximum(self) -> PrimerVector:
        maximum_vector = PrimerVector(0, np.zeros(3), np.zeros(3))
        for segment in self.segments:
            maximum_in_segment = segment.maximum
            if maximum_in_segment.magnitude > maximum_vector.magnitude:
                maximum_vector = maximum_in_segment

        return maximum_vector


class PrimerVectorAnalyzer:
    """
    Analyzes trajectories using primer vector theory. Implements primer vector propagation and analysis based on Lawden's necessary conditions for optimal impulsive trajectories.
    """
    def __init__(
        self: 'PrimerVectorAnalyzer',
        mu: float,
        debug: bool = False
    ) -> None:
        """Initialize with gravitational parameter."""
        self.mu = mu
        self.debug = debug

    def _log(
        self: 'PrimerVectorAnalyzer',
        msg: str
    ) -> None:
        if self.debug:
            print(f"[Primer Analyzer] {msg}")
    
    def _dynamics(
        self: 'PrimerVectorAnalyzer',
        t: float,
        y: NDArray
    ) -> NDArray:
        phi = y[:36].reshape((6, 6))
        
        r = y[36:39]
        v = y[39:42]
        
        r_mag = norm(r)
        
        r_dot = v
        v_dot = -self.mu * r / (r_mag**3)
        
        I3 = np.eye(3)
        outer_rr = np.outer(r, r)
        dAcc_dr = -self.mu / (r_mag**3) * (I3 - 3 * outer_rr / (r_mag**2))
        
        J = np.block([
            [np.zeros((3, 3)),        I3       ],
            [dAcc_dr         , np.zeros((3, 3))]
        ])
        
        phi_dot = J @ phi
        phi_dot_flat = phi_dot.flatten()

        return np.concatenate([phi_dot_flat, r_dot, v_dot])
    
    def analyze_segment(
        self: 'PrimerVectorAnalyzer',
        segment: Segment,
        initial_burn: Burn,
        final_burn: Burn
    ) -> PrimerVectorSegment:
        """
        Analyze a trajectory segment between two burns.
        """
        dv_o = initial_burn.delta_v
        dv_f = final_burn.delta_v
        
        p_o = dv_o / norm(dv_o)
        p_f = dv_f / norm(dv_f)

        phi0 = np.eye(6).flatten()
        y0 = np.concatenate([phi0, segment.initial_state.position, segment.initial_state.velocity])
        
        sol = solve_ivp(
            self._dynamics,
            (segment.initial_state.time, segment.final_state.time),
            y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-10
        )
        
        states: List[PrimerVector] = []
        for i in range(len(sol.t)):
            phi = sol.y[0:36, i].reshape((6,6))
            M_to = phi[0:3, 0:3]
            N_to = phi[0:3, 3:6]
            S_to = phi[3:6, 0:3] 
            T_to = phi[3:6, 3:6]
            
            phi_f = sol.y[0:36,-1].reshape((6,6))
            M_fo = phi_f[0:3, 0:3]
            N_fo = phi_f[0:3, 3:6]
            
            p_dot_o = inv(N_fo) @ (p_f - M_fo @ p_o)
            
            p = (N_to @ inv(N_fo) @ (dv_f/norm(dv_f)) + (M_to - N_to @ inv(N_fo) @ M_fo) @ (dv_o / norm(dv_o)))
            
            p_dot = S_to @ p_o + T_to @ p_dot_o
            
            states.append(PrimerVector(sol.t[i], p, p_dot))
            
        return PrimerVectorSegment(states)

    def analyze_trajectory(
        self: 'PrimerVectorAnalyzer',
        trajectory: Trajectory
    ) -> PrimerVectorTrajectory:
        """
        Analyze a full trajectory consisting of multiple segments.
        
        Args:
            trajectory (Trajectory): Full trajectory to analyze.
            
        Returns:
            primer_trajectory (PrimerVectorTrajectory): Primer vector trajectory containing analysis results.
        """
        primer_segments = []
        for i in range(len(trajectory.segments)):
            segment = trajectory.segments[i]
            primer_segment = self.analyze_segment(segment, trajectory.burns[i], trajectory.burns[i + 1])
            primer_segments.append(primer_segment)
            
        return PrimerVectorTrajectory(primer_segments)
    
    def midcourse_correction_matrix(
        self: 'PrimerVectorAnalyzer',
        segment: Segment,
        midcourse_time: float
    ) -> NDArray:
        """
        Calculate the matrix A that is used to calculate the initial guess for the midpoint burn position.

        Args:
            segment (Segment): The segment to add the midcourse burn to.
            midcourse_time (float): The initial guess at the midcourse correction burn time.

        Returns:
            A (NDArray): The matrix used to calculate the initial position of the midcourse burn.
        """
        phi0 = np.eye(6).flatten()
        y0_initial = np.concatenate([phi0, segment.initial_state.position, segment.initial_state.velocity])
        
        sol_mo = solve_ivp(
            self._dynamics,
            (segment.initial_state.time, midcourse_time),
            y0_initial,
            method='RK45',
            rtol=1e-10,
            atol=1e-10
        )

        midcourse_r = sol_mo.y[36:39, -1]
        midcourse_v = sol_mo.y[39:42, -1]

        y0_midcourse = np.concatenate([phi0, midcourse_r, midcourse_v])

        sol_fm = solve_ivp(
            self._dynamics,
            (midcourse_time, segment.final_state.time),
            y0_midcourse,
            method='RK45',
            rtol=1e-10,
            atol=1e-10
        )
        
        phi_mo = sol_mo.y[0:36, -1].reshape((6,6))
        N_mo = np.array(phi_mo[0:3, 3:6])
        T_mo = np.array(phi_mo[3:6, 3:6])

        phi_fm = sol_fm.y[0:36, -1].reshape((6,6))
        M_fm = np.array(phi_fm[0:3, 0:3])
        N_fm = np.array(phi_fm[0:3, 3:6])

        A = -(M_fm.T @ inv(N_fm.T).T + T_mo @ inv(N_mo))

        return A