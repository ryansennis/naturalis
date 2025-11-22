from dataclasses import dataclass
from naturalis.mathematics.primer import PrimerVector, PrimerVectorAnalyzer
from naturalis.dynamics.orbit import OrbitalState, Segment, Trajectory, Burn
from naturalis.dynamics.propagator import OrbitalPropagator
from naturalis.solvers.lambert import LambertSolverType
from numpy.linalg import norm, solve
from numpy.typing import NDArray
from scipy.optimize import minimize, OptimizeResult
from typing import Tuple

import numpy as np

@dataclass
class CoastOptimizationOutput():
    t_0: float
    t_f: float
    success: bool
    status: int
    message: str

    @staticmethod
    def from_minimization_output(output: OptimizeResult) -> 'CoastOptimizationOutput':
        return CoastOptimizationOutput(
            output.x[0],
            output.x[1],
            output.success,
            output.status,
            output.message
        )
    
@dataclass
class MidpointOptimizationOutput():
    time: float
    position: NDArray
    success: bool
    status: int
    message: str

    @staticmethod
    def from_minimization_output(output: OptimizeResult) -> 'MidpointOptimizationOutput':
        return MidpointOptimizationOutput(
            output.x[0],
            np.array(output.x[1:4]),
            output.success,
            output.status,
            output.message
        )

class NBurnSolver:
    """
    N-impulse trajectory optimizer using primer vector theory.
    
    This solver starts with a two-impulse Lambert solution and systematically
    improves it by analyzing the primer vector to identify opportunities for:
        1. Adding terminal coast arcs when primer slopes indicate improvement 
        2. Adding midcourse impulses where primer magnitude exceeds unity
        3. Optimizing existing impulse locations using primer vector gradients
    """
    def __init__(
        self: 'NBurnSolver', 
        mu: float,
        propagator: OrbitalPropagator,
        lambert: LambertSolverType,
        debug: bool = False
    ) -> None:
        '''
        N-impulse trajectory optimizer using primer vector theory.

        Args:
            mu (float): The standard gravitational parameter of the primary body.
            debug (bool): Flag to determine logging output. Defaults to False: no logging output.

        Returns:
            (None)
        '''
        self.mu = mu
        self.debug = debug
        self.propagator = propagator
        self.lambert = lambert.value(mu, debug)
        self.primer = PrimerVectorAnalyzer(mu, debug)

    def _log(
        self: 'NBurnSolver',
        msg: str
    ) -> None:
        '''
        Logs a message to the output.

        Args:
            msg (str): The message to log.

        Returns:
            (None)
        '''
        if self.debug:
            print(f"[N Burn] {msg}")

    def _add_terminal_coasts(
        self: 'NBurnSolver',
        initial_state: OrbitalState,
        final_state: OrbitalState,
        trajectory: Trajectory,
        tol: float
    ) -> Trajectory:
        '''
        Checks the given trajectory for viable terminal coasts, and adds the coasts if they are viable.
        
        Args:
            initial_state (OrbitalState): The initial state of the craft before the first Lambert burn.
            final_state (OrbitalState): The final state of the craft after the second Lambert burn.
            trajectory (Trajectory): The Lambert trajectory to add terminal coasts to.
            tol (float): The numerical tolerance used to stop parameter optimization.
        
        Returns:
            coast_trajectory (Trajectory): Trajectory with viable terminal coasts added.

        Notes:
            This function could return a trajectory with just an initial coast, just a final coast, both an initial and a final coast, or the original trajectory.
        '''
        self._log("Optimizing initial coast time")
        t_initial = [trajectory.burns[0].time, trajectory.burns[1].time]

        def objective(t_guess: NDArray) -> NDArray[np.float64]:
            initial_coast_state = self.propagator.propagate_state_by_time(initial_state, t_guess[0])
            final_coast_state = self.propagator.propagate_state_by_time(final_state, t_guess[1])

            candidate_trajectory = self.lambert.solve(initial_coast_state, final_coast_state)

            return np.array([candidate_trajectory.total_delta_v])

        result = CoastOptimizationOutput.from_minimization_output(
            minimize(
                objective, 
                t_initial,
                method="BFGS",
                tol=tol
            )
        )

        initial_coast_state = self.propagator.propagate_state_by_time(initial_state, result.t_0)
        final_coast_state = self.propagator.propagate_state_by_time(final_state, result.t_f)

        best_trajectory = self.lambert.solve(initial_coast_state, final_coast_state)
        best_cost = best_trajectory.total_delta_v

        if result.t_0 > t_initial[0]:
            best_trajectory.initial_coast = Segment(initial_state, initial_coast_state)

        if result.t_f < t_initial[1]:
            best_trajectory.final_coast = Segment(final_coast_state, final_state)
        
        self._log(f"Coast optimization complete!")

        self._log(f"Initial Coast time: {(result.t_0 - t_initial[0]):.2f} s")
        self._log(f"Final Coast time: -{(t_initial[0] - result.t_f):.2f} s")
        self._log(f"Delta-v improvement: {(trajectory.total_delta_v - best_cost):.4f}")

        return best_trajectory
    
    def _check_optimality(
        self: 'NBurnSolver',
        trajectory: Trajectory,
        tol: float = 1e-4
    ) -> bool:
        """
        Checks a given trajectory for primer vector optimality via Lawden's necessary conditions.

        Args:
            trajectory (Trajectory): The trajectory to check for optimality.
            tol (float): Tolerance for numerical comparisons.

        Returns:
            optimal (bool): Boolean that is True if the trajectory satisfies necessary conditions.
        """
        self._log("Checking trajectory optimality")
        primer_trajectory = self.primer.analyze_trajectory(trajectory)

        # Condition 1: Continuity
        for i in range(len(primer_trajectory.segments) - 1):
            primer_minus = primer_trajectory.segments[i].vectors[-1]
            primer_plus = primer_trajectory.segments[i + 1].vectors[0]

            p_minus_mag = norm(primer_minus.value)
            p_dot_minus_mag = norm(primer_minus.derivative)

            p_plus_mag = norm(primer_plus.value)
            P_dot_plus_mag = norm(primer_plus.derivative)

            if abs(p_minus_mag - p_plus_mag) > tol:
                self._log(f"Primer vector is discontinuous at burn {i + 2}")
                return False
            
            if abs(p_dot_minus_mag - P_dot_plus_mag) > tol:
                self._log(f"Primer vector derivative is discontinuous at burn {i + 2}")
                return False
        
        # Condition 2: p(t) <= 1 for all t
        for segment in primer_trajectory.segments:
            for vector in segment.vectors:
                if vector.magnitude > 1.0 + tol:
                    self._log(f"Primer magnitude exceeds 1 at t={vector.time:.2f}: {vector.magnitude:.2f}")
                    return False
        
        # Condition 3: p is a unit vector in the burn direction
        for i, burn in enumerate(trajectory.burns):
            burn_time = burn.time
            burn_direction = burn.direction
            
            for segment in primer_trajectory.segments:
                for vector in segment.vectors:
                    if abs(vector.time - burn_time) < tol:
                        primer_direction = np.array(vector.value) / vector.magnitude
                        alignment = np.dot(primer_direction, burn_direction)
                        if abs(1.0 - alignment) > tol:
                            self._log(f"Primer not aligned with burn at burn {i}: {alignment}")
                            return False
                        
                        if abs(vector.magnitude - 1.0) > tol:
                            self._log(f"Primer magnitude not 1 at burn {i}: {vector.magnitude}")
                            return False
        
        # Condition 4: dp/dt = 0 at intermediate impulses
        for i in range(1, len(trajectory.burns) - 1):
            burn_time = trajectory.burns[i].time
            
            for segment in primer_trajectory.segments:
                for vector in segment.vectors:
                    if abs(vector.time - burn_time) < tol:
                        derivative_mag = norm(vector.derivative)
                        if derivative_mag > tol:
                            self._log(f"Primer derivative not zero at intermediate burn {i}: {derivative_mag}")
                            return False
        
        self._log("Trajectory satisfies all necessary conditions")
        return True
    
    def _get_segment_boundaries(
        self: 'NBurnSolver',
        initial_state: OrbitalState,
        final_state: OrbitalState,
        trajectory: Trajectory,
        index: int
    ) -> Tuple[OrbitalState, OrbitalState]:
        if len(trajectory.segments) == 1:
            return initial_state, final_state
        elif len(trajectory.segments) == 2:
            if index == 0:
                return initial_state, trajectory.segments[1].initial_state
            else:
                return trajectory.segments[0].final_state, final_state
        else:
            if index == 0:
                return initial_state, trajectory.segments[1].initial_state
            elif index == len(trajectory.segments) - 1:
                return trajectory.segments[-2].final_state, final_state
            else:
                return trajectory.segments[index - 1].final_state, trajectory.segments[index + 1].initial_state
        
    def _add_midcourse_correction(
        self: 'NBurnSolver',
        initial_state: OrbitalState,
        final_state: OrbitalState,
        trajectory: Trajectory
    ) -> Tuple[Trajectory, int]:
        """
        Adds a midcourse correction at the point of maximum primer magnitude.
        
        Args:
            trajectory (Trajectory): The trajectory to modify.
            
        Returns:
            midcourse_trajectory (Trajectory): The trajectory with added midcourse burn.
        """
        self._log("Adding midcourse correction")
        primer_trajectory = self.primer.analyze_trajectory(trajectory)
        midcourse_primer = primer_trajectory.maximum
        midcourse_time = midcourse_primer.time
        
        if midcourse_primer.magnitude <= 1.0:
            self._log("Primer magnitude does not exceed 1.0 - no midcourse needed")
            return trajectory, -1
        
        index = None
        for i, segment in enumerate(trajectory.segments):
            if segment.initial_state.time <= midcourse_time <= segment.final_state.time:
                index = i
                initial_state, final_state = self._get_segment_boundaries(
                    initial_state, final_state, trajectory, index
                )
                break
        
        if index is None:
            self._log("No suitable segment found")
            return trajectory, -1
        
        self._log(f"Adding burn to segment {index + 1}")
        
        segment = trajectory.segments[index]
        midcourse_state = self.propagator.propagate_state_to_time(segment.initial_state, midcourse_time)
        midcourse_position = midcourse_state.position

        new_trajectory = trajectory

        A = self.primer.midcourse_correction_matrix(segment, midcourse_time)
        primer_vector = np.array(midcourse_primer.value)
        A_inv_p = solve(A, primer_vector)
            
        beta = 0.1
        improvement_found = False

        initial_dv = trajectory.burns[index].magnitude + trajectory.burns[index + 1].magnitude

        first_trajectory, second_trajectory = Trajectory([], []), Trajectory([], [])
        perturbed_position = np.array([])

        while not improvement_found:
            epsilon = beta * norm(midcourse_position) / norm(A_inv_p)
            delta_r = epsilon * A_inv_p
            
            perturbed_position = midcourse_position + delta_r
            perturbed_state = OrbitalState(self.mu, midcourse_time, perturbed_position, midcourse_state.velocity)
            
            first_trajectory = self.lambert.solve(initial_state, perturbed_state)
            second_trajectory = self.lambert.solve(perturbed_state, final_state)

            v_minus = first_trajectory.segments[0].final_state.velocity
            v_plus = second_trajectory.segments[0].initial_state.velocity

            new_cost = norm(v_plus - v_minus) + norm(first_trajectory.burns[0].delta_v) + norm(second_trajectory.burns[1].delta_v)

            if new_cost < initial_dv:
                improvement_found = True
                self._log(f"Found improvement with beta={beta}, cost reduced by {(initial_dv - new_cost)}")
            elif beta < 1e-6:
                improvement_found = True
                self._log(f"Couldn't find improvement with beta > 1e-6, accepting anyway.")
            else:
                self._log(f"No improvement with beta={beta}, reducing to {beta*0.5}")
                beta *= 0.5

        v_minus = first_trajectory.segments[0].final_state.velocity
        v_plus = second_trajectory.segments[0].initial_state.velocity

        midcourse_delta_v = v_plus - v_minus
        midcourse_burn = Burn(midcourse_time, midcourse_delta_v, perturbed_position)
        
        new_segments = trajectory.segments.copy()
        new_burns = trajectory.burns.copy()

        new_segments[index] = first_trajectory.segments[0]
        new_segments.insert(index + 1, second_trajectory.segments[0])

        new_burns[index] = first_trajectory.burns[0]

        new_burns.insert(index + 1, midcourse_burn)

        new_burns[index + 2] = second_trajectory.burns[1]

        new_trajectory = Trajectory(
            segments=new_segments,
            burns=new_burns,
            initial_coast=trajectory.initial_coast
        )

        new_trajectory.initial_coast = trajectory.initial_coast

        return new_trajectory, index
    
    def _optimize_midcourse_correction(
        self: 'NBurnSolver',
        initial_state: OrbitalState,
        final_state: OrbitalState,
        trajectory: Trajectory,
        index: int,
        tol: float
    ) -> Trajectory:
        """Optimizes midcourse burn using primer vector gradients."""
        burn = trajectory.burns[index + 1]

        if len(trajectory.segments) > 2:
            if index == 0:
                final_state = trajectory.segments[2].initial_state
            elif index == len(trajectory.segments) - 2:
                initial_state = trajectory.segments[index - 1].final_state
            else:
                initial_state = trajectory.segments[index - 1].final_state
                final_state = trajectory.segments[index + 2].initial_state
        
        def objective(params):
            dt, dx, dy, dz = params
            new_time = burn.time + dt
            new_position = burn.position + np.array([dx, dy, dz])
                
            state = OrbitalState(self.mu, new_time, new_position, np.zeros(3))
            pre_traj = self.lambert.solve(initial_state, state)
            post_traj = self.lambert.solve(state, final_state)
            
            delta_v = post_traj.segments[0].initial_state.velocity - pre_traj.segments[0].final_state.velocity
            total_dv = (
                norm(pre_traj.burns[0].delta_v) + 
                norm(delta_v) + 
                norm(post_traj.burns[1].delta_v)
            )
            
            return total_dv
        
        x0 = np.zeros(4)
        
        self._log(f"Starting gradient-based optimization")
        result = MidpointOptimizationOutput.from_minimization_output(
            minimize(
                objective, 
                x0, 
                method='BFGS',
                tol=tol
            )
        )
        
        self._log(f"Optimization successful: {result.message}")
        
        new_time = burn.time + result.time
        new_position = burn.position + np.array(result.position)
        
        state = OrbitalState(self.mu, new_time, new_position, np.zeros(3))
        pre_traj = self.lambert.solve(initial_state, state)
        post_traj = self.lambert.solve(state, final_state)
        
        delta_v = post_traj.segments[0].initial_state.velocity - pre_traj.segments[0].final_state.velocity
        
        new_burn = Burn(new_time, delta_v, new_position)
        
        new_segments = trajectory.segments.copy()
        new_burns = trajectory.burns.copy()
        
        new_segments[index] = pre_traj.segments[0]
        new_segments[index + 1] = post_traj.segments[0]
        
        new_burns[index] = pre_traj.burns[0]
        new_burns[index + 1] = new_burn
        new_burns[index + 2] = post_traj.burns[1]

        new_trajectory = Trajectory(new_segments, new_burns, trajectory.initial_coast)
        
        self._log(f"Final cost: {new_trajectory.total_delta_v:.6f}")
        
        return new_trajectory
    
    def solve(
        self: 'NBurnSolver', 
        initial_state: OrbitalState, 
        final_state: OrbitalState,
        max_burns: int = 6,
        tol: float = 1e-12
    ) -> Trajectory:
        """
        Find optimal N-impulse trajectory between two states using primer vector theory.
        
        Args:
            initial_state (OrbitalState): Initial orbital state.
            final_state (OrbitalState): Final target orbital state.
            max_burns (int): Maximum number of burns allowed in the solution.
            tol (float): Tolerance for optimization convergence.
        
        Returns:
            optimal_trajectory (Trajectory): Optimized trajectory satisfying necessary conditions.
        """
        self._log("=== Starting N-burn optimization ===")
        self._log("Finding initial Lambert solution")
        
        trajectory = self.lambert.solve(initial_state, final_state)
        best_delta_v = trajectory.total_delta_v
        self._log(f"Initial Lambert solution cost: {best_delta_v:.4f} km/s")

        # self._add_terminal_coasts(
        #     initial_state=initial_state,
        #     final_state=final_state,
        #     trajectory=trajectory,
        #     tol=tol
        # )

        best_trajectory = trajectory
        while (len(best_trajectory.burns) < max_burns):
            
            if self._check_optimality(trajectory):
                break
            
            guess_trajectory, index = self._add_midcourse_correction(initial_state, final_state, best_trajectory)

            if index < 0:
                break
            
            for i in range(len(guess_trajectory.segments) - 2):
                guess_trajectory = self._optimize_midcourse_correction(initial_state, final_state, guess_trajectory, i, tol)
            
            self._log(f"After midcourse burn: {guess_trajectory.total_delta_v:.4f} km/s")

            if guess_trajectory.total_delta_v >= best_trajectory.total_delta_v:
                self._log(f"Burn addition did not improve cost, ending optimization...")
                break

            best_trajectory = guess_trajectory

            if len(best_trajectory.burns) == max_burns:
                self._log(f"Maximum burns ({max_burns}) reached, ending optimization...")
        
        self._log("=== Optimization complete ===")
        self._log(f"Final solution: {len(best_trajectory.burns)} burns, ΔV: {best_trajectory.total_delta_v:.4f} km/s")
        
        return best_trajectory