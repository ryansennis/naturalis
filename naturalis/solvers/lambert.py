from naturalis.dynamics.orbit import OrbitalState, Segment, Trajectory, Burn
from numpy.linalg import norm
from typing import List, Tuple
from enum import Enum

import numpy as np


class BaseLambertSolver():
    def __init__(
        self,
        mu: float,
        debug: bool = False
    ) -> None:
        """
        Initialize Lambert solver.
        
        Args:
            mu (float): Gravitational parameter (km^3/s^2).
            debug (bool): Enable debug logging.
        """
        self.mu = mu
        self.debug = debug

    def solve(
        self: 'BaseLambertSolver',
        r1: OrbitalState,
        r2: OrbitalState
    ) -> Trajectory:
        return Trajectory([Segment(r1, r2)], [])

class IzzoLambertSolver(BaseLambertSolver):
    def __init__(
            self,
            mu: float,
            debug: bool = False
        ) -> None:
            """
            Initialize Izzo Lambert solver.
            
            Args:
                mu (float): Gravitational parameter (km^3/s^2).
                debug (bool): Enable debug logging.
            """
            super().__init__(mu, debug)

    def _log(
        self,
        msg: str
    ) -> None:
        """Log debug message if debug mode is enabled."""
        if self.debug:
            print(f"[Lambert] {msg}")

    def _initial_guess(
        self,
        T: float,
        l: float,
        M: int
    ) -> float:
        """
        Generate initial guess for x parameter.
        
        Args:
            T (float): Normalized time of flight.
            l (float): Geometry parameter.
            M (int): Number of revolutions.
            
        Returns:
            x_0 (float): Initial guess for x.
        """
        if M == 0:
            # Single revolution
            T_0 = np.arccos(l) + l * np.sqrt(1 - l**2) + M * np.pi  # Equation 19
            T_1 = 2 * (1 - l**3) / 3  # Equation 21

            if T >= T_0:
                x_0 = (T_0 / T) ** (2 / 3) - 1
            elif T < T_1:
                x_0 = 5 / 2 * T_1 / T * (T_1 - T) / (1 - l**5) + 1
            else:
                x_0 = np.exp(np.log(2) * np.log(T / T_0) / np.log(T_1 / T_0)) - 1

            return x_0

        else:
            # Multiple revolution
            x_0l = (((M * np.pi + np.pi) / (8 * T)) ** (2 / 3) - 1) / (((M * np.pi + np.pi) / (8 * T)) ** (2 / 3) + 1)
            x_0r = (((8 * T) / (M * np.pi)) ** (2 / 3) - 1) / (((8 * T) / (M * np.pi)) ** (2 / 3) + 1)

            # Filter out the solution
            x_0 = np.min(np.array([x_0l, x_0r]))

            return x_0
    
    def _hyp2f1b(
        self,
        x: float
    ) -> float:
        """
        Hypergeometric function 2F1(3, 1, 5/2, x).
        
        Used for near-parabolic orbits where x is close to 1.
        See Battin's "An Introduction to the Mathematics and Methods of Astrodynamics".
        
        Args:
            x (float): Argument of hypergeometric function.
            
        Returns:
            result (float): Value of 2F1(3, 1, 5/2, x).
        """
        if x >= 1.0:
            return np.inf
        else:
            res = 1.0
            term = 1.0
            ii = 0
            while True:
                term = term * (3 + ii) * (1 + ii) / (5 / 2 + ii) * x / (ii + 1)
                res_old = res
                res += term
                if res_old == res:
                    return res
                ii += 1

    def _compute_psi(
        self,
        x: float,
        y: float,
        l: float
    ) -> float:
        """
        Compute psi parameter based on orbit type.
        
        Args:
            x (float): x parameter.
            y (float): y parameter.
            l (float): Geometry parameter.
            
        Returns:
            psi (float): Computed psi value.
        """
        if -1 <= x < 1:
            # Elliptic motion
            return np.arccos(x * y + l * (1 - x**2))
        elif x > 1:
            # Hyperbolic motion
            return np.arcsinh((y - x * l) * np.sqrt(x**2 - 1))
        else:
            # Parabolic motion
            return 0.0
    
    def _tof_equation_y(
        self,
        x: float,
        y: float,
        T0: float,
        l: float,
        M: int
    ) -> float:
        """
        Time of flight equation (function of y).
        
        Args:
            x (float): x parameter.
            y (float): y parameter.
            T0 (float): Target normalized time of flight.
            l (float): Geometry parameter.
            M (int): Number of revolutions.
            
        Returns:
            residual (float): Difference between computed and target TOF.
        """
        if M == 0 and np.sqrt(0.6) < x < np.sqrt(1.4):
            # Use hypergeometric function for near-parabolic case
            eta = y - l * x
            S_1 = (1 - l - x * eta) * 0.5
            Q = 4 / 3 * self._hyp2f1b(S_1)
            T_ = (eta**3 * Q + 4 * l * eta) * 0.5
        else:
            psi = self._compute_psi(x, y, l)
            T_ = np.divide(
                np.divide(psi + M * np.pi, np.sqrt(np.abs(1 - x**2))) - x + l * y,
                (1 - x**2),
            )

        return T_ - T0

    def _compute_T_min(
        self,
        l: float,
        M: int,
        max_iter: int,
        atol: float,
        rtol: float
    ) -> float:
        """
        Compute minimum possible transfer time for M revolutions.
        
        Args:
            l (float): Geometry parameter.
            M (int): Number of revolutions.
            max_iter (int): Maximum iterations for convergence.
            atol (float): Absolute tolerance.
            rtol (float): Relative tolerance.
            
        Returns:
            T_min (float): Minimum time of flight.
        """
        if l == 1:
            x_T_min = 0.0
            T_min = self._tof_equation(x_T_min, 0.0, l, M)
        else:
            if M == 0:
                T_min = 0.0
            else:
                x_i = 0.1
                T_i = self._tof_equation(x_i, 0.0, l, M)
                x_T_min = self._halley(x_i, T_i, l, max_iter, atol, rtol)
                T_min = self._tof_equation(x_T_min, 0.0, l, M)
            
        return T_min
        
    def _tof_equation_x(
        self,
        x: float,
        y: float,
        T: float,
        l: float
    ) -> float:
        """First derivative of TOF equation with respect to x."""
        return (3 * T * x - 2 + 2 * l**3 * x / y) / (1 - x**2)

    def _tof_equation_x2(
        self,
        x: float,
        y: float,
        T: float,
        dT: float,
        l: float
    ) -> float:
        """Second derivative of TOF equation with respect to x."""
        return (3 * T + 5 * x * dT + 2 * (1 - l**2) * l**3 / y**3) / (1 - x**2)

    def _tof_equation_x3(
        self,
        x: float,
        y: float,
        T: float,
        dT: float,
        ddT: float,
        l: float
    ) -> float:
        """Third derivative of TOF equation with respect to x."""
        return (7 * x * ddT + 8 * dT - 6 * (1 - l**2) * l**5 * x / y**5) / (1 - x**2)
        
    def _halley(
        self,
        x0: float,
        T0: float,
        l: float,
        max_iter: int,
        atol: float,
        rtol: float
    ):
        """
        Find a minimum of time of flight equation using Halley's method.
        
        Halley's method is a root-finding algorithm that uses first, second,
        and third derivatives for cubic convergence.
        """
        for _ in range(max_iter):
            y = self._compute_y(x0, l)
            fder = self._tof_equation_x(x0, y, T0, l)
            fder2 = self._tof_equation_x2(x0, y, T0, fder, l)
            if fder2 == 0:
                raise RuntimeError("Derivative was zero")
            fder3 = self._tof_equation_x3(x0, y, T0, fder, fder2, l)

            x = x0 - 2 * fder * fder2 / (2 * fder2**2 - fder * fder3)

            if abs(x - x0) < rtol * np.abs(x0) + atol:
                return x
            
            x0 = x
        
        raise RuntimeError("Halley did not converge!")

    def _householder(
        self,
        x0: float,
        T0: float,
        l: float,
        M: int,
        max_iter: int,
        atol: float,
        rtol: float
    ) -> float:
        """
        Find a zero of time of flight equation using Householder's method.
        
        Householder's method is a generalization of Newton's method with
        improved convergence properties.
        """
        for _ in range(max_iter):
            y = self._compute_y(x0, l)
            fval = self._tof_equation_y(x0, y, T0, l, M)
            T = fval + T0
            fder = self._tof_equation_x(x0, y, T, l)
            fder2 = self._tof_equation_x2(x0, y, T, fder, l)
            fder3 = self._tof_equation_x3(x0, y, T, fder, fder2, l)

            x = x0 - fval * ((fder**2 - fval * fder2 / 2) / (fder * (fder**2 - fval * fder2) + fder3 * fval**2 / 6))

            if abs(x - x0) < rtol * np.abs(x0) + atol:
                return x
            
            x0 = x

        raise RuntimeError("Householder did not converge!")
    
    def _tof_equation(
        self,
        x: float,
        T0: float,
        l: float,
        M: int
    ):
        """Time of flight equation."""
        return self._tof_equation_y(x, self._compute_y(x, l), T0, l, M)
    
    def _compute_y(
        self,
        x: float,
        l: float
    ):
        """Compute y parameter from x and l."""
        return np.sqrt(1 - l**2 * (1 - x**2))

    def _find_xy(
        self,
        l: float,
        T: float,
        M: int,
        max_iter: int,
        atol: float,
        rtol: float
    ) -> Tuple[List[float], List[float]]:
        """
        Compute all x, y pairs for single and multiple revolution Lambert problems.
        
        Returns both left and right branch solutions for multi-revolution cases.
        """
        assert abs(l) < 1

        M_max = np.floor(T / np.pi)
        T_00 = np.arccos(l) + l * np.sqrt(1 - l**2)

        # Refine maximum number of revolutions if necessary
        if T < T_00 + M_max * np.pi and M_max > 0:
            T_min = self._compute_T_min(l, M_max, max_iter, atol, rtol)
            if T_min > T:
                M_max = M_max - 1

        # Check if requested revolutions is possible
        if M > M_max:
            raise ValueError(f"The requested number of revolutions M={M} exceeds the maximum possible M_max={M_max}")

        x_list, y_list = [], []

        # Single revolution case
        if M == 0:
            x_0 = self._initial_guess(T, l, M)
            x = self._householder(x_0, T, l, M, max_iter, atol, rtol)
            y = self._compute_y(x, l)
            x_list.append(x)
            y_list.append(y)
        else:
            # Multi-revolution case - try both left and right branch solutions
            # Left branch solution
            x_0l = self._initial_guess(T, l, M)
            try:
                x_l = self._householder(x_0l, T, l, M, max_iter, atol, rtol)
                y_l = self._compute_y(x_l, l)
                x_list.append(x_l)
                y_list.append(y_l)
            except RuntimeError as e:
                self._log(f"Left branch solution failed: {e}")

            # Right branch solution
            x_0r = (((8 * T) / (M * np.pi)) ** (2 / 3) - 1) / (((8 * T) / (M * np.pi)) ** (2 / 3) + 1)
            
            if abs(x_0r - x_0l) > 0.1:  # Only if initial guesses are sufficiently different
                try:
                    x_r = self._householder(x_0r, T, l, M, max_iter, atol, rtol)
                    y_r = self._compute_y(x_r, l)
                    
                    # Check if this is not a duplicate of the left branch solution
                    if len(x_list) == 0 or abs(x_r - x_list[0]) > rtol:
                        x_list.append(x_r)
                        y_list.append(y_r)
                except RuntimeError as e:
                    self._log(f"Right branch solution failed: {e}")

        if len(x_list) == 0:
            raise ValueError(f"No solutions found for M={M} and T={T}")

        return x_list, y_list
    
    def _reconstruct(
        self,
        x: float,
        y: float,
        r1: float,
        r2: float,
        l: float,
        gamma: float,
        rho: float,
        sigma: float
    ) -> Tuple[float, float, float, float]:
        """
        Reconstruct solution velocity vectors from x, y parameters.
        
        Returns radial and tangential velocity components at r1 and r2.
        """
        V_r1 = gamma * ((l * y - x) - rho * (l * y + x)) / r1
        V_r2 = -gamma * ((l * y - x) + rho * (l * y + x)) / r2
        V_t1 = gamma * sigma * (y + l * x) / r1
        V_t2 = gamma * sigma * (y + l * x) / r2
        return V_r1, V_r2, V_t1, V_t2

    def solve(
        self,
        r1: OrbitalState,
        r2: OrbitalState,
        M: int = 0,
        prograde: bool = True,
        max_iter: int = 50,
        atol: float = 1e-5,
        rtol: float = 1e-7
    ) -> Trajectory:
        """
        Solve Lambert's problem using Izzo's algorithm.

        Args:
            r1 (OrbitalState): Initial orbital state.
            r2 (OrbitalState): Final orbital state.
            M (int): Number of revolutions. Must be >= 0.
            prograde (bool): If True, prograde motion. Otherwise, retrograde.
            max_iter (int): Maximum number of iterations.
            atol (float): Absolute tolerance.
            rtol (float): Relative tolerance.

        Returns:
            trajectory (Trajectory): The trajectory corresponding to the Lambert solution.
        """
        tof = r2.time - r1.time
        if tof <= 0:
            raise ValueError("Time of flight must be positive")

        c = r2.position - r1.position
        c_norm, r1_norm, r2_norm = norm(c), norm(r1.position), norm(r2.position)

        r1_unit = r1.position / norm(r1.position)
        r2_unit = r2.position / norm(r2.position)
        dot_product = np.dot(r1_unit, r2_unit)
        
        # Handle near 180° transfer case
        if abs(dot_product) > 1.0 - rtol:
            if dot_product < 0:
                # Nearly 180° transfer (anti-parallel vectors)
                self._log("Detected nearly 180° transfer - applying perturbation")
                
                # Choose a perturbation direction perpendicular to both vectors
                perturbation = np.cross(r1_unit, [0, 0, 1])
                if norm(perturbation) < 1e-10:
                    # If r1_unit is parallel to [0, 0, 1], try a different vector
                    perturbation = np.cross(r1_unit, [1, 0, 0])
                
                perturbation = perturbation / norm(perturbation) * 1e-8 * r2_norm
                
                # Create a copy of r2 with the perturbed position
                perturbed_r2 = OrbitalState(
                    mu=r2.mu,
                    time=r2.time,
                    position=r2.position + perturbation,
                    velocity=r2.velocity.copy()
                )
                
                # Use the perturbed state for calculations
                r2 = perturbed_r2
                # Recalculate needed values
                c = r2.position - r1.position
                c_norm = norm(c)
                r2_norm = norm(r2.position)
                r2_unit = r2.position / r2_norm

        s = (r1_norm + r2_norm + c_norm) * 0.5

        i_r1, i_r2 = r1.position / r1_norm, r2.position / r2_norm
        i_h = np.cross(i_r1, i_r2)
        i_h = i_h / norm(i_h)

        l = np.sqrt(1 - min(1.0, c_norm / s))

        if i_h[2] < 0:
            l = -l
            i_t1, i_t2 = np.cross(i_r1, i_h), np.cross(i_r2, i_h)
        else:
            i_t1, i_t2 = np.cross(i_h, i_r1), np.cross(i_h, i_r2)

        l, i_t1, i_t2 = (l, i_t1, i_t2) if prograde else (-l, -i_t1, -i_t2)

        T = np.sqrt(2 * self.mu / s**3) * tof

        x_list, y_list = self._find_xy(l, T, M, max_iter, atol, rtol)

        gamma = np.sqrt(self.mu * s / 2)
        rho = (r1_norm - r2_norm) / c_norm
        sigma = np.sqrt(1 - rho**2)

        trajectories: List[Trajectory] = []

        for x, y in zip(x_list, y_list):
            V_r1, V_r2, V_t1, V_t2 = self._reconstruct(x, y, float(r1_norm), float(r2_norm), l, gamma, float(rho), sigma)

            v1 = V_r1 * (r1.position / r1_norm) + V_t1 * i_t1
            v2 = V_r2 * (r2.position / r2_norm) + V_t2 * i_t2

            initial_state = OrbitalState(self.mu, r1.time, r1.position, v1)
            final_state = OrbitalState(self.mu, r2.time, r2.position, v2)
            segment = Segment(initial_state, final_state)

            delta_v1 = v1 - r1.velocity
            delta_v2 = r2.velocity - v2

            initial_burn = Burn(r1.time, delta_v1, r1.position)
            final_burn = Burn(r2.time, delta_v2, r2.position)

            trajectory = Trajectory([segment], [initial_burn, final_burn])
            trajectories.append(trajectory)

        return min(trajectories, key=lambda t: t.total_delta_v)

class LambertSolverType(Enum):
    IZZO = IzzoLambertSolver