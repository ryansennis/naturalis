from dataclasses import dataclass
from numpy.typing import NDArray
from typing import List, Optional, Union

import numpy as np

@dataclass
class OrbitalState:
    """
    Seven-element state for objects in orbit around a planet:
        mu = standard gravitational parameter (km^3 s^-2)
        time = state time (s)
        position = state position ([km, km, km])
        velocity = state velocity ([km/s, km/s, km/s])
    """
    mu: float
    time: float
    position: NDArray
    velocity: NDArray

    def __post_init__(self: 'OrbitalState'):
        assert self.mu >= 0
        assert self.time >= 0
        assert len(self.position) == 3
        assert len(self.velocity) == 3

        self.position = np.array(self.position)
        self.velocity = np.array(self.velocity)
    
    def copy(self: 'OrbitalState') -> 'OrbitalState':
        return OrbitalState(self.mu, self.time, self.position.copy(), self.velocity.copy())
    
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

@dataclass
class Burn:
    """Represents an impulsive maneuver (instantaneous change in velocity)."""
    time: float
    delta_v: NDArray
    position: NDArray
    
    def __post_init__(self):
        """Validate the vectors after initialization."""
        self.delta_v = np.array(self.delta_v)
        self.position = np.array(self.position)
            
    @property
    def magnitude(self) -> float:
        """Get the magnitude of the burn's delta-v vector."""
        return float(np.linalg.norm(self.delta_v))
    
    @property
    def direction(self) -> NDArray:
        """Get the unit direction of the burn's delta-v vector."""
        return self.delta_v / self.magnitude
    
    def copy(self) -> 'Burn':
        """Get a deep copy of this `Burn` object."""
        return Burn(self.time, self.delta_v.copy(), self.position.copy())

@dataclass
class Segment:
    """Represents a trajectory segment between burn points."""
    initial_state: OrbitalState
    final_state: OrbitalState

    def copy(self) -> 'Segment':
        """Get a deep copy of this Segment."""
        return Segment(self.initial_state.copy(), self.final_state.copy())

    @property
    def duration(self) -> float:
        """Return the time duration of this segment in seconds."""
        return self.final_state.time - self.initial_state.time
        
@dataclass 
class Trajectory:
    """Represents a complete trajectory with multiple segments and burns."""
    segments: List[Segment]
    burns: List[Burn]
    initial_coast: Optional[Segment] = None
    final_coast: Optional[Segment] = None
                
    @property 
    def total_delta_v(self) -> float:
        """Calculate total delta-v across all burns."""
        dv = sum(burn.magnitude for burn in self.burns)
        return dv

    @property
    def duration(self) -> float:
        """Return the time duration of this trajectory in seconds."""
        return self.segments[-1].final_state.time - self.segments[0].initial_state.time
    
    def copy(self) -> 'Trajectory':
        """Get a deep copy of this Trajectory."""
        segments = [segment.copy() for segment in self.segments]
        burns = [burn.copy() for burn in self.burns]
        return Trajectory(
            segments,
            burns,
            self.initial_coast.copy() if self.initial_coast is not None else None
        )
    
    def get_segment_containing_time(
        self,
        time: float
    ) -> Union[Segment, None]:
        """
        Find the segment that contains the given time. Does not look at terminal coasts.
        
        Args:
            time (float): Time to search for.
            
        Returns:
            segment (Segment | None): The segment containing the time, or None if not found.
        """
        for segment in self.segments:
            if segment.initial_state.time <= time <= segment.final_state.time:
                return segment
        
        return None