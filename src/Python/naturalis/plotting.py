"""
Plotting utilities for orbital trajectories.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from mpl_toolkits.mplot3d import Axes3D


def plot_trajectory_3d(
    trajectory: List[Tuple[np.ndarray, np.ndarray]],
    show_earth: bool = True,
    earth_radius: float = 6378.137,  # Earth radius in km
    title: str = "Orbital Trajectory",
    ax: Optional[Axes3D] = None
) -> Axes3D:
    """
    Plot 3D orbital trajectory.
    
    Args:
        trajectory: List of (r, v) tuples where r and v are 3D numpy arrays
        show_earth: Whether to display Earth as a sphere at origin
        earth_radius: Radius of Earth sphere in km (default: 6378.137)
        title: Plot title
        ax: Optional existing 3D axes (if None, creates new figure)
        
    Returns:
        The 3D axes object
    """
    # Extract position vectors from trajectory
    positions = np.array([r for r, v in trajectory])
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1.5, label='Trajectory')
    
    # Mark start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               color='green', s=50, marker='o', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               color='red', s=50, marker='s', label='End')
    
    # Plot Earth as sphere if requested
    if show_earth:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = earth_radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = earth_radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='blue')
    
    # Set labels and title
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # Set equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                         positions[:, 1].max() - positions[:, 1].min(),
                         positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return ax


def plot_trajectory_2d(
    trajectory: List[Tuple[np.ndarray, np.ndarray]],
    plane: str = "xy",
    show_earth: bool = True,
    earth_radius: float = 6378.137,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot 2D projection of orbital trajectory.
    
    Args:
        trajectory: List of (r, v) tuples where r and v are 3D numpy arrays
        plane: Which plane to project onto ("xy", "xz", or "yz")
        show_earth: Whether to display Earth as a circle at origin
        earth_radius: Radius of Earth circle in km
        title: Plot title (if None, auto-generated from plane)
        ax: Optional existing axes (if None, creates new figure)
        
    Returns:
        The axes object
    """
    # Extract position vectors
    positions = np.array([r for r, v in trajectory])
    
    # Select coordinates based on plane
    plane = plane.lower()
    if plane == "xy":
        x_data = positions[:, 0]
        y_data = positions[:, 1]
        x_label, y_label = "X (km)", "Y (km)"
    elif plane == "xz":
        x_data = positions[:, 0]
        y_data = positions[:, 2]
        x_label, y_label = "X (km)", "Z (km)"
    elif plane == "yz":
        x_data = positions[:, 1]
        y_data = positions[:, 2]
        x_label, y_label = "Y (km)", "Z (km)"
    else:
        raise ValueError(f"Plane must be 'xy', 'xz', or 'yz', got '{plane}'")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot trajectory
    ax.plot(x_data, y_data, 'b-', linewidth=1.5, label='Trajectory')
    
    # Mark start and end points
    ax.scatter(x_data[0], y_data[0], color='green', s=50, marker='o', label='Start', zorder=3)
    ax.scatter(x_data[-1], y_data[-1], color='red', s=50, marker='s', label='End', zorder=3)
    
    # Plot Earth as circle if requested
    if show_earth:
        circle = plt.Circle((0, 0), earth_radius, color='blue', alpha=0.3, zorder=1)
        ax.add_patch(circle)
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is None:
        title = f"Trajectory Projection ({plane.upper()} plane)"
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    return ax


def plot_trajectory_all_planes(
    trajectory: List[Tuple[np.ndarray, np.ndarray]],
    show_earth: bool = True,
    earth_radius: float = 6378.137,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot trajectory in all three 2D projection planes side by side.
    
    Args:
        trajectory: List of (r, v) tuples
        show_earth: Whether to display Earth
        earth_radius: Radius of Earth in km
        figsize: Figure size tuple
        
    Returns:
        The figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for i, plane in enumerate(["xy", "xz", "yz"]):
        plot_trajectory_2d(trajectory, plane=plane, show_earth=show_earth, 
                          earth_radius=earth_radius, ax=axes[i])
    
    plt.tight_layout()
    return fig
