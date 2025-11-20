from naturalis.dynamics.orbit import OrbitalState, Segment, Trajectory
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go

def _add_earth_to_plot(figure: go.Figure) -> None:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)

    a_earth = 6378
    b_earth = 6357

    x_earth = a_earth * np.outer(np.cos(u), np.sin(v))
    y_earth = b_earth * np.outer(np.sin(u), np.sin(v))
    z_earth = b_earth * np.outer(np.ones(np.size(u)), np.cos(v))

    figure.add_trace(
        go.Surface(
            x=x_earth,
            y=y_earth,
            z=z_earth,
            name='Earth',
            surfacecolor=np.zeros_like(x_earth),
            colorscale='Blues',
            showscale=False,
            showlegend=True,
            hoverinfo='none'
        )
    )

    a_atm = a_earth + 90
    b_atm = b_earth + 90

    x_atm = a_atm * np.outer(np.cos(u), np.sin(v))
    y_atm = b_atm * np.outer(np.sin(u), np.sin(v))
    z_atm = b_atm * np.outer(np.ones(np.size(u)), np.cos(v))

    figure.add_trace(
        go.Surface(
            x=x_atm,
            y=y_atm,
            z=z_atm,
            name='Atmosphere',
            surfacecolor=np.zeros_like(x_earth),
            colorscale='Greys',
            opacity=0.2,
            showscale=False,
            showlegend=True,
            hoverinfo='none'
        )
    )

def plot_orbital_states(states: List[OrbitalState], figure: go.Figure) -> None:

    _add_earth_to_plot(figure)

    for state in states:
        figure.add_trace(
            go.Scatter3d(
                x=[state.position[0]],
                y=[state.position[1]],
                z=[state.position[2]],
                showlegend=False,
                hovertemplate=(
                    f"Time: {state.time}<br>"
                    f"Position: [{state.position[0]:.3f}, {state.position[1]:.3f}, {state.position[2]:.3f}] km<br>"
                    f"Velocity: [{state.velocity[0]:.3f}, {state.velocity[1]:.3f}, {state.velocity[2]:.3f}] km/s"
                )
            )
        )