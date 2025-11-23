from naturalis.dynamics.orbit import OrbitalState, OrbitalParameters, Segment, Trajectory, Burn
from naturalis.dynamics.propagator import OrbitalPropagator
from naturalis.solvers.lambert import LambertSolution
from naturalis.solvers.n_burn import NBurnSolution
from typing import List, Optional
from numpy.linalg import norm

import numpy as np
import plotly.graph_objects as go

def _plot_atmosphere(
    figure: go.Figure,
    semi_axes: List[float],
    height: float
) -> None:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)

    a_atm = semi_axes[0] + height
    b_atm = semi_axes[1] + height
    c_atm = semi_axes[2] + height

    x_atm = a_atm * np.outer(np.cos(u), np.sin(v))
    y_atm = b_atm * np.outer(np.sin(u), np.sin(v))
    z_atm = c_atm * np.outer(np.ones(np.size(u)), np.cos(v))

    figure.add_trace(
        go.Surface(
            x=x_atm,
            y=y_atm,
            z=z_atm,
            name='Atmosphere',
            surfacecolor=np.zeros_like(x_atm),
            colorscale='Greys',
            opacity=0.2,
            legendgroup="bodies",
            legendgrouptitle_text="Celestial Bodies",
            showscale=False,
            showlegend=True,
            hoverinfo='none'
        )
    )

def plot_earth(
    figure: go.Figure,
    atmosphere: bool
) -> None:
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
            legendgroup="bodies",
            legendgrouptitle_text="Celestial Bodies",
            hoverinfo='none'
        )
    )

    if atmosphere:
        _plot_atmosphere(figure, [a_earth, b_earth, b_earth], 90)

def plot_orbital_state(
    figure: go.Figure,
    state: OrbitalState,
    name: str,
    color: str
) -> None:
    figure.add_trace(
        go.Scatter3d(
            x=[state.position[0]],
            y=[state.position[1]],
            z=[state.position[2]],
            showlegend=True,
            mode="markers",
            marker=dict(size=3, color=color),
            name=name,
            legendgroup="states",
            legendgrouptitle_text="Orbital States",
            hovertemplate=(
                f"<b>Time</b>: {state.time} s<br>"
                f"<b>Position</b>: [{state.position[0]:.3f}, {state.position[1]:.3f}, {state.position[2]:.3f}] km<br>"
                f"<b>Velocity</b>: [{state.velocity[0]:.3f}, {state.velocity[1]:.3f}, {state.velocity[2]:.3f}] km/s"
                "<extra></extra>"
            )
        )
    )

def plot_orbital_states(
    figure: go.Figure,
    states: List[OrbitalState]
) -> None:
    for i, state in enumerate(states):
        figure.add_trace(
            go.Scatter3d(
                x=[state.position[0]],
                y=[state.position[1]],
                z=[state.position[2]],
                showlegend=True,
                mode="markers",
                marker=dict(size=7),
                name=f"State {i + 1}",
                legendgroup="states",
                legendgrouptitle_text="Orbital States",
                hovertemplate=(
                    f"<b>Time</b>: {state.time} s<br>"
                    f"<b>Position</b>: [{state.position[0]:.3f}, {state.position[1]:.3f}, {state.position[2]:.3f}] km<br>"
                    f"<b>Velocity</b>: [{state.velocity[0]:.3f}, {state.velocity[1]:.3f}, {state.velocity[2]:.3f}] km/s"
                    "<extra></extra>"
                )
            )
        )

def plot_segment(
    figure: go.Figure,
    segment: Segment,
    propagator: OrbitalPropagator,
    name: str,
    color: Optional[str] = None,
    legendgroup: str = "segments",
    legendgrouptitle_text: str = "Segments"
) -> None:
    states = propagator.propagate_segment(segment)

    t = [state.time for state in states]
    x = [state.position[0] for state in states]
    y = [state.position[1] for state in states]
    z = [state.position[2] for state in states]

    v_x = [state.velocity[0] for state in states]
    v_y = [state.velocity[1] for state in states]
    v_z = [state.velocity[2] for state in states]

    figure.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            customdata=np.stack((t, v_x, v_y, v_z), axis=-1),
            mode="lines",
            marker=dict(size=3, color=color),
            showlegend=True,
            name=name,
            legendgroup=legendgroup,
            legendgrouptitle_text=legendgrouptitle_text,
            hovertemplate=(
                "<b>Time</b>: %{customdata[0]} s<br>"
                "<b>Position</b>: [%{x:.3f}, %{y:.3f}, %{z:.3f}] km<br>"
                "<b>Velocity</b>: [%{customdata[1]:.3f}, %{customdata[2]:.3f}, %{customdata[3]:.3f}] km/s"
                "<extra></extra>"
            )
        )
    )

def plot_segments(
    figure: go.Figure,
    segments: List[Segment],
    propagator: OrbitalPropagator
) -> None:
    for i, segment in enumerate(segments):
        plot_segment(
            figure,
            segment,
            propagator,
            f'Segment {i + 1}'
        )

def plot_burn(
    figure: go.Figure,
    burn: Burn,
    name: Optional[str] = None,
    showlegend: Optional[bool] = False
) -> None:
    if burn.position is not None:
        figure.add_trace(
            go.Scatter3d(
                x=[burn.position[0]],
                y=[burn.position[1]],
                z=[burn.position[2]],
                showlegend=showlegend,
                mode="markers",
                marker=dict(size=3, color='darkorange'),
                name=name,
                legendgroup="burns",
                legendgrouptitle_text="Burns",
                hovertemplate=(
                    f"<b>Time</b>: {burn.time} s<br>"
                    f"<b>Position</b>: [{burn.position[0]:.3f}, {burn.position[1]:.3f}, {burn.position[2]:.3f}] km<br>"
                    f"<b>Delta-V</b>: [{burn.delta_v[0]:.3f}, {burn.delta_v[1]:.3f}, {burn.delta_v[2]:.3f}] km/s"
                    "<extra></extra>"
                )
            )
        )

def plot_burns(
    figure: go.Figure,
    burns: List[Burn]
) -> None:
    for i, burn in enumerate(burns):
        plot_burn(
            figure=figure,
            burn=burn,
            name=f"Burns" if i == 0 else None,
            showlegend=True if i == 0 else False
        )

def plot_coast(
    figure: go.Figure,
    coast: Segment,
    name: str,
    color: str,
    propagator: OrbitalPropagator
) -> None:
    states = propagator.propagate_segment(coast)

    t = [state.time for state in states]
    x = [state.position[0] for state in states]
    y = [state.position[1] for state in states]
    z = [state.position[2] for state in states]

    v_x = [state.velocity[0] for state in states]
    v_y = [state.velocity[1] for state in states]
    v_z = [state.velocity[2] for state in states]

    figure.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            customdata=np.stack((t, v_x, v_y, v_z), axis=-1),
            mode="lines",
            marker=dict(size=7, color=color),
            showlegend=True,
            name=name,
            legendgroup="coasts",
            legendgrouptitle_text="Coasts",
            hovertemplate=(
                "<b>Time</b>: %{customdata[0]} s<br>"
                "<b>Position</b>: [%{x:.3f}, %{y:.3f}, %{z:.3f}] km<br>"
                "<b>Velocity</b>: [%{customdata[1]:.3f}, %{customdata[2]:.3f}, %{customdata[3]:.3f}] km/s"
                "<extra></extra>"
            )
        )
    )

def plot_terminal_coasts(
    figure: go.Figure,
    initial_coast: Optional[Segment],
    final_coast: Optional[Segment],
    propagator: OrbitalPropagator
) -> None:
    if initial_coast is not None:
        plot_coast(
            figure=figure,
            coast=initial_coast,
            name="Initial Coast",
            color="blue",
            propagator=propagator
        )

        plot_orbital_state(
            figure=figure,
            state=initial_coast.initial_state,
            name="Initial State",
            color="green"
        )

    if final_coast is not None:
        plot_coast(
            figure=figure,
            coast=final_coast,
            name="Final Coast",
            color="purple",
            propagator=propagator
        )

        plot_orbital_state(
            figure=figure,
            state=final_coast.final_state,
            name="Final State",
            color="red"
        )


def plot_trajectory(
    figure: go.Figure,
    trajectory: Trajectory,
    propagator: OrbitalPropagator
) -> None:
    plot_burns(
        figure=figure,
        burns=trajectory.burns
    )

    plot_segments(
        figure=figure,
        segments=trajectory.segments,
        propagator=propagator
    )

    plot_terminal_coasts(
        figure=figure,
        initial_coast=trajectory.initial_coast,
        final_coast=trajectory.final_coast,
        propagator=propagator
    )

    states: List[OrbitalState] = []
    for segment in trajectory.segments:
        states += propagator.propagate_segment(segment=segment)

    max_radius = norm(max(states, key=lambda state: norm(state.position)).position)
    
    figure.update_layout(
        scene=dict(
            aspectmode="cube",
            xaxis=dict(range=[-max_radius, max_radius]),
            yaxis=dict(range=[-max_radius, max_radius]),
            zaxis=dict(range=[-max_radius, max_radius])
        ),
        autosize=True
    )

def plot_orbit(
    figure: go.Figure,
    state: OrbitalState,
    propagator: OrbitalPropagator,
    name: str,
    color: str
) -> None:
    orbital_period = OrbitalParameters.from_state(state).period

    final_state = propagator.propagate_state_by_time(
        state,
        orbital_period
    )

    orbit = Segment(state, final_state)

    plot_segment(
        figure,
        orbit,
        propagator,
        name,
        color,
        "orbits",
        "Orbits"
    )

def plot_solution(
    figure: go.Figure,
    solution: LambertSolution,
    propagator: OrbitalPropagator
) -> None:
    plot_trajectory(
        figure=figure,
        trajectory=solution.trajectory,
        propagator=propagator
    )

    plot_orbit(
        figure=figure,
        state=solution.initial_state,
        propagator=propagator,
        name="Initial Orbit",
        color="blue"
    )

    plot_orbit(
        figure=figure,
        state=solution.final_state,
        propagator=propagator,
        name="Final Orbit",
        color="purple"
    )

    solver_type = "Lambert" if type(solution) is LambertSolution else "N-Burn"
    plot_title = f"{solver_type} Solution Plot - {solution.trajectory.total_delta_v:.3f} km/s"

    figure.update_layout(
        title=dict(
            text=plot_title
        )
    )