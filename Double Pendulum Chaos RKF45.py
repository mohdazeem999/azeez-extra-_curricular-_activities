"""
Adaptive RKF45 simulation and animation of a planar double pendulum.

State vector:
    y = [theta1, omega1, theta2, omega2]

Angles are measured from the downward vertical.
All quantities use SI units.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import matplotlib

# This must appear before importing matplotlib.pyplot.
# TkAgg opens a separate interactive animation window in a normal VS Code run.
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray


State = NDArray[np.float64]


@dataclass(frozen=True)
class PendulumParameters:
    """Physical parameters of the double pendulum."""

    gravity: float = 9.81
    length1: float = 1.0
    length2: float = 1.0
    mass1: float = 1.0
    mass2: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.gravity,
            self.length1,
            self.length2,
            self.mass1,
            self.mass2,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("All physical parameters must be positive.")


@dataclass(frozen=True)
class SolverOptions:
    """Numerical settings for the adaptive RKF45 integrator."""

    absolute_tolerance: float = 1e-9
    relative_tolerance: float = 1e-7
    minimum_step: float = 1e-8
    maximum_step: float = 0.05
    safety_factor: float = 0.9
    minimum_factor: float = 0.1
    maximum_factor: float = 4.0

    def __post_init__(self) -> None:
        if self.absolute_tolerance <= 0.0:
            raise ValueError("absolute_tolerance must be positive.")
        if self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive.")
        if self.minimum_step <= 0.0:
            raise ValueError("minimum_step must be positive.")
        if self.maximum_step < self.minimum_step:
            raise ValueError("maximum_step must be at least minimum_step.")
        if not 0.0 < self.safety_factor <= 1.0:
            raise ValueError("safety_factor must lie in (0, 1].")
        if self.minimum_factor <= 0.0:
            raise ValueError("minimum_factor must be positive.")
        if self.maximum_factor < self.minimum_factor:
            raise ValueError("maximum_factor must exceed minimum_factor.")


@dataclass(frozen=True)
class StepResult:
    """Result of one attempted RKF45 step."""

    state: State
    next_step: float
    accepted: bool
    normalized_error: float


@dataclass(frozen=True)
class AdvanceResult:
    """Result of advancing the simulation through a time interval."""

    state: State
    next_step: float
    accepted_steps: int
    rejected_steps: int


PARAMETERS = PendulumParameters()
SOLVER_OPTIONS = SolverOptions()

INITIAL_STATE: State = np.array(
    [
        np.radians(120.0),
        0.0,
        np.radians(120.01),
        0.0,
    ],
    dtype=np.float64,
)


def validate_state(state: State) -> State:
    """Validate and return a four-component finite state vector."""
    array = np.asarray(state, dtype=np.float64)

    if array.shape != (4,):
        raise ValueError(
            "State must have exactly four values: "
            "[theta1, omega1, theta2, omega2]."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError("State must contain only finite numerical values.")

    return array


def _derivatives_unchecked(
    state: State,
    parameters: PendulumParameters,
) -> State:
    """Evaluate the double-pendulum differential equations."""
    theta1, omega1, theta2, omega2 = state

    g = parameters.gravity
    l1 = parameters.length1
    l2 = parameters.length2
    m1 = parameters.mass1
    m2 = parameters.mass2

    angle_difference = theta2 - theta1

    denominator = (
        2.0 * m1
        + m2
        - m2 * np.cos(2.0 * angle_difference)
    )

    acceleration1 = (
        -g * (2.0 * m1 + m2) * np.sin(theta1)
        -m2 * g * np.sin(theta1 - 2.0 * theta2)
        -2.0
        * m2
        * np.sin(angle_difference)
        * (
            omega2**2 * l2
            + omega1**2 * l1 * np.cos(angle_difference)
        )
    ) / (l1 * denominator)

    acceleration2 = (
        2.0
        * np.sin(angle_difference)
        * (
            omega1**2 * l1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta1)
            + omega2**2 * l2 * m2 * np.cos(angle_difference)
        )
    ) / (l2 * denominator)

    return np.array(
        [omega1, acceleration1, omega2, acceleration2],
        dtype=np.float64,
    )


def derivatives(
    state: State,
    parameters: PendulumParameters = PARAMETERS,
) -> State:
    """Return d/dt[theta1, omega1, theta2, omega2]."""
    validated_state = validate_state(state)
    return _derivatives_unchecked(validated_state, parameters)


def total_energy(
    state: State,
    parameters: PendulumParameters = PARAMETERS,
) -> float:
    """Return the total mechanical energy of the double pendulum."""
    theta1, omega1, theta2, omega2 = validate_state(state)

    m1 = parameters.mass1
    m2 = parameters.mass2
    l1 = parameters.length1
    l2 = parameters.length2
    g = parameters.gravity

    kinetic_energy = (
        0.5 * (m1 + m2) * l1**2 * omega1**2
        + 0.5 * m2 * l2**2 * omega2**2
        + m2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2)
    )

    potential_energy = (
        -(m1 + m2) * g * l1 * np.cos(theta1)
        -m2 * g * l2 * np.cos(theta2)
    )

    return float(kinetic_energy + potential_energy)


def rkf45_step(
    state: State,
    step: float,
    parameters: PendulumParameters = PARAMETERS,
    options: SolverOptions = SOLVER_OPTIONS,
) -> StepResult:
    """
    Attempt one adaptive Runge-Kutta-Fehlberg 4(5) integration step.

    A fourth-order and fifth-order estimate are calculated. Their scaled
    difference estimates local truncation error.
    """
    if step <= 0.0:
        raise ValueError("step must be positive.")

    y = validate_state(state)
    h = float(step)

    k1 = h * _derivatives_unchecked(y, parameters)

    k2 = h * _derivatives_unchecked(
        y + k1 / 4.0,
        parameters,
    )

    k3 = h * _derivatives_unchecked(
        y + 3.0 * k1 / 32.0 + 9.0 * k2 / 32.0,
        parameters,
    )

    k4 = h * _derivatives_unchecked(
        y
        + 1932.0 * k1 / 2197.0
        - 7200.0 * k2 / 2197.0
        + 7296.0 * k3 / 2197.0,
        parameters,
    )

    k5 = h * _derivatives_unchecked(
        y
        + 439.0 * k1 / 216.0
        - 8.0 * k2
        + 3680.0 * k3 / 513.0
        - 845.0 * k4 / 4104.0,
        parameters,
    )

    k6 = h * _derivatives_unchecked(
        y
        - 8.0 * k1 / 27.0
        + 2.0 * k2
        - 3544.0 * k3 / 2565.0
        + 1859.0 * k4 / 4104.0
        - 11.0 * k5 / 40.0,
        parameters,
    )

    fourth_order = (
        y
        + 25.0 * k1 / 216.0
        + 1408.0 * k3 / 2565.0
        + 2197.0 * k4 / 4104.0
        - k5 / 5.0
    )

    fifth_order = (
        y
        + 16.0 * k1 / 135.0
        + 6656.0 * k3 / 12825.0
        + 28561.0 * k4 / 56430.0
        - 9.0 * k5 / 50.0
        + 2.0 * k6 / 55.0
    )

    scale = (
        options.absolute_tolerance
        + options.relative_tolerance
        * np.maximum(np.abs(y), np.abs(fifth_order))
    )

    normalized_error = float(
        np.max(np.abs(fifth_order - fourth_order) / scale)
    )

    accepted = normalized_error <= 1.0

    if normalized_error == 0.0:
        factor = options.maximum_factor
    else:
        factor = options.safety_factor * normalized_error ** (-0.2)

    factor = float(
        np.clip(
            factor,
            options.minimum_factor,
            options.maximum_factor,
        )
    )

    next_step = float(
        np.clip(
            h * factor,
            options.minimum_step,
            options.maximum_step,
        )
    )

    if not accepted and h <= options.minimum_step:
        raise RuntimeError(
            "RKF45 reached the minimum step size "
            "without satisfying the requested tolerance."
        )

    next_state = fifth_order if accepted else y.copy()

    return StepResult(
        state=next_state,
        next_step=next_step,
        accepted=accepted,
        normalized_error=normalized_error,
    )


def advance(
    state: State,
    duration: float,
    parameters: PendulumParameters = PARAMETERS,
    *,
    step: float = 0.01,
    options: SolverOptions = SOLVER_OPTIONS,
) -> AdvanceResult:
    """Advance the pendulum state through exactly the requested duration."""
    if duration < 0.0:
        raise ValueError("duration must not be negative.")
    if step <= 0.0:
        raise ValueError("step must be positive.")

    current_state = validate_state(state).copy()
    current_step = float(
        np.clip(step, options.minimum_step, options.maximum_step)
    )

    elapsed = 0.0
    accepted_steps = 0
    rejected_steps = 0

    while elapsed < duration:
        remaining_time = duration - elapsed
        trial_step = min(current_step, remaining_time)

        result = rkf45_step(
            current_state,
            trial_step,
            parameters,
            options,
        )

        current_step = result.next_step

        if result.accepted:
            current_state = result.state
            elapsed += trial_step
            accepted_steps += 1
        else:
            rejected_steps += 1

    return AdvanceResult(
        state=current_state,
        next_step=current_step,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
    )


def get_positions(
    state: State,
    parameters: PendulumParameters = PARAMETERS,
) -> tuple[float, float, float, float]:
    """Convert angular coordinates into Cartesian bob positions."""
    theta1, _, theta2, _ = validate_state(state)

    x1 = parameters.length1 * np.sin(theta1)
    y1 = -parameters.length1 * np.cos(theta1)

    x2 = x1 + parameters.length2 * np.sin(theta2)
    y2 = y1 - parameters.length2 * np.cos(theta2)

    return float(x1), float(y1), float(x2), float(y2)


def main() -> None:
    """Run the interactive double-pendulum animation."""
    backend = matplotlib.get_backend().lower()

    non_interactive_backends = {
        "agg",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
    }

    if backend in non_interactive_backends:
        raise RuntimeError(
            f"Matplotlib is using the non-interactive '{backend}' backend. "
            "Run this script in a desktop Python environment with Tk support."
        )

    state = INITIAL_STATE.copy()
    step = 0.01
    simulation_time = 0.0

    frame_duration = 0.02
    max_trail_length = 1_000

    trail_x: deque[float] = deque(maxlen=max_trail_length)
    trail_y: deque[float] = deque(maxlen=max_trail_length)

    initial_energy = total_energy(state)

    figure, axis = plt.subplots(figsize=(8, 8))

    figure.patch.set_facecolor("#1c1c1d")
    axis.set_facecolor("#080909")

    axis.set_xlim(-2.5, 2.5)
    axis.set_ylim(-2.5, 2.5)
    axis.set_aspect("equal", adjustable="box")

    axis.set_xlabel("x position (m)", color="lightgray")
    axis.set_ylabel("y position (m)", color="lightgray")
    axis.tick_params(colors="gray")

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#444444")
    axis.spines["bottom"].set_color("#444444")

    axis.plot(
        0.0,
        0.0,
        marker="o",
        color="#d69e16",
        markersize=8,
        zorder=5,
    )

    rod1, = axis.plot(
        [],
        [],
        color="white",
        alpha=0.6,
        linewidth=2.2,
        zorder=2,
    )

    rod2, = axis.plot(
        [],
        [],
        color="white",
        alpha=0.6,
        linewidth=2.2,
        zorder=2,
    )

    bob1, = axis.plot(
        [],
        [],
        marker="o",
        color="#4af0c0",
        markersize=12,
        zorder=4,
    )

    bob2, = axis.plot(
        [],
        [],
        marker="o",
        color="#f05070",
        markersize=12,
        zorder=4,
    )

    trail, = axis.plot(
        [],
        [],
        color="#f05070",
        alpha=0.35,
        linewidth=1.0,
        zorder=1,
    )

    title = axis.set_title("", color="white", pad=15)

    def initialise_animation():
        """Create the first visible frame."""
        rod1.set_data([], [])
        rod2.set_data([], [])
        bob1.set_data([], [])
        bob2.set_data([], [])
        trail.set_data([], [])
        title.set_text("Adaptive RKF45 double pendulum")

        return rod1, rod2, bob1, bob2, trail, title

    def update(_frame_number: int):
        """Advance the physics and redraw the pendulum."""
        nonlocal state, step, simulation_time

        result = advance(
            state,
            frame_duration,
            parameters=PARAMETERS,
            step=step,
            options=SOLVER_OPTIONS,
        )

        state = result.state
        step = result.next_step
        simulation_time += frame_duration

        x1, y1, x2, y2 = get_positions(state)

        rod1.set_data([0.0, x1], [0.0, y1])
        rod2.set_data([x1, x2], [y1, y2])

        bob1.set_data([x1], [y1])
        bob2.set_data([x2], [y2])

        trail_x.append(x2)
        trail_y.append(y2)
        trail.set_data(list(trail_x), list(trail_y))

        current_energy = total_energy(state)
        relative_energy_drift = (
            (current_energy - initial_energy)
            / max(abs(initial_energy), 1.0)
        )

        title.set_text(
            "Adaptive RKF45 Double Pendulum\n"
            f"t = {simulation_time:6.2f} s   "
            f"theta1 = {np.degrees(state[0]):7.2f} deg   "
            f"E = {current_energy:9.5f} J   "
            f"dE/E0 = {relative_energy_drift:+.2e}"
        )

        return rod1, rod2, bob1, bob2, trail, title

    animation = FuncAnimation(
        figure,
        update,
        init_func=initialise_animation,
        interval=20,
        blit=True,
        cache_frame_data=False,
    )

    # Keeps the animation object alive until the window closes.
    _ = animation

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
