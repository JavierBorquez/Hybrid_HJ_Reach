"""Dog1D modes, guards, resets, target and reference projection order."""
import jax.numpy as jnp
import numpy as np

import hj_reachability as hj
from hybrid_reach.model import HybridModel, Mode, ProjectionStage, Transition


class ForwardMotion(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, speed):
        self.speed = float(speed)
        super().__init__("min", "max", hj.sets.Box(jnp.array([0.]), jnp.array([1.])),
                         hj.sets.Box(jnp.zeros(1), jnp.zeros(1)))

    def open_loop_dynamics(self, state, time):
        return jnp.zeros(1)

    def control_jacobian(self, state, time):
        return jnp.array([[self.speed]])

    def disturbance_jacobian(self, state, time):
        return jnp.zeros((1, 1))


def identity(states):
    return states


def everywhere(states):
    return np.ones(states.shape[:-1], dtype=bool)


def under_table(states):
    return (states[..., 0] >= 5) & (states[..., 0] <= 6)


def make_model(variant="absorbing"):
    if variant not in ("absorbing", "recoverable", "continuous_only"):
        raise ValueError(f"Unknown Dog1D variant: {variant}")
    modes = tuple(Mode(name, ForwardMotion(speed)) for name, speed in
                  (("walking", 3), ("crawling", 1), ("frozen", 0)))
    forced = ProjectionStage("forced", (Transition(0, 2, under_table, identity),))
    controlled = [Transition(0, 1, everywhere, identity), Transition(1, 0, everywhere, identity)]
    if variant == "recoverable":
        controlled.append(Transition(2, 1, everywhere, identity))
    # Match the source ordering, including optional walking->crawling eligibility
    # inside the table guard. Recovery adds one edge to the same simultaneous stage.
    stages = (forced,) if variant == "continuous_only" else (forced, ProjectionStage("controlled", tuple(controlled)))
    return HybridModel(modes, stages)


def make_problem(points=301, variant="absorbing", *, coordinates=None):
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([-1.]), jnp.array([11.])), (points,))
    if coordinates is not None:
        # Accept reference coordinates to eliminate representation differences in
        # guards at x=5 and x=6. Retain the same uniform spacing/domain metadata.
        x = jnp.asarray(coordinates)
        if x.shape != (points,) or not np.allclose(np.asarray(x), np.linspace(-1, 11, points), atol=1e-13, rtol=0):
            raise ValueError("Expected uniform reference coordinates on [-1, 11]")
        grid = grid.replace(coordinate_vectors=(x,), states=x[:, None])
    target = jnp.broadcast_to(jnp.abs(grid.states[:, 0] - 9.5) - 0.5, (3, points))
    return make_model(variant), grid, target
