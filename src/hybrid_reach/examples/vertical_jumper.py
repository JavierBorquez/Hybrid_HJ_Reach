"""Vertical stance/flight jumper, state (height, vertical velocity), force in N."""
from dataclasses import dataclass
import hj_reachability as hj
import jax.numpy as jnp
import numpy as np
from .. import HybridModel, Mode, ProjectionStage, Transition, GuardEvent


@dataclass(frozen=True)
class Parameters:
    mass: float = 1.
    gravity: float = 9.81
    leg_length: float = .5
    force_max: float = 30.
    ceiling: float = 2.8
    goal_height: float = 1.1
    goal_radius: float = .1


class VerticalDynamics(hj.Dynamics):
    def __init__(self, mode, parameters):
        self.mode, self.p = mode, parameters
        hi = parameters.force_max if mode == "stance" else 0.
        super().__init__("min", "max", hj.sets.Box(jnp.zeros(1), jnp.array([hi])),
                         hj.sets.Box(jnp.zeros(0), jnp.zeros(0)))

    def __call__(self, state, control, disturbance, time):
        return jnp.array([state[1], control[0]/self.p.mass-self.p.gravity])

    def optimal_control_and_disturbance(self, state, time, grad_value):
        active = grad_value[1] > 0 if self.control_mode == "max" else grad_value[1] < 0
        return jnp.where(active, self.control_space.hi, 0.), jnp.zeros(0)

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        g = self.p.gravity
        return jnp.array([abs(state[1]), jnp.maximum(g, abs(self.control_space.hi[0]/self.p.mass-g))])


def failure(states, mode, p=Parameters()):
    return jnp.minimum(states[..., 0], p.ceiling-states[..., 0])


def goal(states, mode, p=Parameters()):
    return abs(states[..., 0]-p.goal_height)-p.goal_radius


def make_system(shape=(81, 81), parameters=Parameters()):
    p = parameters
    if min(p.mass, p.gravity, p.leg_length, p.force_max, p.goal_radius) <= 0:
        raise ValueError("Physical parameters must be positive")
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([-.15, -8.]), jnp.array([3., 8.])), shape)
    def launch(x):
        return (x[..., 0] > p.leg_length+1e-10) | ((abs(x[..., 0]-p.leg_length) <= 1e-10) & (x[..., 1] > 0))
    def land(x):
        return (x[..., 0] < p.leg_length-1e-10) | ((abs(x[..., 0]-p.leg_length) <= 1e-10) & (x[..., 1] <= 0))
    identity = lambda x: x
    model = HybridModel(tuple(Mode(q, VerticalDynamics(q, p)) for q in ("stance", "flight")),
        (ProjectionStage("forced", (
            Transition(0, 1, launch, identity, name="launch"),
            Transition(1, 0, land, identity, name="land"))),))
    events = (GuardEvent("launch", lambda x: x[0]-p.leg_length, (0,), 1),
              GuardEvent("land", lambda x: x[0]-p.leg_length, (1,), -1))
    return model, grid, events
