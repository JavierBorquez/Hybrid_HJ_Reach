"""A point piece that can steer, must drop, and finally comes to rest.

The tutorial uses this model to teach dynamics, Hamiltonian control, guards,
reset maps and geometry. All example choices live here, outside the solver.
"""
from dataclasses import dataclass

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np

from ..model import HybridModel, Mode, ProjectionStage, Transition
from ..problems import Problem
from ..simulation import GuardEvent


STEER, DROP, REST = 0, 1, 2
MODE_NAMES = ("steer", "drop", "rest")


@dataclass(frozen=True)
class Parameters:
    y_high: float = 4.
    y_low: float = 1.
    u_max: float = 1.
    slow_speed: float = 1.
    fast_speed: float = 5.
    guard_tolerance: float = 1e-10
    x_bounds: tuple = (-3., 3.)
    y_bounds: tuple = (0., 7.)
    # Rectangles are (left, right, bottom, top).
    target: tuple = (.6, 1.2, 1.4, 1.7)
    obstacles: tuple = ((-2.2, -1.6, 3., 3.35),
                        (.3, .9, 2.7, 3.05),
                        (1.8, 2.4, 1.9, 2.25))

    def __post_init__(self):
        numbers = (*self.x_bounds, *self.y_bounds, self.y_low, self.y_high,
                   self.u_max, self.slow_speed, self.fast_speed, self.guard_tolerance)
        if not np.isfinite(numbers).all() or min(self.u_max, self.slow_speed, self.fast_speed) <= 0:
            raise ValueError("Bounds must be finite and speeds positive")
        if self.guard_tolerance < 0 or self.guard_tolerance > 1e-6*(self.y_high-self.y_low):
            raise ValueError("Guard tolerance must be nonnegative and small relative to the band")
        if not self.x_bounds[0] < self.x_bounds[1]:
            raise ValueError("Expected increasing horizontal bounds")
        if not self.y_bounds[0] < self.y_low < self.y_high < self.y_bounds[1]:
            raise ValueError("Switching heights must lie strictly inside the domain")
        for left, right, bottom, top in (self.target, *self.obstacles):
            if not (np.isfinite((left, right, bottom, top)).all()
                    and self.x_bounds[0] < left < right < self.x_bounds[1]
                    and self.y_low < bottom < top < self.y_high):
                raise ValueError("Tutorial rectangles must lie inside the drop band")


class FallingDynamics(hj.Dynamics):
    def __init__(self, mode, parameters=Parameters()):
        if mode not in (STEER, DROP, REST):
            raise ValueError("Unknown falling-piece mode")
        self.mode, self.p = mode, parameters
        self.horizontal_speed = parameters.u_max if mode == STEER else 0.
        self.fall_speed = (parameters.slow_speed, parameters.fast_speed, 0.)[mode]
        control_space = hj.sets.Box(jnp.array([-self.horizontal_speed]),
                                    jnp.array([self.horizontal_speed]))
        disturbance_space = hj.sets.Box(jnp.zeros(0), jnp.zeros(0))
        super().__init__("min", "max", control_space, disturbance_space)

    def __call__(self, state, control, disturbance, time):
        vx = control[0] if self.mode == STEER else 0.
        return jnp.array([vx, -self.fall_speed])

    def optimal_control_and_disturbance(self, state, time, grad_value):
        # H = Vx*u - Vy*fall_speed. Only Vx*u depends on the control.
        direction = -1. if self.control_mode == "min" else 1.
        control = direction*self.horizontal_speed*jnp.sign(grad_value[0])
        return jnp.array([control]), jnp.zeros(0)

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        # Maximum absolute coordinate speeds over the entire input box.
        return jnp.array([self.horizontal_speed, self.fall_speed])


def rectangle(states, bounds):
    """Negative inside a rectangle, zero on its boundary; supports batches."""
    left, right, bottom, top = bounds
    x, y = states[..., 0], states[..., 1]
    return jnp.maximum(jnp.maximum(left-x, x-right),
                       jnp.maximum(bottom-y, y-top))


def goal(states, mode, parameters=Parameters()):
    return rectangle(states, parameters.target)


def failure(states, mode, parameters=Parameters()):
    # Side-wall contact is failure; the rest band and its floor are safe.
    x = states[..., 0]
    field = jnp.minimum(x-parameters.x_bounds[0], parameters.x_bounds[1]-x)
    for bounds in parameters.obstacles:
        field = jnp.minimum(field, rectangle(states, bounds))
    return field


def mode_at(state, parameters=Parameters()):
    """The physical band owns its lower boundary in the destination mode."""
    y = np.asarray(state)[..., 1]
    return np.where(y > parameters.y_high, STEER,
                    np.where(y > parameters.y_low, DROP, REST))


def make_system(shape=(101, 141), parameters=Parameters()):
    p = parameters
    if len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 5 for n in shape):
        raise ValueError("Expected two grid axes with at least five points")
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([p.x_bounds[0], p.y_bounds[0]]),
                    jnp.array([p.x_bounds[1], p.y_bounds[1]])), shape)
    modes = tuple(Mode(name, FallingDynamics(q, p)) for q, name in enumerate(MODE_NAMES))
    # Guard membership has explicit height slack, covering ODE accumulation near
    # a feedback endpoint. No state or prediction is moved/clipped by this test.
    tolerance = p.guard_tolerance
    identity = lambda states: states
    steer_to_drop = Transition(STEER, DROP,
        lambda states: states[..., 1] <= p.y_high+tolerance, identity, name="lose_steering")
    drop_to_rest = Transition(DROP, REST,
        lambda states: states[..., 1] <= p.y_low+tolerance, identity, name="come_to_rest")
    # Bellman composition is downstream-first: rest -> drop values, then steer.
    # Online paths traverse these stages in reverse, so both resets also work
    # for a supplied source mode below both guards, without implicit closure.
    model = HybridModel(modes, (ProjectionStage("forced", (drop_to_rest,)),
                               ProjectionStage("forced", (steer_to_drop,))))
    events = (GuardEvent("lose_steering", lambda state: state[1]-p.y_high, (STEER,), -1),
              GuardEvent("come_to_rest", lambda state: state[1]-p.y_low, (DROP,), -1))
    return model, grid, events


def make_problem(kind, shape=(101, 141), parameters=Parameters()):
    model, grid, events = make_system(shape, parameters)
    g = lambda states, q: goal(states, q, parameters)
    f = lambda states, q: failure(states, q, parameters)
    if kind == "reach":
        problem = Problem.reach(model, grid, g)
    elif kind == "avoid":
        problem = Problem.avoid(model, grid, f)
    elif kind == "reach_avoid":
        problem = Problem.reach_avoid(model, grid, g, f)
    else:
        raise ValueError("Expected reach, avoid or reach_avoid")
    return problem, events


def physical_flow(state, mode, control, disturbance=None, time=0., parameters=Parameters()):
    """NumPy forward flow, kept independent of JAX Hamiltonian evaluation."""
    speed = (parameters.slow_speed, parameters.fast_speed, 0.)[mode]
    return np.array([float(control[0]) if mode == STEER else 0., -speed])


def nominal_control(state, mode, parameters=Parameters()):
    """An arbitrary admissible policy: drift right while steering is available."""
    return np.array([.8*parameters.u_max if mode == STEER else 0.])


def audit_trajectory(trajectory, parameters=Parameters(), *, check_failure=True):
    """Exact segment replay and wall/rectangle contact for constant controls.

    Each rectangle field along a line is a maximum of four affine functions.
    Its minimum occurs at an endpoint or an intersection of two of these lines.
    Taking the minimum across rectangles/walls gives the segment failure field.
    """
    states, times = trajectory["state"], trajectory["time"]
    endpoints = trajectory["segment_end_state"]
    max_error, clearance = 0., np.inf
    for i, control in enumerate(trajectory["control"]):
        start, stop = states[i], endpoints[i]
        dt, mode = times[i+1]-times[i], int(trajectory["mode"][i])
        expected = start+dt*physical_flow(start, mode, control, parameters=parameters)
        max_error = max(max_error, float(np.max(abs(expected-stop))))
        limit = parameters.u_max if mode == STEER else 0.
        if np.any(abs(control) > limit+1e-12):
            raise AssertionError("Inadmissible serialized tutorial control")
        if check_failure:
            clearance = min(clearance, float(min(start[0]-parameters.x_bounds[0],
                parameters.x_bounds[1]-start[0], stop[0]-parameters.x_bounds[0],
                parameters.x_bounds[1]-stop[0])))
            for left, right, bottom, top in parameters.obstacles:
                a = np.array([left-start[0], start[0]-right, bottom-start[1], start[1]-top])
                delta = stop-start
                b = np.array([-delta[0], delta[0], -delta[1], delta[1]])
                candidates = [0., 1.]
                for j in range(4):
                    for k in range(j):
                        if b[j] != b[k]:
                            root = (a[k]-a[j])/(b[j]-b[k])
                            if 0 < root < 1:
                                candidates.append(float(root))
                clearance = min(clearance, min(float(np.max(a+t*b)) for t in candidates))
    # Preserve the terminal collision/root even when it ends before a flow sample.
    if check_failure:
        clearance = min(clearance, float(failure(np.asarray(trajectory["final_state"]),
                                                  trajectory["final_mode"], parameters)))
    return dict(physical_replay_max_error=max_error,
                minimum_failure_field=None if not check_failure else float(clearance))
