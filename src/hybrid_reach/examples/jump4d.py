"""Planar jumper: stance, ballistic flight, and absorbing terrain collision."""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

import hj_reachability as hj
from hybrid_reach.model import HybridModel, Mode, ProjectionStage, Transition


DOMAIN_LOWER = (-.1, -.1, -10., -10.)
DOMAIN_UPPER = (4.1, 1.9, 10., 10.)
PAPER_SHAPE = (51, 31, 21, 21)
SOURCE_SHAPE = (43, 21, 21, 21)


@dataclass(frozen=True)
class JumpParameters:
    mass: float = 1.
    gravity: float = 9.81
    leg_length: float = .5
    force_max: tuple[float, float] = (30., 30.)
    platform_left: float = 3.
    platform_right: float = 4.
    platform_height: float = 1.
    target_center: tuple[float, float] = (3.5, 1.5)
    target_radius: float = .25
    terrain_closed: bool = True

    def __post_init__(self):
        if self.mass <= 0 or self.gravity <= 0 or self.leg_length <= 0:
            raise ValueError("Mass, gravity, and leg length must be positive")
        if len(self.force_max) != 2 or any(u < 0 for u in self.force_max):
            raise ValueError("Expected two nonnegative force limits")
        if self.platform_right <= self.platform_left or self.target_radius <= 0:
            raise ValueError("Invalid platform or target geometry")


def parameters_for(variant):
    if variant == 'paper':
        return JumpParameters()
    if variant == 'matlab':
        return JumpParameters(force_max=(15., 30.), terrain_closed=False)
    raise ValueError(f"Unknown Jump4D variant: {variant}")


class JumpDynamics(hj.Dynamics):
    """Forces in stance; no control in flight or after freezing.

    The paper variant bounds dissipation over the full admissible input box.
    Only the MATLAB comparison variant uses HelperOC's gradient-endpoint bound.
    Both choose an admissible zero force at a Hamiltonian tie.
    """
    def __init__(self, mode, parameters=None, *, source_dissipation=False, control_mode='min'):
        if mode not in ('stance', 'flight', 'frozen'):
            raise ValueError(f"Unknown jumper mode: {mode}")
        self.mode = mode
        self.parameters = parameters or JumpParameters()
        self.source_dissipation = source_dissipation
        upper = self.parameters.force_max if mode == 'stance' else (0., 0.)
        if control_mode not in ('min', 'max'):
            raise ValueError("Control mode must be min or max")
        super().__init__(control_mode, 'max' if control_mode == 'min' else 'min', hj.sets.Box(jnp.zeros(2), jnp.array(upper)),
                         hj.sets.Box(jnp.zeros(0), jnp.zeros(0)))

    def __call__(self, state, control, disturbance, time):
        if self.mode == 'frozen':
            return jnp.zeros(4, dtype=state.dtype)
        p = self.parameters
        ax = control[0]/p.mass if self.mode == 'stance' else 0.
        az = control[1]/p.mass-p.gravity if self.mode == 'stance' else -p.gravity
        return jnp.array([state[2], state[3], ax, az])

    def optimal_control_and_disturbance(self, state, time, grad_value):
        upper = self.control_space.hi
        active = grad_value[2:] < 0 if self.control_mode == 'min' else grad_value[2:] > 0
        return jnp.where(active, upper, 0.), jnp.zeros(0)

    def hamiltonian(self, state, time, value, grad_value):
        u, d = self.optimal_control_and_disturbance(state, time, grad_value)
        flow = self(state, u, d, time)
        result = jnp.array(0., dtype=grad_value.dtype)
        for dim in range(4):
            result = result+grad_value[dim]*flow[dim]
        return result

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        if self.mode == 'frozen':
            return jnp.zeros(4, dtype=state.dtype)
        if self.source_dissipation:
            ulo, d = self.optimal_control_and_disturbance(state, time, grad_value_box.lo)
            uhi, _ = self.optimal_control_and_disturbance(state, time, grad_value_box.hi)
            return jnp.maximum(jnp.abs(self(state, ulo, d, time)),
                               jnp.abs(self(state, uhi, d, time)))
        p = self.parameters
        ax = p.force_max[0]/p.mass if self.mode == 'stance' else 0.
        az = max(p.gravity, abs(p.force_max[1]/p.mass-p.gravity)) if self.mode == 'stance' else p.gravity
        return jnp.array([jnp.abs(state[2]), jnp.abs(state[3]), ax, az])


def transition_guards(states, variant='paper', parameters=None):
    """Physical guards for paper; literal builder guards for source comparison."""
    p = parameters or parameters_for(variant)
    if variant not in ('paper', 'matlab'):
        raise ValueError(f"Unknown Jump4D variant: {variant}")
    # Avoid changing an intended boundary decision because a grid coordinate
    # represents .5 as .5000000000000001. This tolerance has no cell-sized buffer.
    tol = 1e-12 if variant == 'paper' else 0.
    x, z = states[..., 0], states[..., 1]
    if p.terrain_closed:
        platform = (x >= p.platform_left-tol) & (x <= p.platform_right+tol) & (z <= p.platform_height+tol)
        terrain = (z <= tol) | platform
    else:
        platform = (x > p.platform_left+tol) & (x < p.platform_right-tol) & (z < p.platform_height-tol)
        terrain = (z < -tol) | platform
    stance_collision = terrain | ((z > 1.5*p.leg_length) if variant == 'matlab' else False)
    return {
        'q12': z > p.leg_length+tol,
        'q21': z < p.leg_length-tol,
        'q13': stance_collision,
        'q23': terrain,
    }


def make_model(grid, variant='paper', *, parameters=None, reset_coordinates=None):
    p = parameters or parameters_for(variant)
    modes = tuple(Mode(name, JumpDynamics(name, p, source_dissipation=variant == 'matlab'))
                  for name in ('stance', 'flight', 'frozen'))
    if variant == 'matlab':
        if reset_coordinates is None:
            reset_coordinates = tuple(np.arange(n)*float(dx)+lo
                                      for n, dx, lo in zip(grid.shape, grid.spacings, np.asarray(grid.domain.lo)))
        source = np.stack(np.meshgrid(*reset_coordinates, indexing='ij'), axis=-1)
        # All continuous resets are identity in the source, but a reconstructed
        # coordinate -> ceil round trip can shift an axis by one grid sample.
        reset = lambda states: source
        sampling = 'ceil'
    else:
        source = np.asarray(grid.states)
        reset = lambda states: states
        sampling = 'nearest'
    guards = transition_guards(source, variant, p)
    edges = tuple(Transition(a, b, lambda states, mask=guards[name]: mask, reset, sampling)
                  for a, b, name in ((0, 1, 'q12'), (1, 0, 'q21'), (0, 2, 'q13'), (1, 2, 'q23')))
    if variant == 'matlab':
        # Literal source order, including composition of ceil round trips.
        stages = tuple(ProjectionStage('forced', (edge,)) for edge in edges)
    else:
        # Mode exchanges read one snapshot. Collision has explicit priority.
        stages = (ProjectionStage('forced', edges[:2]), ProjectionStage('forced', edges[2:]))
    return HybridModel(modes, stages)


def make_problem(shape=PAPER_SHAPE, variant='paper', *, parameters=None,
                 coordinates=None, reset_coordinates=None,
                 domain_lower=DOMAIN_LOWER, domain_upper=DOMAIN_UPPER):
    shape = tuple(shape)
    if len(shape) != 4 or any(n < 4 for n in shape):
        raise ValueError("Jump4D needs four grid axes with at least four samples")
    domain_lower,domain_upper=tuple(domain_lower),tuple(domain_upper)
    if len(domain_lower)!=4 or len(domain_upper)!=4 or any(
            not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo for lo,hi in zip(domain_lower,domain_upper)):
        raise ValueError('Expected four finite increasing domain bounds')
    p = parameters or parameters_for(variant)
    vectors = tuple(np.asarray(v) for v in coordinates) if coordinates is not None else tuple(
        np.linspace(lo, hi, n) for lo, hi, n in zip(domain_lower, domain_upper, shape))
    if tuple(len(v) for v in vectors) != shape:
        raise ValueError("Coordinates do not match grid shape")
    spacings = tuple((hi-lo)/(n-1) for lo, hi, n in zip(domain_lower, domain_upper, shape))
    for vector, n, dx, lo, hi in zip(vectors, shape, spacings, domain_lower, domain_upper):
        if not np.isfinite(vector).all() or not np.allclose(vector, np.linspace(lo, hi, n), atol=1e-12, rtol=0):
            raise ValueError("Expected uniform coordinates over the declared domain")
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array(domain_lower), jnp.array(domain_upper)), shape)
    vectors = tuple(jnp.asarray(v) for v in vectors)
    grid = grid.replace(coordinate_vectors=vectors, spacings=spacings,
                        states=jnp.stack(jnp.meshgrid(*vectors, indexing='ij'), axis=-1))
    target_position = jnp.linalg.norm(grid.states[..., :2]-jnp.array(p.target_center), axis=-1)-p.target_radius
    target = jnp.broadcast_to(target_position, (3,)+shape)
    return make_model(grid, variant, parameters=p, reset_coordinates=reset_coordinates), grid, target
