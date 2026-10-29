"""IJRR planar jumper avoid tube: two modes, floor and two pillars."""
from dataclasses import dataclass

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np

from hybrid_reach.model import HybridModel, Mode, ProjectionStage, Transition
from .jump4d import JumpDynamics, JumpParameters


@dataclass(frozen=True)
class FilterJumpParameters:
    mass: float = 1.
    gravity: float = 9.81
    leg_length: float = .5
    force_max: tuple[float, float] = (30., 30.)
    pillars: tuple[tuple[float, float, float], ...] = ((2., 2.4, .6), (4.5, 4.9, .6))

    def __post_init__(self):
        if min(self.mass, self.gravity, self.leg_length) <= 0:
            raise ValueError('Mass, gravity, and leg length must be positive')
        if len(self.force_max) != 2 or any(u < 0 for u in self.force_max):
            raise ValueError('Expected two nonnegative force limits')
        if any(right <= left or height <= 0 for left, right, height in self.pillars):
            raise ValueError('Invalid pillar geometry')


def failure_value(states, parameters=None):
    """Negative inside failure, zero at closed contact, positive outside.

    A minimum of implicit floor/rectangle functions represents their union.
    This is a level-set representation, not a Euclidean distance everywhere.
    """
    p = parameters or FilterJumpParameters()
    x, z = states[..., 0], states[..., 1]
    value = z
    for left, right, height in p.pillars:
        rectangle = jnp.maximum(jnp.maximum(left-x, x-right), z-height)
        value = jnp.minimum(value, rectangle)
    return value


def transition_guards(states, parameters=None):
    """One guard definition shared by grid projection and online simulation."""
    p = parameters or FilterJumpParameters()
    z,vz = states[...,1],states[...,3]
    boundary = abs(z-p.leg_length) <= 1e-12
    return {'launch':(z > p.leg_length+1e-12) | (boundary & (vz > 0)),
            'land':(z < p.leg_length-1e-12) | (boundary & (vz <= 0))}


def resolve_mode(state, mode, parameters=None):
    if mode not in (0,1): raise ValueError('Expected stance or flight mode')
    guards = transition_guards(np.asarray(state),parameters)
    return 1-mode if bool(guards['launch' if mode==0 else 'land']) else mode


def make_problem(shape=(66, 26, 23, 31), *, parameters=None,
                 domain_lower=(0., -.2, -.4, -6.), domain_upper=(6.5, 2.3, 8.4, 6.),
                 control_mode='max'):
    p = parameters or FilterJumpParameters()
    shape = tuple(shape)
    if len(shape) != 4 or any(n < 4 for n in shape):
        raise ValueError('Expected four axes with at least four points each')
    if len(domain_lower) != 4 or len(domain_upper) != 4 or any(
            not np.isfinite(a+b) or b <= a for a,b in zip(domain_lower,domain_upper)):
        raise ValueError('Expected four finite increasing domain bounds')
    coordinates = tuple(np.linspace(a,b,n) for a,b,n in zip(domain_lower,domain_upper,shape))
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array(domain_lower),jnp.array(domain_upper)),shape)
    vectors = tuple(jnp.asarray(v) for v in coordinates)
    grid = grid.replace(coordinate_vectors=vectors,
        spacings=tuple((b-a)/(n-1) for a,b,n in zip(domain_lower,domain_upper,shape)),
        states=jnp.stack(jnp.meshgrid(*vectors,indexing='ij'),axis=-1))
    dynamics_parameters = JumpParameters(mass=p.mass,gravity=p.gravity,
        leg_length=p.leg_length,force_max=p.force_max)
    modes = tuple(Mode(name,JumpDynamics(name,dynamics_parameters,control_mode=control_mode))
                  for name in ('stance','flight'))
    # Identity resets have an exact index-preserving gather. At the shared
    # boundary, velocity resolves departure direction to agree with rollouts.
    model = HybridModel(modes,(ProjectionStage('forced',(
        Transition(0,1,lambda s:transition_guards(s,p)['launch'],lambda s:s),
        Transition(1,0,lambda s:transition_guards(s,p)['land'],lambda s:s)),
        reduction='max'),))
    failure = jnp.broadcast_to(failure_value(grid.states,p),(2,)+shape)
    return model,grid,failure
