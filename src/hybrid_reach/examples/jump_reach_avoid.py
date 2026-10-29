"""IJRR two-mode jumper: reach a goal disk while avoiding floor and pillars."""
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
from .jump_filter import FilterJumpParameters,failure_value,make_problem as make_jump_problem


@dataclass(frozen=True)
class ReachAvoidJumpParameters(FilterJumpParameters):
    target_center: tuple[float,float] = (6.,.5)
    target_radius: float = .25

    def __post_init__(self):
        super().__post_init__()
        if len(self.target_center)!=2 or not np.isfinite(self.target_center).all():
            raise ValueError('Expected a finite planar target center')
        if not np.isfinite(self.target_radius) or self.target_radius<=0:
            raise ValueError('Expected a positive finite target radius')


def goal_value(states,parameters=None):
    p=parameters or ReachAvoidJumpParameters()
    return jnp.linalg.norm(states[...,:2]-jnp.asarray(p.target_center),axis=-1)-p.target_radius


def make_problem(shape=(66,36,27,41),*,parameters=None,
                 domain_lower=(0.,-.2,-.4,-7.),domain_upper=(6.5,3.3,10.,7.)):
    p=parameters or ReachAvoidJumpParameters()
    model,grid,failure=make_jump_problem(shape,parameters=p,domain_lower=domain_lower,
        domain_upper=domain_upper,control_mode='min')
    target=jnp.broadcast_to(goal_value(grid.states,p),(2,)+tuple(shape))
    # Positive inside failure, opposite sign to the avoid/filter example.
    return model,grid,target,-failure
