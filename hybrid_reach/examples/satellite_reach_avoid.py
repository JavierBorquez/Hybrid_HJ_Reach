"""IJRR satellite hBRAT: nonlinear Hill dynamics and a positional docking box."""
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
from .satellite import SatelliteParameters,physical_coordinates,make_problem as make_satellite


@dataclass(frozen=True)
class ReachAvoidSatelliteParameters(SatelliteParameters):
    obstacle_centers: tuple[tuple[float,float], ...] = ((-2000.,6000.),(2000.,7000.),(0.,2000.))
    goal_center: tuple[float,float] = (0.,200.)
    goal_half_width: tuple[float,float] = (400.,200.)

    def __post_init__(self):
        super().__post_init__()
        if len(self.goal_center)!=2 or not np.isfinite(self.goal_center).all():raise ValueError('Expected a finite planar goal center')
        if len(self.goal_half_width)!=2 or any(not np.isfinite(v) or v<=0 for v in self.goal_half_width):raise ValueError('Expected positive finite goal half-widths')


def goal_value(states,parameters=None,*,xp=jnp):
    p=parameters or ReachAvoidSatelliteParameters()
    return xp.max(xp.abs(states[...,:2]-xp.asarray(p.goal_center))-xp.asarray(p.goal_half_width),axis=-1)


def constant_boundary(values,pad_width):
    """Flat ghost values avoid inventing lower costs outside the grid."""
    return jnp.pad(values,((pad_width,pad_width),),mode='edge')


def make_problem(shape=(41,31,15,15,11),*,parameters=None,
                 domain_lower=(-6000.,-1000.,-40.,-40.,0.),
                 domain_upper=(6000.,10000.,40.,40.,1.),coordinate_scales=(2500.,2000.,10.,10.),
                 bound_constraint=True,boundary='constant'):
    p=parameters or ReachAvoidSatelliteParameters()
    model,grid,failure=make_satellite(shape,parameters=p,domain_lower=domain_lower,
        domain_upper=domain_upper,coordinate_scales=coordinate_scales,control_mode='min')
    if boundary=='constant':grid=grid.replace(boundary_conditions=(constant_boundary,)*5)
    elif boundary!='library':raise ValueError('Expected constant or library boundary treatment')
    goal=jnp.broadcast_to(goal_value(physical_coordinates(grid.states,coordinate_scales),p),(3,)+tuple(shape))
    constraint=-failure
    if bound_constraint:
        # g >= -min(half_width) everywhere. Truncating l below this bound
        # leaves max(g(at_stop), max_past(l)) exactly unchanged, while preventing
        # ENO/extrapolation from creating impossible arbitrarily negative costs.
        constraint=jnp.maximum(constraint,-min(p.goal_half_width))
    return model,grid,goal,constraint
