"""Nonlinear planar Hill spacecraft, with idle/thrust/cooldown modes.

The fifth continuous coordinate is mode-normalized timer progress, not seconds:
phase = elapsed/thrust_duration in thrust and elapsed/cooldown_duration in
cooldown. It is held at zero in idle and reset to zero at every switch.
"""
from dataclasses import dataclass

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np

from hybrid_reach.model import HybridModel,Mode,ProjectionStage,Transition


@dataclass(frozen=True)
class SatelliteParameters:
    mu: float = 398600e9
    orbital_radius: float = 7000000.
    acceleration_max: float = 1.
    thrust_duration: float = 60.
    cooldown_duration: float = 300.
    obstacle_radius: float = 1000.
    obstacle_centers: tuple[tuple[float,float], ...] = ((-2000.,6000.),(5000.,4000.),(2000.,7000.),(0.,2000.))

    def __post_init__(self):
        if any(not np.isfinite(v) or v<=0 for v in (self.mu,self.orbital_radius,
                self.acceleration_max,self.thrust_duration,self.cooldown_duration,self.obstacle_radius)):
            raise ValueError('Physical parameters must be finite and positive')
        if any(len(c)!=2 or not np.isfinite(c).all() for c in self.obstacle_centers):
            raise ValueError('Expected finite planar obstacle centers')

    @property
    def omega(self): return np.sqrt(self.mu/self.orbital_radius**3)

    def duration(self,mode):
        if mode==0: return np.inf
        if mode==1: return self.thrust_duration
        if mode==2: return self.cooldown_duration
        raise ValueError('Unknown mode')


def drift_acceleration(state,parameters=None,*,xp=jnp):
    """Algebraically exact Hill forces without subtracting large gravity terms."""
    p=parameters or SatelliteParameters()
    along,radial,vp,vr=(state[...,i] for i in range(4))
    R=p.orbital_radius; w=p.omega
    relative_norm_squared=2*radial/R+(radial*radial+along*along)/(R*R)
    delta=xp.expm1(-1.5*xp.log1p(relative_norm_squared))  # (R/rho)^3 - 1
    return xp.stack([-2*w*vr-w*w*along*delta,
                      2*w*vp-w*w*(R+radial)*delta],axis=-1)


def physical_flow(state,control,parameters=None):
    """NumPy physical (four-state) vector field for feedback ODE integration."""
    state=np.asarray(state);p=parameters or SatelliteParameters()
    return np.r_[state[2:4],drift_acceleration(state,p,xp=np)+np.asarray(control)]


def grid_coordinates(states,scales=None,*,xp=jnp):
    """Physical -> uniform computational coordinates; phase is unchanged."""
    if scales is None:return xp.asarray(states)
    s=xp.asarray(states);a=xp.asarray(scales)
    return xp.concatenate((xp.arcsinh(s[...,:4]/a),s[...,4:]),axis=-1)


def physical_coordinates(states,scales=None,*,xp=jnp):
    if scales is None:return xp.asarray(states)
    s=xp.asarray(states);a=xp.asarray(scales)
    return xp.concatenate((a*xp.sinh(s[...,:4]),s[...,4:]),axis=-1)


class SatelliteDynamics(hj.Dynamics):
    def __init__(self,mode,parameters=None,coordinate_scales=None,*,control_mode='max'):
        if mode not in (0,1,2):raise ValueError('Unknown satellite mode')
        self.mode=mode;self.parameters=parameters or SatelliteParameters()
        self.coordinate_scales=coordinate_scales
        upper=self.parameters.acceleration_max if mode==1 else 0.
        if control_mode not in ('min','max'):raise ValueError('Control mode must be min or max')
        super().__init__(control_mode,'max' if control_mode=='min' else 'min',hj.sets.Box(jnp.full(2,-upper),jnp.full(2,upper)),
                         hj.sets.Box(jnp.zeros(0),jnp.zeros(0)))

    def __call__(self,state,control,disturbance,time):
        state=physical_coordinates(state,self.coordinate_scales)
        rate=0. if self.mode==0 else 1/self.parameters.duration(self.mode)
        force=control if self.mode==1 else jnp.zeros(2)
        physical=jnp.concatenate((state[2:4],drift_acceleration(state,self.parameters)+force,jnp.array([rate])))
        return physical/self.jacobian(state)

    def jacobian(self,physical):
        if self.coordinate_scales is None:return jnp.ones(5)
        return jnp.concatenate((jnp.sqrt(jnp.asarray(self.coordinate_scales)**2+physical[:4]**2),jnp.ones(1)))

    def optimal_control_and_disturbance(self,state,time,grad_value):
        # Zero is an admissible tie choice, including both unactuated modes.
        direction=-1 if self.control_mode=='min' else 1
        return direction*jnp.sign(grad_value[2:4])*self.control_space.hi,jnp.zeros(0)

    def partial_max_magnitudes(self,state,time,value,grad_value_box):
        state=physical_coordinates(state,self.coordinate_scales)
        drift=drift_acceleration(state,self.parameters)
        rate=0. if self.mode==0 else 1/self.parameters.duration(self.mode)
        return jnp.concatenate((jnp.abs(state[2:4]),jnp.abs(drift)+self.control_space.hi,jnp.array([rate])))/self.jacobian(state)


def failure_value(states,parameters=None):
    p=parameters or SatelliteParameters()
    result=states[...,1]  # Closed lower half-plane, independent of along-track position.
    for center in p.obstacle_centers:
        result=jnp.minimum(result,jnp.linalg.norm(states[...,:2]-jnp.array(center),axis=-1)-p.obstacle_radius)
    return result


def reset_phase(states):
    result=np.array(states,copy=True)
    result[...,4]=0.
    return result


def forced_destination(mode,phase):
    if mode not in (0,1,2):raise ValueError('Unknown mode')
    return ((mode+1)%3) if mode!=0 and phase>=1.-1e-12 else mode


def make_problem(shape=(31,26,15,15,11),*,parameters=None,
                 domain_lower=(-4500.,-1000.,-30.,-30.,0.),
                 domain_upper=(8500.,10000.,30.,30.,1.),coordinate_scales=None,control_mode='max'):
    p=parameters or SatelliteParameters();shape=tuple(shape)
    if len(shape)!=5 or any(n<4 for n in shape):raise ValueError('Expected five grid axes with at least four points')
    if len(domain_lower)!=5 or len(domain_upper)!=5 or any(
            not np.isfinite(a+b) or b<=a for a,b in zip(domain_lower,domain_upper)):
        raise ValueError('Expected finite increasing domain bounds')
    if domain_lower[-1]!=0 or domain_upper[-1]!=1:raise ValueError('Normalized timer domain must be [0,1]')
    if domain_lower[1]<=-p.orbital_radius:raise ValueError('Domain includes the gravity singularity')
    if coordinate_scales is not None and (len(coordinate_scales)!=4 or any(
            not np.isfinite(a) or a<=0 for a in coordinate_scales)):
        raise ValueError('Expected four positive finite coordinate scales')
    grid=hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(grid_coordinates(domain_lower,coordinate_scales),grid_coordinates(domain_upper,coordinate_scales)),shape)
    modes=tuple(Mode(name,SatelliteDynamics(i,p,coordinate_scales,control_mode=control_mode)) for i,name in enumerate(('idle','thrust','cooldown')))
    optional=Transition(0,1,lambda s:np.ones(s.shape[:-1],dtype=bool),reset_phase)
    forced=tuple(Transition(a,b,lambda s:s[...,4]>=1.-1e-12,reset_phase) for a,b in ((1,2),(2,0)))
    # Cooldown exit may immediately engage thrust. Apply the optional idle
    # value first, so the forced q3->q1 reset sees that admissible decision.
    model=HybridModel(modes,(ProjectionStage('controlled',(optional,),control_mode),
                             ProjectionStage('forced',forced,control_mode)))
    failure=jnp.broadcast_to(failure_value(physical_coordinates(grid.states,coordinate_scales),p),(3,)+shape)
    return model,grid,failure
