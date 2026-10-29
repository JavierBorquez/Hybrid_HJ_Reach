"""Two-aircraft collision avoidance: modes, rotating resets and timer guards."""
import jax.numpy as jnp
import numpy as np

import hj_reachability as hj
from hybrid_reach.model import HybridModel, Mode, ProjectionStage, Transition


class AircraftDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """Relative (x, y, timer) dynamics with fixed relative heading 2*pi/3.

    Public variant uses admissible input endpoints at Hamiltonian ties and
    dissipation bounds over all corners of the full admissible input box.
    """
    def __init__(self, turning=False, variant="fixed_speed", heading=2*np.pi/3, omega=1.):
        if variant not in ("fixed_speed", "optimal_speed"):
            raise ValueError(f"Unknown aircraft variant: {variant}")
        self.turning = bool(turning)
        self.variant = variant
        self.omega = float(omega) if turning else 0.
        self.cos_heading = float(np.cos(heading))
        self.sin_heading = float(np.sin(heading))
        fixed = variant == "fixed_speed"
        super().__init__("max", "min",
            hj.sets.Box(jnp.array([3. if fixed else 1.5]), jnp.array([3.])),
            hj.sets.Box(jnp.array([4. if fixed else 2.]), jnp.array([4.])))

    def open_loop_dynamics(self, state, time):
        return jnp.array([self.omega*state[1], -self.omega*state[0], float(self.turning)])

    def control_jacobian(self, state, time):
        return jnp.array([[-1.], [0.], [0.]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[self.cos_heading], [self.sin_heading], [0.]])

    def __call__(self, state, control, disturbance, time):
        u, d = control[0], disturbance[0]
        return jnp.array([-u+d*self.cos_heading+self.omega*state[1],
                          d*self.sin_heading-self.omega*state[0], float(self.turning)])

    def optimal_control_and_disturbance(self, state, time, grad_value):
        return super().optimal_control_and_disturbance(state, time, grad_value)

    def hamiltonian(self, state, time, value, grad_value):
        # Match genericHam's sequential dimension sum.
        control, disturbance = self.optimal_control_and_disturbance(state,time,grad_value)
        flow = self(state,control,disturbance,time)
        result = jnp.array(0.,dtype=grad_value.dtype)
        for dim in range(3):
            result = result+grad_value[dim]*flow[dim]
        return result

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        return jnp.max(jnp.stack([jnp.abs(self(state, u, d, time))
                                 for u in (self.control_space.lo, self.control_space.hi)
                                 for d in (self.disturbance_space.lo, self.disturbance_space.hi)]), axis=0)


def make_model(grid, variant="fixed_speed", *, reset_coordinates=None):
    straight = AircraftDynamics(False, variant)
    turning = AircraftDynamics(True, variant)
    modes = (Mode("straight_before", straight), Mode("turning", turning), Mode("straight_after", straight))
    vectors = [np.asarray(v) for v in grid.coordinate_vectors]
    if reset_coordinates is None:
        # MATLAB's builders use index*dx+min, not the physical grid's colon
        # coordinates. Retain this geometry because ceil amplifies ulp differences.
        reset_coordinates = [np.arange(n)*float(dx)+float(lo)
                             for n, dx, lo in zip(grid.shape, grid.spacings, grid.domain.lo)]
    source = np.stack(np.meshgrid(*reset_coordinates, indexing="ij"), axis=-1)
    zero_i = int(np.argmin(abs(vectors[2])))
    pi_i = int(np.argmin(abs(vectors[2]-np.pi)))
    if zero_i+1 >= len(vectors[2]) or pi_i+1 >= len(vectors[2]):
        raise ValueError("Timer grid must include one point past the nearest 0 and pi samples")

    def reset_at(timer):
        def reset(states):
            return np.stack([-source[..., 1], source[..., 0], np.full(grid.shape, timer)], axis=-1)
        return reset

    everywhere = lambda states: np.ones(grid.shape, dtype=bool)
    timer_exit = lambda states: source[..., 2] > np.pi
    # Source order differs from Dog1D: optional max first, then forced exit.
    controlled = ProjectionStage("controlled", (
        Transition(0, 1, everywhere, reset_at(vectors[2][zero_i+1]), "ceil"),), "max")
    forced = ProjectionStage("forced", (
        Transition(1, 2, timer_exit, reset_at(vectors[2][pi_i+1]), "ceil"),), "min")
    return HybridModel(modes, (controlled, forced))


def make_problem(points=31, variant="fixed_speed", *, coordinates=None, reset_coordinates=None):
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([-15., -15., -.1]), jnp.array([15., 15., 3.4])), (points, points, 11))
    # ToolboxLS computes range/(N-1); linspace's retstep rounds the timer
    # spacing differently (.35000000000000003 rather than .35).
    grid = grid.replace(spacings=(30./(points-1),30./(points-1),.35))
    if coordinates is not None:
        vectors = tuple(jnp.asarray(v) for v in coordinates)
        if tuple(len(v) for v in vectors) != tuple(grid.shape):
            raise ValueError("Reference coordinates do not match the selected grid")
        grid = grid.replace(coordinate_vectors=vectors,
                            states=jnp.stack(jnp.meshgrid(*vectors, indexing="ij"), axis=-1))
    target = jnp.broadcast_to(jnp.linalg.norm(grid.states[..., :2], axis=-1)-5., (3,)+tuple(grid.shape))
    return make_model(grid, variant, reset_coordinates=reset_coordinates), grid, target
