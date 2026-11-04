"""Physical model adapters for the shared policy and event simulation pipeline."""
from dataclasses import dataclass
import numpy as np
from ..coordinates import IdentityCoordinates, AsinhCoordinates, PhysicalPolicy
from ..policies import ValuePolicy, HybridFeedback
from ..filtering import AvoidFilter
from ..lookahead import HoldLookahead
from ..simulation import GuardEvent, simulate
from ..geometry import clearance_bound
from .satellite import physical_flow


@dataclass
class PaperSimulation:
    problem: object
    parameters: object
    system: str
    coordinates: object
    events: tuple

    def clearance(self, state, mode=0):
        state = np.asarray(state)
        p = self.parameters
        if self.system == 'satellite':
            return float(min([state[1], *[np.linalg.norm(state[:2]-c)-p.obstacle_radius
                                         for c in p.obstacle_centers]]))
        x, z = state[:2]
        return float(min([z, *[max(left-x, x-right, z-top) for left, right, top in p.pillars]]))

    def goal(self, state, mode=0):
        state = np.asarray(state)
        p = self.parameters
        if self.system == 'satellite':
            return float(np.max(abs(np.asarray(state[:2])-p.goal_center)-p.goal_half_width))
        return float(np.linalg.norm(state[:2]-p.target_center)-p.target_radius)

    def flow(self, state, mode, control, disturbance=None, time=0.):
        p = self.parameters
        if self.system == 'satellite':
            u = control if mode == 1 else np.zeros(2)
            return np.r_[physical_flow(state, u, p), 0. if mode == 0 else 1/p.duration(mode)]
        u = np.asarray(control) if mode == 0 else np.zeros(2)
        return np.r_[state[2:], [u[0]/p.mass, u[1]/p.mass-p.gravity]]

    def computational_passive_flow(self, state, mode):
        physical = self.coordinates.decode(state)
        return self.flow(physical, mode, np.zeros(2))/self.coordinates.inverse_jacobian(physical)

    def _speed_bound(self):
        low = self.coordinates.decode(np.array([v[0] for v in self.problem.grid.coordinate_vectors]))
        high = self.coordinates.decode(np.array([v[-1] for v in self.problem.grid.coordinate_vectors]))
        return float(np.linalg.norm(np.maximum(abs(low[2:4]), abs(high[2:4]))))

    def hold_lookahead(self, result, feedback_dt):
        policy = ValuePolicy(self.problem, result)
        def bound(solution, stop, mode):
            return clearance_bound(solution, 0., stop,
                lambda x: self.clearance(self.coordinates.decode(x), mode), self._speed_bound())
        return HoldLookahead(policy, self.computational_passive_flow, step=feedback_dt,
            modes=(0,), failure=lambda x, q: self.clearance(self.coordinates.decode(x), q),
            clearance_bound=bound)

    def controller(self, result, *, feedback_dt, margin=0., nominal=None, filtered=True):
        nominal = nominal or (lambda x, q: np.zeros(2))
        if self.problem.kind == 'avoid':
            hold = self.hold_lookahead(result, feedback_dt) if self.system == 'satellite' else None
            hlrf = AvoidFilter(self.problem, result, margin=margin, lookahead=hold)
            if filtered:
                computational = lambda x, q, tau: hlrf(x, q, tau,
                    lambda z, m: nominal(self.coordinates.decode(z), m))
            else:
                # Nominal benchmark still obeys deterministic forced resets.
                from ..filtering import Decision
                def computational(x, q, tau):
                    path = next(p for p in hlrf.candidates(x, q, tau) if not p[2])
                    z, m, choices, trace, value = path
                    return Decision(z, m, np.asarray(nominal(self.coordinates.decode(z), m)),
                                    value, False, False, value >= 0, trace)
        else:
            hold = self.hold_lookahead(result, feedback_dt) if self.system == 'satellite' else None
            computational = HybridFeedback(self.problem, result, lookahead=hold)
        return PhysicalPolicy(computational, self.coordinates)

    def run(self, result, initial, *, duration=None, feedback_dt=.01, margin=0., nominal=None,
            filtered=True, initial_mode=0, finish_x=None):
        duration = float(result.horizons[-1]) if duration is None else duration
        controller = self.controller(result, feedback_dt=feedback_dt, margin=margin,
                                     nominal=nominal, filtered=filtered)
        goal = self.goal if self.problem.kind != 'avoid' else None
        if finish_x is not None:
            goal = lambda x, q: finish_x-x[0]
        def audit(solution, start, stop, mode, control):
            # Constant-force jumper velocity is affine; its norm is convex, so
            # the endpoint maximum bounds speed on the entire segment.
            speed = self._speed_bound() if self.system == 'satellite' else max(
                np.linalg.norm(solution.y[2:4, 0]), np.linalg.norm(solution.y[2:4, -1]))
            return dict(clearance_lower_bound=clearance_bound(solution, start, stop,
                lambda x: self.clearance(x, mode), speed,
                tolerance=.01 if self.system == 'satellite' else .0001))
        return simulate(self.problem, controller, initial, initial_mode,
            duration=duration, feedback_dt=feedback_dt, events=self.events,
            failure=self.clearance, goal=goal, coordinates=self.coordinates, flow=self.flow,
            max_step=1. if self.system == 'satellite' else .01,
            rtol=2e-11, atol=1e-8 if self.system == 'satellite' else 1e-11,
            fixed_axes=(lambda q: (4,) if q == 0 else ()) if self.system == 'satellite' else None,
            segment_audit=audit)


def jumper_adapter(problem, parameters):
    leg = parameters.leg_length
    events = (GuardEvent('launch', lambda x: x[1]-leg, (0,), 1),
              GuardEvent('land', lambda x: x[1]-leg, (1,), -1))
    return PaperSimulation(problem, parameters, 'jumper', IdentityCoordinates(), events)


def satellite_adapter(problem, parameters, scales):
    coordinates = IdentityCoordinates() if scales is None else AsinhCoordinates(tuple(scales))
    events = (GuardEvent('thrust_complete', lambda x: x[4]-1., (1,), 1, (4, 1)),
              GuardEvent('cooldown_complete', lambda x: x[4]-1., (2,), 1, (4, 1)))
    return PaperSimulation(problem, parameters, 'satellite', coordinates, events)
