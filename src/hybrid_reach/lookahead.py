"""Explicit commitment values for optional waiting in an unactuated mode."""
import numpy as np
from scipy.integrate import solve_ivp


class HoldLookahead:
    """Evaluate one hold step then retain all future hybrid choices.

    flow(state, mode) is a COMPUTATIONAL-coordinate unactuated vector field.
    Only designated modes are evaluated, and no guard may be crossed during the
    hold. clearance_bound(optional) supplies a conservative running clearance;
    otherwise sampled failure is checked and is explicitly only an approximation.
    """
    def __init__(self, policy, flow, *, step, modes, failure=None, clearance_bound=None):
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Hold step must be positive and finite')
        self.policy, self.flow, self.step = policy, flow, step
        self.modes, self.failure, self.clearance_bound = tuple(modes), failure, clearance_bound

    def __call__(self, state, mode, tau, projected_value):
        if mode not in self.modes or tau == 0:
            return projected_value
        step = min(self.step, tau)
        axes = self.policy.result.coordinates
        low, high = np.array([v[0] for v in axes]), np.array([v[-1] for v in axes])
        events = []
        for i in range(len(state)):
            # Coordinates held exactly at an endpoint are legal; exclude the
            # identically zero event that would terminate at the initial instant.
            speed = np.asarray(self.flow(state, mode))[i]
            if not (state[i] == low[i] and speed == 0):
                def below(t, x, i=i): return x[i]-low[i]
                below.terminal, below.direction = True, -1
                events.append(below)
            if not (state[i] == high[i] and speed == 0):
                def above(t, x, i=i): return high[i]-x[i]
                above.terminal, above.direction = True, -1
                events.append(above)
        solution = solve_ivp(lambda t, x: self.flow(x, mode), (0., step), state,
            method='DOP853', rtol=1e-9, atol=1e-11, dense_output=True, events=events or None,
            max_step=step/2)
        if not solution.success:
            raise RuntimeError(solution.message)
        bad = -np.inf if self.policy.result.kind == 'avoid' else np.inf
        if solution.t[-1] < step-1e-10:
            return bad
        predicted = solution.y[:, -1]
        if np.any(predicted < low) or np.any(predicted > high):
            return bad
        value = self.policy.value(predicted, mode, max(0., tau-step))
        samples = solution.sol(np.linspace(0., step, 5)).T
        if self.clearance_bound is not None:
            clearance = self.clearance_bound(solution, step, mode)
        elif self.failure is not None:
            clearance = min(float(self.failure(x, mode)) for x in samples)
        else:
            clearance = np.inf
        if self.policy.result.kind == 'avoid':
            return min(projected_value, value, clearance)
        if self.policy.result.kind == 'reach_avoid':
            return max(value, -clearance)
        return value
