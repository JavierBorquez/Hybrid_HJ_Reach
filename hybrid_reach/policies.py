"""Local CPU queries of saved, finite-horizon value functions."""
import numpy as np
import jax.numpy as jnp
import jax
from .interpolation import LocalInterpolator


class ValuePolicy:
    def __init__(self, problem, result):
        if problem.kind != result.kind:
            raise ValueError("Problem/result objective mismatch")
        self.model = problem.configured_model()
        if tuple(m.name for m in self.model.modes) != result.mode_names:
            raise ValueError("Problem/result modes mismatch")
        coordinates = tuple(np.asarray(v) for v in problem.grid.coordinate_vectors)
        def same_axis(a, b):
            tolerance = 8*np.finfo(np.result_type(a.dtype, b.dtype)).eps*max(1., float(np.max(abs(b))))
            return a.shape == b.shape and np.allclose(a, b, rtol=0., atol=tolerance)
        if len(coordinates) != len(result.coordinates) or any(
                not same_axis(a, b) for a, b in zip(coordinates, result.coordinates)):
            raise ValueError("Problem/result coordinates mismatch")
        self.result = result
        self.interpolator = LocalInterpolator(result.coordinates)
        self.cpu = jax.devices('cpu')[0]
        self.controls = tuple(jax.jit(lambda x, g, d=m.dynamics: d.optimal_control(x, 0., g))
                              for m in self.model.modes)

    def query(self, state, mode, time_to_go):
        return self._sample(state, mode, time_to_go, tuple(range(len(self.result.coordinates))))

    def value(self, state, mode, time_to_go):
        return self._sample(state, mode, time_to_go, ())[0]

    def _sample(self, state, mode, time_to_go, derivative_axes):
        """Return V and all spatial derivatives, without full-frame gradients.

        No horizon extension or stationary reuse is implicit. A single saved
        frame can only be queried at that frame's horizon.
        """
        if not isinstance(mode, (int, np.integer)) or not 0 <= mode < len(self.model.modes):
            raise ValueError("Unknown mode")
        times = self.result.horizons
        if not np.isfinite(time_to_go) or time_to_go < times[0] or time_to_go > times[-1]:
            raise ValueError("Time-to-go outside saved horizons")
        hi = min(np.searchsorted(times, time_to_go, side="right"), len(times)-1)
        lo = max(0, hi-1)
        weight = 0. if hi == lo else (time_to_go-times[lo])/(times[hi]-times[lo])
        value, gradient = 0., np.zeros(len(derivative_axes))
        for frame, w in ((lo, 1-weight), (hi, weight)):
            if w == 0:
                continue
            v, g = self.interpolator.sample(self.result.values[frame, mode], state,
                                             derivative_axes)
            value += w*v
            gradient += w*g
        return value, gradient

    def control(self, state, mode, time_to_go):
        value, gradient = self.query(state, mode, time_to_go)
        with jax.default_device(self.cpu):
            control = np.asarray(self.controls[mode](np.asarray(state), gradient))
        return control, value

    def query_physical(self, state, mode, time_to_go, coordinates):
        value, gradient = self.query(coordinates.encode(state), mode, time_to_go)
        return value, coordinates.physical_gradient(gradient, state)


class HybridFeedback:
    """Goal-seeking reach/RA feedback with explicit optional hold commitment.

    Controlled source modes need lookahead. This sampled scheduling policy is
    not a reach/RA LRF. A legal no-optional-switch path wins numerical ties.
    """
    def __init__(self, problem, result, *, lookahead=None, tie_tolerance=1e-9):
        if problem.kind == 'avoid':
            raise ValueError('Use AvoidFilter for avoid problems')
        self.policy = ValuePolicy(problem, result)
        self.lookahead, self.tie_tolerance = lookahead, tie_tolerance
        optional = {e.source for s in self.policy.model.stages if s.kind == 'controlled' for e in s.transitions}
        if optional and (lookahead is None or not optional.issubset(lookahead.modes)):
            raise ValueError('Controlled source modes require explicit hold lookahead')
        failure = problem.fields()[1]
        self.failure = None if failure is None else np.asarray(failure)

    def __call__(self, state, mode, time_to_go):
        from .paths import enumerate_paths
        from .filtering import Decision
        paths = enumerate_paths(self.policy, self.failure, state, mode, time_to_go, self.lookahead)
        best = min(p[4] for p in paths)
        selected = next((p for p in paths if not p[2] and p[4] <= best+self.tie_tolerance),
                        min(paths, key=lambda p: p[4]))
        x, q, choices, trace, value = selected
        control, _ = self.policy.control(x, q, time_to_go)
        return Decision(x, q, control, value, False, bool(choices), value <= 0, trace)


class ContinuousFeedback(HybridFeedback):
    """Convenience goal-seeking feedback for deterministic forced-only models."""
    def __init__(self, problem, result):
        if any(s.kind == 'controlled' for s in problem.model.stages):
            raise ValueError('Provide an explicit policy for controlled discrete transitions')
        super().__init__(problem, result)
