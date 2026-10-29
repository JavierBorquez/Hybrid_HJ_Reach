"""Avoid-only hybrid least-restrictive filtering with post-reset control queries."""
from dataclasses import dataclass
import numpy as np
from .policies import ValuePolicy


@dataclass
class Decision:
    state: np.ndarray
    mode: int
    control: np.ndarray
    value: float
    continuous_intervention: bool
    discrete_intervention: bool
    safe_candidate: bool
    transitions: tuple


class AvoidFilter:
    """Accept safe nominal discrete/continuous choices; otherwise maximize safety.

    Finite-horizon, sampled numerical HLRF, not a safety certificate. Forced
    nondeterminism is supported by the solver but rejected by this online filter:
    the simulator must not choose an environmental transition on the user's behalf.
    Dynamics, states and derivatives use the same computational coordinates.
    """
    def __init__(self, problem, result, *, margin=0., lookahead=None):
        if problem.kind != "avoid":
            raise ValueError("HLRF is available only for avoid problems")
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("Margin must be finite and nonnegative")
        self.policy = ValuePolicy(problem, result)
        self.model = self.policy.model
        from hj_reachability.boundary_conditions import periodic
        self._grid_boundary_conditions = problem.grid.boundary_conditions
        if any(bc is periodic for bc in self._grid_boundary_conditions):
            raise ValueError("Online HLRF currently supports nonperiodic grids only")
        if any(not hasattr(m.dynamics.control_space, "lo") or not hasattr(m.dynamics.control_space, "hi")
               for m in self.model.modes):
            raise ValueError("Online HLRF currently requires box control spaces")
        self.failure = np.asarray(problem.fields()[1])
        self.margin = margin
        self.lookahead = lookahead

    def candidates(self, state, mode, time_to_go):
        from .paths import enumerate_paths
        return enumerate_paths(self.policy, self.failure, state, mode, time_to_go, self.lookahead)

    def __call__(self, state, mode, time_to_go, nominal_control, *, nominal_edges=()):
        """nominal_control(state, post_reset_mode) returns an admissible input.

        nominal_edges is a tuple of (stage index, edge index) controlled choices;
        an empty tuple requests staying wherever optional. Forced edges still run.
        """
        paths = self.candidates(state, mode, time_to_go)
        nominal_edges = tuple(nominal_edges)
        requested = next((i for i, p in enumerate(paths) if p[2] == nominal_edges), None)
        accepted = requested is not None and paths[requested][4] >= self.margin
        selected = requested if accepted else int(np.argmax([p[4] for p in paths]))
        x, q, choices, trace, value = paths[selected]
        safety_control, _ = self.policy.control(x, q, time_to_go)
        nominal = np.asarray(nominal_control(x.copy(), q), dtype=float)
        space = self.model.modes[q].dynamics.control_space
        if (nominal.shape != safety_control.shape or not np.isfinite(nominal).all()
                or np.any(nominal < np.asarray(space.lo)) or np.any(nominal > np.asarray(space.hi))):
            raise ValueError("Nominal control outside the mode's input box")
        intervene = value <= self.margin
        return Decision(x, q, safety_control if intervene else nominal, value,
                        bool(intervene), choices != nominal_edges,
                        bool(value >= self.margin), trace)

    @classmethod
    def from_problem(cls, problem, result, **kwargs):
        return cls(problem, result, **kwargs)
