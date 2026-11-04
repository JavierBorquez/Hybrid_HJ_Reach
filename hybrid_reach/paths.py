"""One physical traversal of explicitly composed Bellman transition operators."""
import numpy as np


def reset_state(state, edge, axes):
    result = np.asarray(edge.reset(state), dtype=float)
    if result.shape != state.shape or not np.isfinite(result).all():
        raise ValueError('Invalid active reset')
    lower, upper = np.array([a[0] for a in axes]), np.array([a[-1] for a in axes])
    if edge.boundary == 'clip':
        result = np.clip(result, lower, upper)
    elif np.any(result < lower) or np.any(result > upper):
        raise ValueError('Active reset outside computational domain')
    return result


def enumerate_paths(policy, failure, state, mode, tau, evaluator=None):
    state = np.asarray(state, dtype=float)
    policy.value(state, mode, tau)
    def clearance(x, q):
        return np.inf if failure is None else policy.interpolator.sample(failure[q], x)[0]
    paths = [(state, mode, (), (), clearance(state, mode))]
    for stage_index, stage in reversed(tuple(enumerate(policy.model.stages))):
        next_paths = []
        for x, q, choices, trace, running in paths:
            edges = [(i, e) for i, e in enumerate(stage.transitions)
                     if e.source == q and bool(np.asarray(e.guard(x)))]
            if stage.kind == 'forced' and len(edges) > 1:
                raise ValueError('Online feedback requires deterministic forced transitions')
            if stage.kind == 'controlled' or not edges:
                next_paths.append((x, q, choices, trace, running))
            for i, edge in edges:
                reset = reset_state(x.copy(), edge, policy.result.coordinates)
                choice = ((stage_index, i),)
                next_paths.append((reset, edge.destination,
                    choices+choice if stage.kind == 'controlled' else choices,
                    trace+choice, min(running, clearance(reset, edge.destination))))
        paths = next_paths
    candidates = []
    for x, q, choices, trace, running in paths:
        value = policy.value(x, q, tau)
        if evaluator is not None:
            value = evaluator(x, q, tau, value)
        if policy.result.kind == 'avoid':
            value = min(value, running)
        elif policy.result.kind == 'reach_avoid':
            value = max(value, -running)
        candidates.append((x, q, choices, trace, value))
    return candidates
