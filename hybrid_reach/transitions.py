"""Pure JAX value projection; no example-specific mode connections."""
import jax.numpy as jnp


def project_values(values, stages, *, constraint=None, upper_constraint=None):
    """Apply staged transitions to values shaped (mode, *continuous_grid).

    Each stage has simultaneous edge evaluation. Separate stages allow explicit
    composition where a reference implementation relies on reset ordering.
    constraint, when supplied, is a running failure cost (positive inside
    failure). Apply it before and after every stage: an intermediate reset
    cannot erase failure before a subsequent stage reads that mode's value.
    upper_constraint is the avoid clearance: failure at an intermediate reset
    cannot be erased by a later reset. Reach and reach-avoid never reapply a
    source goal after a mandatory reset. No implicit reset closure is performed.
    """
    if constraint is not None:
        if constraint.shape != values.shape:
            raise ValueError("Constraint must have the same shape as values")
        values = jnp.maximum(values, constraint)
    if upper_constraint is not None:
        values = jnp.minimum(values, upper_constraint)
    for stage in stages:
        snapshot = values
        reduce = jnp.minimum if stage.reduction == "min" else jnp.maximum
        neutral = jnp.inf if stage.reduction == "min" else -jnp.inf
        candidates = jnp.full_like(values, neutral)
        active = jnp.zeros(values.shape, dtype=bool)
        for edge in stage.transitions:
            mapped = snapshot[edge.destination].reshape(-1)[edge.indices]
            candidate = jnp.where(edge.mask, mapped, neutral)
            candidates = candidates.at[edge.source].set(reduce(candidates[edge.source], candidate))
            active = active.at[edge.source].set(active[edge.source] | edge.mask)
        if stage.kind == "controlled":
            values = reduce(snapshot, candidates)
        else:
            values = jnp.where(active, candidates, snapshot)
        if constraint is not None:
            values = jnp.maximum(values, constraint)
        if upper_constraint is not None:
            values = jnp.minimum(values, upper_constraint)
    if constraint is not None:
        values = jnp.maximum(values, constraint)
    return values
