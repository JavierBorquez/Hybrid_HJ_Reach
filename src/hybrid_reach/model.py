"""Host-side model definitions and precomputed reset-value gather maps."""
from dataclasses import dataclass
from typing import Callable, Literal

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Mode:
    name: str
    dynamics: object


@dataclass(frozen=True)
class Transition:
    source: int
    destination: int
    guard: Callable
    reset: Callable
    sampling: Literal["nearest", "ceil"] = "nearest"
    boundary: Literal["error", "clip"] = "error"
    name: str = ""


@dataclass(frozen=True)
class ProjectionStage:
    """All edges in a stage read the same snapshot; stages execute in order.

    A forced stage replaces a source value wherever any outgoing guard is true.
    A controlled stage also includes the option to retain the current value.
    Guards express eligibility, including mode invariants if the model needs it.
    """
    kind: Literal["forced", "controlled"]
    transitions: tuple[Transition, ...]
    reduction: Literal["min", "max"] = "min"


@dataclass(frozen=True)
class HybridModel:
    modes: tuple[Mode, ...]
    stages: tuple[ProjectionStage, ...]


@dataclass(frozen=True)
class CompiledTransition:
    source: int
    destination: int
    mask: object
    indices: object


@dataclass(frozen=True)
class CompiledStage:
    kind: str
    reduction: str
    transitions: tuple[CompiledTransition, ...]


def compile_stages(model, grid):
    """Evaluate guards and reset maps once, then keep the gather arrays on device.

    Active nonperiodic reset destinations must lie inside the grid unless an
    edge explicitly requests clipping. Inactive reset outputs are not sampled.
    Periodic coordinates wrap. No mode coordinate is differenced.
    """
    states = np.asarray(grid.states)
    vectors = [np.asarray(v) for v in grid.coordinate_vectors]
    shape = tuple(grid.shape)
    if not model.modes or len({mode.name for mode in model.modes}) != len(model.modes):
        raise ValueError("Expected nonempty modes with unique names")
    for vector, spacing in zip(vectors, grid.spacings):
        if len(vector) < 3 or not np.allclose(np.diff(vector), spacing, rtol=1e-5, atol=1e-10):
            raise ValueError("Reset compiler requires a uniform computational grid")
    from hj_reachability.boundary_conditions import periodic
    stages = []
    for stage in model.stages:
        if stage.kind not in ("forced", "controlled") or stage.reduction not in ("min", "max"):
            raise ValueError("Invalid projection stage")
        edges = []
        for edge in stage.transitions:
            if not (0 <= edge.source < len(model.modes) and 0 <= edge.destination < len(model.modes)):
                raise ValueError("Transition refers to a nonexistent mode")
            if edge.sampling not in ("nearest", "ceil"):
                raise ValueError("Reset sampling must be 'nearest' or 'ceil'")
            if edge.boundary not in ("error", "clip"):
                raise ValueError("Reset boundary must be error or clip")
            mask = np.broadcast_to(np.asarray(edge.guard(states), dtype=bool), shape).copy()
            reset_states = np.asarray(edge.reset(states))
            if reset_states.shape != states.shape or not np.isfinite(reset_states[mask]).all():
                raise ValueError("Reset must return finite active states with the original grid shape")
            reset_states = np.where(mask[..., None], reset_states, states)
            indices = []
            for axis, vector in enumerate(vectors):
                coordinate = (reset_states[..., axis] - vector[0]) / float(grid.spacings[axis])
                # MATLAB round on nonnegative grid coordinates rounds half upward.
                index = np.floor(coordinate + 0.5) if edge.sampling == "nearest" else np.ceil(coordinate)
                if grid.boundary_conditions[axis] is periodic:
                    index = index % shape[axis]
                else:
                    scale = max(1., abs(vector[0]), abs(vector[-1]))
                    tol = 8*np.finfo(vector.dtype).eps*scale
                    outside = (reset_states[..., axis] < vector[0]-tol) | (reset_states[..., axis] > vector[-1]+tol)
                    if edge.boundary == "error" and np.any(mask & outside):
                        raise ValueError(f"Active reset {edge.name or (edge.source, edge.destination)} leaves axis {axis}")
                    index = np.clip(index, 0, shape[axis] - 1)
                indices.append(index.astype(np.int32))
            flat = np.ravel_multi_index(tuple(indices), shape).astype(np.int32)
            edges.append(CompiledTransition(edge.source, edge.destination, jnp.asarray(mask), jnp.asarray(flat)))
        stages.append(CompiledStage(stage.kind, stage.reduction, tuple(edges)))
    return tuple(stages)
