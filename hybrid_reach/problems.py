"""Explicit autonomous reachability objectives; all input sets are negative inside."""
from copy import copy
from dataclasses import dataclass, replace
from typing import Literal

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Problem:
    model: object
    grid: object
    kind: Literal["reach", "avoid", "reach_avoid"]
    goal: object = None
    failure: object = None

    def __post_init__(self):
        if self.kind not in ("reach", "avoid", "reach_avoid"):
            raise ValueError("Expected reach, avoid or reach_avoid")
        if (self.goal is not None) != (self.kind != "avoid"):
            raise ValueError("Reach/reach_avoid require a goal; avoid takes failure only")
        if (self.failure is not None) != (self.kind != "reach"):
            raise ValueError("Avoid/reach_avoid require failure; reach takes goal only")

    def fields(self):
        shape = (len(self.model.modes),) + tuple(self.grid.shape)
        def evaluate(spec):
            if spec is None:
                return None
            # Callables receive the grid and integer mode so sets may vary by mode.
            data = jnp.stack([jnp.broadcast_to(jnp.asarray(spec(self.grid.states, q)), self.grid.shape)
                              for q in range(shape[0])]) if callable(spec) else jnp.broadcast_to(jnp.asarray(spec), shape)
            data = data.astype(jnp.result_type(data, 0.))
            if not np.isfinite(np.asarray(data)).all():
                raise ValueError("Set fields must be finite")
            return data
        return evaluate(self.goal), evaluate(self.failure)

    def configured_model(self):
        """Configure controller/continuous adversary without mutating user objects.

        Controlled stages belong to the controller. Forced-stage reductions are
        explicit model choices (including adversarial nondeterminism).
        Dynamics must honor control_mode and disturbance_mode in their optimizers.
        """
        controller = "max" if self.kind == "avoid" else "min"
        adversary = "min" if controller == "max" else "max"
        modes = []
        for mode in self.model.modes:
            dynamics = copy(mode.dynamics)
            dynamics.control_mode, dynamics.disturbance_mode = controller, adversary
            modes.append(replace(mode, dynamics=dynamics))
        stages = tuple(replace(s, reduction=controller) if s.kind == "controlled" else s
                       for s in self.model.stages)
        return replace(self.model, modes=tuple(modes), stages=stages)

    @classmethod
    def reach(cls, model, grid, goal):
        return cls(model, grid, "reach", goal=goal)

    @classmethod
    def avoid(cls, model, grid, failure):
        return cls(model, grid, "avoid", failure=failure)

    @classmethod
    def reach_avoid(cls, model, grid, goal, failure):
        return cls(model, grid, "reach_avoid", goal=goal, failure=failure)
