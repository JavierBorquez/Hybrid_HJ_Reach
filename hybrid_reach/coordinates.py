"""Explicit state charts for physical simulation and computational HJ grids."""
from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class IdentityCoordinates:
    def encode(self, state):
        return np.asarray(state, dtype=float)

    def decode(self, state):
        return np.asarray(state, dtype=float)

    def inverse_jacobian(self, physical_state):
        """Diagonal dx/dz for converting computational flow to physical flow."""
        return np.ones_like(physical_state, dtype=float)

    def physical_gradient(self, gradient, physical_state):
        return np.asarray(gradient)/self.inverse_jacobian(physical_state)


@dataclass(frozen=True)
class AsinhCoordinates(IdentityCoordinates):
    scales: tuple

    def __post_init__(self):
        scales = np.asarray(self.scales, dtype=float)
        if scales.ndim != 1 or len(scales) == 0 or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('Expected positive finite asinh scales')

    def encode(self, state):
        state = np.array(state, dtype=float, copy=True)
        state[..., :len(self.scales)] = np.arcsinh(state[..., :len(self.scales)]/self.scales)
        return state

    def decode(self, state):
        state = np.array(state, dtype=float, copy=True)
        state[..., :len(self.scales)] = np.sinh(state[..., :len(self.scales)])*self.scales
        return state

    def inverse_jacobian(self, physical_state):
        result = np.ones_like(physical_state, dtype=float)
        n = len(self.scales)
        result[..., :n] = np.sqrt(np.asarray(self.scales)**2+np.asarray(physical_state)[..., :n]**2)
        return result


@dataclass
class PhysicalPolicy:
    """Keep a controller computational; expose physical states to the simulator."""
    controller: object
    coordinates: object

    def __call__(self, physical_state, mode, time_to_go):
        encoded = self.coordinates.encode(physical_state)
        decision = self.controller(encoded, mode, time_to_go)
        state = np.array(physical_state, copy=True) if not decision.transitions and np.array_equal(
            decision.state, encoded) else self.coordinates.decode(decision.state)
        return replace(decision, state=state)
