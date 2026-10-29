"""Numerical reference tolerances use float64; device choice remains caller-owned."""
import jax

jax.config.update("jax_enable_x64", True)
