"""Every public paper recipe builds and takes an actual PDE coupling step."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import jax.numpy as jnp
from hybrid_reach import solve

spec = importlib.util.spec_from_file_location("paper_recipes", Path(__file__).parents[1]/"examples/papers/run.py")
recipes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipes)


@pytest.mark.parametrize("name", recipes.CASES)
def test_paper_recipe_without_private_files(name):
    problem, horizon, dt = recipes.make_case(name)
    result = solve(problem, [0., min(.02, horizon)], save_indices=(0, 1))
    assert result.values.shape[0] == 2
    assert np.isfinite(np.asarray(result.values)).all()


def test_aircraft_tie_is_admissible_and_hamiltonian_uses_full_bounds():
    from hybrid_reach.examples.aircoll3d import AircraftDynamics
    from itertools import product
    dynamics = AircraftDynamics(True, "optimal_speed")
    x = jnp.array([1., 2., .4])
    u, d = dynamics.optimal_control_and_disturbance(x, 0., jnp.zeros(3))
    assert 1.5 <= float(u[0]) <= 3. and 2. <= float(d[0]) <= 4.
    for grad in np.random.default_rng(4).normal(size=(5, 3)):
        expected = max(min(float(np.dot(grad, dynamics(x, jnp.array([u]), jnp.array([d]), 0.)))
                           for d in (2., 4.)) for u in (1.5, 3.))
        assert float(dynamics.hamiltonian(x, 0., 0., grad)) == pytest.approx(expected)
