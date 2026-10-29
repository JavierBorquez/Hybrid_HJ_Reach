"""Independent objective, reset, feedback and event checks for the public API."""
import numpy as np
import pytest
import hj_reachability as hj
import jax.numpy as jnp

from hybrid_reach import (Mode, HybridModel, Transition, ProjectionStage, Problem,
                          solve, Result, ValuePolicy, AvoidFilter, ContinuousFeedback,
                          simulate, GuardEvent, compile_stages)
from hybrid_reach.transitions import project_values


class Translation(hj.Dynamics):
    def __init__(self, speed=1., controlled=False):
        self.speed, self.controlled = speed, controlled
        super().__init__("min", "max", hj.sets.Box(jnp.array([-1.]), jnp.array([1.])),
                         hj.sets.Box(jnp.zeros(0), jnp.zeros(0)))
    def __call__(self, state, control, disturbance, time):
        return jnp.array([control[0] if self.controlled else self.speed])
    def optimal_control_and_disturbance(self, state, time, grad):
        u = jnp.where(grad[0] >= 0, 1., -1.)
        return jnp.array([u if self.control_mode == "max" else -u]), jnp.zeros(0)
    def partial_max_magnitudes(self, state, time, value, grad_box):
        return jnp.array([1. if self.controlled else abs(self.speed)])


def system(n=81, controlled=False):
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([-1.]), jnp.array([3.])), (n,))
    return HybridModel((Mode("move", Translation(controlled=controlled)),), ()), grid


def all_states(x):
    return np.ones(x.shape[:-1], dtype=bool)


def identity(x):
    return x


def fake_result(problem, fields, times=(0., 1.)):
    fields = np.asarray(fields)
    return Result(problem.kind, tuple(np.asarray(v) for v in problem.grid.coordinate_vectors),
        np.array(times), np.stack([fields for _ in times]), tuple(m.name for m in problem.model.modes))


def test_reach_and_barrier_reach_avoid_against_translation():
    model, grid = system(161)
    x = grid.states[..., 0]
    goal = abs(x-1.)-.1
    failure = abs(x-.4)-.1
    reach = Problem.reach(model, grid, goal)
    ra = Problem.reach_avoid(model, grid, goal, failure)
    times = np.linspace(0., 1.2, 13)
    r = solve(reach, times, save_indices=(0, 12))
    a = solve(ra, times, save_indices=(0, 12))
    assert ValuePolicy(reach, r).query([0.], 0, 1.2)[0] == pytest.approx(-.1, abs=.025)
    assert ValuePolicy(ra, a).query([0.], 0, 1.2)[0] == pytest.approx(.1, abs=.025)


def test_avoid_records_failure_along_path_not_only_endpoint():
    model, grid = system(161)
    failure = abs(grid.states[..., 0]-.4)-.1
    problem = Problem.avoid(model, grid, failure)
    result = solve(problem, np.linspace(0., 1.2, 13), save_indices=(0, 12))
    assert ValuePolicy(problem, result).query([0.], 0, 1.2)[0] == pytest.approx(-.1, abs=.025)


def test_problem_sets_optimization_directions_without_mutation():
    model, grid = system(controlled=True)
    for kind in ("reach", "avoid", "reach_avoid"):
        problem = Problem(model, grid, kind, goal=None if kind == "avoid" else 0.,
                          failure=None if kind == "reach" else 1.)
        d = problem.configured_model().modes[0].dynamics
        u = d.optimal_control(jnp.array([0.]), 0., jnp.array([1.]))
        assert float(u[0]) == (1 if kind == "avoid" else -1)
        assert model.modes[0].dynamics.control_mode == "min"
    assert np.issubdtype(Problem.reach(model, grid, 0).fields()[0].dtype, np.floating)


def test_reset_rejects_active_outside_but_allows_inactive_outputs_and_explicit_clip():
    model, grid = system(5)
    edge = Transition(0, 0, all_states, lambda x: x+1.)
    make = lambda e: HybridModel(model.modes, (ProjectionStage("forced", (e,)),))
    with pytest.raises(ValueError, match="Active reset"):
        compile_stages(make(edge), grid)
    from dataclasses import replace
    compile_stages(make(replace(edge, boundary="clip")), grid)
    compile_stages(make(replace(edge, guard=lambda x: x[..., 0] < 2.)), grid)


@pytest.mark.parametrize("kind", ["reach", "reach_avoid", "avoid"])
def test_immediate_mandatory_reset_and_failure_priority(kind):
    _, grid = system(5)
    modes = (Mode("source", Translation(0.)), Mode("destination", Translation(0.)))
    model = HybridModel(modes, (ProjectionStage("forced", (Transition(0, 1, all_states, identity),)),))
    g = jnp.stack([jnp.full(5, -1.), jnp.ones(5)])
    failure = jnp.stack([jnp.full(5, -.3), jnp.ones(5)])
    problem = Problem(model, grid, kind, goal=None if kind == "avoid" else g,
                      failure=None if kind == "reach" else failure)
    result = solve(problem, [0.])
    expected = -.3 if kind == "avoid" else 1.
    np.testing.assert_allclose(result.values[0, 0], expected)


def test_avoid_projection_preserves_failure_at_intermediate_reset():
    _, grid = system(5)
    model = HybridModel(tuple(Mode(str(i), Translation(0.)) for i in range(3)), (
        ProjectionStage("forced", (Transition(1, 2, all_states, identity),)),
        ProjectionStage("forced", (Transition(0, 1, all_states, identity),))))
    failure = jnp.stack([jnp.ones(5), jnp.full(5, -.4), jnp.ones(5)])
    values = project_values(failure, compile_stages(model, grid), upper_constraint=failure)
    np.testing.assert_allclose(values[0], -.4)
    problem = Problem.avoid(model, grid, failure)
    result = fake_result(problem, np.ones((3, 5)))
    hlrf = AvoidFilter(problem, result)
    decision = hlrf([0.], 0, 1., lambda x, q: np.zeros(1))
    assert decision.mode == 2
    assert decision.value == pytest.approx(-.4)
    trajectory = simulate(problem, lambda x, q, tau: hlrf(x, q, tau, lambda y, p: np.zeros(1)),
        [0.], 0, duration=1., feedback_dt=.1,
        failure=lambda x, q: -.4 if q == 1 else 1.)
    assert trajectory["status"] == "failure" and trajectory["final_mode"] == 1


def test_hlrf_accepts_nominal_safe_and_overrides_unsafe_discrete_choice():
    _, grid = system(5)
    model = HybridModel(tuple(Mode(str(i), Translation(0.)) for i in range(2)),
        (ProjectionStage("controlled", (Transition(0, 1, all_states, identity),), "max"),))
    problem = Problem.avoid(model, grid, jnp.ones((2, 5)))
    values = np.stack([np.full(5, .5), np.full(5, -.5)])
    hlrf = AvoidFilter(problem, fake_result(problem, values))
    nominal = lambda x, q: np.array([.2])
    stay = hlrf([0.], 0, 1., nominal)
    assert stay.mode == 0 and not stay.discrete_intervention
    np.testing.assert_array_equal(stay.control, [.2])
    changed = hlrf([0.], 0, 1., nominal, nominal_edges=((0, 0),))
    assert changed.mode == 0 and changed.discrete_intervention and changed.safe_candidate


def test_hlrf_uses_post_reset_gradient_and_forced_excludes_stay():
    _, grid = system(5, controlled=True)
    modes = tuple(Mode(str(i), Translation(controlled=True)) for i in range(2))
    model = HybridModel(modes, (ProjectionStage("forced", (Transition(0, 1, all_states, identity),)),))
    problem = Problem.avoid(model, grid, 1.)
    x = np.asarray(grid.states[..., 0])
    hlrf = AvoidFilter(problem, fake_result(problem, np.stack([x, -x])))
    decision = hlrf([0.], 0, .5, lambda x, q: np.array([.3]))
    assert decision.mode == 1 and decision.continuous_intervention
    np.testing.assert_array_equal(decision.control, [-1.])
    with pytest.raises(ValueError, match="only for avoid"):
        AvoidFilter(Problem.reach(model, grid, 0.), fake_result(problem, np.ones((2, 5))))


def test_result_roundtrip_queries_remaining_time_and_rejects_extrapolation(tmp_path):
    model, grid = system(5)
    problem = Problem.reach(model, grid, 0.)
    x = np.asarray(grid.states[..., 0])
    result = Result("reach", (x,), np.array([0., 1.]), np.stack([x[None], -x[None]]), ("move",))
    result.save(tmp_path/"run")
    loaded = Result.load(tmp_path/"run")
    assert isinstance(loaded.values, np.memmap)
    policy = ValuePolicy(problem, loaded)
    value, grad = policy.query([.5], 0, .25)
    assert value == pytest.approx(.25)
    np.testing.assert_allclose(grad, [.5])
    with pytest.raises(ValueError):
        policy.query([4.], 0, .5)
    with pytest.raises(ValueError):
        policy.query([0.], 0, 1.1)


def test_sparse_history_runs_all_coupling_steps():
    model, grid = system(41)
    p = Problem.reach(model, grid, abs(grid.states[..., 0]-1.)-.1)
    times = np.linspace(0., .2, 5)
    full = solve(p, times)
    sparse = solve(p, times, save_indices=(0, 4))
    np.testing.assert_array_equal(sparse.values, np.asarray(full.values)[[0, 4]])
    assert sparse.metadata["coupling_steps"] == 4


def test_event_simulator_stops_at_goal_and_applies_physical_guard():
    model, grid = system(41)
    problem = Problem.reach(model, grid, abs(grid.states[..., 0]-1.)-.1)
    result = fake_result(problem, np.asarray(grid.states[..., 0])[None])
    controller = ContinuousFeedback(problem, result)
    trajectory = simulate(problem, controller, [0.], 0, duration=1., feedback_dt=.2,
                          goal=lambda x, q: abs(x[0]-.5)-.01)
    assert trajectory["status"] == "goal_reached"
    assert trajectory["final_time"] == pytest.approx(.49, abs=1e-9)


def test_memory_budget_rejects_execution():
    model, grid = system(21)
    with pytest.raises(MemoryError):
        solve(Problem.reach(model, grid, 0.), [0., .1], memory_budget_bytes=1)


def test_online_stage_composition_matches_backward_bellman_order():
    _, grid = system(5)
    modes = tuple(Mode(str(i), Translation(0.)) for i in range(3))
    # T_forced(T_optional(V)): physical forced exit precedes the optional action.
    model = HybridModel(modes, (
        ProjectionStage("controlled", (Transition(1, 2, all_states, identity),), "max"),
        ProjectionStage("forced", (Transition(0, 1, all_states, identity),))))
    problem = Problem.avoid(model, grid, 1.)
    values = np.stack([np.full(5, -.2), np.full(5, -.3), np.full(5, .5)])
    hlrf = AvoidFilter(problem, fake_result(problem, values))
    decision = hlrf([0.], 0, 1., lambda x, q: np.zeros(1))
    assert decision.mode == 2 and decision.discrete_intervention
    expected = project_values(jnp.asarray(values), compile_stages(problem.configured_model(), grid),
                              upper_constraint=jnp.ones_like(jnp.asarray(values)))
    assert decision.value == pytest.approx(float(expected[0, 1]))
    assert decision.transitions == ((1, 0), (0, 0))


def test_simulator_resolves_guard_before_requerying_control():
    _, grid = system(21)
    modes = (Mode("moving", Translation(1.)), Mode("stopped", Translation(0.)))
    model = HybridModel(modes, (ProjectionStage("forced", (
        Transition(0, 1, lambda x: x[..., 0] >= .5-1e-10, identity),)),))
    problem = Problem.reach(model, grid, 1.)
    policy = ContinuousFeedback(problem, fake_result(problem, np.ones((2, 21))))
    trajectory = simulate(problem, policy, [0.], 0, duration=1., feedback_dt=.3,
        events=(GuardEvent("stop", lambda x: x[0]-.5, (0,), 1.),))
    assert trajectory["final_mode"] == 1
    assert trajectory["final_state"][0] == pytest.approx(.5, abs=1e-9)
    assert trajectory["resets"][0]["time"] == pytest.approx(.5, abs=1e-9)


def test_local_interpolation_matches_full_grid_nonuniform_gradient():
    from scipy.interpolate import RegularGridInterpolator
    from hybrid_reach.interpolation import LocalInterpolator
    axes = (np.array([-2., -.3, .2, 1., 3.]), np.array([-1., .1, .5, 2.]))
    x, y = np.meshgrid(*axes, indexing="ij")
    field = x*x + 2*x*y + y*y*y
    query = np.array([.35, .6])
    local = LocalInterpolator(axes)
    value, gradient = local.sample(field, query, (0, 1))
    expected_value = RegularGridInterpolator(axes, field)(query[None])[0]
    expected_gradient = [RegularGridInterpolator(axes, np.gradient(field, axis, axis=i, edge_order=2))(query[None])[0]
                         for i, axis in enumerate(axes)]
    assert value == pytest.approx(expected_value)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12)
    np.testing.assert_array_equal(local.sample(np.full_like(field, 1234.), query, (0, 1))[1], [0., 0.])
