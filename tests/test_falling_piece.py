"""Analytic column geometry, Hamiltonians, physical switches and feedback."""
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_reach import ContinuousFeedback, Result, ValuePolicy, simulate, solve
from hybrid_reach.examples.falling_piece import (
    DROP, REST, STEER, FallingDynamics, Parameters, audit_trajectory,
    make_problem, mode_at, physical_flow,
)


spec = importlib.util.spec_from_file_location(
    "falling_lesson", Path(__file__).parents[1]/"tutorials/falling_piece.py")
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)


@pytest.fixture(scope="module")
def solved():
    cases = {}
    for kind in ("reach", "avoid", "reach_avoid"):
        problem, events = make_problem(kind)
        result = solve(problem, np.linspace(0., 4., 401), save_indices=tuple(range(0, 401, 5)))
        cases[kind] = problem, events, result
    return cases


def test_hamiltonian_controls_and_speed_bounds_cover_the_input_box():
    for mode in (STEER, DROP, REST):
        dynamics = FallingDynamics(mode, Parameters(u_max=1.7))
        for direction in ("min", "max"):
            dynamics.control_mode = direction
            for gradient in ([-2., 1.], [0., -3.], [1.5, 4.]):
                gradient = jnp.array(gradient)
                u, d = dynamics.optimal_control_and_disturbance(jnp.zeros(2), 0., gradient)
                lo, hi = dynamics.control_space.lo, dynamics.control_space.hi
                assert np.all(u >= lo) and np.all(u <= hi)
                corners = [float(gradient @ dynamics(jnp.zeros(2), c, d, 0.)) for c in (lo, hi)]
                expected = (min if direction == "min" else max)(corners)
                assert float(gradient @ dynamics(jnp.zeros(2), u, d, 0.)) == pytest.approx(expected)
                bound = dynamics.partial_max_magnitudes(None, 0., None, None)
                for control in (lo, hi):
                    assert np.all(abs(dynamics(jnp.zeros(2), control, d, 0.)) <= bound)


def test_switch_times_and_rest_agree_with_closed_form_motion():
    problem, events = make_problem("reach", shape=(21, 29))
    axes = tuple(np.asarray(v) for v in problem.grid.coordinate_vectors)
    result = Result('reach', axes, np.array([0., 4.]),
        np.zeros((2, 3, 21, 29)), ('steer', 'drop', 'rest'))
    trajectory = simulate(problem, ContinuousFeedback(problem, result), [0., 6.2], STEER,
        duration=4., feedback_dt=.17, events=events, flow=physical_flow,
        fixed_axes=lambda q: () if q == STEER else ((0,) if q == DROP else (0, 1)))
    assert trajectory['status'] == 'deadline'
    assert trajectory['final_mode'] == REST
    np.testing.assert_allclose(trajectory['final_state'], [0., 1.], atol=1e-10)
    assert [r['destination'] for r in trajectory['resets']] == [DROP, REST]
    np.testing.assert_allclose([r['time'] for r in trajectory['resets']], [2.2, 2.8], atol=1e-9)
    for reset in trajectory['resets']:
        np.testing.assert_array_equal(reset['before'], reset['after'])
    assert audit_trajectory(trajectory)['physical_replay_max_error'] < 1e-12
    assert int(mode_at([0., 4.])) == DROP
    assert int(mode_at([0., 1.])) == REST


@pytest.mark.parametrize('terminal', ['goal', 'failure'])
def test_located_terminal_event_after_tiny_progress_is_not_a_zeno_contact(terminal):
    from hybrid_reach.examples.falling_piece import failure, goal
    problem, events = make_problem('reach', shape=(21, 29))
    axes = tuple(np.asarray(v) for v in problem.grid.coordinate_vectors)
    result = Result('reach', axes, np.array([0., .1]),
        np.zeros((2, 3, 21, 29)), ('steer', 'drop', 'rest'))
    state = [.95, 1.7+1e-13] if terminal == 'goal' else [.6, 3.05+1e-13]
    trajectory = simulate(problem, ContinuousFeedback(problem, result), state, DROP,
        duration=.1, feedback_dt=.02, events=events, flow=physical_flow,
        goal=(lambda x, q: float(goal(x, q))) if terminal == 'goal' else None,
        failure=(lambda x, q: float(failure(x, q))) if terminal == 'failure' else None)
    assert trajectory['status'] == ('goal_reached' if terminal == 'goal' else 'failure')
    assert 0 <= trajectory['final_time'] < 1e-12
    assert np.all(np.diff(trajectory['time']) > 0)


def test_exact_segment_audit_detects_an_obstacle_missed_by_endpoint_samples():
    trajectory = dict(state=np.array([[.6, 3.4], [.6, 2.4]]),
        time=np.array([0., .2]), mode=np.array([DROP, DROP]), control=np.zeros((1, 1)),
        segment_end_state=np.array([[.6, 2.4]]), final_state=np.array([.6, 2.4]), final_mode=DROP)
    audit = audit_trajectory(trajectory)
    assert audit['physical_replay_max_error'] < 1e-12
    assert audit['minimum_failure_field'] < -.1


def test_reach_brt_matches_independent_alignment_and_flight_time(solved):
    problem, _, result = solved['reach']
    policy = ValuePolicy(problem, result)
    # Full-horizon analytic reach condition: enough steering to overlap [.6,1.2].
    # These probes avoid the grid-scale zero contour, with both reachable sides.
    for x, y in [(-1., 6.2), (.95, 5.8), (2., 5.5), (-1., 4.4), (-2.5, 5.), (2., 4.2)]:
        available = y-4.
        expected = .6-available <= x <= 1.2+available
        assert bool(policy.value([x, y], STEER, 4.) < 0) == expected
    # Short horizons must also include the descent from the switch to the goal.
    assert policy.value([.95, 5.8], STEER, 1.) > 0
    assert policy.value([.95, 5.8], STEER, 3.) < 0
    assert policy.value([.95, .5], REST, 4.) > 0


def test_reach_avoid_excludes_a_reachable_column_blocked_above_the_goal(solved):
    reach = ValuePolicy(solved['reach'][0], solved['reach'][2])
    ra = ValuePolicy(solved['reach_avoid'][0], solved['reach_avoid'][2])
    assert reach.value([.7, 4.05], STEER, 4.) < 0
    assert ra.value([.7, 4.05], STEER, 4.) > 0
    assert ra.value([1.05, 4.3], STEER, 4.) < 0
    # Below that obstacle, the left part of the same goal becomes reachable.
    assert ra.value([.7, 2.4], DROP, 4.) < 0


@pytest.mark.parametrize('kind', ['reach', 'avoid', 'reach_avoid'])
def test_lesson_rollouts_report_successes_failures_and_exact_replay(solved, kind):
    problem, events, result = solved[kind]
    trajectories = lesson.rollouts(problem, events, result, Parameters())
    for name, trajectory in trajectories.items():
        audit = audit_trajectory(trajectory, check_failure=kind != 'reach')
        assert audit['physical_replay_max_error'] < 1e-9
        assert trajectory['status'] not in ('zeno_contact', 'domain_exit')
        if name.startswith('C'):
            assert trajectory['status'] == ('failure' if kind == 'avoid' else 'deadline')
        elif name.endswith('nominal'):
            assert trajectory['status'] == 'failure'
        else:
            assert trajectory['status'] == ('deadline' if kind == 'avoid' else 'goal_reached')
            if kind != 'reach':
                assert audit['minimum_failure_field'] > .03
        if name.endswith('filtered'):
            assert trajectory['continuous_intervention'].any()
