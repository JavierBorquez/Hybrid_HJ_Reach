"""Independent coordinate, clock, physical replay, lookahead and streaming checks."""
import numpy as np
import jax.numpy as jnp
import pytest
from scipy.integrate import solve_ivp
from hybrid_reach import (Problem, Result, solve, solve_streamed, ValuePolicy, HoldLookahead,
    HybridFeedback, AvoidFilter, AsinhCoordinates, PhysicalPolicy, simulate)
from hybrid_reach.examples.satellite import make_problem, SatelliteParameters, physical_flow
from hybrid_reach.examples.adapters import satellite_adapter, jumper_adapter
from hybrid_reach.examples.jump_filter import make_problem as jumper_problem, FilterJumpParameters
from .test_pipeline import system, Translation, all_states, identity
from hybrid_reach import HybridModel, Mode, Transition, ProjectionStage


def constant_result(problem, horizon, value=1.):
    shape = (2, len(problem.model.modes))+tuple(problem.grid.shape)
    return Result(problem.kind, tuple(np.asarray(a) for a in problem.grid.coordinate_vectors),
        np.array([0., horizon]), np.full(shape, value), tuple(m.name for m in problem.model.modes))


def test_coordinate_roundtrip_gradient_and_flow_chain_rule():
    chart = AsinhCoordinates((3., 2., 4., 5.))
    x = np.array([-2., 4., 3., -1., .5])
    np.testing.assert_allclose(chart.decode(chart.encode(x)), x, atol=1e-14)
    coefficients = np.arange(1., 6.)
    analytic = chart.physical_gradient(coefficients, x)
    numerical = []
    for i in range(5):
        d = np.zeros(5); d[i] = 1e-5
        numerical.append((coefficients@chart.encode(x+d)-coefficients@chart.encode(x-d))/(2e-5))
    np.testing.assert_allclose(analytic, numerical, atol=1e-9)
    p = SatelliteParameters()
    model, grid, f = make_problem((5,)*5, parameters=p, coordinate_scales=chart.scales)
    physical = np.array([-2000., 5000., 2., -3., .4]); u = np.array([.3, -.7])
    computational = np.asarray(model.modes[1].dynamics(chart.encode(physical), u, np.zeros(0), 0.))
    expected = np.r_[physical_flow(physical, u, p), 1/60.]
    np.testing.assert_allclose(computational*chart.inverse_jacobian(physical), expected, atol=1e-12)


def test_saved_gpu_axes_allow_only_roundoff_difference_on_cpu():
    model, grid = system(21)
    problem = Problem.reach(model, grid, 0.)
    result = constant_result(problem, 1.)
    result.coordinates = tuple(np.nextafter(a, np.inf) for a in result.coordinates)
    assert ValuePolicy(problem, result).value([0.], 0, .5) == 1.
    result.coordinates = tuple(a+.001 for a in result.coordinates)
    with pytest.raises(ValueError, match='coordinates mismatch'):
        ValuePolicy(problem, result)


def test_unsafe_passive_domain_prediction_blocks_stay_without_clipping():
    _, grid = system(21)
    model = HybridModel((Mode('idle', Translation(0.)), Mode('thrust', Translation(0.))),
        (ProjectionStage('controlled', (Transition(0, 1, all_states, identity),)),))
    problem = Problem.avoid(model, grid, 10.)
    result = constant_result(problem, 1.)
    hold = HoldLookahead(ValuePolicy(problem, result), lambda x, q: np.ones(1), modes=(0,), step=.2)
    decision = AvoidFilter(problem, result, lookahead=hold)([2.9], 0, .5, lambda x, q: np.zeros(1))
    assert decision.mode == 1 and decision.discrete_intervention


@pytest.mark.parametrize('kind', ['reach', 'avoid', 'reach_avoid'])
def test_streamed_chunks_equal_full_history_including_sparse_empty_chunks(tmp_path, kind):
    model, grid = system(41)
    x = grid.states[..., 0]
    problem = Problem(model, grid, kind, goal=None if kind == 'avoid' else abs(x-1.)-.1,
                      failure=None if kind == 'reach' else abs(x-.4)-.1)
    times = np.linspace(0., .4, 9)
    full = solve(problem, times)
    streamed = solve_streamed(problem, times, tmp_path/'stream', save_indices=(0, 6, 8), chunk_steps=2)
    np.testing.assert_allclose(streamed.values, np.asarray(full.values)[[0, 6, 8]], atol=2e-14, rtol=0)
    np.testing.assert_array_equal(Result.load(tmp_path/'stream').values, streamed.values)
    assert streamed.metadata['coupling_steps'] == 8 and len(streamed.metadata['chunks']) == 4
    assert sum(c['compile_seconds'] > 0 for c in streamed.metadata['chunks']) == 2


def test_streaming_does_not_repeat_initial_projection_at_chunk_joins(tmp_path):
    _, grid = system(5)
    model = HybridModel(tuple(Mode(str(q), Translation(0.)) for q in range(7)),
        (ProjectionStage('forced', tuple(Transition(q, q+1, all_states, identity) for q in range(6))),))
    g = jnp.broadcast_to(jnp.arange(7.)[:, None], (7, 5))
    p = Problem.reach(model, grid, g)
    times = np.linspace(0., .3, 4)
    whole = solve(p, times)
    streamed = solve_streamed(p, times, tmp_path/'chain', chunk_steps=1)
    np.testing.assert_array_equal(streamed.values, whole.values)


def test_failed_stream_is_not_loadable_as_complete_result(tmp_path):
    model, grid = system(21)
    directory = tmp_path/'failed'
    with pytest.raises(MemoryError):
        solve_streamed(Problem.reach(model, grid, 0.), [0., .1], directory,
                       chunk_steps=1, memory_budget_bytes=1)
    with pytest.raises(FileNotFoundError):
        Result.load(directory)
    assert 'failed' in (directory/'progress.json').read_text()


@pytest.mark.parametrize('kind', ['avoid', 'reach_avoid'])
def test_idle_hold_looks_at_reduced_time_instead_of_projected_stay(kind):
    _, grid = system(21)
    model = HybridModel((Mode('idle', Translation(0.)), Mode('thrust', Translation(0.))),
        (ProjectionStage('controlled', (Transition(0, 1, all_states, identity),)),))
    p = Problem(model, grid, kind, goal=None if kind == 'avoid' else -1., failure=10.)
    sign = 1. if kind == 'avoid' else -1.
    history = np.stack([np.zeros((2, 21)), np.full((2, 21), sign)])
    result = Result(kind, tuple(np.asarray(a) for a in grid.coordinate_vectors),
        np.array([0., 1.]), history, ('idle', 'thrust'))
    policy = ValuePolicy(p, result)
    hold = HoldLookahead(policy, lambda x, q: np.zeros(1), modes=(0,), step=.1)
    if kind == 'avoid':
        no_hold = AvoidFilter(p, result, margin=.45)([0.], 0, .5, lambda x, q: np.zeros(1))
        assert no_hold.mode == 0
        decision = AvoidFilter(p, result, margin=.45, lookahead=hold)([0.], 0, .5, lambda x, q: np.zeros(1))
    else:
        with pytest.raises(ValueError, match='hold lookahead'):
            HybridFeedback(p, result)
        decision = HybridFeedback(p, result, lookahead=hold)([0.], 0, .5)
    assert decision.mode == 1


def test_physical_satellite_clocks_and_inertial_replay():
    p = SatelliteParameters()
    scales = (6000., 6000., 12., 12.)
    model, grid, f = make_problem((5,)*5, parameters=p, coordinate_scales=scales,
        domain_lower=(-60000., -1000., -60., -60., 0.), domain_upper=(20000., 50000., 60., 60., 1.))
    problem = Problem.avoid(model, grid, f)
    result = constant_result(problem, 400., 10000.)
    adapter = satellite_adapter(problem, p, scales)
    hlrf = AvoidFilter(problem, result)
    controller = PhysicalPolicy(lambda z, q, tau: hlrf(z, q, tau,
        lambda x, mode: np.array([.05, -.01]) if mode == 1 else np.zeros(2),
        nominal_edges=((0, 0),)), adapter.coordinates)
    initial = np.array([-3000., 9000., 0., 0., 0.])
    tr = simulate(problem, controller, initial, 0, duration=400., feedback_dt=13.,
        coordinates=adapter.coordinates, flow=adapter.flow, events=adapter.events,
        failure=adapter.clearance, max_step=1., fixed_axes=lambda q: (4,) if q == 0 else (),
        rtol=2e-11, atol=1e-9)
    assert tr['status'] == 'deadline', (tr['final_time'], tr['final_state'], tr['resets'])
    np.testing.assert_allclose([e['time'] for e in tr['resets']], [0., 60., 360.], atol=1e-8)
    assert tr['resets'][-1]['transitions'] == ((1, 1), (0, 0))
    np.testing.assert_allclose(tr['time_to_go'], 400.-tr['time'][:-1], atol=1e-12)
    for i, u in enumerate(tr['control']):
        q = tr['mode'][i]; dt = tr['time'][i+1]-tr['time'][i]
        rate = 0. if q == 0 else 1/p.duration(int(q))
        assert tr['segment_end_state'][i, 4]-tr['state'][i, 4] == pytest.approx(rate*dt, abs=1e-10)
        if q != 1: np.testing.assert_array_equal(u, [0., 0.])
    # Independent inertial Kepler ODE, carrying accumulated inertial state.
    w, R = p.omega, p.orbital_radius
    def Q(t):
        return np.array([[np.cos(w*t), np.sin(w*t)], [-np.sin(w*t), np.cos(w*t)]])
    def to_inertial(x, t):
        pos = np.array([x[0], R+x[1]])
        return np.r_[Q(t)@pos, Q(t)@(x[2:4]+w*np.array([pos[1], -pos[0]]))]
    y = to_inertial(initial, 0.)
    maximum = 0.
    for i, u in enumerate(tr['control']):
        a, b = tr['time'][i:i+2]
        def rhs(t, s):
            return np.r_[s[2:], -p.mu*s[:2]/np.linalg.norm(s[:2])**3+Q(t)@u]
        sol = solve_ivp(rhs, (a, b), y, method='DOP853', rtol=1e-12, atol=1e-7, max_step=1.)
        y = sol.y[:, -1]
        maximum = max(maximum, np.max(abs(y-to_inertial(tr['segment_end_state'][i], b))))
    assert maximum < 2e-6


def test_only_located_guard_roots_can_be_normalized_at_machine_roundoff():
    from hybrid_reach.simulation import _normalize_guard_face, GuardEvent
    event = GuardEvent('clock', lambda x: x[0]-1., (0,), 1, (0, 1))
    state, correction = _normalize_guard_face(np.array([1.+np.finfo(float).eps]), event,
                                             np.array([0.]), np.array([1.]))
    assert state[0] == 1. and correction['error'] > 0
    with pytest.raises(RuntimeError, match='inconsistent'):
        _normalize_guard_face(np.array([1.0001]), event, np.array([0.]), np.array([1.]))


def test_planar_adapter_resolves_landing_and_never_applies_flight_force():
    p = FilterJumpParameters(pillars=())
    model, grid, f = jumper_problem((9, 9, 5, 5), parameters=p)
    problem = Problem.avoid(model, grid, f)
    adapter = jumper_adapter(problem, p)
    result = constant_result(problem, .4, 1.)
    tr = adapter.run(result, [1., 1., 0., -1.], initial_mode=1, duration=.3,
        feedback_dt=.05, nominal=lambda x, q: np.zeros(2))
    assert tr['resets'][0]['source'] == 1 and tr['resets'][0]['destination'] == 0
    landing = (-1+np.sqrt(1+2*9.81*.5))/9.81
    assert tr['resets'][0]['time'] == pytest.approx(landing, abs=1e-10)
    np.testing.assert_array_equal(tr['control'][tr['mode'][:-1] == 1], 0.)
    assert all(a['clearance_lower_bound'] > 0 for a in tr['segment_audits'])
