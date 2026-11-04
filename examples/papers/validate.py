"""Replay serialized controls independently of the feedback and physical RHS."""
import argparse
import json
from pathlib import Path
import sys

# Prefer the code in this clone when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.integrate import solve_ivp
from hybrid_reach.geometry import clearance_bound
from scenarios import make_scenario, CASES


def validate(scenario, folder):
    reports = json.loads((folder/'rollouts.json').read_text())
    audits = {}
    p = scenario.adapter.parameters
    for name, report in reports.items():
        data = np.load(folder/f'{name}.npz', allow_pickle=False)
        times, states, modes = data['time'], data['state'], data['mode'][:-1]
        endpoints, controls = data['segment_end_state'], data['control']
        assert len(times) == len(controls)+1 and np.all(np.diff(times) > 0)
        assert np.isfinite(states).all() and np.isfinite(endpoints).all()
        np.testing.assert_allclose(data['time_to_go'], scenario.horizon-times[:-1], atol=1e-9, rtol=0)
        geometry_bound, error = scenario.adapter.clearance(states[0]), 0.
        if scenario.adapter.system == 'satellite':
            w, R = p.omega, p.orbital_radius
            def Q(t):
                return np.array([[np.cos(w*t), np.sin(w*t)], [-np.sin(w*t), np.cos(w*t)]])
            def to_inertial(x, t):
                pos = np.array([x[0], R+x[1]])
                return np.r_[Q(t)@pos, Q(t)@(x[2:4]+w*np.array([pos[1], -pos[0]]))]
            def to_physical(y, t):
                pos = Q(t).T@y[:2]
                vel = Q(t).T@y[2:]-w*np.array([pos[1], -pos[0]])
                return np.r_[pos[0], pos[1]-R, vel, 0.]
            accumulated = to_inertial(states[0], times[0])
        for i, u in enumerate(controls):
            a, b = times[i:i+2]; dt = b-a; q = int(modes[i]); x = states[i]
            if scenario.adapter.system == 'satellite':
                assert np.all(abs(u) <= p.acceleration_max+1e-12)
                if q != 1: np.testing.assert_array_equal(u, 0.)
                rate = 0. if q == 0 else 1/p.duration(q)
                assert abs(endpoints[i, 4]-x[4]-rate*dt) < 1e-8
                if q == 0: assert abs(x[4]) < 1e-10
                assert x[4] >= -1e-10 and endpoints[i, 4] <= 1+1e-9
                def rhs(t, y):
                    return np.r_[y[2:], -p.mu*y[:2]/np.linalg.norm(y[:2])**3+Q(t)@u]
                solution = solve_ivp(rhs, (a, b), accumulated, method='DOP853',
                    rtol=1e-12, atol=1e-7, max_step=1., dense_output=True)
                accumulated = solution.y[:, -1]
                error = max(error, float(np.max(abs(to_physical(accumulated, b)[:4]-endpoints[i, :4]))))
                # Dense physical wrapper lets the generic geometry audit operate
                # on independent inertial output, without the Hill vector field.
                class DensePhysical:
                    def sol(self, ts):
                        ts = np.asarray(ts)
                        return np.stack([to_physical(solution.sol(t), t) for t in ts], axis=-1)
                dense = DensePhysical()
            else:
                assert np.all(u >= 0) and np.all(u <= p.force_max)
                if q == 1: np.testing.assert_array_equal(u, 0.)
                acceleration = np.array([u[0]/p.mass, u[1]/p.mass-p.gravity]) if q == 0 else np.array([0., -p.gravity])
                expected = np.r_[x[:2]+x[2:]*dt+.5*acceleration*dt*dt, x[2:]+acceleration*dt]
                error = max(error, float(np.max(abs(expected-endpoints[i]))))
                class DensePhysical:
                    def sol(self, ts):
                        t = np.asarray(ts)-a
                        return np.concatenate((x[:2, None]+x[2:, None]*t+.5*acceleration[:, None]*t*t,
                                               x[2:, None]+acceleration[:, None]*t), axis=0)
                dense = DensePhysical()
                assert (max(x[1], endpoints[i, 1]) <= p.leg_length+1e-8 if q == 0
                        else min(x[1], endpoints[i, 1]) >= p.leg_length-1e-8)
            speed = scenario.adapter._speed_bound() if scenario.adapter.system == 'satellite' else max(
                np.linalg.norm(x[2:4]), np.linalg.norm(endpoints[i, 2:4]))
            bound = clearance_bound(dense, a, b, scenario.adapter.clearance,
                speed, tolerance=.01 if scenario.adapter.system == 'satellite' else .0001)
            geometry_bound = min(geometry_bound, bound)
        assert error < (1e-5 if scenario.adapter.system == 'satellite' else 1e-8), error
        for reset in report['resets']:
            before, after = np.array(reset['before']), np.array(reset['after'])
            np.testing.assert_allclose(before[:4], after[:4], atol=1e-8, rtol=0)
            if scenario.adapter.system == 'satellite':
                assert after[4] == 0.
                if reset['source'] in (1, 2): assert abs(before[4]-1.) < 1e-8
            else:
                assert abs(before[1]-p.leg_length) < 1e-8
        positive = geometry_bound > 0
        success = report['status'] in ('deadline', 'goal_reached') and positive
        goal_value = None
        if scenario.problem.kind == 'reach_avoid':
            goal_value = scenario.adapter.goal(states[-1])
            success = report['status'] == 'goal_reached' and goal_value <= 1e-7 and positive
        audits[name] = dict(status=report['status'], physical_replay_max_error=error,
            independent_clearance_lower_bound=float(geometry_bound), terminal_goal_value=goal_value,
            successful_finite_trajectory=bool(success), input_clock_and_time_to_go_checks='passed')
    (folder/'validation.json').write_text(json.dumps(audits, indent=2, allow_nan=False)+'\n')
    print(json.dumps(audits, indent=2))
    return audits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--preset', choices=['smoke', 'demo', 'reference'], default='demo')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    validate(make_scenario(args.case, args.preset), args.output or Path('outputs/papers')/f'{args.case}_{args.preset}')
