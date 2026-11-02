"""Run the guided lesson: python tutorials/falling_piece.py --kind all."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from hybrid_reach import AvoidFilter, ContinuousFeedback, Decision, Result, simulate, solve
from hybrid_reach.examples.falling_piece import (
    Parameters, audit_trajectory, failure, goal, make_problem, mode_at,
    nominal_control, physical_flow,
)


def rollouts(problem, events, result, p):
    if problem.kind == "avoid":
        starts = {"A": [-1.7, 6.6], "B": [1.3, 4.2], "C (too late)": [-1.9, 4.2]}
        hlrf = AvoidFilter(problem, result, margin=.12)

        def safety_feedback(x, q, tau):
            return hlrf(x, q, tau, lambda z, m: hlrf.policy.control(z, m, tau)[0])

        def nominal_feedback(x, q, tau):
            # Only forced transitions: exactly one legal reset path.
            z, m, _, trace, value = hlrf.candidates(x, q, tau)[0]
            return Decision(z, m, nominal_control(z, m, p), value, False, False, value > 0, trace)

        def filtered_feedback(x, q, tau):
            return hlrf(x, q, tau, lambda z, m: nominal_control(z, m, p))

        controllers = {"safety": safety_feedback, "nominal": nominal_feedback,
                       "filtered": filtered_feedback}
    else:
        starts = {"A (aligned)": [.95, 5.8], "B (steer)": [-1., 6.2],
                  "C (too late)": [-1., 4.4]}
        controllers = {"optimal": ContinuousFeedback(problem, result)}
    trajectories = {}
    for policy_name, controller in controllers.items():
        for name, state in starts.items():
            # The filter comparison uses one shared start; the safety GIF uses all three.
            if policy_name in ("nominal", "filtered") and name != "A":
                continue
            label = f"{name}: {policy_name}"
            trajectory = simulate(problem, controller, state, int(mode_at(state, p)),
                duration=float(result.horizons[-1]), feedback_dt=.02, events=events,
                flow=lambda x, q, u, d, t: physical_flow(x, q, u, d, t, p),
                fixed_axes=lambda q: () if q == 0 else ((0,) if q == 1 else (0, 1)),
                failure=None if problem.kind == "reach" else lambda x, q: float(failure(x, q, p)),
                goal=None if problem.kind == "avoid" else lambda x, q: float(goal(x, q, p)))
            trajectories[label] = trajectory
    return trajectories


def run(kind, output, *, shape=(101, 141), horizon=4., frames=81):
    from falling_piece_figures import animate, plot_brt
    p = Parameters()
    problem, events = make_problem(kind, shape, p)
    count = int(np.ceil(horizon/.01))
    horizons = np.linspace(0., horizon, count+1)
    checkpoints = tuple(sorted(set([0, *range(5, count, 5), count])))
    result = solve(problem, horizons, save_indices=checkpoints, memory_budget_bytes=1024**3)
    folder = output/kind
    result.save(folder/'value')
    result = Result.load(folder/'value')
    trajectories = rollouts(problem, events, result, p)
    reports = {}
    for index, (name, trajectory) in enumerate(trajectories.items()):
        filename = f"trajectory_{index}.npz"
        np.savez(folder/filename, **{key: value for key, value in trajectory.items()
                                    if isinstance(value, np.ndarray)})
        audit = audit_trajectory(trajectory, p, check_failure=kind != "reach")
        if audit['physical_replay_max_error'] > 1e-9:
            raise AssertionError("Tutorial trajectory failed independent straight-line replay")
        reports[name] = dict(file=filename, status=trajectory['status'],
            initial_state=trajectory['state'][0].tolist(), final_state=trajectory['final_state'].tolist(),
            final_time=trajectory['final_time'], final_mode=trajectory['final_mode'],
            continuous_interventions=int(trajectory['continuous_intervention'].sum()),
            resets=trajectory['resets'], **audit)
        print(f"  {name}: {trajectory['status']} at {trajectory['final_time']:.3f} s", flush=True)
    primary = {name: tr for name, tr in trajectories.items()
               if name.endswith('safety') or name.endswith('optimal')}
    plot_brt(result, p, primary, folder/'brt.png')
    animate(result, p, primary, folder/'policy.gif', frames=frames)
    if kind == "avoid":
        comparison = {name: tr for name, tr in trajectories.items()
                      if name.endswith('nominal') or name.endswith('filtered')}
        animate(result, p, comparison, folder/'filter.gif', frames=frames)
    summary = dict(objective=kind, parameters=asdict(p), numerical=result.metadata,
                   feedback_dt=.02, filter_margin=.12 if kind == 'avoid' else None,
                   trajectories=reports, coordinate_order=['x', 'y'],
                   representation='point piece; square markers are decorative')
    (folder/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(f"{kind}: {result.metadata['backend']}, solve={result.metadata['execution_seconds']:.2f} s\n"
          f"  Open {folder/'brt.png'} and {folder/'policy.gif'}", flush=True)
    return summary


def main():
    from falling_piece_figures import plot_setup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['all', 'reach', 'avoid', 'reach_avoid'], default='all')
    parser.add_argument('--output', type=Path, default=Path('outputs/falling_piece'))
    parser.add_argument('--shape', nargs=2, type=int, default=(101, 141), metavar=('NX', 'NY'))
    parser.add_argument('--horizon', type=float, default=4.)
    parser.add_argument('--frames', type=int, default=81, help='GIF frames; does not change the solve')
    args = parser.parse_args()
    if not np.isfinite(args.horizon) or args.horizon <= 0 or args.frames < 2:
        parser.error('horizon must be positive and frames must be at least two')
    args.output.mkdir(parents=True, exist_ok=False)
    plot_setup(Parameters(), args.output/'setup.png')
    for kind in ('reach', 'avoid', 'reach_avoid') if args.kind == 'all' else (args.kind,):
        print(f"Solving {kind}...", flush=True)
        run(kind, args.output, shape=tuple(args.shape), horizon=args.horizon, frames=args.frames)


if __name__ == '__main__':
    main()
