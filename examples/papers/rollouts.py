"""Solve, reload, simulate and plot paper examples through the public pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from hybrid_reach import Result, solve_streamed
from hybrid_reach.runtime import GpuMemoryMonitor
from scenarios import make_scenario, CASES
from figures import plot_run


def record_sources(folder, label):
    import hybrid_reach
    package = Path(hybrid_reach.__file__).parent
    paths = [(p, 'hybrid_reach/'+str(p.relative_to(package))) for p in package.rglob('*.py')]
    paths += [(p, 'examples/papers/'+p.name) for p in Path(__file__).parent.glob('*.py')]
    hashes = {}
    for path, relative in paths:
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        target = folder/'code_snapshots'/digest/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[relative] = digest
    (folder/f'{label}_source_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--preset', choices=['smoke', 'demo', 'reference'], default='demo')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--memory-gib', type=float, default=5.)
    parser.add_argument('--solve-only', action='store_true')
    parser.add_argument('--rollouts-only', action='store_true')
    parser.add_argument('--skip-nominal', action='store_true')
    args = parser.parse_args()
    scenario = make_scenario(args.case, args.preset)
    folder = args.output or Path('outputs/papers')/f'{args.case}_{args.preset}'
    if not args.rollouts_only:
        folder.mkdir(parents=True, exist_ok=False)
        config = scenario.configuration()
        config['preset'] = args.preset
        (folder/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
        record_sources(folder, 'solve')
        count = int(round(scenario.horizon/scenario.coupling_dt))
        times = np.linspace(0., scenario.horizon, count+1)
        stride = int(round(scenario.save_dt/scenario.coupling_dt))
        checkpoints = tuple(sorted(set([0, *range(stride, count, stride), count])))
        with GpuMemoryMonitor() as monitor:
            result = solve_streamed(scenario.problem, times, folder/'value', save_indices=checkpoints,
                chunk_steps=stride, memory_budget_bytes=int(args.memory_gib*1024**3),
                progress=lambda d: print(f"T={d['horizon']:g} s, GPU/CPU execute={d['execution_seconds']:.2f} s, "
                    f"compile={d['compile_seconds']:.2f} s", flush=True))
        peak = monitor.save(folder/'gpu_memory.csv')
        result.metadata['gpu_peak_total_device_mib'] = peak
        result.metadata['configuration'] = config
        result._write_metadata(folder/'value')
    result = Result.load(folder/'value')
    if args.solve_only:
        return
    record_sources(folder, 'rollout')
    if result.metadata.get('configuration') != dict(scenario.configuration(), preset=args.preset):
        # JSON roundtrips tuples as lists; compare serialized configuration.
        expected = json.loads(json.dumps(dict(scenario.configuration(), preset=args.preset)))
        if result.metadata.get('configuration') != expected:
            raise ValueError('Saved model configuration differs from this scenario')
    reports, trajectories = {}, {}
    for label, state in scenario.initial_states.items():
        variants = (True, False) if scenario.problem.kind == 'avoid' and not args.skip_nominal else (True,)
        for filtered in variants:
            key = label+('_filtered' if filtered else '_nominal')
            tr = scenario.adapter.run(result, state, feedback_dt=scenario.feedback_dt,
                margin=scenario.margin, nominal=scenario.nominal, filtered=filtered,
                finish_x=scenario.finish_x)
            trajectories[key] = tr
            np.savez(folder/f'{key}.npz', **{k: v for k, v in tr.items() if isinstance(v, np.ndarray)})
            bound = min((a['clearance_lower_bound'] for a in tr['segment_audits']), default=scenario.adapter.clearance(state))
            reports[key] = dict(status=tr['status'], final_time=tr['final_time'], final_state=tr['final_state'].tolist(),
                final_mode=tr['final_mode'], clearance_lower_bound=bound,
                continuous_interventions=int(tr['continuous_intervention'].sum()),
                discrete_interventions=int(tr['discrete_intervention'].sum()), resets=tr['resets'],
                root_corrections=tr['root_corrections'],
                queries_outside_value_condition=int(np.sum(~tr['safe_candidate'])),
                minimum_queried_value=float(np.min(tr['value'])) if len(tr['value']) else None,
                maximum_queried_value=float(np.max(tr['value'])) if len(tr['value']) else None,
                initial_value=float(tr['value'][0]) if len(tr['value']) else None)
            print(key, reports[key]['status'], f"t={tr['final_time']:.3f}, clearance bound={bound:.5g}", flush=True)
    (folder/'rollouts.json').write_text(json.dumps(reports, indent=2, allow_nan=False)+'\n')
    plot_run(scenario, result, trajectories, folder)


if __name__ == '__main__':
    main()
