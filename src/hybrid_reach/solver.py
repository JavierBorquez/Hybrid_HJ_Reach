"""Autonomous hybrid HJVI, with in-memory or bounded-workspace streamed histories."""
from pathlib import Path
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj
from .model import compile_stages
from .transitions import project_values
from .results import Result


def _inputs(horizons, save_indices, cfl):
    horizons = np.asarray(horizons, dtype=float)
    if (horizons.ndim != 1 or len(horizons) < 1 or horizons[0] != 0
            or not np.isfinite(horizons).all() or np.any(np.diff(horizons) <= 0)):
        raise ValueError('Coupling horizons must start at zero and increase strictly')
    if not np.isfinite(cfl) or not 0 < cfl <= 1:
        raise ValueError('CFL must lie in (0, 1]')
    indices = tuple(range(len(horizons))) if save_indices is None else tuple(save_indices)
    if (not indices or indices[0] != 0 or indices[-1] != len(horizons)-1
            or any(not isinstance(i, (int, np.integer)) for i in indices)
            or any(b <= a for a, b in zip(indices, indices[1:]))):
        raise ValueError('Checkpoints must include zero and final index and increase strictly')
    return horizons, indices


class _Stepper:
    def __init__(self, problem, cfl, sequential_modes, upwind_scheme, time_integrator, budget):
        self.problem, self.budget = problem, budget
        self.model = problem.configured_model()
        stages = compile_stages(self.model, problem.grid)
        goal, failure = problem.fields()
        target = failure if problem.kind == 'avoid' else goal
        lower = -failure if problem.kind == 'reach_avoid' else None
        upper = failure if problem.kind == 'avoid' else None
        self.project = lambda v: project_values(v, stages, constraint=lower, upper_constraint=upper)
        terminal = target if lower is None else jnp.maximum(target, lower)
        self.initial = self.project(terminal)
        upwind_scheme = upwind_scheme or hj.finite_differences.upwind_first.ENO3
        time_integrator = time_integrator or hj.time_integration.third_order_total_variation_diminishing_runge_kutta
        def clamp(q, v):
            v = jnp.minimum(v, target[q])
            return v if lower is None else jnp.maximum(v, lower[q])
        settings = tuple(hj.SolverSettings(upwind_scheme=upwind_scheme, time_integrator=time_integrator,
            CFL_number=cfl, value_postprocessor=lambda t, v, q=q: clamp(q, v))
            for q in range(len(self.model.modes)))
        self.cache = {}
        def make_execute(count, checkpoints):
            slots = np.full(count, -1, dtype=np.int32)
            slots[list(checkpoints)] = np.arange(len(checkpoints))
            slots = jnp.asarray(slots)
            @jax.jit
            def execute(times, initial):
                saved = jnp.zeros((len(checkpoints),)+initial.shape, dtype=initial.dtype)
                if 0 in checkpoints:
                    saved = saved.at[0].set(initial)
                def interval(i, carry):
                    previous, history = carry
                    def propagate(q):
                        return hj.step(settings[q], self.model.modes[q].dynamics, problem.grid,
                            -times[i-1], previous[q], -times[i], progress_bar=False)
                    if sequential_modes:
                        branches = tuple(lambda _, q=q: propagate(q) for q in range(len(self.model.modes)))
                        values = jax.lax.fori_loop(0, len(self.model.modes),
                            lambda q, v: v.at[q].set(jax.lax.switch(q, branches, None)), jnp.zeros_like(previous))
                    else:
                        values = jnp.stack([propagate(q) for q in range(len(self.model.modes))])
                    values = self.project(values)
                    if checkpoints:
                        history = jax.lax.cond(slots[i] >= 0,
                            lambda h: h.at[slots[i]].set(values), lambda h: h, history)
                    return values, history
                return jax.lax.fori_loop(1, count, interval, (initial, saved))
            return execute
        self.make_execute = make_execute

    def run(self, times, initial, checkpoints):
        key = (len(times), tuple(checkpoints))
        compile_seconds = 0.
        if key not in self.cache:
            start = time.perf_counter()
            executable = self.make_execute(*key).lower(jnp.asarray(times), initial).compile()
            compile_seconds = time.perf_counter()-start
            analysis = executable.memory_analysis()
            estimate = None if analysis is None else int(analysis.argument_size_in_bytes
                + analysis.output_size_in_bytes + analysis.temp_size_in_bytes-analysis.alias_size_in_bytes)
            if self.budget is not None and estimate is not None and estimate > self.budget:
                raise MemoryError(f'XLA estimated execution bytes {estimate} exceed budget {self.budget}')
            self.cache[key] = executable, estimate
        executable, estimate = self.cache[key]
        start = time.perf_counter()
        final, history = executable(jnp.asarray(times), initial)
        history.block_until_ready()
        elapsed = time.perf_counter()-start
        # Check the endpoint even when the caller saves no interior samples.
        if not np.isfinite(np.asarray(final)).all() or not np.isfinite(np.asarray(history)).all():
            raise FloatingPointError('Solver produced nonfinite values')
        return final, history, dict(compile_seconds=compile_seconds, execution_seconds=elapsed,
                                   estimated_execution_bytes=estimate)


def _result(problem, times, indices, values, cfl, sequential, chunks):
    metadata = dict(status='complete', backend=jax.default_backend(), jax_version=jax.__version__,
        compile_seconds=sum(c['compile_seconds'] for c in chunks),
        execution_seconds=sum(c['execution_seconds'] for c in chunks),
        estimated_execution_bytes=max((c['estimated_execution_bytes'] or 0 for c in chunks), default=0),
        coupling_steps=len(times)-1, coupling_horizons=times.tolist(), save_indices=list(map(int, indices)),
        cfl=cfl, sequential_modes=sequential, dtype=str(values.dtype),
        value_convention='positive_safe' if problem.kind == 'avoid' else 'negative_winning',
        converged=False, certified=False, chunks=chunks)
    return Result(problem.kind, tuple(np.asarray(v) for v in problem.grid.coordinate_vectors),
        times[list(indices)], values, tuple(m.name for m in problem.model.modes), metadata)


def solve(problem, horizons, *, save_indices=None, cfl=.8, sequential_modes=True,
          upwind_scheme=None, time_integrator=None, memory_budget_bytes=None):
    """Each adjacent horizon is a coupling step; checkpoints only thin storage."""
    times, indices = _inputs(horizons, save_indices, cfl)
    stepper = _Stepper(problem, cfl, sequential_modes, upwind_scheme, time_integrator, memory_budget_bytes)
    _, values, stats = stepper.run(times, stepper.initial, indices)
    return _result(problem, times, indices, values, cfl, sequential_modes, [stats])


def solve_streamed(problem, horizons, directory, *, save_indices=None, chunk_steps=20,
                   cfl=.8, sequential_modes=True, upwind_scheme=None, time_integrator=None,
                   memory_budget_bytes=None, progress=None):
    """Continue the same HJVI in chunks; write selected frames to a host memmap.

    Only endpoint/workspace and the current chunk's selected frames are retained
    on device. Initial reset projection runs ONCE, never again at chunk joins.
    Repeated chunk shapes reuse one executable with dynamic times/initial values.
    A partial run has progress.json but no metadata.json, so Result.load fails.
    """
    times, indices = _inputs(horizons, save_indices, cfl)
    if not isinstance(chunk_steps, int) or chunk_steps <= 0:
        raise ValueError('chunk_steps must be a positive integer')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    stepper = _Stepper(problem, cfl, sequential_modes, upwind_scheme, time_integrator, memory_budget_bytes)
    shape = (len(indices),)+stepper.initial.shape
    history = np.lib.format.open_memmap(directory/'values.npy', mode='w+',
        dtype=np.dtype(stepper.initial.dtype), shape=shape)
    current = stepper.initial
    history[0] = np.asarray(current)
    chunks = []
    slot = {int(i): j for j, i in enumerate(indices)}
    try:
        for start in range(0, len(times)-1, chunk_steps):
            stop = min(start+chunk_steps, len(times)-1)
            selected = tuple(i-start for i in indices if start < i <= stop)
            # Always return an endpoint, independent of selected output frames.
            current, saved, stats = stepper.run(times[start:stop+1], current, selected)
            for k, local in enumerate(selected):
                history[slot[start+local]] = np.asarray(saved[k])
            del saved
            history.flush()
            stats.update(start_index=start, stop_index=stop, horizon=float(times[stop]))
            chunks.append(stats)
            (directory/'progress.json').write_text(json.dumps(dict(status='running', chunks=chunks), indent=2)+'\n')
            if progress is not None:
                progress(stats)
        result = _result(problem, times, indices, history, cfl, sequential_modes, chunks)
        result.metadata.update(streamed=True, chunk_steps=chunk_steps)
        result._write_metadata(directory)
        (directory/'progress.json').write_text(json.dumps(dict(status='complete', chunks=chunks), indent=2)+'\n')
        return result
    except Exception as error:
        history.flush()
        (directory/'progress.json').write_text(json.dumps(dict(status='failed', error=str(error), chunks=chunks), indent=2)+'\n')
        raise
