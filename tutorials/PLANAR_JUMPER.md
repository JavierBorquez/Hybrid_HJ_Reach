# Continue with the planar jumper

The [vertical tutorial](README.md) introduces modes, sets and HLRF decisions.
Here the state becomes `(x,z,vx,vz)` and failure includes two pillars plus the
floor. The same solver and filter are used; the physical adapter supplies event
surfaces, ODE flow and conditional clearance checks.

## 1. Build the physical model once

```python
from hybrid_reach import Problem, solve_streamed, Result, AvoidFilter, HybridFeedback
from hybrid_reach.examples.jump_filter import make_problem
from hybrid_reach.examples.jump_reach_avoid import ReachAvoidJumpParameters, goal_value
from hybrid_reach.examples.adapters import jumper_adapter
import numpy as np

p = ReachAvoidJumpParameters()
model, grid, failure = make_problem(
    shape=(81,51,23,31), parameters=p,
    domain_lower=(0.,-.2,-.4,-7.), domain_upper=(6.5,3.3,10.,7.))
```

Stance supplies bounded horizontal/vertical forces. Flight supplies no force.
The model uses one vectorized guard definition for grid projection and online
decisions. The adapter's signed leg-height event locates crossings during ODE
integration. Identity resets preserve the continuous state; control is queried
again after the mode switch.

## 2. Choose avoid or reach-avoid

```python
avoid = Problem.avoid(model, grid, failure)
reach_avoid = Problem.reach_avoid(
    model, grid, goal=lambda states, q: goal_value(states, p), failure=failure)
```

The problem layer changes continuous/discrete controller optimization directions.
There is no second vector field or different reachability solver. The goal is a
position disk, not a terminal-velocity condition.

## 3. Stream the value history

```python
times = np.linspace(0.,2.,801)  # .0025 s hybrid coupling
result = solve_streamed(reach_avoid, times, 'outputs/planar/value',
    save_indices=tuple(range(0,801,20)), chunk_steps=20,
    memory_budget_bytes=5*1024**3)
result = Result.load('outputs/planar/value')
```

Only one chunk's selected frames plus endpoint/workspace remain on GPU. The
compiled update is reused, including when times and initial values change.
Every coupling step still runs. The example saves every .05 s to disk; online
feedback interpolates those frames at decreasing remaining time.

If interrupted, `progress.json` identifies the completed chunks. Partial runs
are not loadable as completed results. Automatic restart from partial files is
not implemented; `--rollouts-only` resumes work from a completed solve.

## 4. Run a physical trajectory

```python
adapter = jumper_adapter(reach_avoid, p)
trajectory = adapter.run(result, [.1,.35,3.,0.], feedback_dt=.005)
```

The adapter delegates to `HybridFeedback` and the common `simulate` function.
There are no controlled mode choices in this jumper, so feedback resolves the
required stance/flight switch and optimizes the continuous force. The simulator
monitors actual failure and goal events, computational domain exits and
between-sample clearance. It reports unresolved contacts instead of advancing
time artificially.

For an avoid result, use the same adapter with an avoid problem and a nominal
force callback:

```python
adapter = jumper_adapter(avoid, p)
# avoid_result must be produced by solve/solve_streamed(avoid, ...).
trajectory = adapter.run(avoid_result, [.5,.25,0.,0.], feedback_dt=.005,
    margin=.02,
    nominal=lambda state, mode: np.array([10.,20.]) if mode == 0 else np.zeros(2),
    finish_x=5.3)
```

Here `finish_x` is only a simulation stop marker beyond the pillars. It does not
turn the avoid PDE into a reach-avoid problem. The avoid HLRF uses nominal control
above its intervention margin and safety control otherwise. Reach-avoid feedback
is goal seeking and is not described as a least-restrictive filter.

## 5. Use the complete example commands

```bash
JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python examples/papers/rollouts.py jump_avoid
python examples/papers/validate.py jump_avoid

JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python examples/papers/rollouts.py jump_reach_avoid
python examples/papers/validate.py jump_reach_avoid
```

Each run writes a memory-mapped value history, configuration, filtered/nominal
trajectories, reset records, trajectory/value-slice/diagnostic PNGs and numerical
metadata. The separate validation command replays the serialized controls using
the analytic constant-force solution, checking force availability, mode-domain
consistency, timing, goal entry and clearance independently of the policy.

The satellite examples use exactly this workflow with a different adapter and
an explicit `AsinhCoordinates` chart. Its continuous clock advances at 1/60 in
thrust and 1/300 in cooldown, and resets to zero at a switch. Optional idle holds
use one-step lookahead; a source value containing immediate thrust engagement
cannot decide whether actually waiting is safe or desirable.
