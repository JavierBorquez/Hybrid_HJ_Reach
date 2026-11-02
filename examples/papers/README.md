# Compute the paper BRTs and BRATs

[Home](../../README.md) · [Tutorial](../../tutorials/README.md) · [IJRR paper](https://doi.org/10.1177/02783649261477777) · [RSS paper](https://roboticsproceedings.org/rss20/p006.pdf)

Start by computing the value function. Its zero level set gives the boundary
of a backward reachable tube (BRT) or backward reach-avoid tube (BRAT). Then
plot slices of that set, and use the saved values for feedback or safety filtering.

Follow the [setup guide](../../docs/SETUP.md) first. All commands run from the
repository root.

## 1. Compute the value function

[`run.py`](run.py) selects an example, builds its hybrid model and solves its
reachability problem. For a small IJRR jumper reach-avoid run on CPU:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/run.py jump_reach_avoid --preset smoke
```

This saves the initial and final value fields in
`outputs/papers/jump_reach_avoid_smoke/`. `smoke` checks the calculation on a
small grid and short horizon. To compute the full-horizon tube on a larger grid,
use `reference`, for example with [CUDA JAX](../../docs/SETUP.md#gpu-execution):

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python examples/papers/run.py jump_reach_avoid --preset reference
```

The four IJRR cases are:

| Case | Model called by `run.py` | Task | Reference horizon |
|---|---|---|---|
| `jump_avoid` | [`jump_filter.py`](../../src/hybrid_reach/examples/jump_filter.py) | Avoid the floor and pillars | 4 s |
| `jump_reach_avoid` | [`jump_reach_avoid.py`](../../src/hybrid_reach/examples/jump_reach_avoid.py) | Reach a target beyond the pillars | 2 s |
| `satellite_avoid` | [`satellite.py`](../../src/hybrid_reach/examples/satellite.py) | Avoid obstacles and the lower boundary | 1200 s |
| `satellite_reach_avoid` | [`satellite_reach_avoid.py`](../../src/hybrid_reach/examples/satellite_reach_avoid.py) | Reach a target around obstacles | 780 s |

Inside `run.py`, `make_case()` calls the corresponding module's `make_problem()`.
That supplies the dynamics, grid, guards, resets and sets. The script wraps these
in `Problem.avoid(...)` or `Problem.reach_avoid(...)`, then calls `solve(...)`.
To change the system, edit its model module; to change the solve's grid or
horizon, edit `make_case()`.

As in the tutorial, the jumper switches between stance and flight. The
spacecraft can choose to start thrust, but must then complete 60 s of thrust
and 300 s of cooldown before it can thrust again.

Read the computed field using the same signs as the tutorial:

- **Avoid:** `V <= 0` is the failure BRT; `V > 0` is the safe set.
- **Reach-avoid:** `V <= 0` is the BRAT, from which the target can be reached
  while avoiding failure within the horizon.

Use a fresh output directory for reruns. For `run.py`, `--output outputs/my_values`
sets the parent directory; the script appends `<case>_<preset>`.
Reference grids can be expensive and have not all been validated. `--memory-gib`
sets the execution-estimate budget (default 4 GiB).

## 2. Plot BRT / BRAT slices

A jumper state has four continuous coordinates; the spacecraft has five.
To see a position slice, fix velocity and, for the spacecraft, its mode clock.
Each mode gets its own panel.

Run this Python snippet after the small solve above. Change the result path to
plot another IJRR case or its `reference` result:

```python
import numpy as np
import matplotlib.pyplot as plt
from hybrid_reach import Result

result = Result.load('outputs/papers/jump_reach_avoid_smoke')
# Satellite grids use asinh coordinates; decode the position axes for plotting.
scales = {'satellite_avoid': (6000., 6000.),
          'satellite_reach_avoid': (2500., 2000.)}.get(result.metadata['case'])
x, y = result.coordinates[:2]
if scales is not None:
    x, y = scales[0]*np.sinh(x), scales[1]*np.sinh(y)

# Select the grid velocities nearest zero, and phase zero for the spacecraft.
fixed = tuple(int(np.argmin(abs(axis))) for axis in result.coordinates[2:])
fig, axes = plt.subplots(1, len(result.mode_names), squeeze=False,
                         figsize=(5*len(result.mode_names), 4))
for q, ax in enumerate(axes[0]):
    field = result.values[(-1, q, slice(None), slice(None), *fixed)]
    ax.contourf(x, y, (field <= 0).T, levels=[-.5, .5, 1.5],
                colors=['white', 'salmon' if result.kind == 'avoid' else 'palegreen'])
    if field.min() < 0 < field.max():
        ax.contour(x, y, field.T, levels=[0.], colors='black')
    ax.set(title=result.mode_names[q], xlabel='x [m]', ylabel='y / height [m]')
tube = 'BRT (red)' if result.kind == 'avoid' else 'BRAT (green)'
fig.suptitle(f'{tube}, horizon {result.horizons[-1]:g} s; nearest-zero velocity slice')
fig.tight_layout()
fig.savefig('value_slices.png', dpi=150)
```

`result.values` is indexed by `(saved horizon, mode, *state coordinates)`.
`-1` selects the full computed horizon. Change `fixed` to inspect other velocities
or clock phases. A short `smoke` horizon may show only a small part of the tube.

## 3. Use the values as feedback or a safety filter

A trajectory queries the field at decreasing time-to-go, so it needs saved
intermediate horizons. `run.py` saves only the endpoints. For feedback, prepare
a time history with [`rollouts.py`](rollouts.py) in solve-only mode:

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python examples/papers/rollouts.py jump_reach_avoid --preset demo --solve-only
```

This computes a new field using the moderate `demo` grid and saves its history
in `outputs/papers/jump_reach_avoid_demo/value/`. It does not reuse the
endpoint-only `run.py` result. [`scenarios.py`](scenarios.py) defines these grids,
horizons, initial states and feedback intervals. `demo` horizons are 3 s for
jumper avoid, 2 s for jumper reach-avoid, 1200 s for spacecraft avoid and 780 s
for spacecraft reach-avoid. For a short CPU check, set `JAX_PLATFORMS=cpu`
and use `--preset smoke`.

Then follow the feedback and draw trajectories without solving again:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/rollouts.py jump_reach_avoid --preset demo --rollouts-only
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/validate.py jump_reach_avoid --preset demo
```

Replace the case in all three commands to run any other IJRR example:

- **Reach-avoid:** choose continuous inputs and available mode switches using
  the value function to reach the goal without collision.
- **Avoid:** apply the nominal policy when safe; otherwise use the safety
  feedback. The script draws both nominal and filtered trajectories.

The [model adapters](../../src/hybrid_reach/examples/adapters.py) build the
controllers and integrate the physical dynamics. Edit `Scenario.nominal()` in
[`scenarios.py`](scenarios.py) to filter your own admissible nominal policy.
The safety filter is available for avoid only.

Open `trajectories.png` in the run directory. `value_slices.png` shows slices
at the first initial state's velocity and phase; `diagnostics.png` shows values,
modes and controls over time. `validate.py` independently replays the saved
controls and records clearance and goal-arrival checks in `validation.json`.
Use the same case, preset and `--output` directory for all three commands.

These are finite-grid results, and exact paper-figure reproduction is incomplete.
The recorded demo trajectories pass independent checks, but the jumper avoid
run has tight clearance. See the [validation record](../../docs/VALIDATION.md)
for measured outcomes and grids. Short smoke runs need not reach the goal.

## 4. Animate the trajectories

After completing the corresponding runs:

```bash
python examples/papers/animate.py outputs/papers/jump_avoid_demo \
  --output outputs/animations/jumper_filter.gif
python examples/papers/animate.py outputs/papers/satellite_reach_avoid_demo \
  --output outputs/animations/satellite_reach_avoid.gif
```

This renders saved paths. Use a fresh output path; `--frames`, `--fps` and
`--hold` adjust playback.

## RSS value-function examples

Use the same `run.py` command with one of these cases:

| Case | System / task | Model module |
|---|---|---|
| `dog_absorbing`, `dog_recoverable` | A dog reaching the other side of a table | [`dog1d.py`](../../src/hybrid_reach/examples/dog1d.py) |
| `aircraft_fixed`, `aircraft_optimal` | Aircraft collision avoidance | [`aircoll3d.py`](../../src/hybrid_reach/examples/aircoll3d.py) |
| `jump_reach` | A jumper reaching a platform | [`jump4d.py`](../../src/hybrid_reach/examples/jump4d.py) |

`make_case()` wraps reach cases in `Problem.reach(...)`; their BRT is `V <= 0`.
Complete RSS trajectory and figure drivers are still pending. The position-slice
snippet above is for the four IJRR cases.
