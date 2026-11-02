# Run the paper examples

[Home](../../README.md) · [Tutorial](../../tutorials/README.md) · [IJRR paper](https://doi.org/10.1177/02783649261477777) · [RSS paper](https://roboticsproceedings.org/rss20/p006.pdf)

The falling-piece tutorial gives us a short window to steer before a forced
switch. Here we use the same idea for a jumping robot and a spacecraft:
which states can stay safe, and which can reach a goal without a collision?

Follow the [setup guide](../../docs/SETUP.md) first. Run all commands from the
repository root.

## 1. Choose a system and a task

The **jumping robot** switches between stance and flight. It can apply forces
while its leg touches the ground; in flight it follows a ballistic arc. Avoid
uses a safety filter to modify a nominal controller near danger. Reach-avoid
chooses forces that bring the robot to a target beyond the pillars.

The **spacecraft** switches between idle, thrust and cooldown. The controller
chooses when to start a 60 s thrust maneuver. A forced 300 s cooldown follows,
so each maneuver must account for the period when thrust is unavailable.
Avoid filters a nominal policy; reach-avoid steers toward a target while
avoiding obstacles.

| Case | Task | Demo horizon |
|---|---|---|
| `jump_avoid` | Clear the pillars with safety filtering | 3 s |
| `jump_reach_avoid` | Reach a target beyond the pillars | 2 s |
| `satellite_avoid` | Stay safe with safety filtering | 1200 s |
| `satellite_reach_avoid` | Reach a target around obstacles | 780 s |

The filter is available for avoid only. Reach-avoid uses goal-seeking feedback
from the backward reach-avoid tube (BRAT).

## 2. Try a short CPU run

Start with a small grid and a short horizon to check your installation:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/rollouts.py jump_reach_avoid --preset smoke
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/validate.py jump_reach_avoid --preset smoke
```

Open `outputs/papers/jump_reach_avoid_smoke/trajectories.png`.
The 0.2 s horizon checks the workflow; it is too short to expect goal arrival.

## 3. Run a complete demo

Use `demo` for the full horizon on a moderate grid. With
[CUDA JAX installed](../../docs/SETUP.md#gpu-execution):

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python examples/papers/rollouts.py jump_reach_avoid --preset demo
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/validate.py jump_reach_avoid --preset demo
```

Replace `jump_reach_avoid` in **both commands** with any case in the table.
The first command solves the value function, follows the feedback and plots
the trajectories. The second independently replays the saved controls and
checks inputs, switches, clearance and goal arrival.

Results go to `outputs/papers/<case>_<preset>/`. For reruns, choose a fresh
`--output` directory and pass it to both commands.

| Output | What to look for |
|---|---|
| `trajectories.png` | Paths relative to obstacles and the target |
| `value_slices.png` | Position slices of the value function |
| `diagnostics.png` | Values, modes and controls over time |
| `validation.json` | `successful_finite_trajectory` for each run |
| `configuration.json` | Grid, horizon, initial states and parameters |
| `rollouts.json`, `*.npz` | Outcomes and sampled trajectories |
| `value/` | Saved value history for feedback |

In the recorded demos, the jumper reached its target at about 1.645 s.
Its filtered avoid trajectory cleared both pillars; the nominal trajectory
collided. Both spacecraft reach-avoid starts reached the goal within 780 s,
and both filtered avoid runs completed 1200 s safely.

These are finite trajectory checks on moderate grids. Exact paper-figure
reproduction is incomplete, and the coarse jumper avoid run has only about
0.000508 m of clearance. See the [validation record](../../docs/VALIDATION.md)
for grids, timings and measured outcomes. Nominal collisions and short smoke
runs can report `successful_finite_trajectory: false` even when replay passes.

## 4. Animate saved trajectories

After completing the corresponding demos, render their saved trajectories:

```bash
python examples/papers/animate.py outputs/papers/jump_avoid_demo \
  --output outputs/animations/jumper_filter.gif
python examples/papers/animate.py outputs/papers/satellite_reach_avoid_demo \
  --output outputs/animations/satellite_reach_avoid.gif
```

This only renders the recorded paths. `--frames`, `--fps` and `--hold` adjust
playback. Use a fresh output path for each render.
The [main-page GIFs](../../README.md) use these validated demo trajectories.

## 5. Change a run or reuse its value function

Edit [`scenarios.py`](scenarios.py) to change a demo's grid, horizon, initial
states or feedback interval. Dynamics, guards and resets live in
[`src/hybrid_reach/examples/`](../../src/hybrid_reach/examples/).
Use a fresh output directory after changing a configuration.

To repeat feedback and plotting from a completed matching solve:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/rollouts.py jump_reach_avoid --preset demo --rollouts-only
```

This replaces the trajectory reports and plots in that run. Use `--solve-only`
on the initial command if you want to separate solving from simulation.

`smoke` uses small grids and short horizons, `demo` uses the settings above,
and `reference` selects larger research grids. The reference runs have not all
been validated. Value histories are streamed to disk; `--memory-gib` changes
the execution-estimate budget (default 5 GiB), rather than the GPU's capacity.

## RSS examples: solve the value function

The RSS cases currently provide value-function solves; complete trajectory and
figure drivers are still pending.

| Case | System |
|---|---|
| `dog_absorbing`, `dog_recoverable` | A dog walking or crawling under a table |
| `aircraft_fixed`, `aircraft_optimal` | Aircraft collision avoidance |
| `jump_reach` | A jumper reaching a platform |

For example:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/run.py dog_absorbing --preset smoke
```

Load the result in Python:

```python
from hybrid_reach import Result

result = Result.load('outputs/papers/dog_absorbing_smoke')
print(result.mode_names)    # walking, crawling, frozen
print(result.values.shape) # (2, 3, 81): saved horizon, mode, position
```

[`run.py`](run.py) supports `smoke` and `reference`. It saves only the initial
and final fields, so use `rollouts.py` for the IJRR feedback examples above.
