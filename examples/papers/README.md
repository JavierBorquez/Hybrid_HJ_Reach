# Run the paper experiments

[Home](../../README.md) · [IJRR paper](https://doi.org/10.1177/02783649261477777) · [Tutorial](../../tutorials/README.md)

This guide maps the paper examples to runnable commands and tells you what each
command produces. Start with the short CPU workflow below. Then run a complete
planar-jumper demo before moving to the larger spacecraft grids.

Install the package with `.[examples,test]` using the [setup guide](../../docs/SETUP.md).
All commands run from the repository root. You do not need MATLAB or downloaded
solution arrays.

## What is reproduced?

The examples come from two studies:

- RSS: *Hamilton-Jacobi Reachability Analysis for Hybrid Systems with Controlled
  and Forced Transitions* (Borquez et al.).
  [Paper](https://roboticsproceedings.org/rss20/p006.pdf).
- IJRR study: *Hamilton-Jacobi Reachability for Hybrid Systems: Unified Goal-Driven
  Control with Safety Guarantees* (Borquez, Peng and Bansal).
  [Published paper](https://doi.org/10.1177/02783649261477777) · [Preprint](https://arxiv.org/abs/2609.17430).

The public package implements these example models with documented changes in
hybrid projection, numerical bounds and feedback. The status is:

| Study / example | Case name | Available workflow | Current evidence |
|---|---|---|---|
| RSS: Dog1D under a table | `dog_absorbing`, `dog_recoverable` | Solve and save values | Small-grid execution tests; full rollout/figure driver pending |
| RSS: aircraft collision avoidance | `aircraft_fixed`, `aircraft_optimal` | Solve and save values | Small-grid execution tests; full rollout/figure driver pending |
| RSS: jumper reaching a platform | `jump_reach` | Solve and save values | Small-grid execution test; full rollout/figure driver pending |
| IJRR: jumper safety filtering, Figs. 3–4 | `jump_avoid` | Solve, simulate, plot and replay | Validated moderate-grid finite trajectory |
| IJRR: jumper reach-avoid, Fig. 7 | `jump_reach_avoid` | Solve, simulate, plot and replay | Validated moderate-grid goal arrival |
| IJRR: spacecraft safety filtering, Figs. 5–6 | `satellite_avoid` | Solve, simulate, plot and replay | Two validated 1200 s finite trajectories |
| IJRR: spacecraft reach-avoid, Fig. 8 | `satellite_reach_avoid` | Solve, simulate, plot and replay | Two validated goal arrivals within 780 s |

**Exact paper-figure reproduction remains incomplete.** In particular, the new
finite-horizon spacecraft avoid runs use different starts and duration from the
archived stationary-field experiment, and the full RSS plotting workflows have
not been migrated. The larger `reference` presets provide research-scale settings
for further experiments; they have not all been validated under the public
solver. Use the moderate `demo` runs for the recorded public trajectory results.

The numerical differences are listed [below](#differences-from-the-paper-and-matlab-runs),
with decision details in [CHANGELOG.md](../../CHANGELOG.md). Successful trajectories
are evidence for those finite numerical runs, not convergence of the full tube.

## Choose a script and a preset

| Script | Purpose | Default preset |
|---|---|---|
| `examples/papers/rollouts.py` | Complete IJRR solve → simulate → plot workflow | `demo` |
| `examples/papers/validate.py` | Independently replay the saved IJRR controls | `demo` |
| `examples/papers/run.py` | Value-only solves for all nine case names | `smoke` |

For `rollouts.py`, `smoke` is a small grid with a short horizon, `demo` uses the
full listed horizon on a moderate grid, and `reference` uses a larger research
grid. `run.py` supports `smoke` and `reference` only. Its `smoke` horizons vary by
case; high-dimensional cases use short horizons.

## 1. Check the workflow on CPU

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/rollouts.py jump_reach_avoid --preset smoke
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/validate.py jump_reach_avoid --preset smoke
```

Open `outputs/papers/jump_reach_avoid_smoke/trajectories.png`. You should also
find `diagnostics.png`, `value_slices.png`, `rollouts.json` and `validation.json`.
The short 0.2 s horizon exercises the pipeline; it need not reach the goal.
A `successful_finite_trajectory: false` entry is therefore possible in this
check even when execution and replay pass.

Both commands must use the same case, preset and output path. For a rerun, add
`--output outputs/my_smoke_run` to **both** commands and choose a new directory.

## 2. Run the complete jumper demo

With a working CUDA JAX installation (see [setup](../../docs/SETUP.md#gpu-execution)):

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python examples/papers/rollouts.py jump_reach_avoid --preset demo
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/validate.py jump_reach_avoid --preset demo
```

The default output is `outputs/papers/jump_reach_avoid_demo/`. In the recorded
validation run the robot entered the target disk at about 1.645 s, before the
2 s deadline, with a conditional clearance lower bound of about 0.132 m.
Compilation/backend differences can alter the numerical trajectory.

For the safety-filter experiment, replace `jump_reach_avoid` with `jump_avoid`
in both commands. That run compares filtered and nominal controls. The recorded
filtered robot passed both pillars at about 1.648 s; the nominal robot collided
at about 0.770 s. The filtered clearance bound was only about 0.000508 m on this
coarse grid, so further refinement matters.

## 3. Run the spacecraft demos

Use either spacecraft case with the same two-command workflow:

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python examples/papers/rollouts.py satellite_reach_avoid --preset demo
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/validate.py satellite_reach_avoid --preset demo
```

Replace the case with `satellite_avoid` for safety filtering. Each spacecraft
case runs two initial states, named `low_velocity` and `high_velocity`. The
recorded reach-avoid arrivals were about 759 s and 723 s. The filtered avoid
trajectories completed 1200 s; both nominal avoid baselines collided.

The model has idle, thrust and cooldown modes. Thrust lasts 60 s and cooldown
lasts 300 s. Its fifth state is a normalized clock. The PDE uses an explicit
asinh coordinate chart; trajectories and figures use physical coordinates.
Optional waiting in idle is evaluated as a one-feedback-step commitment.

The complete demo configurations are:

| Case | Grid | Horizon | Hybrid coupling | Saved-frame interval | Feedback interval |
|---|---|---|---|---|---|
| `jump_avoid` | 71×36×21×31 | 3 s | .005 s | .05 s | .005 s |
| `jump_reach_avoid` | 81×51×23×31 | 2 s | .0025 s | .05 s | .005 s |
| `satellite_avoid` | 41×31×15×15×11 | 1200 s | 2 s | 20 s | 1 s |
| `satellite_reach_avoid` | 41×31×15×15×11 | 780 s | 2 s | 20 s | 1 s |

On the recorded RTX 4060 Laptop, demo PDE execution took approximately 34 s,
62 s, 151 s and 149 s respectively, plus compilation. These times exclude
simulation, plotting and independent replay. See [the validation record](../../docs/VALIDATION.md)
for the environment, memory measurements and full outcomes.

Histories are streamed to disk. The default XLA execution-estimate budget is
5 GiB; total GPU usage can be higher. `--memory-gib` changes that budget, not
the hardware capacity. The larger `reference` runs can be expensive or rejected
by the budget.

## 4. Replay and check the trajectories

`validate.py` replays serialized controls using exact constant-force jumper flow
or independent inertial Kepler spacecraft dynamics. It also checks admissible
inputs, mode clocks, decreasing remaining time, reset records, goal arrival and
conditional clearance between stored samples.

Inspect `validation.json`, especially `successful_finite_trajectory` for each
filtered/goal-seeking run. A zero process exit alone does not mean every
trajectory succeeded: colliding nominal baselines are retained and reported as
unsuccessful. Smoke runs may also report unsuccessful goal arrival because of
their short horizon.

Use the same case, preset and directory as the solve. For example, if you choose
a custom destination:

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python examples/papers/rollouts.py jump_avoid --preset demo --output outputs/my_jump
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/validate.py jump_avoid --preset demo --output outputs/my_jump
```

## Read and reuse the outputs

| File or directory | Contents |
|---|---|
| `trajectories.png` | Physical trajectories, obstacles and goal |
| `diagnostics.png` | Queried values, modes and controls over time |
| `value_slices.png` | Two-dimensional position slices at the initial velocity/phase |
| `configuration.json` | Grid, horizons, initial states and physical parameters |
| `rollouts.json` | Status, clearance, interventions and reset logs for each run |
| `<trajectory_name>.npz` | Sampled states/controls and pre-reset segment endpoints |
| `validation.json` | Independent replay and trajectory outcome checks |
| `value/` | Saved value history, axes, metadata and streamed progress |
| `gpu_memory.csv` | Whole-device memory samples when GPU monitoring is available |
| `code_snapshots/`, `*_source_sha256.json` | Source used for the solve and rollout |

A completed solve can be reused without solving the PDE again:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/rollouts.py jump_reach_avoid --preset demo --rollouts-only
```

This rewrites the trajectory reports and plots using the saved field. It requires
a completed matching solve. To separate the stages initially, add `--solve-only`
to the original solve command, then use `--rollouts-only` later. Partial streamed
solves cannot yet be restarted automatically. `--skip-nominal` omits avoid
baseline trajectories.

## Animate a completed run

Render the saved jumper safety-filter or spacecraft reach-avoid trajectories
without recomputing the value function or feedback. The animation script reads
`configuration.json`, `rollouts.json` and the trajectory `.npz` files:

```bash
python examples/papers/animate.py outputs/papers/jump_avoid_demo \
  --output outputs/animations/jumper_filter.gif
python examples/papers/animate.py outputs/papers/satellite_reach_avoid_demo \
  --output outputs/animations/satellite_reach_avoid.gif
```

Both commands require the corresponding completed run. Use a fresh output path
for rerenders. `--frames`, `--fps` and `--hold` control playback only. Positions
are linearly interpolated between saved physical samples; modes, controls and
intervention flags follow the recorded samples. Each path stops at its recorded
endpoint, retaining nominal collisions. A JSON sidecar records the source hashes,
saved outcomes and rendering settings. The script uses NumPy, Matplotlib and
Pillow from the `examples` extra.

The [main-page previews](../../docs/PAPER_ANIMATIONS.md) use the previously
validated demo trajectories. Their generated GIFs ship with the documentation;
the large numerical histories remain outside the source export.

## RSS value-function recipes

For Dog1D, begin with:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/run.py dog_absorbing --preset smoke
```

This writes values, axes and metadata to `outputs/papers/dog_absorbing_smoke/`.
It does not write a figure or run a controller. You can inspect it in Python:

```python
from hybrid_reach import Result
result = Result.load('outputs/papers/dog_absorbing_smoke')
print(result.mode_names)    # walking, crawling, frozen
print(result.values.shape) # (2, 3, 81): saved horizon, mode, position
```

Use `dog_recoverable`, `aircraft_fixed`, `aircraft_optimal` or `jump_reach` to
select another RSS case. `--preset reference` selects its larger settings.
The value-only script defaults to a 4 GiB execution-estimate budget and saves
only initial/final frames; these histories are too sparse for general
finite-horizon feedback. It also accepts the four IJRR names, but use
`rollouts.py` when you want their complete trajectory and plotting workflow.

## Change an experiment

Edit [`scenarios.py`](scenarios.py) to change an IJRR grid, horizon, initial
state or feedback interval. It assembles the model and records the configuration
with each result. Model definitions are in
[`src/hybrid_reach/examples/`](../../src/hybrid_reach/examples/); physical events
and controller adapters are in
[`adapters.py`](../../src/hybrid_reach/examples/adapters.py).
[`run.py`](run.py) contains the value-only recipe settings.

Use a new output directory when changing a configuration. Keep model choices
explicit and compare refinements separately from paper/reference comparisons.

## Differences from the paper and MATLAB runs

The following choices affect interpretation of newly computed results:

- Immediate hybrid projection applies even at horizon zero. Required resets
  take priority over stopping in a source goal, and intermediate reset failure
  is preserved.
- Dog1D retains explicit table-guard and projection-stage conventions. Aircraft
  retains ceiling reset sampling and timer offsets, but corrects input ties and
  dissipation bounds in this public implementation.
- The RSS platform-jumper recipe uses the `paper` model variant: stance force
  bounds of 30 N on both axes and explicit closed-terrain collision semantics.
  The model also contains a separately named MATLAB-comparison variant; the
  recipe does not select it.
- Public spacecraft feedback uses decreasing time-to-go and stops at the computed
  finite horizon. The avoid demo uses 1200 s and the two starts shared with
  reach-avoid. It does not repeat the archived 2800 s stationary-field rollout
  or establish an infinite-horizon invariant set.
- Reach and reach-avoid feedback are goal seeking. The least-restrictive safety
  filter is supplied for avoid only.

See [CHANGELOG.md](../../CHANGELOG.md), [API.md](../../docs/API.md) and
[VALIDATION.md](../../docs/VALIDATION.md) for the detailed decisions and evidence.
