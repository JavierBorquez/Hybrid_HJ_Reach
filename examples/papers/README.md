# Paper-model solve recipes

The source models for the RSS and IJRR studies are included here without MATLAB,
PDFs, archived trajectories or private fixtures. From the public export root:

```bash
JAX_ENABLE_X64=true python examples/papers/run.py dog_absorbing
JAX_ENABLE_X64=true python examples/papers/run.py jump_reach_avoid
```

Available cases are `dog_absorbing`, `dog_recoverable`, `aircraft_fixed`,
`aircraft_optimal`, `jump_reach`, `jump_avoid`, `jump_reach_avoid`,
`satellite_avoid` and `satellite_reach_avoid`.

The default `smoke` preset uses small grids; higher-dimensional cases use short
horizons. These check that the unified pipeline runs, not that the tube is
resolved. The `reference` preset selects representative paper/research grids
and horizons, but its new solver semantics have **not** been rerun at full scale.
The default 4 GiB XLA execution budget can reject large configurations. Increase
`--memory-gib` only after assessing available memory. These recipes save just
initial/final frames, insufficient for general finite-horizon rollout feedback.

## Full IJRR simulations

All four IJRR cases have an end-to-end solve/reload/simulate/plot entry point:

```bash
JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python examples/papers/rollouts.py jump_avoid
python examples/papers/validate.py jump_avoid
```

Replace the case with `jump_reach_avoid`, `satellite_avoid` or
`satellite_reach_avoid`. `--preset smoke` exercises short small-grid runs;
`--preset demo` (default) uses the full listed horizon on a moderate grid;
`--preset reference` selects the larger research grid. All runs use finite
horizons; satellite avoid simulates 1200 s, not the private example's 2800 s
stationary-field reuse. The new API does not claim a stationary invariant set.

| Demo | Grid | Horizon | Coupling | Saved frames | Feedback |
|---|---|---|---|---|---|
| Jumper avoid | 71×36×21×31 | 3 s | .005 s | .05 s | .005 s |
| Jumper reach-avoid | 81×51×23×31 | 2 s | .0025 s | .05 s | .005 s |
| Satellite avoid | 41×31×15×15×11 | 1200 s | 2 s | 20 s | 1 s |
| Satellite reach-avoid | 41×31×15×15×11 | 780 s | 2 s | 20 s | 1 s |

Outputs go to `outputs/papers/<case>_<preset>/` or `--output <new-directory>`.
The root contains configuration/provenance, numerical trajectories, reset logs,
whole-device GPU memory samples, and three PNGs. `value/` contains portable
NumPy/JSON results and streamed progress. Full histories stay on disk while
only a chunk's selected frames plus endpoint/workspace occupy GPU memory.
The default XLA execution budget is 5 GiB; it is an estimate, not a guarantee
about total device occupancy. Larger grids can be rejected before execution.

A completed solve can be reused with `--rollouts-only`; use `--solve-only` to
skip simulation initially. `--skip-nominal` omits avoid baseline trajectories.
The separate `validate.py` replays serialized controls using exact constant-force
jumper flow or inertial Kepler satellite dynamics, independently of the feedback
and Hill integration RHS. It checks timers, admissible inputs, decreasing
time-to-go, goal arrival and conditional between-sample clearance. Failed nominal
trajectories remain recorded as failures rather than being silently discarded.

Satellite simulation and plots use physical coordinates; the HJ solver uses the
explicit asinh chart. Gradients, guards and resets use computational coordinates
inside the policy. Exact physical resets and the 60 s/300 s clock events are
handled before control is queried again. Idle candidates explicitly commit to
one passive feedback step, preventing dummy projected-stay decisions.

The larger reference presets are provided for further refinement and have not
all been rerun through this migration. Demo trajectories are finite numerical
examples, not convergence proofs. RSS Dog1D/aircraft/three-mode-jumper solve
recipes remain available in `run.py`; their complete online rollout adapters
are still separate migration work.

Intended source conventions remain visible in each model. Dog's table-guard
priority and aircraft's ceiling reset sampling/timer offsets remain explicit
reproduction choices. Aircraft input ties and dissipation bounds are corrected
in this public copy. Universal zero-horizon projection and forced-reset priority
also differ from some archived BRT runs. Do not describe newly generated fields
as bitwise reproductions of those private results.
