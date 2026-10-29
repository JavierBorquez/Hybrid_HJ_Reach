# Preparation validation

Validated 2026-10-23 with Python 3.10, JAX/JAXlib 0.6.2, float64 and
hj_reachability 0.7.0. The native GPU is an NVIDIA RTX 4060 Laptop.

- **29 GPU tests passed** in 36.77 s.
- **29 CPU tests passed** in 20.10 s in a fresh allowlisted source export.
- The exported copy built as `hybrid_reach_gpu-0.2.0.dev0-py3-none-any.whl`.
- The installed wheel ran the complete tutorial on CPU, independently of the
  private package source. Reach and reach-avoid entered the goal band; the avoid
  HLRF completed its 0.8 s horizon without a detected failure.
- Native GPU tutorial also completed all three cases, saving fields,
  trajectories, metadata and plots. Each PDE solve used about 0.105 s GPU
  execution after 1.75–1.81 s compilation. No allocation OOM occurred.

Coverage includes independent forced-translation reach/avoid/RA checks, a failure
barrier before the goal, mandatory resets at zero horizon, intermediate failure,
forced-versus-controlled transitions, reverse Bellman composition, post-reset
gradient selection, goal/guard ODE events, sparse history equivalence, local
nonuniform interpolation, strict reset bounds, aircraft Hamiltonian corners,
execution-budget rejection, all nine paper-model smoke solves and export hashes.

The small tutorial is a numerical smoke test. ENO stencil and feedback decisions
can differ between CPU and GPU; exact trajectory agreement is not an acceptance
criterion. Neither successful rollouts nor this suite certify the full tubes,
prove time convergence, establish infinite-horizon invariance or validate the
new fine-grid paper presets. Complete raw outputs remain in the private
development archive and are excluded by the export tool.

## Planar/satellite migration, 2026-10-25

The expanded GPU suite passes **42 tests** in 47.82 s. New checks cover all
three streamed objectives versus full history, sparse/empty chunks, no repeated
reset projection at joins, failure loading, physical asinh gradients/flows,
idle commitment at reduced time, strict prediction-domain handling, clock
events, CPU/GPU grid roundoff and logged machine-precision root normalization.
The fresh curated export also passes **42 CPU tests** in 24.78 s. Its installable
wheel runs all four complete short smoke commands, including solve, reload,
physical simulation, source snapshots and PNG generation, with no private-source
imports or MATLAB fixtures.

All four moderate-grid demo PDEs ran on GPU. Independent physical replay and
conditional dense-output geometry checks verify six successful finite
filtered/goal-seeking trajectories:

| Case | Outcome | Conditional clearance lower bound |
|---|---|---|
| Jumper avoid | Past both pillars at 1.647720 s | 0.000508 m |
| Jumper reach-avoid | Goal at 1.645302 s | 0.132194 m |
| Satellite avoid, low/high starts | Both finish 1200 s | 146.195 / 150.873 m |
| Satellite reach-avoid, low/high starts | Goal at 758.990 / 723.490 s | 385.993 / 260.069 m |

Nominal avoid baselines collide at 0.770228 s (jumper) and 370.063 / 695.445 s
(satellite). Exact jumper flow reproduces the serialized segments within
5e-16. Independent inertial satellite replay differs by at most 2.4e-8 m in the
reported physical state components. Numerical replay agreement is not physical
docking precision or an interval ODE guarantee.

The jumper avoid clearance is tight on this coarse grid; it is a visible
refinement limitation, not a robust safety certificate. The satellite avoid
starts are shared with the reach-avoid demo, differing from the archived private
avoid figure starts. Feedback uses decreasing time-to-go and stops at the
finite computed horizon; stationary infinite-horizon reuse is not inferred.

Synchronized GPU execution times were 33.96 s (jumper avoid), 62.30 s (jumper
reach-avoid), 151.39 s (satellite avoid) and 149.00 s (satellite reach-avoid),
plus 5.12 / 8.05 / 14.60 / 15.01 s compilation respectively. Whole-device sampled
peaks for the first three were 2344 / 3532 / 3382 MiB of 8188 MiB. The fourth
had a 3557 MiB in-run sample, not a complete peak trace. No allocation OOM
occurred. Execution estimates exclude desktop and other resident allocations.

The initially false satellite domain exit at phase 1.0000000000000002 is retained
as a private diagnostic prefix. The corrected run reuses the same value history;
only already-located event roots within eight float64 epsilons are normalized,
with each nonzero correction logged. No physical trajectory/predicted state is
clipped to rescue an out-of-domain rollout.

## Falling-piece tutorial, 2026-10-29

The CPU suite passes **52 tests**, including ten new falling-piece/terminal-event
cases. The new model's Hamiltonian controls are checked against input-box
corners. Reach probes agree with independent steering-window and descent-time
conditions; reach-avoid excludes a reachable column blocked above the goal.
Exact fall times validate the forced switches and absorbing rest mode. A
straight segment with safe endpoints but an interior collision checks the
independent obstacle audit.

The full 101×141, T=4 s CPU lesson generates all three BRT plots and four policy
GIFs. Reach and reach-avoid starts A/B arrive at 2.26/2.66 s; their too-late start
C misses and comes to rest. Avoid safety starts A/B complete the 4 s horizon;
the too-late start collides at 0.33 s. From the shared filter-comparison start,
nominal rightward drift collides at 2.79 s, while the filtered trajectory finishes
in rest without collision. Each run retains its status and independent
straight-line replay/whole-segment failure-field audit in `summary.json`.

The first diagnostic runs exposed guard-membership roundoff at sampled height
boundaries, terminal goal roots incorrectly treated as unresolved contacts, and
empty event-list handling in rest. The diagnostic outputs are retained privately.
The model's declared 1e-10 guard slack never changes a state; terminal-event
handling now retains located roots without clipping or artificial time advance.
See PUB-22–24 for code locations and export changes. These are CPU finite-grid
examples; no new GPU, paper-reference or convergence result is inferred.
