# API and numerical semantics

## System and problem

`HybridModel(modes, stages)` groups named `Mode` dynamics and ordered
`ProjectionStage` operators. `Transition` holds source, destination, guard,
reset, sampling (`nearest`/`ceil`), boundary (`error`/`clip`) and an optional name.
Mode indices are integers starting at zero. The reset function returns an array
with the same shape as its input. Guards return the corresponding Boolean batch.

`Problem.reach(model, grid, goal)`, `Problem.avoid(model, grid, failure)` and
`Problem.reach_avoid(model, grid, goal, failure)` accept scalar/grid/mode-grid
arrays or set callables `(states, mode_index)`. All sets are negative inside.
Dynamics must honor the optimization flags set by `Problem.configured_model()`.
This makes shallow copies; user dynamics must not change nested mutable model
parameters while solving.

| Objective | Controller | Continuous disturbance | Value interpretation |
|---|---|---|---|
| Reach | min | max | V≤0: reaches goal by horizon |
| Avoid | max | min | V>0: avoids closed failure through horizon |
| Reach-avoid | min | max | V≤0: reaches while respecting constraints |

These are numerical set interpretations, not certified classifications. At
zero-cost boundaries, exact touching and discretization errors require care.
Reach-avoid V=0 alone does not establish strict clearance from a closed failure
set; simulations give failure contact priority.

Writing `g` for the goal and `f` for the negative-inside failure field, the
continuous stopping clamps are `min(V,g)`, `min(V,f)` and `max(-f,min(V,g))`.
Terminal fields are `g`, `f` and `max(g,-f)`, respectively, followed by immediate
hybrid projection even at zero horizon.

Controlled stages are reduced in the controller's direction, including stay.
Forced-stage min/max reduction is an explicit model choice for nondeterminism.
Stages are Bellman operators applied in listed order to snapshots. Online
physical candidate enumeration uses reverse composition order. There is no
automatic transitive closure. Cyclic, instantaneous hybrid behavior must be
resolved by the model, not silently iterated by the solver.

Reach/RA mandatory resets cannot be undone by stopping in the source goal.
Avoid projection retains the minimum clearance at every intermediate reset.
Reach-avoid projection retains the maximum failure cost at every intermediate
reset. The running failure constraint therefore survives multiple stages.

## Solver and results

`solve(problem, horizons, save_indices=None, cfl=.8, sequential_modes=True,
upwind_scheme=None, time_integrator=None, memory_budget_bytes=None)` uses ENO3
and third-order TVD RK by default. Horizons start at zero and increase strictly.
Both zero and final checkpoints must be saved. Every interval is executed,
including those without a saved frame. Sequential modes reuse temporary
workspace while reading the same previous mode snapshot.

The grid's boundary functions remain explicit `hj_reachability` choices. No
universal cost floor or constant ghost extension is silently enabled. The
satellite reach-avoid recipe explicitly retains its justified -200 m running
cost floor and constant ghosts.

`memory_budget_bytes` checks XLA's execution estimate after compilation. It does
not measure whole-device occupancy, compilation memory or all resident inputs;
it cannot guarantee that a large grid will fit. `solve` stores all selected
checkpoints on-device until completion. `solve_streamed(problem, horizons,
directory, save_indices=..., chunk_steps=...)` instead writes checkpoints to a
host memory map, keeping only the current endpoint, chunk output and workspace
on device. Chunk executables reuse dynamic times and initial arrays. The initial
terminal/reset projection runs once, never at continuation joins. This is tested
against full histories for all three objectives. A failed/partial run has no
complete metadata and cannot be loaded as a complete result. Automatic restart
from partial progress is not yet supported. Fine-grid reference presets still
need separate validation.

`Result.save(new_directory)` writes plain NumPy numerical files and JSON. Values
reload with mmap using `Result.load(directory)`. No Python objects are unpickled.
Result metadata is not a serialized model: preserve model code, configurations
and versions along with each research result. `ValuePolicy` checks objective,
mode names and grid coordinates, but cannot prove two arbitrary dynamics
functions or parameter sets are identical.

## Policies and simulation

`ValuePolicy.query(state, mode, time_to_go)` returns an interpolated value and
all computational-coordinate derivatives. It reads only local cell/stencil
entries, including for memory-mapped histories. It never extrapolates in time
or space. `value(...)` omits derivative stencils when only a candidate score or
plot value is required. Controls are evaluated on CPU even for GPU-solved fields.
`query_physical(..., coordinates=chart)` applies the diagonal derivative chain
rule. `IdentityCoordinates` and `AsinhCoordinates` expose encode/decode and
inverse-Jacobian operations; `PhysicalPolicy` wraps a computational controller
for physical-state simulation. Backend grid roundoff is accepted only within
eight epsilons; interpolation always uses the saved axes.

`ContinuousFeedback` supplies Hamiltonian control for deterministic forced-only
models. `HybridFeedback` supports optional goal-seeking transitions when an
explicit `HoldLookahead` is supplied for each controlled source mode. It compares
an immediate reset with one feedback-step commitment to staying, then evaluates
the continuation at reduced time-to-go. Model-supplied passive flow and running
failure checks make this approximation explicit; it is not an optimal continuous
switch-time search. No reach/RA least-restrictive filter is supplied.

`AvoidFilter(problem, result, margin=0., lookahead=None)` implements sampled avoid-only HLRF
decisions. `nominal_control(post_reset_state, post_reset_mode)` must return an
admissible box input. Optional `nominal_edges` requests controlled transitions
as `(stage_index, edge_index)` tuples in physical traversal order. It accepts a
safe admissible nominal choice, otherwise maximizes candidate safety; if none
is safe, it still returns its best legal choice with `safe_candidate=False`.

The result is a `Decision`: post-reset state/mode, continuous control, value,
continuous/discrete intervention flags, safe-candidate flag and transition trace.
Safety fallback runs at or below the numerical margin. A margin is not an error
bound or a convergence claim. Time-varying finite-horizon fields use decreasing
time-to-go, not indefinite stationary reuse. Optional passive waiting must be
evaluated as an actual commitment when its projected value includes immediate
switching. The satellite adapter supplies `HoldLookahead` for idle; a predicted
domain exit blocks that candidate instead of clipping the predicted state.

The initial online implementation supports nonperiodic grids, box controls and
deterministic forced exits. The solver also supports periodic reset wrapping
and forced nondeterminism, but those features need additional online machinery.
PDE resets use sampled values; online resets use the exact map unless clipping
is explicitly declared. Their discrepancy requires refinement checks.

`simulate(...)` accepts signed model-supplied `GuardEvent` surfaces and runs
DOP853 up to the first event or feedback sample, requerying after resets. It
checks intermediate reset failure, admissible inputs and computational domain
faces. Supply an explicit disturbance policy if the disturbance dimension is
nonzero. Autonomous models only; this API does not promise correct handling of
arbitrary time-dependent dynamics through its negative PDE integration time.

The optional coordinate chart maps physical simulator states to the model's
computational guards/resets. A physical `flow` callback may integrate a separate
NumPy RHS, or the default converts computational flow using the chart Jacobian.
`GuardEvent.domain_face=(axis, sign)` explicitly gives a reset event responsibility
for a lower (-1) or upper (+1) domain face, as for satellite phase=1. `fixed_axes`
omits identically zero domain events for held coordinates such as idle phase=0.
The supplied coordinate charts are separable and monotone; arbitrary coupled
coordinate transformations need different domain-event machinery.

An already-located declared face root is canonicalized only within eight
float64 epsilons of that face; each nonzero correction is recorded. Larger
errors raise. This handles a clock root such as 1.0000000000000002 without
treating it as a physical domain exit. Arbitrary states and predictions are
never clipped by this root-only operation.

`segment_audit` may evaluate dense physical ODE output. Provided paper adapters
use conditional Lipschitz clearance checks, refining near obstacles. Constant
acceleration gives the jumper an endpoint speed bound; satellite bounds use the
physical velocity box. Serialized `segment_end_state` preserves pre-reset states,
while trajectory samples contain post-reset states at the next feedback query.
Reset records store both sides, enabling independent clock and physical replay.

Root detection and floating-point ODE tolerances are not interval verification.
Tangential or multiple contacts within one integration step may need independent
model-specific clearance checks. An unresolved zero-time contact is reported
as `zeno_contact`, never bypassed using state clipping or artificial time advance.
