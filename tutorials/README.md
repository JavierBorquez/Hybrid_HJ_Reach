# Bring your own hybrid system: a vertical jumper

The runnable companion is [jumper.py](jumper.py). It performs all three objectives,
saves and reloads their value histories, and writes a plot and trajectory for each.
Start in the public export root with the environment described in the main README.

## 1. Choose continuous states and modes

Use `x=(z,vz)`: height in metres and vertical velocity in metres per second.
There are two modes, stance and flight. In stance, an upward force is available;
in flight the robot is ballistic. This is a reduced model for teaching the API,
not a new reproduction of the planar paper experiment.

The full model is [vertical_jumper.py](../src/hybrid_reach/examples/vertical_jumper.py).
Its continuous dynamics subclass `hj_reachability.Dynamics` and supply:

1. `__call__(state, control, disturbance, time)` for the vector field.
2. `optimal_control_and_disturbance(...)` for Hamiltonian optimization.
3. `partial_max_magnitudes(...)` for valid dissipation/CFL bounds over the entire
   admissible input space, not only a nominal control.

The optimization routine must honor `self.control_mode` and
`self.disturbance_mode`. The problem layer sets them automatically. This example
has no disturbance; the public solver also accepts continuous adversaries.

```python
model, grid, events = make_system(shape=(81, 81))
```

The first version uses one shared uniform computational grid and finitely many
modes. Modes are a value-array axis, never a PDE differentiation coordinate.

## 2. Declare guards and resets

A `Transition` contains source/destination mode indices, a Boolean guard and a
continuous reset function. Functions accept both a single state and a batch of
grid states, using `states[..., axis]`. This example's resets are identity.

```python
Transition(0, 1, launch_guard, lambda states: states, name='launch')
```

Stance changes to flight at the leg-height surface when departing upward;
descending through that surface returns to stance. A forced stage excludes
staying when its guard is active. A controlled stage includes staying.

Stages describe **Bellman operator composition**: all edges within a stage read
the same value snapshot. Later stages read earlier stages' outputs. The online
filter traverses candidate physical paths in reverse stage order, matching that
composition. Nothing infers a priority or iterates a reset chain to closure.
The jumper needs only one simultaneous forced stage.

Active resets outside a nonperiodic grid raise an error. If clipping is intended,
declare `boundary='clip'` on that edge. That is a modeling/numerical choice, not
an automatic fix. PDE reset sampling defaults to nearest-node; `ceil` is explicit.

## 3. Define goal and failure geometry

All public input set functions are **negative inside**, zero on their boundary.
A callable accepts `(states, mode_index)` and may vary by mode.

```python
def failure(states, mode):
    z = states[..., 0]
    return jnp.minimum(z, 2.8-z)       # floor or ceiling contact

def goal(states, mode):
    return abs(states[..., 0]-1.1)-.1  # target height band
```

The height goal does not impose terminal velocity matching. This is intentional.
Contact with the closed failure set is treated as failure in simulation.

## 4. Select an objective and solve

```python
reach = Problem.reach(model, grid, goal)
avoid = Problem.avoid(model, grid, failure)
reach_avoid = Problem.reach_avoid(model, grid, goal, failure)

horizons = np.linspace(0., .8, 81)
result = solve(avoid, horizons, save_indices=(0, 40, 80))
```

Each consecutive horizon interval is a hybrid coupling step. Saving fewer frames
reduces storage without skipping those updates. Keep enough frames for meaningful
time-dependent feedback; sparse interpolation is a separate approximation.

Reach minimizes the goal cost over arrival times up to the horizon. Avoid
maximizes the minimum clearance along the entire trajectory. Reach-avoid
minimizes the maximum of goal cost at arrival and accumulated failure cost.
Failure has priority over overlapping goal contact. Required resets take
precedence over stopping in the source goal.

Reach and reach-avoid winning states have `V<=0`. Avoid states with `V>0` are
numerically safe for the specified remaining horizon; the failure-tube side is
`V<=0`. These statements describe finite-grid approximations.

## 5. Save and reload

```python
result.save('outputs/example')
result = Result.load('outputs/example')
```

The new directory contains `values.npy`, `axes.npz` and `metadata.json`. Values
have shape `(saved_time, mode, *grid_shape)` and reload as a memory map. Numerical
metadata records coupling times, storage indices, precision, backend and timings.
Model code/configuration remains necessary to reconstruct feedback; arbitrary
Python dynamics and guards are not pickled or reconstructed from JSON.

## 6. Add the avoid HLRF

```python
hlrf = AvoidFilter(avoid, result, margin=.03)

def controller(state, mode, time_to_go):
    return hlrf(state, mode, time_to_go,
                nominal_control=lambda post_state, post_mode: np.zeros(1))
```

The filter evaluates legal reset candidates, accepts a safe nominal candidate
when available, and otherwise selects the candidate with highest safety value.
Forced transitions exclude staying. Intermediate failure costs are retained.
The continuous control is queried **in the selected post-reset mode**. Above the
margin, nominal control passes through; at/below it, maximum-safety control runs.
The callback must return an input admissible for that post-reset mode.

For controlled stages, `nominal_edges=((stage_index, edge_index), ...)` requests
optional edges; `()` requests staying where optional. The small tests demonstrate
controlled-mode intervention; the tutorial jumper has only forced switches.

Finite-horizon policies must receive decreasing `time_to_go`. Reusing the final
field indefinitely is not part of this API. If all candidates are unsafe, the
filter reports `safe_candidate=False` while selecting its best legal action.

Reach/reach-avoid use `ContinuousFeedback` for this deterministic forced-switch
jumper. That is goal-seeking Hamiltonian feedback, not a least-restrictive filter.
Controlled discrete reach scheduling requires a separate explicit policy.

## 7. Simulate physical switching events

Boolean guards used on a grid are insufficient for event-resolved simulation.
Supply signed `GuardEvent` surfaces and crossing directions for your model.
The shared simulator stops integration at those events, applies resets, then
queries the controller again before continuing physical flow.

```python
trajectory = simulate(avoid, controller, [.15, 0.], 0,
    duration=.8, feedback_dt=.005, events=events,
    failure=lambda x, q: float(failure(x, q)))
```

It also monitors domain faces and optional goal/failure events, validates input
bounds, and reports unresolved zero-duration contacts as `zeno_contact`.
ODE event detection can miss an enter-and-exit event within one internal step;
this simulator is not a continuous collision certificate. The private research
experiments retain their independent geometry and replay audits.

## 8. Run and inspect

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python tutorials/jumper.py
```

Inspect `outputs/jumper/{reach,avoid,reach_avoid}/values_and_rollout.png`,
`summary.json` and `trajectory.npz`. Choose a new `--output` directory for reruns;
existing result folders are never silently overwritten.

Change the goal height, failure geometry, dynamics or reset map next. After that,
refine the grid, hybrid coupling interval, saved-frame spacing and feedback
interval separately. A successful trajectory alone does not validate the full
computed set.
