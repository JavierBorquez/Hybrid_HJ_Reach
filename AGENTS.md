# Working on hybrid reachability

This directory is the self-contained public source tree. In the research
checkout it is developed privately and exported with `tools/curate.py`.
Prioritize correct hybrid dynamics and HJVI semantics over MATLAB agreement.

## Read first

- `README.md`: setup and entry points.
- `docs/OVERVIEW.md`, `tutorials/README.md`, `examples/papers/README.md`: reader
  paths from concepts to a first run and paper experiments. Keep their commands,
  expected outputs and reproduction-status claims aligned with the actual code.
- `docs/API.md`: signs, reset priority, coordinates and supported features.
- `CHANGELOG.md`: stable PUB decision IDs and intentional source differences.
- `docs/VALIDATION.md`: measured evidence and unresolved numerical limitations.

Keep changes within this tree unless the task also concerns the enclosing
research repository. Examples and tests must work without that repository's
modules, references, MATLAB installation or saved results.

## Environment and import selection

Run commands from the directory containing this file. Use a separate environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[examples,test]'
.venv/bin/python -c "import hybrid_reach; print(hybrid_reach.__file__)"
```

The printed module must belong to this tree's `src/hybrid_reach/`. The enclosing
research checkout has a different package with the same import name. Do not
overwrite its editable installation or use a successful import as proof that
the intended version was loaded.

Only when working inside that private checkout, its existing environment may be
reused without installing this package over the private one:

```bash
PYTHONPATH="$PWD/src" ../.venv/bin/python -c "import hybrid_reach; print(hybrid_reach.__file__)"
```

Use the same `PYTHONPATH` prefix for subsequent commands with that interpreter.
The standalone public tree does not require a parent environment.

## Routine checks

With this tree's environment, the CPU suite and small tutorial are:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true .venv/bin/python -m pytest -q
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true .venv/bin/python tutorials/jumper.py --output outputs/agent_jumper
```

Choose a fresh output directory for reruns. Existing results are deliberately
protected from accidental overwrite. Select focused tests for small changes;
run the full numerical suite when changing solver, projection or policy behavior.

With GPU-enabled JAX already installed:

```bash
JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m pytest -q
JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python examples/papers/rollouts.py jump_avoid --preset smoke --output outputs/agent_jump_avoid
```

Other rollout cases are `jump_reach_avoid`, `satellite_avoid` and
`satellite_reach_avoid`. Start with `--preset smoke`; `demo` is the default and
uses full finite horizons, while `reference` can be expensive. For a completed
demo, independently audit its serialized controls:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  .venv/bin/python examples/papers/validate.py jump_avoid --preset demo --output outputs/my_demo
```

The validator's case/preset must match the saved run. Short reach-avoid smoke
runs exercise execution; they need not reach the goal. Use `--solve-only` and
`--rollouts-only` to separate expensive PDE work from completed-result feedback.
Do not silently substitute CPU execution for a requested GPU validation.

## Architecture and mathematical invariants

- `model.py`, `problems.py`: modes, guards, resets and objective configuration.
  Model-specific connections belong in example/model modules, not the solver.
- `solver.py`, `transitions.py`: generic continuous propagation and hybrid
  projections. Every coupling interval runs even when history storage is sparse.
- `policies.py`, `paths.py`, `lookahead.py`, `filtering.py`: value queries,
  transition candidates, hold commitments and avoid-only HLRF.
- `coordinates.py`, `simulation.py`, `examples/adapters.py`: explicit state
  charts, physical integration, event handling and model adapters.

All input sets are negative inside. Avoid values are positive-safe;
reach/reach-avoid winning values are nonpositive. HLRF is provided for avoid
only; goal-seeking reach/RA feedback is not a least-restrictive filter.

Forced transitions exclude staying. Edges within a Bellman stage read one
snapshot; online physical paths traverse composed stages in reverse order.
Required resets must not be undone by stopping in the source goal. Failure
incurred at a source or intermediate reset must survive later stages.

Streamed continuation applies initial projection once, not again at chunk joins.
Optional idle decisions must account for actual holding: a projected stay value
already includes immediate switching. Query controls after resets, in the
selected mode and correct coordinate chart, at decreasing time-to-go.

Out-of-domain resets error unless clipping is explicitly declared. Never hide a
domain exit or rescue a trajectory by clipping states/predictions. The documented
eight-epsilon normalization applies only to an already-located declared boundary
root and must remain logged. Report unresolved zero-time contacts explicitly.

## GPU memory and numerical claims

Use `solve_streamed` and sequential mode workspace for large histories. Memory
budgets check an XLA estimate, not all resident GPU allocations. Preserve and
report memory failures; do not silently reduce precision, alter physical bounds,
skip coupling updates or change problem semantics to make a run fit.

Distinguish execution smoke tests, independent trajectory audits, numerical
refinement and convergence/certification. Successful finite trajectories do not
establish a converged tube or infinite-horizon invariance. Record actual backend,
settings and limitations. The coarse jumper-avoid demo has tight clearance.

## Records and export

Update `CHANGELOG.md` for a new modeling decision, numerical workaround or
semantic correction, including reason, code location and validation. In the
private research checkout, also update its implementation log when applicable.
Retain failed diagnostics and original evidence rather than rewriting history.

Keep generated fields, trajectories, logs and source snapshots in `outputs/`.
Public exports include lightweight source, tests and documentation only:

```bash
.venv/bin/python tools/curate.py /tmp/hybrid-reach-fresh-export
```

The destination must be new and outside this tree. Verify the exported copy can
import and run independently, and that `AGENTS.md` and the file-hash manifest are
present. Exporting creates files locally; it does not publish a repository.
Preserve this boundary when changing the allowlist: private archives, PDFs,
MATLAB files, environments and Git history do not belong in the public export.
