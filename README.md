# Hybrid reachability with JAX

Development staging area for a curated public source export. This directory is
self-contained: it does not read the private research repository, MATLAB,
paper PDFs, archived arrays or comparison fixtures. Development stays private;
public snapshots are produced with `tools/curate.py`.

The package computes autonomous hybrid **reach**, **avoid** and **reach-avoid**
value functions using `hj_reachability`. It also supplies an **avoid-only hybrid
least-restrictive filter (HLRF)**. Reach/reach-avoid examples use goal-seeking
feedback; no reach/reach-avoid LRF is advertised.

## Start here

Use a separate environment when installing this package: it deliberately has
the same import name as the private research implementation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[examples,test]'
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python tutorials/jumper.py
python -m pytest -q
```

The small vertical jumper works on CPU. An already GPU-enabled JAX installation
can run the identical API with:

```bash
JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python tutorials/jumper.py --output outputs/jumper_gpu
```

The package does not change process-wide JAX precision/device settings. The
validated development environment uses Python 3.10, JAX/JAXlib 0.6.2 and
hj_reachability 0.7.0. GPU setup depends on the local JAX installation.

- [Step-by-step jumper tutorial](tutorials/README.md)
- [Public API, mathematical conventions and current limits](docs/API.md)
- [Paper example recipes](examples/papers/README.md)
- [Planar-jumper tutorial continuation](tutorials/PLANAR_JUMPER.md)
- [Intentional changes from the research implementation](CHANGELOG.md)
- [Validation performed on this preparation](docs/VALIDATION.md)
- [Agent development guide](AGENTS.md)

## One pipeline

```python
from hybrid_reach import Problem, solve, AvoidFilter
from hybrid_reach.examples.vertical_jumper import make_system, failure
import numpy as np

model, grid, events = make_system()
problem = Problem.avoid(model, grid, failure)
result = solve(problem, np.linspace(0., .8, 81), save_indices=(0, 40, 80))
result.save('outputs/my_jumper')
hlrf = AvoidFilter(problem, result, margin=.03)
decision = hlrf([.3, 0.], 0, .8, nominal_control=lambda x, q: np.zeros(1))
```

Define dynamics, guards, resets and sets in your model module. The shared solver
contains no model-specific value mapping. All input set functions are negative
inside. Avoid results are positive-safe; reach/reach-avoid results are
negative-winning. HLRF margins are numerical intervention buffers.

## Public exports and private evidence

```bash
python tools/curate.py /path/to/a/new/public-snapshot
```

This copies an allowlist of source, tests and documentation and writes file
hashes. It excludes generated output, large arrays, PDFs, MATLAB, environment
directories and Git history. No repository is published by this command.
Choose an explicit license and complete the remaining model/policy curation
before a public release; neither is implied by this development snapshot.

The original fine-grid results, failures, refinement studies and detailed logs
stay in the private repository. Small numerical tests ship with the public code.
Successful smoke runs are not convergence or safety certificates.

The IJRR planar jumper and satellite also have complete solve/reload/rollout/plot
commands, using the same core pipeline:

```bash
JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python examples/papers/rollouts.py jump_reach_avoid
python examples/papers/validate.py jump_reach_avoid
```

Use `jump_avoid`, `satellite_avoid` or `satellite_reach_avoid` for the other
cases. These default demo grids are substantially larger than the vertical
tutorial and benefit from a GPU. `--preset smoke` selects short small-grid runs.
Large histories are streamed to disk with `solve_streamed`, while online policy
queries and physical simulation use CPU. See the paper README for memory,
finite-horizon and validation details.
