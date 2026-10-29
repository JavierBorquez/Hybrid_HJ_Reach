# Development guide

[Home](../README.md) · [Setup](SETUP.md) · [API reference](API.md)

## Run the tests

After [installation](SETUP.md), run from the repository root:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m pytest -q
```

The suite includes analytic translation checks, hybrid reset/failure priority,
policy decisions, interpolation, streamed-history equivalence, paper-model
smoke solves and export checks. [VALIDATION.md](VALIDATION.md) records the
previous CPU/GPU measurements and their limits.

## Find the code

| Location | Role |
|---|---|
| `src/hybrid_reach/model.py`, `problems.py` | Mode/transition declarations and objective configuration |
| `src/hybrid_reach/solver.py`, `transitions.py` | Continuous HJ propagation and hybrid value projection |
| `src/hybrid_reach/policies.py`, `filtering.py` | Value queries, goal feedback and avoid HLRF |
| `src/hybrid_reach/simulation.py`, `coordinates.py` | Event simulation and physical/computational coordinates |
| `src/hybrid_reach/examples/` | Example-specific dynamics, geometry, guards, resets and simulation adapters |
| `tutorials/` | Small runnable tutorial and walkthroughs |
| `examples/papers/` | Paper experiment configurations, drivers, plotting and independent replay |
| `tests/` | Small standalone numerical checks |

Keep model-specific choices in model/example modules. Read [AGENTS.md](../AGENTS.md)
for development invariants and [CHANGELOG.md](../CHANGELOG.md) for prior decisions.
Record new numerical workarounds or semantic changes with their reason, location
and validation evidence.

## Create a source snapshot

Development may take place in a private research checkout. The export tool copies
an allowlist of source, tests and documentation into a new standalone directory:

```bash
python tools/curate.py /tmp/hybrid-reach-fresh-export
```

Choose a destination that does not exist and is outside this source tree.
`EXPORT_MANIFEST.json` records SHA-256 hashes. Generated `outputs/`, MATLAB,
PDFs, environments and Git history are excluded. The command creates local
files; it does not publish a repository. Test the exported copy with its own
source path/environment before sharing it.

A public release still needs an explicit license and upstream attribution
review. The larger reference experiments and the remaining RSS rollout/figure
workflows also need further validation or implementation; see the
[paper status table](../examples/papers/README.md#what-is-reproduced).
