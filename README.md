# Hybrid reachability with JAX

Compute which states of a hybrid system can reach a goal, avoid failure, or do
both. A hybrid system combines continuous motion with discrete mode changes:
a jumping robot switches between stance and flight; a spacecraft switches
between idle, thrust and cooldown.

This Python package solves Hamilton–Jacobi reachability problems on CPU or GPU
using `hj_reachability` and JAX. It includes feedback controllers, an avoid-only
hybrid least-restrictive safety filter, and examples from the RSS and IJRR
hybrid reachability studies.

![Falling-piece reach-avoid policy: steer before the drop to reach the clear part of the target](docs/images/falling_piece_reach_avoid_policy.gif)

The [first tutorial](tutorials/README.md) asks you to steer a falling piece
into the right column before steering disappears. It generates BRT plots and
policy GIFs for reach, avoid and reach-avoid on CPU.

| What would you like to do? | Start here |
|---|---|
| Understand the method and the results | [How hybrid reachability works](docs/OVERVIEW.md) |
| Run a small example and learn the API | [Falling-piece tutorial](tutorials/README.md) |
| Run the paper examples | [Paper experiments and reproduction status](examples/papers/README.md) |
| Define your own system | [Tutorial model walkthrough](tutorials/README.md#2-code-the-continuous-dynamics), then [API reference](docs/API.md) |

## Install and run your first example

Use Python 3.10 or newer. From the repository root, create an environment and
install the package with its plotting and test dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[examples,test]'
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python tutorials/falling_piece.py
```

This runs the same two-dimensional falling-piece model with three objectives:
reach a target, avoid obstacles, and reach the target while avoiding obstacles.
It solves each problem, simulates several initial conditions, and writes a BRT
plot and a policy GIF. Avoid also compares an arbitrary nominal policy with its
safety-filtered version. A GPU is optional.

Open `outputs/falling_piece/reach/brt.png` and `policy.gif` first. The
[tutorial](tutorials/README.md) introduces the intuition before showing how to
code the dynamics, boundaries, optimizer, sets and feedback. Each task also
writes `summary.json`, numerical trajectories and a reloadable `value/` directory.
Choose a new `--output` directory for reruns:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python tutorials/falling_piece.py --output outputs/falling_piece_second_run
```

See [setup and troubleshooting](docs/SETUP.md) for GPU use, environment checks
and common errors.

## From a model to a controller

Define the dynamics in each mode, the guards and reset maps between modes, and
the goal/failure sets. The same pipeline handles all three objectives:

```python
import numpy as np
from hybrid_reach import Problem, solve
from hybrid_reach.examples.falling_piece import make_system, goal, failure

model, grid, events = make_system()
problem = Problem.reach_avoid(model, grid, goal, failure)
result = solve(problem, np.linspace(0., 4., 401))
```

The result contains one value field per mode and saved horizon. Feedback uses
its spatial gradient to choose controls; the simulator locates mode changes
and applies resets. Follow the [tutorial](tutorials/README.md) to save, reload
and simulate this result.

## Paper examples

The [paper guide](examples/papers/README.md) maps every example to its command,
model and available outputs. The planar-jumper and spacecraft cases have a
complete solve → simulate → plot → independently replay workflow. For example,
after installation:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  python examples/papers/rollouts.py jump_reach_avoid --preset smoke
```

This short run checks the workflow. Use the guide's `demo` recipe for the full
goal-reaching trajectory; the short `smoke` horizon need not reach the goal.
Dog1D, aircraft and the RSS platform jumper currently have value-function solve
recipes. Their complete rollout/figure workflows remain unfinished.

The public examples implement the paper models with documented numerical and
modeling choices. Exact paper-figure reproduction is still incomplete, and
the larger `reference` presets have not all been validated. See the guide's
[reproduction status](examples/papers/README.md#what-is-reproduced) before
starting a large run.

## Further reading

- [API reference](docs/API.md): signatures, signs, reset ordering and supported features.
- [Vertical jumper](tutorials/VERTICAL_JUMPER.md) and [planar jumper](tutorials/PLANAR_JUMPER.md): additional models after the first tutorial.
- [Validation results](docs/VALIDATION.md): measured runtimes, trajectory outcomes and numerical limits.
- [Development guide](docs/DEVELOPMENT.md): tests, source layout and source exports.
- [Change log](CHANGELOG.md): recorded modeling and numerical decisions.

This is a development version. A repository license still needs to be selected
before public release. The examples and tests run without MATLAB or saved
reference data.
