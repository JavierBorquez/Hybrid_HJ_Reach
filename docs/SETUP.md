# Setup and troubleshooting

[Home](../README.md) · [Tutorial](../tutorials/README.md) · [Paper examples](../examples/papers/README.md)

## CPU installation

Run these commands from the repository root with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[examples,test]'
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python tutorials/falling_piece.py
```

`examples` installs Matplotlib and Pillow for figures/GIFs; `test` installs pytest. The package
uses JAX through `hj_reachability`. The recorded validation environment was
Python 3.10, JAX/JAXlib 0.6.2 and `hj_reachability` 0.7.0; the declared dependency
ranges also permit other versions, which have not all been validated.

Use `JAX_ENABLE_X64=true` for the documented numerical runs. The package leaves
JAX's process-wide device and precision settings to the caller.

## GPU execution

First install a JAX build appropriate for your GPU and driver in this environment.
The package installation alone does not configure CUDA. With GPU-enabled JAX
already installed, check that JAX can see the GPU:

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  python -c "import jax; print(jax.devices())"
```

Then run the same tutorial on CUDA:

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python tutorials/falling_piece.py --output outputs/falling_piece_gpu
```

Using `JAX_PLATFORMS=cuda` makes a missing CUDA backend an error. For a deliberate
CPU run, use `JAX_PLATFORMS=cpu`. Paper demo grids benefit from a GPU; begin with
the [small value-function solve](../examples/papers/README.md#1-compute-the-value-function).

## Common problems

| Symptom | What to check |
|---|---|
| `ModuleNotFoundError: hybrid_reach` | Activate the environment and run `python -m pip install -e '.[examples,test]'` from the repository root. |
| An unexpected API or import | Run `python -c "import hybrid_reach; print(hybrid_reach.__file__)"`. The path should be this repository's `src/hybrid_reach/`. |
| `FileExistsError` on rerun | Outputs are protected from overwriting. Pass a fresh `--output outputs/my_new_run`. |
| CUDA initialization fails | Check the JAX build and driver. Use the explicit CPU command if you intend to run on CPU. |
| `MemoryError` mentioning an XLA estimate | Start with `--preset smoke` or `demo`. The paper scripts' `--memory-gib` option changes an execution-estimate budget; it does not increase available device memory. |
| A short reach-avoid paper run ends at `deadline` | The `smoke` preset has a short horizon. Use `demo` for the documented goal-reaching experiment. |
| A result cannot be loaded after an interrupted solve | A completed run needs `metadata.json` in its value directory. Partial streamed runs retain `progress.json`; automatic restart is not implemented. |
| A validator cannot find outputs | Match the solve's case, `--preset` and `--output`. Read the [paper guide](../examples/papers/README.md#3-use-the-values-as-feedback-or-a-safety-filter). |

## Working inside the enclosing research checkout

This section applies only if this directory is still named `public_export`
inside the private research project. The parent contains a different package
with the same import name. Keep a separate environment when installing this
tree so that its editable installation does not replace the parent's.

To reuse the parent's existing environment without installing over its package,
run from `public_export` and explicitly select this source tree:

```bash
PYTHONPATH="$PWD/src" ../.venv/bin/python -c "import hybrid_reach; print(hybrid_reach.__file__)"
PYTHONPATH="$PWD/src" JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  ../.venv/bin/python tutorials/falling_piece.py --output outputs/falling_piece_local
```

Retain the same `PYTHONPATH` prefix for subsequent commands using that interpreter.
The standalone repository does not require the parent environment or its data.
