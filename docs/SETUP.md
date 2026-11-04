# Setup and troubleshooting

[Home](../README.md) · [Tutorial](../tutorials/README.md) · [Paper examples](../examples/papers/README.md)

## CPU setup

From the cloned repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python tutorials/falling_piece.py
```

The dependencies include `hj_reachability`, JAX, NumPy, SciPy, Matplotlib and
Pillow. The pinned versions were tested with Python 3.10 and JAX/JAXlib 0.6.2.
The scripts load the `hybrid_reach/` code directly from this clone.

Use `JAX_ENABLE_X64=true` for the documented numerical runs. Results go into
`outputs/`; choose a fresh `--output` directory for reruns.

## GPU setup

In the same environment, install a CUDA-enabled JAX build matching version
0.6.2 and your GPU driver. Check that JAX can see the GPU:

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  python -c "import jax; print(jax.devices())"
```

Then run the tutorial on CUDA:

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python tutorials/falling_piece.py --output outputs/falling_piece_gpu
```

`JAX_PLATFORMS=cuda` requires a working CUDA backend. Paper grids benefit from
a GPU; begin with the [small value-function solve](../examples/papers/README.md#1-compute-the-value-function).

## Common problems

| Symptom | What to check |
|---|---|
| A missing dependency | Activate `.venv` and run `python -m pip install -r requirements.txt`. |
| `ModuleNotFoundError: hybrid_reach` in a Python snippet | Start Python from the repository root. The command-line scripts find the clone automatically. |
| `FileExistsError` on rerun | Pass a fresh `--output outputs/my_new_run`. |
| CUDA initialization fails | Check your JAX build and driver. Use `JAX_PLATFORMS=cpu` for a CPU run. |
| `MemoryError` mentioning an XLA estimate | Start with `--preset smoke`. `--memory-gib` changes the execution-estimate budget, not the GPU capacity. |
| A short reach-avoid run ends at `deadline` | The `smoke` horizon can be too short to reach the goal. Use `demo` for the complete IJRR workflow. |
| A result cannot be loaded after an interrupted solve | A completed run needs `metadata.json` in its value directory. Partial streamed solves cannot be restarted automatically. |
| A validator cannot find outputs | Use the same case, `--preset` and `--output` as the solve. See the [paper guide](../examples/papers/README.md#3-use-the-values-as-feedback-or-a-safety-filter). |
