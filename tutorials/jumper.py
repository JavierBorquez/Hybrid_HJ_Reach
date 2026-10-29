"""Run from the export root: python tutorials/jumper.py --kind all."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hybrid_reach import Problem, solve, Result, AvoidFilter, ContinuousFeedback, simulate
from hybrid_reach.examples.vertical_jumper import make_system, failure, goal


def run(kind, output, points=81, horizon=.8):
    model, grid, events = make_system((points, points))
    problem = {"reach": lambda: Problem.reach(model, grid, goal),
               "avoid": lambda: Problem.avoid(model, grid, failure),
               "reach_avoid": lambda: Problem.reach_avoid(model, grid, goal, failure)}[kind]()
    steps = int(np.ceil(horizon/.01))
    horizons = np.linspace(0., horizon, steps+1)
    saved = tuple(sorted(set([0, *range(5, steps, 5), steps])))
    result = solve(problem, horizons, save_indices=saved, memory_budget_bytes=1024**3)
    folder = output/kind
    result.save(folder)
    result = Result.load(folder)  # The online policy uses memory-mapped values.
    if kind == "avoid":
        hlrf = AvoidFilter(problem, result, margin=.03)
        controller = lambda x, q, tau: hlrf(x, q, tau, lambda state, mode: np.zeros(1))
    else:
        controller = ContinuousFeedback(problem, result)
    trajectory = simulate(problem, controller, [.15, 0.], 0, duration=horizon,
        feedback_dt=.005, events=events, failure=lambda x, q: float(failure(x, q)),
        goal=None if kind == "avoid" else lambda x, q: float(goal(x, q)))
    np.savez(folder/"trajectory.npz", **{k: v for k, v in trajectory.items()
                                       if isinstance(v, np.ndarray)})
    summary = dict(result.metadata, status=trajectory["status"],
        solver_status=result.metadata.get("status"), final_time=trajectory["final_time"],
        final_state=trajectory["final_state"].tolist(), resets=trajectory["resets"],
        continuous_interventions=int(trajectory["continuous_intervention"].sum()))
    (folder/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for q, ax in enumerate(axes[:2]):
        values = result.values[-1, q]
        image = ax.pcolormesh(*result.coordinates, values.T, shading="auto", cmap="coolwarm")
        if values.min() < 0 < values.max():
            ax.contour(*result.coordinates, values.T, levels=[0.], colors="black")
        ax.set(title=model.modes[q].name, xlabel="Height [m]", ylabel="Velocity [m/s]")
        fig.colorbar(image, ax=ax, label="Value")
    axes[2].plot(trajectory["time"], trajectory["state"][:, 0])
    axes[2].axhspan(1.0, 1.2, color="green", alpha=.15)
    axes[2].set(title=trajectory["status"], xlabel="Time [s]", ylabel="Height [m]")
    fig.suptitle(f"{kind}: numerical smoke test, T={horizon:g} s")
    fig.tight_layout()
    fig.savefig(folder/"values_and_rollout.png", dpi=140)
    plt.close(fig)
    print(f"{kind}: {trajectory['status']} at {trajectory['final_time']:.3f} s; "
          f"{result.metadata['backend']}, compile={result.metadata['compile_seconds']:.2f} s, "
          f"solve={result.metadata['execution_seconds']:.3f} s", flush=True)
    print(f"  Figure: {folder/'values_and_rollout.png'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["all", "reach", "avoid", "reach_avoid"], default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/jumper"))
    parser.add_argument("--points", type=int, default=81)
    parser.add_argument("--horizon", type=float, default=.8)
    args = parser.parse_args()
    for kind in ("reach", "avoid", "reach_avoid") if args.kind == "all" else (args.kind,):
        run(kind, args.output, args.points, args.horizon)
