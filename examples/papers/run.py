"""Compact paper-model solve recipes, independent of MATLAB and private data.

These exercise the unified solver; they do not claim the archived fine-grid
results or their original solver/reset conventions are reproduced unchanged.
"""
import argparse
from pathlib import Path
import numpy as np
from hybrid_reach import Problem, solve
from hybrid_reach.examples import dog1d, aircoll3d, jump4d, jump_filter
from hybrid_reach.examples import jump_reach_avoid, satellite, satellite_reach_avoid


def make_case(name, preset="smoke"):
    fine = preset == "reference"
    if name.startswith("dog"):
        model, grid, g = dog1d.make_problem(301 if fine else 81,
            "recoverable" if name.endswith("recoverable") else "absorbing")
        return Problem.reach(model, grid, g), 3., .01
    if name.startswith("aircraft"):
        model, grid, failure = aircoll3d.make_problem(151 if fine else 21,
            "optimal_speed" if name.endswith("optimal") else "fixed_speed")
        return Problem.avoid(model, grid, failure), 5., .01
    if name == "jump_reach":
        model, grid, g = jump4d.make_problem((51, 31, 21, 21) if fine else (15, 11, 7, 7))
        return Problem.reach(model, grid, g), 1., .01
    if name == "jump_avoid":
        model, grid, failure = jump_filter.make_problem((131, 51, 31, 41) if fine else (15, 11, 7, 7))
        return Problem.avoid(model, grid, failure), 4. if fine else .1, .005
    if name == "jump_reach_avoid":
        model, grid, g, l = jump_reach_avoid.make_problem((131, 71, 27, 41) if fine else (15, 11, 7, 7))
        return Problem.reach_avoid(model, grid, g, -l), 2. if fine else .1, .0025
    if name == "satellite_avoid":
        model, grid, failure = satellite.make_problem((51, 36, 17, 17, 11) if fine else (7, 7, 5, 5, 5),
            domain_lower=(-60000., -1000., -60., -60., 0.),
            domain_upper=(20000., 50000., 60., 60., 1.), coordinate_scales=(6000., 6000., 12., 12.))
        return Problem.avoid(model, grid, failure), 1200. if fine else 2., 2.
    if name == "satellite_reach_avoid":
        model, grid, g, l = satellite_reach_avoid.make_problem((51, 41, 15, 15, 11) if fine else (7, 7, 5, 5, 5))
        return Problem.reach_avoid(model, grid, g, -l), 780. if fine else 2., 2.
    raise ValueError(name)


CASES = ("dog_absorbing", "dog_recoverable", "aircraft_fixed", "aircraft_optimal",
         "jump_reach", "jump_avoid", "jump_reach_avoid", "satellite_avoid", "satellite_reach_avoid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--preset", choices=["smoke", "reference"], default="smoke")
    parser.add_argument("--output", type=Path, default=Path("outputs/papers"))
    parser.add_argument("--memory-gib", type=float, default=4.)
    args = parser.parse_args()
    problem, horizon, step = make_case(args.case, args.preset)
    times = np.linspace(0., horizon, int(np.ceil(horizon/step))+1)
    result = solve(problem, times, save_indices=(0, len(times)-1),
                   memory_budget_bytes=int(args.memory_gib*1024**3))
    result.metadata.update(case=args.case, preset=args.preset)
    result.save(args.output/f"{args.case}_{args.preset}")
    print(result.metadata)
