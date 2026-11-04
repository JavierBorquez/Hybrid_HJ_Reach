"""Conditional between-sample clearance checks using dense physical ODE output."""
import numpy as np


def clearance_bound(solution, start, stop, clearance, speed_bound, *, tolerance=.01):
    """For a position-1-Lipschitz clearance, samples miss at most v_max*dt/4.

    The bound is conditional on the ODE solution and supplied speed bound; this
    is not interval verification of the ODE. Subdivide near a failure boundary.
    """
    def check(a, b, depth):
        points = solution.sol(np.array([a, (a+b)/2, b])).T
        sampled = np.asarray([clearance(x) for x in points])
        lower = float(sampled.min()-speed_bound*(b-a)/4)
        if lower > 0 or sampled.min() <= 0 or speed_bound*(b-a)/4 <= tolerance or depth >= 20:
            return lower
        middle = (a+b)/2
        return min(check(a, middle, depth+1), check(middle, b, depth+1))
    return check(start, stop, 0)
