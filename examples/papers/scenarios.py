"""Explicit paper-model experiment settings for end-to-end public runs."""
from dataclasses import dataclass, asdict
import numpy as np
from hybrid_reach import Problem
from hybrid_reach.examples import jump_filter, jump_reach_avoid, satellite, satellite_reach_avoid
from hybrid_reach.examples.adapters import jumper_adapter, satellite_adapter


@dataclass
class Scenario:
    name: str
    adapter: object
    initial_states: dict
    horizon: float
    coupling_dt: float
    save_dt: float
    feedback_dt: float
    margin: float
    finish_x: object = None

    @property
    def problem(self): return self.adapter.problem

    def nominal(self, state, mode):
        return np.array([10., 20.]) if self.adapter.system == 'jumper' and mode == 0 else np.zeros(2)

    def configuration(self):
        return dict(name=self.name, shape=list(self.problem.grid.shape), horizon=self.horizon,
            coupling_dt=self.coupling_dt, save_dt=self.save_dt, feedback_dt=self.feedback_dt,
            margin=self.margin, initial_states=self.initial_states,
            parameters=asdict(self.adapter.parameters), coordinates=asdict(self.adapter.coordinates),
            finish_x=self.finish_x)


def make_scenario(name, preset='demo'):
    if preset not in ('smoke', 'demo', 'reference'):
        raise ValueError('Unknown preset')
    smoke, fine = preset == 'smoke', preset == 'reference'
    if name.startswith('jump'):
        shape = (17, 13, 7, 9) if smoke else ((131, 51, 31, 41) if fine else (71, 36, 21, 31))
        if name == 'jump_avoid':
            p = jump_filter.FilterJumpParameters()
            model, grid, f = jump_filter.make_problem(shape, parameters=p)
            problem = Problem.avoid(model, grid, f)
            return Scenario(name, jumper_adapter(problem, p), {'standard': [.5, .25, 0., 0.]},
                .2 if smoke else (4. if fine else 3.), .005, .05, .005, .02, finish_x=5.3)
        if name == 'jump_reach_avoid':
            shape = (17, 13, 7, 9) if smoke else ((131, 71, 27, 41) if fine else (81, 51, 23, 31))
            p = jump_reach_avoid.ReachAvoidJumpParameters()
            model, grid, g, l = jump_reach_avoid.make_problem(shape, parameters=p)
            problem = Problem.reach_avoid(model, grid, g, -l)
            return Scenario(name, jumper_adapter(problem, p), {'standard': [.1, .35, 3., 0.]},
                .2 if smoke else 2., .0025, .05, .005, 0.)
    shape = (9, 9, 5, 5, 5) if smoke else ((51, 36, 17, 17, 11) if fine else (41, 31, 15, 15, 11))
    starts = {'low_velocity': [-3600., 4000., 6., -2., 0.],
              'high_velocity': [-1000., 8000., 0., -18., 0.]}
    if name == 'satellite_avoid':
        p = satellite.SatelliteParameters()
        scales = (6000., 6000., 12., 12.)
        model, grid, f = satellite.make_problem(shape, parameters=p,
            domain_lower=(-60000., -1000., -60., -60., 0.),
            domain_upper=(20000., 50000., 60., 60., 1.), coordinate_scales=scales)
        problem = Problem.avoid(model, grid, f)
        return Scenario(name, satellite_adapter(problem, p, scales), starts,
            4. if smoke else 1200., 2., 2. if smoke else 20., 1., 200.)
    if name == 'satellite_reach_avoid':
        shape = (9, 9, 5, 5, 5) if smoke else ((51, 41, 15, 15, 11) if fine else shape)
        p = satellite_reach_avoid.ReachAvoidSatelliteParameters()
        scales = (2500., 2000., 10., 10.)
        model, grid, g, l = satellite_reach_avoid.make_problem(shape, parameters=p, coordinate_scales=scales)
        problem = Problem.reach_avoid(model, grid, g, -l)
        return Scenario(name, satellite_adapter(problem, p, scales), starts,
            4. if smoke else 780., 2., 2. if smoke else 20., 1., 0.)
    raise ValueError(name)


CASES = ('jump_avoid', 'jump_reach_avoid', 'satellite_avoid', 'satellite_reach_avoid')
