from dwave.system import LeapHybridDQMSampler

from typing import Any, Optional

from QHyper.solvers.converter import Converter
from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver
from QHyper.optimizers.base import Optimizer



class DQM(Solver):
    def __init__(self, problem: Problem, time: float) -> None:
        self.problem: Problem = problem
        self.time: float = time

    def solve(
            self,
            params_inits: dict[str, Any],
            hyper_optimizer: Optional[Optimizer] = None
    ) -> Any:
        converter = Converter()
        dqm = converter.to_dqm(self.problem)
        sampler = LeapHybridDQMSampler()
        
        sampleset = sampler.sample_dqm(dqm, self.time)

        return sampleset