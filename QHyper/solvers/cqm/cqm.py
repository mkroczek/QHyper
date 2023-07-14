import os
from dimod import ConstrainedQuadraticModel
from dwave.system import LeapHybridCQMSampler

from typing import Any, Optional

from QHyper.solvers.converter import Converter
from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver
from QHyper.optimizers.base import Optimizer



token = os.environ['DWAVE_API_TOKEN']


class CQM(Solver):
    """
    Class for solving a problem using the Constrained Quadratic Model (CQM) approach.

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    time : float
        The maximum time allowed for the CQM solver.
    """

    def __init__(self, problem: Problem, time: float) -> None:
        """
        Parameters
        ----------
        problem : Problem
            The problem to be solved.
        time : float
            The maximum time allowed for the CQM solver.
        """
        self.problem: Problem = problem
        self.time: float = time

    def solve(self, params_inits: dict[str, Any] = {}) -> Any:
        """
        Solve the problem using the CQM approach.

        Parameters
        ----------
        params_inits : dict[str, Any], optional
            Initial parameters for the optimization. Default is None.
        hyper_optimizer : Optimizer, optional
            Hyperparameter optimizer. Default is None.

        Returns
        -------
        Any
            The solution to the problem.
        """

        converter = Converter()
        cqm = converter.to_cqm(self.problem)
        sampler = LeapHybridCQMSampler(token=token)
        solutions = sampler.sample_cqm(cqm, self.time)
        correct_solutions = [
            s for s in solutions
            if len(cqm.violations(s, skip_satisfied=True)) == 0
        ]

        print(f"cqm constraints\n {cqm.constraints}")
        print(f"\n\nProblem constraints\n {self.problem.constraints}")
        print(cqm.constraints[0])
        print(cqm.constraints[0].to_polystring())
        print(cqm.constraints[1].to_polystring())

        return correct_solutions[0]
