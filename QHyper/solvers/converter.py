from typing import Any, cast

import dimod
import sympy
from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel
from QHyper.hyperparameter_gen.parser import Expression
from QHyper.problems.base import Problem
from QHyper.util import QUBO, VARIABLES


def dict_to_list(my_dict: QUBO) -> list[tuple[Any, ...]]:
    return [tuple([*key, val]) for key, val in my_dict.items()]


class Converter:
    @staticmethod
    def create_qubo(problem: Problem, weights: list[float]) -> QUBO:
        results: dict[VARIABLES, float] = {}

        if len(weights) != len(problem.constraints) + 1:
            raise Exception(
                f"Expected {len(problem.constraints)+1} weights, "
                f"got {len(weights)}"
            )

        objective_function = Expression(
            problem.objective_function.polynomial * weights[0]
        )
        for key, value in objective_function.as_dict().items():
            if key in results:
                results[key] += value
            else:
                results[key] = value

        constraint_weights = weights[1:]

        for weight, element in zip(constraint_weights, problem.constraints):
            constraint = Expression(element.polynomial**2 * weight)
            for key, value in constraint.as_dict().items():
                if key in results:
                    results[key] += value
                else:
                    results[key] = value

        return results

    @staticmethod
    def to_cqm(problem: Problem) -> ConstrainedQuadraticModel:
        binary_polynomial = dimod.BinaryPolynomial(
            problem.objective_function.as_dict(), dimod.BINARY
        )
        cqm = dimod.make_quadratic_cqm(binary_polynomial)

        # todo this cqm can probably be initialized in some other way
        for var in problem.variables:
            if str(var) not in cqm.variables:
                cqm.add_variable(dimod.BINARY, str(var))

        for i, constraint in enumerate(problem.constraints):
            cqm.add_constraint(
                dict_to_list(constraint.as_dict()), "==", label=i
            )

        return cqm

    @staticmethod
    def to_qubo(problem: Problem) -> tuple[QUBO, float]:
        cqm = Converter.to_cqm(problem)
        bqm, _ = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10)
        return cast(tuple[QUBO, float], bqm.to_qubo())  # (qubo, offset)

    @staticmethod
    def to_dqm(problem: Problem) -> DiscreteQuadraticModel:
        dqm = dimod.DiscreteQuadraticModel()

        def binary_to_discrete(v: sympy.Symbol) -> sympy.Symbol:
            id = int(str(v)[len("s") :])
            discrete_id = int(id // problem.cases)
            return sympy.Symbol(f"x{discrete_id}")

        variables_discrete = [
            str(binary_to_discrete(v))
            for v in problem.variables[:: problem.cases]
        ]
        for var in variables_discrete:
            if var not in dqm.variables:
                dqm.add_variable(problem.cases, var)

        for vars, bias in problem.objective_function.as_dict().items():
            s_i, *s_j = vars
            x_i = binary_to_discrete(s_i)
            xi_idx: int = cast(int, dqm.variables.index(str(x_i)))
            if s_j:
                x_j = binary_to_discrete(*s_j)
                xj_idx: int = cast(int, dqm.variables.index(str(x_j)))
                dqm.set_quadratic(
                    dqm.variables[xi_idx],
                    dqm.variables[xj_idx],
                    {(case, case): bias for case in range(problem.cases)},
                )
            else:
                dqm.set_linear(
                    dqm.variables[xi_idx], [bias for _ in range(problem.cases)]
                )

        return dqm
