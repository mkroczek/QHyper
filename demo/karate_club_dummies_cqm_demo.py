import os
from matplotlib import pyplot as plt
from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
    ObjFunFormula as off,
)
from typing import Any
from QHyper.problems.network_communities.utils import COLORS
import networkx as nx
from QHyper.util import QUBO
from QHyper.solvers.cqm import CQM

name = "karate"


def safe_open(path: str, permission: str) -> Any:
    """
    Open "path" for writing, creating any parent directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, permission)


def write_to_file(solution: Any, solution_file: str) -> None:
    with safe_open(solution_file, "w") as file:
        file.write(str(solution))


def calc(vars: dict[str, Any], poly_dict: QUBO) -> Any:
    cost_function: float = 0
    for key, value in poly_dict.items():
        tmp = 1
        for k in key:
            tmp *= vars[k]
        cost_function += tmp * value
    return cost_function


karate_problem = CommunityDetectionProblem(
    KarateClubNetwork, N_communities=4, obj_func_formula=off.DICT
)
problem = karate_problem

karate_cqm = CQM(problem, time=5)
karate_cqm_solution = karate_cqm.solve()
solution = karate_cqm_solution

print("-----Encoded solution-----")
keyorder = [v for k, dummies in problem.dummy_coefficients.items() for v in dummies]
d = solution
solution = {str(k): d[str(k)] for k in keyorder if str(k) in d}

solution_file = f"demo/demo_output/{name}_cqm_dummies_solution.csv"
write_to_file(solution, solution_file)

# DECODING
decoded_solution = problem.decode_dummies_solution(solution)

color_map = []
for node in problem.G:
    color_map.append(COLORS[decoded_solution[node]])

print("-------Decoded solution------")
keyorder = [int(str(v)[len('x'):]) for v in problem.variables]
d = decoded_solution
decoded_solution = {k: d[k] for k in keyorder if k in d}

decoded_solution_file = f"demo/demo_output/{name}_cqm_dummies_decoded_solution.csv"
write_to_file(decoded_solution, decoded_solution_file)

folder = "demo/demo_output"
data_name = f"{name}_cqm_dummies"
path = f"{folder}/{data_name}.png"

f = plt.figure()
nx.draw(
    problem.G,
    node_color=color_map,
    with_labels=True,
    ax=f.add_subplot(111),
)
f.savefig(path)