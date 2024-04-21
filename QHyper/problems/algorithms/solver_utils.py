from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from QHyper.problems.algorithms.workflow_decomposition import Division
from QHyper.problems.workflow_scheduling import WorkflowSchedulingOneHot, WorkflowSchedulingBinary, Workflow
from QHyper.solvers import Solver, SolverResult


@dataclass(frozen=True)
class WorkflowSchedule:
    cost: float
    time: float
    deadline: float
    machine_assignment: dict[str, str]
    workflow: Workflow
    parts: list[WorkflowSchedule] = field(default_factory=list)


class WorkflowSchedulingSolverDecorator:
    def __init__(self, solver: Solver):
        self.solver: Solver = solver
        self.problem: WorkflowSchedulingOneHot | WorkflowSchedulingBinary = solver.problem

    def solve(self) -> WorkflowSchedule:
        solver_result: SolverResult = self.solver.solve(params_inits={"name": "wsp"})
        machine_assignment = self.problem.decode_solution(solver_result)

        return WorkflowSchedule(
            cost=self.problem.calculate_solution_cost(machine_assignment),
            time=self.problem.calculate_solution_timespan(machine_assignment),
            deadline=self.problem.workflow.deadline,
            machine_assignment=machine_assignment,
            workflow=self.problem.workflow
        )


class DecomposedWorkflowSchedulingSolver:
    def __init__(self, solvers: list[WorkflowSchedulingSolverDecorator], division: Division):
        self.solvers: list[WorkflowSchedulingSolverDecorator] = solvers
        self.division: Division = division

    def pick_faster_machine(self, task: str, machine1: str, machine2: str) -> str:
        machine1_time = self.division.complete_workflow.time_matrix.loc[task, machine1]
        machine2_time = self.division.complete_workflow.time_matrix.loc[task, machine2]
        return machine1 if machine1_time <= machine2_time else machine2

    def merge_machine_assignments(self, machine_assignments: Iterable[dict[str, str]]):
        final_assignment: dict[str, str] = {}
        for assignment in machine_assignments:
            for task, machine in assignment.items():
                if not task in final_assignment:
                    final_assignment[task] = machine
                else:
                    final_assignment[task] = self.pick_faster_machine(task, final_assignment[task], machine)
        return final_assignment

    def solve(self) -> WorkflowSchedule:
        partial_schedules = [s.solve() for s in self.solvers]
        machine_assignments = map(lambda s: s.machine_assignment, partial_schedules)
        merged_machine_assignment = self.merge_machine_assignments(machine_assignments)
        merged_workflow = self.division.complete_workflow
        problem: WorkflowSchedulingOneHot = WorkflowSchedulingOneHot(merged_workflow)

        return WorkflowSchedule(
            cost=problem.calculate_solution_cost(merged_machine_assignment),
            time=problem.calculate_solution_timespan(merged_machine_assignment),
            deadline=merged_workflow.deadline,
            machine_assignment=merged_machine_assignment,
            workflow=merged_workflow,
            parts=partial_schedules
        )
