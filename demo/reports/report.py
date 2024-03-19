import json
import datetime

from QHyper.problems.workflow_scheduling import Workflow


class WorkflowSolution:
    def __init__(self, id: int, machine_assignment: dict, cost: float, time: float):
        self.id = id
        self.machine_assignment = machine_assignment
        self.cost = cost
        self.time = time


class Solution:
    def __init__(self, solver: str, workflows: list, cost: float, time: float):
        self.solver = solver
        self.workflows = workflows
        self.cost = cost
        self.time = time


class Subworkflow:
    def __init__(self, id: int, deadline: float, workflow: Workflow):
        self.id = id
        self.deadline = deadline
        self.workflow = workflow.wf_instance.instance


class Division:
    def __init__(self, method: str, workflows: list):
        self.method = method
        self.workflows = workflows


class ExecutionReport:
    def __init__(self, workflow: str, machines: str, deadline: float, division: Division, solution: Solution):
        self.workflow = workflow
        self.machines = machines
        self.deadline = deadline
        self.division = division
        self.solution = solution
        self.timestamp = datetime.datetime.now().isoformat()

    def write_json(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(obj=self.__dict__, fp=file, default=lambda o: o.__dict__, indent=4)
