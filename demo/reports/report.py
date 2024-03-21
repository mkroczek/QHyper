import datetime
import json
from dataclasses import dataclass, field

from wfcommons.common import Workflow

from QHyper.problems.algorithms.utils import WorkflowSchedule


@dataclass(frozen=True)
class WorkflowSolution:
    cost: float
    time: float
    deadline: float
    machine_assignment: dict[str, str]
    workflow: Workflow


@dataclass
class Solution:
    solver: str
    total_cost: float = 0.0
    total_time: float = 0.0
    parts: list[WorkflowSolution] = field(default_factory=list)

    def add_part(self, workflow_schedule: WorkflowSchedule):
        self.total_cost += workflow_schedule.cost
        self.total_time += workflow_schedule.time
        self.parts.append(WorkflowSolution(
            cost=workflow_schedule.cost,
            time=workflow_schedule.time,
            deadline=workflow_schedule.workflow.deadline,
            machine_assignment=workflow_schedule.machine_assignment,
            workflow=workflow_schedule.workflow.wf_instance.instance
        ))


@dataclass
class ExecutionReport:
    workflow_file: str
    machines_file: str
    deadline: int
    solution: Solution
    timestamp: str = field(default=datetime.datetime.now().isoformat())

    def write_json(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(obj=self.__dict__, fp=file, default=lambda o: o.__dict__, indent=4)
