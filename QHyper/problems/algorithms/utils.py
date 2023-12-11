import pathlib
import tempfile
from copy import copy

from wfcommons.common.workflow import Workflow as WfWorkflow

from QHyper.problems.workflow_scheduling import Workflow


def wfworkflow_to_qhyper_workflow(workflow: WfWorkflow, machines_file: str, deadline: float) -> Workflow:
    with tempfile.NamedTemporaryFile() as temp:
        workflow.write_json(pathlib.Path(temp.name))
        return Workflow(pathlib.Path(temp.name), machines_file, deadline)


def merge_subworkflows(subworkflows: list) -> WfWorkflow:
    workflow = WfWorkflow(name="merged")
    for idx, subworkflow in enumerate(subworkflows):
        for task in subworkflow.tasks.values():
            task_copy = copy(task)
            if "_" in task_copy.name:
                task_copy.category = "connection"
            else:
                task_copy.category = f"{idx}"
            workflow.add_task(task_copy)
    for subworkflow in subworkflows:
        for task in subworkflow.tasks.keys():
            for parent in subworkflow.tasks_parents[task]:
                workflow.add_dependency(parent, task)
    return workflow
