from QHyper.problems.algorithms.workflow_decomposition import SeriesParallelSplit
from QHyper.problems.workflow_scheduling import Workflow


def test_non_sp_dag():
    # given
    tasks_file = "resources/workflows/complex_workflow.json"
    machines_file = "resources/machines/3_machines.json"
    deadline = 50
    workflow = Workflow(tasks_file, machines_file, deadline)

    # when
    division = SeriesParallelSplit().decompose(workflow, 6)

    # then
    assert (len(division.workflows) == 3)


def test_non_sp_dag_without_source_and_sink():
    # given
    tasks_file = "resources/workflows/complex_workflow_no_entry_and_exit.json"
    machines_file = "resources/machines/3_machines.json"
    deadline = 50
    workflow = Workflow(tasks_file, machines_file, deadline)

    # when
    division = SeriesParallelSplit().decompose(workflow, 6)

    # then
    assert (len(division.workflows) == 3)
    assert (len(division.complete_workflow.wf_instance.roots()) == 1)
    assert (len(division.complete_workflow.wf_instance.leaves()) == 1)
