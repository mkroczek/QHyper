import random

from QHyper.problems.workflow_scheduling import WorkflowSchedulingOneHot, Workflow


def test_solution_cost():
    # given
    workflow: Workflow = load_workflow()
    wsp_one_hot: WorkflowSchedulingOneHot = WorkflowSchedulingOneHot(workflow)
    machine_assignment = assign_random_machines(list(workflow.machines.keys()), workflow.task_names)
    expected_cost = sum([workflow.cost_matrix.loc[t, m] for t, m in machine_assignment.items()])

    # when
    cost: float = wsp_one_hot.calculate_solution_cost(machine_assignment)

    # then
    assert (cost == expected_cost)


def test_solution_timespan():
    # given
    workflow: Workflow = load_workflow()
    wsp_one_hot: WorkflowSchedulingOneHot = WorkflowSchedulingOneHot(workflow)
    machine_assignment = {task: 'MachineA' for task in workflow.task_names}
    longest_path = ['Task1', 'Task2', 'Task9', 'Task10']
    expected_timespan = sum([workflow.time_matrix.loc[t, machine_assignment[t]] for t in longest_path])

    # when
    timespan: float = wsp_one_hot.calculate_solution_timespan(machine_assignment)

    # then
    assert (timespan == expected_timespan)


def test_solution_timespan_no_entry_and_exit():
    # given
    tasks_file = "resources/workflows/complex_workflow_no_entry_and_exit.json"
    machines_file = "resources/machines/3_machines.json"
    deadline = 50
    workflow: Workflow = Workflow(tasks_file, machines_file, deadline)
    wsp_one_hot: WorkflowSchedulingOneHot = WorkflowSchedulingOneHot(workflow)
    machine_assignment = {task: 'MachineA' for task in workflow.task_names}
    longest_path = ['Task5', 'Task9']
    expected_timespan = sum([workflow.time_matrix.loc[t, machine_assignment[t]] for t in longest_path])

    # when
    timespan: float = wsp_one_hot.calculate_solution_timespan(machine_assignment)

    # then
    assert (timespan == expected_timespan)


def test_critical_path_value():
    # given
    tasks_file = "resources/workflows/complex_workflow_no_entry_and_exit.json"
    machines_file = "resources/machines/3_machines.json"
    deadline = 50
    workflow: Workflow = Workflow(tasks_file, machines_file, deadline)

    # when
    cpv = workflow.critical_path_value

    # then
    assert (cpv == 22)


def assign_random_machines(machines: list[str], tasks: list[str]):
    return {task: random.choice(machines) for task in tasks}


def load_workflow():
    tasks_file = "resources/workflows/complex_workflow.json"
    machines_file = "resources/machines/3_machines.json"
    deadline = 50
    return Workflow(tasks_file, machines_file, deadline)
