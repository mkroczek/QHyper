from collections import defaultdict

from wfcommons.common.workflow import Workflow as WfWorkflow

from QHyper.problems.algorithms.graph_utils import get_sp_decomposition_tree, SPTreeNode
from QHyper.problems.algorithms.workflow_decomposition import SeriesParallelSplitFinal, Division
from QHyper.problems.workflow_scheduling import Workflow


def test_should_not_add_new_nodes_due_to_max_subgraph_size():
    # given
    workflow: Workflow = load_sp_workflow()
    wf_workflow: WfWorkflow = workflow.wf_instance.workflow
    nodes_count = wf_workflow.number_of_nodes()
    weights: dict = workflow.time_matrix.mean(axis=1).to_dict()
    max_subgraph_size: int = nodes_count
    tree: SPTreeNode = get_sp_decomposition_tree(wf_workflow)
    split_algorithm: SeriesParallelSplitFinal = SeriesParallelSplitFinal()

    # when
    modified_tree: SPTreeNode = split_algorithm.modify_series_nodes(tree, {}, wf_workflow, weights, max_subgraph_size)

    # then
    assert (nodes_count == wf_workflow.number_of_nodes())
    assert (tree.get_graph_nodes() == modified_tree.get_graph_nodes())
    leaves_before = {leaf.name for leaf in tree.leaves}
    leaves_after = {leaf.name for leaf in modified_tree.leaves}
    assert (leaves_before == leaves_after)


def test_should_not_add_new_nodes_due_zero_weights():
    # given
    workflow: Workflow = load_sp_workflow()
    wf_workflow: WfWorkflow = workflow.wf_instance.workflow
    nodes_count = wf_workflow.number_of_nodes()
    weights: dict = defaultdict(lambda: 0)
    max_subgraph_size: int = 2
    tree: SPTreeNode = get_sp_decomposition_tree(wf_workflow)
    split_algorithm: SeriesParallelSplitFinal = SeriesParallelSplitFinal()

    # when
    modified_tree: SPTreeNode = split_algorithm.modify_series_nodes(tree, {}, wf_workflow, weights, max_subgraph_size)

    # then
    assert (nodes_count == wf_workflow.number_of_nodes())
    assert (tree.get_graph_nodes() == modified_tree.get_graph_nodes())
    leaves_before = {leaf.name for leaf in tree.leaves}
    leaves_after = {leaf.name for leaf in modified_tree.leaves}
    assert (leaves_before == leaves_after)


def test_should_add_two_new_nodes():
    # given
    workflow: Workflow = load_sp_workflow()
    wf_workflow: WfWorkflow = workflow.wf_instance.workflow
    nodes_count = wf_workflow.number_of_nodes()
    weights: dict = defaultdict(lambda: 1)
    max_subgraph_size: int = 3
    tree: SPTreeNode = get_sp_decomposition_tree(wf_workflow)
    split_algorithm: SeriesParallelSplitFinal = SeriesParallelSplitFinal()

    # when
    modified_tree: SPTreeNode = split_algorithm.modify_series_nodes(tree, {}, wf_workflow, weights, max_subgraph_size)

    # then
    assert (nodes_count + 2 == wf_workflow.number_of_nodes())
    new_nodes: set = set(modified_tree.get_graph_nodes()).difference(set(tree.get_graph_nodes()))
    assert (len(new_nodes) == 2)
    leaves_before = {leaf.name for leaf in tree.leaves}
    leaves_after = {leaf.name for leaf in modified_tree.leaves}
    assert (leaves_before != leaves_after)


def test_should_modify_all_series_nodes():
    # given
    workflow: Workflow = load_sp_workflow()
    wf_workflow: WfWorkflow = workflow.wf_instance.workflow
    nodes_count = wf_workflow.number_of_nodes()
    weights: dict = defaultdict(lambda: 1)
    max_subgraph_size: int = 2
    tree: SPTreeNode = get_sp_decomposition_tree(wf_workflow)
    split_algorithm: SeriesParallelSplitFinal = SeriesParallelSplitFinal()

    # when
    modified_tree: SPTreeNode = split_algorithm.modify_series_nodes(tree, {}, wf_workflow, weights, max_subgraph_size)

    # then
    assert (nodes_count + 9 == wf_workflow.number_of_nodes())
    new_nodes: set = set(modified_tree.get_graph_nodes()).difference(set(tree.get_graph_nodes()))
    assert (len(new_nodes) == 9)
    leaves_before = {leaf.name for leaf in tree.leaves}
    leaves_after = {leaf.name for leaf in modified_tree.leaves}
    assert (leaves_before != leaves_after)


def test_full_division():
    # given
    tasks_file = "resources/workflows/new_method_3_nodes.json"
    machines_file = "resources/machines/3_machines.json"
    deadline = 12
    workflow: Workflow = Workflow(tasks_file, machines_file, deadline)
    split_algorithm: SeriesParallelSplitFinal = SeriesParallelSplitFinal()

    # when
    division: Division = split_algorithm.decompose(workflow, 2)

    # then
    assert (len(division.workflows) == 2)
    assert (sum(map(lambda w: w.deadline, division.workflows)) == 12)
    workflow1, workflow2 = division.workflows
    assert (workflow1.deadline == 11)
    assert (workflow2.deadline == 1)


def load_sp_workflow() -> Workflow:
    tasks_file = "resources/workflows/complex_workflow.json"
    machines_file = "resources/machines/3_machines.json"
    deadline = 50
    split_algorithm: SeriesParallelSplitFinal = SeriesParallelSplitFinal()
    return split_algorithm.create_sp_workflow(Workflow(tasks_file, machines_file, deadline))
