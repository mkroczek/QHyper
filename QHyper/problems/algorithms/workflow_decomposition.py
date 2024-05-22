import copy
from copy import deepcopy
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from anytree import AnyNode
from wfcommons.common.machine import Machine
from wfcommons.common.task import Task, TaskType
from wfcommons.common.workflow import Workflow as WfWorkflow

from QHyper.problems.algorithms.graph_utils import get_sp_decomposition_tree, apply_weights_on_tree, CompositionNode, \
    Composition, SPTreeNode
from QHyper.problems.algorithms.spization import SpIzationAlgorithm, JavaFacadeSpIzationAlgorithm
from QHyper.problems.algorithms.utils import wfworkflow_to_qhyper_workflow
from QHyper.problems.workflow_scheduling import Workflow


class ArtificialMachine(Machine):
    def __init__(self):
        super().__init__(
            name="Artificial machine",
            cpu={
                "speed": 1,
                "count": 1
            })


class ConnectingTask(Task):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            task_type=TaskType.COMPUTE,
            runtime=0,
            cores=1,
            machine=ArtificialMachine(),
            category="artificial"
        )


@dataclass
class Division:
    method: str
    complete_workflow: Workflow
    workflows: list[Workflow] = field(default_factory=list)


def add_entry_and_exit_tasks(workflow: Workflow) -> Workflow:
    old_workflow: WfWorkflow = workflow.wf_instance.workflow
    workflow_copy: WfWorkflow = create_subworkflow(old_workflow, old_workflow.tasks.keys(), old_workflow.name)
    if len(workflow_copy.roots()) > 1:
        add_entry_task(workflow_copy, ConnectingTask(name="entry_task"))
    if len(workflow_copy.leaves()) > 1:
        add_exit_task(workflow_copy, ConnectingTask(name="exit_task"))
    return wfworkflow_to_qhyper_workflow(workflow_copy, deepcopy(workflow.machines), workflow.deadline)


def add_entry_task(workflow: WfWorkflow, entry_task: Task):
    roots = workflow.roots()
    workflow.add_task(entry_task)
    for root in roots:
        workflow.add_dependency(entry_task.name, root)


def add_exit_task(workflow: WfWorkflow, exit_task: Task):
    leaves = workflow.leaves()
    workflow.add_task(exit_task)
    for leaf in leaves:
        workflow.add_dependency(leaf, exit_task.name)


def create_subworkflows(workflow: Workflow, parts: list) -> list[WfWorkflow]:
    wfworkflow = workflow.wf_instance.workflow
    subworkflows = []
    for pair_id, pair in enumerate(zip(parts[:-1], parts[1:])):
        part1, part2 = pair
        connecting_task = ConnectingTask(name=f"Task{pair_id}_{pair_id + 1}")

        workflow1 = create_subworkflow(wfworkflow, part1, f"subworkflow{pair_id}")
        workflow1_leafs = workflow1.leaves()
        workflow1.add_task(connecting_task)
        for leaf in workflow1_leafs:
            workflow1.add_dependency(leaf, connecting_task.name)

        workflow2 = create_subworkflow(wfworkflow, part2, f"subworkflow{pair_id + 1}")
        workflow2_roots = workflow2.roots()
        workflow2.add_task(connecting_task)
        for root in workflow2_roots:
            workflow2.add_dependency(connecting_task.name, root)

        subworkflows.extend([workflow1, workflow2])

    return subworkflows


def create_subworkflow(parent_workflow: WfWorkflow, tasks: list, name: str) -> WfWorkflow:
    subworkflow = WfWorkflow(
        name=f"workflow{name}"
    )
    for task in tasks:
        subworkflow.add_task(parent_workflow.tasks[task])
    for task in tasks:
        for parent in filter(lambda p: p in tasks, parent_workflow.tasks_parents[task]):
            subworkflow.add_dependency(parent, task)
    return subworkflow


class HeftBasedAlgorithm:
    def calc_rank_up(self, workflow: Workflow, task, rank, mean_times):
        if task in rank:
            return rank[task]
        children = workflow.wf_instance.workflow.tasks_children[task]
        max_child_rank = max(
            [self.calc_rank_up(workflow, child, rank, mean_times) for child in children]) if children else 0
        rank[task] = mean_times[task] + max_child_rank
        return rank[task]

    def select_split_chunks(self, rank_up, n_parts):
        sorted_tasks = sorted(rank_up, key=rank_up.get, reverse=True)
        return np.array_split(sorted_tasks, n_parts)

    def select_split_chunks_enhanced(self, rank_up: dict, n_parts: int, mean_times: dict) -> list:
        summary_mean_times = sum(mean_times.values())
        sum_points = [summary_mean_times / n_parts * i for i in range(1, n_parts)]
        current_point = 0
        current_sum = 0.0
        parts = []
        part = []
        sorted_tasks = sorted(rank_up, key=rank_up.get, reverse=True)
        for task in sorted_tasks:
            print(
                f"Task: {task}, his rank, sum_points = {sum_points}, current_point = {current_point}, current_sum = {current_sum}")
            if mean_times[task] + current_sum < sum_points[current_point]:
                part.append(task)
            else:
                parts.extend(part)
                part = []
                part.append(task)
                current_point += 1
            current_sum += mean_times[task]
        return parts

    def split_deadline(self, deadline: float, chunks: list, mean_times: dict) -> list[int]:
        mean_times_sum = sum(mean_times.values())
        deadline_per_chunk = []
        for chunk in chunks:
            chunk_mean_time_sum = sum(map(lambda task: mean_times[task], chunk))
            deadline_per_chunk.append(round(chunk_mean_time_sum / mean_times_sum * deadline))
        return deadline_per_chunk

    def decompose(self, workflow: Workflow, n_parts: int) -> Division:
        mean_times = workflow.time_matrix.mean(axis=1).to_dict()
        rank_up = {}

        first_task = workflow.wf_instance.roots()[0]

        self.calc_rank_up(workflow, first_task, rank_up, mean_times)
        split_chunks = self.select_split_chunks(rank_up, n_parts)
        deadline = workflow.deadline
        deadline_per_chunk = self.split_deadline(deadline, split_chunks, mean_times)
        subworkflows = create_subworkflows(workflow, split_chunks)

        division = Division("HeftBasedAlgorithm", workflow)

        for subworkflow, subdeadline in zip(subworkflows, deadline_per_chunk):
            division.workflows.append(wfworkflow_to_qhyper_workflow(subworkflow, workflow.machines, subdeadline))

        return division


class SeriesParallelSplit:
    class DivisionTreeNode(AnyNode):
        def __init__(self, workflow: WfWorkflow, deadline: float, children=None):
            super().__init__(children=children)
            self.workflow = workflow
            self.deadline = deadline

    def distribute_deadline(self, tree: SPTreeNode, deadline: float):
        tree.deadline = deadline
        if isinstance(tree, CompositionNode):
            if tree.operation == Composition.PARALLEL:
                [self.distribute_deadline(child, deadline) for child in tree.children]
            elif tree.operation == Composition.SERIES:
                weights_sum = sum(child.weight for child in tree.children)
                [self.distribute_deadline(child, round(child.weight / weights_sum * deadline)) for child in
                 tree.children]
            else:
                raise TypeError(f"Unable to distribute deadline for node of type {type(tree)}")

    def build_division_tree(self, workflow: WfWorkflow, tree: SPTreeNode, max_graph_size: int) -> DivisionTreeNode:
        graph_nodes = tree.get_graph_nodes()
        if len(graph_nodes) <= max_graph_size:
            return self.DivisionTreeNode(create_subworkflow(workflow, graph_nodes, "name"), tree.deadline)
        elif tree.is_leaf:
            raise ValueError(f"Demanded max graph size is {max_graph_size}, \
                    but graph containing {len(graph_nodes)} nodes can't be divided further")
        else:
            children_trees = [self.build_division_tree(workflow, child, max_graph_size) for child in tree.children]
            return self.DivisionTreeNode(None, tree.deadline, children=children_trees)

    def _get_task(self, old_workflow: WfWorkflow, name: str):
        if name in old_workflow.tasks:
            return old_workflow.tasks[name]
        else:
            return ConnectingTask(name)

    def wrap_in_workflow(self, old_workflow: WfWorkflow, sp_dag: nx.DiGraph) -> WfWorkflow:
        node_mapping: dict[str, Task] = {name: self._get_task(old_workflow, name) for name in sp_dag.nodes}
        new_workflow = WfWorkflow(
            name=f"SP_{old_workflow.name}"
        )
        for node in sp_dag.nodes:
            new_workflow.add_task(node_mapping[node])
        for node1, node2 in sp_dag.edges:
            task1_name, task2_name = node_mapping[node1].name, node_mapping[node2].name
            new_workflow.add_dependency(task1_name, task2_name)
        return new_workflow

    def create_sp_workflow(self, workflow: Workflow) -> Workflow:
        old_workflow: WfWorkflow = workflow.wf_instance.workflow
        spization: SpIzationAlgorithm = JavaFacadeSpIzationAlgorithm()
        sp_dag: nx.DiGraph = spization.run(old_workflow)
        new_workflow = self.wrap_in_workflow(old_workflow, sp_dag)
        return wfworkflow_to_qhyper_workflow(new_workflow, workflow.machines, workflow.deadline)

    def decompose(self, workflow: Workflow, max_graph_size: int) -> Division:
        workflow = self.create_sp_workflow(workflow)
        mean_times: dict = workflow.time_matrix.mean(axis=1).to_dict()
        wf_workflow: WfWorkflow = workflow.wf_instance.workflow
        tree: SPTreeNode = get_sp_decomposition_tree(wf_workflow)
        apply_weights_on_tree(tree, mean_times)
        self.distribute_deadline(tree, workflow.deadline)
        division_tree = self.build_division_tree(wf_workflow, tree, max_graph_size)

        division = Division("SeriesParallelSplitAlgorithm", workflow)

        for leaf in division_tree.leaves:
            division.workflows.append(wfworkflow_to_qhyper_workflow(leaf.workflow, workflow.machines, leaf.deadline))

        return division

class SeriesParallelSplitEnhanced:
    class DivisionTreeNode(AnyNode):
        def __init__(self, workflow: WfWorkflow, deadline: float, children=None):
            super().__init__(children=children)
            self.workflow = workflow
            self.deadline = deadline

    class PrunedNode(SPTreeNode):
        def __init__(self, to_override:SPTreeNode):
            self.nodes: set[str] = to_override.get_graph_nodes()
            self.weight = to_override.weight
            super().__init__(to_override.name)

        def get_graph_nodes(self):
            return self.nodes

        def override_node(self, old_node: str, new_node: str, old_node_weight: float):
            self.nodes.remove(old_node)
            self.nodes.add(new_node)
            self.weight -= old_node_weight

    def modify_series_nodes(self, tree: SPTreeNode, override_nodes:dict[str,str], workflow: nx.DiGraph, weights:dict[str,float]):
        if tree.is_leaf:
            if isinstance(tree, self.PrunedNode):
                for node in filter(lambda n: n in tree.get_graph_nodes(), override_nodes):
                    tree.override_node(node, override_nodes[node], weights[node])
            else:
                raise TypeError("Leaf is expected to be of type PrunedTree")
        elif isinstance(tree, CompositionNode):
            if tree.operation == Composition.PARALLEL:
                [self.modify_series_nodes(child, override_nodes, workflow, weights) for child in tree.children]
            elif tree.operation == Composition.SERIES:
                left: SPTreeNode = tree.children[0]
                right: SPTreeNode = tree.children[1]
                common_node = tree.common_nodes[0]
                common_node_substitute = common_node + "_substitute"
                self.add_substitue(workflow, common_node, common_node_substitute)
                tree.common_nodes = []
                self.modify_series_nodes(left, override_nodes, workflow, weights)
                right_override_nodes = override_nodes.copy()
                right_override_nodes[common_node] = common_node_substitute
                self.modify_series_nodes(right, right_override_nodes, workflow, weights)
            else:
                raise TypeError(f"Unable to distribute deadline for node of type {type(tree)}")

    def add_substitue(self, graph: nx.DiGraph, common_node, common_node_substitute):
        graph.add_node(common_node_substitute)
        out_edges = list(graph.out_edges(common_node))
        for (src, dst) in out_edges:
            graph.add_edge(common_node_substitute, dst)
        graph.remove_edges_from(out_edges)
        graph.add_edge(common_node, common_node_substitute)

    def distribute_deadline(self, tree: SPTreeNode, deadline: float):
        tree.deadline = deadline
        if isinstance(tree, CompositionNode):
            if tree.operation == Composition.PARALLEL:
                [self.distribute_deadline(child, deadline) for child in tree.children]
            elif tree.operation == Composition.SERIES:
                weights_sum = sum(child.weight for child in tree.children)
                [self.distribute_deadline(child, round(child.weight / weights_sum * deadline)) for child in
                 tree.children]
            else:
                raise TypeError(f"Unable to distribute deadline for node of type {type(tree)}")

    def prune_tree(self, tree: SPTreeNode, max_graph_size: int) -> SPTreeNode:
        graph_nodes = tree.get_graph_nodes()
        if len(graph_nodes) <= max_graph_size:
            return self.PrunedNode(tree)
        elif tree.is_leaf:
            raise ValueError(f"Demanded max graph size is {max_graph_size}, \
                    but graph containing {len(graph_nodes)} nodes can't be divided further")
        else:
            children_trees = [self.prune_tree(child, max_graph_size) for child in tree.children]
            tree.children = children_trees
            return tree

    def _get_task(self, old_workflow: WfWorkflow, name: str):
        if name in old_workflow.tasks:
            return old_workflow.tasks[name]
        else:
            return ConnectingTask(name)

    def wrap_in_workflow(self, old_workflow: WfWorkflow, sp_dag: nx.DiGraph) -> WfWorkflow:
        node_mapping: dict[str, Task] = {name: self._get_task(old_workflow, name) for name in sp_dag.nodes}
        new_workflow = WfWorkflow(
            name=f"SP_{old_workflow.name}"
        )
        for node in sp_dag.nodes:
            new_workflow.add_task(node_mapping[node])
        for node1, node2 in sp_dag.edges:
            task1_name, task2_name = node_mapping[node1].name, node_mapping[node2].name
            new_workflow.add_dependency(task1_name, task2_name)
        return new_workflow

    def create_sp_workflow(self, workflow: Workflow) -> Workflow:
        old_workflow: WfWorkflow = workflow.wf_instance.workflow
        spization: SpIzationAlgorithm = JavaFacadeSpIzationAlgorithm()
        sp_dag: nx.DiGraph = spization.run(old_workflow)
        new_workflow = self.wrap_in_workflow(old_workflow, sp_dag)
        return wfworkflow_to_qhyper_workflow(new_workflow, workflow.machines, workflow.deadline)

    def verify_weights(self, tree: SPTreeNode):
        if isinstance(tree, CompositionNode):
            if tree.operation == Composition.PARALLEL:
                [self.verify_weights(child) for child in tree.children]
            elif tree.operation == Composition.SERIES:
                weights_sum = sum(child.weight for child in tree.children)
                assert weights_sum == tree.weight
                [self.verify_weights(child) for child in tree.children]
            else:
                raise TypeError(f"Unable to distribute deadline for node of type {type(tree)}")

    def _apply_weights_on_tree(self, tree: SPTreeNode, weights: dict):
        if isinstance(tree, CompositionNode):
            [self._apply_weights_on_tree(child, weights) for child in tree.children]
            if tree.operation == Composition.SERIES:
                tree.weight = sum(child.weight for child in tree.children) - sum(weights[n] for n in tree.common_nodes)
            elif tree.operation == Composition.PARALLEL:
                tree.weight = max(child.weight for child in tree.children)
            else:
                raise TypeError(f"Unable to apply weight to node with operation {tree.operation}")

    def decompose(self, workflow: Workflow, max_graph_size: int) -> Division:
        workflow = self.create_sp_workflow(workflow)
        mean_times: dict = workflow.time_matrix.mean(axis=1).to_dict()
        wf_workflow: WfWorkflow = workflow.wf_instance.workflow
        tree: SPTreeNode = get_sp_decomposition_tree(wf_workflow)
        apply_weights_on_tree(tree, mean_times)
        tree: SPTreeNode = self.prune_tree(tree, max_graph_size)
        workflow_with_substitutes: nx.DiGraph = copy.deepcopy(wf_workflow)
        self.modify_series_nodes(tree, {}, workflow_with_substitutes, mean_times)
        wf_workflow = self.wrap_in_workflow(wf_workflow, workflow_with_substitutes)
        workflow = wfworkflow_to_qhyper_workflow(wf_workflow, workflow.machines, workflow.deadline)
        # mean_times: dict = workflow.time_matrix.mean(axis=1).to_dict()
        self._apply_weights_on_tree(tree, mean_times)
        self.verify_weights(tree)
        self.distribute_deadline(tree, workflow.deadline)

        division = Division("SeriesParallelSplitEnhancedAlgorithm", workflow)

        for leaf in tree.leaves:
            division.workflows.append(wfworkflow_to_qhyper_workflow(create_subworkflow(wf_workflow, leaf.get_graph_nodes(), "name"), workflow.machines, leaf.deadline))

        return division


class SimpleSplit:
    def split_deadline(self, deadline: float, chunks: list) -> list[int]:
        raise NotImplementedError

    def decompose(self, workflow: Workflow, n_parts: int) -> Division:
        wf_workflow = workflow.wf_instance.workflow
        first_task = workflow.wf_instance.roots()[0]
        distances_from_root = nx.shortest_path_length(wf_workflow, first_task)
        sorted_tasks = sorted(distances_from_root, key=distances_from_root.get)
        split_chunks = np.array_split(sorted_tasks, n_parts)

        division = Division("SimpleSplitAlgorithm", workflow)

        for subworkflow, subdeadline in zip(create_subworkflows(workflow, split_chunks),
                                            self.split_deadline(workflow.deadline, split_chunks)):
            division.workflows.append(wfworkflow_to_qhyper_workflow(subworkflow, workflow.machines, subdeadline))

        return division
