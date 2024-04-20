from __future__ import annotations

import pathlib
import tempfile
from copy import copy
from dataclasses import dataclass, field
from typing import Optional, Union, Set, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm
from wfcommons.common.workflow import Workflow as WfWorkflow

from QHyper.problems.workflow_scheduling import Workflow, TargetMachine, WorkflowSchedulingOneHot, \
    WorkflowSchedulingBinary
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


def wfworkflow_to_qhyper_workflow(workflow: WfWorkflow, machines: str | dict[str, TargetMachine],
                                  deadline: float) -> Workflow:
    with tempfile.NamedTemporaryFile() as temp:
        workflow.write_json(pathlib.Path(temp.name))
        return Workflow(pathlib.Path(temp.name), machines, deadline)


def merge_subworkflows(subworkflows: list) -> WfWorkflow:
    workflow = WfWorkflow(name="merged")
    for idx, subworkflow in enumerate(subworkflows):
        for task in subworkflow.tasks.values():
            task_copy = copy(task)
            task_copy.category = task_copy.category or f"{idx}"
            workflow.add_task(task_copy)
    for subworkflow in subworkflows:
        for task in subworkflow.tasks.keys():
            for parent in subworkflow.tasks_parents[task]:
                workflow.add_dependency(parent, task)
    return workflow


def draw(g: nx.DiGraph,
         extension: Optional[str] = 'png',
         with_labels: bool = False,
         ax: Optional[plt.Axes] = None,
         show: bool = False,
         save: Optional[Union[pathlib.Path, str]] = None,
         close: bool = False,
         legend: bool = False,
         node_size: int = 1000,
         linewidths: int = 5,
         subgraph: Set[str] = set()) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a netwrokX DiGraph.

    :param g: graph to be plotted.
    :type g: networkX DiGraph.
    :type extension: extension of the output file.
    :param extension: str.
    :param with_labels: if set, it prints the task types over their nodes.
    :type with_labels: bool.
    :param ax: plot axes.
    :type ax: plt.Axes.
    :param show: if set, displays the plot on screen.
    :type show: bool.
    :param save: path to directory to save the plot.
    :type save: pathlib.Path.
    :param close: if set, automatically closes window that displays plot.
    :type close: bool.
    :param legend: if set, displays legend of the plot.
    :type legend: bool.
    :param node_size: size of the nodes (circles) in the plot.
    :type node_size: int.
    :param linewidths: thickness of the edges in the plot.
    :type linewidths: int.
    :param subgraph: nodes that were added by replication and will be colored green.
    :type subgraph: Set[str].



    :return: the figure and the axis used.
    :rtype:  Tuple[plt.Figure, plt.Axes].
    """
    fig: plt.Figure
    ax: plt.Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.get_figure()

    node_border_colors = {}
    if isinstance(subgraph, dict):
        for color, nodes in subgraph.items():
            for node in nodes:
                node_border_colors[node] = color
    else:
        for node in subgraph:
            node_border_colors[node] = "green"

    pos = nx.nx_agraph.pygraphviz_layout(g, prog='dot')
    type_set = sorted({g.nodes[node]["task"].category or "default" for node in g.nodes})  # not type-hash
    types = {
        t: i for i, t in enumerate(type_set)
    }
    node_color = [types[g.nodes[node]["task"].category or "default"] for node in g.nodes]  # not type-hash
    for node in g.nodes:
        if node in subgraph:
            g.nodes[node]["node_shape"] = "s"
        else:
            g.nodes[node]["node_shape"] = "c"
    edgecolors = [node_border_colors.get(node, "white") for node in g.nodes]
    edge_color = [
        node_border_colors.get(src) if node_border_colors.get(src, -1) == node_border_colors.get(dst, 1) else "black"
        for src, dst in g.edges
    ]
    cmap = cm.get_cmap('rainbow', len(type_set))
    nx.draw(g, pos, node_size=node_size, node_color=node_color, edgecolors=edgecolors, edge_color=edge_color,
            linewidths=linewidths, cmap=cmap, ax=ax, with_labels=with_labels)
    color_lines = [mpatches.Patch(color=cmap(types[t]), label=t) for t in type_set]

    if legend:
        legend = ax.legend(handles=color_lines, loc='lower right')

    if show:
        plt.show()

    if save is not None:
        if extension == 'dot':
            nx.drawing.nx_agraph.write_dot(g, save)
        else:
            fig.savefig(f'{save}').with_suffix(f".{extension.lstrip('.')}")

    if close:
        plt.close(fig)

    return fig, ax
