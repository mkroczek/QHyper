import networkx as nx
import numpy as np
from wfcommons.common.machine import Machine
from wfcommons.common.task import Task, TaskType
from wfcommons.common.workflow import Workflow as WfWorkflow

from QHyper.problems.workflow_scheduling import Workflow


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

    def create_subworkflow(self, parent_workflow: WfWorkflow, tasks: list, name: str) -> WfWorkflow:
        subworkflow = WfWorkflow(
            name=f"workflow{name}"
        )
        for task in tasks:
            subworkflow.add_task(parent_workflow.tasks[task])
        for task in tasks:
            for parent in filter(lambda p: p in tasks, parent_workflow.tasks_parents[task]):
                subworkflow.add_dependency(parent, task)
        return subworkflow

    def artificial_machine(self):
        return Machine(
            name="Artificial machine",
            cpu={
                "speed": 1,
                "count": 1
            }
        )

    def create_subworkflows(self, workflow: Workflow, parts: list) -> list:
        wfworkflow = workflow.wf_instance.workflow
        subworkflows = []
        for pair_id, pair in enumerate(zip(parts[:-1], parts[1:])):
            part1, part2 = pair
            connecting_task = Task(
                name=f"Task{pair_id}_{pair_id + 1}",
                task_type=TaskType.COMPUTE,
                runtime=0,
                cores=1,
                machine=self.artificial_machine()
            )

            workflow1 = self.create_subworkflow(wfworkflow, part1, f"subworkflow{pair_id}")
            workflow1_leafs = workflow1.leaves()
            workflow1.add_task(connecting_task)
            for leaf in workflow1_leafs:
                workflow1.add_dependency(leaf, connecting_task.name)

            workflow2 = self.create_subworkflow(wfworkflow, part2, f"subworkflow{pair_id + 1}")
            workflow2_roots = workflow2.roots()
            workflow2.add_task(connecting_task)
            for root in workflow2_roots:
                workflow2.add_dependency(connecting_task.name, root)

            subworkflows.extend([workflow1, workflow2])

        return subworkflows

    def decompose(self, workflow: Workflow, n_parts=2) -> list:
        mean_times = workflow.time_matrix.mean(axis=1).to_dict()
        rank_up = {}

        first_task = workflow.wf_instance.roots()[0]

        self.calc_rank_up(workflow, first_task, rank_up, mean_times)
        split_chunks = self.select_split_chunks(rank_up, n_parts)
        # split_chunks = select_split_chunks_enhanced(rank_up, n_parts, mean_times)

        return self.create_subworkflows(workflow, split_chunks)


class SimpleSplit:
    def decompose(self, workflow: Workflow, n_parts=2) -> list:
        wf_workflow = workflow.wf_instance.workflow
        first_task = workflow.wf_instance.roots()[0]
        distances_from_root = nx.shortest_path_length(wf_workflow, first_task)
        sorted_tasks = sorted(distances_from_root, key=distances_from_root.get)
        split_chunks = np.array_split(sorted_tasks, n_parts)

        return self.create_subworkflows(workflow, split_chunks)
