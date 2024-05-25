import uuid
from abc import ABC, abstractmethod
from enum import Enum

import networkx as nx
from anytree import Node


class Composition(Enum):
    PARALLEL = 0
    SERIES = 1


class SPTreeNode(ABC, Node):
    @abstractmethod
    def get_graph_nodes(self):
        pass


class CompositionNode(SPTreeNode):
    def __init__(self, left: SPTreeNode, right: SPTreeNode, composition: Composition, common_nodes: list[str]):
        super().__init__(name=uuid.uuid1(), children=[left, right])
        self._left = left
        self._right = right
        self.operation = composition
        self.common_nodes = common_nodes

    def get_graph_nodes(self):
        return set().union(*map(lambda c: c.get_graph_nodes(), self.children))

    @property
    def left_child(self):
        return self._left

    @property
    def right_child(self):
        return self._right


class EdgeNode(SPTreeNode):
    def __init__(self, edge):
        self.u = edge[0]
        self.v = edge[1]
        self.edge = edge
        super().__init__((self.u, self.v))

    def get_graph_nodes(self):
        return {self.u, self.v}


def with_extra_functions(graph: nx.MultiDiGraph):
    def get_sources(self):
        return [n for n, d in self.in_degree() if d == 0]

    def get_sinks(self):
        return [n for n, d in self.out_degree() if d == 0]

    def reduce_parallel(self, e1, e2):
        if e1[0] != e2[0] or e1[1] != e2[1]:
            raise ValueError("Only identical edges can be reduced")
        u, v = e1[0], e1[1]
        e1_tree, e2_tree = self.get_edge_tree(e1), self.get_edge_tree(e2)
        self.remove_edges_from([e1, e2])
        self.add_edge(u, v, tree=CompositionNode(e1_tree, e2_tree, Composition.PARALLEL, [u, v]))

    def reduce_series(self, e1, e2):
        if e1[1] != e2[0]:
            raise ValueError("Edges must be consecutive to be reduced")
        u, w, v = e1[0], e1[1], e2[1]
        e1_tree, e2_tree = self.get_edge_tree(e1), self.get_edge_tree(e2)
        self.remove_edges_from([e1, e2])
        self.remove_node(w)
        self.add_edge(u, v, tree=CompositionNode(e1_tree, e2_tree, Composition.SERIES, [w]))

    def get_edge_tree(self, e) -> EdgeNode:
        return self.edges[e].get('tree', EdgeNode(e))

    graph.get_sources = get_sources
    graph.get_sinks = get_sinks
    graph.reduce_parallel = reduce_parallel
    graph.reduce_series = reduce_series
    graph.get_edge_tree = get_edge_tree
    return graph


@with_extra_functions
class GraphDecorator(nx.MultiDiGraph):
    pass


def label_edges(graph: nx.MultiDiGraph, label_generator) -> None:
    for edge in graph.edges(keys=True):
        graph.edges[edge]['label'] = label_generator(edge)


def reduce_parallels(graph: GraphDecorator, node, retrieve_edges):
    edges = retrieve_edges(node)
    if len(edges) <= 1:
        return
    e1, e2 = edges[:2]
    if e1[0] != e2[0] or e1[1] != e2[1]:
        return
    graph.reduce_parallel(e1, e2)
    return reduce_parallels(graph, node, retrieve_edges)


def process_vertex(graph: GraphDecorator, node, unsatisfied_nodes: set):
    reduce_parallels(graph, node, lambda n: list(graph.in_edges(n, keys=True)))
    reduce_parallels(graph, node, lambda n: list(graph.out_edges(n, keys=True)))
    if graph.in_degree(node) == 1 and graph.out_degree(node) == 1:
        e1, e2 = list(graph.in_edges(node, keys=True))[0], list(graph.out_edges(node, keys=True))[0]
        graph.reduce_series(e1, e2)
        unsatisfied_nodes.update({e1[0], e2[1]})


def is_trivial_sp_dag(graph: nx.DiGraph):
    return graph.number_of_nodes() == 2 and graph.number_of_edges() == 1


def recognize_sp(graph: nx.DiGraph) -> GraphDecorator:
    graph = GraphDecorator(graph)
    sources, sinks = graph.get_sources(), graph.get_sinks()
    if len(sources) != 1 or len(sinks) != 1:
        return graph
    source, sink = sources[0], sinks[0]
    unsatisfied_nodes = {n for n in graph.nodes if n != sink and n != source}
    while unsatisfied_nodes:
        node = unsatisfied_nodes.pop()
        process_vertex(graph, node, unsatisfied_nodes)
    unsatisfied_nodes.update({source, sink})
    while unsatisfied_nodes:
        node = unsatisfied_nodes.pop()
        process_vertex(graph, node, unsatisfied_nodes)
    return graph


def is_sp_dag(graph: nx.DiGraph) -> bool:
    graph = recognize_sp(graph)
    return is_trivial_sp_dag(graph)


def get_sp_decomposition_tree(graph: nx.DiGraph) -> SPTreeNode:
    graph = recognize_sp(graph)
    if is_trivial_sp_dag(graph):
        return graph.get_edge_tree(list(graph.edges(keys=True))[0])
    else:
        raise ValueError("Input graph must be SP to build decomposition tree")


def apply_weights_on_tree(tree: SPTreeNode, weights: dict):
    if tree.is_leaf:
        tree.weight = sum([weights[node] for node in tree.get_graph_nodes()])
    elif isinstance(tree, CompositionNode):
        [apply_weights_on_tree(child, weights) for child in tree.children]
        if tree.operation == Composition.SERIES:
            tree.weight = sum(child.weight for child in tree.children) - sum(weights[n] for n in tree.common_nodes)
        elif tree.operation == Composition.PARALLEL:
            tree.weight = max(child.weight for child in tree.children)
        else:
            raise TypeError(f"Unable to apply weight to node with operation {tree.operation}")
    else:
        raise TypeError(f"Unable to apply weight to node of type {type(tree)}")
