import networkx as nx

from QHyper.problems.algorithms.graph_utils import is_sp_dag
from QHyper.problems.algorithms.spization import Parser, RunnerConfiguration, FORMAT_97_CONFIGURATION, Runner, \
    JavaFacadeSpIzationAlgorithm


def test_encoding():
    # given
    graph: nx.DiGraph = nx.DiGraph()
    nodes = ['Task1', 'Task2', 'Task3']
    edges = [('Task1', 'Task2'), ('Task1', 'Task3'), ('Task3', 'Task2')]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    parser: Parser = Parser(graph)

    expected_encoding = "T: 3\nt0: m1, 1.0 s2: 1 2\nt1: m1, 1.0 s0: \nt2: m1, 1.0 s1: 1\n"

    # when
    encoded: str = parser.encode()

    # then
    assert (encoded == expected_encoding)


def test_decoding():
    # given
    graph_str = "\n".join([
        "T: 3",
        "R: 1",
        "t2: m1, 0.1 s1:1 2",
        "t1: m1, 0.8 s0:",
        "t0: m1, 0.5 s1:2 "
    ])
    graph: nx.DiGraph = nx.DiGraph()
    nodes = ['Task1', 'Task2', 'Task3']
    edges = [('Task1', 'Task2'), ('Task1', 'Task3'), ('Task3', 'Task2')]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    parser: Parser = Parser(graph)

    # when
    out_graph: nx.DiGraph = parser.decode(graph_str)

    # then
    assert (set(out_graph.nodes) == set(graph.nodes))


def test_runner():
    # given
    config: RunnerConfiguration = FORMAT_97_CONFIGURATION
    input_text: str = "T: 3\nt0: m1, 1.0 s2: 1 2\nt1: m1, 1.0 s0: \nt2: m1, 1.0 s1: 1\n"
    runner: Runner = Runner(config)
    expected_output: str = "T: 3\nR: 1\nt2: m1, 1.0 s1:1 \nt1: m1, 1.0 s0:\nt0: m1, 1.0 s1:2 \n"

    # when
    output = runner.run(input_text)

    # then
    assert (output == expected_output)


def test_full_algorithm():
    # given
    graph: nx.DiGraph = nx.DiGraph()
    nodes = [f'Task{idx}' for idx in range(1, 19)]
    edges = [
        ('Task1', 'Task2'), ('Task1', 'Task3'),
        ('Task2', 'Task4'), ('Task2', 'Task5'),
        ('Task3', 'Task11'), ('Task3', 'Task12'), ('Task3', 'Task13'),
        ('Task4', 'Task6'), ('Task4', 'Task7'),
        ('Task5', 'Task7'), ('Task5', 'Task8'), ('Task5', 'Task11'),
        ('Task6', 'Task9'),
        ('Task7', 'Task9'), ('Task7', 'Task10'),
        ('Task8', 'Task9'),
        ('Task9', 'Task18'),
        ('Task10', 'Task18'),
        ('Task11', 'Task17'),
        ('Task12', 'Task17'),
        ('Task13', 'Task14'), ('Task13', 'Task15'),
        ('Task14', 'Task16'),
        ('Task15', 'Task16'),
        ('Task16', 'Task17'),
        ('Task17', 'Task18')]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    expected_nodes = nodes + ['NewNode18', 'NewNode19']
    expected_edges = [
        ('Task1', 'Task2'), ('Task1', 'Task3'),
        ('Task2', 'Task4'), ('Task2', 'Task5'),
        ('Task3', 'Task12'), ('Task3', 'Task13'),
        ('Task4', 'NewNode18'),
        ('Task5', 'NewNode18'),
        ('Task6', 'NewNode19'),
        ('Task7', 'NewNode19'),
        ('Task8', 'NewNode19'),
        ('Task9', 'Task18'),
        ('Task10', 'Task18'),
        ('Task11', 'Task17'),
        ('Task12', 'NewNode18'),
        ('Task13', 'NewNode18'),
        ('Task14', 'Task16'),
        ('Task15', 'Task16'),
        ('Task16', 'Task17'),
        ('Task17', 'Task18'),
        ('NewNode18', 'Task6'), ('NewNode18', 'Task7'), ('NewNode18', 'Task8'), ('NewNode18', 'Task11'),
        ('NewNode18', 'Task14'), ('NewNode18', 'Task15'),
        ('NewNode19', 'Task9'), ('NewNode19', 'Task10'),
    ]

    algorithm: JavaFacadeSpIzationAlgorithm = JavaFacadeSpIzationAlgorithm()

    # when
    out_graph: nx.DiGraph = algorithm.run(graph)

    # then
    nx.draw(out_graph)
    assert (set(out_graph.nodes) == set(expected_nodes))
    assert (set(out_graph.edges) == set(expected_edges))
    assert (is_sp_dag(out_graph))
