from itertools import product

import graph_tool as gt
import numpy as np

import strainzip as sz


def test_generate_dbg():
    np.random.seed(1)
    gt.seed_rng(1)
    sequence = sz.sequence.random_sequence(1000)
    graph = sz.build.annotated_dbg(sequence, k=7, circularize=True, include_rc=True)
    gm = sz.graph_manager.GraphManager(
        unzippers=[
            sz.graph_manager.LengthUnzipper(),
            sz.graph_manager.SequenceUnzipper(),
            sz.graph_manager.ScalarDepthUnzipper(),
        ],
        pressers=[
            sz.graph_manager.LengthPresser(),
            sz.graph_manager.SequencePresser(sep=","),
            sz.graph_manager.ScalarDepthPresser(),
        ],
    )
    gm.validate(graph)


def test_press_unitigs():
    np.random.seed(1)
    gt.seed_rng(1)
    sequence = sz.sequence.random_sequence(1000)
    graph = sz.build.annotated_dbg(sequence, k=5, circularize=True, include_rc=True)
    gm = sz.graph_manager.GraphManager(
        unzippers=[
            sz.graph_manager.LengthUnzipper(),
            sz.graph_manager.SequenceUnzipper(),
            sz.graph_manager.ScalarDepthUnzipper(),
        ],
        pressers=[
            sz.graph_manager.LengthPresser(),
            sz.graph_manager.SequencePresser(sep=","),
            sz.graph_manager.ScalarDepthPresser(),
        ],
    )
    gm.validate(graph)
    degree_stats0 = sz.stats.degree_stats(graph)
    gm.batch_press(
        graph, *[(path, {}) for path in sz.assembly.iter_maximal_unitig_paths(graph)]
    )
    degree_stats1 = sz.stats.degree_stats(graph)
    vertex_stats1 = graph.get_vertices(
        vprops=[graph.vp["length"], graph.vp["depth"], graph.vp["filter"]]
    ).copy()
    gm.batch_press(
        graph,
        *[
            (path, {})
            for path in sz.assembly.iter_maximal_unitig_paths(
                gt.GraphView(graph, vfilt=graph.vp["filter"])
            )
        ]
    )
    degree_stats2 = sz.stats.degree_stats(graph)
    vertex_stats2 = graph.get_vertices(
        vprops=[graph.vp["length"], graph.vp["depth"], graph.vp["filter"]]
    ).copy()
    assert np.array_equal(
        vertex_stats1, vertex_stats2
    ), "Successive pressing all unitigs should be a NoOp."
    assert np.array_equal(
        degree_stats1, degree_stats2
    ), "Successive pressing all unitigs should be a NoOp."


def test_find_tip():
    graph = gt.Graph([(0, 4), (1, 4), (4, 2), (4, 3), (3, 5)])
    tips = sz.assembly.find_tips(graph)
    assert np.array_equal(tips, [0, 1, 2, 5])


def test_find_junctions():
    graph = gt.Graph([(0, 4), (1, 4), (4, 2), (4, 3), (3, 5)])
    junctions = sz.assembly.find_junctions(graph)
    assert np.array_equal(junctions, [4])


def test_split_junctions():
    graph = gt.Graph(
        [
            (0, 4),
            (1, 4),
            (4, 2),
            (4, 3),
            (2, 7),
            (3, 7),
            (7, 5),
            (7, 6),
            (8, 5),
            (5, 9),
            (5, 10),
        ]
    )


def test_unitig_generator_deterministic():
    np.random.seed(1)
    gt.seed_rng(1)
    sequence = sz.sequence.random_sequence(1000)
    graph = sz.build.annotated_dbg(sequence, k=5, circularize=True, include_rc=True)

    k_replicates = 5
    examples = []

    # List unitig paths k times.
    for _ in range(k_replicates):
        examples.append(
            tuple(tuple(p) for p in sz.assembly.iter_maximal_unitig_paths(graph))
        )

    # Check that all are equal to each other.
    for i in range(k_replicates):
        assert examples[0] == examples[i]


def test_simulated_graph_building_and_unitigs_deterministic():
    gm = sz.graph_manager.GraphManager(
        unzippers=[
            sz.graph_manager.LengthUnzipper(),
            sz.graph_manager.SequenceUnzipper(),
            sz.graph_manager.ScalarDepthUnzipper(),
        ],
        pressers=[
            sz.graph_manager.LengthPresser(),
            sz.graph_manager.SequencePresser(sep=","),
            sz.graph_manager.ScalarDepthPresser(),
        ],
    )

    k_replicates = 5
    examples = []
    for i in range(k_replicates):
        np.random.seed(1)
        gt.seed_rng(1)
        sequence = sz.sequence.random_sequence(1000)
        graph = sz.build.annotated_dbg(sequence, k=5, circularize=True, include_rc=True)
        gm.validate(graph)
        gm.batch_press(
            graph,
            *[(path, {}) for path in sz.assembly.iter_maximal_unitig_paths(graph)]
        )
        examples.append(graph)
    # Check that all are equal to each other.
    for i in range(k_replicates):
        assert (examples[0].get_edges() == examples[1].get_edges()).all()
