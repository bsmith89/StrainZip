import graph_tool as gt
import numpy as np

import strainzip as sz


def test_generate_dbg():
    np.random.seed(1)
    gt.seed_rng(1)
    sequence = sz.simulate.random_sequence(1000)
    graph = sz.simulate.annotated_dbg(sequence, k=7, circularize=True, include_rc=True)
    gm = sz.graph_manager.GraphManager(
        unzippers=[
            sz.graph_manager.FilterUnzipper(),
            sz.graph_manager.LengthUnzipper(),
            sz.graph_manager.SequenceUnzipper(),
            sz.graph_manager.ScalarDepthUnzipper(),
        ],
        pressers=[
            sz.graph_manager.FilterPresser(),
            sz.graph_manager.LengthPresser(),
            sz.graph_manager.SequencePresser(sep=","),
            sz.graph_manager.ScalarDepthPresser(),
        ],
    )
    gm.validate(graph)


def test_press_unitigs():
    np.random.seed(1)
    gt.seed_rng(1)
    sequence = sz.simulate.random_sequence(1000)
    graph = sz.simulate.annotated_dbg(sequence, k=5, circularize=True, include_rc=True)
    gm = sz.graph_manager.GraphManager(
        unzippers=[
            sz.graph_manager.FilterUnzipper(),
            sz.graph_manager.LengthUnzipper(),
            sz.graph_manager.SequenceUnzipper(),
            sz.graph_manager.ScalarDepthUnzipper(),
        ],
        pressers=[
            sz.graph_manager.FilterPresser(),
            sz.graph_manager.LengthPresser(),
            sz.graph_manager.SequencePresser(sep=","),
            sz.graph_manager.ScalarDepthPresser(),
        ],
    )
    gm.validate(graph)
    degree_stats0 = sz.stats.degree_stats(graph)
    gm.batch_press(
        graph,
        *[
            (path, {})
            for path in sz.assembly.iter_maximal_unitig_paths(
                gt.GraphView(graph, vfilt=graph.vp["filter"])
            )
        ]
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