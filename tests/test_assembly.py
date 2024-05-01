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
        graph, *[(path, {}) for path in sz.topology.iter_maximal_unitig_paths(graph)]
    )
    degree_stats1 = sz.stats.degree_stats(graph)
    vertex_stats1 = graph.get_vertices(
        vprops=[graph.vp["length"], graph.vp["depth"], graph.vp["filter"]]
    ).copy()
    gm.batch_press(
        graph,
        *[
            (path, {})
            for path in sz.topology.iter_maximal_unitig_paths(
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
    tips = sz.topology.find_tips(graph)
    assert np.array_equal(tips, [0, 1, 2, 5])


def test_find_junctions():
    graph = gt.Graph([(0, 4), (1, 4), (4, 2), (4, 3), (3, 5)])
    junctions = sz.topology.find_junctions(graph)
    assert np.array_equal(junctions, [4])


def test_find_unitigs():
    graph = gt.Graph(
        [
            # A 3-cycle
            (0, 1),
            (1, 2),
            (2, 0),
            # An in-3-lolipop
            (3, 4),
            (4, 5),
            (5, 3),
            (6, 3),
            # An out-3-lolipop
            (7, 8),
            (8, 9),
            (9, 7),
            (7, 10),
            # A 2-cycle
            (11, 12),
            (12, 11),
            # A 1-cycle
            (13, 13),
            # An in-1-lolipop
            (14, 14),
            (15, 14),
            # An out-1-lolipop
            (16, 16),
            (16, 17),
        ]
    )
    graph.add_vertex(1)  # An orphan vertex.
    unitig_paths = set([tuple(p) for p in sz.topology.iter_maximal_unitig_paths(graph)])
    assert unitig_paths == {(0, 1, 2), (3, 4, 5), (8, 9, 7), (11, 12)}


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
            tuple(tuple(p) for p in sz.topology.iter_maximal_unitig_paths(graph))
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
            *[(path, {}) for path in sz.topology.iter_maximal_unitig_paths(graph)]
        )
        examples.append(graph)
    # Check that all are equal to each other.
    for i in range(k_replicates):
        assert (examples[0].get_edges() == examples[1].get_edges()).all()


def test_press_unitigs_with_cycles():
    _graph = gt.Graph()
    _graph.add_edge_list(
        [
            (0, 1),
            (1, 2),
            (2, 3),  # Linear stretch
            (4, 5),
            (5, 6),
            (6, 4),  # 3-cycle
            (7, 7),  # 1-cycle
            (8, 9),
            (9, 10),
            (10, 8),
            (10, 11),  # Lolipop, stick-out
            (12, 13),
            (13, 14),
            (14, 12),
            (15, 14),  # Lolipop, stick-in
        ]
    )
    _graph.vp["filter"] = _graph.new_vertex_property("bool", val=True)
    _graph.set_vertex_filter(_graph.vp["filter"])
    gm = sz.graph_manager.GraphManager()
    gm.validate(_graph)
    # sz.draw.draw_graph(_graph, ink_scale=1, output_size=(200, 200), vertex_text=_graph.vertex_index)
    unitig_paths = set(tuple(u) for u in sz.topology.iter_maximal_unitig_paths(_graph))
    assert unitig_paths == {(0, 1, 2, 3), (14, 12, 13), (4, 5, 6), (8, 9, 10)}
    gm.batch_press(_graph, *[(list(path), {}) for path in unitig_paths])
    # sz.draw.draw_graph(_graph, ink_scale=1, output_size=(200, 200), vertex_text=_graph.vertex_index)
    assert (
        sz.stats.degree_stats(_graph).sort_index().reset_index().values
        == np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 2.0],
                [1.0, 2.0, 1.0],
                [2.0, 1.0, 1.0],
            ]
        )
    ).all()


def test_unzip_lolipops():
    _graph = gt.Graph()
    _graph.add_edge_list(
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (0, 3),  # Out lolipop
            (4, 5),
            (5, 6),
            (6, 4),
            (7, 4),  # In lolipop
        ]
    )
    _graph.vp["filter"] = _graph.new_vertex_property("bool", val=True)
    _graph.set_vertex_filter(_graph.vp["filter"])
    gm = sz.graph_manager.GraphManager()
    gm.validate(_graph)
    # sz.draw.draw_graph(_graph, ink_scale=1, output_size=(200, 200), vertex_text=_graph.vertex_index)
    unitig_paths = [tuple(u) for u in sz.topology.iter_maximal_unitig_paths(_graph)]
    assert set(frozenset(u) for u in unitig_paths) == {
        frozenset([0, 1, 2]),
        frozenset([4, 5, 6]),
    }
    gm.batch_press(_graph, *[(list(path), {}) for path in unitig_paths])
    # sz.draw.draw_graph(_graph, ink_scale=1, output_size=(200, 200), vertex_text=_graph.vertex_index)

    gm.batch_unzip(_graph, (9, [(9, 9), (7, 9)], {}), (8, [(8, 8), (8, 3)], {}))
    # sz.draw.draw_graph(_graph, ink_scale=1, output_size=(200, 200), vertex_text=_graph.vertex_index)
    assert (
        sz.stats.degree_stats(_graph).sort_index().reset_index().values
        == np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 2.0],
                [1.0, 2.0, 1.0],
                [2.0, 1.0, 1.0],
            ]
        )
    ).all()
